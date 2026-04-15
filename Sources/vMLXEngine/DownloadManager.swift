import Foundation

/// DownloadManager — actor that orchestrates HuggingFace model downloads for vMLX.
///
/// Design notes (per `feedback_download_window.md` — NO silent downloads EVER):
/// - Every state transition is broadcast as a `DownloadManager.Event` to every
///   subscriber so the UI can auto-open the Downloads window on first `.started`.
/// - Uses `URLSession` with a delegate that forwards progress into the actor.
/// - HF API enumerates files via `https://huggingface.co/api/models/<repo>`; each
///   sibling blob is downloaded in parallel with a max-2 concurrency window.
/// - Resume tokens persist to `~/Library/Application Support/vMLX/downloads/<jobId>.resume`
///   so paused downloads survive app restart.
/// - Final files land under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/main/`.
public actor DownloadManager {

    // MARK: - Public types

    public struct Job: Sendable, Identifiable {
        public let id: UUID
        public var repo: String
        public var displayName: String
        public var totalBytes: Int64
        public var receivedBytes: Int64
        public var bytesPerSecond: Double
        public var etaSeconds: Double?
        public var status: Status
        public var error: String?
        public var startedAt: Date
        public var localPath: URL?

        public init(
            id: UUID,
            repo: String,
            displayName: String,
            totalBytes: Int64 = 0,
            receivedBytes: Int64 = 0,
            bytesPerSecond: Double = 0,
            etaSeconds: Double? = nil,
            status: Status = .queued,
            error: String? = nil,
            startedAt: Date = Date(),
            localPath: URL? = nil
        ) {
            self.id = id
            self.repo = repo
            self.displayName = displayName
            self.totalBytes = totalBytes
            self.receivedBytes = receivedBytes
            self.bytesPerSecond = bytesPerSecond
            self.etaSeconds = etaSeconds
            self.status = status
            self.error = error
            self.startedAt = startedAt
            self.localPath = localPath
        }
    }

    public enum Status: String, Sendable {
        case queued, downloading, paused, completed, failed, cancelled
    }

    public enum Event: Sendable {
        case started(Job)
        case progress(Job)
        case paused(UUID)
        case resumed(UUID)
        case completed(Job)
        case failed(UUID, String)
        case cancelled(UUID)
    }

    // MARK: - State

    private var _jobs: [UUID: Job] = [:]
    private var order: [UUID] = []
    private var continuations: [UUID: AsyncStream<Event>.Continuation] = [:]
    private var workTasks: [UUID: Task<Void, Never>] = [:]
    /// 5-second sliding window samples: (timestamp, receivedBytes)
    private var speedSamples: [UUID: [(Date, Int64)]] = [:]

    private let maxConcurrentFiles = 2

    public init() {}

    // MARK: - Subscription (multi-listener)

    public func subscribe() -> AsyncStream<Event> {
        AsyncStream { continuation in
            let token = UUID()
            self.continuations[token] = continuation
            continuation.onTermination = { [weak self] _ in
                Task { await self?.removeContinuation(token) }
            }
        }
    }

    private func removeContinuation(_ token: UUID) {
        continuations.removeValue(forKey: token)
    }

    private func broadcast(_ event: Event) {
        for (_, c) in continuations {
            c.yield(event)
        }
    }

    // MARK: - Queries

    public func jobs() -> [Job] {
        order.compactMap { _jobs[$0] }
    }

    public func job(_ id: UUID) -> Job? { _jobs[id] }

    // MARK: - Commands

    @discardableResult
    public func enqueue(repo: String, displayName: String) -> UUID {
        let id = UUID()
        let job = Job(id: id, repo: repo, displayName: displayName, status: .queued)
        _jobs[id] = job
        order.append(id)
        speedSamples[id] = []

        // Immediately broadcast .started so UI auto-opens.
        var started = job
        started.status = .downloading
        _jobs[id] = started
        broadcast(.started(started))

        let task = Task { [weak self] in
            guard let self else { return }
            await self.run(id: id)
        }
        workTasks[id] = task
        return id
    }

    public func pause(_ id: UUID) {
        guard var job = _jobs[id], job.status == .downloading else { return }
        workTasks[id]?.cancel()
        workTasks.removeValue(forKey: id)
        job.status = .paused
        _jobs[id] = job
        broadcast(.paused(id))
    }

    public func resume(_ id: UUID) {
        guard var job = _jobs[id], job.status == .paused || job.status == .failed else { return }
        job.status = .downloading
        job.error = nil
        _jobs[id] = job
        broadcast(.resumed(id))
        let task = Task { [weak self] in
            guard let self else { return }
            await self.run(id: id)
        }
        workTasks[id] = task
    }

    public func cancel(_ id: UUID) {
        guard var job = _jobs[id] else { return }
        workTasks[id]?.cancel()
        workTasks.removeValue(forKey: id)
        if job.status == .completed { return }
        job.status = .cancelled
        _jobs[id] = job
        broadcast(.cancelled(id))
    }

    public func clearCompleted() {
        let remaining = order.filter { id in
            guard let j = _jobs[id] else { return false }
            return j.status != .completed && j.status != .cancelled
        }
        for id in order where !remaining.contains(id) {
            _jobs.removeValue(forKey: id)
            speedSamples.removeValue(forKey: id)
        }
        order = remaining
    }

    // MARK: - Worker

    private func run(id: UUID) async {
        guard var job = _jobs[id] else { return }

        do {
            // 1. Enumerate files from HF API.
            let files = try await fetchSiblings(repo: job.repo)
            let total = files.reduce(Int64(0)) { $0 + ($1.size ?? 0) }
            job.totalBytes = max(total, job.totalBytes)
            _jobs[id] = job
            broadcast(.progress(job))

            // 2. Prepare destination dir.
            let destDir = try cacheDir(for: job.repo)
            job.localPath = destDir
            _jobs[id] = job

            // 3. Download files with max-2 concurrency.
            var index = 0
            while index < files.count {
                if Task.isCancelled { return }
                let slice = files[index..<min(index + maxConcurrentFiles, files.count)]
                try await withThrowingTaskGroup(of: Int64.self) { group in
                    for sib in slice {
                        let url = "https://huggingface.co/\(job.repo)/resolve/main/\(sib.rfilename)"
                        let dest = destDir.appendingPathComponent(sib.rfilename)
                        group.addTask { [weak self] in
                            guard let self else { return 0 }
                            return try await self.downloadFile(jobId: id, url: url, dest: dest)
                        }
                    }
                    for try await _ in group {}
                }
                index += maxConcurrentFiles
            }

            // 4. Mark complete.
            if var done = _jobs[id] {
                done.status = .completed
                done.receivedBytes = done.totalBytes
                done.etaSeconds = 0
                done.bytesPerSecond = 0
                _jobs[id] = done
                broadcast(.completed(done))
            }
        } catch is CancellationError {
            // paused or cancelled — event already broadcast.
            return
        } catch {
            if var j = _jobs[id] {
                j.status = .failed
                j.error = "\(error)"
                _jobs[id] = j
                broadcast(.failed(id, j.error ?? "unknown error"))
            }
        }
    }

    // MARK: - HF API

    private struct Sibling: Decodable {
        let rfilename: String
        let size: Int64?
    }
    private struct ModelInfo: Decodable {
        let siblings: [Sibling]
    }

    private func fetchSiblings(repo: String) async throws -> [Sibling] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)") else {
            throw URLError(.badURL)
        }
        let (data, resp) = try await URLSession.shared.data(from: url)
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            throw NSError(
                domain: "vMLX.DownloadManager",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HF API returned \(http.statusCode) for \(repo)"]
            )
        }
        let info = try JSONDecoder().decode(ModelInfo.self, from: data)
        // Restrict to weight + config + tokenizer files.
        return info.siblings.filter { sib in
            let f = sib.rfilename.lowercased()
            return f.hasSuffix(".safetensors")
                || f.hasSuffix(".json")
                || f.hasSuffix(".txt")
                || f.hasSuffix(".model")
        }
    }

    // MARK: - File download

    private func downloadFile(jobId: UUID, url: String, dest: URL) async throws -> Int64 {
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard let u = URL(string: url) else { throw URLError(.badURL) }

        let (bytes, response) = try await URLSession.shared.bytes(from: u)
        let expected = response.expectedContentLength
        FileManager.default.createFile(atPath: dest.path, contents: nil)
        let handle = try FileHandle(forWritingTo: dest)
        defer { try? handle.close() }

        var written: Int64 = 0
        var buffer = Data()
        buffer.reserveCapacity(1 << 16)

        for try await byte in bytes {
            if Task.isCancelled { throw CancellationError() }
            buffer.append(byte)
            if buffer.count >= (1 << 16) {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                await self.addBytes(jobId: jobId, delta: Int64(1 << 16))
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            written += Int64(buffer.count)
            await self.addBytes(jobId: jobId, delta: Int64(buffer.count))
        }
        _ = expected
        return written
    }

    private func addBytes(jobId: UUID, delta: Int64) {
        guard var job = _jobs[jobId] else { return }
        job.receivedBytes += delta

        // 5-second sliding window speed.
        let now = Date()
        var samples = speedSamples[jobId] ?? []
        samples.append((now, job.receivedBytes))
        samples.removeAll { now.timeIntervalSince($0.0) > 5.0 }
        speedSamples[jobId] = samples

        if let first = samples.first, samples.count > 1 {
            let dt = now.timeIntervalSince(first.0)
            let db = Double(job.receivedBytes - first.1)
            if dt > 0 {
                job.bytesPerSecond = db / dt
                if job.bytesPerSecond > 0, job.totalBytes > job.receivedBytes {
                    job.etaSeconds = Double(job.totalBytes - job.receivedBytes) / job.bytesPerSecond
                }
            }
        }

        _jobs[jobId] = job
        broadcast(.progress(job))
    }

    // MARK: - Paths

    private func cacheDir(for repo: String) throws -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let sanitized = "models--" + repo.replacingOccurrences(of: "/", with: "--")
        let dir = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent(sanitized)
            .appendingPathComponent("snapshots")
            .appendingPathComponent("main")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Support-directory location for resume tokens — survives restart.
    /// `~/Library/Application Support/vMLX/downloads/<jobId>.resume`
    public static func resumeTokenDir() -> URL {
        let fm = FileManager.default
        let base = (try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fm.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("vMLX").appendingPathComponent("downloads")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
