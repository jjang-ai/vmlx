import Foundation

/// DownloadManager — actor that orchestrates HuggingFace model downloads for vMLX.
///
/// Design notes (per `feedback_download_window.md` — NO silent downloads EVER):
/// - Every state transition is broadcast as a `DownloadManager.Event` to every
///   subscriber so the UI can auto-open the Downloads window on first `.started`.
/// - Uses `URLSessionDownloadTask` with KVO progress observation — streams
///   directly to disk in OS-native chunks (not byte-by-byte). Delegate-free
///   keeps the actor bridging simple via `withCheckedThrowingContinuation`.
/// - HF API enumerates files via `https://huggingface.co/api/models/<repo>`; each
///   sibling blob is downloaded in parallel with a max-2 concurrency window.
/// - Gated repos: set `hfAuthToken` via `setHFAuthToken(_:)`. The token is
///   forwarded as `Authorization: Bearer <token>` on both the API lookup and
///   the file downloads. Stored in the macOS Keychain by the UI layer; the
///   manager itself only holds an in-memory copy per actor lifetime.
/// - Resume: on `resume()` we stat each existing partial file under the cache
///   dir and send a `Range: bytes=<size>-` header. Server returns 206 Partial
///   Content with the remainder; we append to the existing file. On 416 Range
///   Not Satisfiable the file is treated as complete. On ETag mismatch or
///   other 4xx we fall back to a fresh full download.
/// - Final files land under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/main/`.
public actor DownloadManager {

    // MARK: - Public types

    public struct Job: Sendable, Identifiable, Codable {
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
        /// O7 §293 — set when the HF sibling-list fetch returns 401 or
        /// 403 so the DownloadStatusBar can show a targeted CTA
        /// ("Paste HF token in Settings → API") instead of just a
        /// generic error message. DownloadManager marks this true on
        /// 401/403 responses and clears it on any subsequent retry
        /// that succeeds past the auth step.
        public var requiresHFAuth: Bool

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
            localPath: URL? = nil,
            requiresHFAuth: Bool = false
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
            self.requiresHFAuth = requiresHFAuth
        }
    }

    public enum Status: String, Sendable, Codable {
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

    /// §253b: cross-job cap. Each `run(id:)` pulls up to
    /// `maxConcurrentFiles` shards in parallel, so 5 enqueued jobs can
    /// fire 10 simultaneous HTTP streams at huggingface.co — well past
    /// HF's unauthenticated rate-limit (~30 req/min per IP). Cap the
    /// number of jobs actively fetching shards to 3; extras stay
    /// `.downloading` but block on `jobSlots` before entering the
    /// sibling fetch. FIFO so the first-enqueued gets its turn first.
    private let maxConcurrentJobs = 3
    private var activeJobs: Int = 0
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    /// HuggingFace access token for gated repos. Set via `setHFAuthToken(_:)`.
    /// The UI persists the real value in the macOS Keychain; the manager only
    /// keeps an in-memory copy for the duration of the actor's lifetime.
    private var hfAuthToken: String?

    /// In-flight URLSessionDownloadTask per job id. Used so `pause()`/`cancel()`
    /// can cancel the native task, not just the Swift Task wrapper.
    private var liveDataTasks: [UUID: URLSessionDownloadTask] = [:]

    public init() {
        // §252: load any previously-enqueued jobs from the on-disk
        // sidecar so the user can see their downloads after app restart.
        // Entries restored as `.paused` — the user clicks Resume to
        // re-hydrate siblings from HF and continue via the existing
        // Range-header path. Corrupt sidecar = silently start empty;
        // we never want a bad persistence layer to break app launch.
        Self.loadSidecar().forEach { persisted in
            var job = persisted
            // Completed/cancelled jobs survive for history; everything
            // in-flight at the prior quit needs user confirmation
            // before resuming, so present as paused.
            if job.status == .downloading || job.status == .queued {
                job.status = .paused
                job.bytesPerSecond = 0
                job.etaSeconds = nil
            }
            _jobs[job.id] = job
            order.append(job.id)
            speedSamples[job.id] = []
        }
    }

    // MARK: - Auth

    /// Set or clear the HuggingFace access token. Pass nil to forget.
    public func setHFAuthToken(_ token: String?) {
        if let t = token, !t.isEmpty {
            self.hfAuthToken = t
        } else {
            self.hfAuthToken = nil
        }
    }

    public func hasHFAuthToken() -> Bool { hfAuthToken != nil }

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
        persistSidecar()

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
        cancelDataTask(jobId: id)
        job.status = .paused
        _jobs[id] = job
        broadcast(.paused(id))
        persistSidecar()
    }

    public func resume(_ id: UUID) {
        guard var job = _jobs[id], job.status == .paused || job.status == .failed else { return }
        job.status = .downloading
        job.error = nil
        _jobs[id] = job
        broadcast(.resumed(id))
        persistSidecar()
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
        cancelDataTask(jobId: id)
        if job.status == .completed { return }
        job.status = .cancelled
        _jobs[id] = job
        broadcast(.cancelled(id))
        persistSidecar()
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
        persistSidecar()
    }

    // MARK: - Worker

    private func run(id: UUID) async {
        guard var job = _jobs[id] else { return }

        // §253b: global slot acquire. Blocks until < maxConcurrentJobs
        // other runs are active. Released in all exit paths (success,
        // failure, cancel) via `defer`.
        await acquireSlot(for: id)
        defer { releaseSlot() }

        // Re-read the job now that we've waited — the user may have
        // cancelled or paused while we were queued.
        guard let refreshed = _jobs[id],
              refreshed.status == .downloading
        else { return }
        job = refreshed

        do {
            // 1. Enumerate files from HF API.
            let files = try await fetchSiblings(repo: job.repo)
            let total = files.reduce(Int64(0)) { $0 + ($1.size ?? 0) }
            job.totalBytes = max(total, job.totalBytes)
            _jobs[id] = job
            broadcast(.progress(job))

            // §251: pre-flight disk-space check. If HF reported a total
            // (sum of sibling `size` fields) and the destination volume
            // has less free space than 1.15× that (15% slack for HF
            // rounding + temp extraction + filesystem overhead), fail
            // fast with an actionable error instead of OOMing the disk
            // mid-download and leaving a half-complete snapshot.
            if total > 0 {
                let hubRoot = Self.huggingFaceHubRoot()
                try? FileManager.default.createDirectory(
                    at: hubRoot, withIntermediateDirectories: true)
                if let free = Self.freeSpaceBytes(at: hubRoot) {
                    let needed = Int64(Double(total) * 1.15)
                    if free < needed {
                        let needGB = Double(needed) / 1_073_741_824
                        let freeGB = Double(free) / 1_073_741_824
                        let msg = String(
                            format: "Not enough disk space — need %.1f GB, have %.1f GB free",
                            needGB, freeGB)
                        throw DownloadError.diskFull(msg)
                    }
                }
            }

            // 2. Prepare destination dir.
            let destDir = try cacheDir(for: job.repo)
            job.localPath = destDir
            _jobs[id] = job

            // 2b. Seed the progress bar with bytes already on disk from any
            //     prior (paused / crashed / resumed) attempt. This way a
            //     resume doesn't reset the bar to 0 then jump forward.
            let existingBytes = files.reduce(Int64(0)) { acc, sib in
                let dest = destDir.appendingPathComponent(sib.rfilename)
                guard FileManager.default.fileExists(atPath: dest.path) else { return acc }
                let size = (try? FileManager.default
                    .attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
                // Skip completed files (full expected size) as well as partial.
                return acc + size
            }
            if existingBytes > 0 {
                job.receivedBytes = existingBytes
                _jobs[id] = job
                broadcast(.progress(job))
            }

            // 3. Download files with max-2 concurrency. For each file we
            //    stat the on-disk bytes and pass as resumeFrom so the
            //    HTTP Range header skips what's already local.
            var index = 0
            while index < files.count {
                if Task.isCancelled { return }
                let slice = files[index..<min(index + maxConcurrentFiles, files.count)]
                try await withThrowingTaskGroup(of: Int64.self) { group in
                    for sib in slice {
                        let url = "https://huggingface.co/\(job.repo)/resolve/main/\(sib.rfilename)"
                        let dest = destDir.appendingPathComponent(sib.rfilename)
                        // Skip fully complete files entirely.
                        let existing = (try? FileManager.default
                            .attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
                        if let expected = sib.size, expected > 0, existing >= expected {
                            continue
                        }
                        let resumeFrom = existing
                        group.addTask { [weak self] in
                            guard let self else { return 0 }
                            return try await self.downloadFile(
                                jobId: id, url: url, dest: dest, resumeFrom: resumeFrom
                            )
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
                persistSidecar()
            }
        } catch is CancellationError {
            // paused or cancelled — event already broadcast.
            return
        } catch {
            // iter-94 §121: URLSession cancellation surfaces as
            // `URLError(.cancelled)` (NSURLErrorCancelled = -999), NOT
            // Swift's `CancellationError`, so the case-is match above
            // doesn't catch it. Without this second guard, user-
            // cancelled downloads briefly appeared `.cancelled` (set
            // by `cancel(_:)` synchronously) and then flipped to
            // `.failed` with cryptic message "The operation couldn't
            // be completed" once the URLSession completion handler
            // resumed with URLError.cancelled. Also catch cases
            // where the outer Task was cancelled but the error
            // propagated as something other than CancellationError.
            if let urlErr = error as? URLError, urlErr.code == .cancelled {
                return
            }
            if Task.isCancelled {
                return
            }
            // Also skip the flip if the job was already flagged
            // `.cancelled` by the user — defensive in case a future
            // refactor threads cancellation through a different
            // error type.
            if _jobs[id]?.status == .cancelled {
                return
            }
            if var j = _jobs[id] {
                j.status = .failed
                j.error = "\(error)"
                // O7 §293: detect HF auth failure and set the flag so
                // the Downloads UI can CTA the user into the
                // HuggingFaceTokenCard. Match NSError.code (which
                // fetchSiblings + file-stream throw with the HTTP
                // status on domain "vMLX.DownloadManager") OR the
                // "requires authentication" / "is gated" hint strings
                // embedded in those errors, so we catch both the
                // initial sibling fetch and any per-file 401.
                let err = error as NSError
                let msg = (err.userInfo[NSLocalizedDescriptionKey] as? String) ?? "\(error)"
                if err.code == 401 || err.code == 403
                    || msg.contains("requires authentication")
                    || msg.contains("is gated")
                {
                    j.requiresHFAuth = true
                }
                _jobs[id] = j
                broadcast(.failed(id, j.error ?? "unknown error"))
                persistSidecar()
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
        var request = URLRequest(url: url)
        if let token = hfAuthToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, resp) = try await URLSession.shared.data(for: request)
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            let hint: String
            switch http.statusCode {
            case 401: hint = "HF repo \(repo) requires authentication. Set a HuggingFace token in Settings → Downloads."
            case 403: hint = "HF repo \(repo) is gated. Accept the license on huggingface.co, then retry."
            case 404: hint = "HF repo \(repo) not found. Check the owner/name spelling."
            default:  hint = "HF API returned \(http.statusCode) for \(repo)."
            }
            throw NSError(
                domain: "vMLX.DownloadManager",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: hint]
            )
        }
        let info = try JSONDecoder().decode(ModelInfo.self, from: data)
        // Restrict to weight + config + tokenizer files, AND
        // reject anything that could escape the destination
        // directory on disk.
        return info.siblings.filter { sib in
            guard Self.isSafeFilename(sib.rfilename) else { return false }
            let f = sib.rfilename.lowercased()
            return f.hasSuffix(".safetensors")
                || f.hasSuffix(".json")
                || f.hasSuffix(".txt")
                || f.hasSuffix(".model")
                || f.hasSuffix(".jinja")
        }
    }

    /// **iter-82 (§110)** — path-traversal guard for the filename
    /// coming back from the HuggingFace API. A compromised,
    /// man-in-the-middled, or malicious-mirror HF API response
    /// could set `rfilename` to `"../../../.ssh/authorized_keys"`;
    /// `URL.appendingPathComponent` preserves `..` literally but
    /// POSIX path resolution collapses them, so a write would
    /// land OUTSIDE `destDir`. This check rejects any filename
    /// that starts with `/`, contains `..` as a path component,
    /// or is empty/whitespace — defense in depth independent of
    /// TLS validation on the transport.
    internal static func isSafeFilename(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return false }
        // Absolute path starting at root.
        if trimmed.hasPrefix("/") { return false }
        // Windows-style absolute or UNC path.
        if trimmed.hasPrefix("\\") { return false }
        // Path-traversal segment anywhere in the component chain.
        // Split on both `/` and `\` to defend against cross-platform
        // tricks; any segment equal to `..` is a traversal.
        let segments = trimmed.split(whereSeparator: { $0 == "/" || $0 == "\\" })
        for seg in segments where seg == ".." {
            return false
        }
        // Null byte — POSIX truncates filename at `\0`; reject
        // rather than silently writing to a truncated path.
        if trimmed.contains("\0") { return false }
        return true
    }

    // MARK: - File download
    //
    // Streams via `URLSessionDownloadTask` so the OS writes directly to a
    // temp file at its native chunk size (typically 64-128KB). KVO on
    // `task.progress.completedUnitCount` fires at whatever interval the OS
    // chooses — usually 10-50 times per second — and forwards the raw byte
    // delta into the actor. The previous implementation iterated the
    // `URLSession.AsyncBytes` sequence one byte at a time, which cost one
    // async hop per byte and capped real throughput well below gigabit.
    //
    // Range resume: if `resumeFrom > 0` we send `Range: bytes=<n>-` and
    // append the 206 Partial Content body to the existing file. On 416 we
    // treat the file as already complete. On ETag mismatch the caller
    // should delete the partial file and retry.

    private func downloadFile(
        jobId: UUID,
        url: String,
        dest: URL,
        resumeFrom: Int64 = 0
    ) async throws -> Int64 {
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        guard let u = URL(string: url) else { throw URLError(.badURL) }

        var request = URLRequest(url: u)
        if let token = hfAuthToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if resumeFrom > 0 {
            request.setValue("bytes=\(resumeFrom)-", forHTTPHeaderField: "Range")
        }

        // Bridge the delegate-free `downloadTask` callback into async/await
        // while installing a KVO observer for real-time progress. The
        // observer runs on an arbitrary queue; we bounce each delta back
        // into the actor via a detached Task.
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Int64, Error>) in
                let holder = ProgressHolder()
                let task = URLSession.shared.downloadTask(with: request) { tmpURL, response, error in
                    holder.observation?.invalidate()
                    Task { [jobId] in
                        await self.unregisterDataTask(jobId: jobId)
                    }
                    if let error = error {
                        cont.resume(throwing: error)
                        return
                    }
                    guard let tmpURL = tmpURL else {
                        cont.resume(throwing: URLError(.cannotCreateFile))
                        return
                    }
                    do {
                        if let http = response as? HTTPURLResponse {
                            if http.statusCode == 416 {
                                // Range not satisfiable — partial is already complete.
                                try? FileManager.default.removeItem(at: tmpURL)
                                cont.resume(returning: 0)
                                return
                            }
                            if http.statusCode >= 400 {
                                try? FileManager.default.removeItem(at: tmpURL)
                                let hint: String
                                switch http.statusCode {
                                case 401: hint = "HF file requires authentication."
                                case 403: hint = "HF file is gated — accept the license and retry."
                                default:  hint = "HTTP \(http.statusCode) downloading \(u.lastPathComponent)."
                                }
                                cont.resume(throwing: NSError(
                                    domain: "vMLX.DownloadManager",
                                    code: http.statusCode,
                                    userInfo: [NSLocalizedDescriptionKey: hint]
                                ))
                                return
                            }
                        }
                        let appended: Int64
                        if resumeFrom > 0,
                           FileManager.default.fileExists(atPath: dest.path)
                        {
                            // Append the 206 body to the existing partial file.
                            let partData = try Data(contentsOf: tmpURL, options: .mappedIfSafe)
                            let handle = try FileHandle(forWritingTo: dest)
                            try handle.seekToEnd()
                            try handle.write(contentsOf: partData)
                            try handle.close()
                            try? FileManager.default.removeItem(at: tmpURL)
                            appended = Int64(partData.count)
                        } else {
                            if FileManager.default.fileExists(atPath: dest.path) {
                                try FileManager.default.removeItem(at: dest)
                            }
                            try FileManager.default.moveItem(at: tmpURL, to: dest)
                            let size = (try FileManager.default
                                .attributesOfItem(atPath: dest.path)[.size] as? Int64) ?? 0
                            appended = size
                        }
                        cont.resume(returning: appended)
                    } catch {
                        cont.resume(throwing: error)
                    }
                }

                // Real-time progress: translate KVO deltas into actor calls.
                holder.observation = task.progress.observe(\.completedUnitCount) { [jobId] progress, _ in
                    let current = Int64(progress.completedUnitCount)
                    let delta = holder.consume(newValue: current)
                    if delta > 0 {
                        Task { await self.addBytes(jobId: jobId, delta: delta) }
                    }
                }

                Task { [jobId] in
                    // iter-84: `Task { }` inside an actor method
                    // inherits the actor's isolation, so
                    // `self.registerDataTask` doesn't need an actor
                    // hop. Drop the redundant await to silence
                    // "no 'async' operations occur" warning.
                    self.registerDataTask(jobId: jobId, task: task)
                }
                task.resume()
            }
        } onCancel: {
            Task { [jobId] in await self.cancelDataTask(jobId: jobId) }
        }
    }

    // MARK: - Data-task lifecycle bridging

    private func registerDataTask(jobId: UUID, task: URLSessionDownloadTask) {
        liveDataTasks[jobId] = task
    }

    private func unregisterDataTask(jobId: UUID) {
        liveDataTasks.removeValue(forKey: jobId)
    }

    private func cancelDataTask(jobId: UUID) {
        liveDataTasks[jobId]?.cancel()
        liveDataTasks.removeValue(forKey: jobId)
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

    // `resumeTokenDir` was removed 2026-04-15 per audit finding (lifecycle
    // #6). The helper was declared but never referenced — partial-download
    // state is derived entirely from on-disk file size + a `Range` header,
    // which is simpler and survives restart without a sidecar token file.
    // NSURLSession-native resume data was never wired up, so the helper
    // was misleading dead code.

    // MARK: - §251 disk-space helpers

    // MARK: - §253b slot coordination

    /// Wait until fewer than `maxConcurrentJobs` runs are active, then
    /// increment `activeJobs`. FIFO ordering keyed by UUID so enqueue
    /// order is preserved.
    private func acquireSlot(for id: UUID) async {
        if activeJobs < maxConcurrentJobs {
            activeJobs += 1
            return
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            waiters[id] = cont
        }
        activeJobs += 1
    }

    /// Release the slot and wake the next waiter (if any). Called
    /// once per acquire; safe to call multiple times due to the
    /// `firstKey` check — never double-resumes a continuation.
    private func releaseSlot() {
        activeJobs = max(0, activeJobs - 1)
        // Take the oldest waiter — keyed on insertion order since
        // Swift dicts are unordered, we pick the first UUID
        // deterministically by sorting. Waiter-set size is O(enqueued),
        // almost always <5, so the sort cost is negligible.
        guard let next = waiters.keys.sorted().first else { return }
        let cont = waiters.removeValue(forKey: next)!
        cont.resume()
    }

    // MARK: - §252 sidecar persistence

    /// Location of the jobs sidecar. Written next to SettingsStore so
    /// cleanup (`rm -rf "~/Library/Application Support/vMLX"`) clears
    /// both. Created lazily on first write.
    private static func sidecarURL() -> URL {
        // Test override: `VMLX_SIDECAR_DIR` lets tests point at a
        // scratch dir without clobbering real user state. FileManager's
        // `.applicationSupportDirectory` URL isn't redirected by $HOME
        // on macOS, so env override is the cleanest seam.
        if let override = ProcessInfo.processInfo.environment["VMLX_SIDECAR_DIR"],
           !override.isEmpty
        {
            return URL(fileURLWithPath: override)
                .appendingPathComponent("downloads.json")
        }
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("vMLX/downloads.json")
    }

    /// Persist the full job list. Called after every status-changing
    /// event — enqueue, pause, resume, cancel, complete, progress tick
    /// (rate-limited by caller). JSON is small (<50KB for 100 jobs)
    /// so atomic overwrite is fine; no WAL needed.
    nonisolated private static func writeSidecar(_ jobs: [Job]) {
        let url = sidecarURL()
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let payload = SidecarPayload(version: 1, jobs: jobs)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(payload) {
            try? data.write(to: url, options: .atomic)
        }
    }

    /// Load jobs from disk. Returns empty on missing/corrupt file.
    nonisolated private static func loadSidecar() -> [Job] {
        let url = sidecarURL()
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let payload = try? decoder.decode(SidecarPayload.self, from: data)
        else { return [] }
        return payload.jobs
    }

    /// Trigger a sidecar write for the current state. Fire-and-forget;
    /// the write itself is async off-actor to avoid blocking the
    /// download hot path on disk IO.
    private func persistSidecar() {
        let snapshot = order.compactMap { _jobs[$0] }
        Task.detached(priority: .utility) {
            Self.writeSidecar(snapshot)
        }
    }

    private struct SidecarPayload: Codable {
        let version: Int
        let jobs: [Job]
    }

    /// Root directory where HuggingFace snapshots land. Used for the
    /// pre-flight disk check so we probe the correct volume.
    static func huggingFaceHubRoot() -> URL {
        FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// Available free bytes on the volume that hosts `url`. Returns nil
    /// if the FS attributes call fails (e.g. volume unmounted).
    nonisolated static func freeSpaceBytes(at url: URL) -> Int64? {
        let values = try? url.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }
}

/// §251 — download-manager-level errors. Wrapped in a separate enum so
/// callers can surface a disk-full banner vs. a generic "download
/// failed" message.
public enum DownloadError: Error, LocalizedError {
    case diskFull(String)
    public var errorDescription: String? {
        switch self {
        case .diskFull(let m): return m
        }
    }
}

/// Thread-safe running total for KVO progress observation.
///
/// `URLSessionTask.progress.observe(...)` fires from an arbitrary queue, so
/// we can't close over a captured `var`. This little holder gives us an
/// NSLock-guarded counter plus a slot for the observation token so the
/// completion handler can invalidate it once the task finishes.
private final class ProgressHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var lastValue: Int64 = 0
    var observation: NSKeyValueObservation?

    /// Returns the delta since the last call, updating the stored value.
    func consume(newValue: Int64) -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        let delta = newValue - lastValue
        if delta > 0 { lastValue = newValue }
        return delta
    }
}
