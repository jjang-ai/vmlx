import Foundation
import CryptoKit

/// Model library — discovery + metadata index for every model on disk vMLX
/// can load. Mirrors the Python-side behaviour in `panel/src/main/process/
/// modelScanner.ts` (scan HF cache + user dirs) and `vmlx_engine/utils/
/// jang_loader.py` (JANG/MXTQ detection), but reimplemented in pure Swift
/// against `ModelDetector` for the three-tier `model_type` discovery.
///
/// Actor-isolated — all disk walking and DB I/O happen inside `scan()`, and
/// the in-memory cache + subscriber list are mutated only on the actor.
public actor ModelLibrary {

    // MARK: - Types

    public struct ModelEntry: Sendable, Identifiable, Hashable {
        public let id: String                    // sha256(canonicalPath)[0..<32]
        public let canonicalPath: URL
        public let displayName: String           // "org/repo" or dir basename
        public let family: String                // e.g. "qwen3", "mistral4", "gemma4"
        public let modality: Modality
        public let totalSizeBytes: Int64
        public let isJANG: Bool
        public let isMXTQ: Bool
        public let quantBits: Int?               // 2/3/4/5/6/8 or nil (fp16)
        public let detectedAt: Date
        public let source: Source
        public let capabilities: ModelCapabilities

        public init(
            id: String,
            canonicalPath: URL,
            displayName: String,
            family: String,
            modality: Modality,
            totalSizeBytes: Int64,
            isJANG: Bool,
            isMXTQ: Bool,
            quantBits: Int?,
            detectedAt: Date,
            source: Source,
            capabilities: ModelCapabilities = .unknown
        ) {
            self.id = id
            self.canonicalPath = canonicalPath
            self.displayName = displayName
            self.family = family
            self.modality = modality
            self.totalSizeBytes = totalSizeBytes
            self.isJANG = isJANG
            self.isMXTQ = isMXTQ
            self.quantBits = quantBits
            self.detectedAt = detectedAt
            self.source = source
            self.capabilities = capabilities
        }

        public func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }

        public static func == (lhs: ModelEntry, rhs: ModelEntry) -> Bool {
            lhs.id == rhs.id
                && lhs.canonicalPath == rhs.canonicalPath
                && lhs.displayName == rhs.displayName
                && lhs.family == rhs.family
                && lhs.modality == rhs.modality
                && lhs.totalSizeBytes == rhs.totalSizeBytes
                && lhs.isJANG == rhs.isJANG
                && lhs.isMXTQ == rhs.isMXTQ
                && lhs.quantBits == rhs.quantBits
                && lhs.detectedAt == rhs.detectedAt
                && lhs.source == rhs.source
                && lhs.capabilities == rhs.capabilities
        }
    }

    public enum Modality: String, Sendable, Hashable {
        case text, vision, embedding, image, rerank, unknown
    }

    public enum Source: Sendable, Hashable {
        case hfCache
        case userDir(URL)
        case downloaded
    }

    // MARK: - State

    private let database: ModelLibraryDB
    private var cache: [ModelEntry] = []
    private var subscribers: [UUID: AsyncStream<[ModelEntry]>.Continuation] = [:]

    /// If last scan finished within this window, `scan(force:false)` returns the
    /// cached result without walking disk again. Tuned for "user clicks Server
    /// tab repeatedly" use-case.
    private let freshnessWindow: TimeInterval = 300  // 5 minutes

    /// Disk-walk spy — injected for unit tests to count disk walks.
    internal var _onDiskWalk: (@Sendable () -> Void)?

    /// FSEvents watcher — started on the first `scan()` call. Fires a
    /// debounced `scan(force: true)` whenever the HF cache or any user-added
    /// dir changes on disk. Held as a strong reference until the library is
    /// torn down; the watcher's own deinit tears down the FSEventStream.
    private var watcher: ModelLibraryWatcher? = nil

    // MARK: - Init

    /// Set by `addUserDir` to invalidate the next freshness check. When a
    /// new user dir is added we MUST rescan regardless of how recently
    /// the last scan ran — otherwise models under the newly added dir
    /// never show up in `entries()` until the 5-minute window elapses.
    private var pendingForcedRescan: Bool = false

    public init(database: ModelLibraryDB) {
        self.database = database
        self.cache = database.all()

        // Auto-register `~/.mlxstudio/models/` as a user dir if present.
        // This is the legacy Electron-app models home; many users have
        // JANG models downloaded there outside the HuggingFace cache.
        // Without this the `GET /v1/models` route would miss them.
        // Idempotent — the DB's upsert ignores duplicates.
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            let legacy = URL(fileURLWithPath: home)
                .appendingPathComponent(".mlxstudio/models")
            if FileManager.default.fileExists(atPath: legacy.path) {
                let existing = Set(database.userDirs().map { $0.standardizedFileURL.path })
                if !existing.contains(legacy.standardizedFileURL.path) {
                    database.addUserDir(legacy)
                    pendingForcedRescan = true
                }
            }
        }
    }

    // MARK: - Public API

    public func scan(force: Bool = false) async -> [ModelEntry] {
        startWatcherIfNeeded()
        let shouldForce = force || pendingForcedRescan
        if shouldForce {
            pendingForcedRescan = false
        }
        if !shouldForce, isFresh() {
            cache = database.all()
            return cache
        }
        _onDiskWalk?()

        let now = Date()
        let userDirs = database.userDirs()
        var discovered: [ModelEntry] = []
        var seenIds = Set<String>()

        // HF cache: ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/
        for url in walkHFCache() {
            if let entry = buildEntry(dir: url, source: .hfCache, now: now) {
                if seenIds.insert(entry.id).inserted {
                    discovered.append(entry)
                }
            }
        }

        // User-added dirs, depth 3
        for root in userDirs {
            for url in walkUserDir(root, maxDepth: 3) {
                if let entry = buildEntry(dir: url, source: .userDir(root), now: now) {
                    if seenIds.insert(entry.id).inserted {
                        discovered.append(entry)
                    }
                }
            }
        }

        // Diff against DB → upsert + purge
        let existing = database.all()
        let existingIds = Set(existing.map(\.id))
        let newIds = Set(discovered.map(\.id))

        for e in discovered { database.upsert(e) }
        let toPurge = existingIds.subtracting(newIds)
        database.purge(toPurge)

        cache = database.all()
        broadcast()
        return cache
    }

    public func entries() -> [ModelEntry] {
        cache
    }

    public func entry(byId id: String) -> ModelEntry? {
        cache.first(where: { $0.id == id }) ?? database.byId(id)
    }

    /// Delete a model's on-disk files and drop it from the library cache.
    ///
    /// Fixes vmlx #57 (OPEN): no user-facing way to remove downloaded models.
    /// The entry's `canonicalPath` directory is removed recursively via
    /// `FileManager`, the row is dropped from the DB, the in-memory cache is
    /// pruned, and subscribers are notified. Returns `true` on success.
    ///
    /// Safety: refuses to delete anything whose `canonicalPath` is not a
    /// subdirectory of a known models location (HF cache or a registered
    /// user dir), to avoid accidental rm -rf on unrelated paths if the DB
    /// row is stale.
    @discardableResult
    public func deleteEntry(byId id: String) async throws -> Bool {
        guard let entry = entry(byId: id) else { return false }

        let path = entry.canonicalPath
        let fm = FileManager.default

        // Safety fence: the directory must live under a known roots list.
        let knownRoots: [URL] = {
            var roots: [URL] = []
            if let home = ProcessInfo.processInfo.environment["HOME"] {
                roots.append(URL(fileURLWithPath: home)
                    .appendingPathComponent(".cache/huggingface/hub"))
                roots.append(URL(fileURLWithPath: home)
                    .appendingPathComponent(".mlxstudio/models"))
            }
            roots.append(contentsOf: database.userDirs())
            return roots.map { $0.standardizedFileURL }
        }()
        let target = path.standardizedFileURL
        let isUnderKnown = knownRoots.contains { root in
            target.path.hasPrefix(root.path + "/")
                || target.path == root.path
        }
        guard isUnderKnown else {
            throw EngineError.unsupportedModelType(
                "refuse to delete \(target.path): not under a known model root")
        }

        guard fm.fileExists(atPath: target.path) else {
            // Already gone on disk — still drop the DB + cache entries.
            database.purge([id])
            cache.removeAll { $0.id == id }
            broadcast()
            return true
        }

        try fm.removeItem(at: target)
        database.purge([id])
        cache.removeAll { $0.id == id }
        broadcast()
        return true
    }

    public func subscribe() -> AsyncStream<[ModelEntry]> {
        let id = UUID()
        return AsyncStream { cont in
            self.subscribers[id] = cont
            cont.yield(self.cache)
            cont.onTermination = { [weak self] _ in
                Task { await self?.removeSubscriber(id) }
            }
        }
    }

    private func removeSubscriber(_ id: UUID) {
        subscribers[id] = nil
    }

    public func addUserDir(_ url: URL) {
        database.addUserDir(url)
    }

    public func removeUserDir(_ url: URL) {
        database.removeUserDir(url)
    }

    public func userDirs() -> [URL] {
        database.userDirs()
    }

    // MARK: - Freshness

    private func isFresh() -> Bool {
        guard let last = database.mostRecentDetectedAt() else { return false }
        return Date().timeIntervalSince(last) < freshnessWindow
    }

    private func broadcast() {
        for (_, cont) in subscribers {
            cont.yield(cache)
        }
    }

    // MARK: - Disk walking

    private func walkHFCache() -> [URL] {
        let fm = FileManager.default
        let hub = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        guard fm.fileExists(atPath: hub.path) else { return [] }

        var out: [URL] = []
        guard let topLevel = try? fm.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: [.isDirectoryKey]) else { return [] }

        // ONE snapshot per models--<org>--<repo> directory. HF cache routinely
        // accumulates multiple revs per repo (a fully-downloaded commit plus
        // partials from subsequent metadata-only fetches). Before 2026-04-16
        // we emitted every rev with a config.json — which surfaced empty
        // stub snapshots in the picker, and clicking one hung the loader at
        // "100% Ready" on a random-weight graph (see Load.swift empty-weight
        // guard for the downstream symptom). Now we pick exactly one rev
        // per repo using this order:
        //   1. If `refs/main` exists and the referenced snapshot has ≥1
        //      weight file (.safetensors / .bin / .gguf), use it.
        //   2. Otherwise, pick the snapshot with the largest aggregate
        //      weight size. Ties broken by newest mtime.
        //   3. If no snapshot has any weights, emit NONE (the repo is a
        //      metadata-only stub — don't clutter the picker).
        for modelDir in topLevel where modelDir.lastPathComponent.hasPrefix("models--") {
            let snapshots = modelDir.appendingPathComponent("snapshots", isDirectory: true)
            guard let revs = try? fm.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil) else { continue }

            // Try refs/main first — the "canonical" rev for this repo.
            var refMainURL: URL? = nil
            let refsMain = modelDir
                .appendingPathComponent("refs", isDirectory: true)
                .appendingPathComponent("main")
            if let sha = try? String(contentsOf: refsMain, encoding: .utf8) {
                let trimmed = sha.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    let candidate = snapshots.appendingPathComponent(trimmed, isDirectory: true)
                    let cfg = candidate.appendingPathComponent("config.json")
                    if fm.fileExists(atPath: cfg.path),
                       totalWeightBytes(in: candidate) > 0
                    {
                        refMainURL = candidate
                    }
                }
            }
            if let picked = refMainURL {
                out.append(picked)
                continue
            }

            // Fall back to largest-weight snapshot.
            var best: (url: URL, size: Int64, mtime: Date)? = nil
            for rev in revs {
                let cfg = rev.appendingPathComponent("config.json")
                guard fm.fileExists(atPath: cfg.path) else { continue }
                let size = totalWeightBytes(in: rev)
                guard size > 0 else { continue }
                let mtime = (try? fm.attributesOfItem(atPath: rev.path))?[.modificationDate]
                    as? Date ?? Date.distantPast
                if let cur = best {
                    if size > cur.size || (size == cur.size && mtime > cur.mtime) {
                        best = (rev, size, mtime)
                    }
                } else {
                    best = (rev, size, mtime)
                }
            }
            if let picked = best?.url {
                out.append(picked)
            }
            // else: repo is metadata-only — skip silently so the picker
            // doesn't show a dead entry.
        }
        return out
    }

    private func walkUserDir(_ root: URL, maxDepth: Int) -> [URL] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path) else { return [] }

        var out: [URL] = []
        // BFS with depth tracking — avoids the recursive enumerator's
        // unbounded descent into node_modules-style subtrees.
        var queue: [(URL, Int)] = [(root, 0)]
        // Sub-module names used by diffusion pipelines (Z-Image, Flux, SD).
        // Each of these ships its own `config.json` but isn't a standalone
        // model — they're referenced by a parent `model_index.json`. Without
        // this guard the library pollutes with "text_encoder", "transformer",
        // "vae" stub entries.
        let diffusionSubmodules: Set<String> = [
            "transformer", "text_encoder", "text_encoder_2", "text_encoder_3",
            "vae", "unet", "scheduler", "tokenizer", "tokenizer_2",
            "safety_checker", "feature_extractor", "image_encoder",
        ]
        while let (dir, depth) = queue.popLast() {
            if fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) {
                // Reject diffusion sub-module dirs (parent owns model_index.json).
                let name = dir.lastPathComponent.lowercased()
                let parent = dir.deletingLastPathComponent()
                let parentIsPipeline = fm.fileExists(
                    atPath: parent.appendingPathComponent("model_index.json").path)
                if diffusionSubmodules.contains(name) || parentIsPipeline {
                    continue
                }
                out.append(dir)
                // Don't descend into a matched model dir.
                continue
            }
            if depth >= maxDepth { continue }
            guard let kids = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            for k in kids {
                var isDir: ObjCBool = false
                if fm.fileExists(atPath: k.path, isDirectory: &isDir), isDir.boolValue {
                    queue.append((k, depth + 1))
                }
            }
        }
        return out
    }

    // MARK: - Entry construction

    private func buildEntry(dir: URL, source: Source, now: Date) -> ModelEntry? {
        let canonical = dir.resolvingSymlinksInPath()
        // CapabilityDetector is the source of truth for family + parser
        // metadata. ModelDetector is still used internally by the engine
        // load path; the two agree on model_type resolution.
        let caps = CapabilityDetector.detect(at: dir)
        let family = caps.family == "unknown" ? caps.modelType : caps.family
        var modality: Modality = {
            switch caps.modality {
            case .vision: return .vision
            case .embedding: return .embedding
            case .image: return .image
            case .rerank: return .rerank
            case .unknown: return .unknown
            case .text: return .text
            }
        }()

        // Pull config.json for JANG/MXTQ/quant bits + modality refinements.
        let cfgURL = dir.appendingPathComponent("config.json")
        let json = (try? Data(contentsOf: cfgURL))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]

        let jangCfgURL = dir.appendingPathComponent("jang_config.json")
        let jangJson = (try? Data(contentsOf: jangCfgURL))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]

        let isJANG = FileManager.default.fileExists(atPath: jangCfgURL.path)
            || json["jang_config"] != nil
            || json["jang"] != nil

        let isMXTQ = detectMXTQ(config: json, jang: jangJson)
        let quantBits = detectQuantBits(config: json, jang: jangJson)

        // Modality refinement: embedding / image / rerank best-effort.
        if let mt = json["model_type"] as? String {
            let lower = mt.lowercased()
            if lower.contains("embed") || lower == "bert" || lower == "xlm-roberta" {
                modality = .embedding
            } else if lower.contains("rerank") {
                modality = .rerank
            }
        }
        // Image: diffusion pipelines ship a `model_index.json`, not `config.json`.
        // Most of those won't be picked up by this scanner (no config.json), but
        // if a user points at a Flux/SD dir that has a config.json inside the
        // transformer subdir, family will still be text — flag by path hint.
        let lastComponent = dir.lastPathComponent.lowercased()
        if lastComponent.contains("flux")
            || lastComponent.contains("z-image")
            || lastComponent.contains("sdxl")
            || lastComponent.contains("schnell") {
            modality = .image
        }

        let displayName = deriveDisplayName(dir: dir, source: source, config: json)
        let sizeBytes = totalWeightBytes(in: dir)
        // Zero-weight guard: any dir that has config.json but no
        // .safetensors/.bin/.gguf shards is a stub — either a partial HF
        // download, a standalone tokenizer repo, or a config-only metadata
        // release. Emitting it into the picker means a user can click it
        // and hang the loader evaluating uninitialized parameters. Image
        // / diffusion pipelines are the one exception: they don't carry
        // top-level weight files (the transformer / vae subdirs do), so
        // allow them through.
        if sizeBytes == 0 && modality != .image {
            return nil
        }
        let id = sha256(canonical.path)

        return ModelEntry(
            id: id,
            canonicalPath: canonical,
            displayName: displayName,
            family: family,
            modality: modality,
            totalSizeBytes: sizeBytes,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            quantBits: quantBits,
            detectedAt: now,
            source: source,
            capabilities: caps
        )
    }

    /// Resolve a human-readable display name for a model directory. Tries
    /// in order:
    ///
    ///   1. HF cache layout (`.../models--<org>--<repo>/snapshots/<rev>`) —
    ///      walks up two parents and rebuilds `org/repo`. Handles repo
    ///      names with embedded hyphens correctly by only splitting on
    ///      the FIRST `--` after the `models--` prefix.
    ///
    ///   2. `config.json` `_name_or_path` field — every HF model ships
    ///      this; a clean `org/repo` string. Best fallback when the dir
    ///      layout is non-standard (vendored checkouts, user-added dirs
    ///      that point directly at a snapshot hash).
    ///
    ///   3. `jang_config.json` `model_name` field — JANG models stamp
    ///      this during conversion so we honor it next.
    ///
    ///   4. If the dir itself looks like a snapshot hash (40-char hex),
    ///      walk up one parent and retry. Prevents the UI from ever
    ///      showing something like `11de96878523501bcaa86104e3c186de07ff9068`.
    ///
    ///   5. Final fallback: the dir's last path component as-is.
    private func deriveDisplayName(
        dir: URL,
        source: Source,
        config: [String: Any]
    ) -> String {
        // User-owned dirs: the FOLDER NAME is authoritative. The user
        // chose to name the dir `Qwen3.6-35B-A3B-JANGTQ2` or
        // `MiniMax-M2.7-JANGTQ-CRACK` — those quant/crack suffixes are
        // how they tell variants apart in the model list. Reaching
        // into `config.json._name_or_path` or `jang_config.json`
        // strips those suffixes (both point at the upstream base name
        // like `Qwen3.6-35B-A3B`) and makes every variant of the same
        // base show up as a "duplicate" with identical display name.
        // Walk up if the leaf is a hash (unlikely for userDir).
        if case .userDir = source {
            let last = dir.lastPathComponent
            if !looksLikeHash(last) && !last.isEmpty {
                return last
            }
            let up = dir.deletingLastPathComponent().lastPathComponent
            if !up.isEmpty && !looksLikeHash(up) { return up }
            return last.isEmpty ? dir.path : last
        }

        // HF cache layout. Split on the FIRST `--` after `models--`
        // so repos with hyphens (`Qwen3.5-VL-307B-A17B`) don't get
        // mangled. For HF paths we keep the original 3-field
        // resolution because `snapshots/<SHA>/` is NOT a usable
        // display name.
        let parent = dir.deletingLastPathComponent()        // snapshots/
        let modelDir = parent.deletingLastPathComponent()   // models--org--repo
        let name = modelDir.lastPathComponent
        if name.hasPrefix("models--") {
            let trimmed = String(name.dropFirst("models--".count))
            if let sep = trimmed.range(of: "--") {
                let org = String(trimmed[..<sep.lowerBound])
                let repo = String(trimmed[sep.upperBound...])
                if !org.isEmpty, !repo.isEmpty {
                    return "\(org)/\(repo)"
                }
            }
        }

        // HF non-standard layout fallbacks (snapshot-hash leaf or
        // user-added HF-like dirs). Prefer structured metadata over
        // the path because path info here is a hash.
        if let np = config["_name_or_path"] as? String,
           !np.isEmpty, !np.hasPrefix("/"), !looksLikeHash(np)
        {
            return np
        }
        if let jc = config["jang_config"] as? [String: Any],
           let mn = jc["model_name"] as? String,
           !mn.isEmpty, !looksLikeHash(mn)
        {
            return mn
        }
        let jangURL = dir.appendingPathComponent("jang_config.json")
        if let jdata = try? Data(contentsOf: jangURL),
           let jobj = try? JSONSerialization.jsonObject(with: jdata) as? [String: Any]
        {
            if let sm = jobj["source_model"] as? [String: Any],
               let nm = sm["name"] as? String,
               !nm.isEmpty, !looksLikeHash(nm)
            {
                return nm
            }
            if let nm = jobj["model_name"] as? String,
               !nm.isEmpty, !looksLikeHash(nm)
            {
                return nm
            }
        }

        let last = dir.lastPathComponent
        if looksLikeHash(last) {
            let up = parent.lastPathComponent
            if !up.isEmpty, !looksLikeHash(up), up != "snapshots" {
                return up
            }
            let upUp = modelDir.lastPathComponent
            if !upUp.isEmpty, !looksLikeHash(upUp) {
                return upUp
            }
        }

        return last
    }

    /// Heuristic: 32+ hex chars → commit/snapshot hash, not a real name.
    private func looksLikeHash(_ s: String) -> Bool {
        guard s.count >= 32 else { return false }
        return s.allSatisfy { $0.isHexDigit }
    }

    private func totalWeightBytes(in dir: URL) -> Int64 {
        let fm = FileManager.default
        // Follow symlinks (HF cache snapshots → blobs/).
        guard let enumerator = fm.enumerator(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: []
        ) else { return 0 }

        var total: Int64 = 0
        for case let url as URL in enumerator {
            let ext = url.pathExtension.lowercased()
            guard ext == "safetensors" || ext == "bin" || ext == "gguf" else { continue }
            // Resolve symlinks so we count the blob, not the link size.
            let resolved = url.resolvingSymlinksInPath()
            if let attrs = try? fm.attributesOfItem(atPath: resolved.path),
               let size = attrs[.size] as? NSNumber {
                total += size.int64Value
            }
        }
        return total
    }

    private func detectMXTQ(config: [String: Any], jang: [String: Any]) -> Bool {
        if let q = config["quantization"] as? [String: Any],
           let method = q["method"] as? String,
           method.lowercased().contains("mxtq") { return true }
        if let q = jang["quantization"] as? [String: Any],
           let method = q["method"] as? String,
           method.lowercased().contains("mxtq") { return true }
        if config["mxtq_seed"] != nil || config["mxtq_bits"] != nil { return true }
        if jang["mxtq_seed"] != nil || jang["mxtq_bits"] != nil { return true }
        return false
    }

    private func detectQuantBits(config: [String: Any], jang: [String: Any]) -> Int? {
        if let q = config["quantization"] as? [String: Any],
           let b = q["bits"] as? Int { return b }
        if let q = jang["quantization"] as? [String: Any] {
            if let widths = q["bit_widths_used"] as? [Int], let first = widths.first {
                return first
            }
            if let b = q["bits"] as? Int { return b }
        }
        return nil
    }

    private func sha256(_ s: String) -> String {
        let digest = SHA256.hash(data: Data(s.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(32))
    }

    // MARK: - Test hooks

    internal func _setDiskWalkSpy(_ spy: (@Sendable () -> Void)?) {
        self._onDiskWalk = spy
    }

    /// Test hook: swap in a watcher whose `_simulateEvent()` can be called
    /// from XCTest without waiting for real FSEvents.
    internal func _installWatcher(_ w: ModelLibraryWatcher) {
        self.watcher = w
    }

    internal func _currentWatcher() -> ModelLibraryWatcher? { watcher }

    // MARK: - Watcher

    /// Lazily instantiate the FSEvents watcher. Watches the HF cache dir
    /// (whether or not it exists yet — FSEvents tolerates non-existent roots
    /// once the parent is present) plus every user-added dir.
    private func startWatcherIfNeeded() {
        if watcher != nil { return }
        var roots: [URL] = []
        let fm = FileManager.default
        let hfCache = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        if fm.fileExists(atPath: hfCache.path) { roots.append(hfCache) }
        for dir in database.userDirs() where fm.fileExists(atPath: dir.path) {
            roots.append(dir)
        }
        guard !roots.isEmpty else { return }
        watcher = ModelLibraryWatcher(paths: roots) { [weak self] in
            guard let self else { return }
            _ = await self.scan(force: true)
        }
    }
}
