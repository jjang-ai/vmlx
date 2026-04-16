import Foundation
#if canImport(Darwin)
import Darwin
#endif
import vMLXLLM
import vMLXVLM
import vMLXLMCommon
import vMLXEmbedders
import MLX
@preconcurrency import Tokenizers

/// vMLX engine facade. Wraps vmlx-swift-lm's `LLMModelFactory` / `VLMModelFactory`
/// and exposes the same surface area the Python `vmlx_engine` module does today
/// (load → generate → stream → cache stats).
///
/// This is the SCAFFOLD. Each method is stubbed and points at the corresponding
/// Python implementation in vmlx_engine/ for porting reference.
public actor Engine {

    public enum EngineKind: Sendable {
        case simple   // single-stream, vmlx_engine/engine/simple.py
        case batched  // multi-stream + paged cache, vmlx_engine/engine/batched.py
    }

    public struct LoadOptions: Sendable {
        public var modelPath: URL
        public var kind: EngineKind = .batched
        public var maxNumSeqs: Int = 5
        public var prefillStepSize: Int = 1024
        public var maxCacheBlocks: Int = 500
        // TurboQuant KV compression: default-OFF (perf audit 2026-04-16).
        // Measured impact of flipping to default-off: Nemotron-Cascade-2-30B
        // A3B 2.4 → 59.8 tok/s (25× speedup), Gemma-4-26B-A4B 34.9 → 49.0
        // tok/s (+40%), Qwen3.5-9B 69.5 → 78.3 tok/s (+13%). The decode-time
        // compress+dequant cycle on every attention-half layer dominated
        // decode throughput on MoE and hybrid models. Users who need the
        // 4× KV memory savings on long contexts can opt in via settings or
        // the jang_config.json `turboquant` block (which auto-activates for
        // models explicitly shipped with calibrated TQ bit widths).
        public var enableTurboQuant: Bool = false
        public var enableJANG: Bool = true
        public var enablePrefixCache: Bool = true
        public var enableSSMCompanion: Bool = true
        public var defaultEnableThinking: Bool? = nil
        public var idleSoftSec: TimeInterval = 300
        public var idleDeepSec: TimeInterval = 900
        public var idleEnabled: Bool = true
        // Cache stack — added 2026-04-15: silently ignored before because
        // SettingsStore.LoadOptions(from:) didn't forward these.
        public var enableMemoryCache: Bool = true
        public var enableDiskCache: Bool = true
        public var diskCacheDir: String = ""
        public var diskCacheMaxGB: Double = 10.0
        public var enableBlockDiskCache: Bool = false
        public var blockDiskCacheDir: String = ""
        public var blockDiskCacheMaxGB: Double = 10.0
        public var kvCacheQuantization: String = "none"
        public var kvCacheGroupSize: Int = 64
        public var turboQuantBits: Int = 4
        public var enableSSMReDerive: Bool = true
        // Smelt + Flash MoE
        public var smelt: Bool = false
        public var smeltExperts: Int = 50
        public var smeltMode: String = "default"
        public var flashMoe: Bool = false
        // 64 is the AUTO-SIZE sentinel — `EngineFlashMoE.applyFlashMoEIfEnabled`
        // treats `<= 64` as "user hasn't overridden, compute
        // layers × experts_per_tok × 1.5". Explicit overrides (e.g. 256, 512)
        // keep the user-chosen value unchanged. Matches GlobalSettings default.
        public var flashMoeSlotBank: Int = 64
        public var flashMoePrefetch: String = "none"
        public var flashMoeIoSplit: Int = 4
        // DFlash spec-decode
        public var dflash: Bool = false
        public var dflashDrafterPath: String = ""
        public var dflashBlockSize: Int = 16
        public var dflashTopK: Int = 4
        public var dflashNumPaths: Int = 60
        public var dflashTapLayers: String = "10,22,34,46,58"
        public var dflashTargetHiddenDim: Int = 3072
        // Parser overrides (CLI --tool-call-parser / --reasoning-parser)
        public var defaultToolParser: String = ""
        public var defaultReasoningParser: String = ""
        public init(modelPath: URL) { self.modelPath = modelPath }
    }

    public private(set) var loaded: vMLXLMCommon.ModelContainer?
    public private(set) var state: EngineState = .stopped

    /// Currently active LoRA / DoRA adapter on the chat model, or
    /// `nil` when the model is running with unmodified base weights.
    /// Mutated only by `Engine.loadAdapter` / `unloadAdapter` /
    /// `fuseAdapter` in `EngineAdapters.swift`.
    internal var _activeAdapter: ActiveAdapterInfo?

    /// Path the currently-loaded chat model was loaded from. nil when no
    /// model is loaded (or when still loading). Used by `benchmark(suite:)`
    /// to label report rows + for diagnostics.
    public private(set) var loadedModelPath: URL?

    /// Separately-loaded embedding model container. Distinct from `loaded`
    /// (the chat LLM) — users can have BOTH a chat LLM and an embedding
    /// model loaded simultaneously. Populated by `loadEmbeddingModel(at:)`.
    public private(set) var embeddingContainer: vMLXEmbedders.ModelContainer?

    /// Loaded whisper bundle (model + tokenizer + config). Managed by
    /// `EngineTranscribe.swift`. `Any?` to avoid pulling vMLXWhisper into
    /// this file's public surface — the extension casts it back on use.
    internal var _whisperBundle: Any?

    /// Path the embedding model was loaded from — used for response "model"
    /// field. nil when no embedding model has been loaded.
    public private(set) var embeddingModelPath: URL?

    /// Auto-detected capabilities for the currently loaded model. Populated
    /// by `CapabilityDetector.detect(at:)` during `load()`. Reset to nil on
    /// unload. Consumers (parser wiring, chat template overrides, UI badges)
    /// read this via `currentCapabilities()` rather than deriving from name
    /// heuristics or user settings.
    public private(set) var modelCapabilities: ModelCapabilities?

    /// Parsed `jang_config.json` for the loaded chat model, or `nil` when a
    /// standard MLX model is loaded. Populated after the model container is
    /// created and used by `Stream.buildGenerateParameters` so JANG models
    /// honor the TurboQuant bit widths they were calibrated for instead of
    /// the generic `turboQuantBits` default.
    public private(set) var loadedJangConfig: JangConfig?

    /// The `LoadOptions` used for the most recent successful load. Retained
    /// so that `wake()` out of `.standby(.deep)` can replay the load without
    /// requiring the caller to stash and re-supply them. Cleared by
    /// `unload()` and by an explicit new `load()` before the swap, and set
    /// on every successful `load()` completion.
    public private(set) var lastLoadOptions: LoadOptions?

    /// Engine-wide log sink. Mirrors `logger = logging.getLogger("vmlx")` in
    /// `vmlx_engine/server.py` — every subsystem (engine lifecycle, HTTP
    /// server middleware, model loader, cache, tool dispatch, MCP) writes
    /// here and the LogsPanel in the app subscribes to tail it live.
    public let logs = LogStore()

    /// Convenience forwarder so call sites don't need `await self.logs.append(...)`.
    public func log(_ level: LogStore.Level, _ category: String, _ message: String) async {
        await logs.append(level, category: category, message)
    }

    /// Live performance metrics — GPU memory, RAM, CPU, rolling tok/s, queue
    /// depth, and recent request latencies. Subscribed by the Server screen's
    /// PerformancePanel.
    ///
    /// Wired into `Stream.swift` per-decode-chunk and per-finalize (see
    /// `Stream.swift:463` + `:635`). Live token rate is available via
    /// `metrics.subscribe()` without additional plumbing.
    public let metrics = MetricsCollector()

    /// Model library — discovery + metadata index for every model on disk.
    /// Scans `~/.cache/huggingface/hub/` + any user-added dirs. The ServerScreen
    /// model picker binds to this; see `Library/ModelLibrary.swift`.
    public let modelLibrary: ModelLibrary

    /// Engine-owned download manager. Wired into the Ollama `/api/pull`
    /// route + the OpenAI `/v1/admin/models` download endpoint so HTTP
    /// clients that issue `ollama pull <repo>` actually trigger a real
    /// HuggingFace download (instead of the previous silent no-op).
    /// The SwiftUI app uses its own `AppState.downloadManager` for UI
    /// state, but the Engine instance is the canonical one for server
    /// callers (`vmlxctl serve` etc) where there is no AppState.
    public let downloadManager = DownloadManager()

    /// Persistent terminal working directory, threaded through the
    /// `bash` tool dispatcher so a `cd foo` in one tool invocation
    /// affects the next call in the same chat. The SwiftUI Terminal
    /// screen sets this on startup via `setTerminalCwd(_:)`; every
    /// successful bash call with a recovered post-exec `pwd` updates
    /// it via `updateTerminalCwd(_:)`. `nil` means "inherit the
    /// process cwd" (what you'd get from Finder's default).
    public private(set) var terminalCwd: URL? = nil
    public func setTerminalCwd(_ url: URL?) { terminalCwd = url }
    public func updateTerminalCwd(_ url: URL?) {
        // Only bump the persistent cwd on an actual new value so
        // the UI binding doesn't churn on every no-op bash call.
        guard let url, url != terminalCwd else { return }
        terminalCwd = url
    }

    /// 4-tier settings store (global → session → chat → request). Backed by
    /// its own SQLite file at `~/Library/Application Support/vMLX/settings.sqlite3`.
    /// HTTP routes should call `settings.resolved(sessionId:chatId:request:)`
    /// before dispatching to `stream()` so per-request overrides win.
    public let settings: SettingsStore

    /// MCP (Model Context Protocol) server manager. Owns the set of
    /// configured stdio MCP servers and lazily spawns them on first
    /// tool-list or tool-call request. Loaded from `mcp.json` in the
    /// standard search paths on Engine init. Empty by default — no
    /// servers are started until the user adds an entry to their
    /// config file.
    ///
    /// See `MCPServerManager`, `MCPStdioClient`, `MCPConfigLoader`.
    public let mcp: MCPServerManager

    /// Proxy — rescan the model library. `force=true` bypasses the 5-minute
    /// SQLite freshness window and always walks disk.
    public func scanModels(force: Bool = false) async -> [ModelLibrary.ModelEntry] {
        await modelLibrary.scan(force: force)
    }

    /// Subscribe to live metrics snapshots. Hot stream; first value is an
    /// immediate snapshot so the UI paints without waiting a full poll tick.
    public func subscribeMetrics() async -> AsyncStream<MetricsCollector.Snapshot> {
        await metrics.subscribe()
    }

    /// Reset the displayed GPU peak memory to the current active usage — wired
    /// to the "Reset peak" button on `PerformancePanel`'s GPU tile.
    public func resetPeakMemory() async {
        await metrics.resetPeakMemory()
    }

    /// Convenience forwarder — record a batch of tokens (prefill or decode).
    public func recordTokenBatch(prefill: Bool, count: Int, durationMs: Double) async {
        await metrics.recordTokenBatch(prefill: prefill, count: count, durationMs: durationMs)
    }

    /// Multi-subscriber state broadcast. The UI layer should subscribe via
    /// `Engine.subscribeState()` and re-render on every emission. Currently
    /// stubbed as a single-subscriber stream — graduate to a multicast
    /// channel (e.g. AsyncChannel) when more than one consumer needs it.
    private var stateContinuation: AsyncStream<EngineState>.Continuation?
    private lazy var stateStreamStorage: AsyncStream<EngineState> = {
        AsyncStream { cont in
            self.stateContinuation = cont
            cont.yield(self.state)
        }
    }()

    /// Idle lifecycle timer. On `.softSleep` we transition to
    /// `.standby(.soft)` and attempt the (currently stub) `softSleep()`; on
    /// `.deepSleep` we transition to `.standby(.deep)` and attempt
    /// `deepSleep()`. The HTTP routes call `wakeFromStandby()` before serving
    /// any request so the "Waking up…" banner has a chance to render.
    public let idleTimer = IdleTimer()

    /// Handle for the idleTimer subscription Task, so we can cancel it on
    /// `stop()` or engine deinit. The loop lives for the engine's lifetime.
    private var idleWatcher: Task<Void, Never>?

    /// Lazy-loaded vmlx-flux backend. Created on first image gen call.
    /// See `FluxBackend.swift` for the bridge layer. Typed as `Any?` here
    /// so this file doesn't need `import vMLXFlux` — the extension in
    /// FluxBackend.swift does the unsafe cast at the access site.
    internal var fluxBackend: Any?

    /// Live image-gen jobs keyed by UUID. UI subscribes via
    /// `Engine.imageGenStream(jobId:)` which fans out from the per-job bridge.
    internal var fluxJobs: [UUID: FluxJobBridge] = [:]

    /// Unified cache coordinator for prefix reuse across generations.
    /// Initialized on first `load()` based on the resolved settings
    /// (paged block size, max blocks, disk cache toggle). Passed into
    /// `vMLXLMCommon.generate(..., cacheCoordinator:)` so prefix hits are
    /// accounted for and `StreamChunk.Usage.cachedTokens` + `.cacheDetail`
    /// get populated.
    public private(set) var cacheCoordinator: CacheCoordinator?

    /// Currently-driving stream Task, if any. Registered by
    /// `Stream.swift::streamReal` so `Engine.stop()` and the request
    /// watchdog can cancel the in-flight generation directly instead of
    /// waiting for the AsyncStream's onTermination to fire. This is the
    /// knob the stop button needs to actually interrupt a hanging prefill.
    internal var currentStreamTask: Task<Void, Never>?

    /// Task handle for the in-flight `load()` so `stop()` can abort a
    /// wrong-model click mid-download or mid-weight-mmap. Audit
    /// 2026-04-16: previously the load task was a detached Task with
    /// no handle, so the Stop button during loading was inoperative.
    internal var currentLoadTask: Task<Void, Never>?

    // MARK: - JANG-DFlash speculative-decoding state
    //
    // Drafter + target adapter cached on the engine so `Stream.swift`
    // can short-circuit into `JangDFlashSpecDec.cachedGenerate` when
    // dflash is enabled, the target model conforms to JangDFlashTarget,
    // and a drafter checkpoint is loaded. All three must be non-nil for
    // the short-circuit to fire; anything else falls back cleanly to
    // the standard token iterator with a structured log warning.
    //
    // See `EngineDFlash.swift` for the load / lifecycle surface.
    internal var _dflashDrafter: Any?              // JangDFlashDrafter
    internal var _dflashDrafterURL: URL?
    internal var _dflashDrafterConfig: JangDFlashConfig?
    internal var _dflashTarget: Any?               // any JangDFlashTarget
    /// Per-request task registry for `POST /v1/{chat,completions,responses}/{id}/cancel`.
    /// Keyed by the SSE id the route handler returned to the client. Multiple
    /// concurrent streams (rare but possible: parallel non-blocking requests)
    /// each register their own entry so an explicit cancel hits the right
    /// task rather than the most-recent one.
    internal var streamTasksByID: [String: Task<Void, Never>] = [:]

    internal func setCurrentStreamTask(_ task: Task<Void, Never>) {
        self.currentStreamTask = task
    }
    internal func clearCurrentStreamTask() {
        self.currentStreamTask = nil
    }
    internal func cancelCurrentStreamTask() {
        self.currentStreamTask?.cancel()
    }

    /// Register `task` under `id` so a per-id cancel can find it later.
    public func registerStreamTask(id: String, task: Task<Void, Never>) {
        streamTasksByID[id] = task
    }
    /// Remove the registration when the stream finishes naturally.
    public func unregisterStreamTask(id: String) {
        streamTasksByID.removeValue(forKey: id)
    }

    /// Currently bound RemoteEngineClient, when this engine is being used
    /// as a thin proxy to an OpenAI/Ollama/Anthropic-compatible remote.
    /// Set by the chat dispatch layer right before `client.stream(...)`
    /// so a subsequent `cancelStream()` can also cancel the remote
    /// HTTP task. Cleared automatically when a new client takes over.
    private var remoteClient: RemoteEngineClient? = nil

    /// Bind the remote client so cancelStream() can cancel the in-flight
    /// HTTP request. ChatViewModel calls this just before invoking
    /// `client.stream(request:)`. Setting overwrites any previous binding
    /// — there's only ever one in-flight stream per engine actor.
    public func attachRemoteClient(_ client: RemoteEngineClient) {
        self.remoteClient = client
    }

    /// Public cancel hook — the ChatViewModel stop button calls this
    /// through a task hop so the actor can reach `currentStreamTask`
    /// directly. Safe to call when no stream is active (no-op).
    public func cancelStream() {
        currentStreamTask?.cancel()
        if let rc = remoteClient {
            Task { await rc.cancelStream() }
        }
    }

    /// Per-id cancel — used by `POST /v1/{chat,completions,responses}/{id}/cancel`.
    /// Returns `true` if a task was found and cancelled, `false` if no
    /// in-flight task exists for that id (already finished or unknown).
    @discardableResult
    public func cancelStream(id: String) -> Bool {
        if let task = streamTasksByID[id] {
            task.cancel()
            streamTasksByID.removeValue(forKey: id)
            return true
        }
        return false
    }

    public init(
        modelLibraryDB: ModelLibraryDB? = nil,
        settingsDB: SettingsDB? = nil
    ) {
        self.modelLibrary = ModelLibrary(database: modelLibraryDB ?? ModelLibraryDB())
        self.settings = SettingsStore(database: settingsDB ?? SettingsDB())

        // Load mcp.json if present in any of the standard search paths.
        // A missing file is not an error — `MCPConfigLoader.load`
        // returns an empty config so the manager just has no servers.
        // Any parse / validation failure is logged (via the caller's
        // future subscribe path) but doesn't crash startup.
        //
        // Initial load uses the search-path discovery; the user-set
        // `mcpConfigPath` from GlobalSettings is honored on the next
        // `applySettings` call via `reloadMCPConfig(path:)`.
        let mcpConfig: MCPConfig
        do {
            mcpConfig = try MCPConfigLoader.load()
        } catch {
            mcpConfig = MCPConfig()
        }
        self.mcp = MCPServerManager(config: mcpConfig)

        Task { await self.startIdleWatcher() }
        // Apply the user's persisted mcpConfigPath (if any) on the next
        // event loop tick so the SettingsStore has had a chance to load.
        // This fixes the "user picks an mcp.json from settings, nothing
        // happens" bug — previously the path was stored but never read.
        Task { @MainActor in
            let g = await self.settings.global()
            if !g.mcpConfigPath.isEmpty {
                try? await self.reloadMCPConfig(
                    path: URL(fileURLWithPath: g.mcpConfigPath))
            }
        }
    }

    /// Re-load the MCP server catalog from a specific JSON file. Called by
    /// the settings UI when the user picks a new `mcp.json` from the
    /// folder/file picker. Stops every currently-running stdio server
    /// (so the next chat turn that needs a tool starts the new servers
    /// lazily), then installs the new config on the manager.
    ///
    /// `nil` resets to the default search-path discovery
    /// (`./mcp.json` → `~/.config/vmlx/mcp.json` → legacy paths).
    public func reloadMCPConfig(path: URL?) async throws {
        let cfg: MCPConfig
        if let path {
            cfg = try MCPConfigLoader.load(path: path)
            await logs.append(.info, category: "mcp",
                "Loaded mcp.json from \(path.path) — \(cfg.servers.count) server(s)")
        } else {
            cfg = (try? MCPConfigLoader.load()) ?? MCPConfig()
            await logs.append(.info, category: "mcp",
                "Reloaded mcp.json from default search paths — \(cfg.servers.count) server(s)")
        }
        // Stop any running stdio servers so the next tool dispatch
        // starts fresh against the new spec. Lazy start on first call
        // happens automatically via `executeTool`.
        await mcp.stopAll()
        await mcp.setConfig(cfg)
    }

    /// Apply a new global settings snapshot. Updates the idle timer config
    /// AND re-loads the MCP server catalog when the path changes, so any
    /// edit in the UI takes effect without an app restart.
    public func applySettings(_ g: GlobalSettings) async {
        let prev = await settings.global()
        await settings.setGlobal(g)
        await idleTimer.setConfig(.init(
            softAfter: g.idleSoftSec,
            deepAfter: g.idleDeepSec,
            enabled: g.idleEnabled
        ))
        // MCP path changed → reload the catalog so the new server list
        // is live before the next chat turn requests a tool. Skipped
        // when the path is unchanged so identical-snapshot writes (the
        // common settings-debounce flush) don't churn the stdio
        // sub-processes.
        if prev.mcpConfigPath != g.mcpConfigPath {
            let url = g.mcpConfigPath.isEmpty
                ? nil : URL(fileURLWithPath: g.mcpConfigPath)
            try? await reloadMCPConfig(path: url)
        }
    }

    /// Apply a session-level settings change via SettingsStore and fill in
    /// defaults that require engine-side state (notably auto-allocated port).
    /// If `session.port` is nil, walks 8000..8999 to find the first free TCP
    /// port and writes it back into the SessionSettings before persisting.
    @discardableResult
    public func applySessionSettings(_ id: UUID, _ incoming: SessionSettings) async -> SessionSettings {
        var s = incoming
        if s.port == nil {
            // Best-effort port allocation. If the entire 8000+1000 range
            // is taken (extremely unlikely), fall through to nil so the
            // user can pick manually rather than crashing on bind. The
            // session's server start path will surface a clean banner.
            s.port = (try? Self.firstAvailablePort(startingAt: 8000)) ?? 8000
        }
        await settings.setSession(id, s)
        return s
    }

    /// Find the first TCP port >= `start` that is not in use on localhost.
    /// Uses synchronous POSIX bind with SO_REUSEADDR=off so a free port means
    /// the OS returned success. Falls back to `start` if the full range is
    /// exhausted (shouldn't happen in practice).
    /// Walk a port range starting at `start` and return the first one
    /// that's actually bindable. Throws `EngineError.portInUse` after
    /// exhausting the range — the previous fallback `return start`
    /// silently returned a known-bad port that crashed downstream with
    /// a confusing error. Build-state audit 2026-04-15 finding #8.
    static func firstAvailablePort(
        startingAt start: Int,
        limit: Int = 1000,
        lan: Bool = false
    ) throws -> Int {
        var port = start
        let end = start + limit
        while port < end {
            if isPortFree(port, lan: lan) { return port }
            port += 1
        }
        throw EngineError.portInUse(start)
    }

    /// Probe whether `port` is currently bindable. By default checks
    /// 127.0.0.1 only — fast enough for the per-session allocator.
    /// When the caller is about to bind to 0.0.0.0 (LAN mode), pass
    /// `lan: true` to also probe the wildcard address; otherwise a
    /// peer-bound listener can slip past loopback-only detection and
    /// the bind will fail later with a confusing "address in use".
    /// Audit finding #4.
    public static func isPortFree(_ port: Int, lan: Bool = false) -> Bool {
        #if canImport(Darwin)
        if !probe(port: port, addr: 0x7f000001) { return false }   // 127.0.0.1
        if lan {
            if !probe(port: port, addr: 0x00000000) { return false } // 0.0.0.0
        }
        return true
        #else
        return true
        #endif
    }

    #if canImport(Darwin)
    private static func probe(port: Int, addr: UInt32) -> Bool {
        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { Darwin.close(sock) }
        var sin = sockaddr_in()
        sin.sin_family = sa_family_t(AF_INET)
        sin.sin_port = UInt16(port).bigEndian
        sin.sin_addr.s_addr = addr.bigEndian
        // SO_REUSEADDR so the probe doesn't false-positive on a TIME_WAIT
        // remnant from an earlier bind in this same process.
        var yes: Int32 = 1
        _ = Darwin.setsockopt(
            sock, SOL_SOCKET, SO_REUSEADDR,
            &yes, socklen_t(MemoryLayout<Int32>.size))
        let bindResult = withUnsafePointer(to: &sin) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(sock, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bindResult == 0
    }
    #endif

    private func startIdleWatcher() {
        idleWatcher?.cancel()
        idleWatcher = Task { [weak self] in
            guard let self else { return }
            let events = await self.idleTimer.subscribe()
            for await event in events {
                if Task.isCancelled { break }
                await self.handleIdleEvent(event)
            }
        }
    }

    private func handleIdleEvent(_ event: IdleTimer.Event) async {
        switch event {
        case .softSleep:
            // Only transition from .running — if we're already in standby or
            // stopped or loading, the timer fired stale; ignore it.
            if case .running = state {
                transition(.standby(.soft))
                do {
                    try await softSleep()
                } catch {
                    await log(.warn, "lifecycle",
                              "softSleep stub threw: \(error) — state pinned to standby(.soft)")
                }
            }
        case .deepSleep:
            if case .standby(.soft) = state {
                transition(.standby(.deep))
                do {
                    try await deepSleep()
                } catch {
                    await log(.warn, "lifecycle",
                              "deepSleep stub threw: \(error) — state pinned to standby(.deep)")
                }
            }
        }
    }

    /// JIT wake entry point. HTTP routes must call this before `stream()` so
    /// the UI banner has time to flash and state matches reality. No-op when
    /// the engine is already `.running`.
    /// P2-LIFE-3: idle timer bump from interactive UI surfaces. The chat
    /// view model calls this whenever the user types in the input field
    /// (debounced) so the model doesn't deep-sleep during active typing
    /// even when no API request has fired yet. Safe no-op when the timer
    /// is disabled or the engine is already awake.
    public func bumpIdleTimer() async {
        await idleTimer.reset()
    }

    public func wakeFromStandby() async {
        switch state {
        case .standby:
            do {
                try await wake()
            } catch {
                await log(.warn, "lifecycle",
                          "wake stub threw: \(error) — transitioning state anyway")
            }
            transition(.running)
            await idleTimer.reset()
        default:
            break
        }
    }

    /// Subscribe to engine lifecycle transitions. Hot stream — replays current
    /// state to new subscribers, then yields every transition.
    public func subscribeState() -> AsyncStream<EngineState> {
        stateStreamStorage
    }

    private func transition(_ next: EngineState) {
        state = next
        stateContinuation?.yield(next)
    }

    /// Load a model from disk with progress events.
    ///
    /// Mirrors `vmlx_engine/engine/{simple,batched}.py::load_model` plus the
    /// Electron-side load progress regex table at
    /// `panel/src/main/sessions.ts:140-198`. As long as the underlying
    /// `loadContainer` API has no per-shard callback, we emit phase-level
    /// transitions only — better than nothing, and the UI shows a moving bar.
    public func load(_ opts: LoadOptions) -> AsyncThrowingStream<LoadEvent, Error> {
        AsyncThrowingStream { continuation in
            let loadTask = Task { [weak self] in
                guard let self else { return }
                // Audit 2026-04-16: register ourselves so `Engine.stop()`
                // can reach the load task. Self-deregister on exit via
                // the `defer` at the closure tail.

                // Emit + fail are declared as @Sendable closures instead
                // of local `func` decls so they satisfy the @Sendable
                // parameter contract on `runLoadProgressClock(emit:)`.
                // A local function is not implicitly Sendable even when
                // all its captures are, so passing one triggers a
                // "converting non-Sendable function value to @Sendable"
                // warning under Swift-6 strict concurrency.
                let emit: @Sendable (LoadProgress) async -> Void = { p in
                    await self.transition(.loading(p))
                    continuation.yield(.progress(p))
                }
                let fail: @Sendable (String) async -> Void = { msg in
                    await self.logs.append(.error, category: "engine", "Load failed: \(msg)")
                    await self.transition(.error(msg))
                    continuation.yield(.failed(msg))
                    continuation.finish()
                }

                // NOTE: No per-load timeout watchdog. Python's `vmlx-engine
                // serve` has no `--load-timeout` flag and the loader
                // runs to completion regardless of wall-clock. We match
                // that behavior here. If a hung HF download or deadlocked
                // Metal init is observed in practice, a proper fix lands
                // upstream; until then no arbitrary 600s kill switch.
                do {
                    await self.logs.append(
                        .info, category: "engine",
                        "Loading model from \(opts.modelPath.lastPathComponent)")

                    // Phase 1: detection (fraction 0 → 0.05)
                    await emit(LoadProgress(
                        phase: .reading,
                        fraction: 0.02,
                        label: "Detecting model"
                    ))
                    let config = try ModelDetector.detect(at: opts.modelPath)
                    // Hand-written TokenizerLoader — replaces the
                    // vmlx-swift-lm `#huggingFaceTokenizerLoader()` macro.
                    // We dropped the MLXHuggingFaceMacros target when we
                    // vendored the tree because the swift-syntax + .macro
                    // target resolution was flaky on 6.2.3. This struct
                    // produces the same `any TokenizerLoader` the macro
                    // would have.
                    let tokenizerLoader = TransformersTokenizerLoader()

                    // Capability auto-detection: JANG stamped > model_type
                    // table > bronze heuristic. Never throws. Bronze-tier
                    // fallthroughs get logged so users can promote the
                    // model_type to the silver allowlist.
                    let caps = CapabilityDetector.detect(at: opts.modelPath) { msg in
                        Task { [weak self] in
                            await self?.logs.append(.warn, category: "engine", msg)
                        }
                    }
                    await self.setCapabilities(caps)
                    await self.logs.append(
                        .info, category: "engine",
                        "Capabilities detected via \(caps.detectionSource.rawValue): " +
                        "family=\(caps.family) " +
                        "reasoning=\(caps.reasoningParser ?? "nil") " +
                        "tool=\(caps.toolParser ?? "nil") " +
                        "cache=\(caps.cacheType) " +
                        "thinkInTemplate=\(caps.thinkInTemplate)"
                    )

                    let modality = config.modality == .vision ? "VLM" : "LLM"
                    await self.logs.append(
                        .info, category: "engine",
                        "Detected modality=\(modality); loading weights")
                    await emit(LoadProgress(
                        phase: .reading,
                        fraction: 0.05,
                        label: "Preparing tokenizer"
                    ))

                    // Phase 2: weight load (fraction 0.05 → 0.85). mlx-swift
                    // doesn't surface per-shard progress, so we drive a
                    // time-based pseudo-progress clock in a background task
                    // that ramps from 0.05 → 0.85 over an estimated
                    // duration based on file size. Real progress will plug
                    // in if mlx-swift ever exposes a delegate.
                    let estimatedBytes = estimateModelBytes(at: opts.modelPath)
                    await self.setExpectedBytes(estimatedBytes)
                    // Audit 2026-04-16: release the prior model's container
                    // BEFORE the progress clock captures its baseline. If
                    // we skip this, a second load on a running engine
                    // snapshots a baseline that already includes the old
                    // weights → the new load's GPU delta is artificially
                    // small and the progress bar moves slower than reality.
                    // ARC may still hold the container briefly via in-flight
                    // streams, but explicit `loaded = nil` gives it the
                    // earliest release opportunity.
                    await self.setLoaded(nil)
                    let progressClock = Task { [weak self] in
                        await self?.runLoadProgressClock(
                            estimatedBytes: estimatedBytes,
                            startFraction: 0.05,
                            endFraction: 0.85,
                            phase: .applying,
                            label: "Loading \(modality) weights",
                            emit: emit
                        )
                    }

                    // Pre-load guard (Audit 2026-04-16 VL pipeline gap #2):
                    // silver table marks llava/cogvlm/molmo/internvl/florence2/
                    // got_ocr2/phi3v/phi4mm/minicpmv as `isMLLM: true` so the
                    // picker shows a vision badge, but no Swift factory entry
                    // exists → loadContainer throws a confusing
                    // `ModelFactoryError.unsupportedModelType` mid-way. Fail
                    // fast with a clear, user-facing reason BEFORE touching
                    // the weights.
                    let unsupportedVLMs: Set<String> = [
                        "llava", "llava_next", "cogvlm", "cogvlm2",
                        "florence2", "got_ocr2", "molmo", "minicpmv",
                        "internvl_chat", "phi4mm", "phi3v",
                    ]
                    if config.modality == .vision,
                       unsupportedVLMs.contains(config.modelType.lowercased())
                    {
                        progressClock.cancel()
                        throw EngineError.notImplemented(
                            "\(config.modelType) is recognized as a VL model but has no Swift implementation yet. Use the Electron vMLX.app or a different model family (Qwen2/2.5/3-VL, Gemma 3/4 VLM, PaliGemma, Idefics3, SmolVLM, Pixtral are supported).")
                    }
                    let container: vMLXLMCommon.ModelContainer
                    do {
                        switch config.modality {
                        case .text:
                            container = try await LLMModelFactory.shared.loadContainer(
                                from: opts.modelPath, using: tokenizerLoader)
                        case .vision:
                            container = try await VLMModelFactory.shared.loadContainer(
                                from: opts.modelPath, using: tokenizerLoader)
                        }
                    } catch {
                        // Audit 2026-04-16: progressClock was only cancelled
                        // on the happy path. When loadContainer threw, the
                        // 4Hz poller leaked, emitting .loading events after
                        // the engine had already transitioned to .error.
                        progressClock.cancel()
                        throw error
                    }
                    progressClock.cancel()
                    await emit(LoadProgress(
                        phase: .applying,
                        fraction: 0.85,
                        label: "Weights loaded"
                    ))

                    // Phase 3: cache + idle + metrics setup (fraction 0.85 → 0.90)
                    await self.setLoaded(container)
                    await self.setLoadedModelPath(opts.modelPath)
                    await self.setLastLoadOptions(opts)
                    // Pull the model's declared JangTurboQuant (if any) from
                    // the container so `Stream.buildGenerateParameters` can
                    // apply it to the cache without a re-parse on hot path.
                    let loadedJang = await container.perform { (ctx: ModelContext) -> JangConfig? in
                        ctx.jangConfig
                    }
                    await self.setLoadedJangConfig(loadedJang)
                    await self.setupCacheCoordinator(opts: opts)
                    // Phase 3.5: Flash MoE apply.
                    //
                    // When `settings.flashMoe == true` and the loaded model
                    // conforms to `FlashMoEReplaceable`, walk the decoder
                    // layers and swap in streaming shims that page experts
                    // from disk via a slot-bank LRU. Mirrors Python
                    // `vmlx_engine/models/flash_moe_integration.py:apply_flash_moe`.
                    // Non-conforming models silently skip — Flash MoE is
                    // opt-in per-model (see `FlashMoEApply.swift`).
                    await self.applyFlashMoEIfEnabled(
                        container: container, opts: opts)
                    // Bind a JANG-DFlash target adapter if the model
                    // supports it. Only MiniMax family today — every
                    // other model returns nil and `dflashIsReady()`
                    // stays false so the stream path falls back to
                    // the standard iterator.
                    await self.bindDFlashTargetIfEligible(container: container)
                    // Restore any previously-configured DFlash drafter
                    // from persisted settings so the speculative-decode
                    // path survives app relaunch. Runs AFTER bind so the
                    // shape check can see the target adapter.
                    await self.autoLoadDFlashDrafterIfConfigured()
                    await self.metrics.setQueueDepth(0)
                    await self.idleTimer.setConfig(.init(
                        softAfter: opts.idleSoftSec,
                        deepAfter: opts.idleDeepSec,
                        enabled: opts.idleEnabled
                    ))
                    await self.idleTimer.reset()
                    await emit(LoadProgress(
                        phase: .warmup,
                        fraction: 0.90,
                        label: "Initializing cache"
                    ))

                    // Phase 4: REAL warmup (fraction 0.90 → 0.98). Run a
                    // 1-token dummy generation so Metal shaders are JIT-
                    // compiled, kernels are cached, and the first REAL
                    // request doesn't pay the cold-start latency. This is
                    // the #1 reason "model ready" lies — fix it here.
                    await emit(LoadProgress(
                        phase: .warmup,
                        fraction: 0.92,
                        label: "Compiling Metal shaders"
                    ))
                    try await self.runWarmup(container: container)
                    await emit(LoadProgress(
                        phase: .finalizing,
                        fraction: 0.98,
                        label: "Ready"
                    ))

                    // Phase 5: done — NOW the model is actually ready to speak.
                    await self.logs.append(
                        .info, category: "engine",
                        "Model ready (\(modality)) — warmup complete")
                    await emit(LoadProgress(
                        phase: .finalizing,
                        fraction: 1.0,
                        label: "Ready"
                    ))
                    await self.transition(.running)
                    continuation.yield(.done)
                    continuation.finish()
                } catch {
                    await fail("\(error)")
                }
                // Self-deregister: clear the engine's handle so subsequent
                // `stop()` calls don't cancel a completed task.
                await self.clearCurrentLoadTask()
            }
            // Register the load task so `stop()` can cancel it. Done
            // AFTER Task creation (not inside the closure) because the
            // stream builder can't hop to the actor here; a detached Task
            // is used for the register hop.
            Task { [weak self] in
                await self?.setCurrentLoadTask(loadTask)
            }
        }
    }

    internal func setCurrentLoadTask(_ task: Task<Void, Never>) {
        self.currentLoadTask = task
    }
    internal func clearCurrentLoadTask() {
        self.currentLoadTask = nil
    }

    private func setLoaded(_ c: vMLXLMCommon.ModelContainer?) {
        self.loaded = c
    }

    private func setLoadedModelPath(_ url: URL?) {
        self.loadedModelPath = url
    }

    private func setLoadedJangConfig(_ cfg: JangConfig?) {
        self.loadedJangConfig = cfg
    }

    private func setLastLoadOptions(_ opts: LoadOptions?) {
        self.lastLoadOptions = opts
    }

    private func setCapabilities(_ c: ModelCapabilities?) {
        self.modelCapabilities = c
    }

    /// Snapshot of the current model's auto-detected capabilities. Returns
    /// nil before the first successful load.
    public func currentCapabilities() -> ModelCapabilities? {
        modelCapabilities
    }

    /// Estimate total byte count of safetensors + weight files in the
    /// model directory. Used to gauge weight-load duration so the
    /// time-based pseudo-progress clock ramps at a realistic rate.
    private nonisolated func estimateModelBytes(at directory: URL) -> Int64 {
        // HF cache layout uses symlinks from `snapshots/<hash>/*.safetensors`
        // to blobs under `blobs/<sha>`. `.fileSizeKey` on the symlink
        // itself reports tiny (50-100 B) — we must resolve + stat the
        // target. Audit 2026-04-16: the integration harness caught
        // `0 B total` on HF cache dirs; same bug here caused the real
        // progress bar to show "677 MB / 76 B" = 892% and saturate at
        // the 98% clamp instead of tracking actual load.
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for url in contents {
            let ext = url.pathExtension.lowercased()
            guard ext == "safetensors" || ext == "bin" || ext == "gguf" else { continue }
            // Follow symlink → stat target. Falls back to .fileSizeKey
            // if resolvingSymlinksInPath returns same URL (non-symlink).
            let resolved = url.resolvingSymlinksInPath()
            if let attrs = try? FileManager.default.attributesOfItem(atPath: resolved.path),
               let size = attrs[.size] as? Int64
            {
                total += size
            } else if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    /// Real load-progress poller. Drives the bar from `startFraction` →
    /// `endFraction` based on measured MLX GPU memory growth (+ process
    /// RSS as fallback when the GPU probe reads zero pre-warmup).
    ///
    /// Approach: we pre-scanned the model directory's safetensors files
    /// into `expectedBytes` (sum of on-disk weight bytes). When
    /// `loadContainer` runs, MLX maps those tensors into unified memory
    /// and the GPU active-memory counter rises accordingly.
    ///
    /// Nuances handled:
    /// - **Baseline drift**: capture GPU + RSS at clock start; emit
    ///   fractions based on deltas so leftover cache from a prior load
    ///   doesn't skew the bar.
    /// - **JANG dequant peak**: MXTQ/JANG repack can peak at 1.3–1.5x
    ///   the final resident size. When `delta > expectedBytes`, clamp
    ///   to `endFraction - 0.02` (show 83%) rather than overflow past
    ///   the bar's right edge.
    /// - **Zero-probe fallback**: on first-ever load in a process the
    ///   Metal device may not be initialized, so `MLX.Memory.snapshot`
    ///   returns 0. Fall back to process RSS in that case.
    /// - **Prequantized models**: in-memory ≈ disk bytes → progress
    ///   tracks 1:1 honest.
    /// - **Cancellation-safe**: parent cancels the Task when the real
    ///   load call returns; label includes "…" sentinel so the caller
    ///   can replace with "Weights loaded" on completion.
    ///
    /// Audit 2026-04-16 UX: replaces prior time-based pseudo-progress
    /// clock that lied to users during the longest phase.
    private func runLoadProgressClock(
        estimatedBytes: Int64,
        startFraction: Double,
        endFraction: Double,
        phase: LoadProgress.Phase,
        label: String,
        emit: @escaping @Sendable (LoadProgress) async -> Void
    ) async {
        // Baseline snapshot so we measure growth from THIS load only.
        let baselineGPU = readMLXActiveMemory()
        let baselineRSS = readProcessResidentBytes()
        // Fallback time estimate for when the memory probe hasn't moved
        // yet (pre-mmap phase). Used ONLY to interpolate the first 1-2s
        // before real bytes show up — cap at 10% of the span.
        let fallbackStart = Date()
        let fallbackBudget: TimeInterval = 4.0  // ease-in over 4s
        while !Task.isCancelled {
            let gpuNow = readMLXActiveMemory()
            let rssNow = readProcessResidentBytes()
            // Use whichever grew more — GPU when Metal is live, RSS for
            // the pre-Metal / JANG-dequant windows where weights are
            // still in CPU-side buffers before the MLX upload.
            let gpuDelta = max(0, gpuNow - baselineGPU)
            let rssDelta = max(0, rssNow - baselineRSS)
            let bestDelta = max(gpuDelta, rssDelta)

            let span = endFraction - startFraction
            let fraction: Double
            let detail: String
            if expectedBytes > 0, bestDelta > 0 {
                // Real-data path: clamp to [0, 0.98 * span] so the bar
                // never fully completes until `loadContainer` returns.
                let raw = Double(bestDelta) / Double(max(1, expectedBytes))
                let clamped = min(0.98, raw)
                fraction = startFraction + span * clamped
                detail = "\(Self.prettyBytes(bestDelta)) / \(Self.prettyBytes(expectedBytes))"
            } else {
                // Fallback: ease the first 10% of the span over 4s so
                // the user sees immediate motion even if no bytes have
                // landed yet. Stays stuck at 10% until real data flows.
                let elapsed = Date().timeIntervalSince(fallbackStart)
                let t = min(1.0, elapsed / fallbackBudget)
                fraction = startFraction + span * 0.10 * t
                detail = "initializing"
            }
            await emit(LoadProgress(
                phase: phase,
                fraction: fraction,
                label: "\(label) (\(detail))"
            ))
            try? await Task.sleep(nanoseconds: 250_000_000)  // 4 Hz
        }
    }

    /// expectedBytes is captured once at load-time by the caller and
    /// copied into the Engine so the poller can read it without another
    /// actor hop. Updated in `load()` right before the poller starts.
    private var expectedBytes: Int64 = 0

    /// Actor-isolated setter so the non-isolated `load` closure can
    /// prime `expectedBytes` before spawning the progress clock Task.
    private func setExpectedBytes(_ n: Int64) { expectedBytes = n }

    /// Snapshot MLX active memory. Zero when Metal is not yet initialized
    /// (first load in a fresh process). Caller compensates.
    private nonisolated func readMLXActiveMemory() -> Int64 {
        // Honor the same XCTest gate as MetricsCollector.
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            return 0
        }
        let snap = MLX.Memory.snapshot()
        return Int64(snap.activeMemory + snap.cacheMemory)
    }

    /// Process resident set size via mach_task_basic_info. Valid even
    /// before Metal is touched, so this is the fallback signal during
    /// the initial mmap phase of `loadContainer`.
    private nonisolated func readProcessResidentBytes() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }

    /// Short human-readable byte count for progress labels.
    /// 1_234_567_890 → "1.23 GB".
    static func prettyBytes(_ n: Int64) -> String {
        let gb = Double(n) / 1_073_741_824
        if gb >= 1 { return String(format: "%.2f GB", gb) }
        let mb = Double(n) / 1_048_576
        if mb >= 1 { return String(format: "%.0f MB", mb) }
        return "\(n) B"
    }

    /// Real warmup pass. Runs a 1-token dummy `generate()` on the loaded
    /// container so Metal shaders JIT-compile, the KV cache allocator
    /// initializes, and the model graph is fully materialized. After this
    /// returns, the first REAL `Engine.stream` call won't pay the cold
    /// start penalty. This is the fix for "loading bar finishes but
    /// first response lags for 3 seconds".
    private func runWarmup(container: vMLXLMCommon.ModelContainer) async throws {
        let warmupChat = [Chat.Message.user("Hi")]
        let userInput = UserInput(chat: warmupChat)
        // Build params as a `let` so the captured copy is immutable —
        // Swift-6 strict concurrency rejects captures of mutable `var`
        // in @Sendable closures. GenerateParameters is a struct so we
        // use the mutating-then-freeze pattern.
        let params: GenerateParameters = {
            var p = GenerateParameters()
            p.temperature = 0.0
            p.maxTokens = 1
            p.prefillStepSize = 512
            return p
        }()
        let stream: AsyncStream<Generation> = try await container.perform {
            (ctx: ModelContext) in
            let lmInput = try await ctx.processor.prepare(input: userInput)
            return try vMLXLMCommon.generate(
                input: lmInput,
                parameters: params,
                context: ctx
            )
        }
        for await _ in stream { /* drain — one token, we don't care what */ }
        await self.logs.append(.info, category: "engine", "Warmup pass complete")
    }

    /// Build and install the cache coordinator after a model load succeeds.
    /// Mirrors the settings-tier mapping: prefix cache toggle, block size,
    /// max blocks, disk cache toggle + dir + size. Passes through to
    /// `vMLXLMCommon.generate(..., cacheCoordinator:)` on every stream.
    private func setupCacheCoordinator(opts: LoadOptions) async {
        let g = await settings.global()
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = g.usePagedCache && g.enablePrefixCache
        cfg.pagedBlockSize = g.pagedCacheBlockSize
        cfg.maxCacheBlocks = g.maxCacheBlocks
        // PROMPT-LEVEL L2 disk cache — keyed by token sequence hash,
        // stores per-prompt KV arrays via `DiskCache`. Reads from
        // `g.enableDiskCache` / `g.diskCacheDir` / `g.diskCacheMaxGB`
        // which is what the CLI `--enable-disk-cache` flag and the
        // SwiftUI API tab toggle both write to.
        //
        // PRIOR BUG: this used to read `g.enableBlockDiskCache` +
        // `g.blockDiskCache{Dir,MaxGB}` — fields that exist for
        // Python-parity future block-level store but are currently
        // orphaned. The CLI flag wrote to the correct fields; the
        // coordinator read from the wrong ones; the result was that
        // `--enable-disk-cache` silently did nothing, every L2 live
        // test showed `disk.enabled: false`.
        //
        // MODEL-CLASS SAFETY GUARD: disable disk cache for model
        // classes whose restore path is known to be unstable. Live
        // tested 2026-04-13:
        //   • Plain LLM (Qwen3, Llama, Gemma 4 text) — STABLE ✅
        //   • VL JANG repacked       — CRASHES on T2 restore with
        //     `SmallVector out of range` in mlx_array_dim. Per-layer
        //     shape mismatch (likely sliding-window attention layers
        //     with rotating KV that the disk store doesn't serialize).
        //   • Hybrid SSM             — disk path stores KV only, the
        //     SSM companion state is in a separate cache. Restoring
        //     KV without companion = garbled output even when no
        //     fatal occurs. Skip until SSM-aware disk path lands.
        // Both classes still get full L1 paged + (hybrid) SSM
        // companion in-memory caching — only the L2 disk tier is
        // suppressed.
        // `ModelCapabilities.modality` is a `Modality` enum with String
        // raw value. Compare against the rawValue or the enum case
        // directly — `modality == "vision"` would compile (because
        // both Modality and String are Comparable through some path)
        // but never match, so the guard below would silently no-op
        // and VL JANG would still hit the disk-restore crash.
        let isVL: Bool = (self.modelCapabilities?.modality.rawValue == "vision")
        // Mirror the hybrid detection logic from the lower block so we
        // can decide BEFORE constructing CacheCoordinator. Same source
        // (CapabilityDetector → ModelCapabilities.cacheType).
        let isHybridForDisk = self.modelCapabilities?.cacheType == "hybrid"
        let userWantsDisk = g.enableDiskCache
        // L2 disk cache: ALL model classes supported (verified live
        // 2026-04-13 PM). The v2 unified format from vmlx-swift-lm@14457d1
        // handles plain LLM (KVCacheSimple), hybrid SSM (Mamba per-layer
        // via `.mamba` LayerKind), TQ-compressed
        // (`TurboQuantKVCache.restoreCompressed`), and QuantizedKVCache
        // via `.qkv`. Three latent bugs were fixed during the port:
        //
        //   1. **Scalar metadata round-trip.** `MLXArray(Int32(...))`
        //      produces a 0-dim scalar that doesn't survive the
        //      multi-key safetensors round-trip — half of the
        //      `__layer_kind_*__` and `__mamba_*_offset__` arrays came
        //      back missing or as garbage float-bit-patterns. Fixed in
        //      `TQDiskSerializer.swift` via the new `metaInt32(...)`
        //      helper that writes 1-element 1D arrays.
        //   2. **Cache-hit input rank.** The cache-hit path fed a 1D
        //      `[1]` token tensor to `model.prepare`, but VL models
        //      call `inputs.dim(1)` for sequence length and a 1D shape
        //      fatal-trapped with `SmallVector out of range. at
        //      array.cpp:335`. Fixed in `Evaluate.swift` and
        //      `BatchEngine.swift` cache-hit branches by using
        //      `[.newAxis]` to make the shape `[1, 1]`.
        //   3. **Mamba double-write.** The disk-restore branch called
        //      the legacy `restoreSSMStates(...)` after the v2
        //      per-layer `.mamba` restore had already populated the
        //      Mamba caches. Fixed by removing the legacy call from
        //      both engine paths' disk branches.
        //
        // Live verified on Qwen3.5-VL-4B-JANG hybrid (24 Mamba + 8
        // KVCacheSimple): T1 store, T2 restart, **19/19 prompt tokens
        // restored from disk**, coherent generation.
        cfg.enableDiskCache = userWantsDisk
        cfg.diskCacheMaxGB = Float(g.diskCacheMaxGB)
        if !g.diskCacheDir.isEmpty {
            cfg.diskCacheDir = URL(fileURLWithPath: g.diskCacheDir)
        }
        cfg.enableMemoryCache = g.enableMemoryCache
        cfg.memoryCachePercent = g.memoryCachePercent
        cfg.memoryCacheTTLMinutes = g.memoryCacheTTLMinutes
        // Surface the locals in the log so we can still see model class.
        _ = isVL
        _ = isHybridForDisk
        if userWantsDisk {
            await logs.append(
                .info, category: "cache",
                "L2 disk cache enabled (v2 unified format).")
        }
        cfg.modelKey = opts.modelPath.lastPathComponent
        let coord = CacheCoordinator(config: cfg)

        // Flip hybrid mode if the loaded model has interleaved SSM layers
        // (Nemotron-H, Qwen3-Next/GatedDelta, Jamba, FalconH1, LFM2/LFM2MoE,
        // GraniteMoeHybrid, MiMoV2Flash, BaichuanM1, or any config carrying
        // `hybrid_override_pattern`). CapabilityDetector already honours the
        // Python `is_hybrid_ssm_config` rules, so we just mirror the result
        // here. When true, CacheCoordinator.fetch/store walks the
        // SSMStateCache companion tier alongside the paged KV cache.
        let isHybrid = self.modelCapabilities?.cacheType == "hybrid"
        if isHybrid {
            coord.setHybrid(true)
        }
        // MLA ⊥ TurboQuant: log the implicit skip at load time so the
        // operator understands why their TQ flag is being ignored. The
        // actual gate happens in `Stream.buildGenerateParameters` so
        // every request inherits the exclusion regardless of how the
        // setting flips at runtime.
        if self.modelCapabilities?.cacheType == "mla" && g.enableTurboQuant {
            await logs.append(
                .info, category: "cache",
                "MLA model detected — TurboQuant disabled for this load (MLA uses native latent KV; TQ would silently no-op).")
        }
        self.cacheCoordinator = coord
        await logs.append(
            .info, category: "cache",
            "CacheCoordinator: paged=\(cfg.usePagedCache) "
            + "blocks=\(cfg.maxCacheBlocks) disk=\(cfg.enableDiskCache) "
            + "hybrid=\(isHybrid)")
    }

    /// Stop the engine and release the model.
    public func stop() {
        // Cancel any in-flight generation first so the stop button is
        // actually responsive (see Stream.swift cancellation commentary).
        currentStreamTask?.cancel()
        currentStreamTask = nil
        // Audit 2026-04-16: also cancel a mid-flight load so Stop during
        // a wrong-model click actually aborts the weight mmap / JANG
        // repack. Prior behavior was to only cancel streams, leaving
        // long loads uninterruptible.
        currentLoadTask?.cancel()
        currentLoadTask = nil
        loaded = nil
        loadedModelPath = nil
        loadedJangConfig = nil
        lastLoadOptions = nil
        _activeAdapter = nil
        // DFlash drafter + target adapter are tied to the loaded target —
        // free them here so a subsequent load of a DIFFERENT target
        // doesn't dispatch through a dead adapter. `dflashDrafterPath`
        // settings persists so auto-load will rebuild on the next load
        // if the next target is compatible.
        _dflashDrafter = nil
        _dflashDrafterURL = nil
        _dflashDrafterConfig = nil
        _dflashTarget = nil
        idleWatcher?.cancel()
        idleWatcher = nil
        transition(.stopped)
        // Flush any pending debounced settings writes so in-memory
        // edits from the last 500ms don't get dropped when the app is
        // terminating. Caller responsibility is to await the engine
        // actor after this method returns if they want the flush to
        // complete before exit — the vMLXApp `willTerminate` hook at
        // `vMLXApp.swift:107-122` already uses a semaphore + Task for
        // this. Here we fire-and-forget because `stop()` is a sync
        // method that can be called from non-async contexts.
        Task {
            await settings.flushPending()
            await logs.append(.info, category: "engine", "Engine stopped")
        }
    }

    /// Generate a streaming completion. Mirrors `BatchedEngine.generate`.
    ///
    /// Generation flow:
    /// - Reasoning suppression (when `enableThinking=false` and the model
    ///   template does not already stamp `<think>` tags): see
    ///   `Stream.buildChatMessages`:1210 which injects an empty
    ///   `<think>\n</think>\n\n` assistant-role prefix so the model skips
    ///   the reasoning block. §15 NO-REGRESSION fallthrough routes any
    ///   leaked reasoning tokens to visible content.
    /// - Tool-call parser dispatch: see `Stream.swift` toolCallParser
    ///   wiring, model-aware via `CapabilityDetector.toolParser`.
    /// - Prefix cache hit accounting: see `CacheCoordinator.prefixHit`
    ///   and `MetricsCollector.recordTokenBatch(prefill:count:...)`.
    public func stream(
        request: ChatRequest
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        // Real generation loop — see `Stream.swift` for the implementation.
        streamReal(request: request)
    }

    /// `id`-tagged variant. Registers the underlying generation task in
    /// `streamTasksByID[id]` so `cancelStream(id:)` can target this
    /// specific request. Routes that mint an OpenAI/Anthropic SSE id call
    /// this so per-request cancellation works under parallel load.
    public func stream(
        request: ChatRequest,
        id: String
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        let upstream = streamReal(request: request)
        return AsyncThrowingStream { continuation in
            let pump = Task { [weak self] in
                do {
                    for try await chunk in upstream {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await self?.unregisterStreamTask(id: id)
            }
            Task { [weak self] in
                await self?.registerStreamTask(id: id, task: pump)
            }
            continuation.onTermination = { _ in pump.cancel() }
        }
    }

    // MARK: - HTTP route entry points (stubs — wired but not implemented)

    /// Load an embedding model into `embeddingContainer`. Separate from the
    /// chat `loaded` container — users can have both loaded simultaneously.
    public func loadEmbeddingModel(at modelPath: URL) async throws {
        let tokenizerLoader = TransformersTokenizerLoader()
        let container = try await vMLXEmbedders.loadModelContainer(
            from: modelPath, using: tokenizerLoader)
        self.embeddingContainer = container
        self.embeddingModelPath = modelPath
        await logs.append(.info, category: "engine",
            "Embedding model ready (\(modelPath.lastPathComponent))")
    }

    /// Generate embeddings for a batch of inputs using the currently-loaded
    /// embedding model. Runs tokenize -> forward -> pool -> normalize for
    /// each input. Returns one Float vector per input.
    public func generateEmbeddings(inputs: [String]) async throws -> [[Float]] {
        guard let container = self.embeddingContainer else {
            throw EngineError.notImplemented(
                "Engine.generateEmbeddings - no embedding model loaded (call loadEmbeddingModel first)")
        }
        return await container.perform {
            (model, tokenizer, pooling) -> [[Float]] in
            let encoded = inputs.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
            let padTok = tokenizer.eosTokenId ?? 0
            let maxLen = max(1, encoded.reduce(0) { max($0, $1.count) })
            let padded = MLX.stacked(
                encoded.map { row -> MLXArray in
                    let pad = Array(repeating: padTok, count: maxLen - row.count)
                    return MLXArray(row + pad)
                })
            let mask = (padded .!= padTok)
            let tokenTypes = MLXArray.zeros(like: padded)
            let modelOutput = model(
                padded,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask)
            let result = pooling(modelOutput, mask: mask, normalize: true, applyLayerNorm: false)
            MLX.eval(result)
            return result.map { $0.asArray(Float.self) }
        }
    }

    /// OpenAI-format embeddings response builder. Decodes the request dict,
    /// calls `generateEmbeddings`, and shapes the response.
    public func embeddings(request: [String: Any]) async throws -> [String: Any] {
        let inputs: [String]
        if let s = request["input"] as? String {
            inputs = [s]
        } else if let arr = request["input"] as? [String] {
            inputs = arr
        } else {
            throw EngineError.invalidRequest("embeddings: missing or invalid 'input' field — expected string or [string]")
        }
        let vectors = try await generateEmbeddings(inputs: inputs)
        let modelName = (request["model"] as? String)
            ?? self.embeddingModelPath?.lastPathComponent
            ?? "unknown"
        let data: [[String: Any]] = vectors.enumerated().map { (i, v) in
            ["object": "embedding", "index": i, "embedding": v]
        }
        var totalTokens = 0
        if let container = self.embeddingContainer {
            totalTokens = await container.perform { _, tokenizer, _ in
                inputs.reduce(0) { $0 + tokenizer.encode(text: $1, addSpecialTokens: true).count }
            }
        }
        return [
            "object": "list",
            "data": data,
            "model": modelName,
            "usage": [
                "prompt_tokens": totalTokens,
                "total_tokens": totalTokens,
            ] as [String: Any],
        ]
    }

    /// Image generation via dict-form request. Python source:
    /// `vmlx_engine/server.py:3429 create_image`.
    ///
    /// OpenAI wire format:
    ///     {
    ///       "model": "flux1-schnell",
    ///       "prompt": "...",
    ///       "n": 1,
    ///       "size": "1024x1024",
    ///       "response_format": "b64_json" | "url",
    ///       "seed": 42
    ///     }
    ///
    /// Delegates to the typed SwiftUI-facing `generateImage(prompt:model:settings:)`
    /// which flows through `FluxBackend` → `vMLXFlux.FluxEngine`. The
    /// underlying model-specific `.generate()` bodies are still
    /// scaffolded (see `APP-SURFACE-AUDIT-2026-04-13.md`), so this
    /// endpoint will currently throw until those land — but the full
    /// request/response plumbing is now real so the day the DiT
    /// ports go in, the route works end-to-end with no further wiring.
    public func generateImage(request: [String: Any]) async throws -> [String: Any] {
        guard let prompt = request["prompt"] as? String, !prompt.isEmpty else {
            throw EngineError.invalidRequest("generateImage: missing 'prompt' field")
        }
        let model = (request["model"] as? String) ?? "flux1-schnell"

        // Parse `size` as "WxH"; default to square 1024.
        var width = 1024
        var height = 1024
        if let size = request["size"] as? String {
            let parts = size.split(separator: "x")
            if parts.count == 2,
               let w = Int(parts[0]), let h = Int(parts[1])
            {
                width = w; height = h
            }
        }
        let n = (request["n"] as? Int) ?? 1
        let seed = (request["seed"] as? Int) ?? -1
        let responseFormat = (request["response_format"] as? String) ?? "url"

        var settings = ImageGenSettings()
        settings.width = width
        settings.height = height
        settings.numImages = max(1, n)
        settings.seed = seed
        // Steps + guidance come from the model's default unless the
        // caller explicitly overrides. OpenAI's wire spec doesn't have
        // them, so we only read from vMLX extensions.
        if let steps = request["steps"] as? Int { settings.steps = steps }
        if let guidance = request["guidance_scale"] as? Double {
            settings.guidance = guidance
        }

        let url = try await self.generateImage(
            prompt: prompt, model: model, settings: settings
        )

        let created = Int(Date().timeIntervalSince1970)
        var entry: [String: Any] = [:]
        if responseFormat == "b64_json" {
            if let data = try? Data(contentsOf: url) {
                entry["b64_json"] = data.base64EncodedString()
            }
        } else {
            entry["url"] = url.absoluteString
        }
        return [
            "created": created,
            "data": [entry],
        ]
    }

    /// Image edit via dict-form request. Python source:
    /// `vmlx_engine/server.py:3632 create_image_edit`.
    ///
    /// OpenAI wire format: multipart body with `image` + `mask` + prompt.
    /// In practice callers hit this via the typed `editImage(prompt:source:mask:...)`
    /// method on the SwiftUI Image screen. This dict-form wraps that
    /// typed call for external HTTP clients once the server-side
    /// multipart parsing is hooked up.
    public func editImage(request: [String: Any]) async throws -> [String: Any] {
        guard let prompt = request["prompt"] as? String, !prompt.isEmpty else {
            throw EngineError.invalidRequest("editImage: missing 'prompt' field")
        }
        guard let imageB64 = request["image"] as? String,
              let source = Data(base64Encoded: imageB64)
        else {
            throw EngineError.invalidRequest("editImage: missing 'image' field (expected base64-encoded PNG/JPEG)")
        }
        var mask: Data? = nil
        if let maskB64 = request["mask"] as? String {
            mask = Data(base64Encoded: maskB64)
        }
        let strength = (request["strength"] as? Double) ?? 0.6
        let model = (request["model"] as? String) ?? ""
        var settings = ImageGenSettings()
        if let size = request["size"] as? String {
            let parts = size.split(separator: "x")
            if parts.count == 2,
               let w = Int(parts[0]), let h = Int(parts[1])
            {
                settings.width = w
                settings.height = h
            }
        }
        if let seed = request["seed"] as? Int { settings.seed = seed }
        if let steps = request["steps"] as? Int { settings.steps = steps }
        if let g = request["guidance"] as? Double { settings.guidance = g }
        let url = try await self.editImage(
            prompt: prompt,
            model: model,
            source: source,
            mask: mask,
            strength: strength,
            settings: settings
        )
        return [
            "created": Int(Date().timeIntervalSince1970),
            "data": [["url": url.absoluteString]],
        ]
    }

    // MARK: - Typed image API (wired by the Swift Image screen)
    //
    // The concrete methods live in `FluxBackend.swift` — this file just
    // declares the property stores (`fluxBackend`, `fluxJobs`) that the
    // extension uses. The actual `generateImage(prompt:model:settings:)`,
    // `editImage(...)`, and `imageGenStream(jobId:)` implementations all
    // route through `vMLXFlux.FluxEngine` via the bridge layer.

    /// Rerank endpoint. Python source: `vmlx_engine/server.py:3083 create_rerank`.
    ///
    /// Request shape (OpenAI/Cohere-compatible):
    ///     {
    ///       "model": "<embedding model>",
    ///       "query": "<query string>",
    ///       "documents": ["doc1", "doc2", ...],
    ///       "top_n": 5,
    ///       "return_documents": true
    ///     }
    ///
    /// Response shape:
    ///     {
    ///       "object": "list",
    ///       "model": "...",
    ///       "results": [
    ///         { "index": 2, "relevance_score": 0.87, "document": {"text": "..."} },
    ///         ...
    ///       ]
    ///     }
    ///
    /// Implementation notes: for now we score via cosine similarity
    /// over embeddings rather than a dedicated cross-encoder model.
    /// This is the same approach the SentenceTransformers "bi-encoder"
    /// rerank does and it works with every embedding model we already
    /// support. A true cross-encoder port (BGE-reranker, Cohere, etc.)
    /// is deferred until we have a dedicated rerank loader.
    public func rerank(request: [String: Any]) async throws -> [String: Any] {
        guard let query = request["query"] as? String, !query.isEmpty else {
            throw EngineError.invalidRequest("rerank: missing 'query' field")
        }
        let rawDocs = (request["documents"] as? [Any]) ?? []
        // Documents can be plain strings OR `{text: "..."}` objects.
        var documents: [String] = []
        for entry in rawDocs {
            if let s = entry as? String { documents.append(s); continue }
            if let obj = entry as? [String: Any], let t = obj["text"] as? String {
                documents.append(t)
            }
        }
        if documents.isEmpty {
            return [
                "object": "list",
                "model": request["model"] ?? "",
                "results": [] as [Any],
            ]
        }
        let topN = (request["top_n"] as? Int) ?? documents.count
        let returnDocs = (request["return_documents"] as? Bool) ?? true

        // Compute embeddings via the existing path by reusing the
        // embeddings() entry point: query + documents all in one batch.
        var inputs: [String] = [query]
        inputs.append(contentsOf: documents)
        let embReq: [String: Any] = [
            "model": request["model"] ?? "",
            "input": inputs,
        ]
        let embResponse = try await embeddings(request: embReq)
        guard let data = embResponse["data"] as? [[String: Any]],
              data.count == inputs.count
        else {
            throw EngineError.notImplemented("rerank: embedding backend returned malformed data")
        }

        func vec(_ row: [String: Any]) -> [Double] {
            (row["embedding"] as? [Double]) ?? []
        }

        let queryVec = vec(data[0])
        var scored: [(index: Int, score: Double, text: String)] = []
        scored.reserveCapacity(documents.count)
        for i in 0..<documents.count {
            let docVec = vec(data[i + 1])
            let score = Self.cosineSimilarity(queryVec, docVec)
            scored.append((i, score, documents[i]))
        }
        // Sort by descending similarity.
        scored.sort { $0.score > $1.score }
        let top = Array(scored.prefix(max(1, topN)))

        let results: [[String: Any]] = top.map { row in
            var out: [String: Any] = [
                "index": row.index,
                "relevance_score": row.score,
            ]
            if returnDocs {
                out["document"] = ["text": row.text]
            }
            return out
        }
        return [
            "object": "list",
            "model": request["model"] ?? "",
            "results": results,
        ]
    }

    /// Cosine similarity between two equal-length vectors.
    /// Returns 0 for zero-length, mismatched-length, or all-zero inputs.
    private static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard !a.isEmpty, a.count == b.count else { return 0 }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = (normA.squareRoot() * normB.squareRoot())
        return denom > 0 ? dot / denom : 0
    }

    /// Soft sleep — keep weights in unified memory, flush caches so GPU can
    /// reclaim scratch memory while the session is idle. The next request
    /// wakes instantly (no reload) via `wakeFromStandby`. Python analog:
    /// `vmlx_engine/server.py:1880 admin_soft_sleep`.
    public func softSleep() async throws {
        // Audit 2026-04-16: if a stream is mid-generate, cancel it first
        // so the cacheCoordinator clear doesn't race with concurrent
        // fetch/store from the generation loop. CacheCoordinator's paged
        // and disk tiers aren't actor-isolated against `.clear()` being
        // invoked while another thread is inside fetch/store, so this
        // preemptive cancel is the straightforward race guard.
        if currentStreamTask != nil {
            currentStreamTask?.cancel()
            currentStreamTask = nil
            await logs.append(.info, category: "engine",
                "soft sleep: cancelled in-flight stream to avoid cache race")
        }
        // Clear the cache coordinator's in-memory tiers. Weights stay loaded.
        cacheCoordinator?.clear()
        // Drop the DFlash drafter + cached target adapter too. Their KV
        // state tracks the target model's now-cleared cache, so leaving
        // them live across a soft-sleep → wake cycle would feed the
        // next generation stale state and produce garbage. The drafter
        // checkpoint path is preserved as `_dflashDrafterURL` so a
        // subsequent wake can rebind via `autoLoadDFlashDrafterIfConfigured`.
        // Audit 2026-04-15 (lifecycle #4).
        _dflashDrafter = nil
        _dflashTarget = nil
        await logs.append(.info, category: "engine",
            "soft sleep — caches cleared, DFlash adapters cleared, weights retained")
        transition(.standby(.soft))
    }

    /// Deep sleep — unload weights entirely but keep the process alive.
    /// The next request will trigger a full reload from disk. Python analog:
    /// `vmlx_engine/server.py:1927 admin_deep_sleep`.
    public func deepSleep() async throws {
        cacheCoordinator?.clear()
        loaded = nil
        loadedModelPath = nil
        loadedJangConfig = nil
        cacheCoordinator = nil
        await logs.append(.info, category: "engine", "deep sleep — weights unloaded")
        transition(.standby(.deep))
    }

    /// Wake from standby. For `.standby(.soft)` this is a no-op (weights
    /// are still resident). For `.standby(.deep)` the retained
    /// `lastLoadOptions` is replayed, re-invoking `load()` so the engine
    /// is usable again on the next request. If no options are retained
    /// (first start, explicit unload), transition to `.stopped` so the
    /// caller can surface a "pick a model" banner. Python analog:
    /// `vmlx_engine/server.py:2020 admin_wake`.
    ///
    /// Optional `override` parameter lets the caller swap to a different
    /// model on wake (useful for `POST /admin/wake {model:"..."}` flows).
    public func wake(override: LoadOptions? = nil) async throws {
        switch state {
        case .standby(.soft):
            await logs.append(.info, category: "engine", "wake (soft) — instant")
            transition(.running)
        case .standby(.deep):
            // Deep wake — prefer an explicit override (model swap on wake)
            // over the retained LoadOptions. Either way, replay load() and
            // drain its stream to completion before returning so the
            // caller's next request sees a fully-warmed engine.
            guard let opts = override ?? lastLoadOptions else {
                await logs.append(.warn, category: "engine",
                    "wake (deep) — no retained LoadOptions; stopping")
                transition(.stopped)
                return
            }
            // P1-LIFE-2: when waking with an OVERRIDE model (different
            // path), drop the previous model's CacheCoordinator so the
            // new model gets a fresh paged cache + disk path. Reusing
            // the old coordinator would feed the new model a token-key
            // chain hashed under the wrong modelKey, plus layer counts
            // wouldn't match.
            if override != nil, override?.modelPath != lastLoadOptions?.modelPath {
                cacheCoordinator?.clear()
                cacheCoordinator = nil
                await logs.append(
                    .info, category: "cache",
                    "wake (deep) override model swap — cleared previous coordinator")
            }
            await logs.append(
                .info, category: "engine",
                "wake (deep) — replaying load from \(opts.modelPath.lastPathComponent)")
            let stream = self.load(opts)
            do {
                for try await event in stream {
                    if case .failed(let msg) = event {
                        // Retry-loop escape: if the retained LoadOptions
                        // point at a model that is now permanently unloadable
                        // (deleted from disk, corrupted config, etc.) the
                        // next `wakeFromStandby` would re-try the same failing
                        // path forever. Clear lastLoadOptions and transition
                        // to .stopped so the UI can surface a "pick a model"
                        // banner. Audit 2026-04-15 (lifecycle #5).
                        lastLoadOptions = nil
                        transition(.stopped)
                        throw EngineError.notImplemented("wake reload failed: \(msg)")
                    }
                }
            } catch {
                // Load threw (e.g. EngineError.modelNotFound from a deleted
                // model dir). Same retry-loop escape: clear options so the
                // next wake call doesn't re-try the broken path.
                lastLoadOptions = nil
                transition(.stopped)
                throw error
            }
        case .running:
            return
        default:
            throw EngineError.notImplemented("wake called in state \(state)")
        }
    }

    /// Cache stats — snapshot of prefix cache + paged cache + disk L2.
    /// Reads from the live `cacheCoordinator` if it exists, otherwise
    /// returns a "not loaded" placeholder. Mirrors the shape of the
    /// Python `cache_stats` endpoint in `vmlx_engine/server.py:2141`.
    ///
    /// Returned dict (top-level keys when loaded):
    /// - `loaded`, `isHybrid`
    /// - `paged`: enabled, blockSize, maxBlocks, blocksInUse, blocksFree,
    ///   utilizationPct, hitCount, missCount, hitRate, evictions
    /// - `disk`: enabled, maxGB, currentGB, entryCount, hitCount,
    ///   missCount, hitRate, storeCount, directory
    /// - `ssmCompanion`: enabled, hitCount, missCount, hitRate, maxEntries
    /// - `prefixCache`: enabled, size
    ///
    /// Stats fields that the upstream vmlx-swift-lm sub-caches don't expose
    /// publicly (e.g. SSM entry count, paged byte usage) are omitted rather
    /// than faked. See report comments inside for the current gaps.
    public func cacheStats() async throws -> [String: Any] {
        guard let coord = cacheCoordinator else {
            return ["loaded": false]
        }

        var out: [String: Any] = [
            "loaded": true,
            "isHybrid": coord.isHybrid,
        ]

        // Paged (L1) ---------------------------------------------------------
        if let paged = coord.pagedCache {
            let s = paged.stats
            // Block 0 is a null sentinel — usable pool = maxBlocks - 1.
            let usable = max(paged.maxBlocks - 1, 1)
            let inUse = max(s.allocatedBlocks, 0)
            let free = max(s.freeBlocks, 0)
            let util = min(100.0, (Double(inUse) / Double(usable)) * 100.0)
            let total = s.cacheHits + s.cacheMisses
            let hitRate = total > 0 ? Double(s.cacheHits) / Double(total) : 0.0
            out["paged"] = [
                "enabled": true,
                "blockSize": paged.blockSize,
                "maxBlocks": paged.maxBlocks,
                "blocksInUse": inUse,
                "blocksFree": free,
                "utilizationPct": util,
                "hitCount": s.cacheHits,
                "missCount": s.cacheMisses,
                "hitRate": hitRate,
                "evictions": s.evictions,
            ] as [String: Any]
        } else {
            out["paged"] = ["enabled": false] as [String: Any]
        }

        // Disk (L2) ----------------------------------------------------------
        if let disk = coord.diskCache {
            // Walk the cacheDir to compute live entry count + current size.
            // DiskCache doesn't expose these publicly; filesystem walk is
            // the cheapest portable path and the directory is small (1 dir
            // of .safetensors files + sqlite index).
            //
            // Swift-6 async context note: `FileManager.enumerator(at:...)`
            // returns a `DirectoryEnumerator` whose `makeIterator` is
            // marked unavailable from async contexts (sync-only iterator
            // over potentially blocking disk reads). We use the flat
            // `contentsOfDirectory(at:)` variant instead — it's a single
            // readdir call, returns an array synchronously, and plays
            // fine with strict concurrency. The directory is small so
            // one-shot enumeration is cheaper than a lazy enumerator
            // anyway.
            var entryCount = 0
            var currentBytes: Int64 = 0
            if let contents = try? FileManager.default.contentsOfDirectory(
                at: disk.cacheDir,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
            ) {
                for url in contents {
                    guard url.pathExtension == "safetensors" else { continue }
                    entryCount += 1
                    if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                        currentBytes += Int64(size)
                    }
                }
            }
            let currentGB = Double(currentBytes) / 1_073_741_824.0
            let total = disk.hits + disk.misses
            let hitRate = total > 0 ? Double(disk.hits) / Double(total) : 0.0
            out["disk"] = [
                "enabled": true,
                "maxGB": Double(coord.config.diskCacheMaxGB),
                "currentGB": currentGB,
                "entryCount": entryCount,
                "hitCount": disk.hits,
                "missCount": disk.misses,
                "hitRate": hitRate,
                "storeCount": disk.stores,
                "directory": disk.cacheDir.path,
            ] as [String: Any]
        } else {
            out["disk"] = ["enabled": false] as [String: Any]
        }

        // Memory cache (L1.5 byte-budgeted) ----------------------------------
        if let mem = coord.memoryCache {
            let s = mem.snapshotStats()
            out["memory"] = [
                "enabled": true,
                "entryCount": s.entryCount,
                "currentMemoryMB": Double(s.currentMemoryBytes) / 1_048_576.0,
                "maxMemoryMB": Double(s.maxMemoryBytes) / 1_048_576.0,
                "hitCount": s.hits,
                "missCount": s.misses,
                "hitRate": s.hitRate,
                "evictions": s.evictions,
                "utilizationPct": s.memoryUtilization * 100.0,
            ] as [String: Any]
        } else {
            out["memory"] = ["enabled": false] as [String: Any]
        }

        // SSM companion ------------------------------------------------------
        let ssm = coord.ssmStateCache
        let ssmTotal = ssm.hits + ssm.misses
        let ssmHitRate = ssmTotal > 0 ? Double(ssm.hits) / Double(ssmTotal) : 0.0
        out["ssmCompanion"] = [
            "enabled": coord.isHybrid,
            "hitCount": ssm.hits,
            "missCount": ssm.misses,
            "hitRate": ssmHitRate,
            "maxEntries": coord.config.ssmMaxEntries,
        ] as [String: Any]

        // Prefix cache (same underlying paged pool) --------------------------
        out["prefixCache"] = [
            "enabled": coord.config.usePagedCache,
            "size": coord.config.maxCacheBlocks,
        ] as [String: Any]

        return out
    }

    /// Clear all cache tiers without changing engine state.
    /// Called from the CachePanel "Clear caches" footer action.
    /// Unlike `softSleep()`, weights stay hot and state stays `.running`.
    public func clearCaches() async {
        cacheCoordinator?.clear()
        await logs.append(.info, category: "cache", "caches cleared via panel action")
    }

    /// `GET /v1/cache/entries` backing — returns a per-tier entry summary
    /// suitable for JSON serialization. Mirrors Python's
    /// `/v1/cache/entries` shape: top-level `entries` count plus per-tier
    /// breakdown so a client can show "how many prompts are cached
    /// where". Entries are derived from cacheStats() plus any per-tier
    /// detail the coordinator can expose. Returns an empty payload when
    /// no model is loaded.
    public func cacheEntries() async throws -> [String: Any] {
        guard cacheCoordinator != nil else {
            return ["object": "cache.entries", "entries": [] as [Any], "tiers": [:] as [String: Any]]
        }
        let stats = try await cacheStats()
        var resp: [String: Any] = ["object": "cache.entries"]
        resp["stats"] = stats
        // Best-effort: copy through the tier counts surfaced by stats.
        // The CacheCoordinator does not currently expose a flat list of
        // cached prompt hashes (would require adding an enumerator to
        // pagedCache + diskCache). When that lands, replace this with the
        // real list. For now the count is enough for a UI that wants
        // "N prompts cached" + "M MB on disk".
        return resp
    }

    /// `POST /v1/cache/warm` backing — runs a one-token generation per
    /// prompt so the prefix cache populates naturally. This is the same
    /// approach Python uses (it just sends a 1-token chat request per
    /// warmup string). Returns the count of prompts processed.
    public func cacheWarm(prompts: [String], model: String) async throws -> Int {
        var warmed = 0
        for prompt in prompts {
            let req = ChatRequest(
                model: model.isEmpty ? (loadedModelPath?.path ?? "") : model,
                messages: [.init(role: "user", content: .string(prompt))],
                stream: false,
                maxTokens: 1)
            let stream = self.stream(request: req)
            do {
                for try await _ in stream { /* drain */ }
                warmed += 1
            } catch {
                await logs.append(.warn, category: "cache",
                    "warm failed for prompt[\(warmed)]: \(error)")
            }
        }
        return warmed
    }

    /// Test-only: inject a pre-built `CacheCoordinator` so `cacheStats()` can
    /// be exercised without loading a real model. Not part of the public API
    /// surface — guarded by `internal` + `@testable import`.
    internal func _setCacheCoordinatorForTesting(_ coord: CacheCoordinator?) {
        self.cacheCoordinator = coord
    }

    // MARK: - Benchmark
    //
    // Benchmark surface exposed to the Server screen `BenchmarkPanel`.
    // Three suites run real generation through `Engine.stream` (no
    // shortcuts): `.decode256` measures decode tok/s after warmup,
    // `.prefill1024` measures TTFT over a ~1024-token prompt, and
    // `.cacheTurn5` measures prefix-cache hit ratio across 5 multi-turn
    // requests. Each suite yields `BenchEvent.progress` periodically
    // and a final `BenchEvent.done(report:)` (or `.failed(msg)` on error).
    //
    // Port parity with `vmlx_engine/benchmark.py`: the Python driver
    // additionally computes TPOT / generation_tps / processing_tps
    // splits from the same observations — Swift now surfaces those
    // fields directly on `BenchReport`. Peak MLX GPU memory is sampled
    // from `MLX.GPU.peakMemory` at the end of each run.

    public enum BenchSuite: String, Sendable, CaseIterable, Codable {
        case decode256       // "Decode speed (256 tok)"
        case prefill1024     // "Prefill speed (1024 tok)"
        case cacheTurn5      // "Cache turn (5 multi-turn)"

        public var displayName: String {
            switch self {
            case .decode256:   return "Decode speed (256 tok)"
            case .prefill1024: return "Prefill speed (1024 tok)"
            case .cacheTurn5:  return "Cache turn (5 multi-turn)"
            }
        }
    }

    /// Result of a single benchmark run. Matches Python's `BenchmarkResult`
    /// derivation rules (TPOT excludes the first token; processing_tps
    /// is prompt_tokens / ttft_sec). Extended fields are optional so
    /// existing `BenchRow` SQLite decoders that only read the 4 original
    /// headline columns still round-trip cleanly.
    public struct BenchReport: Sendable, Codable {
        public var suite: BenchSuite
        public var modelId: String
        public var date: Date
        // Headline (kept for back-compat with BenchRow schema).
        public var tokensPerSec: Double     // generation tokens / run time
        public var ttftMs: Double           // time to first emitted content
        public var totalMs: Double          // end-to-end wall time
        public var cacheHitRate: Double     // cachedTokens / promptTokens across turns
        public var notes: String

        // Extended fields (Python benchmark.py parity).
        /// Time Per Output Token, excluding TTFT. `(totalMs - ttftMs) / (generated - 1)`.
        public var tpotMs: Double?
        /// Generation throughput — decoded tokens / (totalMs - ttftMs), in tok/s.
        /// Subtly different from `tokensPerSec` which uses total wall time.
        public var generationTps: Double?
        /// Prompt-processing throughput — promptTokens / ttftMs, in tok/s.
        public var processingTps: Double?
        /// Peak MLX GPU memory observed during the run, in gigabytes.
        public var peakMemoryGB: Double?
        /// Observed prompt + completion token counts (from the last stream's usage).
        public var promptTokens: Int?
        public var completionTokens: Int?

        public init(
            suite: BenchSuite, modelId: String, date: Date = Date(),
            tokensPerSec: Double, ttftMs: Double, totalMs: Double,
            cacheHitRate: Double, notes: String = "",
            tpotMs: Double? = nil,
            generationTps: Double? = nil,
            processingTps: Double? = nil,
            peakMemoryGB: Double? = nil,
            promptTokens: Int? = nil,
            completionTokens: Int? = nil
        ) {
            self.suite = suite; self.modelId = modelId; self.date = date
            self.tokensPerSec = tokensPerSec; self.ttftMs = ttftMs
            self.totalMs = totalMs; self.cacheHitRate = cacheHitRate
            self.notes = notes
            self.tpotMs = tpotMs
            self.generationTps = generationTps
            self.processingTps = processingTps
            self.peakMemoryGB = peakMemoryGB
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
        }
    }

    /// Aggregated summary across N benchmark runs. Port of Python's
    /// `BenchmarkSummary` stat fields — mean/min/max/p50/p95 across
    /// TTFT + TPOT + latency.
    public struct BenchSummary: Sendable, Codable {
        public var suite: BenchSuite
        public var modelId: String
        public var date: Date
        public var runs: Int

        // TTFT stats (ms).
        public var ttftMean: Double
        public var ttftMin: Double
        public var ttftMax: Double
        public var ttftP50: Double
        public var ttftP95: Double

        // TPOT stats (ms). 0 when the suite isn't a decode suite.
        public var tpotMean: Double
        public var tpotMin: Double
        public var tpotMax: Double

        // Throughput stats (tok/s).
        public var generationTpsMean: Double
        public var generationTpsMax: Double
        public var processingTpsMean: Double

        // Latency stats (ms).
        public var latencyMean: Double
        public var latencyMin: Double
        public var latencyMax: Double
        public var latencyP50: Double
        public var latencyP95: Double

        /// Headline number for the panel — pick the most representative
        /// single value for the suite. Decode → `generationTpsMean`,
        /// prefill → `processingTpsMean`, cache → hit rate from the last run.
        public var headlineTokensPerSec: Double

        /// All individual runs that made up the summary.
        public var runReports: [BenchReport]

        public init(
            suite: BenchSuite, modelId: String, date: Date = Date(),
            runs: Int,
            ttftMean: Double, ttftMin: Double, ttftMax: Double, ttftP50: Double, ttftP95: Double,
            tpotMean: Double, tpotMin: Double, tpotMax: Double,
            generationTpsMean: Double, generationTpsMax: Double, processingTpsMean: Double,
            latencyMean: Double, latencyMin: Double, latencyMax: Double,
            latencyP50: Double, latencyP95: Double,
            headlineTokensPerSec: Double,
            runReports: [BenchReport]
        ) {
            self.suite = suite; self.modelId = modelId; self.date = date
            self.runs = runs
            self.ttftMean = ttftMean; self.ttftMin = ttftMin; self.ttftMax = ttftMax
            self.ttftP50 = ttftP50; self.ttftP95 = ttftP95
            self.tpotMean = tpotMean; self.tpotMin = tpotMin; self.tpotMax = tpotMax
            self.generationTpsMean = generationTpsMean
            self.generationTpsMax = generationTpsMax
            self.processingTpsMean = processingTpsMean
            self.latencyMean = latencyMean; self.latencyMin = latencyMin
            self.latencyMax = latencyMax
            self.latencyP50 = latencyP50; self.latencyP95 = latencyP95
            self.headlineTokensPerSec = headlineTokensPerSec
            self.runReports = runReports
        }
    }

    public enum BenchEvent: Sendable {
        case progress(Double, String)  // (fraction 0..1, label)
        case done(BenchReport)
        case failed(String)
    }

    /// Run a benchmark suite against the currently-loaded chat model.
    ///
    /// - `.decode256`  — 10-token warmup, then time a 256-token generation
    ///   on a fixed short prompt. Reports tokensPerSec + TTFT.
    /// - `.prefill1024` — synthetic ~1024-token prompt, measure first-token
    ///   latency + promptTokensPerSecond.
    /// - `.cacheTurn5` — 5 sequential turns where each turn's prompt
    ///   includes all prior turns. Measure prefix-cache hit ratio by
    ///   querying the coordinator before each turn.
    ///
    /// Emits `.progress` periodically so BenchmarkPanel's live bar advances
    /// and a final `.done(report:)` (or `.failed` on error).
    public func benchmark(suite: BenchSuite) -> AsyncThrowingStream<BenchEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    guard await self.loaded != nil else {
                        continuation.yield(.failed("No model loaded — call Engine.load first"))
                        continuation.finish()
                        return
                    }
                    await self.logs.append(
                        .info, category: "bench",
                        "benchmark started: \(suite.displayName)")
                    continuation.yield(.progress(0.0, "Starting \(suite.displayName)"))

                    let report: BenchReport
                    switch suite {
                    case .decode256:
                        report = try await self.runDecodeBench(
                            suite: suite, maxTokens: 256, continuation: continuation)
                    case .prefill1024:
                        report = try await self.runPrefillBench(
                            suite: suite, continuation: continuation)
                    case .cacheTurn5:
                        report = try await self.runCacheMultiturnBench(
                            suite: suite, continuation: continuation)
                    }

                    await self.logs.append(
                        .info, category: "bench",
                        "benchmark complete: \(suite.displayName) " +
                        "tok/s=\(String(format: "%.1f", report.tokensPerSec)) " +
                        "ttft=\(String(format: "%.0f", report.ttftMs))ms")
                    continuation.yield(.progress(1.0, "Done"))
                    continuation.yield(.done(report))
                    continuation.finish()
                } catch {
                    await self.logs.append(
                        .error, category: "bench",
                        "benchmark failed: \(suite.displayName) — \(error)")
                    continuation.yield(.failed("\(error)"))
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Benchmark suite implementations

    private func currentBenchModelId() -> String {
        loadedModelPath?.lastPathComponent ?? "unknown"
    }

    /// Decode-speed benchmark. 10 warmup tokens (not timed) followed by a
    /// `maxTokens`-token generation on a fixed short prompt. Drains the
    /// real `Engine.stream` path so the benchmark sees the same prefix
    /// cache, scheduler, and parser wiring a real request would hit.
    private func runDecodeBench(
        suite: BenchSuite,
        maxTokens: Int,
        continuation: AsyncThrowingStream<BenchEvent, Error>.Continuation
    ) async throws -> BenchReport {
        let prompt = "The quick brown fox jumps over the lazy dog."

        // Warmup — 10 tokens, discard timing.
        let warmup = ChatRequest(
            model: currentBenchModelId(),
            messages: [.init(role: "user", content: .string(prompt))],
            stream: true,
            maxTokens: 10,
            temperature: 0.0,
            enableThinking: false
        )
        continuation.yield(.progress(0.05, "Warmup (10 tok)"))
        for try await _ in self.stream(request: warmup) { /* drain */ }

        // Real run — `maxTokens` decoded tokens, timed.
        let req = ChatRequest(
            model: currentBenchModelId(),
            messages: [.init(role: "user", content: .string(prompt))],
            stream: true,
            maxTokens: maxTokens,
            temperature: 0.0,
            enableThinking: false
        )
        let start = Date()
        var firstTokenAt: Date? = nil
        var tokenCount = 0
        var usage: StreamChunk.Usage? = nil
        for try await chunk in self.stream(request: req) {
            if let c = chunk.content, !c.isEmpty {
                if firstTokenAt == nil { firstTokenAt = Date() }
                tokenCount += 1
                if tokenCount % 16 == 0 {
                    let frac = 0.1 + 0.85 * (Double(tokenCount) / Double(maxTokens))
                    continuation.yield(.progress(
                        min(0.95, frac),
                        "\(tokenCount)/\(maxTokens) tok"
                    ))
                }
            }
            if let u = chunk.usage { usage = u }
        }
        let end = Date()
        let totalSec = end.timeIntervalSince(start)
        let decodedTokens = usage?.completionTokens ?? tokenCount
        let ttftMs = firstTokenAt.map { $0.timeIntervalSince(start) * 1000 } ?? 0
        let ttftSec = ttftMs / 1000.0
        let tps = totalSec > 0 ? Double(decodedTokens) / totalSec : 0

        // TPOT = (total − ttft) / (generated − 1). Matches Python
        // BenchmarkResult.__post_init__. Guard on `decodedTokens > 1`
        // so the denominator is non-zero for trivially-short runs.
        let decodeSec = max(0, totalSec - ttftSec)
        let tpotMs: Double? = decodedTokens > 1 && decodeSec > 0
            ? (decodeSec * 1000) / Double(decodedTokens - 1)
            : nil
        let generationTps: Double? = decodedTokens > 1 && decodeSec > 0
            ? Double(decodedTokens - 1) / decodeSec
            : nil
        let processingTps: Double? = ttftSec > 0 && (usage?.promptTokens ?? 0) > 0
            ? Double(usage!.promptTokens) / ttftSec
            : nil
        let peakGB = currentPeakMLXMemoryGB()

        return BenchReport(
            suite: suite,
            modelId: currentBenchModelId(),
            tokensPerSec: tps,
            ttftMs: ttftMs,
            totalMs: totalSec * 1000,
            cacheHitRate: 0,
            notes: "decoded=\(decodedTokens)",
            tpotMs: tpotMs,
            generationTps: generationTps,
            processingTps: processingTps,
            peakMemoryGB: peakGB,
            promptTokens: usage?.promptTokens,
            completionTokens: decodedTokens
        )
    }

    /// Prefill-speed benchmark. Builds a synthetic ~1024-token prompt (a
    /// short sentence repeated ~200 times) and measures first-token
    /// latency. Reports `tokensPerSec` as the prompt-prefill rate
    /// (promptTokenCount / ttft_sec) and surfaces TTFT as the headline.
    private func runPrefillBench(
        suite: BenchSuite,
        continuation: AsyncThrowingStream<BenchEvent, Error>.Continuation
    ) async throws -> BenchReport {
        // ~5 words/sentence × 200 repeats → ~1024 BPE tokens on most
        // Latin-alphabet tokenizers. Matches the Python bench default.
        let sentence = "The quick brown fox jumps. "
        let prompt = String(repeating: sentence, count: 200)
        let req = ChatRequest(
            model: currentBenchModelId(),
            messages: [.init(role: "user", content: .string(prompt))],
            stream: true,
            maxTokens: 8,
            temperature: 0.0,
            enableThinking: false
        )
        continuation.yield(.progress(0.1, "Prefilling ~1024 tok"))
        let start = Date()
        var firstTokenAt: Date? = nil
        var usage: StreamChunk.Usage? = nil
        for try await chunk in self.stream(request: req) {
            if let c = chunk.content, !c.isEmpty, firstTokenAt == nil {
                firstTokenAt = Date()
                continuation.yield(.progress(0.8, "First token"))
            }
            if let u = chunk.usage { usage = u }
        }
        let end = Date()
        let totalSec = end.timeIntervalSince(start)
        let ttftSec = firstTokenAt.map { $0.timeIntervalSince(start) } ?? totalSec
        let promptTokens = usage?.promptTokens ?? 1024
        let completionTokens = usage?.completionTokens ?? 0
        let prefillTps = ttftSec > 0 ? Double(promptTokens) / ttftSec : 0

        // For the prefill suite, processing_tps IS the headline. Still
        // populate generation_tps + tpot for the tail of the short
        // 8-token generation so the extended fields are consistent.
        let decodeSec = max(0, totalSec - ttftSec)
        let tpotMs: Double? = completionTokens > 1 && decodeSec > 0
            ? (decodeSec * 1000) / Double(completionTokens - 1)
            : nil
        let generationTps: Double? = completionTokens > 1 && decodeSec > 0
            ? Double(completionTokens - 1) / decodeSec
            : nil
        let peakGB = currentPeakMLXMemoryGB()

        return BenchReport(
            suite: suite,
            modelId: currentBenchModelId(),
            tokensPerSec: prefillTps,
            ttftMs: ttftSec * 1000,
            totalMs: totalSec * 1000,
            cacheHitRate: 0,
            notes: "promptTokens=\(promptTokens)",
            tpotMs: tpotMs,
            generationTps: generationTps,
            processingTps: prefillTps,
            peakMemoryGB: peakGB,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )
    }

    /// Cache-multiturn benchmark. 5 sequential turns where each turn's
    /// prompt includes all prior turns. Measures prefix-cache hit rate by
    /// inspecting `StreamChunk.Usage.cachedTokens` after each turn. The
    /// headline `cacheHitRate` is cumulative `cachedTokens / promptTokens`
    /// across turns 2..5 (turn 1 is pure prefill and excluded from the
    /// ratio because it has nothing to hit against).
    private func runCacheMultiturnBench(
        suite: BenchSuite,
        continuation: AsyncThrowingStream<BenchEvent, Error>.Continuation
    ) async throws -> BenchReport {
        let turns = 5
        var messages: [ChatRequest.Message] = [
            .init(role: "system", content: .string("You are a terse benchmark assistant."))
        ]
        var totalCached = 0
        var totalPrompt = 0
        let runStart = Date()
        var firstTokenAt: Date? = nil
        var totalGenTokens = 0
        for i in 0..<turns {
            continuation.yield(.progress(
                Double(i) / Double(turns),
                "Turn \(i + 1)/\(turns)"
            ))
            messages.append(.init(
                role: "user",
                content: .string("Turn \(i + 1): repeat the word 'ack' exactly once.")
            ))
            let req = ChatRequest(
                model: currentBenchModelId(),
                messages: messages,
                stream: true,
                maxTokens: 16,
                temperature: 0.0,
                enableThinking: false
            )
            var assistant = ""
            var usage: StreamChunk.Usage? = nil
            for try await chunk in self.stream(request: req) {
                if let c = chunk.content {
                    assistant += c
                    if firstTokenAt == nil, !c.isEmpty { firstTokenAt = Date() }
                }
                if let u = chunk.usage { usage = u }
            }
            messages.append(.init(role: "assistant", content: .string(assistant)))
            if let u = usage {
                if i > 0 {
                    totalCached += u.cachedTokens
                    totalPrompt += u.promptTokens
                }
                totalGenTokens += u.completionTokens
            }
        }
        let end = Date()
        let totalSec = end.timeIntervalSince(runStart)
        let hitRate = totalPrompt > 0 ? Double(totalCached) / Double(totalPrompt) : 0
        let tps = totalSec > 0 ? Double(totalGenTokens) / totalSec : 0
        let ttftMs = firstTokenAt.map { $0.timeIntervalSince(runStart) * 1000 } ?? 0
        let peakGB = currentPeakMLXMemoryGB()
        return BenchReport(
            suite: suite,
            modelId: currentBenchModelId(),
            tokensPerSec: tps,
            ttftMs: ttftMs,
            totalMs: totalSec * 1000,
            cacheHitRate: hitRate,
            notes: "cached=\(totalCached)/\(totalPrompt)",
            // Cache suite is dominated by wall time across all turns;
            // a per-run TPOT split isn't meaningful. Leave those nil
            // and let the panel render the `cacheHitRate` headline.
            peakMemoryGB: peakGB,
            promptTokens: totalPrompt,
            completionTokens: totalGenTokens
        )
    }

    // MARK: - Benchmark helpers

    /// Snapshot of the MLX GPU peak memory (gigabytes). Returns `nil`
    /// when the MLX runtime isn't initialized (for example during an
    /// early engine-load failure before the first kernel dispatch).
    private func currentPeakMLXMemoryGB() -> Double? {
        // `MLX.Memory.peakMemory` is a free accessor that reads
        // the runtime's own counter — no Metal library load required.
        // We guard defensively because the first-run non-initialized
        // path returns 0 which we want to report as `nil` (absent),
        // not 0.0 (accurately-zero).
        let bytes = MLX.Memory.peakMemory
        guard bytes > 0 else { return nil }
        return Double(bytes) / 1_073_741_824.0
    }

    // MARK: - Multi-run benchmark aggregation

    /// Run a benchmark suite N times back-to-back and aggregate the
    /// per-run reports into a `BenchSummary` with p50/p95 stats.
    ///
    /// Emits `.progress(runIndex / runs, "Run k/N")` between runs and
    /// forwards the individual `.progress` events from each underlying
    /// run with a scaled fraction. On any per-run failure, yields
    /// `.failed(msg)` for the failed run and continues with the rest
    /// — the final summary only includes the successful runs.
    public func benchmark(
        suite: BenchSuite, runs: Int
    ) -> AsyncThrowingStream<BenchEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                guard runs >= 1 else {
                    continuation.yield(.failed("runs must be >= 1"))
                    continuation.finish()
                    return
                }
                var reports: [BenchReport] = []
                for i in 0..<runs {
                    continuation.yield(.progress(
                        Double(i) / Double(runs),
                        "Run \(i + 1)/\(runs)"
                    ))
                    do {
                        // Run the single-shot suite in-line and capture
                        // the final report. We re-enter the existing
                        // `benchmark(suite:)` entrypoint so the per-run
                        // logging + error path stays consistent.
                        let innerStream = await self.benchmark(suite: suite)
                        var captured: BenchReport? = nil
                        for try await ev in innerStream {
                            if case .done(let r) = ev { captured = r }
                            if case .failed(let msg) = ev {
                                await self.logs.append(
                                    .warn, category: "bench",
                                    "run \(i + 1)/\(runs) failed: \(msg)")
                                continuation.yield(.failed(msg))
                            }
                        }
                        if let r = captured { reports.append(r) }
                    } catch {
                        await self.logs.append(
                            .error, category: "bench",
                            "run \(i + 1)/\(runs) exception: \(error)")
                        continuation.yield(.failed("\(error)"))
                    }
                }
                guard !reports.isEmpty else {
                    continuation.yield(.failed("all runs failed"))
                    continuation.finish()
                    return
                }
                let modelId = await self.loadedModelPath?.lastPathComponent ?? "unknown"
                let summary = Self.summarize(
                    suite: suite,
                    modelId: modelId,
                    reports: reports
                )
                continuation.yield(.progress(1.0, "Done (\(reports.count) runs)"))
                // We still yield a `.done(BenchReport)` so existing UI
                // code that listens for it renders the headline row.
                // The full summary is attached to the report's `notes`
                // field as pretty text for now; a dedicated
                // `.doneSummary(BenchSummary)` variant could land later.
                var last = reports[reports.count - 1]
                last.notes = "runs=\(reports.count) "
                    + "p50_ttft=\(String(format: "%.0f", summary.ttftP50))ms "
                    + "p95_ttft=\(String(format: "%.0f", summary.ttftP95))ms "
                    + "mean_tps=\(String(format: "%.1f", summary.generationTpsMean))"
                continuation.yield(.done(last))
                continuation.finish()
            }
        }
    }

    /// Aggregate a list of `BenchReport`s into a single `BenchSummary`.
    /// Public so test code can exercise the math without running a
    /// real generation loop.
    public static func summarize(
        suite: BenchSuite,
        modelId: String,
        reports: [BenchReport]
    ) -> BenchSummary {
        precondition(!reports.isEmpty, "summarize requires at least one report")
        let ttfts = reports.map { $0.ttftMs }
        let lats = reports.map { $0.totalMs }
        let tpots = reports.compactMap { $0.tpotMs }
        let genTps = reports.compactMap { $0.generationTps }
        let procTps = reports.compactMap { $0.processingTps }

        // Python's headline tokens/sec is the mean of generation_tps
        // for decode suites, falling back to the tokens_per_sec field
        // for suites that don't split (cache/prefill).
        let headlineTps: Double
        switch suite {
        case .decode256:
            headlineTps = genTps.isEmpty
                ? (reports.map { $0.tokensPerSec }.reduce(0, +) / Double(reports.count))
                : (genTps.reduce(0, +) / Double(genTps.count))
        case .prefill1024:
            headlineTps = procTps.isEmpty
                ? (reports.map { $0.tokensPerSec }.reduce(0, +) / Double(reports.count))
                : (procTps.reduce(0, +) / Double(procTps.count))
        case .cacheTurn5:
            headlineTps = reports.map { $0.tokensPerSec }.reduce(0, +)
                / Double(reports.count)
        }

        return BenchSummary(
            suite: suite, modelId: modelId, runs: reports.count,
            ttftMean: mean(ttfts), ttftMin: ttfts.min() ?? 0,
            ttftMax: ttfts.max() ?? 0,
            ttftP50: percentile(ttfts, 50),
            ttftP95: percentile(ttfts, 95),
            tpotMean: mean(tpots), tpotMin: tpots.min() ?? 0,
            tpotMax: tpots.max() ?? 0,
            generationTpsMean: mean(genTps),
            generationTpsMax: genTps.max() ?? 0,
            processingTpsMean: mean(procTps),
            latencyMean: mean(lats), latencyMin: lats.min() ?? 0,
            latencyMax: lats.max() ?? 0,
            latencyP50: percentile(lats, 50),
            latencyP95: percentile(lats, 95),
            headlineTokensPerSec: headlineTps,
            runReports: reports
        )
    }

    /// Mean of a Double array, zero when empty (matches Python's
    /// `statistics.mean(xs) if xs else 0.0`).
    private static func mean(_ xs: [Double]) -> Double {
        xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count)
    }

    /// Nearest-rank percentile. Port of Python's
    /// `calculate_percentile(data, percentile)` at benchmark.py:318.
    /// Returns 0 for empty input, the single value for a 1-element
    /// input, and otherwise `sorted[ceil((p/100) * n) - 1]` clamped
    /// to the valid index range.
    public static func percentile(_ xs: [Double], _ p: Double) -> Double {
        guard !xs.isEmpty else { return 0 }
        guard xs.count > 1 else { return xs[0] }
        let sorted = xs.sorted()
        // Parenthesize carefully: we want ceil of the full product,
        // not ceil of just the count. Swift's `.rounded(.up)` on the
        // raw Double binds tighter than the multiply, so we force it
        // into a single expression.
        let product = (p / 100.0) * Double(sorted.count)
        let rank = Int(product.rounded(.up))
        let idx = max(0, min(sorted.count - 1, rank - 1))
        return sorted[idx]
    }
}

public enum EngineError: Error, CustomStringConvertible {
    case notImplemented(String)
    /// Request rejected for structural reasons (missing required field,
    /// malformed JSON, invalid enum value, etc). Distinct from
    /// `.notImplemented` — this isn't a missing feature, it's a bad
    /// input. HTTP routes map this to 400, not 501. Used by
    /// `Engine.embeddings`, `generateImage`, `editImage`, `rerank` for
    /// the pre-flight validation shims that used to throw
    /// `.notImplemented("missing 'input' field")` and mislead clients.
    case invalidRequest(String)
    case modelNotFound(URL)
    case unsupportedModelType(String)
    /// Emitted by `performStreamingGeneration` when the model has emitted
    /// the exact same tool call 3 times in a row. Prevents the "nuclear
    /// retry" loop from `project_session_2026_03_21.md`.
    case toolCallRepetition(String)
    /// Emitted when the request watchdog (settings.requestTimeout) fires.
    /// The stream also yields a `finishReason: "timeout"` chunk before
    /// throwing this, so UIs that only read chunks still see the timeout.
    case requestTimeout(TimeInterval)
    /// Emitted by `Server.run()` when the configured port is already bound
    /// by another process or another Engine instance. Fixes mlxstudio #44
    /// (gateway/session port overlap → fatal trap) by surfacing a clean
    /// error instead of letting Hummingbird crash on bind failure.
    case portInUse(Int)
    /// `loadAdapter` / `fuseAdapter` / `unloadAdapter` called but no
    /// chat model is loaded.
    case notLoaded
    /// An adapter directory is missing `adapter_config.json` or
    /// `adapters.safetensors`. The string is the missing file's path.
    case adapterMissingFile(String)
    /// `unloadAdapter` called on an adapter that has already been
    /// permanently fused into the base weights. The caller must
    /// reload the model from disk to recover the pre-fusion state.
    case adapterAlreadyFused
    /// `fuseAdapter` / `unloadAdapter` called but no adapter is active.
    case adapterNotLoaded
    /// The request set `tool_choice="required"` or
    /// `tool_choice={function:...}` but the model emitted text only
    /// or called a different function. Audit round 5 enforcement —
    /// mirrors OpenAI spec contract so clients don't get silent
    /// text in place of the tool call they asked for.
    case toolChoiceNotSatisfied(String)
    /// Prompt token count exceeds the `GlobalSettings.maxPromptTokens`
    /// ceiling. Rejected pre-flight so MLX Metal allocation
    /// fatalError doesn't crash the process. Set `maxPromptTokens: 0`
    /// to disable (not recommended in production).
    case promptTooLong(tokens: Int, limit: Int)

    public var description: String {
        switch self {
        case .notImplemented(let s): return "not implemented: \(s)"
        case .invalidRequest(let s): return "invalid request: \(s)"
        case .modelNotFound(let u):  return "model not found: \(u.path)"
        case .unsupportedModelType(let t): return "unsupported model_type: \(t)"
        case .toolCallRepetition(let sig):
            return "tool call repetition detected: \(sig)"
        case .requestTimeout(let t):
            return "request timed out after \(t)s"
        case .portInUse(let p):
            return "port \(p) is already in use"
        case .notLoaded:
            return "no chat model is loaded"
        case .adapterMissingFile(let p):
            return "adapter directory missing required file: \(p)"
        case .adapterAlreadyFused:
            return "adapter has been fused into the base weights — reload the model to recover"
        case .adapterNotLoaded:
            return "no LoRA adapter is currently active"
        case .toolChoiceNotSatisfied(let msg):
            return "tool_choice not satisfied: \(msg)"
        case .promptTooLong(let tokens, let limit):
            return "prompt too long: \(tokens) tokens exceeds the configured maxPromptTokens ceiling of \(limit). Shorten the prompt or raise `maxPromptTokens` in Server settings."
        }
    }
}
