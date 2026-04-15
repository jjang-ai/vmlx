import ArgumentParser
import Foundation
import MLX
import MLXFast
import vMLXEngine
import vMLXLLM
import vMLXLMCommon
import vMLXServer

/// Wrapper around MLX.asyncEval so callers don't have to spell out the
/// raw identifier scattered across this file. `asyncEval` kicks off GPU
/// materialization without blocking the caller — sufficient here
/// because every subsequent `.item()` or shape access will block on
/// completion anyway.
@inline(__always)
private func materialize(_ a: MLXArray) { asyncEval(a) }

@main
struct VMLX: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vmlx",
        abstract: "vMLX — local LLM server (Swift)",
        subcommands: [Serve.self, Chat.self, Pull.self, List.self, DFlashSmoke.self],
        defaultSubcommand: Serve.self
    )
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Start the OpenAI-compatible server")

    @Option(name: .shortAndLong) var model: String
    @Option(name: .shortAndLong) var host: String = "127.0.0.1"
    @Option(name: .shortAndLong) var port: Int = 8000
    @Option(name: .long) var apiKey: String?
    @Flag(name: .long, help: "Print progress as JSON-lines to stderr for scripting.")
    var jsonProgress: Bool = false

    // L2 disk cache flags — parity with Python `vmlx_engine serve
    // --enable-disk-cache --disk-cache-dir --disk-cache-max-gb`.
    // Persists prompt KV caches to SSD so they survive server restarts.
    // Acts as an L2 cache: on L1 (paged in-memory) miss, checks disk
    // before recomputing. Safe for all model types including hybrid
    // SSM (companion state is stored alongside in the same coordinator).
    @Flag(name: .long, help: "Persist prompt KV caches to disk (L2). Survives restarts.")
    var enableDiskCache: Bool = false
    @Option(name: .long, help: "Directory for L2 disk cache. Default: ~/Library/Application Support/vMLX/disk_cache")
    var diskCacheDir: String?
    @Option(name: .long, help: "Maximum L2 disk cache size in GB (default 10).")
    var diskCacheMaxGb: Double = 10.0

    // Reasoning + tool parser overrides — explicit names from
    // ParserRegistry (qwen3, deepseek_r1, mistral, gemma4, gpt_oss, auto)
    // for reasoning; (hermes, qwen, llama, mistral, deepseek, kimi,
    // granite, nemotron, step3p5, xlam, functionary, glm47, minimax,
    // gemma4, native) for tools. Empty string = let CapabilityDetector
    // pick from the model. Mirrors `cli.py --reasoning-parser` /
    // `--tool-call-parser`.
    @Option(name: .long, help: "Override reasoning parser. Empty = auto-detect from model. Names: qwen3, deepseek_r1, mistral, gemma4, gpt_oss, auto.")
    var reasoningParser: String = ""
    @Option(name: .long, help: "Override tool-call parser. Empty = auto-detect from model. Names: hermes, qwen, llama, mistral, deepseek, kimi, granite, nemotron, step3p5, xlam, functionary, glm47, minimax, gemma4, native.")
    var toolCallParser: String = ""

    // Default sampling overrides (server-wide). Each one sets the
    // corresponding GlobalSettings field; per-request fields still
    // override these via the 4-tier resolver. Mirrors `cli.py`.
    @Option(name: .long, help: "Default temperature (0..2). Per-request overrides win.")
    var defaultTemperature: Double?
    @Option(name: .long, help: "Default top-p (0..1).")
    var defaultTopP: Double?
    @Option(name: .long, help: "Default repetition penalty.")
    var defaultRepetitionPenalty: Double?
    @Option(name: .long, help: "Default enable_thinking (true|false). Per-request overrides win.")
    var defaultEnableThinking: Bool?

    // Chat template overrides — mirror `cli.py --chat-template`.
    // `--chat-template` accepts either a path to a .jinja file OR an
    // inline template string (auto-detected by checking for file
    // existence). `--chat-template-kwargs` is a JSON blob threaded into
    // every render via `UserInput.additionalContext`.
    @Option(name: .long, help: "Path to a .jinja chat template OR an inline template string.")
    var chatTemplate: String?
    @Option(name: .long, help: "JSON blob of default chat_template_kwargs threaded into every render.")
    var chatTemplateKwargs: String?

    // Networking — rate limit + TLS. Match `cli.py --rate-limit` and
    // `--ssl-keyfile`/`--ssl-certfile`.
    @Option(name: .long, help: "Per-IP rate limit (requests/min). 0 disables.")
    var rateLimit: Int = 0
    @Option(name: .long, help: "PEM TLS key file. Set together with --ssl-certfile to enable HTTPS.")
    var sslKeyfile: String?
    @Option(name: .long, help: "PEM TLS cert file. Set together with --ssl-keyfile to enable HTTPS.")
    var sslCertfile: String?

    func run() async throws {
        let engine = Engine()

        // Override GlobalSettings with any CLI cache flags the user passed.
        // These flow through `Engine.load` → `CacheCoordinatorConfig` →
        // `DiskCache(cacheDir:maxSizeGB:modelKey:)`, same path the SwiftUI
        // app settings use. Persisting the override means the next
        // `resolved()` call in Stream.swift picks it up automatically.
        // Apply CLI-supplied GlobalSettings overrides BEFORE loading so the
        // parser/template choices are visible during model load (caps
        // detection consults `defaultReasoningParser`/`defaultToolParser`
        // when set). Each flag is opt-in: nil/empty leaves the existing
        // value alone.
        do {
            var g = await engine.settings.global()
            var dirty = false
            if !reasoningParser.isEmpty { g.defaultReasoningParser = reasoningParser; dirty = true }
            if !toolCallParser.isEmpty { g.defaultToolParser = toolCallParser; dirty = true }
            if let t = defaultTemperature { g.defaultTemperature = t; dirty = true }
            if let p = defaultTopP { g.defaultTopP = p; dirty = true }
            if let r = defaultRepetitionPenalty { g.defaultRepetitionPenalty = r; dirty = true }
            if let e = defaultEnableThinking { g.defaultEnableThinking = e; dirty = true }
            if let tpl = chatTemplate, !tpl.isEmpty {
                // Auto-detect: if the value names an existing file, read it
                // in as the template body. Otherwise treat it as inline.
                let fm = FileManager.default
                if fm.fileExists(atPath: tpl),
                   let body = try? String(contentsOfFile: tpl, encoding: .utf8)
                {
                    g.chatTemplate = body
                } else {
                    g.chatTemplate = tpl
                }
                dirty = true
            }
            if let kw = chatTemplateKwargs, !kw.isEmpty {
                g.chatTemplateKwargs = kw
                dirty = true
            }
            if rateLimit > 0 { g.rateLimit = rateLimit; dirty = true }
            if let key = sslKeyfile, !key.isEmpty { g.sslKeyFile = key; dirty = true }
            if let cert = sslCertfile, !cert.isEmpty { g.sslCertFile = cert; dirty = true }
            if dirty { await engine.settings.setGlobal(g) }
        }

        if enableDiskCache {
            var g = await engine.settings.global()
            g.enableDiskCache = true
            if let dir = diskCacheDir, !dir.isEmpty {
                g.diskCacheDir = dir
            } else if g.diskCacheDir.isEmpty {
                // Default location parallels Python's
                // `~/.cache/vmlx/disk_cache` but on macOS we use the
                // platform-appropriate Application Support dir.
                let appSupport = FileManager.default.urls(
                    for: .applicationSupportDirectory, in: .userDomainMask
                ).first?.appendingPathComponent("vMLX/disk_cache")
                g.diskCacheDir = appSupport?.path
                    ?? (NSString(string: "~/Library/Application Support/vMLX/disk_cache")
                        .expandingTildeInPath)
            }
            g.diskCacheMaxGB = diskCacheMaxGb
            await engine.settings.setGlobal(g)
            FileHandle.standardError.write(Data(
                "[cli] L2 disk cache ON at \(g.diskCacheDir) (max \(diskCacheMaxGb) GB)\n".utf8))
        }

        // Drain the AsyncThrowingStream to completion BEFORE starting the
        // HTTP listener. Prior bug: `try await engine.load(...)` returned
        // immediately because `load` returns a stream handle, not an async
        // function — the server came up before the model was ready and
        // the first request hit "no model loaded".
        //
        // LoadOptions must pull from the (possibly-overridden) GlobalSettings
        // so the CLI `--enable-disk-cache` flag actually flows into
        // `CacheCoordinatorConfig`. Default `LoadOptions(modelPath:)`
        // ignores GlobalSettings entirely, which is why the cache flag
        // was silently dropped before this change.
        let resolved = await engine.settings.resolved(sessionId: nil, chatId: nil)
        let loadOpts = Engine.LoadOptions(
            modelPath: URL(fileURLWithPath: model),
            from: resolved
        )
        let loadStream = await engine.load(loadOpts)
        do {
            for try await event in loadStream {
                switch event {
                case .progress(let p):
                    if jsonProgress {
                        let line = #"{"phase":"\#(p.phase.rawValue)","message":"\#(p.label)","fraction":\#(p.fraction)}"#
                        FileHandle.standardError.write(Data((line + "\n").utf8))
                    } else {
                        FileHandle.standardError.write(
                            Data("[load] \(p.phase.rawValue): \(p.label)\n".utf8)
                        )
                    }
                case .done:
                    break
                case .failed(let reason):
                    throw EngineError.modelNotFound(URL(fileURLWithPath: model))
                        .context("\(reason)")
                }
            }
        } catch {
            FileHandle.standardError.write(Data("[load] failed: \(error)\n".utf8))
            throw ExitCode.failure
        }

        // Install a graceful shutdown handler for SIGTERM + SIGINT.
        //
        // Without this, a `docker stop`, `systemctl restart`, or a
        // Ctrl-C at the terminal kills the process mid-request and
        // any in-flight settings writes queued on the 500ms
        // SettingsStore debounce window are lost. The handler flushes
        // pending writes + calls `engine.stop()` (which cancels any
        // active stream task and releases the model), then exits
        // cleanly. Bounded to 2s so a wedged store can't block
        // termination forever — matches the vMLXApp `willTerminate`
        // timeout pattern.
        //
        // We use `DispatchSourceSignal` rather than the legacy
        // `signal()` callback because that API is unsafe from Swift
        // concurrency contexts — DispatchSource fires its handler on
        // a queue we own and plays nicely with the async runtime.
        // Apple's Dispatch sources require the matching
        // `signal(..., SIG_IGN)` first so the default action
        // (process exit) doesn't race the source handler.
        let shutdownQueue = DispatchQueue(label: "vmlx.shutdown")
        let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: shutdownQueue)
        let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: shutdownQueue)
        signal(SIGTERM, SIG_IGN)
        signal(SIGINT, SIG_IGN)
        nonisolated(unsafe) let engineCapture = engine
        let shutdownHandler: @Sendable () -> Void = {
            FileHandle.standardError.write(Data(
                "[vmlx] graceful shutdown: flushing settings...\n".utf8))
            let sem = DispatchSemaphore(value: 0)
            Task.detached {
                await engineCapture.settings.flushPending()
                await engineCapture.stop()
                sem.signal()
            }
            _ = sem.wait(timeout: .now() + .seconds(2))
            Darwin.exit(0)
        }
        sigterm.setEventHandler(handler: shutdownHandler)
        sigint.setEventHandler(handler: shutdownHandler)
        sigterm.resume()
        sigint.resume()

        let resolvedSettings = await engine.settings.global()
        let scheme = (!resolvedSettings.sslKeyFile.isEmpty && !resolvedSettings.sslCertFile.isEmpty)
            ? "https" : "http"
        let server = Server(
            engine: engine, host: host, port: port, apiKey: apiKey,
            tlsKeyPath: resolvedSettings.sslKeyFile.isEmpty ? nil : resolvedSettings.sslKeyFile,
            tlsCertPath: resolvedSettings.sslCertFile.isEmpty ? nil : resolvedSettings.sslCertFile,
            rateLimitPerMinute: resolvedSettings.rateLimit
        )
        print("vmlx serving \(model) at \(scheme)://\(host):\(port)")
        try await server.run()
    }
}

/// Thin REPL. Reads a line from stdin, streams the response, repeats.
/// No chat history on disk — just in-memory multi-turn for one session.
struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Interactive chat (REPL)")

    @Option(name: .shortAndLong) var model: String
    @Option(name: .long) var system: String?

    func run() async throws {
        let engine = Engine()
        print("Loading \(model)…")
        let loadStream = await engine.load(.init(modelPath: URL(fileURLWithPath: model)))
        for try await event in loadStream {
            if case .failed(let reason) = event {
                FileHandle.standardError.write(Data("[load] failed: \(reason)\n".utf8))
                throw ExitCode.failure
            }
        }
        print("Ready. Type a message, or /quit to exit.")

        var messages: [ChatRequest.Message] = []
        if let sys = system {
            messages.append(ChatRequest.Message(role: "system", content: .string(sys)))
        }

        while let line = readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            if trimmed == "/quit" || trimmed == "/exit" { break }

            messages.append(ChatRequest.Message(role: "user", content: .string(trimmed)))
            let req = ChatRequest(model: model, messages: messages, stream: true)
            var assistant = ""
            let stream = await engine.stream(request: req)
            do {
                for try await chunk in stream {
                    if let delta = chunk.content {
                        print(delta, terminator: "")
                        fflush(stdout)
                        assistant += delta
                    }
                }
                print("")
                messages.append(
                    ChatRequest.Message(role: "assistant", content: .string(assistant))
                )
            } catch {
                print("\n[error] \(error)")
            }
        }
    }
}

struct Pull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Download a model from HuggingFace")
    @Argument var repo: String

    func run() async throws {
        // Use the standalone DownloadManager actor (same one the SwiftUI
        // app drives via its Downloads window). Subscribe to events and
        // block until we see `.completed` or `.failed` for our job.
        let manager = DownloadManager()
        let displayName = repo.split(separator: "/").last.map(String.init) ?? repo
        let jobId = await manager.enqueue(repo: repo, displayName: displayName)

        print("Downloading \(repo)…")
        let events = await manager.subscribe()
        for await event in events {
            switch event {
            case .progress(let job) where job.id == jobId:
                let pct = job.totalBytes > 0
                    ? Double(job.receivedBytes) / Double(job.totalBytes) * 100
                    : 0
                print(String(format: "  %.1f%% (%.1f MB/s)",
                    pct, job.bytesPerSecond / 1e6),
                      terminator: "\r")
                fflush(stdout)
            case .completed(let job) where job.id == jobId:
                print("\nDone: \(job.localPath?.path ?? "(unknown path)")")
                return
            case .failed(let id, let reason) where id == jobId:
                print("\nFailed: \(reason)")
                throw ExitCode.failure
            case .cancelled(let id) where id == jobId:
                print("\nCancelled.")
                throw ExitCode.failure
            default:
                break
            }
        }
    }
}

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ls",
        abstract: "List downloaded models"
    )

    func run() async throws {
        let engine = Engine()
        _ = await engine.modelLibrary.scan(force: false)
        let entries = await engine.modelLibrary.entries()
        if entries.isEmpty {
            print("No models found. Downloads live under ~/.cache/huggingface/hub or user-configured directories.")
            return
        }
        // Column widths picked to accommodate the longest `org/repo-family`.
        print(String(format: "%-50s %-10s %-8s %12s  %s",
            "NAME", "FAMILY", "QUANT", "SIZE", "PATH"))
        for e in entries {
            let quant = e.quantBits.map { "Q\($0)" } ?? "fp16"
            let size = formatBytes(e.totalSizeBytes)
            print(String(format: "%-50s %-10s %-8s %12s  %s",
                e.displayName, e.family, quant, size, e.canonicalPath.path))
        }
    }

    private func formatBytes(_ b: Int64) -> String {
        let gb = Double(b) / 1e9
        if gb >= 1 { return String(format: "%.2f GB", gb) }
        let mb = Double(b) / 1e6
        return String(format: "%.1f MB", mb)
    }
}

// MARK: - EngineError context helper

extension EngineError {
    /// Attach a message to an existing case by wrapping in `unsupportedModelType`.
    /// Only used by the CLI for friendlier failure strings.
    fileprivate func context(_ s: String) -> EngineError {
        .unsupportedModelType("\(self): \(s)")
    }
}

// MARK: - dflash-smoke subcommand
//
// End-to-end JANG-DFlash + DDTree pipeline smoke. Loads a MiniMax
// target via the standard Engine path, loads a JangDFlashDrafter
// checkpoint (or constructs a random-init one for a dry-run), and
// runs ONE spec-dec cycle:
//
//     target forward (prompt, tap capture)
//       → bonus token + per-layer hidden taps
//       → drafter 1-step denoising forward
//       → softmax → top-k per slot
//       → lattice beam → prefix trie → ancestry mask
//       → target verify forward (tree attention mask)
//       → greedy walker → accepted token sequence
//
// Purpose: validate that the pipeline runs end-to-end on real MLX
// weights before any trained drafter exists. With a random-init
// drafter, expected behavior is low-to-zero acceptance rate and
// coherent walker output — the point is to prove the plumbing.

struct DFlashSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dflash-smoke",
        abstract: "End-to-end JANG-DFlash + DDTree pipeline smoke test"
    )

    @Option(name: .shortAndLong, help: "Path to the target model (must be MiniMax-JANG-*).")
    var model: String

    @Option(
        name: .long,
        help: "Path to a JangDFlashDrafter safetensors checkpoint. If omitted, uses a randomly-initialized drafter (pipeline-only smoke).")
    var drafter: String?

    @Option(name: .long, help: "Prompt text to seed the pipeline.")
    var prompt: String = "The Roman Empire reached its greatest territorial extent"

    @Option(name: .long, help: "Spec-dec block size B (DFlash default 16).")
    var blockSize: Int = 16

    @Option(name: .long, help: "Top-K per slot for DDTree expansion.")
    var topK: Int = 4

    @Option(name: .long, help: "Max paths kept after lattice beam.")
    var numPaths: Int = 60

    @Option(
        name: .long,
        help: "Comma-separated target layer indices to tap. Default: 5 evenly spaced across 62 MiniMax layers.")
    var tapLayersOpt: String = "10,22,34,46,58"

    @Option(name: .long, help: "Max new tokens to generate (multi-block loop).")
    var maxNewTokens: Int = 16

    @Flag(name: .long, help: "Use the v2 cached-KV generate path (persistent target KV cache + tap accumulation).")
    var cached: Bool = false

    @Flag(name: .long, help: "Interactive loop: read prompts from stdin line-by-line, model stays loaded between prompts.")
    var loop: Bool = false

    func run() async throws {
        let tapIdx = Set(tapLayersOpt.split(separator: ",").compactMap { Int($0) })
        guard !tapIdx.isEmpty else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] no valid tap layers parsed from \(tapLayersOpt)\n".utf8))
            throw ExitCode.failure
        }

        // Load target via the standard engine path.
        let engine = Engine()
        let loadOpts = Engine.LoadOptions(modelPath: URL(fileURLWithPath: model))
        let loadStream = await engine.load(loadOpts)
        for try await ev in loadStream {
            switch ev {
            case .progress(let p):
                FileHandle.standardError.write(Data(
                    "[load] \(p.phase.rawValue): \(p.label)\n".utf8))
            case .done: break
            case .failed(let reason):
                FileHandle.standardError.write(Data(
                    "[load] failed: \(reason)\n".utf8))
                throw ExitCode.failure
            }
        }

        guard let container = await engine.loaded else {
            FileHandle.standardError.write(Data("[dflash-smoke] no loaded container\n".utf8))
            throw ExitCode.failure
        }

        let drafterPath = drafter.map { URL(fileURLWithPath: $0) }
        let B = blockSize
        let k = topK
        let m = numPaths
        let maxN = maxNewTokens
        let useCached = cached

        if loop {
            // Interactive loop: read prompts from stdin, one per line.
            // Model stays loaded between prompts. Empty line skips;
            // `:q` or EOF exits. Prints the response followed by a
            // separator line to make the output easy to consume from
            // pipes and test harnesses.
            FileHandle.standardError.write(Data(
                "[dflash-smoke] loop mode — enter prompts, one per line, ':q' or Ctrl-D to exit\n".utf8))
            while let line = readLine() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed == ":q" || trimmed == ":quit" || trimmed == ":exit" {
                    break
                }
                if trimmed.isEmpty { continue }
                let start = Date()
                let result: DFlashSmokeImpl.GenerateResult
                do {
                    result = try await container.perform { (ctx: ModelContext) in
                        try DFlashSmokeImpl.runGenerate(
                            ctx: ctx,
                            drafterPath: drafterPath,
                            prompt: trimmed,
                            blockSize: B,
                            topK: k,
                            numPaths: m,
                            tapLayers: tapIdx,
                            maxNewTokens: maxN,
                            cached: useCached
                        )
                    }
                } catch {
                    FileHandle.standardError.write(Data(
                        "[dflash-smoke] generation failed: \(error)\n".utf8))
                    print("---")
                    continue
                }
                let wall = Date().timeIntervalSince(start)
                print(result.decodedText)
                let effRate = Double(result.generatedTokens.count) / max(wall, 1e-6)
                let nTok = result.generatedTokens.count
                let nBlk = result.blockOutcomes.count
                let accStr = String(format: "%.2f", result.meanAcceptedPerBlock)
                let rateStr = String(format: "%.2f", effRate)
                let statsLine = "[dflash] \(nTok) tok, \(nBlk) blocks, \(accStr) acc/blk, \(rateStr) tok/s\n"
                FileHandle.standardError.write(Data(statsLine.utf8))
                print("---")
            }
            return
        }

        // Single-shot mode (default) — matches session-1 output format.
        let start = Date()
        let promptCopy = prompt
        let result = try await container.perform { (ctx: ModelContext) in
            try DFlashSmokeImpl.runGenerate(
                ctx: ctx,
                drafterPath: drafterPath,
                prompt: promptCopy,
                blockSize: B,
                topK: k,
                numPaths: m,
                tapLayers: tapIdx,
                maxNewTokens: maxN,
                cached: useCached
            )
        }
        let wall = Date().timeIntervalSince(start)

        print("")
        print("=== DFlash generate ===")
        print("  prompt:          \"\(prompt)\"")
        print("  block size B:    \(blockSize)")
        print("  top-k per slot:  \(topK)")
        print("  m paths:         \(numPaths)")
        print("  tap layers:      \(tapIdx.sorted())")
        print("  drafter:         \(drafter ?? "(random init)")")
        print("  max new tokens:  \(maxNewTokens)")
        print("  blocks run:      \(result.blockOutcomes.count)")
        print("  generated:       \(result.generatedTokens.count) tokens")
        print("  mean accept/blk: \(String(format: "%.2f", result.meanAcceptedPerBlock))")
        print("  target wall sum: \(String(format: "%.3fs", result.totalTargetWallSec))")
        print("  drafter wall sum:\(String(format: "%.3fs", result.totalDrafterWallSec))")
        print("  verify wall sum: \(String(format: "%.3fs", result.totalVerifyWallSec))")
        print("  total wall:      \(String(format: "%.3fs", wall))")
        print("  eff tok/s:       \(String(format: "%.2f", Double(result.generatedTokens.count) / max(wall, 1e-6)))")
        print("")
        print("=== Text ===")
        print(result.decodedText)
        print("")
        print("  pipeline:        OK")
    }
}

/// Non-Sendable impl lives in an enum so it can return a plain Sendable
/// struct across the container actor boundary.
enum DFlashSmokeImpl {
    struct GenerateResult: Sendable {
        var generatedTokens: [Int]
        var decodedText: String
        var blockOutcomes: [JangDFlashBlockOutcome]
        var totalTargetWallSec: Double
        var totalDrafterWallSec: Double
        var totalVerifyWallSec: Double
        var meanAcceptedPerBlock: Double
    }

    static func runGenerate(
        ctx: ModelContext,
        drafterPath: URL?,
        prompt: String,
        blockSize: Int,
        topK: Int,
        numPaths: Int,
        tapLayers: Set<Int>,
        maxNewTokens: Int,
        cached: Bool
    ) throws -> GenerateResult {
        guard let minimax = ctx.model as? MiniMaxModel else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] loaded model is not MiniMaxModel (got \(type(of: ctx.model)))\n".utf8))
            throw ExitCode.validationFailure
        }
        let target = MiniMaxDFlashTarget(minimax)

        let drafterCfg = JangDFlashConfig(blockSize: blockSize)
        let drafter: JangDFlashDrafter
        if let drafterPath {
            drafter = try JangDFlashLoader.loadNew(config: drafterCfg, from: drafterPath)
            FileHandle.standardError.write(Data(
                "[dflash-smoke] loaded drafter from \(drafterPath.path)\n".utf8))
        } else {
            drafter = JangDFlashDrafter(drafterCfg)
            FileHandle.standardError.write(Data(
                "[dflash-smoke] using random-init drafter (pipeline smoke only)\n".utf8))
        }

        var specCfg = JangDFlashSpecConfig()
        specCfg.blockSize = blockSize
        specCfg.topK = topK
        specCfg.numPaths = numPaths
        specCfg.tapLayers = tapLayers
        let specDec = JangDFlashSpecDec(target: target, drafter: drafter, cfg: specCfg)

        let promptIDs = ctx.tokenizer.encode(text: prompt)
        guard !promptIDs.isEmpty else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] tokenizer returned empty token list\n".utf8))
            throw ExitCode.failure
        }

        let eosSet: Set<Int>
        if let eosId = ctx.tokenizer.eosTokenId {
            eosSet = [eosId]
        } else {
            eosSet = []
        }

        var outcomes: [JangDFlashBlockOutcome] = []
        let onBlockCallback: (JangDFlashBlockOutcome) -> Void = { outcome in
            outcomes.append(outcome)
            let acc = outcome.acceptedTokens.count
            let tree = outcome.treeSize
            let targetS = String(format: "%.3fs", outcome.targetWallSec)
            let draftS = String(format: "%.3fs", outcome.drafterWallSec)
            let verifyS = String(format: "%.3fs", outcome.verifyWallSec)
            let line = "[dflash] block \(outcomes.count): accepted=\(acc) tree=\(tree) "
                + "target=\(targetS) draft=\(draftS) verify=\(verifyS)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        let generated: [Int]
        if cached {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] using cached-KV generate path\n".utf8))
            generated = try specDec.cachedGenerate(
                promptIDs: promptIDs,
                maxNewTokens: maxNewTokens,
                eosTokenIDs: eosSet,
                onBlock: onBlockCallback
            )
        } else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] using v1 cacheless generate path\n".utf8))
            generated = try specDec.generate(
                promptIDs: promptIDs,
                maxNewTokens: maxNewTokens,
                eosTokenIDs: eosSet,
                onBlock: onBlockCallback
            )
        }

        let decoded = ctx.tokenizer.decode(tokenIds: generated)

        let sumT = outcomes.reduce(0.0) { $0 + $1.targetWallSec }
        let sumD = outcomes.reduce(0.0) { $0 + $1.drafterWallSec }
        let sumV = outcomes.reduce(0.0) { $0 + $1.verifyWallSec }
        let meanAcc: Double = outcomes.isEmpty ? 0 :
            Double(outcomes.reduce(0) { $0 + $1.acceptedTokens.count }) / Double(outcomes.count)

        return GenerateResult(
            generatedTokens: generated,
            decodedText: decoded,
            blockOutcomes: outcomes,
            totalTargetWallSec: sumT,
            totalDrafterWallSec: sumD,
            totalVerifyWallSec: sumV,
            meanAcceptedPerBlock: meanAcc
        )
    }

    // Legacy single-cycle path preserved below for reference but no
    // longer called. Can be deleted once the multi-block loop lands
    // in production; kept here as a known-good shape-checker.
    struct CycleResult: Sendable {
        var targetWallSec: Double
        var drafterWallSec: Double
        var verifyWallSec: Double
        var treeSize: Int
        var acceptedTokens: [Int]
    }

    static func runOneCycle(
        ctx: ModelContext,
        drafterPath: URL?,
        prompt: String,
        blockSize: Int,
        topK: Int,
        numPaths: Int,
        tapLayers: Set<Int>
    ) throws -> CycleResult {
        guard let minimax = ctx.model as? MiniMaxModel else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] loaded model is not MiniMaxModel (got \(type(of: ctx.model)))\n".utf8))
            throw ExitCode.validationFailure
        }
        let target = MiniMaxDFlashTarget(minimax)

        let drafterCfg = JangDFlashConfig(blockSize: blockSize)

        let drafter: JangDFlashDrafter
        if let drafterPath {
            drafter = try JangDFlashLoader.loadNew(config: drafterCfg, from: drafterPath)
            FileHandle.standardError.write(Data(
                "[dflash-smoke] loaded drafter from \(drafterPath.path)\n".utf8))
        } else {
            drafter = JangDFlashDrafter(drafterCfg)
            FileHandle.standardError.write(Data(
                "[dflash-smoke] using random-init drafter (pipeline smoke only)\n".utf8))
        }

        var specCfg = JangDFlashSpecConfig()
        specCfg.blockSize = blockSize
        specCfg.topK = topK
        specCfg.numPaths = numPaths
        specCfg.tapLayers = tapLayers
        let specDec = JangDFlashSpecDec(target: target, drafter: drafter, cfg: specCfg)

        // Tokenize
        let promptIDs = ctx.tokenizer.encode(text: prompt)
        guard !promptIDs.isEmpty else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] tokenizer returned empty token list\n".utf8))
            throw ExitCode.failure
        }
        let inputArr = MLXArray(promptIDs.map { Int32($0) }).reshaped(1, promptIDs.count)

        // Target forward with tap capture
        let tTarget0 = Date()
        let (targetLogits, taps) = target.forwardWithTaps(
            inputs: inputArr,
            cache: nil,
            tapLayers: tapLayers,
            providedMask: nil
        )
        materialize(targetLogits)
        for (_, t) in taps { materialize(t) }
        let targetWall = Date().timeIntervalSince(tTarget0)

        // Bonus token
        let finalLogits = targetLogits[0, promptIDs.count - 1]
        let bonusArr = argMax(finalLogits, axis: -1)
        materialize(bonusArr)
        let bonusID = Int(bonusArr.item(Int32.self))

        // Block input
        var blockIDsArr = [Int32](repeating: Int32(drafterCfg.maskTokenId), count: blockSize)
        blockIDsArr[0] = Int32(bonusID)
        let block = MLXArray(blockIDsArr).reshaped(1, blockSize)

        // Tap concatenation → slice/pad to block length
        let hCtxFull = specDec.buildTapConcatenation(taps: taps)
        let Tctx = hCtxFull.dim(1)
        let sliceStart = max(0, Tctx - blockSize)
        let hCtxBlock = hCtxFull[0..., sliceStart ..< Tctx, 0...]
        let hCtxPadded: MLXArray
        if hCtxBlock.dim(1) < blockSize {
            let pad = MLXArray.zeros(
                [1, blockSize - hCtxBlock.dim(1), hCtxBlock.dim(2)],
                dtype: hCtxBlock.dtype
            )
            hCtxPadded = concatenated([pad, hCtxBlock], axis: 1)
        } else {
            hCtxPadded = hCtxBlock
        }

        // Drafter forward
        let tDraft0 = Date()
        let drafterLogits = drafter(block, hTaps: hCtxPadded)
        materialize(drafterLogits)
        let drafterWall = Date().timeIntervalSince(tDraft0)

        // Top-k per slot
        let drafterProbs = specDec.softmaxLastAxis(drafterLogits[0..., 1..., 0...])
        materialize(drafterProbs)
        let (vals, ids) = specDec.topKPerSlot(probs: drafterProbs, k: topK)

        // Beam + trie
        let paths = DDTreeBuilder.beamTopMLattice(vals: vals, ids: ids, m: numPaths)
        guard !paths.isEmpty else {
            FileHandle.standardError.write(Data(
                "[dflash-smoke] beam returned zero paths\n".utf8))
            throw ExitCode.failure
        }
        let flatTree = DDTreeBuilder.flatten(paths: paths)
        let n = flatTree.flatTokens.count

        // Build additive mask for verify
        let totalLen: Int = promptIDs.count + n
        let negInf: Float = -1e9
        var biasFlat = [Float](repeating: negInf, count: totalLen * totalLen)
        for i in 0 ..< promptIDs.count {
            for j in 0 ... i {
                biasFlat[i * totalLen + j] = 0
            }
        }
        for i in 0 ..< n {
            let treeRow = promptIDs.count + i
            for j in 0 ..< promptIDs.count {
                biasFlat[treeRow * totalLen + j] = 0
            }
            for j in 0 ..< n where flatTree.ancestryMask[i][j] {
                biasFlat[treeRow * totalLen + (promptIDs.count + j)] = 0
            }
        }
        let bias = MLXArray(biasFlat, [totalLen, totalLen])

        let verifyIds: [Int32] =
            promptIDs.map { Int32($0) } + flatTree.flatTokens.map { Int32($0) }
        let verifyInput = MLXArray(verifyIds).reshaped(1, totalLen)

        let tVerify0 = Date()
        let (verifyLogits, _) = target.forwardWithTaps(
            inputs: verifyInput,
            cache: nil,
            tapLayers: [],
            providedMask: .array(bias)
        )
        materialize(verifyLogits)
        let verifyWall = Date().timeIntervalSince(tVerify0)

        // Extract argmax at each tree node.
        let treeLogits = verifyLogits[0, promptIDs.count ..< totalLen]
        let treeArg = argMax(treeLogits, axis: -1)
        materialize(treeArg)
        let treeArgArr = treeArg.asArray(Int32.self).map { Int($0) }

        let accepted = JangDFlashSpecDec.walkAcceptGreedy(
            flatTokens: flatTree.flatTokens,
            ancestryMask: flatTree.ancestryMask,
            targetArgmax: treeArgArr,
            bonusToken: bonusID
        )

        return CycleResult(
            targetWallSec: targetWall,
            drafterWallSec: drafterWall,
            verifyWallSec: verifyWall,
            treeSize: n,
            acceptedTokens: accepted
        )
    }
}
