import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTLS
import NIOSSL
import vMLXEngine
import vMLXLMCommon

/// Multi-engine gateway server.
///
/// The default vMLX listener (`Server`) is bound to one Engine actor — fine
/// for a single-model setup but a pain when the user runs N sessions, each
/// on a different port, and wants a single base URL their SDK can talk to.
///
/// `GatewayServer` solves that by exposing one Hummingbird listener that
/// reads each request's `model` field and forwards to the matching session's
/// Engine. Routes that need fan-out (chat/completions, completions,
/// embeddings, ollama chat, ollama generate, anthropic messages) dispatch
/// per-request; routes that don't (`/v1/models`, `/health`, `/admin/*`)
/// either union across engines or pin to the default.
///
/// Per-session servers keep running unchanged. Gateway is purely additive —
/// flip it on in GlobalSettings and a new listener appears at the configured
/// gateway port without touching any existing session bind.
public struct GatewayServer {

    /// Resolves a request's `model` field to the Engine that should handle
    /// it. Returns nil if the model is unknown — the route handler turns
    /// that into a 404 with a helpful list of available models.
    public typealias EngineResolver = @Sendable (_ model: String) async -> Engine?

    /// Returns every engine currently registered with the gateway, used by
    /// `/v1/models` to enumerate all loaded models in one response and by
    /// the model-not-found 404 to suggest valid alternatives.
    public typealias EngineEnumerator = @Sendable () async -> [Engine]

    public let host: String
    public let port: Int
    public let apiKey: String?
    public let adminToken: String?
    public let logLevel: LogStore.Level
    public let resolver: EngineResolver
    public let enumerate: EngineEnumerator
    /// Default engine used by routes that aren't model-keyed (admin,
    /// metrics, mcp, image gen). The first registered session works.
    public let defaultEngine: Engine
    /// Allowed CORS origins — same contract as `Server.allowedOrigins`.
    /// iter-49 companion fix: gateway was also hardcoded to `.all`.
    public let allowedOrigins: [String]
    /// iter-135 §161: live-mutable credential box. See Server.swift.
    public let authTokens: AuthTokenBox

    public init(
        host: String = "127.0.0.1",
        port: Int = 8080,
        apiKey: String? = nil,
        adminToken: String? = nil,
        logLevel: LogStore.Level = .info,
        defaultEngine: Engine,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator,
        allowedOrigins: [String] = ["*"]
    ) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.adminToken = adminToken
        self.logLevel = logLevel
        self.defaultEngine = defaultEngine
        self.resolver = resolver
        self.enumerate = enumerate
        self.allowedOrigins = allowedOrigins
        self.authTokens = AuthTokenBox(apiKey: apiKey, adminToken: adminToken)
    }

    /// iter-135 §161: swap live credentials without restarting. Mirrors
    /// `Server.applyAuthCredentials`.
    public func applyAuthCredentials(apiKey: String?, adminToken: String?) {
        authTokens.update(apiKey: apiKey, adminToken: adminToken)
    }

    public func run() async throws {
        guard Engine.isPortFree(port) else {
            await defaultEngine.logs.append(
                .error, category: "gateway",
                "Cannot start gateway: port \(port) is already in use on \(host)"
            )
            throw EngineError.portInUse(port)
        }

        let router = Router()

        // §331 — allowlist gate before Hummingbird's CORSMiddleware.
        // Matches Server.swift — see that file for the detailed comment.
        if allowedOrigins.count >= 2,
           !allowedOrigins.contains("*"),
           !allowedOrigins.filter({ !$0.isEmpty }).isEmpty
        {
            router.add(middleware: CORSAllowlistMiddleware(
                allowedOrigins: allowedOrigins))
        }
        // iter-49: CORS now honors `allowedOrigins` (previously
        // hardcoded `.all`). See `Server.resolveAllowOrigin`.
        router.add(middleware: CORSMiddleware(
            allowOrigin: Server.resolveAllowOrigin(allowedOrigins),
            allowHeaders: [
                .accept, .authorization, .contentType, .origin, .userAgent,
                // iter-76 (§104): accept x-api-key (Anthropic SDK) +
                // x-admin-token for parity with per-session server.
                HTTPField.Name("x-api-key")!,
                HTTPField.Name("x-admin-token")!,
            ],
            allowMethods: [.get, .post, .put, .delete, .options, .head]
        ))
        // iter-135 §161: box-based middleware — see Server.swift note.
        router.add(middleware: BearerAuthMiddleware(tokens: authTokens))
        router.add(middleware: AdminAuthMiddleware(tokens: authTokens))
        router.add(middleware: RequestLoggerMiddleware(engine: defaultEngine, minLevel: logLevel))

        await defaultEngine.logs.append(
            .info, category: "gateway",
            "Starting gateway HTTP server on \(host):\(port)"
        )

        let resolver = self.resolver
        let enumerate = self.enumerate

        // GET /v1/models — union of every registered engine's library.
        router.get("/v1/models") { _, _ -> Response in
            let engines = await enumerate()
            // iter-104 §182: collect the set of currently-loaded model
            // paths across every engine so the union response can flag
            // which ids are live. Prior gateway code had no per-entry
            // loaded flag at all — identical response whether the model
            // was resident or not.
            var loadedPaths: Set<String> = []
            for engine in engines {
                if let lp = await engine.loadedModelPath {
                    loadedPaths.insert(lp.path)
                }
            }
            var seen = Set<String>()
            var data: [[String: Any]] = []
            for engine in engines {
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                for e in entries where !seen.contains(e.displayName) {
                    seen.insert(e.displayName)
                    let isLoaded = loadedPaths.contains(e.canonicalPath.path)
                    data.append([
                        "id": e.displayName,
                        "object": "model",
                        // iter-104 §182: entry's detectedAt, not request
                        // time. See OpenAIRoutes /v1/models §182 comment
                        // for the full rationale (prior `Int(Date().time...
                        // IntervalSince1970)` stamped every entry with
                        // the current request time).
                        "created": Int(e.detectedAt.timeIntervalSince1970),
                        // iter-104 §182: shared owned_by heuristic. The
                        // gateway previously used "jjang-ai" while
                        // OpenAIRoutes used "dealignai" for the same
                        // isJANG==true condition — clients hitting both
                        // endpoints saw inconsistent ownership. Now both
                        // call the shared helper which parses org from
                        // `org/repo` displayName.
                        "owned_by": OpenAIRoutes.deriveOwnedBy(e.displayName),
                        // iter-117 §143: redact per-model canonicalPath.
                        // Same rationale as OpenAIRoutes /v1/models.
                        "vmlx": [
                            "family": e.family,
                            "modality": e.modality.rawValue,
                            "size_bytes": e.totalSizeBytes,
                            "is_jang": e.isJANG,
                            "is_mxtq": e.isMXTQ,
                            "quant_bits": e.quantBits as Any,
                            "path": OpenAIRoutes.redactHomeDir(e.canonicalPath.path),
                            // iter-104 §182: loaded flag (gateway path).
                            "loaded": isLoaded,
                        ],
                    ])
                }
            }
            return Self.json([
                "object": "list",
                "data": data,
            ])
        }

        // GET /metrics — Prometheus-format metrics per registered engine.
        // Each engine gets its own `session="session_N"` label set so
        // scrapers can distinguish concurrent sessions inside one gateway.
        MetricsRoutes.registerGateway(on: router, enumerate: enumerate)

        // GET /health — gateway-level health check.
        //
        // iter-118 §144: /health is bearer-auth-exempt (§104) so LAN
        // monitors can probe liveness without the API key. Previously
        // the response included `models: [all displayNames]` — sourced
        // from `engine.modelLibrary.entries()` which enumerates EVERY
        // installed model (not just loaded). That made the endpoint a
        // free unauth model-library enumeration: any LAN peer could
        // scrape the entire list of what the user has on disk. Now we
        // return only the currently-loaded model name per session —
        // matches the per-session /health which only exposes the loaded
        // model (AdminRoutes.swift:41-50). Plus `loaded_models_count`
        // for liveness assertions that don't leak names either.
        router.get("/health") { _, _ -> Response in
            let engines = await enumerate()
            var loadedModels: [String] = []
            // Iter 143 — gateway /health parity with iter-140 per-session
            // /health: emit per-engine cache_summary + arch flags +
            // continuous_batching marker so monitors scraping the
            // gateway port see the same observability surface as a
            // direct hit on a session port. Per-session stays
            // authoritative for the full per-stat breakdown
            // (/v1/cache/stats); /health is the compact summary tier.
            var perSession: [[String: Any]] = []
            // Aggregates rolled across every engine. Lets a single
            // /health probe answer "what's our cluster cache hit rate".
            var aggHits = 0
            var aggMisses = 0
            var aggInflight = 0
            var aggWaiting = 0
            var aggDiskBytesUsed: Int64 = 0
            for engine in engines {
                let state = engine.state
                let stateStr: String = {
                    switch state {
                    case .stopped: return "stopped"
                    case .loading: return "loading"
                    case .running: return "running"
                    case .standby(.soft): return "soft_sleep"
                    case .standby(.deep): return "deep_sleep"
                    case .error: return "error"
                    }
                }()
                var entry: [String: Any] = [
                    "state": stateStr,
                ]
                if let lp = await engine.loadedModelPath {
                    // Mirror the per-session /health displayName
                    // resolution so gateway users see the same
                    // repo-shape name.
                    let entries = await engine.modelLibrary.entries()
                    let canonical = lp.resolvingSymlinksInPath().standardizedFileURL
                    let name = entries.first(where: {
                        $0.canonicalPath.standardizedFileURL == canonical
                    })?.displayName ?? lp.lastPathComponent
                    loadedModels.append(name)
                    entry["model"] = name
                }
                let lock = await engine.generationLock
                let inflight = await lock.isHeld ? 1 : 0
                let waiting = await lock.waitingCount
                entry["inflight"] = inflight
                entry["waiting"] = waiting
                aggInflight += inflight
                aggWaiting += waiting
                if state == .running, let stats = try? await engine.cacheStats() {
                    var summary: [String: Any] = [:]
                    if let paged = stats["paged"] as? [String: Any] {
                        var p: [String: Any] = [:]
                        if let enabled = paged["enabled"] { p["enabled"] = enabled }
                        if let hr = paged["hitRate"] { p["hit_rate"] = hr }
                        if let h = paged["hitCount"] {
                            p["hits"] = h
                            if let hi = h as? Int { aggHits += hi }
                        }
                        if let m = paged["missCount"] {
                            p["misses"] = m
                            if let mi = m as? Int { aggMisses += mi }
                        }
                        if let bs = paged["blockSize"] { p["block_size"] = bs }
                        if let bu = paged["blocksInUse"] { p["blocks_in_use"] = bu }
                        summary["paged"] = p
                    }
                    if let disk = stats["disk"] as? [String: Any] {
                        var d: [String: Any] = [:]
                        if let enabled = disk["enabled"] { d["enabled"] = enabled }
                        if let hr = disk["hitRate"] { d["hit_rate"] = hr }
                        if let bytes = disk["bytesUsed"] {
                            d["bytes_used"] = bytes
                            if let bi = bytes as? Int64 { aggDiskBytesUsed += bi }
                            else if let bi = bytes as? Int { aggDiskBytesUsed += Int64(bi) }
                        }
                        if let cap = disk["bytesCap"] { d["bytes_cap"] = cap }
                        summary["disk"] = d
                    }
                    if let ssm = stats["ssm"] as? [String: Any] {
                        summary["ssm"] = ssm
                    }
                    if let jp = stats["jangPress"] as? [String: Any] {
                        summary["jangpress"] = jp
                    }
                    if let arch = stats["architecture"] as? [String: Any] {
                        var a: [String: Any] = [:]
                        if let h = arch["hybridSSMActive"] { a["hybrid_ssm"] = h }
                        if let s = arch["slidingWindowActive"] { a["sliding_window"] = s }
                        if let t = arch["turboQuantActive"] { a["turbo_quant"] = t }
                        summary["architecture"] = a
                    }
                    if !summary.isEmpty { entry["cache_summary"] = summary }
                }
                perSession.append(entry)
            }
            // Aggregate hit rate (totals make a single dashboard tile
            // possible). Avoid divide-by-zero when no engine has yet
            // touched the paged tier.
            let totalAccesses = aggHits + aggMisses
            let aggHitRate: Double = totalAccesses > 0
                ? Double(aggHits) / Double(totalAccesses)
                : 0.0
            let aggregate: [String: Any] = [
                "paged": [
                    "hits": aggHits,
                    "misses": aggMisses,
                    "hit_rate": aggHitRate,
                ] as [String: Any],
                "disk": [
                    "bytes_used": aggDiskBytesUsed,
                ] as [String: Any],
                "inflight": aggInflight,
                "waiting": aggWaiting,
            ]
            // Iter 143 — gateway `continuous_batching` is true only
            // when ANY engine has built its BatchEngine instance.
            // Mirrors per-session honesty marker. Until callers
            // actually invoke `engine.batchEngine(...)`, we run
            // through GenerationLock-serial and report false.
            var anyBatchEngineLive = false
            for e in engines {
                if (await e.batchEngineInstance) != nil {
                    anyBatchEngineLive = true
                    break
                }
            }
            return Self.json([
                "status": "ok",
                "engine": "vmlx-swift-gateway",
                "scheduling": anyBatchEngineLive ? "batch-engine" : "serial-fifo",
                "continuous_batching": anyBatchEngineLive,
                "sessions": engines.count,
                "loaded_models": loadedModels,
                "loaded_models_count": loadedModels.count,
                "engines": perSession,
                "aggregate": aggregate,
            ])
        }

        // POST /v1/chat/completions — the main fan-out. Reads model field,
        // resolves to the matching engine, forwards the full request.
        router.post("/v1/chat/completions") { req, ctx -> Response in
            return try await Self.handleOpenAIChat(
                req: req, ctx: ctx,
                resolver: resolver,
                enumerate: enumerate,
                defaultEngine: defaultEngine
            )
        }

        // POST /v1/completions — legacy text completion. Same dispatch.
        router.post("/v1/completions") { req, ctx -> Response in
            return try await Self.handleOpenAICompletions(
                req: req,
                resolver: resolver,
                enumerate: enumerate
            )
        }

        // POST /v1/embeddings — model-keyed but not streaming.
        router.post("/v1/embeddings") { req, ctx -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let model = obj["model"] as? String
            else {
                return Self.errorJSON(.badRequest, "Missing or invalid `model` field")
            }
            guard let engine = await resolver(model) else {
                return await Self.modelNotFound(model: model, enumerate: enumerate)
            }
            // iter-62: JIT wake — mirror §89 from per-session Server.
            // A soft-slept engine returned 503 notLoaded on gateway
            // embeddings requests, breaking RAG clients polling the
            // multi-model gateway port.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.embeddings(request: obj)
                return Self.json(result)
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // §363 — image generation/edit routes forward to defaultEngine.
        // Image gen is a resident singleton (FluxBackend) held by the
        // default Engine, so "gateway pins to default engine" is correct
        // for images — there's literally one backend shared across all
        // sessions. Previously the gateway returned 404 here and the user
        // had to figure out which per-session port the image model lived
        // on. Now OpenAI clients pointed at the gateway work out of the
        // box.
        router.post("/v1/images/generations") { req, ctx -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 64 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "Invalid JSON")
            }
            await defaultEngine.wakeFromStandby()
            do {
                let result = try await defaultEngine.generateImage(request: obj)
                return Self.json(result)
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }
        router.post("/v1/images/edits") { req, ctx -> Response in
            var req = req
            // JSON-with-base64 only on the gateway. The multipart/form-data
            // variant (OpenAI SDK default) requires file-field parsing that
            // lives in the per-session Server. Clients hitting the gateway
            // with multipart should use the per-session port for now, or
            // switch their SDK to base64 JSON which is the vMLX extension.
            let body = try await req.collectBody(upTo: 128 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "Gateway /v1/images/edits accepts JSON with base64-encoded image only. Use the per-session port for multipart uploads.")
            }
            await defaultEngine.wakeFromStandby()
            do {
                let result = try await defaultEngine.editImage(request: obj)
                return Self.json(result)
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // iter 138 — gateway protocol fan-out for Ollama, Anthropic, and
        // OpenAI Responses. Per user directive 2026-05-03: "make sure the
        // api ollama and anthropic and chat and respones all hook up to
        // gatwway port and is passed thru to the proper model".
        //
        // Each route reads the `model` field from the request body,
        // resolves to the matching engine via the shared resolver, then
        // delegates to the per-session handler logic (which already
        // exists in vMLXServer/Routes/{Anthropic,Ollama,OpenAI}Routes.swift).
        // The per-session implementations stay authoritative; the
        // gateway just dispatches.
        //
        // Streaming encoders (Anthropic SSE event family, Ollama NDJSON,
        // Responses event family) are reused via the shared SSEEncoder /
        // JSONLEncoder modules — each protocol's encoder is already
        // gateway-safe (no engine actor isolation required).

        // POST /v1/messages — Anthropic Messages API on gateway.
        router.post("/v1/messages") { req, _ -> Response in
            return try await Self.handleAnthropicMessages(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        // POST /api/chat — Ollama chat on gateway.
        router.post("/api/chat") { req, _ -> Response in
            return try await Self.handleOllamaChat(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        // POST /api/generate — Ollama text-completion on gateway.
        router.post("/api/generate") { req, _ -> Response in
            return try await Self.handleOllamaGenerate(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        // POST /v1/responses — OpenAI Responses API on gateway.
        router.post("/v1/responses") { req, _ -> Response in
            return try await Self.handleResponsesAPI(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        // Iter 143 — Ollama discovery routes on the gateway. These are
        // model-list / version probes that Ollama clients (Open WebUI,
        // GitHub Copilot, LangChain Ollama, ollama-js) hit BEFORE
        // /api/chat to populate model pickers and feature-gate. Without
        // them on the gateway, those clients see an empty list + silently
        // disable tool-calling. iter-138 added the actual generation
        // routes (/api/chat, /api/generate) but not these read-only
        // probes — the missing pieces.

        // GET /api/version — same Ollama version pin as the per-session
        // route. Static; no model resolution needed.
        router.get("/api/version") { _, _ -> Response in
            return Self.json(["version": "0.12.6"])
        }

        // GET /api/tags — union the model libraries across every
        // registered engine. The per-session route serves only its own
        // engine's library; the gateway aggregates so a single SDK base
        // URL sees every available model.
        router.get("/api/tags") { _, _ -> Response in
            let engines = await enumerate()
            // Walk every engine, scan + collect, dedupe by canonical
            // path so a model that appears in multiple engines'
            // libraries (sharing the same HF cache) is reported once.
            var seen = Set<String>()
            var models: [[String: Any]] = []
            for engine in engines {
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                for e in entries {
                    let key = e.canonicalPath.standardizedFileURL.path
                    if seen.contains(key) { continue }
                    seen.insert(key)
                    let iso = ISO8601DateFormatter().string(from: e.detectedAt)
                    models.append([
                        "name": e.displayName,
                        "model": e.displayName,
                        "modified_at": iso,
                        "size": e.totalSizeBytes,
                        "digest": e.id,
                        "details": [
                            "parent_model": "",
                            "format": e.isJANG ? "jang" : (e.isMXTQ ? "mxtq" : "mlx"),
                            "family": e.family,
                            "families": [e.family],
                            "parameter_size": OllamaRoutes.humanParamSize(
                                e.totalSizeBytes, quantBits: e.quantBits),
                            "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                        ] as [String: Any],
                    ])
                }
            }
            return Self.json(["models": models])
        }

        // GET /api/ps — every loaded model across every engine. SDK
        // clients hit this to know what's resident before deciding
        // whether to send a request that would trigger a load.
        router.get("/api/ps") { _, _ -> Response in
            let engines = await enumerate()
            var models: [[String: Any]] = []
            let iso = ISO8601DateFormatter().string(from: Date())
            for engine in engines {
                guard let path = await engine.loadedModelPath else { continue }
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                let canonical = path.resolvingSymlinksInPath().standardizedFileURL
                let matched = entries.first(where: {
                    $0.canonicalPath.standardizedFileURL == canonical
                })
                let displayName = matched?.displayName ?? path.lastPathComponent
                var model: [String: Any] = [
                    "name": displayName,
                    "model": displayName,
                    "expires_at": iso,
                    "size_vram": 0,
                ]
                if let e = matched {
                    model["size"] = e.totalSizeBytes
                    model["digest"] = e.id
                    model["details"] = [
                        "parent_model": "",
                        "format": e.isJANG ? "jang" : (e.isMXTQ ? "mxtq" : "mlx"),
                        "family": e.family,
                        "families": [e.family],
                        "parameter_size": OllamaRoutes.humanParamSize(
                            e.totalSizeBytes, quantBits: e.quantBits),
                        "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                    ] as [String: Any]
                } else {
                    model["size"] = 0
                    model["digest"] = ""
                    model["details"] = [:] as [String: Any]
                }
                models.append(model)
            }
            return Self.json(["models": models])
        }

        // POST /api/show — find a named model anywhere in the gateway's
        // engines. Body: `{"name": "<displayName>"}` or `{"model": ...}`.
        router.post("/api/show") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 1024 * 1024)
            let data = Data(buffer: body)
            let obj = (try? JSONSerialization.jsonObject(with: data)
                       as? [String: Any]) ?? [:]
            let name = (obj["name"] as? String)
                ?? (obj["model"] as? String)
                ?? ""
            if name.isEmpty {
                return Self.errorJSON(
                    .badRequest, "missing 'name' or 'model'")
            }
            let engines = await enumerate()
            for engine in engines {
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                guard let e = entries.first(where: { $0.displayName == name }) else {
                    continue
                }
                let caps = OllamaCapabilities.capabilities(
                    family: e.family, modality: e.modality)
                let modelfile = """
                    # vMLX model
                    FROM \(e.displayName)
                    # format: \(e.isJANG ? "jang" : "mlx")

                    """
                return Self.json([
                    "license": "",
                    "modelfile": modelfile,
                    "parameters": "",
                    "template": "",
                    "details": [
                        "parent_model": "",
                        "format": e.isJANG ? "jang" : "mlx",
                        "family": e.family,
                        "families": [e.family],
                        "parameter_size": OllamaRoutes.humanParamSize(
                            e.totalSizeBytes, quantBits: e.quantBits),
                        "quantization_level": e.quantBits.map { "Q\($0)" } ?? "",
                    ] as [String: Any],
                    "capabilities": caps,
                ])
            }
            return Self.errorJSON(.notFound, "model not found: \(name)")
        }

        // POST /api/embeddings + /api/embed — Ollama embedding routes
        // dispatched by `model` field. Both shapes (legacy `prompt`,
        // newer `input`) are translated to the OpenAI-compatible
        // request body the engine.embeddings path consumes.
        router.post("/api/embeddings") { req, _ -> Response in
            return try await Self.handleOllamaEmbed(
                req: req, resolver: resolver, enumerate: enumerate)
        }
        router.post("/api/embed") { req, _ -> Response in
            return try await Self.handleOllamaEmbed(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        // POST /v1/messages/count_tokens — Anthropic preflight on
        // gateway. Iter 143: real implementation; resolves engine by
        // body's `model` field then delegates to the per-session
        // helper which uses the loaded tokenizer.
        router.post("/v1/messages/count_tokens") { req, _ -> Response in
            return try await Self.handleAnthropicCountTokens(
                req: req, resolver: resolver, enumerate: enumerate)
        }

        router.get("/v1/_gateway/info") { _, _ -> Response in
            return Self.json([
                "supported": [
                    "GET  /v1/models",
                    "POST /v1/chat/completions",
                    "POST /v1/completions",
                    "POST /v1/embeddings",
                    "POST /v1/messages (Anthropic — iter 138)",
                    "POST /v1/messages/count_tokens (Anthropic preflight — iter 143)",
                    "POST /api/chat (Ollama — iter 138)",
                    "POST /api/generate (Ollama — iter 138)",
                    "GET  /api/version (Ollama — iter 143)",
                    "GET  /api/tags (Ollama discovery — iter 143)",
                    "GET  /api/ps (Ollama loaded-model probe — iter 143)",
                    "POST /api/show (Ollama model details — iter 143)",
                    "POST /api/embeddings (Ollama — iter 143)",
                    "POST /api/embed (Ollama — iter 143)",
                    "POST /v1/responses (OpenAI Responses — iter 138)",
                    "POST /v1/images/generations",
                    "POST /v1/images/edits (JSON + base64 only)",
                    "GET  /health",
                ],
                "unsupported_in_gateway": [
                    "POST /v1/images/edits (multipart/form-data) — JSON+base64 works here, multipart requires per-session port",
                    "/v1/audio/* — use the per-session port",
                    "/v1/mcp/* — per-session by definition (each session owns its own MCP subprocess pool)",
                    "/api/pull, /api/delete, /api/blobs/* — model-management routes are per-session (download/delete touches a specific engine's library)",
                ],
                "note": "Gateway dispatches by `model` field. List loaded models at /v1/models or /api/tags.",
            ])
        }

        // Build + run the listener. TLS handled by the calling layer if
        // the user wants gateway-level SSL — for v1 we ship plain HTTP and
        // assume the gateway sits behind a reverse proxy in production.
        let config = ApplicationConfiguration(
            address: .hostname(host, port: port),
            serverName: "vmlx-gateway"
        )
        let app = Application(router: router, configuration: config)
        try await app.runService()
    }

    // MARK: - Shared dispatch helpers

    /// Parse the `model` field, resolve the engine, then call the per-protocol
    /// handler. Returns an OpenAI-shape error response on misses so SDK
    /// clients see a consistent error envelope regardless of protocol.
    static func handleByModel<Context: RequestContext>(
        req: Request, ctx: Context,
        modelKey: String,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator,
        invoke: (Engine, Data) async throws -> Response
    ) async throws -> Response {
        var req = req
        let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj[modelKey] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `\(modelKey)` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        return try await invoke(engine, data)
    }

    /// /v1/chat/completions handler — same shape as
    /// OpenAIRoutes.register's body but the engine is per-request.
    static func handleOpenAIChat<Context: RequestContext>(
        req: Request, ctx: Context,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator,
        defaultEngine: Engine
    ) async throws -> Response {
        var req = req
        let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        var chatReq: ChatRequest
        do {
            chatReq = try JSONDecoder().decode(ChatRequest.self, from: data)
        } catch {
            return errorJSON(.badRequest, "invalid request: \(error)")
        }
        chatReq.applyMaxCompletionTokensAlias()
        do { try chatReq.validate() }
        catch let err as ChatRequestValidationError {
            return errorJSON(.badRequest, err.description)
        } catch {
            return errorJSON(.badRequest, "invalid request: \(error)")
        }

        guard let engine = await resolver(chatReq.model) else {
            return await modelNotFound(model: chatReq.model, enumerate: enumerate)
        }

        let isStream = chatReq.stream ?? false
        let id = "chatcmpl-\(UUID().uuidString.prefix(8).lowercased())"
        let created = Int(Date().timeIntervalSince1970)
        await engine.wakeFromStandby()

        if isStream {
            let stream = await engine.stream(request: chatReq, id: id)
            var headers: HTTPFields = [:]
            headers[.contentType] = "text/event-stream; charset=utf-8"
            headers[.cacheControl] = "no-cache"
            headers[.connection] = "keep-alive"
            return Response(
                status: .ok,
                headers: headers,
                body: SSEEncoder.chatCompletionStream(
                    id: id, model: chatReq.model, created: created,
                    includeUsage: chatReq.streamOptions?.includeUsage ?? false,
                    includeReasoning: chatReq.includeReasoning ?? true,
                    upstream: stream
                )
            )
        }

        var content = "", reasoning = ""
        var toolCalls: [ChatRequest.ToolCall] = []
        var seenToolCallKeys = Set<String>()
        var finishReason: String? = nil
        var usage: StreamChunk.Usage? = nil
        let stream = await engine.stream(request: chatReq, id: id)
        do {
            for try await chunk in stream {
                if let c = chunk.content { content += c }
                if let r = chunk.reasoning { reasoning += r }
                if let tcs = chunk.toolCalls {
                    ToolCallDeduper.appendUnique(
                        tcs, to: &toolCalls, seen: &seenToolCallKeys)
                }
                if let fr = chunk.finishReason { finishReason = fr }
                if let u = chunk.usage { usage = u }
            }
        } catch let err as EngineError {
            return OpenAIRoutes.mapEngineError(err)
        } catch {
            return errorJSON(.internalServerError, "\(error)")
        }
        var message: [String: Any] = ["role": "assistant", "content": content]
        if (chatReq.includeReasoning ?? true) && !reasoning.isEmpty {
            message["reasoning_content"] = reasoning
        }
        if !toolCalls.isEmpty {
            message["tool_calls"] = toolCalls.map { tc in
                [
                    "id": tc.id,
                    "type": "function",
                    "function": [
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    ] as [String: Any],
                ] as [String: Any]
            }
        }
        var obj: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": chatReq.model,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": finishReason ?? "stop",
            ] as [String: Any]],
        ]
        if let u = usage {
            // iter-130 §205: gateway non-stream chat/completions was
            // the last hold-out missing the timing-envelope hard rule.
            // Direct per-engine /v1/chat/completions (OpenAIRoutes:232)
            // has emitted tokens_per_second + ttft_ms + prefill_ms +
            // total_ms + cache_detail since iter-64 §118; iter-126
            // §201 added prompt_tokens_per_second. The multi-session
            // gateway fan-out handler mirrors the same shape but kept
            // emitting only the baseline token counts, so dashboards
            // that scraped the gateway (for cross-engine SLO rollups)
            // saw `tokens_per_second: undefined` while single-engine
            // /v1/chat/completions had full timings. Mirror the
            // OpenAIRoutes usage-envelope 1:1.
            var usageObj: [String: Any] = [
                "prompt_tokens": u.promptTokens,
                "completion_tokens": u.completionTokens,
                "total_tokens": u.promptTokens + u.completionTokens,
                "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
            ]
            if let tps = u.tokensPerSecond { usageObj["tokens_per_second"] = tps }
            if let pps = u.promptTokensPerSecond { usageObj["prompt_tokens_per_second"] = pps }
            if let ttft = u.ttftMs { usageObj["ttft_ms"] = ttft }
            if let prefill = u.prefillMs { usageObj["prefill_ms"] = prefill }
            if let total = u.totalMs { usageObj["total_ms"] = total }
            if let detail = u.cacheDetail { usageObj["cache_detail"] = detail }
            obj["usage"] = usageObj
        }
        return json(obj)
    }

    /// /v1/completions handler — legacy text-completion wire shape.
    /// Gateway must not reuse `handleOpenAIChat`: legacy clients send
    /// `prompt`, expect `choices[0].text`, and use the flat logprobs
    /// envelope. This mirrors the per-session OpenAIRoutes implementation
    /// while adding model-keyed engine resolution.
    static func handleOpenAICompletions(
        req: Request,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var req = req
        let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return errorJSON(.badRequest, "invalid JSON body")
        }
        let model = (obj["model"] as? String) ?? "default"
        let prompt: String
        if let s = obj["prompt"] as? String {
            prompt = s
        } else if let arr = obj["prompt"] as? [String] {
            prompt = arr.joined(separator: "\n")
        } else {
            return errorJSON(.badRequest, "missing 'prompt'")
        }

        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }

        let stopList: [String]? = {
            if let arr = obj["stop"] as? [String], !arr.isEmpty { return arr }
            if let s = obj["stop"] as? String, !s.isEmpty { return [s] }
            return nil
        }()
        var chatReq = ChatRequest(
            model: model,
            messages: [.init(role: "user", content: .string(prompt))],
            stream: obj["stream"] as? Bool,
            maxTokens: obj["max_tokens"] as? Int,
            temperature: obj["temperature"] as? Double,
            topP: obj["top_p"] as? Double,
            topK: obj["top_k"] as? Int,
            minP: obj["min_p"] as? Double,
            repetitionPenalty: obj["repetition_penalty"] as? Double,
            stop: stopList,
            seed: obj["seed"] as? Int
        )
        chatReq.frequencyPenalty = obj["frequency_penalty"] as? Double
        chatReq.presencePenalty = obj["presence_penalty"] as? Double
        if let so = obj["stream_options"] as? [String: Any],
           let include = so["include_usage"] as? Bool {
            chatReq.streamOptions = .init(includeUsage: include)
        }
        if let flag = obj["logprobs"] as? Bool {
            chatReq.logprobs = flag
        } else if let n = obj["logprobs"] as? Int {
            chatReq.logprobs = true
            chatReq.topLogprobs = n
        }
        if let n = obj["top_logprobs"] as? Int {
            chatReq.topLogprobs = n
        }
        do {
            try chatReq.validate()
        } catch let err as ChatRequestValidationError {
            return errorJSON(.badRequest, err.description)
        } catch {
            return errorJSON(.badRequest, "invalid request: \(error)")
        }

        await engine.wakeFromStandby()
        let id = "cmpl-\(UUID().uuidString.prefix(8).lowercased())"
        let created = Int(Date().timeIntervalSince1970)
        let isStream = chatReq.stream ?? false

        if isStream {
            let stream = await engine.stream(request: chatReq, id: id)
            var headers: HTTPFields = [:]
            headers[.contentType] = "text/event-stream; charset=utf-8"
            headers[.cacheControl] = "no-cache"
            headers[.connection] = "keep-alive"
            if let n = HTTPField.Name("x-vmlx-trace-id") {
                headers[n] = id
            }
            return Response(
                status: .ok,
                headers: headers,
                body: SSEEncoder.textCompletionStream(
                    id: id, model: model, created: created, upstream: stream
                )
            )
        }

        var content = ""
        var finishReason: String? = nil
        var usage: StreamChunk.Usage? = nil
        var allLogprobs: [TokenLogprob] = []
        let stream = await engine.stream(request: chatReq, id: id)
        do {
            for try await chunk in stream {
                if let c = chunk.content { content += c }
                if let fr = chunk.finishReason { finishReason = fr }
                if let u = chunk.usage { usage = u }
                if let lps = chunk.logprobs { allLogprobs.append(contentsOf: lps) }
            }
        } catch let err as EngineError {
            return OpenAIRoutes.mapEngineError(err)
        } catch {
            return errorJSON(.internalServerError, "\(error)")
        }

        if let fimFlag = obj["truncate_fim"] as? Bool, fimFlag {
            content = FIMTruncator.truncate(content)
        }

        var choice0: [String: Any] = [
            "text": content,
            "index": 0,
            "finish_reason": finishReason ?? "stop",
        ]
        if !allLogprobs.isEmpty {
            var tokens: [String] = []
            var tokenLogprobs: [Float] = []
            var textOffsets: [Int] = []
            var topLogprobsArr: [[String: Float]] = []
            var runningOffset = 0
            for lp in allLogprobs {
                tokens.append(lp.token)
                tokenLogprobs.append(lp.logprob)
                textOffsets.append(runningOffset)
                runningOffset += lp.token.utf8.count
                if lp.topLogprobs.isEmpty {
                    topLogprobsArr.append([:])
                } else {
                    var dict: [String: Float] = [:]
                    for alt in lp.topLogprobs { dict[alt.token] = alt.logprob }
                    topLogprobsArr.append(dict)
                }
            }
            choice0["logprobs"] = [
                "tokens": tokens,
                "token_logprobs": tokenLogprobs,
                "top_logprobs": topLogprobsArr,
                "text_offset": textOffsets,
            ] as [String: Any]
        }

        var out: [String: Any] = [
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [choice0],
        ]
        if let u = usage {
            var usageObj: [String: Any] = [
                "prompt_tokens": u.promptTokens,
                "completion_tokens": u.completionTokens,
                "total_tokens": u.promptTokens + u.completionTokens,
                "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
            ]
            if let tps = u.tokensPerSecond { usageObj["tokens_per_second"] = tps }
            if let pps = u.promptTokensPerSecond { usageObj["prompt_tokens_per_second"] = pps }
            if let ttft = u.ttftMs { usageObj["ttft_ms"] = ttft }
            if let prefill = u.prefillMs { usageObj["prefill_ms"] = prefill }
            if let total = u.totalMs { usageObj["total_ms"] = total }
            if let detail = u.cacheDetail { usageObj["cache_detail"] = detail }
            out["usage"] = usageObj
        }
        return json(out)
    }

    // MARK: - Iter 138 protocol fan-out (Anthropic / Ollama / Responses)

    /// Gateway dispatcher for POST /v1/messages — Anthropic Messages API.
    static func handleAnthropicMessages(
        req: Request, resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        // Sniff body for `model` then delegate. We collect the full body
        // here and re-wrap it because AnthropicRoutes.handleMessages
        // re-collects on its own — but we need the body once first to
        // resolve the engine. Use a 32 MB cap matching the inline route.
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        // Re-wrap body for the per-session helper. Hummingbird requests
        // are single-pass; the helper expects to call collectBody(),
        // so we synthesize a fresh Request with the buffered body.
        let buf = ByteBufferAllocator().buffer(data: data)
        let replay = Request(head: req.head, body: .init(buffer: buf))
        return try await AnthropicRoutes.handleMessages(req: replay, engine: engine)
    }

    /// Gateway dispatcher for POST /api/chat — Ollama chat.
    static func handleOllamaChat(
        req: Request, resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        let buf = ByteBufferAllocator().buffer(data: data)
        let replay = Request(head: req.head, body: .init(buffer: buf))
        return try await OllamaRoutes.handleChat(req: replay, engine: engine)
    }

    /// Gateway dispatcher for POST /api/generate — Ollama text completion.
    static func handleOllamaGenerate(
        req: Request, resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        let buf = ByteBufferAllocator().buffer(data: data)
        let replay = Request(head: req.head, body: .init(buffer: buf))
        return try await OllamaRoutes.handleGenerate(req: replay, engine: engine)
    }

    /// Gateway dispatcher for POST /v1/responses — OpenAI Responses API.
    /// Iter 139: fully wired now that OpenAIRoutes.handleResponses is
    /// extracted as a public static helper. Resolves engine by model
    /// then delegates.
    static func handleResponsesAPI(
        req: Request, resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        let buf = ByteBufferAllocator().buffer(data: data)
        let replay = Request(head: req.head, body: .init(buffer: buf))
        return try await OpenAIRoutes.handleResponses(req: replay, engine: engine)
    }

    /// Iter 143 — gateway dispatcher for POST /api/embeddings + /api/embed.
    /// Translates Ollama's `prompt` / `input` shape to the OpenAI body the
    /// engine.embeddings path consumes, then unwraps the result back into
    /// Ollama's response envelope. Mirrors the per-session
    /// `ollamaEmbedHandler` body in `OllamaRoutes.swift:591`.
    static func handleOllamaEmbed(
        req: Request,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 16 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            return errorJSON(.badRequest, "invalid JSON body")
        }
        let modelName = (obj["model"] as? String) ?? ""
        guard !modelName.isEmpty else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(modelName) else {
            return await modelNotFound(model: modelName, enumerate: enumerate)
        }
        var openAI: [String: Any] = ["model": modelName]
        if let prompt = obj["prompt"] {
            openAI["input"] = prompt
        } else if let input = obj["input"] {
            openAI["input"] = input
        } else {
            return errorJSON(.badRequest, "missing 'prompt' or 'input'")
        }
        await engine.wakeFromStandby()
        do {
            let result = try await engine.embeddings(request: openAI)
            let dataArr = (result["data"] as? [[String: Any]]) ?? []
            let vectors: [[Float]] = dataArr.compactMap {
                $0["embedding"] as? [Float]
            }
            let promptEvalCount: Int = {
                if let usage = result["usage"] as? [String: Any],
                   let tokens = usage["prompt_tokens"] as? Int
                { return tokens }
                return 0
            }()
            if vectors.count == 1 {
                return json([
                    "embedding": vectors[0],
                    "model": result["model"] ?? "",
                    "prompt_eval_count": promptEvalCount,
                    "total_duration": 0,
                    "load_duration": 0,
                ])
            }
            return json([
                "embeddings": vectors,
                "model": result["model"] ?? "",
                "prompt_eval_count": promptEvalCount,
                "total_duration": 0,
                "load_duration": 0,
            ])
        } catch let err as EngineError {
            return OpenAIRoutes.mapEngineError(err)
        } catch {
            return errorJSON(.internalServerError, "\(error)")
        }
    }

    /// Iter 143 — gateway dispatcher for POST /v1/messages/count_tokens.
    /// Resolves engine by `model` field then delegates to the per-session
    /// helper which uses the loaded tokenizer.
    static func handleAnthropicCountTokens(
        req: Request,
        resolver: @escaping EngineResolver,
        enumerate: @escaping EngineEnumerator
    ) async throws -> Response {
        var sniffReq = req
        let body = try await sniffReq.collectBody(upTo: 32 * 1024 * 1024)
        let data = Data(buffer: body)
        guard let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
              let model = obj["model"] as? String
        else {
            return errorJSON(.badRequest, "Missing or invalid `model` field")
        }
        guard let engine = await resolver(model) else {
            return await modelNotFound(model: model, enumerate: enumerate)
        }
        let buf = ByteBufferAllocator().buffer(data: data)
        let replay = Request(head: req.head, body: .init(buffer: buf))
        return try await AnthropicRoutes.handleCountTokens(req: replay, engine: engine)
    }

    static func modelNotFound(
        model: String,
        enumerate: @escaping EngineEnumerator
    ) async -> Response {
        let engines = await enumerate()
        var available: [String] = []
        for engine in engines {
            for entry in await engine.modelLibrary.entries() {
                available.append(entry.displayName)
            }
        }
        let body: [String: Any] = [
            "error": [
                "message": "Model `\(model)` is not loaded by any active session",
                "type": "not_found",
                "available_models": available,
            ] as [String: Any]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var headers: HTTPFields = [:]
        headers[.contentType] = "application/json"
        return Response(
            status: .notFound,
            headers: headers,
            body: .init(byteBuffer: .init(data: data))
        )
    }

    // MARK: - Helpers (mirror OpenAIRoutes for response shaping)

    static func json(_ obj: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
        var headers: HTTPFields = [:]
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: .init(byteBuffer: .init(data: data))
        )
    }

    static func errorJSON(_ status: HTTPResponse.Status, _ message: String) -> Response {
        let body: [String: Any] = [
            "error": [
                "message": message,
                "type": status == .badRequest ? "invalid_request_error" : "server_error",
            ] as [String: Any]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: body)) ?? Data()
        var headers: HTTPFields = [:]
        headers[.contentType] = "application/json"
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: .init(data: data))
        )
    }
}
