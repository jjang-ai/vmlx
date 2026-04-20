import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTLS
import NIOSSL
import vMLXEngine

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
        router.add(middleware: BearerAuthMiddleware(apiKey: apiKey))
        router.add(middleware: AdminAuthMiddleware(adminToken: adminToken))
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
            let created = Int(Date().timeIntervalSince1970)
            var seen = Set<String>()
            var data: [[String: Any]] = []
            for engine in engines {
                _ = await engine.modelLibrary.scan(force: false)
                let entries = await engine.modelLibrary.entries()
                for e in entries where !seen.contains(e.displayName) {
                    seen.insert(e.displayName)
                    data.append([
                        "id": e.displayName,
                        "object": "model",
                        "created": created,
                        "owned_by": e.isJANG ? "jjang-ai" : "mlx-community",
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

        // GET /health — gateway-level health check. Lists every routed model.
        router.get("/health") { _, _ -> Response in
            let engines = await enumerate()
            var models: [String] = []
            for engine in engines {
                let entries = await engine.modelLibrary.entries()
                models.append(contentsOf: entries.map(\.displayName))
            }
            return Self.json([
                "status": "ok",
                "engine": "vmlx-swift-gateway",
                "sessions": engines.count,
                "models": models,
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
            return try await Self.handleOpenAIChat(
                req: req, ctx: ctx,
                resolver: resolver,
                enumerate: enumerate,
                defaultEngine: defaultEngine
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

        // Ollama (`/api/chat`, `/api/generate`) and Anthropic (`/v1/messages`)
        // are NOT exposed on the gateway in v1. Each protocol has a non-
        // trivial encoder (NDJSON for Ollama, anthropic SSE event family
        // for messages) and gracefully sharing those across engines is a
        // separate refactor. Per-session ports still serve them correctly.
        // GET /unsupported-protocols returns the documented list so SDK
        // probing surfaces a useful 404 instead of a silent connection.
        router.get("/v1/_gateway/info") { _, _ -> Response in
            return Self.json([
                "supported": [
                    "GET  /v1/models",
                    "POST /v1/chat/completions",
                    "POST /v1/completions",
                    "POST /v1/embeddings",
                    "GET  /health",
                ],
                "unsupported_in_gateway": [
                    "POST /api/chat (Ollama) — use the per-session port",
                    "POST /api/generate (Ollama) — use the per-session port",
                    "POST /v1/messages (Anthropic) — use the per-session port",
                    "/v1/images/* — use the per-session port",
                    "/v1/audio/* — use the per-session port",
                    "/v1/mcp/* — use the per-session port",
                ],
                "note": "Gateway dispatches by `model` field. List loaded models at /v1/models.",
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
        var finishReason: String? = nil
        var usage: StreamChunk.Usage? = nil
        let stream = await engine.stream(request: chatReq, id: id)
        do {
            for try await chunk in stream {
                if let c = chunk.content { content += c }
                if let r = chunk.reasoning { reasoning += r }
                if let tcs = chunk.toolCalls { toolCalls.append(contentsOf: tcs) }
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
            obj["usage"] = [
                "prompt_tokens": u.promptTokens,
                "completion_tokens": u.completionTokens,
                "total_tokens": u.promptTokens + u.completionTokens,
                "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
            ] as [String: Any]
        }
        return json(obj)
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
