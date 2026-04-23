import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTLS
import NIOSSL
import os
import vMLXEngine

/// vMLX HTTP server. Hummingbird 2.x, sandbox-safe (no fork/exec, no Python sidecar).
///
/// Route parity with `vmlx_engine/server.py` — see `Routes/*.swift` for per-endpoint
/// mappings and Python line numbers.
public struct Server {

    public let engine: Engine
    public let host: String
    public let port: Int
    public let apiKey: String?
    public let adminToken: String?
    public let logLevel: LogStore.Level
    public let tlsKeyPath: String?
    public let tlsCertPath: String?
    public let rateLimitPerMinute: Int
    /// Allowed CORS origins. `["*"]` (default) → fully permissive
    /// `Access-Control-Allow-Origin: *`. A single entry other than
    /// `"*"` maps to Hummingbird's `.custom(origin)`. Two or more
    /// non-wildcard entries route through `CORSAllowlistMiddleware`
    /// (see §331) which enforces a strict allowlist on every request
    /// instead of echoing arbitrary origins back — the prior
    /// "TODO follow-up middleware" note is closed. Live-swap is wired
    /// via §152 so /admin/cors/update propagates without restart.
    public let allowedOrigins: [String]
    /// iter-135 §161: mutable auth credentials the middleware reads
    /// per-request. See `AuthTokenBox`. Callers swap values via
    /// `applyAuthCredentials(apiKey:adminToken:)` without tearing down
    /// the listener.
    public let authTokens: AuthTokenBox

    public init(
        engine: Engine,
        host: String = "127.0.0.1",
        port: Int = 8000,
        apiKey: String? = nil,
        adminToken: String? = nil,
        logLevel: LogStore.Level = .info,
        tlsKeyPath: String? = nil,
        tlsCertPath: String? = nil,
        rateLimitPerMinute: Int = 0,
        allowedOrigins: [String] = ["*"]
    ) {
        self.engine = engine
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.adminToken = adminToken
        self.logLevel = logLevel
        self.tlsKeyPath = tlsKeyPath
        self.tlsCertPath = tlsCertPath
        self.rateLimitPerMinute = rateLimitPerMinute
        self.allowedOrigins = allowedOrigins
        self.authTokens = AuthTokenBox(apiKey: apiKey, adminToken: adminToken)
    }

    /// iter-135 §161: swap the live credentials. Next incoming HTTP
    /// request will authenticate against the new values — no server
    /// restart needed. Used by `APIKeyManager.revoke` so the
    /// iter-96 §123 "immediately lose access" promise becomes true.
    public func applyAuthCredentials(apiKey: String?, adminToken: String?) {
        authTokens.update(apiKey: apiKey, adminToken: adminToken)
    }

    /// Translate the user's `corsOrigins` list into Hummingbird's
    /// `AllowOrigin` enum. See the `allowedOrigins` doc for the
    /// mapping contract.
    static func resolveAllowOrigin(
        _ origins: [String]
    ) -> CORSMiddleware<BasicRequestContext>.AllowOrigin {
        let nonEmpty = origins.filter { !$0.isEmpty }
        if nonEmpty.isEmpty { return .all }
        if nonEmpty == ["*"] { return .all }
        if nonEmpty.count == 1 { return .custom(nonEmpty[0]) }
        return .originBased
    }

    public func run() async throws {
        // Fixes mlxstudio #44: preflight bind check. Before this, two servers
        // on the same port (or a port already held by another process) would
        // let Hummingbird fatal-trap on bind failure. Surface a clean
        // `EngineError.portInUse` instead.
        guard Engine.isPortFree(port) else {
            await engine.logs.append(
                .error, category: "server",
                "Cannot start: port \(port) is already in use on \(host)"
            )
            throw EngineError.portInUse(port)
        }

        let router = Router()

        // §331 — allowlist gate BEFORE Hummingbird's CORSMiddleware.
        // For 2+-entry allowlists, resolveAllowOrigin maps to
        // `.originBased` which echoes ANY Origin header — no gating.
        // The gate strips the Origin header for disallowed origins so
        // the downstream CORSMiddleware skips Allow-Origin emission
        // entirely (matching the "CORS denied" browser behavior).
        // Preflight OPTIONS from disallowed origins gets a 403.
        if allowedOrigins.count >= 2,
           !allowedOrigins.contains("*"),
           !allowedOrigins.filter({ !$0.isEmpty }).isEmpty
        {
            router.add(middleware: CORSAllowlistMiddleware(
                allowedOrigins: allowedOrigins))
        }
        // Middleware: CORS — now honors `allowedOrigins` from the
        // session's cors_origins setting (mirrors Python fastapi
        // CORSMiddleware config). Pre-iter-49 this was hardcoded
        // `allowOrigin: .all`, so the cors_origins setting was a
        // dead UI field. See `resolveAllowOrigin` for the list→enum
        // mapping contract.
        router.add(middleware: CORSMiddleware(
            allowOrigin: Self.resolveAllowOrigin(allowedOrigins),
            allowHeaders: [
                .accept, .authorization, .contentType, .origin, .userAgent,
                // iter-76 (§104): x-api-key is the Anthropic SDK's
                // authentication header. Browser-hosted clients hitting
                // /v1/messages need this in the CORS allowlist so
                // BearerAuthMiddleware can read it.
                HTTPField.Name("x-api-key")!,
                // x-admin-token mirrors the admin-auth sibling header.
                HTTPField.Name("x-admin-token")!,
            ],
            allowMethods: [.get, .post, .put, .delete, .options, .head]
        ))
        // iter-135 §161: hand the middleware the shared AuthTokenBox
        // instead of a literal so credential swaps propagate to the
        // running server. Before §161 the literal was captured at init
        // and stale until the next server restart, making the
        // iter-96 §123 revoke-dialog "immediately lose access" promise
        // a lie.
        router.add(middleware: BearerAuthMiddleware(tokens: authTokens))
        router.add(middleware: AdminAuthMiddleware(tokens: authTokens))
        // Per-IP rate limit (no-op if rateLimitPerMinute == 0).
        if rateLimitPerMinute > 0 {
            router.add(middleware: RateLimitMiddleware(
                requestsPerMinute: rateLimitPerMinute))
            await engine.logs.append(.info, category: "server",
                "rate limit: \(rateLimitPerMinute) req/min/IP")
        }
        // Per-request access log into Engine.logs (category "server").
        router.add(middleware: RequestLoggerMiddleware(engine: engine, minLevel: logLevel))
        await engine.logs.append(
            .info, category: "server",
            "Starting HTTP server on \(host):\(port)"
        )

        // Kick off a background library scan so the first `/v1/models`
        // or `/api/tags` hit doesn't have to walk the HF cache inline.
        // The scan is idempotent inside the freshness window so the
        // route handlers' defensive `scan(force:false)` calls become
        // no-ops. Detached so server bind isn't gated on disk I/O.
        Task.detached { [engine] in
            _ = await engine.modelLibrary.scan(force: false)
        }

        // Route groups
        AdminRoutes.register(on: router, engine: engine)
        AdapterRoutes.register(on: router, engine: engine)
        OpenAIRoutes.register(on: router, engine: engine)
        OllamaRoutes.register(on: router, engine: engine)
        AnthropicRoutes.register(on: router, engine: engine)
        MCPRoutes.register(on: router, engine: engine)
        MetricsRoutes.register(on: router, engine: engine)

        // TLS: if both key and cert paths are present, build a TLSConfiguration
        // and use the TLS-wrapped HTTP/1 server. Otherwise plain HTTP. The
        // configuration uses NIOSSL's PEM file loaders so any self-signed
        // dev cert (`mkcert`) or production LetsEncrypt fullchain.pem +
        // privkey.pem combo just works.
        let baseConfig = ApplicationConfiguration(
            address: .hostname(host, port: port),
            serverName: "vmlx"
        )
        if let keyPath = tlsKeyPath, !keyPath.isEmpty,
           let certPath = tlsCertPath, !certPath.isEmpty
        {
            do {
                let certs = try NIOSSLCertificate.fromPEMFile(certPath)
                    .map { NIOSSLCertificateSource.certificate($0) }
                let key = try NIOSSLPrivateKey(file: keyPath, format: .pem)
                var tlsConfig = TLSConfiguration.makeServerConfiguration(
                    certificateChain: certs,
                    privateKey: .privateKey(key))
                tlsConfig.minimumTLSVersion = .tlsv12
                let app = try Application(
                    router: router,
                    server: .tls(.http1(), tlsConfiguration: tlsConfig),
                    configuration: baseConfig
                )
                await engine.logs.append(.info, category: "server",
                    "TLS enabled (cert=\(certPath), key=\(keyPath))")
                try await app.runService()
                return
            } catch {
                await engine.logs.append(.error, category: "server",
                    "TLS setup failed: \(error) — falling back to plain HTTP")
            }
        }
        let app = Application(
            router: router,
            configuration: baseConfig
        )
        try await app.runService()
    }
}

/// Sliding-window per-IP rate limiter. Each client IP is tracked in an
/// in-memory dictionary keyed by `peerAddress`; old timestamps are pruned
/// on every request. When a client exceeds `requestsPerMinute` in the
/// trailing 60s window, the middleware returns a 429 with a clean JSON
/// `{"error": {"message": "rate limit exceeded", ...}}` body.
public struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    public let requestsPerMinute: Int

    // **iter-77 (§105)** — migrated from `NSLock` to
    // `OSAllocatedUnfairLock` with `.withLock` scoped access. The
    // previous implementation invoked bare lock / unlock calls on
    // the `state` struct's NSLock directly inside this async method.
    // Under Swift 6 strict concurrency those APIs are marked
    // unavailable from async contexts (a bare lock call that
    // suspends on an actor
    // hop between it and `unlock()` would violate the sync mutex
    // contract). SourceKit flagged this as a warning across multiple
    // iterations. `.withLock` is the scoped form that works correctly
    // in async code because the closure body is synchronous and the
    // lock is released before any `await` can happen.
    private let state = OSAllocatedUnfairLock<[String: [Date]]>(initialState: [:])

    public init(requestsPerMinute: Int) {
        self.requestsPerMinute = requestsPerMinute
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let key = peerKey(from: request, context: context)
        let now = Date()
        let requestsPerMinute = self.requestsPerMinute
        let allowed = state.withLock { hits -> Bool in
            var window = (hits[key] ?? []).filter { now.timeIntervalSince($0) < 60 }
            if window.count >= requestsPerMinute {
                return false
            }
            window.append(now)
            hits[key] = window
            return true
        }
        if !allowed {
            let body = #"{"error":{"message":"rate limit exceeded","type":"rate_limit_exceeded"}}"#
            var buf = ByteBuffer()
            buf.writeBytes(body.utf8)
            return Response(
                status: .tooManyRequests,
                headers: [
                    .contentType: "application/json",
                    .retryAfter: "60",
                ],
                body: .init(byteBuffer: buf))
        }
        return try await next(request, context)
    }

    private func peerKey(from request: Request, context: Context) -> String {
        // Prefer X-Forwarded-For when set (common behind reverse proxies);
        // pick the left-most XFF entry which is the original client per
        // RFC 7239.
        if let xff = request.headers[.init("x-forwarded-for")!]?.split(separator: ",").first {
            return String(xff).trimmingCharacters(in: .whitespaces)
        }
        // P1-LIFE-1: real peer extraction. Hummingbird 2.x exposes the
        // peer SocketAddress on `BasicRequestContext.channel`. We can't
        // strongly type the protocol here without a generic constraint
        // (Context is the user's RequestContext type), so use Mirror
        // reflection — this is a fall-back path for when XFF isn't
        // present, and the structural lookup avoids forcing every caller
        // to specialize the middleware on a concrete context type.
        let mirror = Mirror(reflecting: context)
        for child in mirror.children {
            if child.label == "channel" {
                let inner = Mirror(reflecting: child.value)
                for c in inner.children where c.label == "remoteAddress" {
                    if let addr = c.value as? CustomStringConvertible {
                        return String(describing: addr)
                    }
                }
            }
        }
        return "default"
    }
}
