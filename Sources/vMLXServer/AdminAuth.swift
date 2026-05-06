import Foundation
import HTTPTypes
import Hummingbird

/// iter-ralph §232 (M3): per-IP failed-auth counter for admin token.
/// Keeps a rolling window of recent auth misses keyed on peer IP.
/// Purges entries older than `windowSeconds`; once a peer exceeds
/// `maxFailures` inside the window, returns 429 until the window
/// scrolls off. No allocations on the success path (success clears
/// the peer's slot).
actor AdminAuthRateLimiter {
    static let shared = AdminAuthRateLimiter()

    private let maxFailures = 5
    private let windowSeconds: TimeInterval = 60

    private var failures: [String: [Date]] = [:]

    /// Returns `(shouldBlock, retryAfterSeconds)` — `shouldBlock` true
    /// when the peer has already burned its budget. When true, the
    /// caller should emit 429 WITHOUT running the token comparison so
    /// we don't give timing-oracle feedback.
    func shouldBlock(peer: String, now: Date = Date()) -> (Bool, Int) {
        let cutoff = now.addingTimeInterval(-windowSeconds)
        var window = failures[peer] ?? []
        window = window.filter { $0 > cutoff }
        failures[peer] = window
        if window.count >= maxFailures {
            let oldest = window.min() ?? now
            let retryAfter = Int(ceil(windowSeconds - now.timeIntervalSince(oldest)))
            return (true, max(1, retryAfter))
        }
        return (false, 0)
    }

    func recordFailure(peer: String, now: Date = Date()) {
        var window = failures[peer] ?? []
        window.append(now)
        // Bound runaway growth even inside the window.
        if window.count > maxFailures * 4 {
            window.removeFirst(window.count - maxFailures * 4)
        }
        failures[peer] = window
    }

    func recordSuccess(peer: String) {
        failures.removeValue(forKey: peer)
    }
}

/// Admin / cache-control endpoint auth middleware.
///
/// Gates the destructive side of the vMLX HTTP API — lifecycle, cache
/// flush, anything that could nuke a running inference session — behind a
/// dedicated admin token. Ordinary chat clients and LAN peers on the
/// friendly `/v1/*` surface are untouched.
///
/// Paths that require the token:
///   * `/admin/*`        — soft/deep sleep, wake, restart, shutdown
///   * `/v1/cache/*`     — warm, flush, stats, entry enumeration
///
/// If `adminToken` is nil or empty the middleware is a no-op (matches the
/// dev default where nobody has set one yet). Once a token is configured
/// via the UI or CLI, unauthenticated calls to the above paths get a 401
/// with a clean JSON body and a `WWW-Authenticate: Bearer realm="admin"`
/// header so SDK clients know what to do.
///
/// The token is accepted in either of:
///   * `Authorization: Bearer <token>` (OpenAI-style)
///   * `X-Admin-Token: <token>`        (compat with existing curl examples)
public struct AdminAuthMiddleware<Context: RequestContext>: RouterMiddleware {
    // iter-135 §161: box OR literal. Same pattern as BearerAuthMiddleware.
    let literal: String?
    let tokens: AuthTokenBox?

    public init(adminToken: String?) {
        if let t = adminToken, !t.isEmpty {
            self.literal = t
        } else {
            self.literal = nil
        }
        self.tokens = nil
    }

    public init(tokens: AuthTokenBox) {
        self.literal = nil
        self.tokens = tokens
    }

    /// Effective admin token resolved per-request. Empty-string is
    /// treated as nil (no gate) to match the literal init's
    /// normalization.
    var adminToken: String? {
        if let box = tokens {
            let t = box.adminToken
            return (t?.isEmpty == false) ? t : nil
        }
        return literal
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // No token configured → open access, same behaviour as before.
        guard let expected = adminToken else {
            return try await next(request, context)
        }

        // Only gate admin + cache + destructive model-mutation paths.
        // Regular inference routes (`/v1/chat/completions`,
        // `/v1/embeddings`, `/api/chat`, `/api/generate`, etc.) must
        // remain accessible with just the user API key.
        //
        // **iter-75 (§103)** — previously only `/admin/*` and
        // `/v1/cache/*` were gated. The audit surfaced four
        // destructive routes that bypassed the gate:
        //   - POST /v1/adapters/load   (arbitrary-path LoRA load)
        //   - POST /v1/adapters/unload (unload current adapter)
        //   - POST /v1/adapters/fuse   (PERMANENT weight fusion)
        //   - DELETE /api/delete       (PERMANENT on-disk model delete)
        // A LAN peer with the user API key could wipe models off
        // disk or fuse a rogue adapter into base weights. Now
        // gated alongside /admin/ and /v1/cache/. The safe read
        // path `GET /v1/adapters` (list-only) stays open so SDKs
        // can still inspect the active-adapter state.
        let path = request.uri.path
        let method = request.method
        let isAdmin = path.hasPrefix("/admin/") || path == "/admin"
        let isCache = path.hasPrefix("/v1/cache/") || path == "/v1/cache"
        let isAdapterMutation =
            path == "/v1/adapters/load"
            || path == "/v1/adapters/unload"
            || path == "/v1/adapters/fuse"
        let isOllamaDelete = (path == "/api/delete" && method == .delete)
        guard isAdmin || isCache || isAdapterMutation || isOllamaDelete else {
            return try await next(request, context)
        }

        // iter-ralph §232 (M3): rate-limit admin-token brute-force.
        // Peer key: X-Forwarded-For leftmost → request remote address
        // → "default". Matches RateLimitMiddleware.peerKey logic.
        let peer: String = {
            if let xff = request.headers[.init("x-forwarded-for")!]?
                .split(separator: ",").first {
                return String(xff).trimmingCharacters(in: .whitespaces)
            }
            let mirror = Mirror(reflecting: context)
            for child in mirror.children where child.label == "channel" {
                let inner = Mirror(reflecting: child.value)
                for c in inner.children where c.label == "remoteAddress" {
                    if let addr = c.value as? CustomStringConvertible {
                        return String(describing: addr)
                    }
                }
            }
            return "default"
        }()

        let (blocked, retryAfter) =
            await AdminAuthRateLimiter.shared.shouldBlock(peer: peer)
        if blocked {
            let body = #"{"error":{"message":"too many admin auth attempts, slow down","type":"rate_limit_error","retry_after":\#(retryAfter)}}"#
            var resp = Response(
                status: .tooManyRequests,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: body))
            )
            resp.headers[HTTPField.Name("Retry-After")!] = String(retryAfter)
            return resp
        }

        // Accept Bearer or X-Admin-Token.
        let bearer = request.headers[.authorization] ?? ""
        let xheader = request.headers[HTTPField.Name("X-Admin-Token")!] ?? ""

        // iter-87 §115: admin-token comparison must be constant-time
        // for the same reason as the BearerAuth API-key check — a LAN
        // attacker on a 0.0.0.0-bound server could otherwise recover
        // the admin token byte-by-byte via timing side-channel, and
        // the admin token gates destructive endpoints (soft/deep
        // sleep, cache flush, adapter fuse, on-disk model delete).
        let bearerOK = BearerAuthMiddleware<Context>.constantTimeEquals(bearer, "Bearer \(expected)")
        let headerOK = BearerAuthMiddleware<Context>.constantTimeEquals(xheader, expected)

        guard bearerOK || headerOK else {
            // iter-ralph §232 (M3): record the miss for per-IP bucket.
            await AdminAuthRateLimiter.shared.recordFailure(peer: peer)
            // Iter 144 — build error body via JSONSerialization rather
            // than string interpolation. Pre-fix, a request to a path
            // containing `"` (e.g. `/admin/wake?x="}`) produced a
            // malformed JSON envelope. Echoing the path also gives
            // attackers free confirmation of guarded paths — but the
            // path itself is non-secret (it's in the request line);
            // the real concern was JSON well-formedness for SDK
            // consumers that fail closed on parse errors.
            let envelope: [String: Any] = [
                "error": [
                    "message": "admin token required",
                    "type": "auth_error",
                    "path": path,
                ] as [String: Any]
            ]
            let bodyData = (try? JSONSerialization.data(withJSONObject: envelope))
                ?? Data(#"{"error":{"message":"admin token required","type":"auth_error"}}"#.utf8)
            var resp = Response(
                status: .unauthorized,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: bodyData))
            )
            resp.headers[.wwwAuthenticate] = "Bearer realm=\"admin\""
            return resp
        }
        // Success — clear any prior-miss window for this peer.
        await AdminAuthRateLimiter.shared.recordSuccess(peer: peer)
        return try await next(request, context)
    }
}
