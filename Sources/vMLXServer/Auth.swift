import Foundation
import HTTPTypes
import Hummingbird
import os

/// iter-135 §161: thread-safe container for the live API key + admin
/// token. `BearerAuthMiddleware` / `AdminAuthMiddleware` hold a
/// reference and read on every request, so the HTTP server can
/// swap credentials while running (revoke the active API key, or
/// pick up a global-settings change) without tearing down and
/// restarting the listener.
///
/// Before §161, the middleware captured `apiKey: String?` at init.
/// `APIKeyManager.revoke` + `applySettings` updated SQLite but the
/// running middleware kept using the old value — the iter-96 §123
/// revoke-dialog promise ("any client using this key will
/// immediately lose access") was a lie until the next server
/// restart. AuthTokenBox fixes that by turning the token fields
/// into a reference the middleware can re-read per request.
public final class AuthTokenBox: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()
    private var _apiKey: String?
    private var _adminToken: String?

    public init(apiKey: String? = nil, adminToken: String? = nil) {
        self._apiKey = apiKey
        self._adminToken = adminToken
    }

    public var apiKey: String? { lock.withLock { _apiKey } }
    public var adminToken: String? { lock.withLock { _adminToken } }

    /// Atomically swap the stored credentials. Next incoming request
    /// will read these values. In-flight requests already past the
    /// middleware gate are not affected (acceptable — the outgoing
    /// response goes out on the same socket they entered on).
    public func update(apiKey: String?, adminToken: String?) {
        lock.withLock {
            self._apiKey = apiKey
            self._adminToken = adminToken
        }
    }
}

/// Bearer-token authentication middleware.
///
/// Mirrors `vmlx_engine/server.py::verify_api_key`. If `apiKey` is nil,
/// all requests pass (matching Python's default when no `VMLX_API_KEY`
/// env var is set).
///
/// **iter-76 (§104)**: accepts the API key in either of:
///   - `Authorization: Bearer <key>`   (OpenAI / Ollama SDK convention)
///   - `x-api-key: <key>`              (Anthropic SDK convention)
///
/// The dual-header handling means an Anthropic client hitting our
/// `/v1/messages` endpoint with their SDK's default `x-api-key` header
/// authenticates correctly, rather than getting a confusing 401 that
/// only went away after they manually injected an `Authorization:
/// Bearer` header. Before this change, the only accepted format was
/// `Authorization: Bearer`, which broke any SDK that didn't ship the
/// OpenAI header convention out of the box.
///
/// **iter-76 (§104)** also exempts `GET /health` from the bearer
/// check so external monitoring probes (uptime checks, tray pulse,
/// load balancers) can verify liveness without holding the API key.
/// /health returns only `{state, model_name}` — no token-material
/// or sensitive user data — so leaking it to unauth'd probes is
/// safe. Matches the standard "health endpoints are public"
/// convention used by FastAPI, nginx, Kubernetes, etc.
public struct BearerAuthMiddleware<Context: RequestContext>: RouterMiddleware {
    // iter-135 §161: optional box OR literal — box wins when both set.
    // The literal init preserves back-compat with call-sites that haven't
    // migrated; the box init lets callers swap credentials live.
    let literal: String?
    let tokens: AuthTokenBox?

    public init(apiKey: String?) {
        self.literal = apiKey
        self.tokens = nil
    }

    public init(tokens: AuthTokenBox) {
        self.literal = nil
        self.tokens = tokens
    }

    /// Effective key resolved per-request — reads from box when present,
    /// literal when the back-compat init was used.
    var apiKey: String? { tokens?.apiKey ?? literal }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard let key = apiKey, !key.isEmpty else {
            return try await next(request, context)
        }

        // /health is public so monitors can probe without a token.
        // Only the exact path; not a prefix match, to avoid an
        // unintended backdoor like `/health/something-sensitive`.
        if request.uri.path == "/health" && request.method == .get {
            return try await next(request, context)
        }

        let header = request.headers[.authorization] ?? ""
        let xApiKey = request.headers[HTTPField.Name("x-api-key")!] ?? ""
        let expectedBearer = "Bearer \(key)"
        // iter-87 §115: use constant-time comparison so an attacker on
        // LAN (when server binds 0.0.0.0) can't recover the API key
        // byte-by-byte by measuring response-time differences between
        // a one-byte-wrong header and a fully wrong header. Swift's
        // stdlib `==` on `String` is variable-time (early-exit on
        // first mismatch).
        let bearerOK = Self.constantTimeEquals(header, expectedBearer)
        let xApiKeyOK = Self.constantTimeEquals(xApiKey, key)

        guard bearerOK || xApiKeyOK else {
            let body = #"{"error":{"message":"unauthorized","type":"auth_error"}}"#
            var resp = Response(
                status: .unauthorized,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: body))
            )
            resp.headers[.wwwAuthenticate] = "Bearer"
            return resp
        }
        return try await next(request, context)
    }

    /// Constant-time string comparison for token / API-key checks.
    ///
    /// Mitigates timing-oracle attacks where an attacker on LAN
    /// measures the round-trip time of auth attempts to recover the
    /// expected key byte-by-byte. The loop always iterates over the
    /// longer of the two UTF-8 byte sequences and XORs every byte, so
    /// the elapsed time is independent of where the first mismatch
    /// occurs. Length-mismatch also flips the accumulator so a short
    /// prefix of a longer key doesn't return true.
    static func constantTimeEquals(_ a: String, _ b: String) -> Bool {
        let ab = Array(a.utf8)
        let bb = Array(b.utf8)
        // Force the loop to a fixed count derived from max length so a
        // length mismatch doesn't leak via loop-iteration count.
        let n = max(ab.count, bb.count)
        var diff: UInt8 = UInt8(ab.count ^ bb.count) & 0xFF
        for i in 0..<n {
            let x: UInt8 = i < ab.count ? ab[i] : 0
            let y: UInt8 = i < bb.count ? bb[i] : 0
            diff |= (x ^ y)
        }
        return diff == 0
    }
}
