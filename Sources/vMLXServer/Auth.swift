import Foundation
import HTTPTypes
import Hummingbird

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
    let apiKey: String?

    public init(apiKey: String?) { self.apiKey = apiKey }

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
