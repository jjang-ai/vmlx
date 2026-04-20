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
        let bearerOK = header == expectedBearer
        let xApiKeyOK = xApiKey == key

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
}
