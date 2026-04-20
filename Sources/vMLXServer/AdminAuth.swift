import Foundation
import HTTPTypes
import Hummingbird

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
    let adminToken: String?

    public init(adminToken: String?) {
        if let t = adminToken, !t.isEmpty {
            self.adminToken = t
        } else {
            self.adminToken = nil
        }
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

        // Accept Bearer or X-Admin-Token.
        let bearer = request.headers[.authorization] ?? ""
        let xheader = request.headers[HTTPField.Name("X-Admin-Token")!] ?? ""

        let bearerOK = bearer == "Bearer \(expected)"
        let headerOK = xheader == expected

        guard bearerOK || headerOK else {
            let body = #"{"error":{"message":"admin token required","type":"auth_error","path":"\#(path)"}}"#
            var resp = Response(
                status: .unauthorized,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: body))
            )
            resp.headers[.wwwAuthenticate] = "Bearer realm=\"admin\""
            return resp
        }
        return try await next(request, context)
    }
}
