import Foundation
import HTTPTypes
import Hummingbird

/// Bearer-token authentication middleware.
///
/// Mirrors `vmlx_engine/server.py::verify_api_key`. If `apiKey` is nil, all requests pass
/// (matching Python's default when no `VMLX_API_KEY` env var is set). Otherwise we require
/// `Authorization: Bearer <key>` exactly.
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
        let header = request.headers[.authorization] ?? ""
        let expected = "Bearer \(key)"
        guard header == expected else {
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
