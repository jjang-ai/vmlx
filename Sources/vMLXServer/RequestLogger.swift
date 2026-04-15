import Foundation
import HTTPTypes
import Hummingbird
import vMLXEngine

/// Hummingbird middleware that logs every HTTP request into the engine's
/// `LogStore` (category `"server"`).
///
/// Mirrors the `@app.middleware("http")` request/response logger in
/// `vmlx_engine/server.py` — method, path, status, and elapsed milliseconds
/// are captured per request and routed through the same log stream the
/// LogsPanel tails. Log level is chosen by status code so 5xx responses
/// bubble up to the "error" filter in the UI automatically.
public struct RequestLoggerMiddleware<Context: RequestContext>: RouterMiddleware {
    let engine: Engine
    let minLevel: LogStore.Level

    public init(engine: Engine, minLevel: LogStore.Level = .info) {
        self.engine = engine
        self.minLevel = minLevel
    }

    public func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let start = DispatchTime.now()
        let method = request.method.rawValue
        let path = request.uri.path

        do {
            let response = try await next(request, context)
            let elapsedMs = Self.elapsedMs(since: start)
            let status = response.status.code
            let level: LogStore.Level = {
                if status >= 500 { return .error }
                if status >= 400 { return .warn }
                return .info
            }()
            if level >= minLevel {
                await engine.logs.append(
                    level, category: "server",
                    "\(method) \(path) -> \(status) (\(Self.fmt(elapsedMs))ms)"
                )
            }
            return response
        } catch {
            let elapsedMs = Self.elapsedMs(since: start)
            await engine.logs.append(
                .error, category: "server",
                "\(method) \(path) -> thrown \(error) (\(Self.fmt(elapsedMs))ms)"
            )
            throw error
        }
    }

    private static func elapsedMs(since start: DispatchTime) -> Double {
        let ns = DispatchTime.now().uptimeNanoseconds &- start.uptimeNanoseconds
        return Double(ns) / 1_000_000.0
    }

    private static func fmt(_ ms: Double) -> String {
        String(format: "%.1f", ms)
    }
}
