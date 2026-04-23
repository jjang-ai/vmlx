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
///
/// **iter-79 sensitive-data invariant (§107)**:
///
///   - This middleware deliberately NEVER logs auth headers (any
///     header-subscript lookup by the Authorization / x-api-key /
///     x-admin-token names), request or response payload bytes,
///     or the URL query string. Only method, path, status, and
///     elapsed ms land in the LogStore — plus on throw, the
///     thrown error's interpolated description.
///
///   - Route handlers throughout the server catch their own errors
///     and return a structured JSON 4xx/5xx response, so only
///     unexpected throws propagate up to this middleware. When
///     they do, the interpolated text is typically a Foundation IO
///     error (e.g. "Connection closed") or a Swift DecodingError
///     (which carries a coding-path key list, NOT the decoded
///     value).
///
///   - LogStore itself is bounded (5000-line ring buffer, see
///     `LogStore.init(capacity:)`) so even a log-pump adversary
///     can't indefinitely inflate engine memory via spammy
///     requests.
///
/// The §107 regression guard pins this invariant: if a future
/// change adds any sensitive-data-reader pattern to this file the
/// test fires.
public struct RequestLoggerMiddleware<Context: RequestContext>: RouterMiddleware {
    let engine: Engine

    /// `minLevel` is retained for source compatibility but no longer
    /// gates the append. `LogStore.append` owns the threshold check via
    /// its live-swappable `_globalMinLevel` (R4 §305 / S3 §309 /
    /// §319 O2-live-swap fix). Middleware-level filtering here would
    /// shadow `/admin/log-level` bumps — `warn`-initialized middleware
    /// would silently drop info-level request lines even after an
    /// operator lowered the global threshold to `trace`/`debug`. Pass
    /// every line to LogStore and let the global filter decide.
    public init(engine: Engine, minLevel: LogStore.Level = .info) {
        self.engine = engine
        _ = minLevel
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
            // R1 §302: stamp trace id on the log line so the
            // APIScreen request log + any downstream consumer can
            // correlate a row with the client's response envelope.
            // Populated by OpenAIRoutes Q1 — SSE + non-streaming
            // chat/completions/responses set `x-vmlx-trace-id`.
            let tidSuffix: String
            if let hname = HTTPField.Name("x-vmlx-trace-id"),
               let tid = response.headers[hname], !tid.isEmpty
            {
                tidSuffix = " [tid=\(tid)]"
            } else {
                tidSuffix = ""
            }
            await engine.logs.append(
                level, category: "server",
                "\(method) \(path) -> \(status) (\(Self.fmt(elapsedMs))ms)\(tidSuffix)"
            )
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
