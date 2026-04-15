import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine

/// MCP (Model Context Protocol) routes.
///
/// Python source: `vmlx_engine/server.py`
///   - GET  /v1/mcp/tools      — line 3912
///   - GET  /v1/mcp/servers    — line 3931
///   - POST /v1/mcp/execute    — line 3955
///
/// All routes dispatch through `engine.mcp` (an `MCPServerManager`
/// actor) which owns the stdio subprocess lifecycles and the
/// aggregated tool catalog. A missing `mcp.json` is not an error —
/// the manager simply reports zero servers and the routes return
/// empty arrays.
public enum MCPRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        // GET /v1/mcp/tools — flattened tool catalog across every
        // connected MCP server. Tool names are namespaced as
        // `server__tool` so clients can pass the returned `name`
        // directly to `/v1/mcp/execute` or OpenAI function-calling
        // without any massaging on their side.
        router.get("/v1/mcp/tools") { _, _ -> Response in
            let tools = await engine.mcp.listTools()
            let data: [[String: Any]] = tools.map { $0.toOpenAIFormat() }
            return OpenAIRoutes.json(["tools": data])
        }

        // GET /v1/mcp/servers — runtime status of every configured
        // server. Includes disconnected entries (the config rows)
        // alongside connected ones so the UI can render a full list
        // and offer a "start" button per disconnected entry.
        router.get("/v1/mcp/servers") { _, _ -> Response in
            let statuses = await engine.mcp.listServers()
            return OpenAIRoutes.json([
                "servers": statuses.map { $0.toDictionary() }
            ])
        }

        // POST /v1/mcp/execute — run one tool by its namespaced name.
        //
        // Body: `{"name": "<server>__<tool>", "arguments": {...}}`
        // Return: `{"content": "<joined text blocks>", "is_error": bool}`
        //
        // If the tool belongs to a server that's not yet started, the
        // manager lazily starts it on this call (matches Python).
        router.post("/v1/mcp/execute") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 8 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON")
            }
            guard let name = obj["name"] as? String, !name.isEmpty else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing 'name'")
            }
            let arguments = (obj["arguments"] as? [String: Any]) ?? [:]
            do {
                let result = try await engine.mcp.executeTool(
                    namespaced: name, arguments: arguments
                )
                return OpenAIRoutes.json([
                    "tool_name": result.toolName,
                    "content": result.content,
                    "is_error": result.isError,
                ])
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /mcp/:server/** — raw JSON-RPC passthrough.
        //
        // Body: `{"params": {...}}` (or `{}` for no params).
        // Return: raw MCP JSON-RPC result dict.
        //
        // Examples:
        //   POST /mcp/filesystem/resources/list     — list resources
        //   POST /mcp/github/prompts/get {"params":{"name":"summarize"}}
        //   POST /mcp/github/tools/call  {"params":{"name":"..."}}
        //
        // Uses Hummingbird's `**` recursive capture so multi-segment
        // MCP method names (`resources/list`, `tools/call`, etc.) that
        // contain slashes match this route. A single-segment `:method`
        // parameter would silently drop every real MCP method — session
        // 2026-04-14 deep nuance audit #24.
        //
        // The server is lazily started on first call. Unknown server
        // names return 404; protocol/transport errors return 500.
        router.post("/mcp/:server/**") { req, ctx -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 8 * 1024 * 1024)
            let data = Data(buffer: body)

            guard let serverName = ctx.parameters.get("server") else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing :server")
            }
            // `**` captures everything after `/mcp/:server/` into the
            // Hummingbird `:**:` key. That's the MCP method (possibly
            // multi-segment like `resources/list`).
            guard let method = ctx.parameters.get(":**:"),
                  !method.isEmpty
            else {
                return OpenAIRoutes.errorJSON(.badRequest, "missing :method")
            }

            var params: [String: Any]? = nil
            if !data.isEmpty,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Accept either `{params: {...}}` envelope or the raw
                // params dict at the top level.
                params = (obj["params"] as? [String: Any]) ?? obj
            }

            do {
                let result = try await engine.mcp.rawCall(
                    server: serverName, method: method, params: params
                )
                return OpenAIRoutes.json(result)
            } catch let e as MCPError {
                let status: HTTPResponse.Status
                switch e {
                case .serverNotFound: status = .notFound
                case .timeout: status = .gatewayTimeout
                default: status = .internalServerError
                }
                return OpenAIRoutes.errorJSON(status, "\(e)")
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
        }
    }
}
