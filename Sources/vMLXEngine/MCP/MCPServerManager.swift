// SPDX-License-Identifier: Apache-2.0
//
// MCP server manager. Owns one `MCPStdioClient` per configured server,
// handles lazy startup + handshake, surfaces status for the
// `/v1/mcp/servers` route, and exposes the flattened tool catalog
// for `/v1/mcp/tools`.
//
// Port of `vmlx_engine/mcp/manager.py` — same responsibilities but
// Swift-native concurrency instead of threads. The manager itself is
// an actor so multiple route handlers can call `listTools()` /
// `executeTool(...)` concurrently without clobbering each other.

import Foundation

public actor MCPServerManager {

    // MARK: - State

    private var config: MCPConfig
    private var clients: [String: MCPStdioClient] = [:]
    private var statuses: [String: MCPServerStatus] = [:]
    /// Cached tool catalog per server, populated after `initialize` +
    /// `tools/list`. Refreshed when the caller explicitly re-starts a
    /// server or after a protocol error.
    private var tools: [String: [MCPTool]] = [:]

    // MARK: - Init

    public init(config: MCPConfig = MCPConfig()) {
        self.config = config
        // Seed statuses so /v1/mcp/servers always has a row per config entry.
        for (name, server) in config.servers {
            self.statuses[name] = MCPServerStatus(
                name: name,
                state: .disconnected,
                transport: server.transport
            )
        }
    }

    // MARK: - Config swap

    /// Replace the active config. Any currently-running clients whose
    /// server entry was removed or changed are torn down. New entries
    /// become discoverable but are not started until first use (lazy).
    public func setConfig(_ newConfig: MCPConfig) async {
        // Stop clients whose config disappeared or changed materially.
        let oldNames = Set(config.servers.keys)
        let newNames = Set(newConfig.servers.keys)
        let toStop = oldNames.subtracting(newNames).union(
            oldNames.intersection(newNames).filter {
                config.servers[$0] != newConfig.servers[$0]
            }
        )
        for name in toStop {
            if let client = clients.removeValue(forKey: name) {
                await client.stop()
            }
            tools.removeValue(forKey: name)
            statuses[name] = nil
        }
        // Add fresh rows for new entries.
        for name in newNames.subtracting(oldNames) {
            if let server = newConfig.servers[name] {
                statuses[name] = MCPServerStatus(
                    name: name,
                    state: .disconnected,
                    transport: server.transport
                )
            }
        }
        config = newConfig
    }

    public func currentConfig() -> MCPConfig { config }

    // MARK: - Startup

    /// Start a specific server if it isn't already running. Idempotent.
    public func startServer(_ name: String) async throws {
        guard let server = config.servers[name] else {
            throw MCPError.serverNotFound(name: name)
        }
        guard server.enabled else {
            // Disabled servers aren't an error — they're just skipped.
            return
        }
        if clients[name] != nil { return }

        statuses[name] = MCPServerStatus(
            name: name,
            state: .connecting,
            transport: server.transport
        )
        let client = MCPStdioClient(server: server)
        do {
            try await client.start()
            try await client.initialize()
            let discovered = try await client.listTools()
            clients[name] = client
            tools[name] = discovered
            statuses[name] = MCPServerStatus(
                name: name,
                state: .connected,
                transport: server.transport,
                toolsCount: discovered.count,
                error: nil,
                lastConnected: Date()
            )
        } catch {
            await client.stop()
            statuses[name] = MCPServerStatus(
                name: name,
                state: .error,
                transport: server.transport,
                toolsCount: 0,
                error: String(describing: error),
                lastConnected: statuses[name]?.lastConnected
            )
            throw error
        }
    }

    /// Bring every enabled server up in parallel. Failed servers are
    /// recorded as `.error` state but do not abort the overall start.
    public func startAll() async {
        await withTaskGroup(of: Void.self) { group in
            for (name, server) in config.servers where server.enabled {
                group.addTask {
                    try? await self.startServer(name)
                }
            }
        }
    }

    /// Stop every running client and clear the tool cache.
    public func stopAll() async {
        for (name, client) in clients {
            await client.stop()
            if let s = statuses[name] {
                statuses[name] = MCPServerStatus(
                    name: s.name,
                    state: .disconnected,
                    transport: s.transport,
                    toolsCount: 0,
                    error: nil,
                    lastConnected: s.lastConnected
                )
            }
        }
        clients.removeAll()
        tools.removeAll()
    }

    // MARK: - Query

    /// Flattened tool catalog across every connected server, with
    /// namespacing via `server__tool` in `fullName`.
    public func listTools() -> [MCPTool] {
        tools.values.flatMap { $0 }
    }

    public func listServers() -> [MCPServerStatus] {
        config.servers.keys.sorted().compactMap { statuses[$0] }
    }

    // MARK: - Execution

    /// Execute a tool by its namespaced name (`server__tool`). If the
    /// server isn't running yet, it's lazily started on first call.
    public func executeTool(
        namespaced fullName: String,
        arguments: [String: Any]
    ) async throws -> MCPToolResult {
        // Split `server__tool`. A server name can't contain `__` so
        // this is unambiguous (matches Python split).
        let parts = fullName.components(separatedBy: "__")
        guard parts.count >= 2 else {
            throw MCPError.protocolError(reason: "tool name missing server prefix")
        }
        let serverName = parts[0]
        let toolName = parts.dropFirst().joined(separator: "__")

        guard config.servers[serverName] != nil else {
            throw MCPError.serverNotFound(name: serverName)
        }
        if clients[serverName] == nil {
            try await startServer(serverName)
        }
        guard let client = clients[serverName] else {
            throw MCPError.serverNotFound(name: serverName)
        }
        return try await client.callTool(name: toolName, arguments: arguments)
    }

    /// Raw JSON-RPC passthrough — forwards an arbitrary `method` +
    /// `params` to the named MCP server and returns the raw result
    /// dictionary. Used by `POST /mcp/<server>/<method>` for clients
    /// that want direct protocol access (e.g. `resources/list`,
    /// `prompts/get`, custom server extensions).
    ///
    /// The server is lazily started if not yet running, mirroring
    /// `executeTool`. Errors bubble up as `MCPError.serverNotFound`
    /// or `MCPError.protocolError` / `.timeout` from the transport.
    public func rawCall(
        server serverName: String,
        method: String,
        params: [String: Any]?
    ) async throws -> [String: Any] {
        guard config.servers[serverName] != nil else {
            throw MCPError.serverNotFound(name: serverName)
        }
        if clients[serverName] == nil {
            try await startServer(serverName)
        }
        guard let client = clients[serverName] else {
            throw MCPError.serverNotFound(name: serverName)
        }
        return try await client.call(method: method, params: params)
    }
}
