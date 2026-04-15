// SPDX-License-Identifier: Apache-2.0
//
// Swift port of `vmlx_engine/mcp/types.py`.
//
// Data model for the Model Context Protocol client:
// - `MCPTransport` — stdio vs sse (http coming later)
// - `MCPServerConfig` — one server entry in mcp.json
// - `MCPConfig` — root config with `servers` dict
// - `MCPTool` — normalized tool shape (server + name + schema)
// - `MCPToolResult` — result of a `tools/call` execution
// - `MCPServerStatus` — per-server runtime state for the /v1/mcp/servers route
//
// Parity note: Python's namespaced name is `server__tool` (double
// underscore). We match that exactly so wire-level integration with
// OpenAI-style clients round-trips cleanly.

import Foundation

// MARK: - Transport + state enums

/// Which transport an MCP server speaks. `stdio` is a local subprocess;
/// `sse` is a remote endpoint over HTTP+SSE. HTTP (plain JSON-RPC) is
/// not yet implemented but would slot in here.
public enum MCPTransport: String, Sendable, Codable {
    case stdio
    case sse
}

/// The current connection state of an MCP server.
public enum MCPServerState: String, Sendable, Codable, Equatable {
    case disconnected
    case connecting
    case connected
    case error
}

// MARK: - Server config

/// Configuration for a single MCP server.
///
/// Matches the Python `MCPServerConfig` dataclass shape so the same
/// `mcp.json` file loads in both runtimes. Security validation is
/// deferred to `MCPSecurity.validate(_:)` — the initializer just
/// parses the fields.
public struct MCPServerConfig: Sendable, Codable, Equatable {
    public var name: String
    public var transport: MCPTransport

    // stdio
    public var command: String?
    public var args: [String]?
    public var env: [String: String]?

    // sse
    public var url: String?

    // common
    public var enabled: Bool
    public var timeout: Double

    /// When true, bypass `MCPSecurity.validate` at load time. Only for
    /// development use — documented in the Python source with the same
    /// name so switching runtimes doesn't break configs.
    public var skipSecurityValidation: Bool

    public init(
        name: String,
        transport: MCPTransport = .stdio,
        command: String? = nil,
        args: [String]? = nil,
        env: [String: String]? = nil,
        url: String? = nil,
        enabled: Bool = true,
        timeout: Double = 30.0,
        skipSecurityValidation: Bool = false
    ) {
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args
        self.env = env
        self.url = url
        self.enabled = enabled
        self.timeout = timeout
        self.skipSecurityValidation = skipSecurityValidation
    }

    enum CodingKeys: String, CodingKey {
        case name
        case transport
        case command
        case args
        case env
        case url
        case enabled
        case timeout
        case skipSecurityValidation = "skip_security_validation"
    }

    /// Minimal sanity check — the security validator runs as a separate
    /// pass in `MCPConfig.load`. Returns `nil` on valid, or an error
    /// string describing what's wrong.
    public func validateShape() -> String? {
        if transport == .stdio, command == nil || command!.isEmpty {
            return "stdio transport requires 'command'"
        }
        if transport == .sse, url == nil || url!.isEmpty {
            return "sse transport requires 'url'"
        }
        return nil
    }
}

// MARK: - Root config

/// Root MCP config loaded from `mcp.json`.
public struct MCPConfig: Sendable, Codable, Equatable {
    public var servers: [String: MCPServerConfig]
    public var maxToolCalls: Int
    public var defaultTimeout: Double

    public init(
        servers: [String: MCPServerConfig] = [:],
        maxToolCalls: Int = 10,
        defaultTimeout: Double = 30.0
    ) {
        self.servers = servers
        self.maxToolCalls = maxToolCalls
        self.defaultTimeout = defaultTimeout
    }

    enum CodingKeys: String, CodingKey {
        case servers
        case mcpServers
        case maxToolCalls = "max_tool_calls"
        case defaultTimeout = "default_timeout"
    }

    /// Custom decoder that accepts both `servers` and the Claude-style
    /// `mcpServers` key (Anthropic's desktop app uses the latter).
    /// If a name appears in BOTH sections, the `servers` entry wins —
    /// same precedence as the Python loader — and the collision is
    /// recorded on the `decodeWarnings` thread-local so callers can
    /// surface it via logs without making the decoder throw.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var raw: [String: MCPServerConfig] = [:]
        if let rawClaude = try c.decodeIfPresent(
            [String: RawServerEntry].self, forKey: .mcpServers
        ) {
            for (name, entry) in rawClaude {
                raw[name] = entry.toConfig(name: name)
            }
        }
        if let rawServers = try c.decodeIfPresent(
            [String: RawServerEntry].self, forKey: .servers
        ) {
            for (name, entry) in rawServers {
                if raw[name] != nil {
                    MCPConfig.appendDecodeWarning(
                        "server '\(name)' appears in both 'servers' and 'mcpServers' — 'servers' wins"
                    )
                }
                raw[name] = entry.toConfig(name: name)
            }
        }
        self.servers = raw
        self.maxToolCalls = try c.decodeIfPresent(Int.self, forKey: .maxToolCalls) ?? 10
        self.defaultTimeout = try c.decodeIfPresent(Double.self, forKey: .defaultTimeout) ?? 30.0
    }

    /// Thread-local decode warnings, captured across a single
    /// `JSONDecoder().decode(MCPConfig.self, from:)` call. The
    /// `MCPConfigLoader` drains this after decode so warnings can
    /// land in the engine log without making the decoder throw.
    nonisolated(unsafe) private static var _decodeWarnings: [String] = []
    private static let decodeWarningsLock = NSLock()

    public static func drainDecodeWarnings() -> [String] {
        decodeWarningsLock.lock()
        defer { decodeWarningsLock.unlock() }
        let out = _decodeWarnings
        _decodeWarnings = []
        return out
    }

    fileprivate static func appendDecodeWarning(_ message: String) {
        decodeWarningsLock.lock()
        _decodeWarnings.append(message)
        decodeWarningsLock.unlock()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(servers, forKey: .servers)
        try c.encode(maxToolCalls, forKey: .maxToolCalls)
        try c.encode(defaultTimeout, forKey: .defaultTimeout)
    }
}

/// Intermediate decoder for `MCPConfig` — the config file omits `name`
/// (the dict key is the name) so we inject it when converting.
private struct RawServerEntry: Codable {
    var transport: MCPTransport?
    var command: String?
    var args: [String]?
    var env: [String: String]?
    var url: String?
    var enabled: Bool?
    var timeout: Double?
    var skipSecurityValidation: Bool?

    enum CodingKeys: String, CodingKey {
        case transport, command, args, env, url, enabled, timeout
        case skipSecurityValidation = "skip_security_validation"
    }

    func toConfig(name: String) -> MCPServerConfig {
        // Auto-detect transport if omitted: a `url` field implies SSE,
        // otherwise stdio. Matches Python's same permissive behavior.
        let t: MCPTransport
        if let explicit = transport {
            t = explicit
        } else if url != nil {
            t = .sse
        } else {
            t = .stdio
        }
        return MCPServerConfig(
            name: name,
            transport: t,
            command: command,
            args: args,
            env: env,
            url: url,
            enabled: enabled ?? true,
            timeout: timeout ?? 30.0,
            skipSecurityValidation: skipSecurityValidation ?? false
        )
    }
}

// MARK: - Tools

/// A normalized tool description returned by `tools/list` on an MCP
/// server. `serverName` + `name` disambiguate across servers (two
/// servers might both expose a `write_file` tool); `fullName` is the
/// `server__tool` form that's used as the OpenAI function name.
public struct MCPTool: Sendable, Equatable {
    public let serverName: String
    public let name: String
    public let description: String
    /// JSON-Schema-shaped `inputSchema` field from the MCP server.
    /// Stored as raw `Data` so we can round-trip it into OpenAI format
    /// without re-encoding type-erased dictionaries.
    public let inputSchemaJSON: Data

    public init(
        serverName: String,
        name: String,
        description: String,
        inputSchemaJSON: Data
    ) {
        self.serverName = serverName
        self.name = name
        self.description = description
        self.inputSchemaJSON = inputSchemaJSON
    }

    public var fullName: String { "\(serverName)__\(name)" }

    /// Convert to OpenAI function-calling shape for embedding into a
    /// chat-completion `tools` array.
    public func toOpenAIFormat() -> [String: Any] {
        let schema: Any
        if let obj = try? JSONSerialization.jsonObject(with: inputSchemaJSON) {
            schema = obj
        } else {
            schema = ["type": "object", "properties": [:] as [String: Any]]
        }
        return [
            "type": "function",
            "function": [
                "name": fullName,
                "description": description,
                "parameters": schema,
            ] as [String: Any],
        ]
    }
}

/// Result from a `tools/call` execution.
public struct MCPToolResult: Sendable, Equatable {
    public let toolName: String
    public let content: String
    public let isError: Bool

    public init(toolName: String, content: String, isError: Bool = false) {
        self.toolName = toolName
        self.content = content
        self.isError = isError
    }

    /// Convert to the OpenAI-style `role: "tool"` message that feeds
    /// back into the next generation pass.
    public func toMessage(toolCallId: String) -> [String: Any] {
        [
            "role": "tool",
            "tool_call_id": toolCallId,
            "content": isError ? "Error: \(content)" : content,
        ]
    }
}

// MARK: - Server status

/// Runtime status of an MCP server for `/v1/mcp/servers`.
public struct MCPServerStatus: Sendable, Equatable {
    public let name: String
    public let state: MCPServerState
    public let transport: MCPTransport
    public let toolsCount: Int
    public let error: String?
    public let lastConnected: Date?

    public init(
        name: String,
        state: MCPServerState,
        transport: MCPTransport,
        toolsCount: Int = 0,
        error: String? = nil,
        lastConnected: Date? = nil
    ) {
        self.name = name
        self.state = state
        self.transport = transport
        self.toolsCount = toolsCount
        self.error = error
        self.lastConnected = lastConnected
    }

    public func toDictionary() -> [String: Any] {
        var out: [String: Any] = [
            "name": name,
            "state": state.rawValue,
            "transport": transport.rawValue,
            "tools_count": toolsCount,
        ]
        if let error = error { out["error"] = error }
        if let lastConnected = lastConnected {
            out["last_connected"] = lastConnected.timeIntervalSince1970
        }
        return out
    }
}

// MARK: - Errors

public enum MCPError: Error, CustomStringConvertible {
    case configNotFound(path: String)
    case configInvalid(reason: String)
    case serverNotFound(name: String)
    case transportNotSupported(MCPTransport)
    case processFailure(reason: String)
    case protocolError(reason: String)
    case securityViolation(reason: String)
    case timeout(server: String, method: String)

    public var description: String {
        switch self {
        case .configNotFound(let p):   return "mcp: config not found at \(p)"
        case .configInvalid(let r):    return "mcp: invalid config — \(r)"
        case .serverNotFound(let n):   return "mcp: server '\(n)' not found"
        case .transportNotSupported(let t):
            return "mcp: transport '\(t.rawValue)' not yet implemented"
        case .processFailure(let r):   return "mcp: subprocess error — \(r)"
        case .protocolError(let r):    return "mcp: protocol error — \(r)"
        case .securityViolation(let r): return "mcp: security violation — \(r)"
        case .timeout(let s, let m):
            return "mcp: timeout waiting for \(s).\(m)"
        }
    }
}
