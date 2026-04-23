//
//  MCPClipboardImport.swift
//  §340 — Parse a Claude-Desktop-compatible MCP JSON payload into
//         `[MCPServerConfig]` for one-click import.
//
//  Claude Desktop + Cursor + Windsurf + Zed all share the same
//  top-level shape:
//
//    {
//      "mcpServers": {
//        "filesystem": {
//          "command": "npx",
//          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
//          "env": { "LOG_LEVEL": "debug" }
//        },
//        "notion": {
//          "url": "https://mcp.notion.com/sse",
//          "transport": "sse"
//        }
//      }
//    }
//
//  Users already have these configs sitting in
//  `~/Library/Application Support/Claude/claude_desktop_config.json`
//  (or similar) — being able to paste one block and have vMLX import
//  every server is the shortest path from "I know how to use Claude
//  Desktop" to "my vMLX has my MCP tools".
//
//  The parser is intentionally lenient:
//    - Accepts `mcpServers` OR the vMLX-native `servers` key.
//    - Defaults transport to `stdio` when `command` is present,
//      `sse` when `url` is present, `stdio` otherwise.
//    - `enabled` defaults true; `timeout` defaults 30s — matches the
//      MCPServerConfig initializer.
//    - Skips entries that fail validateShape() rather than aborting
//      the whole paste — partial imports are better than "one typo
//      lost all five servers".
//
//  Returns the list of valid server configs plus any per-server
//  error messages so the UI can show a banner like
//  "Imported 4 servers. Skipped: 'broken' (stdio transport requires
//  'command')".

import Foundation

public enum MCPClipboardImport {

    public struct Result: Sendable {
        public let servers: [MCPServerConfig]
        /// `(serverName, reason)` pairs for entries we couldn't import.
        public let skipped: [(name: String, reason: String)]

        public var totalParsed: Int { servers.count + skipped.count }
        public var hasAnyImported: Bool { !servers.isEmpty }
    }

    public enum ImportError: Error, CustomStringConvertible {
        case invalidJSON
        case noServerBlock
        case emptyServerBlock

        public var description: String {
            switch self {
            case .invalidJSON:
                return "Clipboard didn't contain valid JSON."
            case .noServerBlock:
                return "JSON had no `mcpServers` or `servers` block to import."
            case .emptyServerBlock:
                return "`mcpServers` block was empty — nothing to import."
            }
        }
    }

    /// Parse an mcp-style JSON payload. Returns validated server
    /// configs plus per-server error messages for skipped entries.
    public static func parse(json: String) throws -> Result {
        guard let data = json.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            throw ImportError.invalidJSON
        }
        // Both key spellings.
        let block: [String: Any]
        if let mcp = root["mcpServers"] as? [String: Any] {
            block = mcp
        } else if let srv = root["servers"] as? [String: Any] {
            block = srv
        } else {
            // Some users paste just the inner dict. Only accept if
            // every value is itself a dict-shaped server entry —
            // otherwise we'd mis-parse random JSON as "server configs".
            let looksLikeInnerBlock = !root.isEmpty
                && root.values.allSatisfy { $0 is [String: Any] }
            if looksLikeInnerBlock {
                block = root
            } else {
                throw ImportError.noServerBlock
            }
        }
        guard !block.isEmpty else {
            throw ImportError.emptyServerBlock
        }

        var out: [MCPServerConfig] = []
        var skipped: [(String, String)] = []
        // Sort for deterministic import order — useful when the user
        // imports from a fixed config + expects stable UI ordering.
        for name in block.keys.sorted() {
            guard let entry = block[name] as? [String: Any] else {
                skipped.append((name, "entry is not a JSON object"))
                continue
            }
            let cfg = buildConfig(name: name, entry: entry)
            if let err = cfg.validateShape() {
                skipped.append((name, err))
                continue
            }
            out.append(cfg)
        }

        return Result(servers: out, skipped: skipped)
    }

    /// Build an MCPServerConfig from a single Claude-Desktop-shape entry.
    /// Loose defaults: any missing field falls back to the
    /// MCPServerConfig init defaults.
    private static func buildConfig(
        name: String,
        entry: [String: Any]
    ) -> MCPServerConfig {
        let command = entry["command"] as? String
        let args = entry["args"] as? [String]
        let env = entry["env"] as? [String: String]
        let url = entry["url"] as? String

        // Transport: explicit field > url-implies-sse > stdio default.
        let transport: MCPTransport = {
            if let s = entry["transport"] as? String,
               let t = MCPTransport(rawValue: s.lowercased())
            {
                return t
            }
            if url != nil { return .sse }
            return .stdio
        }()

        let enabled = (entry["enabled"] as? Bool) ?? true
        let timeout = (entry["timeout"] as? Double)
            ?? (entry["timeout_seconds"] as? Double)
            ?? 30.0
        let skipSec = (entry["skip_security_validation"] as? Bool)
            ?? (entry["skipSecurityValidation"] as? Bool)
            ?? false

        return MCPServerConfig(
            name: name,
            transport: transport,
            command: command,
            args: args,
            env: env,
            url: url,
            enabled: enabled,
            timeout: timeout,
            skipSecurityValidation: skipSec
        )
    }
}
