// SPDX-License-Identifier: Apache-2.0
//
// MCP config loader. Port of `vmlx_engine/mcp/config.py`.
//
// Search order (first hit wins):
//   1. explicit path passed to `load(path:)`
//   2. $VMLX_MCP_CONFIG env var
//   3. ./mcp.json
//   4. ~/.config/vmlx/mcp.json
//   5. ~/.config/vmlx-engine/mcp.json (legacy path from the Python side)
//
// Only JSON is supported for now. YAML is not widely used by MCP
// clients anyway (VS Code / Claude Desktop both use JSON).

import Foundation

public enum MCPConfigLoader {

    /// Environment variable name that points at an explicit config path.
    public static let envVar = "VMLX_MCP_CONFIG"

    /// Default search paths, in priority order.
    public static var defaultSearchPaths: [URL] {
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let home = fm.homeDirectoryForCurrentUser
        return [
            cwd.appendingPathComponent("mcp.json"),
            home.appendingPathComponent(".config/vmlx/mcp.json"),
            home.appendingPathComponent(".config/vmlx-engine/mcp.json"),
        ]
    }

    /// Find the first config file on disk, honoring the explicit path
    /// argument then the env var then the default search paths.
    public static func findConfigFile(path explicit: URL? = nil) -> URL? {
        let fm = FileManager.default
        if let explicit = explicit, fm.fileExists(atPath: explicit.path) {
            return explicit
        }
        if let envPath = ProcessInfo.processInfo.environment[envVar],
           !envPath.isEmpty
        {
            let url = URL(fileURLWithPath: (envPath as NSString).expandingTildeInPath)
            if fm.fileExists(atPath: url.path) { return url }
        }
        for candidate in defaultSearchPaths {
            if fm.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// Load and validate the MCP config from disk. Returns an empty
    /// `MCPConfig` when no file is found — a missing config file means
    /// "no MCP servers configured", not an error.
    public static func load(path: URL? = nil) throws -> MCPConfig {
        guard let url = findConfigFile(path: path) else {
            return MCPConfig()  // empty
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let cfg: MCPConfig
        do {
            cfg = try decoder.decode(MCPConfig.self, from: data)
        } catch {
            throw MCPError.configInvalid(
                reason: "\(url.lastPathComponent): \(error.localizedDescription)"
            )
        }

        // Shape + security validation in a second pass so we can
        // report ALL problems at once rather than stopping at the
        // first bad entry.
        var errors: [String] = []
        for (name, server) in cfg.servers {
            if let reason = server.validateShape() {
                errors.append("\(name): \(reason)")
            }
            if !server.skipSecurityValidation {
                if let violation = MCPSecurity.validate(server) {
                    errors.append("\(name): \(violation)")
                }
            }
        }
        if !errors.isEmpty {
            throw MCPError.configInvalid(
                reason: errors.joined(separator: "; ")
            )
        }
        return cfg
    }

    /// Persist an `MCPConfig` back to disk as pretty-printed JSON. Used
    /// by the in-app CRUD UI so edits round-trip to the file the user
    /// chose (or to the first default search path when none was set).
    ///
    /// Returns the URL that was written. Creates parent directories as
    /// needed. Atomic write (temp-file + rename) so a crash mid-write
    /// can't truncate the user's config.
    @discardableResult
    public static func save(config: MCPConfig, to explicit: URL? = nil) throws -> URL {
        let fm = FileManager.default
        let target: URL = {
            if let explicit = explicit { return explicit }
            if let found = findConfigFile() { return found }
            // No existing file — land the new one at the primary
            // default search path and create the parent dir.
            return defaultSearchPaths[1]  // ~/.config/vmlx/mcp.json
        }()
        let parent = target.deletingLastPathComponent()
        if !fm.fileExists(atPath: parent.path) {
            try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(config)
        try data.write(to: target, options: .atomic)
        return target
    }
}
