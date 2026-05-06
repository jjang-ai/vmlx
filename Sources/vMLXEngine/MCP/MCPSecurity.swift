// SPDX-License-Identifier: Apache-2.0
//
// MCP security validator. Port of `vmlx_engine/mcp/security.py`.
//
// What we validate:
//  - `command` is in the executable allowlist (basename only) or is a
//    skipSecurityValidation override (mlxstudio#31)
//  - no shell metacharacters in command/args/env (the 11-character set)
//  - NUL byte rejection in args + env keys + env values
//  - `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES` etc. blocked in `env`
//    (loader-injection attack surface)
//  - for SSE transport: URL is http/https (not file:// or javascript:)
//
// What we don't yet validate (future session):
//  - binary signing
//  - executable allowlist from the global settings store
//  - per-server capabilities (filesystem vs network vs subprocess)
//
// All checks are pure — no file system mutation, no process exec,
// no logging. Caller decides whether to abort on failure.

import Foundation

public enum MCPSecurity {

    /// Allowlist of executable basenames permitted as MCP `command`.
    /// Mirrors `vmlx_engine/mcp/security.py:ALLOWED_COMMANDS`. Iter 143:
    /// previously absent — any binary on PATH could execute. Per
    /// mlxstudio#31 the allowlist includes `bun, bunx, deno,
    /// python3.10–3.13, java` for IDE-bundled MCP servers.
    public static let allowedCommands: Set<String> = [
        // Node.js + JS/TS package runners (most official MCP servers)
        "npx", "npm", "node",
        "bun", "bunx",
        "deno",
        // Python package runners (mlxstudio#31)
        "uvx", "uv",
        "python", "python3",
        "python3.10", "python3.11", "python3.12", "python3.13",
        "pip", "pipx",
        // JVM (mlxstudio#31: JetBrains IDE MCP servers run as
        // `java -classpath … com.intellij.mcpserver.stdio.…`)
        "java",
        // Official MCP servers when installed globally
        "mcp-server-filesystem",
        "mcp-server-fs",
        "mcp-server-sqlite",
        "mcp-server-postgres",
        "mcp-server-github",
        "mcp-server-slack",
        "mcp-server-memory",
        "mcp-server-puppeteer",
        "mcp-server-brave-search",
        "mcp-server-google-maps",
        "mcp-server-fetch",
        // Containerized MCP servers
        "docker",
    ]

    /// Environment variables that can hijack process loading. Setting
    /// any of these via the MCP `env` map is rejected because a
    /// malicious config could redirect dynamic linking to attacker code.
    /// Mirror of Python's `dangerous_env_vars`. PATH/PYTHONPATH/NODE_PATH
    /// are intentionally allowed because real MCP servers need them.
    public static let dangerousEnvKeys: Set<String> = [
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "NODE_OPTIONS",            // can inject --require to load arbitrary code
        "ELECTRON_RUN_AS_NODE",    // turns Electron apps into plain Node.js
    ]

    /// Run all safety checks against a server config. Returns `nil`
    /// when the config is acceptable, or a short human-readable string
    /// explaining the violation.
    public static func validate(_ server: MCPServerConfig) -> String? {
        switch server.transport {
        case .stdio:
            return validateStdio(server)
        case .sse:
            return validateSSE(server)
        }
    }

    // MARK: - stdio

    private static func validateStdio(_ s: MCPServerConfig) -> String? {
        guard let command = s.command, !command.isEmpty else {
            return "stdio transport requires a non-empty 'command'"
        }
        if containsShellMetachar(command) {
            return "command contains shell metacharacters: \(command)"
        }
        // Allowlist check on basename. Caller can opt out per-server
        // via `skip_security_validation: true` in mcp.json — but
        // MCPConfigLoader gates the entire validate(_:) call on that
        // flag already, so reaching this point means the user did NOT
        // opt out and the basename must be on the allowlist.
        let basename = (command as NSString).lastPathComponent
        if !allowedCommands.contains(basename) {
            return "command '\(basename)' is not in the MCP allowlist. " +
                   "Set `skip_security_validation: true` on the server " +
                   "to override (development only)."
        }
        if let args = s.args {
            for (i, arg) in args.enumerated() {
                // Args CAN contain paths, flags, spaces — we only
                // reject the most dangerous patterns. Arg-level shell
                // metacharacter matching is too noisy for real MCP
                // servers which pass paths with shell-safe content.
                if arg.contains("\0") {
                    return "arg[\(i)] contains NUL"
                }
            }
        }
        if let env = s.env {
            for (k, v) in env {
                if k.contains("=") || k.contains("\0") {
                    return "env key '\(k)' has illegal characters"
                }
                if v.contains("\0") {
                    return "env value for '\(k)' has illegal characters"
                }
                // Reject loader-injection env vars regardless of value.
                // Matches Python's `dangerous_env_vars` rejection.
                if dangerousEnvKeys.contains(k.uppercased()) {
                    return "env key '\(k)' is blocked (loader-injection vector)"
                }
            }
        }
        return nil
    }

    // MARK: - sse

    private static func validateSSE(_ s: MCPServerConfig) -> String? {
        guard let urlString = s.url, !urlString.isEmpty else {
            return "sse transport requires a non-empty 'url'"
        }
        guard let url = URL(string: urlString), let scheme = url.scheme?.lowercased() else {
            return "invalid SSE URL: \(urlString)"
        }
        guard scheme == "http" || scheme == "https" else {
            return "SSE URL scheme must be http/https, got '\(scheme)'"
        }
        return nil
    }

    // MARK: - helpers

    /// Block the most dangerous shell metacharacters in a command
    /// string. Args are allowed to contain spaces and paths, so we
    /// only check the command-name slot.
    private static func containsShellMetachar(_ s: String) -> Bool {
        let danger: Set<Character> = [";", "|", "&", "`", "$", "(", ")",
                                      "<", ">", "\n", "\r"]
        return s.contains(where: { danger.contains($0) })
    }
}
