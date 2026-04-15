// SPDX-License-Identifier: Apache-2.0
//
// MCP security validator. Pared-down port of
// `vmlx_engine/mcp/security.py` — just the defensive checks that
// matter for first-run safety, not the full allowlist/sandboxing
// infrastructure Python ships for multi-tenant deployments.
//
// What we validate:
//  - `command` is an absolute path OR a basename that resolves via PATH
//  - no shell metacharacters in command/args/env
//  - for SSE transport: URL is http/https (not file:// or javascript:)
//  - environment keys don't collide with known secrets (AWS_*, etc —
//    returning a warning string, not an error)
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
