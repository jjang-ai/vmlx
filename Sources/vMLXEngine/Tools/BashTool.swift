import Foundation

/// Full-shell `bash` tool — gives the model unrestricted terminal access.
///
/// This intentionally bypasses any sandboxing or allow-listing. The Swift
/// rewrite ships as a Developer ID notarized DMG (NOT Mac App Store) so the
/// sandbox isn't in play. See `project_swift_terminal_mode.md`.
///
/// Registered as an OpenAI-compatible function tool so models invoke it via
/// standard tool-calling. The HTTP server intercepts `bash` tool calls,
/// executes them server-side via `BashTool.run`, and feeds the result back
/// into the generation loop.
public actor BashTool {

    public struct Result: Sendable {
        public var stdout: String
        public var stderr: String
        public var exitCode: Int32
        public var timedOut: Bool
        public var killed: Bool
        /// Working directory after the command ran. Extracted from a
        /// trailing `__VMLX_PWD__:<path>` marker the foreground wrapper
        /// always prints. `nil` when the marker was missing (e.g.
        /// background mode, timeout, or wrapper not installed). Used
        /// by Terminal mode to persist cwd across calls so a `cd foo`
        /// in one tool invocation affects the next.
        public var newCwd: URL?
    }

    public struct Invocation: Sendable {
        public var command: String
        public var cwd: URL?
        public var timeoutSeconds: Double?
        public var runInBackground: Bool

        public init(command: String,
                    cwd: URL? = nil,
                    timeoutSeconds: Double? = 120,
                    runInBackground: Bool = false) {
            self.command = command
            self.cwd = cwd
            self.timeoutSeconds = timeoutSeconds
            self.runInBackground = runInBackground
        }
    }

    /// OpenAI function-tool schema for `bash`. Inject this into the request
    /// `tools` array whenever Terminal mode is active.
    public nonisolated static var openAISchema: ChatRequest.Tool {
        // JSON Schema for bash args. Must be present so the model emits
        // a `command` string rather than calling with `{}`.
        let schema: JSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "command": .object([
                    "type": .string("string"),
                    "description": .string("Shell command to run via /bin/zsh -c."),
                ]),
                "cwd": .object([
                    "type": .string("string"),
                    "description": .string("Optional working directory. Defaults to the Terminal cwd."),
                ]),
                "timeout": .object([
                    "type": .string("number"),
                    "description": .string("Optional timeout in seconds (default 120)."),
                ]),
            ]),
            "required": .array([.string("command")]),
        ])
        return ChatRequest.Tool(
            type: "function",
            function: .init(
                name: "bash",
                description: "Run a shell command via /bin/zsh -c. Full filesystem and network access. Returns stdout, stderr, exit code.",
                parameters: schema
            )
        )
    }

    private var background: [UUID: Process] = [:]

    public init() {}

    /// Sentinel used to transport the post-command working directory
    /// back through stdout. Chosen to be astronomically unlikely to
    /// appear in real command output.
    private static let pwdMarker = "__VMLX_PWD_MARK_7f4c2b::"

    /// Wrap the user's script so the final working directory is captured
    /// on a guaranteed trailing line. The wrapper form is:
    ///
    ///     { <user command>; } 2>&1
    ///     __status=$?
    ///     printf '\n%s%s\n' "__VMLX_PWD_MARK_7f4c2b::" "$(pwd)"
    ///     exit $__status
    ///
    /// This preserves the user's exit code, includes cwd changes made
    /// by `cd` anywhere inside the block, and survives failed commands
    /// because `printf` runs unconditionally after the brace group.
    /// `set -e`-style users can opt out by passing their own preamble.
    private static func wrapCommand(_ command: String) -> String {
        return
            "{ \(command) ; } ; __vmlx_status=$? ; " +
            "printf '\\n%s%s\\n' '\(pwdMarker)' \"$(pwd)\" ; " +
            "exit $__vmlx_status"
    }

    /// Strip the trailing `__VMLX_PWD_MARK_...:<path>` sentinel out of
    /// stdout and return the recovered cwd alongside the cleaned text.
    /// Returns `nil` for the cwd when the marker isn't present (wrapper
    /// not installed, timed out before printf ran, etc.).
    private static func extractCwd(from stdout: String) -> (String, URL?) {
        guard let markerRange = stdout.range(
            of: pwdMarker, options: .backwards
        ) else {
            return (stdout, nil)
        }
        let pathStart = markerRange.upperBound
        let rest = stdout[pathStart...]
        // Take everything up to the next newline.
        let pathEnd = rest.firstIndex(of: "\n") ?? rest.endIndex
        let rawPath = String(rest[..<pathEnd]).trimmingCharacters(in: .whitespaces)
        // Clean: drop the preceding newline + marker line from stdout.
        // The marker is emitted as `\n<marker><path>\n` so the cleanup
        // start index is one character BEFORE the marker if that char
        // is a newline (otherwise just before the marker itself).
        var cleanEnd = markerRange.lowerBound
        if cleanEnd > stdout.startIndex {
            let prior = stdout.index(before: cleanEnd)
            if stdout[prior] == "\n" {
                cleanEnd = prior
            }
        }
        let cleaned = String(stdout[..<cleanEnd])
        let url = rawPath.isEmpty ? nil : URL(fileURLWithPath: rawPath)
        return (cleaned, url)
    }

    /// Synchronously run a command and return its result. Streams stdout/stderr
    /// internally and joins on completion (or timeout).
    public func run(_ inv: Invocation) async -> Result {
        let process = Process()
        process.launchPath = "/bin/zsh"
        // Wrap the user command so we can recover the post-exec `pwd`
        // from stdout. Skipped for background mode since the printf
        // tail only runs after the user script finishes.
        let scriptToRun = inv.runInBackground
            ? inv.command
            : Self.wrapCommand(inv.command)
        process.arguments = ["-c", scriptToRun]
        if let cwd = inv.cwd { process.currentDirectoryURL = cwd }

        let outPipe = Pipe(); let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError  = errPipe

        do {
            try process.run()
        } catch {
            return Result(stdout: "", stderr: "vMLX: failed to spawn /bin/zsh: \(error)",
                          exitCode: -1, timedOut: false, killed: false,
                          newCwd: nil)
        }

        // Run in background mode: register and return immediately.
        if inv.runInBackground {
            let id = UUID()
            background[id] = process
            return Result(stdout: "[backgrounded id=\(id.uuidString)]",
                          stderr: "", exitCode: 0, timedOut: false, killed: false,
                          newCwd: nil)
        }

        // Foreground with optional timeout. We poll because Process.waitUntilExit
        // blocks the actor; instead we yield in a Task and race the timeout.
        // Audit R2 (P2): also race a cancellation watcher that calls
        // process.terminate() when the parent task is cancelled. Without
        // this, clicking Stop while a long-running command (e.g. `find /`,
        // `sleep 100`) is in flight leaves the subprocess running until
        // it finishes naturally — wasted CPU + the user's stop button is
        // a lie. The timeout watcher at line 187 already had the right
        // pattern; we just add a third task that watches for cancellation
        // and terminates the subprocess too. Task.sleep is cancel-aware
        // so the watcher exits promptly when the parent is cancelled.
        let timeoutSec = inv.timeoutSeconds ?? 120
        let deadline = Date().addingTimeInterval(timeoutSec)
        var timedOut = false
        var killed = false

        await withTaskGroup(of: Void.self) { group in
            group.addTask { [process] in
                while process.isRunning {
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
            }
            group.addTask { [process] in
                while process.isRunning && Date() < deadline {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                if process.isRunning {
                    process.terminate()
                }
            }
            // R2: cancellation watcher. Polls Task.isCancelled and
            // terminates the subprocess on flip.
            group.addTask { [process] in
                while process.isRunning {
                    if Task.isCancelled {
                        if process.isRunning { process.terminate() }
                        return
                    }
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
            }
            await group.next()
            group.cancelAll()
        }
        if Task.isCancelled {
            killed = true
        }

        if Date() >= deadline && process.isRunning == false {
            timedOut = true
            killed = true
        }

        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let rawStdout = String(data: outData, encoding: .utf8) ?? ""
        let stderr = String(data: errData, encoding: .utf8) ?? ""

        // Recover the post-exec cwd from the trailing marker line
        // (see `wrapCommand`). `cleanStdout` is the user-facing stdout
        // with the marker line stripped; `recoveredCwd` is populated
        // when the command actually ran to completion.
        let (cleanStdout, recoveredCwd) = Self.extractCwd(from: rawStdout)

        return Result(stdout: cleanStdout,
                      stderr: stderr,
                      exitCode: process.terminationStatus,
                      timedOut: timedOut,
                      killed: killed,
                      newCwd: recoveredCwd)
    }

    /// Kill a backgrounded process by ID.
    public func kill(_ id: UUID) {
        background[id]?.terminate()
        background.removeValue(forKey: id)
    }
}
