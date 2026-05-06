// SPDX-License-Identifier: Apache-2.0
//
// Minimal JSON-RPC 2.0 client over stdio for MCP.
//
// Port of `vmlx_engine/mcp/client.py`. Handles:
//  - line-delimited JSON framing (one request or response per line)
//  - monotonic request IDs
//  - `initialize` handshake
//  - `tools/list` + `tools/call`
//  - response correlation via a per-ID continuation map
//  - graceful shutdown on process exit
//
// Now handled (mlxstudio#31):
//  - SSE transport (HTTP+SSE legacy MCP servers like Docker Toolkit / JetBrains)
//
// Not yet handled (deferred to a later session):
//  - bidirectional requests (server → client `sampling/createMessage`)
//  - notifications (one-way messages from server)
//
// The actor model isolates the request ID counter, pending-response
// table, and subprocess pipe handles so multiple concurrent `call`s
// are safe.

import Foundation

/// JSON-RPC 2.0 client actor for talking to a single MCP server.
///
/// Usage:
///     let client = MCPStdioClient(server: cfg)
///     try await client.start()
///     let info = try await client.initialize()
///     let tools = try await client.listTools()
///     let result = try await client.callTool(name: "echo", arguments: [:])
///     await client.stop()
public actor MCPStdioClient {

    // MARK: - Config

    public let server: MCPServerConfig

    // MARK: - Process state

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    /// Iter 144 — stderr pipe must be drained or the subprocess
    /// deadlocks at the ~64 KB kernel buffer cap. Held here so
    /// `stop()` can detach the readability handler before we let go
    /// of the Pipe (matches the stdoutPipe pattern).
    private var stderrPipe: Pipe?
    private var readTask: Task<Void, Never>?
    /// Buffered stdout bytes waiting for a newline.
    private var readBuffer = Data()

    // SSE transport state (mlxstudio#31). Active when server.transport == .sse.
    private var sseStreamTask: Task<Void, Never>?
    /// URL the server announces via the `endpoint` SSE event for client→
    /// server JSON-RPC POSTs. Falls back to the configured SSE URL if no
    /// endpoint event is observed (some servers accept POSTs to the same URL).
    private var sseSendURL: URL?
    /// One-shot continuation that resumes once the server announces its
    /// messages endpoint (or the stream EOF/errors). `start()` waits on this
    /// so callers don't issue requests before the channel is ready.
    private var sseEndpointReady: CheckedContinuation<Void, Error>?
    /// Buffered SSE event lines until the blank-line terminator.
    private var sseEventBuffer: [String] = []

    // MARK: - Protocol state

    private var nextId: Int = 1
    /// Continuations keyed on request id. Each entry resumes the caller
    /// of `send(_:)` when the matching response arrives.
    private var pending: [Int: CheckedContinuation<[String: Any], Error>] = [:]
    private(set) public var protocolVersion: String = "2024-11-05"
    private(set) public var serverInfo: [String: Any] = [:]

    // MARK: - Init

    public init(server: MCPServerConfig) {
        self.server = server
    }

    // MARK: - Lifecycle

    /// Start the subprocess and begin reading stdout. Must be called
    /// before `initialize()`. Throws if the process can't launch.
    public func start() async throws {
        switch server.transport {
        case .stdio:
            try startStdio()
        case .sse:
            try await startSSE()
        }
    }

    private func startStdio() throws {
        guard let command = server.command else {
            throw MCPError.configInvalid(reason: "stdio server has no command")
        }

        let proc = Process()
        // Resolve the command: absolute path → exec directly; bare name
        // → route through `/usr/bin/env` so PATH resolution applies.
        if command.hasPrefix("/") {
            proc.executableURL = URL(fileURLWithPath: command)
            proc.arguments = server.args ?? []
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = [command] + (server.args ?? [])
        }
        try launchAndStoreStreams(proc)
    }

    /// HTTP + SSE transport (mlxstudio#31). Implements the legacy MCP
    /// "HTTP+SSE" channel:
    ///   • GET <url>      — server-sent event stream of JSON-RPC frames.
    ///                      First event with `event: endpoint` carries the
    ///                      messages POST URL (with session id).
    ///   • POST <messages-url> — client→server JSON-RPC requests.
    ///
    /// We start the SSE stream task immediately, then wait up to
    /// `server.timeout` seconds for the `endpoint` event before declaring
    /// readiness. Servers that don't announce a separate POST URL keep
    /// using the original SSE URL — many simple bridges (Docker MCP
    /// Toolkit, single-endpoint servers) work this way.
    private func startSSE() async throws {
        guard let urlString = server.url, let url = URL(string: urlString) else {
            throw MCPError.configInvalid(reason: "sse server has no url")
        }
        sseSendURL = url  // default; overridden by `endpoint` event if present

        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue("no-cache", forHTTPHeaderField: "Cache-Control")
        // Iter 130 (vmlx#131): explicit `headers` dict — primary
        // source for SSE/HTTP auth (Authorization: Bearer, X-API-Key,
        // etc.). Required for Exa / GitHub / other auth-gated remote
        // MCP servers.
        if let headers = server.headers {
            for (k, v) in headers {
                req.setValue(v, forHTTPHeaderField: k)
            }
        }
        // Deprecated env-key scrape — kept for back-compat with
        // existing configs that stash auth headers in `env`. New
        // configs SHOULD use `headers`.
        if let overrides = server.env {
            for (k, v) in overrides where k.lowercased().hasPrefix("authorization") || k.hasPrefix("X-") || k.hasPrefix("x-") {
                req.setValue(v, forHTTPHeaderField: k)
            }
        }
        req.timeoutInterval = max(60, server.timeout)

        // Use bytes(for:) to get an AsyncBytes stream + line decoder.
        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            throw MCPError.processFailure(
                reason: "\(server.name) SSE GET \(url) returned HTTP \(http.statusCode)"
            )
        }

        sseStreamTask = Task { [weak self] in
            do {
                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    await self?.handleSSELine(line)
                }
            } catch {
                // Stream errored — clean up.
            }
            await self?.handleEOF()
        }

        // Wait for either the endpoint event or a short grace timeout.
        // Servers that don't announce an endpoint can be used immediately
        // (sseSendURL stays = url). Use a 2s grace, not the full server
        // timeout, since most servers announce within milliseconds.
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sseEndpointReady = cont
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await self?.fireEndpointReady()
            }
        }
    }

    private func fireEndpointReady() {
        if let cont = sseEndpointReady {
            sseEndpointReady = nil
            cont.resume()
        }
    }

    /// Parse one SSE protocol line. Buffers `event:`, `data:`, `id:` lines
    /// until a blank line, then dispatches the assembled event.
    private func handleSSELine(_ line: String) {
        if line.isEmpty {
            // End of an event — assemble and dispatch.
            var eventName = "message"
            var dataParts: [String] = []
            for raw in sseEventBuffer {
                if raw.hasPrefix("event:") {
                    eventName = String(raw.dropFirst("event:".count))
                        .trimmingCharacters(in: .whitespaces)
                } else if raw.hasPrefix("data:") {
                    var d = String(raw.dropFirst("data:".count))
                    if d.hasPrefix(" ") { d.removeFirst() }
                    dataParts.append(d)
                }
                // `id:` and `retry:` lines are ignored — JSON-RPC doesn't need them.
            }
            sseEventBuffer.removeAll(keepingCapacity: true)
            let payload = dataParts.joined(separator: "\n")
            switch eventName {
            case "endpoint":
                // Server announces the messages POST URL. Some servers send
                // an absolute URL; others a path relative to the SSE URL.
                if let resolved = resolveEndpoint(payload) {
                    sseSendURL = resolved
                }
                fireEndpointReady()
            case "message", "":
                // JSON-RPC payload — feed into the same dispatch the stdio
                // path uses. `appendAndDispatch` parses line-delimited JSON,
                // so terminate with `\n` to match.
                if let data = (payload + "\n").data(using: .utf8) {
                    appendAndDispatch(chunk: data)
                }
            case "ping":
                // Keepalive — ignore.
                break
            default:
                // Unknown event type — log via standard error path; treat
                // as message so we don't drop legitimate JSON-RPC.
                if let data = (payload + "\n").data(using: .utf8) {
                    appendAndDispatch(chunk: data)
                }
            }
            return
        }
        sseEventBuffer.append(line)
    }

    private func resolveEndpoint(_ raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let abs = URL(string: trimmed), abs.scheme != nil { return abs }
        // Relative — resolve against original SSE URL.
        if let base = server.url, let baseURL = URL(string: base) {
            return URL(string: trimmed, relativeTo: baseURL)?.absoluteURL
        }
        return nil
    }

    /// Configure pipes, run the subprocess, and start the read loop.
    /// Throws the underlying `proc.run()` error when launch fails so
    /// the caller's `MCPServerStatus.error` reflects the real reason
    /// instead of a stale "stdin pipe is nil" on the next `call()`.
    private func launchAndStoreStreams(_ proc: Process) throws {
        let stdin = Pipe()
        let stdout = Pipe()
        // Iter 144 — drain stderr or the subprocess deadlocks at ~64 KB.
        // Pre-fix the manager set `proc.standardError = Pipe()` but
        // never attached a `readabilityHandler`, so once the kernel
        // pipe buffer filled (~64 KB on macOS), the subprocess
        // BLOCKED on `write(stderr, ...)`. For a long-running MCP
        // server emitting routine warnings/debug, this is hours of
        // operation away from a hard hang. Drain to engine logs and
        // tag with the server name so `/admin/logs` shows the real
        // failure cause.
        let stderr = Pipe()
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stderr

        // Merge env with the caller's overrides. We keep the parent
        // environment as the baseline so stdio servers inherit PATH
        // and user locale automatically.
        var env = ProcessInfo.processInfo.environment
        if let overrides = server.env {
            for (k, v) in overrides { env[k] = v }
        }
        proc.environment = env

        do {
            try proc.run()
        } catch {
            throw MCPError.processFailure(
                reason: "\(server.name) failed to launch: \(error.localizedDescription)"
            )
        }

        self.process = proc
        self.stdinPipe = stdin
        self.stdoutPipe = stdout
        self.stderrPipe = stderr
        // Drain stderr — see comment at the Pipe() above for why.
        // Discard text quietly (most MCP servers write copious debug
        // to stderr); future enhancement could route to engine logs
        // gated on a verbose flag.
        let serverName = self.server.name
        stderr.fileHandleForReading.readabilityHandler = { fh in
            let chunk = fh.availableData
            if chunk.isEmpty { return }
            // Bound the per-line forwarding to avoid log spam from
            // pathological servers; truncate at 4 KB.
            let cap = min(chunk.count, 4096)
            if let s = String(data: chunk.prefix(cap), encoding: .utf8),
               !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                FileHandle.standardError.write(Data(
                    "[mcp \(serverName) stderr] \(s)\n".utf8))
            }
        }
        // Use `readabilityHandler` instead of a blocking read loop.
        // The previous impl called `handle.availableData` inside an
        // actor Task which blocked the actor's executor thread until
        // data or EOF arrived — meaning a quiet server would wedge
        // concurrent `call()` / `stop()` requests behind the read.
        // `readabilityHandler` fires on a dispatch queue for each
        // chunk so we can hop back into the actor without holding it.
        let handle = stdout.fileHandleForReading
        handle.readabilityHandler = { [weak self] fh in
            let chunk = fh.availableData
            Task { [weak self] in
                if chunk.isEmpty {
                    await self?.handleEOF()
                    return
                }
                await self?.appendAndDispatch(chunk: chunk)
            }
        }
        // Termination handler so EOF + subprocess exit both flow
        // through `handleEOF` from the same actor context.
        proc.terminationHandler = { [weak self] _ in
            Task { [weak self] in
                await self?.handleEOF()
            }
        }
    }

    /// Shut the subprocess down gracefully and release all pending
    /// continuations with a cancellation error.
    public func stop() {
        // Detach the readability handler BEFORE closing anything so we
        // don't get a dangling "readability fired on a released pipe"
        // crash from Dispatch.
        if let stdout = stdoutPipe {
            stdout.fileHandleForReading.readabilityHandler = nil
        }
        // Iter 144 — detach stderr handler before nilling the pipe;
        // same Dispatch-source-on-released-pipe crash class.
        if let stderr = stderrPipe {
            stderr.fileHandleForReading.readabilityHandler = nil
        }
        if let proc = process {
            proc.terminationHandler = nil
            if proc.isRunning {
                proc.terminate()
            }
        }
        readTask?.cancel()
        readTask = nil
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil

        // Cancel SSE stream task + clean up endpoint-readiness latch.
        sseStreamTask?.cancel()
        sseStreamTask = nil
        sseSendURL = nil
        sseEventBuffer.removeAll()
        if let cont = sseEndpointReady {
            sseEndpointReady = nil
            cont.resume(throwing: MCPError.processFailure(reason: "client stopped"))
        }

        // Fail anything still waiting.
        for (_, cont) in pending {
            cont.resume(throwing: MCPError.processFailure(reason: "client stopped"))
        }
        pending.removeAll()
    }

    // MARK: - Handshake

    /// Send the JSON-RPC `initialize` request to kick off the session.
    /// Caches `serverInfo` + `protocolVersion` for later.
    public func initialize() async throws {
        let params: [String: Any] = [
            "protocolVersion": protocolVersion,
            "capabilities": [:] as [String: Any],
            "clientInfo": [
                "name": "vmlx-swift",
                "version": "0.1.0",
            ] as [String: Any],
        ]
        let response = try await call(method: "initialize", params: params)
        if let info = response["serverInfo"] as? [String: Any] {
            self.serverInfo = info
        }
        if let version = response["protocolVersion"] as? String {
            self.protocolVersion = version
        }
        // Send the `initialized` notification so the server knows the
        // handshake is complete.
        try await sendNotification(method: "notifications/initialized", params: nil)
    }

    // MARK: - Tool operations

    /// `tools/list` — returns the full tool catalog from the server.
    public func listTools() async throws -> [MCPTool] {
        let response = try await call(method: "tools/list", params: nil)
        guard let rawTools = response["tools"] as? [[String: Any]] else {
            return []
        }
        return rawTools.compactMap { raw in
            guard let name = raw["name"] as? String else { return nil }
            let description = (raw["description"] as? String) ?? ""
            let schemaAny = raw["inputSchema"] ?? ["type": "object"]
            let schemaData = (try? JSONSerialization.data(
                withJSONObject: schemaAny
            )) ?? Data("{\"type\":\"object\"}".utf8)
            return MCPTool(
                serverName: server.name,
                name: name,
                description: description,
                inputSchemaJSON: schemaData
            )
        }
    }

    /// `tools/call` — execute a tool on this server.
    public func callTool(
        name: String,
        arguments: [String: Any]
    ) async throws -> MCPToolResult {
        let params: [String: Any] = [
            "name": name,
            "arguments": arguments,
        ]
        let response = try await call(method: "tools/call", params: params)

        // MCP responses have `content: [{type, text}, ...]` and an
        // `isError` boolean. We join all text parts into one string
        // for the simpler OpenAI-style `tool` message round-trip.
        let isError = (response["isError"] as? Bool) ?? false
        var joined = ""
        if let content = response["content"] as? [[String: Any]] {
            for block in content {
                if let text = block["text"] as? String {
                    if !joined.isEmpty { joined += "\n" }
                    joined += text
                }
            }
        }
        return MCPToolResult(toolName: name, content: joined, isError: isError)
    }

    // MARK: - Core call

    /// Send a JSON-RPC request and wait for its response. Thread-safe;
    /// multiple concurrent callers are ok because each gets its own
    /// unique id and matching continuation.
    public func call(
        method: String,
        params: [String: Any]?
    ) async throws -> [String: Any] {
        let id = nextId
        nextId += 1

        var message: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        ]
        if let params = params {
            message["params"] = params
        }

        let line = try makeLine(message)
        try await write(line)

        // Install a continuation and wait.
        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<[String: Any], Error>) in
            pending[id] = cont

            // Per-server timeout watchdog.
            let timeoutNs = UInt64(server.timeout * 1_000_000_000)
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: timeoutNs)
                await self?.timeout(id: id, method: method)
            }
        }
    }

    private func timeout(id: Int, method: String) {
        guard let cont = pending.removeValue(forKey: id) else { return }
        cont.resume(throwing: MCPError.timeout(server: server.name, method: method))
    }

    /// One-way notification — no response expected, so no continuation.
    private func sendNotification(method: String, params: [String: Any]?) async throws {
        var message: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method,
        ]
        if let params = params { message["params"] = params }
        let line = try makeLine(message)
        try await write(line)
    }

    private func write(_ line: String) async throws {
        switch server.transport {
        case .stdio:
            guard let stdin = stdinPipe else {
                throw MCPError.processFailure(reason: "stdin pipe is nil")
            }
            guard let data = line.data(using: .utf8) else {
                throw MCPError.protocolError(reason: "message not UTF-8 encodable")
            }
            try stdin.fileHandleForWriting.write(contentsOf: data)
        case .sse:
            guard let postURL = sseSendURL else {
                throw MCPError.processFailure(
                    reason: "SSE send URL not yet known (waiting for `endpoint` event)"
                )
            }
            // SSE servers expect a single JSON object per POST, no newline.
            let body = line.hasSuffix("\n") ? String(line.dropLast()) : line
            guard let data = body.data(using: .utf8) else {
                throw MCPError.protocolError(reason: "message not UTF-8 encodable")
            }
            var req = URLRequest(url: postURL)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
            req.httpBody = data
            req.timeoutInterval = max(30, server.timeout)
            // Iter 130 (vmlx#131): explicit headers dict — primary
            // source. Used by every POST so auth-gated servers stay
            // authenticated past the initial GET handshake.
            if let headers = server.headers {
                for (k, v) in headers {
                    req.setValue(v, forHTTPHeaderField: k)
                }
            }
            // Deprecated env-key scrape (back-compat).
            if let overrides = server.env {
                for (k, v) in overrides where k.lowercased().hasPrefix("authorization") || k.hasPrefix("X-") || k.hasPrefix("x-") {
                    req.setValue(v, forHTTPHeaderField: k)
                }
            }
            let (respData, response) = try await URLSession.shared.data(for: req)
            if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
                throw MCPError.processFailure(
                    reason: "SSE POST returned HTTP \(http.statusCode): \(String(data: respData, encoding: .utf8) ?? "")"
                )
            }
            // Some "Streamable HTTP" servers return the JSON-RPC reply
            // inline in the POST response (Content-Type application/json),
            // others reply only via the SSE stream. Feed the inline body
            // through appendAndDispatch when present so both shapes work.
            if !respData.isEmpty,
               let http = response as? HTTPURLResponse,
               (http.value(forHTTPHeaderField: "Content-Type") ?? "").contains("application/json")
            {
                var bytes = respData
                if bytes.last != 0x0a {
                    bytes.append(0x0a)  // ensure newline-terminated for line decoder
                }
                appendAndDispatch(chunk: bytes)
            }
        }
    }

    private func makeLine(_ message: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: message, options: [.sortedKeys]
        )
        guard let s = String(data: data, encoding: .utf8) else {
            throw MCPError.protocolError(reason: "JSON encode failed")
        }
        return s + "\n"
    }

    // MARK: - Read loop
    //
    // The actual read is driven by `FileHandle.readabilityHandler`
    // installed in `launchAndStoreStreams` — that fires on a dispatch
    // queue and hops into the actor via `Task { appendAndDispatch }`,
    // so this file no longer needs a blocking `availableData` loop.
    // `appendAndDispatch` + `processLine` + `handleEOF` are still
    // actor-isolated and re-used from both the readability callback
    // and the subprocess termination handler.

    private func appendAndDispatch(chunk: Data) {
        readBuffer.append(chunk)
        while let newlineIdx = readBuffer.firstIndex(of: 0x0A) {
            let lineData = readBuffer.subdata(in: 0..<newlineIdx)
            readBuffer.removeSubrange(0...newlineIdx)
            guard !lineData.isEmpty else { continue }
            processLine(lineData)
        }
    }

    private func processLine(_ data: Data) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        // We only handle responses (have an "id"). Notifications from
        // the server (missing "id") are ignored for now.
        guard let id = obj["id"] as? Int, let cont = pending.removeValue(forKey: id) else {
            return
        }
        if let err = obj["error"] as? [String: Any] {
            let msg = (err["message"] as? String) ?? "unknown error"
            cont.resume(throwing: MCPError.protocolError(reason: msg))
            return
        }
        if let result = obj["result"] as? [String: Any] {
            cont.resume(returning: result)
        } else {
            // Result could be a scalar / array / null — wrap for uniform return.
            cont.resume(returning: ["result": obj["result"] ?? NSNull()])
        }
    }

    private func handleEOF() {
        for (_, cont) in pending {
            cont.resume(throwing: MCPError.processFailure(reason: "subprocess exited"))
        }
        pending.removeAll()
        if let proc = process, proc.isRunning {
            proc.terminate()
        }
        process = nil
    }
}
