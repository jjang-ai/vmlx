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
// Not yet handled (deferred to a later session):
//  - SSE transport (reads events from an HTTP stream)
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
    private var readTask: Task<Void, Never>?
    /// Buffered stdout bytes waiting for a newline.
    private var readBuffer = Data()

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
    public func start() throws {
        guard server.transport == .stdio else {
            throw MCPError.transportNotSupported(server.transport)
        }
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

    /// Configure pipes, run the subprocess, and start the read loop.
    /// Throws the underlying `proc.run()` error when launch fails so
    /// the caller's `MCPServerStatus.error` reflects the real reason
    /// instead of a stale "stdin pipe is nil" on the next `call()`.
    private func launchAndStoreStreams(_ proc: Process) throws {
        let stdin = Pipe()
        let stdout = Pipe()
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = Pipe()

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
        try sendNotification(method: "notifications/initialized", params: nil)
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
        try write(line)

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
    private func sendNotification(method: String, params: [String: Any]?) throws {
        var message: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method,
        ]
        if let params = params { message["params"] = params }
        let line = try makeLine(message)
        try write(line)
    }

    private func write(_ line: String) throws {
        guard let stdin = stdinPipe else {
            throw MCPError.processFailure(reason: "stdin pipe is nil")
        }
        guard let data = line.data(using: .utf8) else {
            throw MCPError.protocolError(reason: "message not UTF-8 encodable")
        }
        try stdin.fileHandleForWriting.write(contentsOf: data)
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
