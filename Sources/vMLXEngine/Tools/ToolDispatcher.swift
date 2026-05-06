import Foundation

/// Server-side tool dispatcher. When `Engine.stream` emits a tool call
/// (via Harmony-style `.toolCall(_)` or the streaming-accumulator buffer),
/// the HTTP route layer calls `Engine.executeToolCall(_:cwd:)` to actually
/// run the tool and produce a `tool` role message that gets appended to
/// the conversation for the next turn.
///
/// Terminal mode always registers `bash` as an available tool. Future
/// tools (web search, fetch URL, file read/write, etc.) plug in here
/// alongside `bash` — the dispatcher switches on `function.name`.
///
/// This file is the "trusted" side of the tool sandbox: `bash` gives the
/// model full shell access, so the caller MUST have gated it on user
/// acknowledgment (the Terminal mode first-run warning) before invoking.
public struct ToolDispatchResult: Sendable {
    public var toolCallId: String
    public var name: String
    public var content: String   // the JSON-or-text body to feed back as a tool message
    public var isError: Bool
    public init(toolCallId: String, name: String, content: String, isError: Bool = false) {
        self.toolCallId = toolCallId
        self.name = name
        self.content = content
        self.isError = isError
    }
}

extension Engine {

    /// Execute a tool call server-side and return the result formatted as
    /// a tool-message payload ready to append to `ChatRequest.messages`.
    ///
    /// Dispatch order:
    /// 1. `bash` — in-process shell tool
    /// 2. `server__tool` (contains `__`) — MCP tool via `engine.mcp.executeTool`
    /// 3. Everything else — error
    public func executeToolCall(
        _ call: ChatRequest.ToolCall,
        cwd: URL? = nil,
        timeoutSeconds: Double = 120
    ) async -> ToolDispatchResult {
        await log(.info, "tool", "executing \(call.function.name) id=\(call.id)")

        let name = call.function.name
        // In-process bash tool wins over any MCP server that happens
        // to expose a tool literally named "bash". Falls back to the
        // engine's persistent `terminalCwd` when the caller doesn't
        // supply an explicit cwd — that's how Terminal mode threads
        // `cd foo` across calls.
        if name == "bash" {
            let effectiveCwd = cwd ?? terminalCwd
            return await executeBashTool(call, cwd: effectiveCwd, timeoutSeconds: timeoutSeconds)
        }

        // §429 — VL screenshot tool. Captures the screen to a temp PNG
        // and returns the path. Terminal then auto-attaches the PNG
        // as an `image_url` content part on the next user message so
        // the VL model actually SEES the pixels.
        if name == "screenshot" {
            return await executeScreenshotTool(call)
        }

        // §431 — Headless WKWebView browser tool for VL agents. Same
        // rendezvous as `screenshot` (drains via consumeLatestScreenshots
        // after the agentic loop). The model gets a stateful browser
        // it can drive iteratively — open, click, type, scroll —
        // and SEES the page after every action.
        if name == "browser" {
            return await executeBrowserTool(call)
        }

        // MCP namespaced names are `server__tool`. Route through the
        // MCPServerManager which starts the server lazily on first use.
        if name.contains("__") {
            return await executeMCPTool(call)
        }

        await log(.warn, "tool", "unknown tool: \(name)")
        return ToolDispatchResult(
            toolCallId: call.id,
            name: name,
            content: "{\"error\":\"Unknown tool '\(name)'\"}",
            isError: true
        )
    }

    /// Dispatch a namespaced `server__tool` call to the running MCP
    /// server. Parses the model's JSON `arguments` string into a
    /// `[String: Any]` for the JSON-RPC 2.0 `tools/call` request.
    private func executeMCPTool(
        _ call: ChatRequest.ToolCall
    ) async -> ToolDispatchResult {
        let argsData = call.function.arguments.data(using: .utf8) ?? Data()
        let rawArguments = (try? JSONSerialization.jsonObject(with: argsData))
            as? [String: Any] ?? [:]
        // §338 (vmlx#47) — LLMs occasionally emit tool arguments with
        // numeric values wrapped in quotes (`{"page":"3"}` instead of
        // `{"page":3}`). MCP servers that do strict JSON-Schema
        // validation reject these as `Invalid arguments`. Walk the
        // tool's declared `inputSchema` and coerce numeric/bool
        // strings to their declared leaf types before dispatch.
        // Non-destructive: values the schema doesn't cover or that
        // don't cleanly coerce pass through unchanged.
        var arguments = rawArguments
        if let tool = await self.mcp.findTool(namespaced: call.function.name),
           let schema = (try? JSONSerialization.jsonObject(
                with: tool.inputSchemaJSON)) as? [String: Any]
        {
            arguments = coerceToolArguments(rawArguments, schema: schema)
        }
        do {
            let result = try await self.mcp.executeTool(
                namespaced: call.function.name,
                arguments: arguments
            )
            await log(.info, "tool",
                "mcp tool \(call.function.name) ok (\(result.content.count) chars)")
            return ToolDispatchResult(
                toolCallId: call.id,
                name: call.function.name,
                content: result.content,
                isError: result.isError
            )
        } catch {
            await log(.warn, "tool",
                "mcp tool \(call.function.name) failed: \(error)")
            let errJSON = "{\"error\":\"\(String(describing: error).replacingOccurrences(of: "\"", with: "\\\""))\"}"
            return ToolDispatchResult(
                toolCallId: call.id,
                name: call.function.name,
                content: errJSON,
                isError: true
            )
        }
    }

    /// §431 — Run the `browser` tool. Drives a persistent headless
    /// WKWebView the VL model can interact with. After every action
    /// the post-action snapshot is staged on the engine via
    /// `recordScreenshot(path:)` so the Terminal auto-continue path
    /// attaches it to the model's next user message.
    private func executeBrowserTool(
        _ call: ChatRequest.ToolCall
    ) async -> ToolDispatchResult {
        #if canImport(WebKit) && canImport(AppKit)
        let argsData = call.function.arguments.data(using: .utf8) ?? Data()
        let argsJson = (try? JSONSerialization.jsonObject(with: argsData)) as? [String: Any] ?? [:]

        let action = (argsJson["action"] as? String) ?? "screenshot"
        var inv = BrowserTool.Invocation(action: action)
        if let s = argsJson["url"] as? String { inv.url = s }
        if let s = argsJson["selector"] as? String { inv.selector = s }
        if let s = argsJson["text"] as? String { inv.text = s }
        if let s = argsJson["script"] as? String { inv.script = s }
        if let n = argsJson["x"] as? Double { inv.x = n }
        if let n = argsJson["x"] as? Int { inv.x = Double(n) }
        if let n = argsJson["y"] as? Double { inv.y = n }
        if let n = argsJson["y"] as? Int { inv.y = Double(n) }
        if let n = argsJson["delta_y"] as? Double { inv.deltaY = n }
        if let n = argsJson["delta_y"] as? Int { inv.deltaY = Double(n) }
        if let b = argsJson["visible"] as? Bool { inv.visible = b }
        if let n = argsJson["width"] as? Int { inv.width = n }
        if let n = argsJson["height"] as? Int { inv.height = n }

        let tool = await self.browserToolInstance()
        let result = await tool.run(inv)

        if let path = result.screenshotPath {
            await self.recordScreenshot(path: path)
        }

        var dict: [String: Any] = [
            "action": action,
            "ok": result.error == nil,
        ]
        if let u = result.pageURL { dict["page_url"] = u }
        if let t = result.pageTitle { dict["page_title"] = t }
        if let p = result.screenshotPath {
            dict["screenshot_path"] = p.path
            if let w = result.widthHint { dict["width"] = w }
            if let h = result.heightHint { dict["height"] = h }
            dict["note"] = "PNG attached to your next input — describe what you see and decide the next action."
        }
        if let r = result.evalResult { dict["eval_result"] = r }
        if let err = result.error { dict["error"] = err }

        let body = (try? JSONSerialization.data(withJSONObject: dict))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        return ToolDispatchResult(
            toolCallId: call.id,
            name: "browser",
            content: body,
            isError: result.error != nil
        )
        #else
        return ToolDispatchResult(
            toolCallId: call.id, name: "browser",
            content: "{\"error\":\"browser tool requires macOS WebKit/AppKit\"}",
            isError: true)
        #endif
    }

    /// §429 — Run the `screenshot` tool. Captures the screen via
    /// /usr/sbin/screencapture, persists the path on the engine actor
    /// so the Terminal UI can pick it up after the agentic loop ends
    /// and attach the image to the next user turn for VL inference.
    private func executeScreenshotTool(
        _ call: ChatRequest.ToolCall
    ) async -> ToolDispatchResult {
        let argsData = call.function.arguments.data(using: .utf8) ?? Data()
        let argsJson = (try? JSONSerialization.jsonObject(with: argsData)) as? [String: Any] ?? [:]

        var inv = ScreenshotTool.Invocation()
        if let region = argsJson["region"] as? [Any] {
            let ints = region.compactMap { ($0 as? Int) ?? Int($0 as? Double ?? .nan) }
            if ints.count == 4 { inv.region = ints }
        }
        if let d = argsJson["delay"] as? Double { inv.delaySeconds = d }
        if let d = argsJson["delay"] as? Int { inv.delaySeconds = Double(d) }
        if let t = argsJson["target"] as? String { inv.target = t }

        let tool = ScreenshotTool()
        let result = await tool.run(inv)

        // Persist the path on the engine actor so the chat session
        // (typically TerminalScreen.runViaEngine) can pick it up after
        // the engine.stream() agentic loop concludes and attach the
        // PNG as an image_url part on the next user message.
        if result.error == nil {
            await self.recordScreenshot(path: result.path)
        }

        // Format result as JSON. The model sees this as the tool
        // message body — keep it concise.
        var dict: [String: Any] = [
            "saved": result.error == nil,
            "path": result.path.path,
            "bytes": result.savedBytes,
        ]
        if let w = result.widthHint { dict["width"] = w }
        if let h = result.heightHint { dict["height"] = h }
        if let err = result.error {
            dict["error"] = err
        } else {
            dict["note"] = "PNG saved. The image will be attached to your next input so you can SEE it. Don't try to call screenshot again unless the user requests a new capture."
        }
        let body = (try? JSONSerialization.data(withJSONObject: dict))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        return ToolDispatchResult(
            toolCallId: call.id,
            name: "screenshot",
            content: body,
            isError: result.error != nil
        )
    }

    /// Run the `bash` tool. Parses the `command` field out of the JSON
    /// arguments, dispatches to the shared `BashTool` actor, and formats
    /// the result as a JSON blob.
    private func executeBashTool(
        _ call: ChatRequest.ToolCall,
        cwd: URL?,
        timeoutSeconds: Double
    ) async -> ToolDispatchResult {
        // Parse arguments: expected `{"command": "...", "cwd": "...", "timeout": 60}`
        let argsData = call.function.arguments.data(using: .utf8) ?? Data()
        let argsJson = (try? JSONSerialization.jsonObject(with: argsData)) as? [String: Any] ?? [:]
        guard let command = argsJson["command"] as? String else {
            return ToolDispatchResult(
                toolCallId: call.id, name: "bash",
                content: "{\"error\":\"missing 'command' argument\"}",
                isError: true)
        }
        let effectiveCwd: URL? = {
            if let cwdString = argsJson["cwd"] as? String {
                return URL(fileURLWithPath: cwdString)
            }
            return cwd
        }()
        let effectiveTimeout = (argsJson["timeout"] as? Double) ?? timeoutSeconds

        // Dispatch to the shared BashTool actor.
        let bash = BashTool()
        let result = await bash.run(.init(
            command: command,
            cwd: effectiveCwd,
            timeoutSeconds: effectiveTimeout,
            runInBackground: false
        ))

        // Persist any post-exec cwd change back to the engine actor
        // so the NEXT bash invocation in this chat starts where the
        // previous one ended (cd foo / pushd / etc). `updateTerminalCwd`
        // is a no-op when `newCwd` is nil or matches the current value.
        updateTerminalCwd(result.newCwd)

        // Format the result as a JSON body. Keep it concise — the model
        // has to read this back and its context window is precious.
        let maxBytes = 16_384
        let stdout = truncate(result.stdout, maxBytes: maxBytes / 2)
        let stderr = truncate(result.stderr, maxBytes: maxBytes / 2)
        var payload: [String: Any] = [
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": Int(result.exitCode),
        ]
        if result.timedOut { payload["timed_out"] = true }
        if result.killed { payload["killed"] = true }

        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
        let body = String(data: data, encoding: .utf8) ?? "{}"

        await log(.info, "tool",
            "bash exit=\(result.exitCode) stdout=\(stdout.count)B stderr=\(stderr.count)B")

        return ToolDispatchResult(
            toolCallId: call.id,
            name: "bash",
            content: body,
            isError: result.exitCode != 0 || result.killed
        )
    }

    /// Append a tool result to the chat history and build a follow-up
    /// request that the server can pass back to `engine.stream(request:)`
    /// to continue the generation loop.
    public nonisolated func appendToolResult(
        to request: ChatRequest,
        result: ToolDispatchResult
    ) -> ChatRequest {
        var newMessages = request.messages
        // First, ensure the assistant's tool_call message is represented.
        // The caller is expected to have already appended the assistant
        // message with `toolCalls` set; we just add the tool response.
        //
        // Iter 144 — frame errors with an explicit "Error:" prefix so
        // the model can recover instead of treating the error like a
        // successful return value. OpenAI's spec puts `is_error` in
        // tool-message metadata, but the chat-template render only
        // sees `content` for most templates, so prefixing is the
        // template-agnostic guardrail. The reviewer audit caught that
        // `result.isError` was set but never surfaced to the model;
        // a tool that returned `{"error":"timeout"}` would be treated
        // as success and the model would keep iterating.
        let framedContent: String =
            result.isError
                ? "Error: \(result.content)"
                : result.content
        newMessages.append(ChatRequest.Message(
            role: "tool",
            content: .string(framedContent),
            name: result.name,
            toolCalls: nil,
            toolCallId: result.toolCallId
        ))
        return ChatRequest(
            model: request.model,
            messages: newMessages,
            stream: request.stream,
            maxTokens: request.maxTokens,
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            repetitionPenalty: request.repetitionPenalty,
            stop: request.stop,
            seed: request.seed,
            enableThinking: request.enableThinking,
            reasoningEffort: request.reasoningEffort,
            tools: request.tools,
            toolChoice: request.toolChoice
        )
    }

    /// Truncate a string to a max byte count, preserving UTF-8 validity.
    private func truncate(_ s: String, maxBytes: Int) -> String {
        let bytes = s.utf8
        guard bytes.count > maxBytes else { return s }
        // Walk back from maxBytes to find a valid UTF-8 boundary.
        let data = Data(bytes.prefix(maxBytes))
        if let str = String(data: data, encoding: .utf8) {
            return str + "\n...[truncated]"
        }
        // Fall back to character-level truncation.
        return String(s.prefix(maxBytes / 4)) + "\n...[truncated]"
    }
}
