import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine

/// Anthropic Messages API.
///
/// Python source: `vmlx_engine/server.py:2664 create_anthropic_message`.
/// Streaming SSE format:
///   event: message_start      data: {...}
///   event: content_block_start data: {...}
///   event: content_block_delta data: {"delta":{"type":"text_delta","text":"Hi"}}
///   event: content_block_stop
///   event: message_delta       data: {"delta":{"stop_reason":"end_turn"}, "usage":{...}}
///   event: message_stop
///
/// Supports `thinking_budget`, `tool_use` blocks, vision content blocks (image source).
public enum AnthropicRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        router.post("/v1/messages") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let anthropicBody = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid JSON")
            }
            let model = (anthropicBody["model"] as? String) ?? "default"
            let isStream = (anthropicBody["stream"] as? Bool) ?? false

            // Translate Anthropic → ChatRequest. Full logic: server.py:2664.
            let chatReq = Self.anthropicToChatRequest(anthropicBody)
            // 2026-04-18 validate-parity fix — OpenAI + gateway paths
            // call `chatReq.validate()` before streaming; Anthropic was
            // silently skipping the range check, so a wild temperature=99
            // or negative max_tokens hit the engine as HTTP 200 instead
            // of a clean 400. Live-triggered by the production audit.
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return OpenAIRoutes.errorJSON(.badRequest, err.description)
            } catch {
                return OpenAIRoutes.errorJSON(.badRequest, "invalid request: \(error)")
            }
            await engine.wakeFromStandby()
            // 4-tier settings resolution happens inside `Engine.stream`.
            let upstream = await engine.stream(request: chatReq)

            if isStream {
                var headers: HTTPFields = [:]
                headers[.contentType] = "text/event-stream; charset=utf-8"
                headers[.cacheControl] = "no-cache"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: Self.streamingBody(model: model, upstream: upstream)
                )
            }

            // Non-streaming — collect reasoning, content, and tool calls
            // into Anthropic's content-block array so clients see the same
            // shape they'd get from `client.messages.create(stream=False)`.
            var thinking = ""
            var content = ""
            var toolCalls: [ChatRequest.ToolCall] = []
            var usage: StreamChunk.Usage? = nil
            var finishReason: String? = nil
            do {
                for try await chunk in upstream {
                    if let r = chunk.reasoning { thinking += r }
                    if let c = chunk.content { content += c }
                    if let tcs = chunk.toolCalls { toolCalls.append(contentsOf: tcs) }
                    if let u = chunk.usage { usage = u }
                    if let fr = chunk.finishReason { finishReason = fr }
                }
            } catch let err as EngineError {
                return OpenAIRoutes.mapEngineError(err)
            } catch {
                return OpenAIRoutes.errorJSON(.internalServerError, "\(error)")
            }
            var blocks: [[String: Any]] = []
            if !thinking.isEmpty {
                blocks.append([
                    "type": "thinking",
                    "thinking": thinking,
                ] as [String: Any])
            }
            if !content.isEmpty || (blocks.isEmpty && toolCalls.isEmpty) {
                // Ensure at least one text block when no reasoning/tools.
                blocks.append([
                    "type": "text",
                    "text": content,
                ] as [String: Any])
            }
            for tc in toolCalls {
                // Parse `arguments` JSON string into a dict for `input`.
                let argsData = tc.function.arguments.data(using: .utf8) ?? Data("{}".utf8)
                let input = (try? JSONSerialization.jsonObject(with: argsData))
                    as? [String: Any] ?? [:]
                blocks.append([
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": input,
                ] as [String: Any])
            }
            let obj: [String: Any] = [
                "id": "msg_\(UUID().uuidString.prefix(12).lowercased())",
                "type": "message",
                "role": "assistant",
                "content": blocks,
                "model": model,
                "stop_reason": Self.mapStopReason(finishReason),
                "usage": Self.usageEnvelope(usage),
            ]
            return OpenAIRoutes.json(obj)
        }
    }

    static func streamingBody(
        model: String,
        upstream: AsyncThrowingStream<StreamChunk, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()
            let messageId = "msg_\(UUID().uuidString.prefix(12).lowercased())"

            func emit(_ event: String, _ data: [String: Any]) async throws {
                let json = SSEEncoder.asciiJSON(data)
                let frame = "event: \(event)\ndata: \(json)\n\n"
                var b = allocator.buffer(capacity: frame.utf8.count)
                b.writeString(frame)
                try await writer.write(b)
            }

            try await emit("message_start", [
                "type": "message_start",
                "message": [
                    "id": messageId,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": NSNull(),
                    "usage": ["input_tokens": 0, "output_tokens": 0] as [String: Any],
                ] as [String: Any],
            ])

            // Lazy, multi-block output. Anthropic's extended-thinking wire
            // format uses separate `content_block`s (each with its own
            // `index`) for text, thinking, and tool_use. Open each block
            // only when we see its first relevant chunk, and stop it
            // before opening the next block type. This mirrors
            // `vmlx_engine/api/anthropic_adapter.py::stream_anthropic_events`.
            enum OpenBlock { case none, thinking(Int), text(Int), toolUse(Int, String) }
            var blockIndex = -1
            var open: OpenBlock = .none

            func stopOpen() async throws {
                switch open {
                case .none: return
                case .thinking(let i), .text(let i), .toolUse(let i, _):
                    try await emit("content_block_stop", [
                        "type": "content_block_stop",
                        "index": i,
                    ])
                }
                open = .none
            }

            func startThinking() async throws {
                try await stopOpen()
                blockIndex += 1
                try await emit("content_block_start", [
                    "type": "content_block_start",
                    "index": blockIndex,
                    "content_block": ["type": "thinking", "thinking": ""] as [String: Any],
                ])
                open = .thinking(blockIndex)
            }

            func startText() async throws {
                try await stopOpen()
                blockIndex += 1
                try await emit("content_block_start", [
                    "type": "content_block_start",
                    "index": blockIndex,
                    "content_block": ["type": "text", "text": ""] as [String: Any],
                ])
                open = .text(blockIndex)
            }

            func startToolUse(id: String, name: String) async throws {
                try await stopOpen()
                blockIndex += 1
                try await emit("content_block_start", [
                    "type": "content_block_start",
                    "index": blockIndex,
                    "content_block": [
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": [:] as [String: Any],
                    ] as [String: Any],
                ])
                open = .toolUse(blockIndex, id)
            }

            var finishReason: String? = nil
            var usage: StreamChunk.Usage? = nil
            // Iter-18: wrap the upstream with the shared heartbeat
            // helper so thinking-model prefill doesn't trigger a
            // nginx / SDK idle-timeout disconnect. Anthropic SSE uses
            // named events; emit the `ping` event (Anthropic's own
            // keep-alive convention, documented in their streaming
            // spec) so the anthropic-sdk-python/node parsers don't
            // complain about unknown event names.
            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)
            do {
                for try await event in merged {
                    if case .heartbeat = event {
                        try await emit("ping", [
                            "type": "ping",
                        ])
                        continue
                    }
                    guard case .chunk(let chunk) = event else { continue }
                    if let r = chunk.reasoning, !r.isEmpty {
                        switch open {
                        case .thinking: break
                        default: try await startThinking()
                        }
                        if case .thinking(let i) = open {
                            try await emit("content_block_delta", [
                                "type": "content_block_delta",
                                "index": i,
                                "delta": ["type": "thinking_delta", "thinking": r] as [String: Any],
                            ])
                        }
                    }
                    if let c = chunk.content, !c.isEmpty {
                        switch open {
                        case .text: break
                        default: try await startText()
                        }
                        if case .text(let i) = open {
                            try await emit("content_block_delta", [
                                "type": "content_block_delta",
                                "index": i,
                                "delta": ["type": "text_delta", "text": c] as [String: Any],
                            ])
                        }
                    }
                    if let tcs = chunk.toolCalls, !tcs.isEmpty {
                        // Tool-use streaming: Anthropic wire format wants ONE
                        // `content_block_start` + N `input_json_delta`s + ONE
                        // `content_block_stop` per logical tool call. If the
                        // model's tool_call arguments arrive in multiple
                        // chunks for the SAME `tc.id`, we must NOT re-open
                        // the block every chunk — deep audit #8 flagged that
                        // the original code emitted one start/stop pair per
                        // delta, which SDK clients reject as malformed.
                        for tc in tcs {
                            let alreadyOpen: Bool
                            if case .toolUse(_, let openId) = open, openId == tc.id {
                                alreadyOpen = true
                            } else {
                                alreadyOpen = false
                            }
                            if !alreadyOpen {
                                try await startToolUse(id: tc.id, name: tc.function.name)
                            }
                            if case .toolUse(let i, _) = open,
                               !tc.function.arguments.isEmpty
                            {
                                try await emit("content_block_delta", [
                                    "type": "content_block_delta",
                                    "index": i,
                                    "delta": [
                                        "type": "input_json_delta",
                                        "partial_json": tc.function.arguments,
                                    ] as [String: Any],
                                ])
                            }
                        }
                    }
                    if let fr = chunk.finishReason { finishReason = fr }
                    if let u = chunk.usage { usage = u }
                }
            } catch {
                try await stopOpen()
                try await emit("error", [
                    "type": "error",
                    "error": ["type": "api_error", "message": "\(error)"] as [String: Any],
                ])
                try await writer.finish(nil)
                return
            }

            // Ensure at least one text block exists so downstream Anthropic
            // parsers don't choke on an empty message. Mirrors server.py.
            if blockIndex < 0 {
                try await startText()
            }
            try await stopOpen()
            try await emit("message_delta", [
                "type": "message_delta",
                "delta": ["stop_reason": Self.mapStopReason(finishReason)] as [String: Any],
                "usage": Self.usageEnvelope(usage),
            ])
            try await emit("message_stop", ["type": "message_stop"])
            try await writer.finish(nil)
        }
    }

    static func mapStopReason(_ finish: String?) -> String {
        switch finish {
        case "length": return "max_tokens"
        case "tool_calls": return "tool_use"
        default: return "end_turn"
        }
    }

    /// **iter-65 (§94)** — Anthropic usage envelope with optional timing
    /// fields. Anthropic's published schema requires `input_tokens` +
    /// `output_tokens`; extra keys are tolerated by `anthropic-sdk-*`
    /// parsers (both Python and TypeScript SDKs dispatch on known keys
    /// and ignore the rest). We add the same `tokens_per_second`,
    /// `ttft_ms`, `prefill_ms`, `total_ms` quartet that OpenAI (iter-64)
    /// and Ollama (iter-63) envelopes already carry, so every API
    /// surface emits comparable timings for observability parity.
    ///
    /// `includeInputTokens` gates whether `input_tokens` is emitted.
    /// The streaming `message_delta` final event historically carried
    /// only `output_tokens` (Anthropic's own spec — the client already
    /// has `input_tokens` from `message_start`), but iter-65 includes
    /// it there too because clients observed the `message_start` value
    /// was a stub `0` until usage was known. Emitting the real value
    /// on `message_delta` keeps the stream truthful without breaking
    /// the non-stream contract.
    static func usageEnvelope(
        _ usage: StreamChunk.Usage?,
        includeInputTokens: Bool = true
    ) -> [String: Any] {
        var out: [String: Any] = [
            "output_tokens": usage?.completionTokens ?? 0,
        ]
        if includeInputTokens {
            out["input_tokens"] = usage?.promptTokens ?? 0
        }
        guard let u = usage else { return out }
        if let tps = u.tokensPerSecond { out["tokens_per_second"] = tps }
        if let ttft = u.ttftMs { out["ttft_ms"] = ttft }
        if let prefill = u.prefillMs { out["prefill_ms"] = prefill }
        if let total = u.totalMs { out["total_ms"] = total }
        return out
    }

    /// Minimal Anthropic → ChatRequest. Full translation: server.py:2664 create_anthropic_message.
    static func anthropicToChatRequest(_ body: [String: Any]) -> ChatRequest {
        var messages: [ChatRequest.Message] = []
        // iter-106 §184: Anthropic's Messages API accepts `system`
        // in two shapes. (1) A bare string — the common case for
        // hand-rolled curl + most SDK quickstarts. (2) An array of
        // `{type: "text", text: "...", cache_control?}` blocks —
        // this is the shape Anthropic's Python + TypeScript SDKs
        // emit when prompt caching is enabled, and we were silently
        // dropping it (the old `as? String` cast returned nil on
        // arrays, no fallback, the system prompt evaporated).
        // Accept both shapes; for the array form, concatenate text
        // blocks in order and drop cache_control markers (vMLX's
        // own prefix cache keys on prompt text, not Anthropic's
        // cache_control contract, so the markers have nothing to
        // bind to on our side).
        if let system = body["system"] as? String {
            messages.append(ChatRequest.Message(
                role: "system",
                content: .string(system),
                name: nil, toolCalls: nil, toolCallId: nil
            ))
        } else if let systemBlocks = body["system"] as? [[String: Any]] {
            var flat = ""
            for block in systemBlocks {
                let type = (block["type"] as? String) ?? "text"
                guard type == "text" else { continue }
                if let text = block["text"] as? String, !text.isEmpty {
                    if !flat.isEmpty { flat += "\n" }
                    flat += text
                }
            }
            if !flat.isEmpty {
                messages.append(ChatRequest.Message(
                    role: "system",
                    content: .string(flat),
                    name: nil, toolCalls: nil, toolCallId: nil
                ))
            }
        }
        if let raw = body["messages"] as? [[String: Any]] {
            for m in raw {
                let role = (m["role"] as? String) ?? "user"
                let contentVal: ChatRequest.ContentValue?
                // Collected `tool_use` blocks are lifted to the parent
                // assistant message's `toolCalls` field. `tool_result`
                // blocks are split out into their own role=tool messages.
                var collectedToolCalls: [ChatRequest.ToolCall] = []
                var collectedToolResults: [(id: String, text: String)] = []
                if let s = m["content"] as? String {
                    contentVal = .string(s)
                } else if let parts = m["content"] as? [[String: Any]] {
                    // Anthropic content blocks → OpenAI-style parts.
                    //
                    // Anthropic wire format for a vision block is:
                    //   {"type":"image","source":{"type":"base64",
                    //    "media_type":"image/jpeg","data":"<b64>"}}
                    // plus `{"type":"text","text":"..."}` blocks.
                    //
                    // We convert each block into a `ChatRequest.ContentPart`
                    // so `Stream.extractImages` + the VLM pipeline see the
                    // image. Previously every image block was silently
                    // dropped and text-only was concatenated — Anthropic
                    // VLM clients got a blank response with no indication
                    // why.
                    var converted: [ChatRequest.ContentPart] = []
                    for block in parts {
                        let type = block["type"] as? String ?? ""
                        switch type {
                        case "text":
                            if let t = block["text"] as? String, !t.isEmpty {
                                converted.append(ChatRequest.ContentPart(
                                    type: "text",
                                    text: t,
                                    imageUrl: nil
                                ))
                            }
                        case "image":
                            // Two Anthropic source shapes: base64 (inline)
                            // and url (external).
                            if let source = block["source"] as? [String: Any] {
                                let sourceType = source["type"] as? String ?? ""
                                if sourceType == "base64",
                                   let data = source["data"] as? String
                                {
                                    let mediaType = (source["media_type"] as? String)
                                        ?? "image/png"
                                    // OpenAI-style `data:` URL so downstream
                                    // `extractImages` can decode it.
                                    let dataURL = "data:\(mediaType);base64,\(data)"
                                    converted.append(ChatRequest.ContentPart(
                                        type: "image_url",
                                        text: nil,
                                        imageUrl: .init(url: dataURL)
                                    ))
                                } else if sourceType == "url",
                                          let url = source["url"] as? String
                                {
                                    converted.append(ChatRequest.ContentPart(
                                        type: "image_url",
                                        text: nil,
                                        imageUrl: .init(url: url)
                                    ))
                                }
                            }
                        case "tool_use":
                            // Assistant-side tool invocation. Anthropic:
                            //   {"type":"tool_use","id":"toolu_x",
                            //    "name":"get_weather","input":{...}}
                            if let id = block["id"] as? String,
                               let name = block["name"] as? String
                            {
                                let inputDict = block["input"] as? [String: Any] ?? [:]
                                let argsData = (try? JSONSerialization.data(
                                    withJSONObject: inputDict)) ?? Data("{}".utf8)
                                let argsJSON = String(data: argsData, encoding: .utf8) ?? "{}"
                                // Build via Codable since ToolCall.Function
                                // has no public memberwise init.
                                let wrapped: [String: Any] = [
                                    "id": id,
                                    "type": "function",
                                    "function": [
                                        "name": name,
                                        "arguments": argsJSON,
                                    ] as [String: Any],
                                ]
                                if let data = try? JSONSerialization.data(withJSONObject: wrapped),
                                   let call = try? JSONDecoder().decode(
                                    ChatRequest.ToolCall.self, from: data)
                                {
                                    collectedToolCalls.append(call)
                                }
                            }
                        case "tool_result":
                            // User-side tool output. Anthropic content can
                            // be either a plain string or an array of text
                            // blocks; we flatten to a single string.
                            guard let id = block["tool_use_id"] as? String
                            else { break }
                            var text = ""
                            if let s = block["content"] as? String {
                                text = s
                            } else if let arr = block["content"] as? [[String: Any]] {
                                for b in arr {
                                    if let t = b["text"] as? String { text += t }
                                }
                            }
                            collectedToolResults.append((id: id, text: text))
                        case "document":
                            // Anthropic PDF / plaintext document block.
                            // Shape: `{type:document, source:{type, media_type, data}}`
                            // - base64 + application/pdf → data URL, passes
                            //   through to multimodal models as image_url
                            //   (mlx-vlm's PDF decoder accepts this form).
                            // - text/plain → lift `data` into a text part
                            //   so any LLM can read it.
                            // - url source → forward raw URL as image_url
                            //   (http loader in ContentPart decodes it).
                            guard let source = block["source"] as? [String: Any]
                            else { break }
                            let srcType = (source["type"] as? String) ?? ""
                            let mediaType = (source["media_type"] as? String) ?? ""
                            let rawData = (source["data"] as? String) ?? ""
                            if srcType == "base64",
                               mediaType.hasPrefix("application/") {
                                let dataURL = "data:\(mediaType);base64,\(rawData)"
                                converted.append(.init(
                                    type: "image_url",
                                    imageUrl: .init(url: dataURL)))
                            } else if srcType == "text"
                                      || mediaType.hasPrefix("text/") {
                                converted.append(.init(
                                    type: "text", text: rawData))
                            } else if srcType == "url",
                                      let url = source["url"] as? String {
                                converted.append(.init(
                                    type: "image_url",
                                    imageUrl: .init(url: url)))
                            }
                            // Optional title/context fields drop into text.
                            if let title = block["title"] as? String,
                               !title.isEmpty {
                                converted.append(.init(
                                    type: "text", text: "[\(title)]"))
                            }
                        case "server_tool_use":
                            // Server-invoked tool call (e.g. Anthropic's
                            // built-in web_search). Same shape as tool_use
                            // — id, name, input — so map identically.
                            guard let id = block["id"] as? String,
                                  let name = block["name"] as? String
                            else { break }
                            let input = block["input"] as? [String: Any] ?? [:]
                            let argsData =
                                (try? JSONSerialization.data(withJSONObject: input))
                                ?? Data("{}".utf8)
                            let args = String(data: argsData, encoding: .utf8) ?? "{}"
                            let wrapped: [String: Any] = [
                                "id": id,
                                "type": "function",
                                "function": ["name": name, "arguments": args],
                            ]
                            if let data = try? JSONSerialization.data(withJSONObject: wrapped),
                               let call = try? JSONDecoder().decode(
                                ChatRequest.ToolCall.self, from: data)
                            {
                                collectedToolCalls.append(call)
                            }
                        case "web_search_tool_result":
                            // Server tool result — flatten like tool_result.
                            guard let id = block["tool_use_id"] as? String
                            else { break }
                            var text = ""
                            if let s = block["content"] as? String {
                                text = s
                            } else if let arr = block["content"] as? [[String: Any]] {
                                for b in arr {
                                    if let t = b["text"] as? String {
                                        text += t
                                    } else if let title = b["title"] as? String,
                                              let url = b["url"] as? String {
                                        text += "\(title) — \(url)\n"
                                    }
                                }
                            }
                            collectedToolResults.append((id: id, text: text))
                        default:
                            // Unknown block type — drop silently. Real
                            // text/image content in the same message is
                            // preserved above.
                            break
                        }
                    }
                    contentVal = converted.isEmpty
                        ? .string("")
                        : .parts(converted)
                } else {
                    contentVal = nil
                }
                // Assistant messages may carry tool_use blocks → map to
                // ChatRequest.Message.toolCalls so downstream tool-call
                // history is preserved for multi-turn.
                let effectiveToolCalls = collectedToolCalls.isEmpty
                    ? nil : collectedToolCalls
                messages.append(ChatRequest.Message(
                    role: role, content: contentVal, name: nil,
                    toolCalls: effectiveToolCalls, toolCallId: nil
                ))
                // Split tool_result blocks out into their own role=tool
                // messages (OpenAI shape) so the model sees them as tool
                // responses rather than user text. Anthropic attaches them
                // to the user message; vMLX/OpenAI use a separate turn.
                for result in collectedToolResults {
                    messages.append(ChatRequest.Message(
                        role: "tool",
                        content: .string(result.text),
                        name: nil,
                        toolCalls: nil,
                        toolCallId: result.id
                    ))
                }
            }
        }
        // Tool spec translation. Anthropic format is:
        //   {"name":"...","description":"...","input_schema":{...}}
        // OpenAI-style (what vMLX's ChatRequest.Tool expects) is:
        //   {"type":"function","function":{"name","description","parameters"}}
        var tools: [ChatRequest.Tool]? = nil
        if let rawTools = body["tools"] as? [[String: Any]], !rawTools.isEmpty {
            // Translate each Anthropic tool block into an OpenAI-shaped
            // `{type, function:{name, description, parameters}}` dict, then
            // decode as a `ChatRequest.Tool` via Codable. We avoid
            // constructing the struct literally because `Tool.Function`
            // doesn't expose a public memberwise init.
            var out: [ChatRequest.Tool] = []
            for t in rawTools {
                guard let name = t["name"] as? String else { continue }
                var fn: [String: Any] = ["name": name]
                if let d = t["description"] as? String { fn["description"] = d }
                fn["parameters"] = t["input_schema"] as? [String: Any] ?? [:]
                let wrapped: [String: Any] = ["type": "function", "function": fn]
                if let data = try? JSONSerialization.data(withJSONObject: wrapped),
                   let tool = try? JSONDecoder().decode(ChatRequest.Tool.self, from: data)
                {
                    out.append(tool)
                }
            }
            if !out.isEmpty { tools = out }
        }

        // Tool choice translation. Anthropic uses {"type":"auto"|"any"|
        // "tool","name":"..."}. OpenAI-style ChatRequest.ToolChoice has
        // `.auto / .required / .function(name)`.
        var toolChoice: ChatRequest.ToolChoice? = nil
        if let tc = body["tool_choice"] as? [String: Any],
           let tcType = tc["type"] as? String
        {
            switch tcType {
            case "auto": toolChoice = .auto
            case "any":  toolChoice = .required
            case "none": toolChoice = ChatRequest.ToolChoice.none
            case "tool":
                if let name = tc["name"] as? String {
                    toolChoice = .function(name: name)
                }
            default: break
            }
        }

        // thinking.budget_tokens → rough reasoning_effort bucket, matching
        // vmlx_engine/api/anthropic_adapter.py::_budget_to_effort.
        var effort: String? = nil
        if let think = body["thinking"] as? [String: Any],
           let budget = think["budget_tokens"] as? Int
        {
            if budget >= 10_000 { effort = "high" }
            else if budget >= 2_000 { effort = "medium" }
            else if budget > 0 { effort = "low" }
        }

        return ChatRequest(
            model: (body["model"] as? String) ?? "default",
            messages: messages,
            stream: body["stream"] as? Bool,
            maxTokens: body["max_tokens"] as? Int,
            temperature: body["temperature"] as? Double,
            topP: body["top_p"] as? Double,
            topK: body["top_k"] as? Int,
            minP: nil,
            repetitionPenalty: nil,
            stop: body["stop_sequences"] as? [String],
            seed: nil,
            enableThinking: (body["thinking"] as? [String: Any])?["type"] as? String == "enabled",
            reasoningEffort: effort,
            tools: tools,
            toolChoice: toolChoice,
            includeReasoning: body["include_reasoning"] as? Bool
        )
    }
}
