import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine
import vMLXLMCommon

/// OpenAI-compatible Server-Sent Events encoder.
///
/// Wire format (mirrors `vmlx_engine/server.py:_dump_sse_json` / `stream_chat_completion`):
///
///     data: {"id":"chatcmpl-XXXX","object":"chat.completion.chunk","created":1234567890,
///            "model":"<name>","choices":[{"index":0,"delta":{...},"finish_reason":null}]}\n\n
///     ...
///     data: [DONE]\n\n
///
/// JSON is emitted with `ensure_ascii=True` equivalent — non-ASCII characters are
/// encoded as `\uXXXX` escapes so HTTP chunk boundaries cannot split UTF-8 bytes.
/// Fields with nil values are omitted (OpenAI `exclude_none=True`).
public enum SSEEncoder {

    /// Build the ResponseBody that streams an OpenAI chat.completion.chunk SSE.
    public static func chatCompletionStream(
        id: String,
        model: String,
        created: Int,
        includeUsage: Bool,
        includeReasoning: Bool = true,
        upstream: AsyncThrowingStream<StreamChunk, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()

            // 1. First chunk: role=assistant
            let roleChunk = Self.chunkJSON(
                id: id, model: model, created: created,
                delta: ["role": "assistant"],
                finishReason: nil
            )
            try await writer.write(allocator.buffer(string: "data: \(roleChunk)\n\n"))

            var finishReason: String? = nil
            var lastUsage: StreamChunk.Usage? = nil
            var toolCallIndex = 0
            // iter-90 §118: track whether the stream ended due to an
            // upstream throw. When true, the final stop-finish chunk
            // and the usage frame are skipped — emitting them after
            // an error event was misleading to clients that treat
            // `finish_reason: "stop"` as "clean completion".
            var hadError = false

            // Merge upstream with a periodic heartbeat (see
            // `sseMergeWithHeartbeat` docs). Protects thinking models
            // from idle-timeout disconnects on LAN / proxied deployments.
            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)

            do {
                for try await event in merged {
                    switch event {
                    case .heartbeat:
                        // SSE comment — ignored by EventSource parsers
                        // but flushes TCP so the connection stays warm.
                        try await writer.write(
                            allocator.buffer(string: ": keep-alive\n\n"))
                        continue
                    case .chunk(let chunk):
                        if let reasoning = chunk.reasoning, !reasoning.isEmpty, includeReasoning {
                            let j = Self.chunkJSON(
                                id: id, model: model, created: created,
                                delta: ["reasoning_content": reasoning],
                                finishReason: nil
                            )
                            try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                        }
                        if let content = chunk.content, !content.isEmpty {
                            let j = Self.chunkJSON(
                                id: id, model: model, created: created,
                                delta: ["content": content],
                                finishReason: nil
                            )
                            try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                        }
                        if let tcs = chunk.toolCalls, !tcs.isEmpty {
                            for tc in tcs {
                                let delta: [String: Any] = [
                                    "tool_calls": [[
                                        "index": toolCallIndex,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": [
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        ] as [String: Any],
                                    ] as [String: Any]]
                                ]
                                let j = Self.chunkJSON(
                                    id: id, model: model, created: created,
                                    delta: delta, finishReason: nil
                                )
                                try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                                toolCallIndex += 1
                            }
                        }
                        if let fr = chunk.finishReason { finishReason = fr }
                        if let u = chunk.usage { lastUsage = u }
                        // Emit logprobs data when present.
                        if let lps = chunk.logprobs, !lps.isEmpty {
                            let contentArr: [[String: Any]] = lps.map { lp in
                                var entry: [String: Any] = [
                                    "token": lp.token,
                                    "logprob": lp.logprob,
                                ]
                                if let bo = lp.byteOffset {
                                    entry["bytes_offset"] = bo
                                }
                                if !lp.topLogprobs.isEmpty {
                                    entry["top_logprobs"] = lp.topLogprobs.map { alt in
                                        [
                                            "token": alt.token,
                                            "logprob": alt.logprob,
                                        ] as [String: Any]
                                    }
                                }
                                return entry
                            }
                            let logprobsDict: [String: Any] = ["content": contentArr]
                            let j = Self.chunkJSON(
                                id: id, model: model, created: created,
                                delta: ["logprobs": logprobsDict],
                                finishReason: nil
                            )
                            try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                        }
                    }
                }
            } catch {
                hadError = true
                let errObj: [String: Any] = [
                    "error": [
                        "message": "\(error)",
                        "type": "engine_error",
                    ] as [String: Any]
                ]
                let j = Self.asciiJSON(errObj)
                try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
            }

            // Final chunk: empty delta + finish_reason. Skipped on
            // error path — emitting a `finish_reason: "stop"` frame
            // after an error event signals "clean completion" to
            // OpenAI SDK clients and triggers their post-completion
            // hooks (log usage, close UI spinner, etc) as if the
            // generation succeeded.
            if !hadError {
                let finalChunk = Self.chunkJSON(
                    id: id, model: model, created: created,
                    delta: [:],
                    finishReason: finishReason ?? "stop"
                )
                try await writer.write(allocator.buffer(string: "data: \(finalChunk)\n\n"))
            }

            if !hadError, includeUsage, let u = lastUsage {
                var usageDict: [String: Any] = [
                    "prompt_tokens": u.promptTokens,
                    "completion_tokens": u.completionTokens,
                    "total_tokens": u.promptTokens + u.completionTokens,
                    "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
                ]
                if let tps = u.tokensPerSecond { usageDict["tokens_per_second"] = tps }
                if let ppt = u.promptTokensPerSecond { usageDict["prompt_tokens_per_second"] = ppt }
                if let ttft = u.ttftMs { usageDict["ttft_ms"] = ttft }
                if let pm = u.prefillMs { usageDict["prefill_ms"] = pm }
                if let tm = u.totalMs { usageDict["total_ms"] = tm }
                let usageObj: [String: Any] = [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usageDict,
                ]
                let j = Self.asciiJSON(usageObj)
                try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
            }

            try await writer.write(allocator.buffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }
    }

    /// SSE encoder for the legacy `/v1/completions` (text_completion) API.
    ///
    /// Same wire shape as `chatCompletionStream` but emits `choices[0].text`
    /// instead of `choices[0].delta.content`, and uses the object label
    /// `text_completion.chunk`. Matches OpenAI's original completion spec.
    public static func textCompletionStream(
        id: String,
        model: String,
        created: Int,
        upstream: AsyncThrowingStream<StreamChunk, Error>,
        echoPrompt: String? = nil
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()
            var finishReason: String? = nil
            // iter-90 §118: mirror chatCompletionStream — skip the
            // final finish_reason=stop frame if an upstream throw
            // terminated the loop.
            var hadError = false
            // Echo: emit prompt text as a single delta before the
            // completion stream begins. The prompt text arrives as a
            // single chunk; no token-by-token splitting needed.
            if let prompt = echoPrompt {
                let obj: [String: Any] = [
                    "id": id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "index": 0,
                        "text": prompt,
                        "finish_reason": NSNull(),
                    ] as [String: Any]],
                ]
                let j = Self.asciiJSON(obj)
                try? await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
            }
            // Legacy logprobs format — accumulate across chunks for text_offset.
            var allTokens: [String] = []
            var allTokenLogprobs: [Any] = []
            var allTopLogprobs: [[String: Any]] = []
            var allTextOffset: [Int] = []
            var textOffsetCursor = 0
            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)
            do {
                for try await event in merged {
                    switch event {
                    case .heartbeat:
                        try await writer.write(
                            allocator.buffer(string: ": keep-alive\n\n"))
                        continue
                    case .chunk(let chunk):
                        var deltaText: String? = nil
                        if let content = chunk.content, !content.isEmpty {
                            deltaText = content
                        }
                        // Emit logprobs deltas when present, even if text is empty.
                        // Format: legacy completions shape with tokens, token_logprobs,
                        // top_logprobs, text_offset arrays (not chat-completions content array).
                        if let lps = chunk.logprobs, !lps.isEmpty {
                            var chunkTokens: [String] = []
                            var chunkTokenLogprobs: [Any] = []
                            var chunkTopLogprobs: [[String: Any]] = []
                            var chunkTextOffset: [Int] = []
                            for lp in lps {
                                chunkTextOffset.append(textOffsetCursor)
                                chunkTokens.append(lp.token)
                                // NaN logprobs (position 0) → JSON null.
                                chunkTokenLogprobs.append(
                                    lp.logprob.isNaN ? NSNull() : lp.logprob)
                                var dict: [String: Any] = [:]
                                if !lp.logprob.isNaN {
                                    dict[lp.token] = lp.logprob
                                }
                                for alt in lp.topLogprobs {
                                    dict[alt.token] = alt.logprob
                                }
                                chunkTopLogprobs.append(dict)
                                textOffsetCursor += lp.token.utf8.count
                            }
                            let logprobsDict: [String: Any] = [
                                "tokens": chunkTokens,
                                "token_logprobs": chunkTokenLogprobs,
                                "top_logprobs": chunkTopLogprobs,
                                "text_offset": chunkTextOffset,
                            ]
                            // Accumulate for final logprobs object if needed later.
                            allTokens.append(contentsOf: chunkTokens)
                            allTokenLogprobs.append(contentsOf: chunkTokenLogprobs)
                            allTopLogprobs.append(contentsOf: chunkTopLogprobs)
                            allTextOffset.append(contentsOf: chunkTextOffset)
                            let obj: [String: Any] = [
                                "id": id,
                                "object": "text_completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [[
                                    "index": 0,
                                    "text": deltaText ?? "",
                                    "finish_reason": NSNull(),
                                    "logprobs": logprobsDict,
                                ] as [String: Any]],
                            ]
                            let j = Self.asciiJSON(obj)
                            try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                        } else if let text = deltaText {
                            let obj: [String: Any] = [
                                "id": id,
                                "object": "text_completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [[
                                    "index": 0,
                                    "text": text,
                                    "finish_reason": NSNull(),
                                ] as [String: Any]],
                            ]
                            let j = Self.asciiJSON(obj)
                            try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
                        }
                        if let fr = chunk.finishReason { finishReason = fr }
                    }
                }
            } catch {
                hadError = true
                let errObj: [String: Any] = [
                    "error": [
                        "message": "\(error)",
                        "type": "engine_error",
                    ] as [String: Any]
                ]
                let j = Self.asciiJSON(errObj)
                try await writer.write(allocator.buffer(string: "data: \(j)\n\n"))
            }

            if !hadError {
                let finalObj: [String: Any] = [
                    "id": id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "index": 0,
                        "text": "",
                        "finish_reason": finishReason ?? "stop",
                    ] as [String: Any]],
                ]
                let finalJ = Self.asciiJSON(finalObj)
                try await writer.write(allocator.buffer(string: "data: \(finalJ)\n\n"))
            }
            try await writer.write(allocator.buffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }
    }

    /// SSE encoder for OpenAI `/v1/responses` (Responses API).
    ///
    /// Emits the Responses-shape event stream. Each SSE frame is prefixed
    /// with `event: <type>\n` so SDK clients that dispatch on event name
    /// (like the official `openai-python` >= 1.40 Responses client) parse
    /// cleanly. Events emitted:
    ///
    ///   - `response.created`              — once at stream start
    ///   - `response.output_item.added`    — one per reasoning/message/fn_call item
    ///   - `response.reasoning_summary_text.delta` — per reasoning chunk
    ///   - `response.output_text.delta`    — per content chunk
    ///   - `response.function_call_arguments.delta` — per tool-call chunk
    ///   - `response.output_item.done`     — one per closed item
    ///   - `response.completed`            — final event with usage
    ///   - `data: [DONE]\n\n`              — terminator (compat w/ chat SSE parsers)
    public static func responsesStream(
        id: String,
        model: String,
        created: Int,
        upstream: AsyncThrowingStream<StreamChunk, Error>
    ) -> ResponseBody {
        ResponseBody { writer in
            let allocator = ByteBufferAllocator()

            func send(_ eventName: String, _ payload: [String: Any]) async throws {
                let j = Self.asciiJSON(payload)
                try await writer.write(
                    allocator.buffer(string: "event: \(eventName)\ndata: \(j)\n\n"))
            }

            // response.created
            try await send("response.created", [
                "type": "response.created",
                "response": [
                    "id": id,
                    "object": "response",
                    "created_at": created,
                    "status": "in_progress",
                    "model": model,
                    "output": [] as [Any],
                ] as [String: Any],
            ])

            // Lazy block state — we open blocks only when the first chunk
            // of that kind arrives, mirroring Anthropic's multi-block SSE.
            enum OpenBlock {
                case none
                case reasoning(itemId: String, outputIndex: Int)
                case message(itemId: String, outputIndex: Int)
                case toolCall(itemId: String, outputIndex: Int, callId: String, name: String)
            }

            var current: OpenBlock = .none
            var nextOutputIndex = 0
            var finishReason: String? = nil
            var lastUsage: StreamChunk.Usage? = nil
            // iter-90 §118: track whether the upstream stream threw so
            // the final event can differentiate success vs failure.
            var hadError = false
            var seenToolCalls: [String: (outputIndex: Int, itemId: String, name: String)] = [:]
            // Accumulate each tool_call's streaming arguments by call id so
            // `response.output_item.done` can emit the FINAL assembled
            // argument string (not the hardcoded empty) on close. Deep
            // audit #1 — SDK clients that read `item.arguments` on the
            // done event otherwise see blank. Keyed by call id.
            var toolArgsByCallId: [String: String] = [:]
            var accumulatedText = ""
            var accumulatedReasoning = ""

            func closeCurrent() async throws {
                switch current {
                case .none:
                    break
                case .reasoning(let itemId, let outputIndex):
                    try await send("response.output_item.done", [
                        "type": "response.output_item.done",
                        "output_index": outputIndex,
                        "item": [
                            "type": "reasoning",
                            "id": itemId,
                            "summary": [[
                                "type": "summary_text",
                                "text": accumulatedReasoning,
                            ] as [String: Any]],
                        ] as [String: Any],
                    ])
                case .message(let itemId, let outputIndex):
                    try await send("response.output_text.done", [
                        "type": "response.output_text.done",
                        "item_id": itemId,
                        "output_index": outputIndex,
                        "content_index": 0,
                        "text": accumulatedText,
                    ])
                    try await send("response.output_item.done", [
                        "type": "response.output_item.done",
                        "output_index": outputIndex,
                        "item": [
                            "type": "message",
                            "id": itemId,
                            "role": "assistant",
                            "status": "completed",
                            "content": [[
                                "type": "output_text",
                                "text": accumulatedText,
                                "annotations": [] as [Any],
                            ] as [String: Any]],
                        ] as [String: Any],
                    ])
                case .toolCall(let itemId, let outputIndex, let callId, let name):
                    try await send("response.output_item.done", [
                        "type": "response.output_item.done",
                        "output_index": outputIndex,
                        "item": [
                            "type": "function_call",
                            "id": itemId,
                            "call_id": callId,
                            "name": name,
                            "arguments": toolArgsByCallId[callId] ?? "",
                            "status": "completed",
                        ] as [String: Any],
                    ])
                }
                current = .none
            }

            func openReasoning() async throws -> (String, Int) {
                if case .reasoning(let id, let idx) = current { return (id, idx) }
                try await closeCurrent()
                let itemId = "rs_\(UUID().uuidString.prefix(8).lowercased())"
                let outputIndex = nextOutputIndex
                nextOutputIndex += 1
                accumulatedReasoning = ""
                try await send("response.output_item.added", [
                    "type": "response.output_item.added",
                    "output_index": outputIndex,
                    "item": [
                        "type": "reasoning",
                        "id": itemId,
                        "summary": [] as [Any],
                    ] as [String: Any],
                ])
                current = .reasoning(itemId: itemId, outputIndex: outputIndex)
                return (itemId, outputIndex)
            }

            func openMessage() async throws -> (String, Int) {
                if case .message(let id, let idx) = current { return (id, idx) }
                try await closeCurrent()
                let itemId = "msg_\(UUID().uuidString.prefix(8).lowercased())"
                let outputIndex = nextOutputIndex
                nextOutputIndex += 1
                accumulatedText = ""
                try await send("response.output_item.added", [
                    "type": "response.output_item.added",
                    "output_index": outputIndex,
                    "item": [
                        "type": "message",
                        "id": itemId,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [] as [Any],
                    ] as [String: Any],
                ])
                current = .message(itemId: itemId, outputIndex: outputIndex)
                return (itemId, outputIndex)
            }

            func openToolCall(callId: String, name: String) async throws -> (String, Int) {
                // Deep audit #3: previously, if `seenToolCalls[callId]`
                // returned early, deltas landed on the OLD itemId while
                // `current == .none`, so the closing event for the
                // resumed block was never sent. Fix: if `current` is
                // already this tool call, reuse as-is. If `current` is
                // a different block (or .none), close it and rebind
                // `current` to the prior tool call so subsequent deltas
                // emit under the right index AND `closeCurrent()` can
                // fire the matching done event.
                if let existing = seenToolCalls[callId] {
                    if case .toolCall(_, _, let openCallId, _) = current,
                       openCallId == callId
                    {
                        return (existing.itemId, existing.outputIndex)
                    }
                    try await closeCurrent()
                    current = .toolCall(
                        itemId: existing.itemId,
                        outputIndex: existing.outputIndex,
                        callId: callId,
                        name: existing.name)
                    return (existing.itemId, existing.outputIndex)
                }
                try await closeCurrent()
                let itemId = "fc_\(UUID().uuidString.prefix(8).lowercased())"
                let outputIndex = nextOutputIndex
                nextOutputIndex += 1
                seenToolCalls[callId] = (outputIndex: outputIndex, itemId: itemId, name: name)
                try await send("response.output_item.added", [
                    "type": "response.output_item.added",
                    "output_index": outputIndex,
                    "item": [
                        "type": "function_call",
                        "id": itemId,
                        "call_id": callId,
                        "name": name,
                        "arguments": "",
                        "status": "in_progress",
                    ] as [String: Any],
                ])
                current = .toolCall(itemId: itemId, outputIndex: outputIndex,
                                    callId: callId, name: name)
                return (itemId, outputIndex)
            }

            let merged = sseMergeWithHeartbeat(
                upstream: upstream, interval: sseHeartbeatInterval)
            do {
                for try await event in merged {
                    switch event {
                    case .heartbeat:
                        try await writer.write(
                            allocator.buffer(string: ": keep-alive\n\n"))
                        continue
                    case .chunk(let chunk):
                        if let r = chunk.reasoning, !r.isEmpty {
                            let (itemId, outputIndex) = try await openReasoning()
                            accumulatedReasoning += r
                            try await send("response.reasoning_summary_text.delta", [
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": itemId,
                                "output_index": outputIndex,
                                "summary_index": 0,
                                "delta": r,
                            ])
                        }
                        if let c = chunk.content, !c.isEmpty {
                            let (itemId, outputIndex) = try await openMessage()
                            accumulatedText += c
                            try await send("response.output_text.delta", [
                                "type": "response.output_text.delta",
                                "item_id": itemId,
                                "output_index": outputIndex,
                                "content_index": 0,
                                "delta": c,
                            ])
                        }
                        if let tcs = chunk.toolCalls, !tcs.isEmpty {
                            for tc in tcs {
                                let (itemId, outputIndex) = try await openToolCall(
                                    callId: tc.id, name: tc.function.name)
                                if !tc.function.arguments.isEmpty {
                                    // Accumulate the partial arg string for
                                    // this call id so `closeCurrent()` can
                                    // emit the full assembled arguments in
                                    // the terminal `output_item.done` payload.
                                    toolArgsByCallId[tc.id, default: ""] += tc.function.arguments
                                    try await send("response.function_call_arguments.delta", [
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": itemId,
                                        "output_index": outputIndex,
                                        "delta": tc.function.arguments,
                                    ])
                                }
                            }
                        }
                        if let fr = chunk.finishReason { finishReason = fr }
                        if let u = chunk.usage { lastUsage = u }
                    }
                }
            } catch {
                hadError = true
                let errObj: [String: Any] = [
                    "type": "error",
                    "error": [
                        "message": "\(error)",
                        "type": "engine_error",
                    ] as [String: Any],
                ]
                try await send("error", errObj)
            }

            try await closeCurrent()

            // iter-90 §118: on error, emit `response.failed` with
            // status=failed instead of `response.completed` with
            // status=completed. The OpenAI Responses spec uses
            // `response.failed` / `response.incomplete` for error
            // and interrupted paths; emitting `response.completed`
            // after an error event gave SDK clients inconsistent
            // signals (they'd mark the response a success in their
            // telemetry and close the UI spinner as if everything
            // went fine).
            let finalStatus: String = hadError
                ? "failed"
                : (finishReason == "length" ? "incomplete" : "completed")
            let finalEvent: String = hadError
                ? "response.failed"
                : "response.completed"
            var completed: [String: Any] = [
                "type": finalEvent,
                "response": [
                    "id": id,
                    "object": "response",
                    "created_at": created,
                    "status": finalStatus,
                    "model": model,
                ] as [String: Any],
            ]
            if !hadError, let u = lastUsage {
                var r = completed["response"] as! [String: Any]
                // iter-67 (§96) — include tokens_per_second / ttft_ms /
                // prefill_ms / total_ms in the `response.completed`
                // usage dict for parity with chat/completions,
                // Ollama, and Anthropic surfaces.
                r["usage"] = OpenAIRoutes.responsesUsageEnvelope(u)
                completed["response"] = r
            }
            try await send(finalEvent, completed)

            try await writer.write(allocator.buffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }
    }

    // MARK: - JSON helpers (ASCII-safe, exclude_none equivalent)

    static func chunkJSON(
        id: String, model: String, created: Int,
        delta: [String: Any], finishReason: String?
    ) -> String {
        var choice: [String: Any] = [
            "index": 0,
            "delta": delta,
        ]
        choice["finish_reason"] = finishReason ?? NSNull()
        let obj: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        ]
        return asciiJSON(obj)
    }

    /// JSON-serialize with ASCII-only output (escapes non-ASCII as \uXXXX).
    static func asciiJSON(_ object: Any) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys]
        ) else { return "{}" }
        var out = String()
        out.reserveCapacity(data.count)
        for byte in data {
            if byte < 0x80 {
                out.append(Character(UnicodeScalar(byte)))
            } else {
                // JSONSerialization already emits valid UTF-8; convert runs to \uXXXX.
                // Fall back to re-decoding the whole blob once — simpler & correct.
                return Self.asciiEscapeAll(String(data: data, encoding: .utf8) ?? "{}")
            }
        }
        return out
    }

    private static func asciiEscapeAll(_ s: String) -> String {
        var out = String()
        out.reserveCapacity(s.count)
        for scalar in s.unicodeScalars {
            if scalar.value < 0x80 {
                out.append(Character(scalar))
            } else if scalar.value <= 0xFFFF {
                out.append(String(format: "\\u%04x", scalar.value))
            } else {
                // Surrogate pair
                let v = scalar.value - 0x10000
                let hi = 0xD800 + (v >> 10)
                let lo = 0xDC00 + (v & 0x3FF)
                out.append(String(format: "\\u%04x\\u%04x", hi, lo))
            }
        }
        return out
    }
}

extension ByteBufferAllocator {
    fileprivate func buffer(string: String) -> ByteBuffer {
        var b = self.buffer(capacity: string.utf8.count)
        b.writeString(string)
        return b
    }
}

// MARK: - Heartbeat merging
//
// Thinking models (Qwen3.6-JANGTQ2, GLM-5.1, DeepSeek R1) can spend
// 20-40 seconds in prefill + initial reasoning before emitting their
// first visible token. If vMLX is LAN-bound behind an idle-timeout
// proxy (nginx default 60s) OR the client has a short HTTP read
// timeout (ollama-js default 30s, some custom SDKs ~20s), the TCP
// connection is torn down before the first token arrives and the
// user sees a "connection reset" rather than the completion.
//
// Fix: merge the upstream chunk stream with a periodic heartbeat
// signal. When heartbeat fires, we emit a single SSE comment line
// (`: keep-alive\n\n`) — valid per the SSE spec, ignored by
// EventSource / ChatGPT / Ollama parsers, but puts bytes on the wire
// so the TCP connection stays live. Local 127.0.0.1 servers get
// this essentially for free (negligible cost) and LAN / proxied
// deployments stop dropping thinking-model streams.
//
// The heartbeat is cancelled when the upstream terminates so the
// final [DONE] isn't followed by a stray keep-alive.

/// Unified event type threaded through the merged heartbeat stream.
enum SSEMergedEvent: @unchecked Sendable {
    case chunk(StreamChunk)
    case heartbeat
}

/// Merge `upstream` chunks with a periodic heartbeat. The returned
/// stream yields `.chunk(c)` for every upstream element (in order)
/// and `.heartbeat` every `interval` seconds in between. Errors and
/// termination propagate from the upstream. Pass `interval <= 0` to
/// disable heartbeating entirely.
func sseMergeWithHeartbeat(
    upstream: AsyncThrowingStream<StreamChunk, Error>,
    interval: TimeInterval
) -> AsyncThrowingStream<SSEMergedEvent, Error> {
    AsyncThrowingStream { continuation in
        // 2026-04-18 iter-18 fix: a strongly-held ref to the heartbeat
        // task so the upstream completion closure can cancel it even
        // when the consumer drains the stream normally (no-cancel path
        // — `onTermination` doesn't always fire on clean completion,
        // and without explicit cancellation the heartbeat loop would
        // run forever, yielding into a finished continuation).
        // Wrapped in a class box for cross-Task write access.
        final class HeartbeatHandle: @unchecked Sendable {
            var task: Task<Void, Never>?
        }
        let hb = HeartbeatHandle()

        let upstreamTask = Task {
            do {
                for try await c in upstream {
                    continuation.yield(.chunk(c))
                }
                // Clean upstream end — cancel the heartbeat BEFORE
                // finishing the continuation so no stray .heartbeat
                // events arrive between finish() and the consumer's
                // return-from-loop.
                hb.task?.cancel()
                continuation.finish()
            } catch {
                hb.task?.cancel()
                continuation.finish(throwing: error)
            }
        }
        if interval > 0 {
            hb.task = Task {
                let nanos = UInt64(interval * 1_000_000_000)
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: nanos)
                    if Task.isCancelled { return }
                    continuation.yield(.heartbeat)
                }
            }
        }
        continuation.onTermination = { _ in
            upstreamTask.cancel()
            hb.task?.cancel()
        }
    }
}

/// Heartbeat interval used by every SSE-producing route. Env override
/// `VMLX_SSE_HEARTBEAT_SEC` lets ops tune for a specific proxy
/// without rebuilding. 15s is a conservative middle ground that
/// protects nginx's 60s default while staying cheap on 127.0.0.1.
let sseHeartbeatInterval: TimeInterval = {
    if let raw = ProcessInfo.processInfo.environment["VMLX_SSE_HEARTBEAT_SEC"],
       let v = TimeInterval(raw), v >= 0 {
        return v
    }
    return 15
}()
