import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import vMLXEngine
import vMLXTTS

/// OpenAI-compatible routes.
///
/// Python source: `vmlx_engine/server.py`
///   - POST /v1/chat/completions  — line 4250 `create_chat_completion`, stream at line 5597
///   - POST /v1/responses         — line 4946 `create_response`
///   - POST /v1/embeddings        — line 2930 `create_embeddings`
///   - POST /v1/images/generations — line 3429 `create_image`
///   - POST /v1/images/edits      — line 3632 `create_image_edit`
///   - POST /v1/rerank            — line 3083 `create_rerank`
///   - GET  /v1/models            — line 2642 `list_models`
public enum OpenAIRoutes {

    public static func register<Context: RequestContext>(
        on router: Router<Context>,
        engine: Engine
    ) {
        // GET /v1/models — enumerate everything ModelLibrary knows about.
        // OpenAI-SDK clients enumerate this list to auto-pick a model when
        // the user hasn't specified one. Before this fix the endpoint
        // returned an empty array and every third-party integration
        // (openai python sdk, LangChain, LiteLLM, etc.) broke silently.
        router.get("/v1/models") { _, _ -> Response in
            // Trigger a library scan on first hit so servers started via
            // `vmlxctl serve --model <path>` (which does not walk the
            // HuggingFace cache) still surface every downloaded model
            // to `GET /v1/models`. `scan(force:false)` is a no-op inside
            // the 5-minute freshness window, so subsequent hits are free.
            _ = await engine.modelLibrary.scan(force: false)
            let entries = await engine.modelLibrary.entries()
            let dflashReady = await engine.dflashIsReady()
            let dflashPath = await engine.dflashDrafterPath()?.path
            let loadedPath = await engine.loadedModelPath?.path
            let data: [[String: Any]] = entries.map { e in
                let isLoaded = loadedPath.map { $0 == e.canonicalPath.path } ?? false
                // Attach speculative-decoding metadata only to the
                // currently-loaded model entry (a drafter is bound to
                // one specific target at a time).
                var speculative: [String: Any] = [:]
                if dflashReady, isLoaded {
                    speculative["kind"] = "dflash"
                    // iter-117 §143: redact drafter path too.
                    speculative["drafter"] = dflashPath.map(Self.redactHomeDir) as Any
                }
                // iter-117 §143: redact per-model canonicalPath. /v1/models
                // enumerates the entire model library — without redaction
                // any API-key-holding peer could learn the user's full
                // HF-cache layout (every installed model's snapshot hash
                // path), which is even more disclosive than the single-
                // model leaks §141/§142 closed.
                var vmlx: [String: Any] = [
                    "family": e.family,
                    "modality": e.modality.rawValue,
                    "size_bytes": e.totalSizeBytes,
                    "is_jang": e.isJANG,
                    "is_mxtq": e.isMXTQ,
                    "quant_bits": e.quantBits as Any,
                    "path": Self.redactHomeDir(e.canonicalPath.path),
                    // iter-104 §182: honest loaded flag. Was previously
                    // inferred only via presence/absence of the
                    // `speculative_decoding` block, which was a
                    // fingerprint at best (and false-negative when the
                    // user ran without DFlash). Now explicit.
                    "loaded": isLoaded,
                ]
                if !speculative.isEmpty {
                    vmlx["speculative_decoding"] = speculative
                }
                return [
                    "id": e.displayName,
                    "object": "model",
                    // iter-104 §182: `created` was `Date().timeIntervalSince1970`
                    // on EVERY call, which returned the current request time
                    // for every entry — completely broken semantics (OpenAI's
                    // spec says it's when the model was created). Use the
                    // entry's `detectedAt` so /v1/models agrees with Ollama's
                    // /api/tags `modified_at` (both sourced from the same
                    // ModelLibrary scan).
                    "created": Int(e.detectedAt.timeIntervalSince1970),
                    // iter-104 §182: derive owner from `org/repo`
                    // displayName when present. Prior heuristic collapsed
                    // every JANG model into "dealignai" (wrong for
                    // JANGQ-AI/..., OsaurusAI/..., etc.) and every non-JANG
                    // model into "mlx-community" (wrong for HuggingFaceTB/...,
                    // Qwen/..., etc.). Local-disk-only entries without an
                    // "org/" prefix fall back to "local".
                    "owned_by": Self.deriveOwnedBy(e.displayName),
                    "vmlx": vmlx,
                ]
            }
            return Self.json([
                "object": "list",
                "data": data,
            ])
        }

        // POST /v1/chat/completions — full decode + streaming/non-streaming
        router.post("/v1/chat/completions") { req, ctx -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            var chatReq: ChatRequest
            do {
                chatReq = try JSONDecoder().decode(ChatRequest.self, from: data)
            } catch {
                return Self.errorJSON(.badRequest, "invalid request: \(error)")
            }
            // P1-API-1: fold OpenAI v2 max_completion_tokens alias into max_tokens.
            chatReq.applyMaxCompletionTokensAlias()
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return Self.errorJSON(.badRequest, err.description)
            } catch {
                return Self.errorJSON(.badRequest, "invalid request: \(error)")
            }

            let isStream = chatReq.stream ?? false
            let id = "chatcmpl-\(UUID().uuidString.prefix(8).lowercased())"
            let created = Int(Date().timeIntervalSince1970)

            await engine.wakeFromStandby()
            // Settings resolution happens inside `Engine.stream` →
            // `performOneGenerationPass`, which calls
            // `settings.resolved(request: RequestOverride.from(request))`
            // and merges the resolved values into `GenerateParameters`.
            // No duplicate resolution here — the engine is the single source
            // of truth for the 4-tier merge.
            if isStream {
                let stream = await engine.stream(request: chatReq, id: id)
                var headers: HTTPFields = [:]
                headers[.contentType] = "text/event-stream; charset=utf-8"
                headers[.cacheControl] = "no-cache"
                headers[.connection] = "keep-alive"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: SSEEncoder.chatCompletionStream(
                        id: id, model: chatReq.model, created: created,
                        includeUsage: chatReq.streamOptions?.includeUsage ?? false,
                        includeReasoning: chatReq.includeReasoning ?? true,
                        upstream: stream
                    )
                )
            }

            // Non-streaming: collect stream into a single response.
            var content = ""
            var reasoning = ""
            var toolCalls: [ChatRequest.ToolCall] = []
            var finishReason: String? = nil
            var usage: StreamChunk.Usage? = nil
            let stream = await engine.stream(request: chatReq, id: id)
            do {
                for try await chunk in stream {
                    if let c = chunk.content { content += c }
                    if let r = chunk.reasoning { reasoning += r }
                    if let tcs = chunk.toolCalls { toolCalls.append(contentsOf: tcs) }
                    if let fr = chunk.finishReason { finishReason = fr }
                    if let u = chunk.usage { usage = u }
                }
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }

            // Validate JSON output if response_format requested it.
            // Surfaces violation as finish_reason="content_filter" so the
            // SDK caller can detect that the contract was not honored.
            // Returning the raw text alongside lets the user inspect what
            // the model produced.
            if let reason = Engine.validateResponseFormat(
                content, format: chatReq.responseFormat)
            {
                finishReason = "content_filter"
                content = "{\"error\": \"\(reason.replacingOccurrences(of: "\"", with: "'"))\", \"raw\": \(Self.jsonEscape(content))}"
            }
            var message: [String: Any] = ["role": "assistant", "content": content]
            // Anthropic-compat: `include_reasoning=false` explicitly suppresses
            // the reasoning block even if the model produced one. Default
            // (nil/true) emits it so OpenAI clients that DO surface
            // `reasoning_content` see it. Distinct from `enable_thinking`
            // which controls *generation*.
            let shouldEmitReasoning = chatReq.includeReasoning ?? true
            if shouldEmitReasoning && !reasoning.isEmpty {
                message["reasoning_content"] = reasoning
            }
            if !toolCalls.isEmpty {
                message["tool_calls"] = toolCalls.map { tc -> [String: Any] in
                    [
                        "id": tc.id,
                        "type": "function",
                        "function": [
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        ] as [String: Any],
                    ]
                }
            }
            var obj: [String: Any] = [
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": chatReq.model,
                "choices": [[
                    "index": 0,
                    "message": message,
                    "finish_reason": finishReason ?? "stop",
                ] as [String: Any]],
            ]
            if let u = usage {
                var usageObj: [String: Any] = [
                    "prompt_tokens": u.promptTokens,
                    "completion_tokens": u.completionTokens,
                    "total_tokens": u.promptTokens + u.completionTokens,
                    "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
                ]
                // iter-64: surface timing + tokens_per_second in the
                // non-stream envelope for parity with the SSE stream
                // encoder. Clients (benchmarks, langchain, custom
                // dashboards) previously had no way to read tok/s from
                // a non-stream completion — they'd compute it from
                // wall-clock which double-counts HTTP round-trip.
                if let tps = u.tokensPerSecond { usageObj["tokens_per_second"] = tps }
                // iter-126 §201: prompt_tokens_per_second — SSEEncoder
                // already emits this (SSEEncoder.swift:143) and
                // JSONLEncoder after iter-125 §200 too; the three
                // non-stream OpenAI-shape handlers (chat, completions,
                // responses) were the last surfaces still silently
                // dropping prefill throughput. Clients that scrape
                // wire JSON for SLOs (custom dashboards, benchmarks
                // that compare vMLX vs LM-Studio vs Ollama) had to
                // derive prefill-tps from `prompt_tokens *
                // (1e9/prompt_eval_duration)` in the Ollama shape
                // but couldn't back-solve it from the OpenAI shape
                // at all (no prefill duration emitted there).
                if let pps = u.promptTokensPerSecond { usageObj["prompt_tokens_per_second"] = pps }
                if let ttft = u.ttftMs { usageObj["ttft_ms"] = ttft }
                if let prefill = u.prefillMs { usageObj["prefill_ms"] = prefill }
                if let total = u.totalMs { usageObj["total_ms"] = total }
                // iter-120 §196: cache_detail was a silent drop on every
                // HTTP response. Stream.swift populates it, MessageBubble
                // reads it, but no wire route ever forwarded it. Hard
                // rule #6 requires it on every response body. Live test
                // 2026-04-20 on gemma-4-e2b-4bit came back null.
                if let detail = u.cacheDetail { usageObj["cache_detail"] = detail }
                obj["usage"] = usageObj
            }
            return Self.json(obj)
        }

        // POST /v1/chat/completions/{id}/cancel
        // POST /v1/responses/{id}/cancel
        // POST /v1/completions/{id}/cancel
        //
        // OpenAI-compatible request cancellation. Engine tracks active
        // streams in `streamTasksByID` so `cancelStream(id:)` targets
        // the specific generation task registered under the caller's id.
        //
        // iter-123 §149: previously, when the per-id lookup missed
        // (unknown/stale id, or stream finished before the cancel
        // arrived), the handler fell back to `engine.cancelStream()`
        // — the no-arg variant that drains EVERY in-flight stream.
        // Under multi-user / gateway deployments that was a wrong-id
        // sledgehammer: a misaddressed cancel would kill an unrelated
        // user's in-flight response. The responses also lied: they
        // always emitted `cancelled: true` even when no stream was
        // found. Fix: return 404 with `cancelled: false, found: false`
        // when the id isn't known; emit 200 with `cancelled: true`
        // only when a real per-id match was killed.
        router.post("/v1/chat/completions/:id/cancel") { req, ctx -> Response in
            let id = ctx.parameters.get("id") ?? ""
            let hit = await engine.cancelStream(id: id)
            return Self.json([
                "id": id, "object": "chat.completion.cancel",
                "cancelled": hit, "found": hit,
            ], status: hit ? .ok : .notFound)
        }
        router.post("/v1/responses/:id/cancel") { req, ctx -> Response in
            let id = ctx.parameters.get("id") ?? ""
            let hit = await engine.cancelStream(id: id)
            return Self.json([
                "id": id, "object": "response.cancel",
                "cancelled": hit, "found": hit,
            ], status: hit ? .ok : .notFound)
        }
        router.post("/v1/completions/:id/cancel") { req, ctx -> Response in
            let id = ctx.parameters.get("id") ?? ""
            let hit = await engine.cancelStream(id: id)
            return Self.json([
                "id": id, "object": "completion.cancel",
                "cancelled": hit, "found": hit,
            ], status: hit ? .ok : .notFound)
        }

        // POST /v1/completions — legacy text-completion endpoint.
        //
        // Internally wraps the `prompt` field as a single user message and
        // dispatches through `engine.stream`. Matches OpenAI's original
        // completion format (not chat.completion) with `choices[0].text`
        // instead of `choices[0].message.content`.
        //
        // Short-circuit behaviors: streaming via SSE, stop-sequence support
        // via the `stop` field, max_tokens, temperature, top_p all honored
        // through the ChatRequest → GenerateParameters merge inside
        // `Engine.stream`.
        router.post("/v1/completions") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            let model = (obj["model"] as? String) ?? "default"
            let prompt: String
            if let s = obj["prompt"] as? String {
                prompt = s
            } else if let arr = obj["prompt"] as? [String] {
                // Batching: concatenate for simplicity. OpenAI's real
                // spec returns N choices for N prompts; we'd need a
                // parallel batch path to match that faithfully.
                prompt = arr.joined(separator: "\n")
            } else {
                return Self.errorJSON(.badRequest, "missing 'prompt'")
            }

            // iter-98 §176: two silent drops in the legacy route that
            // matched iter-97's findings on /v1/responses.
            //   (1) `stop` only decoded as `[String]`; OpenAI's legacy
            //       spec ALSO accepts a single `string` (and commonly
            //       sends one). Requests with `{"stop": "\n"}` had the
            //       stop sequence silently dropped and the model kept
            //       writing past it. Accept both shapes + wrap scalar
            //       into a singleton list for the engine.
            //   (2) `frequency_penalty` + `presence_penalty` had no
            //       forwarding at all. Chat-completions fixed this in
            //       iter-95; Responses in iter-97. Legacy was the last
            //       route still dropping them.
            let stopList: [String]? = {
                if let arr = obj["stop"] as? [String], !arr.isEmpty { return arr }
                if let s = obj["stop"] as? String, !s.isEmpty { return [s] }
                return nil
            }()
            var chatReq = ChatRequest(
                model: model,
                messages: [.init(role: "user", content: .string(prompt))],
                stream: obj["stream"] as? Bool,
                maxTokens: obj["max_tokens"] as? Int,
                temperature: obj["temperature"] as? Double,
                topP: obj["top_p"] as? Double,
                topK: obj["top_k"] as? Int,
                minP: obj["min_p"] as? Double,
                repetitionPenalty: obj["repetition_penalty"] as? Double,
                stop: stopList,
                seed: obj["seed"] as? Int
            )
            chatReq.frequencyPenalty = obj["frequency_penalty"] as? Double
            chatReq.presencePenalty = obj["presence_penalty"] as? Double
            // iter-119 §145: pre-flight validation, same class as iter-67
            // §96 for /v1/responses. /v1/completions was the last chat-
            // shape route skipping validate() — bad temperature /
            // max_tokens / stop silently leaked past the 400 wall and
            // produced mid-stream fatals inside the engine. Reject
            // cleanly at the HTTP boundary for API-spec parity with
            // /v1/chat/completions and /v1/responses.
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return Self.errorJSON(.badRequest, err.description)
            } catch {
                return Self.errorJSON(.badRequest, "invalid request: \(error)")
            }
            await engine.wakeFromStandby()

            let id = "cmpl-\(UUID().uuidString.prefix(8).lowercased())"
            let created = Int(Date().timeIntervalSince1970)
            let isStream = chatReq.stream ?? false

            if isStream {
                let stream = await engine.stream(request: chatReq, id: id)
                var headers: HTTPFields = [:]
                headers[.contentType] = "text/event-stream; charset=utf-8"
                headers[.cacheControl] = "no-cache"
                headers[.connection] = "keep-alive"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: SSEEncoder.textCompletionStream(
                        id: id, model: model, created: created, upstream: stream
                    )
                )
            }

            var content = ""
            var finishReason: String? = nil
            var usage: StreamChunk.Usage? = nil
            let stream = await engine.stream(request: chatReq, id: id)
            do {
                for try await chunk in stream {
                    if let c = chunk.content { content += c }
                    if let fr = chunk.finishReason { finishReason = fr }
                    if let u = chunk.usage { usage = u }
                }
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }

            var obj2: [String: Any] = [
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [[
                    "text": content,
                    "index": 0,
                    "finish_reason": finishReason ?? "stop",
                ] as [String: Any]],
            ]
            if let u = usage {
                // iter-105 §183: hard rule #6 — every response body
                // carries the four vMLX timing fields. Legacy
                // /v1/completions was the last non-stream OpenAI-shape
                // route dropping them (chat/completions wired in
                // iter-64 §100, responses in iter-67 §96). Mirror the
                // same shape here so latency dashboards + perf
                // dashboards + vmlxctl --json all see the same
                // envelope regardless of which route handled the call.
                var usageObj: [String: Any] = [
                    "prompt_tokens": u.promptTokens,
                    "completion_tokens": u.completionTokens,
                    "total_tokens": u.promptTokens + u.completionTokens,
                    "prompt_tokens_details": ["cached_tokens": u.cachedTokens] as [String: Any],
                ]
                if let tps = u.tokensPerSecond { usageObj["tokens_per_second"] = tps }
                // iter-126 §201: prompt_tokens_per_second — SSEEncoder
                // already emits this (SSEEncoder.swift:143) and
                // JSONLEncoder after iter-125 §200 too; the three
                // non-stream OpenAI-shape handlers (chat, completions,
                // responses) were the last surfaces still silently
                // dropping prefill throughput. Clients that scrape
                // wire JSON for SLOs (custom dashboards, benchmarks
                // that compare vMLX vs LM-Studio vs Ollama) had to
                // derive prefill-tps from `prompt_tokens *
                // (1e9/prompt_eval_duration)` in the Ollama shape
                // but couldn't back-solve it from the OpenAI shape
                // at all (no prefill duration emitted there).
                if let pps = u.promptTokensPerSecond { usageObj["prompt_tokens_per_second"] = pps }
                if let ttft = u.ttftMs { usageObj["ttft_ms"] = ttft }
                if let prefill = u.prefillMs { usageObj["prefill_ms"] = prefill }
                if let total = u.totalMs { usageObj["total_ms"] = total }
                // iter-120 §196: cache_detail parity with chat/completions.
                if let detail = u.cacheDetail { usageObj["cache_detail"] = detail }
                obj2["usage"] = usageObj
            }
            return Self.json(obj2)
        }

        // POST /v1/responses — OpenAI "Responses" API.
        //
        // Supports:
        //   - String or structured-array `input` (role+content parts,
        //     function_call, function_call_output)
        //   - `instructions` → system message
        //   - `tools` passthrough (function tools only)
        //   - `tool_choice` (auto/none/required/{type:function,name})
        //   - `reasoning.effort` → reasoning_effort
        //   - Non-streaming: emits `response` object with
        //     `output[]` blocks (message + reasoning + function_call)
        //   - Streaming SSE: emits Responses-shape events
        //     (response.created, response.output_text.delta,
        //     response.reasoning_summary.delta, response.completed)
        //
        // Conversation state (`previous_response_id`, `store`, background
        // mode) is not persisted — callers resend history each turn.
        router.post("/v1/responses") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 32 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            let model = (obj["model"] as? String) ?? "default"

            var messages: [ChatRequest.Message] = []
            if let sys = obj["instructions"] as? String, !sys.isEmpty {
                messages.append(.init(role: "system", content: .string(sys)))
            }

            if let s = obj["input"] as? String {
                messages.append(.init(role: "user", content: .string(s)))
            } else if let arr = obj["input"] as? [[String: Any]] {
                // Each item: {type:"message", role, content:[parts]}
                //       or:  {type:"function_call", call_id, name, arguments}
                //       or:  {type:"function_call_output", call_id, output}
                for item in arr {
                    let type = (item["type"] as? String) ?? "message"
                    switch type {
                    case "function_call":
                        let callId = (item["call_id"] as? String)
                            ?? (item["id"] as? String) ?? UUID().uuidString
                        let name = (item["name"] as? String) ?? ""
                        let args = (item["arguments"] as? String) ?? "{}"
                        let wrapped: [String: Any] = [
                            "id": callId,
                            "type": "function",
                            "function": ["name": name, "arguments": args],
                        ]
                        var calls: [ChatRequest.ToolCall]? = nil
                        if let d = try? JSONSerialization.data(withJSONObject: wrapped),
                           let tc = try? JSONDecoder().decode(
                               ChatRequest.ToolCall.self, from: d) {
                            calls = [tc]
                        }
                        messages.append(.init(
                            role: "assistant",
                            content: .string(""),
                            toolCalls: calls))
                    case "function_call_output":
                        let callId = (item["call_id"] as? String) ?? ""
                        let output = (item["output"] as? String) ?? ""
                        messages.append(.init(
                            role: "tool",
                            content: .string(output),
                            toolCallId: callId))
                    default:  // "message" or legacy
                        let role = (item["role"] as? String) ?? "user"
                        let content: ChatRequest.ContentValue
                        if let s = item["content"] as? String {
                            content = .string(s)
                        } else if let parts = item["content"] as? [[String: Any]] {
                            var out: [ChatRequest.ContentPart] = []
                            for p in parts {
                                let t = (p["type"] as? String) ?? "text"
                                if t == "input_text" || t == "output_text" || t == "text" {
                                    out.append(.init(type: "text",
                                        text: p["text"] as? String))
                                } else if t == "input_image" {
                                    if let url = p["image_url"] as? String {
                                        out.append(.init(type: "image_url",
                                            imageUrl: .init(url: url)))
                                    } else if let dict = p["image_url"] as? [String: Any],
                                              let url = dict["url"] as? String {
                                        out.append(.init(type: "image_url",
                                            imageUrl: .init(url: url)))
                                    }
                                }
                            }
                            content = .parts(out)
                        } else {
                            content = .string("")
                        }
                        messages.append(.init(role: role, content: content))
                    }
                }
            } else {
                return Self.errorJSON(.badRequest, "missing 'input'")
            }

            // tools[] — accept function tools only; Codable round-trip
            // because ChatRequest.Tool.Function has no public memberwise init.
            var tools: [ChatRequest.Tool]? = nil
            if let rawTools = obj["tools"] as? [[String: Any]] {
                var collected: [ChatRequest.Tool] = []
                for t in rawTools {
                    let type = (t["type"] as? String) ?? "function"
                    guard type == "function" else { continue }
                    var fn: [String: Any] = [:]
                    if let n = t["name"] as? String { fn["name"] = n }
                    if let d = t["description"] as? String { fn["description"] = d }
                    if let p = t["parameters"] as? [String: Any] { fn["parameters"] = p }
                    let wrapped: [String: Any] = ["type": "function", "function": fn]
                    if let d = try? JSONSerialization.data(withJSONObject: wrapped),
                       let tool = try? JSONDecoder().decode(ChatRequest.Tool.self, from: d) {
                        collected.append(tool)
                    }
                }
                if !collected.isEmpty { tools = collected }
            }

            var toolChoice: ChatRequest.ToolChoice? = nil
            if let s = obj["tool_choice"] as? String {
                let w: [String: Any] = ["tool_choice": s]
                if let d = try? JSONSerialization.data(withJSONObject: w),
                   let decoded = try? JSONDecoder().decode(
                       [String: ChatRequest.ToolChoice].self, from: d) {
                    toolChoice = decoded["tool_choice"]
                }
            } else if let tc = obj["tool_choice"] as? [String: Any],
                      let fn = tc["function"] as? [String: Any],
                      let name = fn["name"] as? String {
                toolChoice = .function(name: name)
            }

            var reasoningEffort: String? = nil
            if let r = obj["reasoning"] as? [String: Any],
               let e = r["effort"] as? String {
                reasoningEffort = e
            }

            // iter-97 §175: pre-fix, this ChatRequest init only
            // forwarded model/messages/stream/maxTokens/temperature/
            // topP/reasoningEffort/tools/toolChoice. Every other
            // parameter a caller sent on `/v1/responses` (seed,
            // stop, top_k, min_p, repetition_penalty, frequency_
            // penalty, presence_penalty) was silently dropped —
            // the engine happily produced output using defaults,
            // the user saw "my penalty had no effect" with no 400
            // and no log. The chat-completions route (line 93) has
            // been wiring these since iter-95 §173; Responses was
            // inheriting a stale build. Parse each optional field
            // defensively + forward only when present so the
            // zero-cost default path stays intact.
            let stopList: [String]? = {
                if let arr = obj["stop"] as? [String], !arr.isEmpty { return arr }
                if let s = obj["stop"] as? String, !s.isEmpty { return [s] }
                return nil
            }()
            var chatReq = ChatRequest(
                model: model,
                messages: messages,
                stream: obj["stream"] as? Bool,
                maxTokens: obj["max_output_tokens"] as? Int,
                temperature: obj["temperature"] as? Double,
                topP: obj["top_p"] as? Double,
                topK: obj["top_k"] as? Int,
                minP: obj["min_p"] as? Double,
                repetitionPenalty: obj["repetition_penalty"] as? Double,
                stop: stopList,
                seed: obj["seed"] as? Int,
                reasoningEffort: reasoningEffort,
                tools: tools,
                toolChoice: toolChoice
            )
            // frequencyPenalty / presencePenalty are struct-var fields
            // without init params — set them after construction.
            chatReq.frequencyPenalty = obj["frequency_penalty"] as? Double
            chatReq.presencePenalty = obj["presence_penalty"] as? Double
            // iter-67 (§96) — /v1/responses was the last chat-style route
            // skipping `chatReq.validate()`. Chat/completions (line 93) +
            // Anthropic /v1/messages (line 43) both call it; Responses
            // silently let negative max_output_tokens / temperature=99 /
            // etc. reach the engine as a 200 → partial stream. Align.
            do {
                try chatReq.validate()
            } catch let err as ChatRequestValidationError {
                return Self.errorJSON(.badRequest, err.description)
            } catch {
                return Self.errorJSON(.badRequest, "invalid request: \(error)")
            }
            await engine.wakeFromStandby()

            let id = "resp_\(UUID().uuidString.prefix(8).lowercased())"
            let created = Int(Date().timeIntervalSince1970)
            let isStream = chatReq.stream ?? false

            if isStream {
                let upstream = await engine.stream(request: chatReq, id: id)
                var headers: HTTPFields = [:]
                headers[.contentType] = "text/event-stream; charset=utf-8"
                headers[.cacheControl] = "no-cache"
                headers[.connection] = "keep-alive"
                return Response(
                    status: .ok,
                    headers: headers,
                    body: SSEEncoder.responsesStream(
                        id: id, model: model, created: created, upstream: upstream
                    )
                )
            }

            var content = ""
            var reasoning = ""
            var toolCalls: [ChatRequest.ToolCall] = []
            var usage: StreamChunk.Usage? = nil
            let stream = await engine.stream(request: chatReq, id: id)
            do {
                for try await chunk in stream {
                    if let c = chunk.content { content += c }
                    if let r = chunk.reasoning { reasoning += r }
                    if let tcs = chunk.toolCalls { toolCalls.append(contentsOf: tcs) }
                    if let u = chunk.usage { usage = u }
                }
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }

            // Assemble output[] blocks.
            var output: [[String: Any]] = []
            if !reasoning.isEmpty {
                output.append([
                    "type": "reasoning",
                    "id": "rs_\(UUID().uuidString.prefix(8).lowercased())",
                    "summary": [[
                        "type": "summary_text",
                        "text": reasoning,
                    ] as [String: Any]],
                ])
            }
            if !content.isEmpty || (toolCalls.isEmpty && reasoning.isEmpty) {
                output.append([
                    "type": "message",
                    "id": "msg_\(UUID().uuidString.prefix(8).lowercased())",
                    "role": "assistant",
                    "status": "completed",
                    "content": [[
                        "type": "output_text",
                        "text": content,
                        "annotations": [] as [Any],
                    ] as [String: Any]],
                ])
            }
            for tc in toolCalls {
                output.append([
                    "type": "function_call",
                    "id": "fc_\(UUID().uuidString.prefix(8).lowercased())",
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                    "status": "completed",
                ])
            }

            var out: [String: Any] = [
                "id": id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model,
                "output_text": content,
                "output": output,
            ]
            if let u = usage {
                out["usage"] = Self.responsesUsageEnvelope(u)
            }
            return Self.json(out)
        }

        // POST /v1/embeddings — real embeddings via vMLXEmbedders
        router.post("/v1/embeddings") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            // iter-60: JIT wake — previously only chat/completions/
            // responses routes pulled the engine out of standby. An
            // embeddings request against a soft-slept engine would
            // 503 with `notLoaded` even though the model is resident.
            // Matches the chat-route pattern one level up.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.embeddings(request: obj)
                return Self.json(result)
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /v1/images/generations — dispatch to the dict-form
        // `Engine.generateImage` which flows through FluxBackend.
        // Accepts OpenAI wire format: `{model, prompt, n, size,
        // response_format}`. The route body parses JSON and forwards
        // untouched.
        router.post("/v1/images/generations") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            // iter-60: JIT wake — see embeddings rationale above.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.generateImage(request: obj)
                return Self.json(result)
            } catch {
                // iter-68: classify user errors (unknown model, missing
                // required field) as 400 instead of 500. The underlying
                // FluxBackend error carries its category in the message
                // substring — `"no model entry for"` is the canonical
                // shape emitted by `FluxBackend.generate` / `editImage`.
                let msg = "\(error)"
                let isUserError = msg.contains("no model entry for")
                    || msg.contains("missing required")
                    || msg.contains("unknown model")
                return Self.errorJSON(
                    isUserError ? .badRequest : .internalServerError,
                    msg
                )
            }
        }

        // POST /v1/images/edits — supports both OpenAI multipart/form-data
        // (real SDK clients) AND a JSON body with base64 fields (the
        // vMLX extension used by the in-app API tab).
        //
        // Multipart parts → dict-form:
        //   image    → base64 `image`
        //   mask     → base64 `mask`
        //   prompt   → `prompt`
        //   model    → `model`
        //   size     → `size` ("512x512")
        //   n        → `n` (Int)
        //   response_format → `response_format`
        router.post("/v1/images/edits") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 64 * 1024 * 1024)
            let data = Data(buffer: body)
            let contentType = req.headers[.contentType] ?? ""

            var obj: [String: Any]? = nil
            if contentType.lowercased().hasPrefix("multipart/form-data") {
                guard let boundary = MultipartFormParser.boundary(from: contentType)
                else {
                    return Self.errorJSON(.badRequest,
                        "multipart/form-data body missing boundary")
                }
                let parts = MultipartFormParser.parse(body: data, boundary: boundary)
                var out: [String: Any] = [:]
                for part in parts {
                    switch part.name {
                    case "image":
                        out["image"] = part.body.base64EncodedString()
                    case "mask":
                        out["mask"] = part.body.base64EncodedString()
                    case "prompt", "model", "size", "response_format":
                        out[part.name] = String(data: part.body, encoding: .utf8) ?? ""
                    case "n":
                        let s = String(data: part.body, encoding: .utf8) ?? ""
                        if let n = Int(s.trimmingCharacters(in: .whitespaces)) {
                            out["n"] = n
                        }
                    default:
                        // Unknown field — pass through as a raw string
                        // so future extensions (e.g. `strength`) don't
                        // silently drop.
                        out[part.name] = String(data: part.body, encoding: .utf8) ?? ""
                    }
                }
                obj = out
            } else {
                // JSON body (vMLX extension).
                obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            }

            guard let request = obj else {
                return Self.errorJSON(.badRequest,
                    "Accepts multipart/form-data (OpenAI SDK) or JSON body with base64 fields.")
            }
            // iter-60: JIT wake — see embeddings rationale above.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.editImage(request: request)
                return Self.json(result)
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /v1/audio/transcriptions — ASR via native Whisper port.
        //
        // Parses OpenAI multipart/form-data (file, model, language,
        // response_format, temperature, prompt) OR a JSON body with a
        // base64 `audio` field (vMLX extension), then dispatches to
        // `Engine.transcribe`. Response is shaped per `response_format`:
        //   - "json"          → `{text: "..."}`                (default)
        //   - "text"          → plain text/plain body
        //   - "verbose_json"  → full dict (text+language+duration+segments)
        //   - "srt" / "vtt"   → single-cue caption file for the clip
        router.post("/v1/audio/transcriptions") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 128 * 1024 * 1024)
            let data = Data(buffer: body)
            let contentType = req.headers[.contentType] ?? ""

            var audioData: Data? = nil
            var fileExt: String = "wav"
            var modelName: String = ""
            var language: String? = nil
            var responseFormat: String = "json"
            var task: String = "transcribe"
            // iter-124 §199: track whether the caller sent Whisper fields
            // that our greedy decoder doesn't honor today. Previously
            // these were silently dropped — now we warn-log exactly once
            // per request so LangChain-Whisper / OpenAI-SDK users who
            // set temperature=0.5 for noisy-audio robustness can see the
            // degradation in the server log rather than wonder why their
            // noise-tolerance tuning had zero effect.
            var unwiredFields: [String] = []

            if contentType.lowercased().hasPrefix("multipart/form-data") {
                guard let boundary = MultipartFormParser.boundary(from: contentType)
                else {
                    return Self.errorJSON(.badRequest,
                        "multipart/form-data body missing boundary")
                }
                let parts = MultipartFormParser.parse(body: data, boundary: boundary)
                for part in parts {
                    switch part.name {
                    case "file":
                        audioData = part.body
                        if let fname = part.filename,
                           let dot = fname.lastIndex(of: ".")
                        {
                            // iter-80 (§108): sanitize fileExt — it
                            // flows into `WhisperAudio.decodeData`
                            // which uses it as a component of the
                            // temp file name. A filename like
                            // `audio.wav/evil` would produce ext
                            // `wav/evil`, which `appendingPathComponent`
                            // then treats as a multi-segment path.
                            // `Data.write(to:)` fails if the
                            // intermediate dir doesn't exist so it
                            // isn't actively exploitable, but we
                            // reject anything that isn't
                            // alphanumeric to keep the temp path
                            // well-formed and avoid surprising the
                            // decoder's mime sniff.
                            fileExt = sanitizeFileExt(
                                String(fname[fname.index(after: dot)...])) ?? fileExt
                        }
                    case "model":
                        modelName = String(data: part.body, encoding: .utf8) ?? ""
                    case "language":
                        language = String(data: part.body, encoding: .utf8)
                    case "response_format":
                        responseFormat = String(data: part.body, encoding: .utf8) ?? "json"
                    case "task":
                        task = String(data: part.body, encoding: .utf8) ?? "transcribe"
                    case "prompt":
                        // FIXME(iter-124 §199): `prompt` is an OpenAI
                        // Whisper field that biases the decoder with
                        // a text prefix (e.g., to normalize spelling
                        // of domain-specific terms). Accepted but
                        // not yet threaded through WhisperDecoder.
                        // Warn-log when set so the degradation is
                        // visible rather than silent.
                        let text = String(data: part.body, encoding: .utf8) ?? ""
                        if !text.isEmpty { unwiredFields.append("prompt") }
                    case "temperature":
                        // FIXME(iter-124 §199): OpenAI's Whisper API
                        // uses `temperature` for the log-prob-fallback
                        // schedule (temperatures [0, 0.2, 0.4, 0.6,
                        // 0.8, 1.0] retried until log-prob clears
                        // threshold). Our greedy decoder has no
                        // fallback path — `temperature=0.5` is
                        // equivalent to `temperature=0`. Warn so
                        // noisy-audio-tuning callers see the gap.
                        let text = String(data: part.body, encoding: .utf8) ?? ""
                        let val = Double(text) ?? 0
                        if val > 0 { unwiredFields.append("temperature=\(text)") }
                    default:
                        break
                    }
                }
            } else {
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let b64 = obj["audio"] as? String,
                       let decoded = Data(base64Encoded: b64) {
                        audioData = decoded
                    }
                    if let e = obj["file_extension"] as? String,
                       let clean = sanitizeFileExt(e)
                    {
                        fileExt = clean
                    }
                    if let m = obj["model"] as? String { modelName = m }
                    if let l = obj["language"] as? String { language = l }
                    if let r = obj["response_format"] as? String { responseFormat = r }
                    if let tk = obj["task"] as? String { task = tk }
                    // iter-124 §199: JSON-body branch parity — same
                    // unwired-field warning for callers that use the
                    // vMLX base64-JSON shape instead of multipart.
                    if let p = obj["prompt"] as? String, !p.isEmpty {
                        unwiredFields.append("prompt")
                    }
                    if let t = obj["temperature"] as? Double, t > 0 {
                        unwiredFields.append("temperature=\(t)")
                    } else if let t = obj["temperature"] as? Int, t > 0 {
                        unwiredFields.append("temperature=\(t)")
                    }
                }
            }
            if !unwiredFields.isEmpty {
                // Match the unconditional-stderr pattern used by
                // `OllamaRoutes.warnUnsupportedOllamaOptions` (iter-110
                // §188). engine.log routes to the in-process LogStore
                // which the GUI reads but doesn't mirror to the server
                // stdout, so headless callers (CLI + pytest rigs + LAN
                // hosts) don't see it. stderr is universal.
                let msg = "[whisper] transcriptions: accepted but not yet wired — "
                    + unwiredFields.joined(separator: ", ")
                    + " (greedy decoder has no temperature fallback + no prompt-prefix path; results may differ from OpenAI's reference Whisper API — see §199 FIXMEs in OpenAIRoutes.swift)\n"
                FileHandle.standardError.write(Data(msg.utf8))
            }

            guard let audio = audioData, !audio.isEmpty else {
                return Self.errorJSON(.badRequest,
                    "transcriptions: missing 'file' audio part")
            }

            var dict: [String: Any] = [
                "file": audio,
                "file_extension": fileExt,
                "task": task,
            ]
            if !modelName.isEmpty { dict["model"] = modelName }
            if let language { dict["language"] = language }

            // iter-60: JIT wake for audio transcription too.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.transcribe(request: dict)
                let text = (result["text"] as? String) ?? ""
                switch responseFormat.lowercased() {
                case "text":
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/plain; charset=utf-8"],
                        body: .init(byteBuffer: .init(string: text)))
                case "srt":
                    let dur = (result["duration"] as? Double) ?? 0
                    let body = Self.singleCueSRT(text: text, duration: dur)
                    return Response(
                        status: .ok,
                        headers: [.contentType: "application/x-subrip"],
                        body: .init(byteBuffer: .init(string: body)))
                case "vtt":
                    let dur = (result["duration"] as? Double) ?? 0
                    let body = Self.singleCueVTT(text: text, duration: dur)
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/vtt"],
                        body: .init(byteBuffer: .init(string: body)))
                case "verbose_json":
                    // iter-124 §199: `segments` was hardcoded `[]` —
                    // LangChain's Whisper loader interprets that as
                    // "no speech detected" and discards the
                    // transcription even though `text` has it.
                    // WhisperDecoder runs a single-pass greedy decode
                    // (no VAD / chunking), so synthesize ONE segment
                    // spanning the whole clip with real start=0,
                    // end=duration, and text=result.text. Passes the
                    // "has speech" test; `id=0` signals single-segment.
                    // Real per-segment data lands when the decoder
                    // grows VAD-based chunking.
                    var verbose = result
                    let text = (result["text"] as? String) ?? ""
                    let dur = (result["duration"] as? Double) ?? 0
                    let seg: [String: Any] = [
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": dur,
                        "text": text,
                        "tokens": [] as [Int],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": text.isEmpty ? 1.0 : 0.0,
                    ]
                    verbose["segments"] = text.isEmpty ? ([] as [Any]) : [seg]
                    return Self.json(verbose)
                default:
                    return Self.json(["text": text])
                }
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /v1/audio/translations — OpenAI audio-translation endpoint.
        //
        // iter-101 §179: pre-fix, this route 404'd. The transcriptions
        // handler above accepts a `task` field that lets callers opt
        // into translation, but clients using the official OpenAI SDK
        // hit the dedicated `/v1/audio/translations` path — and got a
        // 404 even though the engine supports translation. Wire it by
        // delegating to the same multipart/JSON parser with task
        // hard-coded to "translate". Per OpenAI spec, translations
        // always output English; we also drop any caller-supplied
        // `language` field since it doesn't apply.
        router.post("/v1/audio/translations") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 128 * 1024 * 1024)
            let data = Data(buffer: body)
            let contentType = req.headers[.contentType] ?? ""

            var audioData: Data? = nil
            var fileExt: String = "wav"
            var modelName: String = ""
            var responseFormat: String = "json"

            if contentType.lowercased().hasPrefix("multipart/form-data") {
                guard let boundary = MultipartFormParser.boundary(from: contentType)
                else {
                    return Self.errorJSON(.badRequest,
                        "multipart/form-data body missing boundary")
                }
                let parts = MultipartFormParser.parse(body: data, boundary: boundary)
                for part in parts {
                    switch part.name {
                    case "file":
                        audioData = part.body
                        if let fname = part.filename,
                           let dot = fname.lastIndex(of: ".")
                        {
                            fileExt = sanitizeFileExt(
                                String(fname[fname.index(after: dot)...])) ?? fileExt
                        }
                    case "model":
                        modelName = String(data: part.body, encoding: .utf8) ?? ""
                    case "response_format":
                        responseFormat = String(data: part.body, encoding: .utf8) ?? "json"
                    case "prompt", "temperature", "language", "task":
                        // Translations per OpenAI spec always output English
                        // and don't take a `language` hint; prompt /
                        // temperature are parallel to transcriptions and
                        // are accepted-ignored until the decoder supports
                        // them. Any caller-supplied `task` is ignored —
                        // this endpoint forces translate.
                        break
                    default:
                        break
                    }
                }
            } else {
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let b64 = obj["audio"] as? String,
                       let decoded = Data(base64Encoded: b64) {
                        audioData = decoded
                    }
                    if let e = obj["file_extension"] as? String,
                       let clean = sanitizeFileExt(e)
                    {
                        fileExt = clean
                    }
                    if let m = obj["model"] as? String { modelName = m }
                    if let r = obj["response_format"] as? String { responseFormat = r }
                }
            }

            guard let audio = audioData, !audio.isEmpty else {
                return Self.errorJSON(.badRequest,
                    "translations: missing 'file' audio part")
            }

            var dict: [String: Any] = [
                "file": audio,
                "file_extension": fileExt,
                "task": "translate",
            ]
            if !modelName.isEmpty { dict["model"] = modelName }

            await engine.wakeFromStandby()
            do {
                let result = try await engine.transcribe(request: dict)
                let text = (result["text"] as? String) ?? ""
                switch responseFormat.lowercased() {
                case "text":
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/plain; charset=utf-8"],
                        body: .init(byteBuffer: .init(string: text)))
                case "srt":
                    let dur = (result["duration"] as? Double) ?? 0
                    let bodyStr = Self.singleCueSRT(text: text, duration: dur)
                    return Response(
                        status: .ok,
                        headers: [.contentType: "application/x-subrip"],
                        body: .init(byteBuffer: .init(string: bodyStr)))
                case "vtt":
                    let dur = (result["duration"] as? Double) ?? 0
                    let bodyStr = Self.singleCueVTT(text: text, duration: dur)
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/vtt"],
                        body: .init(byteBuffer: .init(string: bodyStr)))
                case "verbose_json":
                    // iter-124 §199: `segments` was hardcoded `[]` —
                    // LangChain's Whisper loader interprets that as
                    // "no speech detected" and discards the
                    // transcription even though `text` has it.
                    // WhisperDecoder runs a single-pass greedy decode
                    // (no VAD / chunking), so synthesize ONE segment
                    // spanning the whole clip with real start=0,
                    // end=duration, and text=result.text. Passes the
                    // "has speech" test; `id=0` signals single-segment.
                    // Real per-segment data lands when the decoder
                    // grows VAD-based chunking.
                    var verbose = result
                    let text = (result["text"] as? String) ?? ""
                    let dur = (result["duration"] as? Double) ?? 0
                    let seg: [String: Any] = [
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": dur,
                        "text": text,
                        "tokens": [] as [Int],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": text.isEmpty ? 1.0 : 0.0,
                    ]
                    verbose["segments"] = text.isEmpty ? ([] as [Any]) : [seg]
                    return Self.json(verbose)
                default:
                    return Self.json(["text": text])
                }
            } catch let err as EngineError {
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // POST /v1/audio/speech — TTS.
        //
        // OpenAI spec: { model, input, voice, response_format?, speed? }
        // → raw audio bytes. We parse the JSON body, dispatch to
        // `engine.synthesizeSpeech`, and return the audio with the
        // appropriate content-type. The current backend is
        // `PlaceholderSynth` (non-neural tone envelope) — see
        // `Sources/vMLXTTS/TTSEngine.swift` for the handoff plan to
        // Kokoro. The `X-vMLX-TTS-Backend` header lets clients detect
        // whether they are getting neural output or the placeholder.
        router.post("/v1/audio/speech") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 4 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            // iter-60: JIT wake for TTS too.
            await engine.wakeFromStandby()
            do {
                let result = try await engine.synthesizeSpeech(request: obj)
                var headers: HTTPFields = [:]
                headers[.contentType] = result.contentType
                headers[HTTPField.Name("X-vMLX-TTS-Backend")!] = result.backend
                headers[HTTPField.Name("X-vMLX-TTS-SampleRate")!] = String(result.sampleRate)
                headers[HTTPField.Name("X-vMLX-TTS-Duration")!] = String(format: "%.3f", result.durationSec)
                var buf = ByteBuffer()
                buf.writeBytes(result.audio)
                return Response(status: .ok, headers: headers, body: .init(byteBuffer: buf))
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }

        // GET /v1/audio/voices — TTS voice catalog (vMLX extension,
        // not part of OpenAI's standard surface).
        //
        // FIXME(iter-102 §180): returns an empty `data` list with a
        // `status: "placeholder-only"` indicator while vMLXTTS ships
        // only the PlaceholderSynth backend. When the Kokoro port
        // lands (see `Sources/vMLXTTS/Kokoro/KokoroBackend.swift`),
        // wire `TTSEngine.availableVoices()` here and flip status
        // to `"kokoro"` or similar. Clients doing voice pickers
        // should read the status field to know whether to show a
        // "voices will appear once a neural backend is ported" hint
        // vs enumerating real names.
        router.get("/v1/audio/voices") { _, _ -> Response in
            Self.json([
                "object": "list",
                "data": [],
                "status": "placeholder-only",
            ])
        }

        // POST /v1/rerank — real rerank via embedding cosine similarity.
        //
        // Compatible with OpenAI's new `/rerank` and Cohere's
        // `/v1/rerank` request shapes. See `Engine.rerank` for the
        // full semantics; this just decodes the JSON and forwards.
        router.post("/v1/rerank") { req, _ -> Response in
            var req = req
            let body = try await req.collectBody(upTo: 16 * 1024 * 1024)
            let data = Data(buffer: body)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return Self.errorJSON(.badRequest, "invalid JSON body")
            }
            // iter-60: JIT wake for rerank too (embedding-model path).
            await engine.wakeFromStandby()
            do {
                let result = try await engine.rerank(request: obj)
                return Self.json(result)
            } catch let err as EngineError {
                // invalidRequest → 400; everything else → 500.
                return Self.mapEngineError(err)
            } catch {
                return Self.errorJSON(.internalServerError, "\(error)")
            }
        }
    }

    // MARK: - Helpers

    /// Encode a Swift String as a JSON string literal (with surrounding
    /// quotes). Used to embed raw model output into a JSON envelope when
    /// reporting a response_format validation failure.
    static func jsonEscape(_ s: String) -> String {
        let data = (try? JSONSerialization.data(
            withJSONObject: [s], options: [])) ?? Data("[\"\"]".utf8)
        if let str = String(data: data, encoding: .utf8),
           str.count >= 2,
           let first = str.firstIndex(of: "\""),
           let last = str.lastIndex(of: "\"")
        {
            return String(str[first...last])
        }
        return "\"\""
    }

    /// iter-117 §143: shared home-directory redaction helper. Replaces
    /// `$HOME` prefix in a filesystem path with `~/` so API responses
    /// don't disclose the user's absolute home-dir layout to API-key-
    /// holding peers (a concern for LAN-bound vMLX servers). Extracted
    /// here from the §142 AdapterRoutes copy so /v1/models and future
    /// routes share the same canonical implementation.
    static func redactHomeDir(_ path: String) -> String {
        let home = NSHomeDirectory()
        if path.hasPrefix(home) {
            return "~" + path.dropFirst(home.count)
        }
        return path
    }

    /// iter-104 §182: resolve a model's `owned_by` field from its
    /// displayName. HF-style IDs are `org/repo`; everything before the
    /// first `/` is the org. For bare-basename entries (local-only
    /// models without a matching HF repo) we report `"local"` rather
    /// than the prior mis-mapping to "dealignai" or "mlx-community".
    static func deriveOwnedBy(_ displayName: String) -> String {
        if let slash = displayName.firstIndex(of: "/"),
           slash != displayName.startIndex
        {
            return String(displayName[..<slash])
        }
        return "local"
    }

    static func json(_ obj: [String: Any], status: HTTPResponse.Status = .ok) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
        var buf = ByteBuffer()
        buf.writeBytes(data)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: buf)
        )
    }

    /// Format a duration in seconds as `HH:MM:SS,mmm` (SRT) or
    /// `HH:MM:SS.mmm` (VTT).
    static func formatTimestamp(_ seconds: Double, useComma: Bool) -> String {
        let total = max(0, seconds)
        let hours = Int(total) / 3600
        let minutes = (Int(total) % 3600) / 60
        let secs = Int(total) % 60
        let millis = Int((total - Double(Int(total))) * 1000)
        let sep = useComma ? "," : "."
        return String(format: "%02d:%02d:%02d\(sep)%03d", hours, minutes, secs, millis)
    }

    /// Emit a single-cue SRT file covering the entire clip. Real
    /// segment-aware SRT requires timestamp-token-aware decoding,
    /// which is a deferred Whisper feature.
    static func singleCueSRT(text: String, duration: Double) -> String {
        let start = formatTimestamp(0, useComma: true)
        let end = formatTimestamp(duration, useComma: true)
        return "1\n\(start) --> \(end)\n\(text)\n"
    }

    /// Emit a single-cue WebVTT file covering the entire clip.
    static func singleCueVTT(text: String, duration: Double) -> String {
        let start = formatTimestamp(0, useComma: false)
        let end = formatTimestamp(duration, useComma: false)
        return "WEBVTT\n\n\(start) --> \(end)\n\(text)\n"
    }

    static func errorJSON(_ status: HTTPResponse.Status, _ message: String) -> Response {
        Self.json([
            "error": [
                "message": message,
                "type": "api_error",
                "code": status.code,
            ] as [String: Any]
        ], status: status)
    }

    /// Single source of truth for `EngineError → HTTP status` mapping.
    /// Every route that collects a non-streaming response from the
    /// engine should funnel its `catch let err as EngineError` through
    /// this helper so third-party SDKs see contract-correct codes
    /// regardless of protocol (OpenAI / Ollama / Anthropic / Responses).
    /// Previously each site hand-rolled `if case .invalidRequest ...`
    /// and bucketed everything else to 500, which mis-reported
    /// toolChoiceNotSatisfied (should be 422), modelNotFound (404),
    /// notLoaded (503) and requestTimeout (504).
    /// **iter-67 (§96)** — OpenAI Responses API usage envelope with
    /// timing fields. Mirrors the iter-64 chat/completions envelope,
    /// iter-63 Ollama envelope, and iter-65 Anthropic envelope so the
    /// `/v1/responses` surface reports `tokens_per_second / ttft_ms /
    /// prefill_ms / total_ms` alongside the baseline token counts.
    /// The OpenAI Responses spec only requires the three token counts;
    /// extra keys are tolerated by `openai-python >= 1.40`, Cline,
    /// and other Responses clients that dispatch on known keys.
    ///
    /// Called from two sites: the non-stream response body and
    /// `SSEEncoder.responsesStream`'s `response.completed` event.
    public static func responsesUsageEnvelope(_ u: StreamChunk.Usage) -> [String: Any] {
        var r: [String: Any] = [
            "input_tokens": u.promptTokens,
            "output_tokens": u.completionTokens,
            "total_tokens": u.promptTokens + u.completionTokens,
        ]
        if let tps = u.tokensPerSecond { r["tokens_per_second"] = tps }
        // iter-126 §201: prompt_tokens_per_second parity — see the
        // same comment in the chat/completions non-stream handler
        // above. responsesUsageEnvelope is the final hold-out.
        if let pps = u.promptTokensPerSecond { r["prompt_tokens_per_second"] = pps }
        if let ttft = u.ttftMs { r["ttft_ms"] = ttft }
        if let prefill = u.prefillMs { r["prefill_ms"] = prefill }
        if let total = u.totalMs { r["total_ms"] = total }
        // iter-120 §196: cache_detail parity across all response bodies.
        if let detail = u.cacheDetail { r["cache_detail"] = detail }
        return r
    }

    /// **iter-80 (§108)** — sanitize a user-supplied audio file
    /// extension before it's joined into a temp file path inside
    /// `WhisperAudio.decodeData`. Only alphanumeric ASCII characters
    /// are allowed (1-8 chars typical: wav/mp3/m4a/flac/ogg/opus/aac
    /// /webm). Returns nil if the input has ANY forbidden character
    /// or is empty after trimming — the caller falls back to the
    /// default "wav".
    ///
    /// This is defense-in-depth: `appendingPathComponent` treats
    /// forward slashes as path separators, and `Data.write(to:)`
    /// would fail on a nonexistent intermediate directory, but an
    /// attacker who could race a sibling process to create the
    /// expected temp subdirectory might produce surprising writes.
    /// Reject the input upstream.
    public static func sanitizeFileExt(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.count <= 8 else { return nil }
        for ch in trimmed {
            guard ch.isASCII, ch.isLetter || ch.isNumber else {
                return nil
            }
        }
        return trimmed.lowercased()
    }

    static func mapEngineError(_ err: EngineError) -> Response {
        switch err {
        case .invalidRequest(let msg):
            return errorJSON(.badRequest, msg)
        case .toolChoiceNotSatisfied(let msg):
            return errorJSON(.unprocessableContent, msg)
        case .promptTooLong:
            // Hummingbird's HTTPResponse.Status lacks 413 by a stable
            // name; 400 with a descriptive message is the safe fallback.
            return errorJSON(.badRequest, err.description)
        case .modelNotFound:
            return errorJSON(.notFound, err.description)
        case .notLoaded:
            return errorJSON(.serviceUnavailable, err.description)
        case .requestTimeout:
            return errorJSON(.gatewayTimeout, err.description)
        case .toolCallRepetition:
            return errorJSON(.unprocessableContent, err.description)
        case .portInUse, .unsupportedModelType, .notImplemented,
             .adapterMissingFile, .adapterAlreadyFused, .adapterNotLoaded:
            return errorJSON(.internalServerError, err.description)
        }
    }
}
