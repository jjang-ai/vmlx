import Foundation
import CoreImage
import MLXRandom
import vMLXLMCommon

// MARK: - Engine.stream — real generation loop
//
// Replaces the `EngineError.notImplemented` stub with a live call into
// vmlx-swift-lm's `ModelContainer.generate(input:parameters:)`. The latter
// returns an `AsyncStream<Generation>` where each element is either:
//   - `.chunk(String)`  — incremental decoded text
//   - `.toolCall(ToolCall)` — parsed tool call (Harmony-style wrap)
//   - `.info(GenerateCompletionInfo)` — final metrics (tokens, t/s)
//
// This file translates each of those events to `StreamChunk` events, with
// vMLX-specific glue:
//
// 1. **Reasoning split** — route text between `<think>` / `</think>` to
//    `chunk.reasoning` unless `enableThinking == false`, in which case we
//    honor §15 NO-REGRESSION and route reasoning → content anyway so
//    always-thinking models don't strand the UI.
// 2. **Reasoning suppression** — when reasoning is off AND no tools, we
//    *could* inject `<think>\n</think>\n` at the prompt layer, but Swift
//    Chat.Message doesn't give us a prompt-injection hook. Instead we
//    rely on the content-splitter to route reasoning to visible content.
// 3. **Tool-call streaming** — feed each chunk through StreamingAccumulator
//    which detects any of the 12 `toolCallMarkers` from server.py's
//    `_TOOL_CALL_MARKERS`. When a marker fires, we buffer the stream and
//    try to parse the full tool call on each subsequent delta.
// 4. **TTFT / tok/s / cache-detail measurement** — record wall-clock for
//    the first non-empty content chunk (TTFT), count decode tokens, surface
//    `tokensPerSecond` + `promptTokensPerSecond` from `GenerateCompletionInfo`
//    in the final `StreamChunk.Usage`.
// 5. **MetricsCollector + LogStore plumbing** — `recordTokenBatch` per
//    prefill + decode burst, `recordRequest` on completion, logs at
//    category `"engine"` at start + end.
// 6. **Idle timer reset** — on stream entry so JIT wake lands the session
//    back in `.running`.
//
// The UserInput → LMInput conversion goes through
// `ModelContext.processor.prepare(input:)` inside a `container.perform`
// closure — mirrors the usage example at Evaluate.swift:1611.

extension Engine {

    /// Replace the notImplemented stub with a real generation loop.
    /// Call site is still `engine.stream(request:)` — same signature.
    ///
    /// Cancellation model
    /// ------------------
    /// vmlx-swift-lm's `TokenIterator.init` runs prefill synchronously (see
    /// `Evaluate.swift:1574` — `prepare(...)` is invoked from the iterator's
    /// initializer before the AsyncStream is even created). That means for
    /// long prefills (large prompts, cold cache), cancellation can't
    /// propagate until *after* the first token lands, because the inner
    /// `for token in iterator` loop's `Task.isCancelled` check (line 1958)
    /// doesn't run until iteration begins. This inherits the known Python
    /// "SimpleEngine prefill not interruptible" bug — we FIX it here by:
    ///
    /// 1. Exposing the driving `Task` on the Engine actor as
    ///    `currentStreamTask`, so `Engine.stop()` (or the `ChatViewModel`
    ///    stop button) can cancel it directly instead of waiting for the
    ///    outer `streamTask` to propagate.
    /// 2. Observing `Task.isCancelled` at MULTIPLE points in
    ///    `performOneGenerationPass` — before each iteration of the tool
    ///    loop, before the `container.perform` hop, and on each event of
    ///    the inner `for await` so that even though prefill itself can't
    ///    be interrupted, the *UI* flips to stopped immediately (we drop
    ///    the first token as soon as it lands and finish the stream).
    /// 3. Wiring a `withTaskCancellationHandler` around the blocking
    ///    `container.perform` call so the outer cancellation flips a
    ///    local flag the post-perform drain loop checks on entry.
    public func streamReal(
        request: ChatRequest
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                // Register the driving task on the actor so `Engine.stop()`
                // and any future admin endpoints can reach it.
                await self.setCurrentStreamTask(Task { })  // placeholder; replaced below
                do {
                    try await self.performStreamingGeneration(
                        request: request,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    await self.log(.error, "engine", "stream failed: \(error)")
                    continuation.finish(throwing: error)
                }
                await self.clearCurrentStreamTask()
            }
            // Store a reference so the engine actor can cancel this
            // specific task from `Engine.stop()`. We register it via a
            // detached fire-and-forget — the actor hop is short and by the
            // time the first event lands the registration has completed.
            Task { [weak self] in
                await self?.setCurrentStreamTask(task)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Hard ceiling used when neither the global nor session/chat
    /// settings provide a value. Matches Python server default.
    /// Most callers should never read this — the effective value
    /// comes from `resolvedGlobal.settings.defaultMaxToolIterations`
    /// which honors the 4-tier inheritance. Kept as a static constant
    /// so the outer loop has something sensible if a future
    /// refactor forgets to thread the setting.
    private static let maxToolIterationsDefault = 10

    /// The outer generation loop — runs one-or-more inner generation
    /// passes, executing tool calls between them. Exits when either:
    /// - the inner pass completes without emitting any tool calls, or
    /// - `maxToolIterations` is reached, or
    /// - the task is cancelled.
    ///
    /// Each tool call is executed via `Engine.executeToolCall(_:cwd:)`
    /// and the result is appended as a `tool` role message via
    /// `Engine.appendToolResult(to:result:)` before the next inner pass.
    /// This gives the `bash` tool (Terminal mode) a true feedback loop
    /// without leaving the engine actor.
    private func performStreamingGeneration(
        request: ChatRequest,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws {
        // 1. Preconditions
        guard self.loaded != nil else {
            throw EngineError.notImplemented(
                "Engine.stream — no model loaded (call Engine.load first)")
        }
        await self.idleTimer.reset()
        await self.log(.info, "engine",
            "stream begin: \(request.messages.count) msgs, model=\(request.model)")
        await self.metrics.incrementActiveRequests()
        // Ensure active-requests decrement no matter how we exit.
        let metricsRef = self.metrics
        defer { Task { await metricsRef.decrementActiveRequests() } }

        // Pull request timeout from resolved settings. Mirrors Python
        // `cli.py --timeout` (default 300s). A value of 0 disables the
        // watchdog entirely — matches the Python semantics. We arm the
        // watchdog BEFORE the first generation pass so runaway prefills
        // don't hold the engine forever.
        let resolvedGlobal = await self.settings.resolved(
            sessionId: nil, chatId: nil, request: nil)
        let timeoutSec: TimeInterval = resolvedGlobal.settings.requestTimeout
        let streamStart = Date()
        let timeoutTask: Task<Void, Never>? = timeoutSec > 0 ? Task { [weak self] in
            let ns = UInt64(timeoutSec * 1_000_000_000)
            try? await Task.sleep(nanoseconds: ns)
            if Task.isCancelled { return }
            // Surface the timeout to the stream before we cancel.
            continuation.yield(StreamChunk(
                finishReason: "timeout",
                usage: StreamChunk.Usage(
                    promptTokens: 0,
                    completionTokens: 0,
                    cachedTokens: 0,
                    totalMs: Date().timeIntervalSince(streamStart) * 1000,
                    isPartial: false
                )
            ))
            await self?.log(.warn, "engine",
                "request timed out after \(timeoutSec)s — cancelling stream")
            // Cancel the driving stream task; the outer Task's
            // isCancelled check will break the tool loop.
            await self?.cancelCurrentStreamTask()
        } : nil
        defer { timeoutTask?.cancel() }

        // Drive the tool-loop. `currentRequest` evolves on each iteration
        // as tool results are appended. The inner pass returns the
        // collected tool calls (if any) so we can execute + append + loop.
        //
        // Repetition detection (NO-REGRESSION §18 / project_session_2026_03_21):
        // if the model emits the exact same (name + arguments) tool call
        // 3 times in a row across tool iterations, abort the loop. This
        // catches the "nuclear retry" regression where a broken tool makes
        // the model keep retrying identical args forever.
        // Resolve the tool-iteration budget from the 4-tier settings.
        // Chat-scope override wins, then session, then global
        // default, then the static fallback. Prior behavior was a
        // hardcoded 10 — audit round 6 finding.
        let resolvedForTools = await self.settings.resolved(
            sessionId: request.sessionId.flatMap(UUID.init(uuidString:)),
            chatId: request.chatId.flatMap(UUID.init(uuidString:)),
            request: nil)
        // P2-STREAM-9 / P0-CAP gap: per-request `max_tool_iterations` wins
        // over session/global defaults. Lets a caller cap a single
        // dangerous request without touching shared settings.
        let maxIters: Int = {
            if let req = request.maxToolIterations, req > 0 { return req }
            let resolved = resolvedForTools.settings.defaultMaxToolIterations
            return max(1, resolved > 0 ? resolved : Self.maxToolIterationsDefault)
        }()

        var currentRequest = request
        var repetitionStreak: [String] = []  // signatures of recent calls
        for iteration in 0..<maxIters {
            if Task.isCancelled { break }
            let collectedCalls = try await performOneGenerationPass(
                request: currentRequest,
                continuation: continuation,
                iteration: iteration
            )
            if Task.isCancelled { break }
            if collectedCalls.isEmpty {
                // tool_choice enforcement — audit round 5 finding.
                //
                // OpenAI spec: when `tool_choice: "required"` is set,
                // the model MUST emit at least one tool call. When
                // `tool_choice: { type: "function", function: { name } }`
                // is set, the model MUST emit a call to that specific
                // function. If the model returns plain text instead,
                // client code expecting a tool call gets a malformed
                // message — the caller's SDK just silently gets a
                // text response and downstream parsers fail.
                //
                // We only enforce on the FIRST iteration because the
                // tool-loop re-enters `performOneGenerationPass` with
                // `tool` role messages appended, at which point the
                // model is supposed to summarize — no tool call on
                // the follow-up is correct behavior.
                if iteration == 0 {
                    if case .required = request.toolChoice {
                        await self.log(.warn, "engine",
                            "tool_choice=required but model emitted no tool call — surfacing as error")
                        throw EngineError.toolChoiceNotSatisfied(
                            "the request set tool_choice=\"required\" but the model returned text only")
                    }
                    if case .function(let name) = request.toolChoice {
                        await self.log(.warn, "engine",
                            "tool_choice={function:\(name)} but model emitted no tool call — surfacing as error")
                        throw EngineError.toolChoiceNotSatisfied(
                            "the request required function `\(name)` but the model returned text only")
                    }
                }
                return
            }

            // When the caller pinned a specific function, also
            // validate that at least one of the collected calls
            // actually matches. Tolerates parallel calls — as long
            // as one is the required function, we accept the batch.
            if iteration == 0, case .function(let requiredName) = request.toolChoice {
                let matched = collectedCalls.contains {
                    $0.function.name == requiredName
                }
                if !matched {
                    let got = collectedCalls.map { $0.function.name }.joined(separator: ", ")
                    await self.log(.warn, "engine",
                        "tool_choice={function:\(requiredName)} got=\(got) — surfacing as error")
                    throw EngineError.toolChoiceNotSatisfied(
                        "the request required function `\(requiredName)` but the model called [\(got)]")
                }
            }

            // Signature = name + sorted args JSON. We check BEFORE
            // executing so a stuck model can't spam the tool runtime.
            let signature = Self.toolCallSignature(collectedCalls)
            repetitionStreak.append(signature)
            if repetitionStreak.count > 3 { repetitionStreak.removeFirst() }
            if repetitionStreak.count == 3
                && Set(repetitionStreak).count == 1
            {
                await self.log(.error, "engine",
                    "tool call repetition detected — aborting loop (signature=\(signature))")
                throw EngineError.toolCallRepetition(signature)
            }

            // Split tool calls: server-side vs client-side.
            //
            // Audit 2026-04-16: previously ALL tool calls were routed
            // through `executeToolCall`. Client-registered tools (via
            // `request.tools`) that weren't `bash` or MCP-namespaced
            // returned `"Unknown tool"` which then got fed back to the
            // model as a tool result — the client never saw
            // `finish_reason: "tool_calls"` and couldn't execute their
            // own tool. Now: if ANY collected call names a tool that's
            // only in `request.tools` (not `bash`, not MCP namespaced),
            // exit the loop with a terminal `finish_reason: "tool_calls"`
            // chunk so the client can execute.
            let internalNames = Self.knownInternalToolNames(
                collectedCalls: collectedCalls)
            let hasClientTool = collectedCalls.contains { !internalNames.contains($0.function.name) }
            if hasClientTool {
                await self.log(.info, "engine",
                    "tool_calls emitted for client-side execution — exiting tool loop")
                continuation.yield(StreamChunk(
                    toolCalls: collectedCalls,
                    finishReason: "tool_calls"
                ))
                return
            }
            // Execute each internal tool call server-side and append results
            // as `tool` role messages.
            // P2-STREAM-7: check cancellation BEFORE and AFTER each tool
            // execution. The "after" check matters when bash takes
            // multiple seconds and the user clicks stop mid-execution —
            // without it, the loop would still record the result and try
            // another iteration.
            for call in collectedCalls {
                if Task.isCancelled { break }
                let result = await executeToolCall(call)
                if Task.isCancelled { break }
                await self.log(
                    .info, "tool",
                    "executed \(call.function.name) iteration=\(iteration) exit=\(result.isError ? "error" : "ok")")
                continuation.yield(StreamChunk(toolStatus: StreamChunk.ToolStatus(
                    toolCallId: result.toolCallId,
                    name: result.name,
                    phase: result.isError ? .error : .done,
                    message: result.content
                )))
                currentRequest = self.appendToolResult(to: currentRequest, result: result)
            }
        }
        await self.log(.warn, "engine",
            "stream: hit maxToolIterations (\(maxIters)) — exiting loop")
    }

    /// Classify tool calls as internal (bash, MCP-namespaced) vs client-
    /// registered. Any call whose name is neither `bash` nor contains
    /// `__` is assumed to be a client-side tool that the model emitted
    /// in response to `request.tools`; the engine should stop the loop
    /// and emit `finish_reason: "tool_calls"` so the client can run it.
    /// Audit 2026-04-16.
    private static func knownInternalToolNames(
        collectedCalls: [ChatRequest.ToolCall]
    ) -> Set<String> {
        var names = Set<String>()
        for c in collectedCalls {
            let n = c.function.name
            if n == "bash" || n.contains("__") {
                names.insert(n)
            }
        }
        return names
    }

    /// Build a stable signature for a batch of tool calls. Used by the
    /// repetition detector in `performStreamingGeneration`. Two batches
    /// with the same signature means the model is stuck re-emitting the
    /// exact same call(s). We sort by name so ordering jitter doesn't
    /// fool the detector.
    static func toolCallSignature(_ calls: [ChatRequest.ToolCall]) -> String {
        let parts = calls
            .map { "\($0.function.name)(\($0.function.arguments))" }
            .sorted()
        return parts.joined(separator: "|")
    }

    /// Run one generation pass (one call to `vMLXLMCommon.generate`).
    /// Forwards content / reasoning / usage / finishReason to the
    /// continuation, but BUFFERS tool calls into the returned array so
    /// the outer loop can execute them before deciding to continue.
    ///
    /// Returns the list of tool calls the model emitted during this pass.
    /// An empty array means the model finished normally — the outer loop
    /// exits.
    @discardableResult
    private func performOneGenerationPass(
        request: ChatRequest,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation,
        iteration: Int
    ) async throws -> [ChatRequest.ToolCall] {
        guard let container = self.loaded else {
            throw EngineError.notImplemented("Engine.stream — model unloaded mid-stream")
        }

        // 2. Settings resolution — 4-tier merge (global → session → chat → request).
        // vMLX-extension fields `session_id` / `chat_id` on the request body
        // scope the merge; vanilla OpenAI clients that omit them get the
        // 2-tier (global + request) resolution. Accept both raw UUID strings
        // and any non-UUID identifier; invalid strings fall back to nil and
        // the resolver degrades to 2-tier gracefully.
        let resolvedSessionId = request.sessionId.flatMap { UUID(uuidString: $0) }
        let resolvedChatId = request.chatId.flatMap { UUID(uuidString: $0) }
        let resolved = await self.settings.resolved(
            sessionId: resolvedSessionId,
            chatId: resolvedChatId,
            request: RequestOverride.from(request))

        // DFlash short-circuit. When enabled + target model supports it +
        // drafter is loaded + request has no tools/images, run JANG-DFlash
        // speculative decoding instead of the standard token iterator.
        // Any precondition miss returns nil → we fall through silently.
        // See `StreamDFlash.swift`.
        if let tc = try await self.tryDFlashGenerationPass(
            request: request,
            resolved: resolved,
            continuation: continuation
        ) {
            return tc
        }

        // Smelt mode is a dead setting in the Swift engine today —
        // the Python bundled engine implements partial expert loading
        // via `vmlx_engine/utils/smelt_loader.py`, but the Swift port
        // has no equivalent load-path consumer yet. When a user enables
        // it here we emit a one-shot warning so they don't silently
        // get full-precision experts while expecting memory savings.
        if resolved.settings.smelt {
            await self.log(.warn, "engine",
                "smelt mode is enabled but not wired in the Swift engine "
                + "(Python-only). Loading full experts. This warning fires "
                + "once per request — set smelt=false in Settings to silence.")
        }
        // Default-off matches Python v1.3.36 server.py and §15 contract:
        // when neither the request nor resolved settings specify a value
        // (OpenAI clients that never send enable_thinking), reasoning must
        // route to content so `delta.content` is never empty.
        // Fixes vmlx #67 (streaming empty content) + #6 (think-seed missing).
        var effectiveThinking: Bool = request.enableThinking
            ?? resolved.enableThinking
            ?? false

        // Parser auto-dispatch from the loaded model's capabilities.
        // CapabilityDetector stamps `toolParser` and `reasoningParser` on the
        // ModelCapabilities snapshot at load time (Tier-1 JANG, Tier-2
        // model_type table, or Tier-3 bronze heuristic). Prefer those over
        // the settings default so users running e.g. Qwen3 get the `qwen3`
        // reasoning parser without manually toggling settings for every
        // model swap. Settings default remains the fallback for unknown
        // models.
        let caps = self.modelCapabilities
        let toolParserName =
            caps?.toolParser ?? resolved.settings.defaultToolParser
        // `thinkInTemplate` — when the chat template already stamps
        // `<think></think>` tags itself (Qwen3, Qwen3.5, MiniMax, Step,
        // NemotronH etc.) we must NOT inject another assistant stub with
        // pre-filled think tags; doing so collides with the template's own
        // stamping and produces doubled tags. This mirrors
        // `vmlx_engine/engine/simple.py:445-453`'s `_should_inject_think`.
        let modelStampsThink = caps?.thinkInTemplate ?? false

        // 3. Build UserInput + GenerateParameters
        //    VLM image flow: `buildChatMessages` is async because decoding
        //    image parts may require base64 decode + HTTP download. The
        //    resulting `Chat.Message.images` array is flattened into
        //    `UserInput.images` automatically by `UserInput(chat:)`. For
        //    text-only requests on a VLM model, every message's `images`
        //    array is empty and the VLM processor takes its text-only path
        //    — preserving the §2026-03-25e text-only short-circuit.
        // Gate the `<think></think>` pre-fill injection: only inject when
        // the model's chat template does NOT already stamp think tags. For
        // templates that already stamp (qwen3, minimax, step3p5, nemotron_h
        // etc. — `thinkInTemplate=true`), the stream-time content router in
        // this file still routes stray reasoning chunks to visible content
        // when `effectiveThinking=false`, honoring §15 NO-REGRESSION.
        // Family-aware image marker. VLM processors do a literal split on
        // their family's image token — the wrong marker silently produces
        // a text-only prompt with zero visual tokens expanded. See
        // `buildChatMessages` comment for the per-family table.
        let imageMarker: String = {
            let fam = (caps?.family ?? "").lowercased()
            let mt = (caps?.modelType ?? "").lowercased()
            if fam.contains("gemma") || mt.contains("gemma") || mt.contains("paligemma") {
                return "<|image|>"
            }
            if fam.contains("mistral") || mt.contains("mistral") || mt.contains("pixtral") {
                return "[IMG]"
            }
            // Qwen family (qwen2_vl / qwen2_5_vl / qwen3_vl / qwen3_5_moe
            // with has_vision=true): emit the fully-expanded triple
            // `<|vision_start|><|image_pad|><|vision_end|>` inline in
            // the text. The Qwen3VL message generator is overridden
            // (see `Qwen3VLMessageGenerator.generate`) to pass
            // `content` as a plain string so the template's
            // `{%- if content is string %} {{- content }}` branch
            // renders the triple verbatim. This sidesteps a
            // swift-jinja bug in the list-of-parts code path where
            // `{%- elif content is iterable %}` with multiple
            // top-level messages dropped the `{type:"image"}`
            // dict to the literal substring `[image]` on T2+ (the
            // T1 render worked because the single-message fast path
            // took a different internal branch). Single-turn VL still
            // works; multi-turn VL now stamps N vision blocks for N
            // frames. Live-confirmed 2026-04-16 on
            // Qwen3.6-35B-A3B-JANGTQ2 with 3-turn image+text+text.
            if fam.contains("qwen") || mt.contains("qwen") {
                return "<|vision_start|><|image_pad|><|vision_end|>"
            }
            // Llava / Idefics / SmolVLM / FastVLM / LFM2 / GlmOcr etc.
            // — their templates DO accept the short `<image>` marker.
            return "<image>"
        }()
        let chatMessages = await Engine.buildChatMessages(
            from: request,
            effectiveThinking: effectiveThinking,
            modelStampsThink: modelStampsThink,
            responseFormatInstruction: Engine.responseFormatInstruction(
                from: request.responseFormat),
            imageMarker: imageMarker)

        // Merge MCP tools into the spec list the model sees.
        //
        // MCP Phase 2: `engine.mcp.listTools()` returns the flat catalog
        // across every connected server (format `server__tool`). We
        // convert each one into a `ChatRequest.Tool` and append it
        // to the request's existing tools (which might include the
        // in-process `bash` tool or user-supplied specs). The
        // dispatcher in `executeToolCall` recognizes the `__`
        // separator and routes calls through `engine.mcp.executeTool`.
        //
        // Skipped when the request has `tool_choice: .none` — MCP
        // tools don't trigger surprise activation when the caller
        // explicitly disabled tool use.
        var mergedTools: [ChatRequest.Tool]? = request.tools
        // Disambiguate ChatRequest.ToolChoice.none (the spec value) from
        // Optional<ChatRequest.ToolChoice>.none (nil = caller omitted it).
        // Nil means "no preference" — MCP tools should still merge.
        let mcpMergeBlocked: Bool = {
            guard let tc = request.toolChoice else { return false }
            if case .none = tc { return true }
            return false
        }()
        // P1-API-2: distinguish "no preference" (request.tools == nil) from
        // "explicit no tools" (request.tools == []). When the caller passes
        // an empty list, they explicitly opted out of tool dispatch — do
        // not merge MCP tools on top, even if MCP servers are connected.
        let userExplicitlyDisabledTools = (request.tools?.isEmpty == true)
        if !mcpMergeBlocked && !userExplicitlyDisabledTools {
            let mcpTools = await self.mcp.listTools()
            if !mcpTools.isEmpty {
                var merged = request.tools ?? []
                // P1-STREAM-4: dedupe MCP tools against user-supplied tools
                // by name. User-supplied tools always win — they may have
                // sandboxing or schema overrides MCP doesn't know about.
                let userToolNames = Set(merged.map { $0.function.name })
                var skipped: [String] = []
                for t in mcpTools {
                    if userToolNames.contains(t.fullName) {
                        skipped.append(t.fullName)
                        continue
                    }
                    // Round-trip the MCP schema JSON through the
                    // engine's `JSONValue` so it embeds cleanly in the
                    // OpenAI tool spec the chat template sees.
                    let parametersJSON: JSONValue? = try? JSONDecoder().decode(
                        JSONValue.self, from: t.inputSchemaJSON)
                    merged.append(ChatRequest.Tool(
                        type: "function",
                        function: ChatRequest.Tool.Function(
                            name: t.fullName,
                            description: t.description.isEmpty
                                ? "MCP tool from server '\(t.serverName)'"
                                : t.description,
                            parameters: parametersJSON
                        )
                    ))
                }
                if !skipped.isEmpty {
                    let names = skipped.joined(separator: ", ")
                    await self.log(.info, "engine",
                        "MCP tool merge: skipped \(skipped.count) name collision(s) — user tools win: \(names)")
                }
                mergedTools = merged
            }
        }

        // Gemma 4 thinking+tools collision guard (Round 16 / mlxstudio#71).
        // Gemma 4's chat template double-stamps `<think>` blocks when tools
        // are present AND reasoning is on, producing garbled output with
        // interleaved tool calls inside reasoning tags. Python v1.3.54
        // auto-disables reasoning when both conditions hold. Port the same
        // behavior to Swift — checked AFTER MCP merge so MCP-added tools
        // also trigger the guard.
        if effectiveThinking,
           let tools = mergedTools, !tools.isEmpty,
           let fam = caps?.family.lowercased(),
           fam.hasPrefix("gemma")
        {
            effectiveThinking = false
            await self.log(.warn, "engine",
                "Gemma + tools: auto-disabling thinking to prevent chat-template double-stamp (parity with Python v1.3.54 mlxstudio#71)")
        }

        let toolSpecs: [ToolSpec]? = mergedTools.map { buildToolSpecs(from: $0) }

        // Wire `enable_thinking` + `reasoning_effort` into the chat template
        // via `UserInput.additionalContext`. This is the Jinja extension
        // some templates (MiniMax M1/M2/M2.5, Mistral 4, Qwen3.5 when the
        // user wants the explicit value) read to decide whether to stamp
        // `<think>` blocks and how much effort to allocate. The Python
        // engine threads these through `_ct_kwargs`; without this wiring
        // the Swift server silently dropped them, wasting tokens on
        // reasoning even in "reasoning off" mode and producing output
        // different from Python. Deep audit 2026-04-14 #1 (HIGH).
        var templateExtras: [String: any Sendable] = [:]
        templateExtras["enable_thinking"] = effectiveThinking
        if let effort = request.reasoningEffort {
            templateExtras["reasoning_effort"] = effort
        }
        if let budget = request.thinkingBudget, budget > 0 {
            templateExtras["thinking_budget"] = budget
        }
        // OpenAI `chat_template_kwargs` — caller-provided per-request
        // template variables (e.g. assistant_prefix, enable_tools). Merged
        // **last** so callers can override engine-set extras intentionally.
        if let kwargs = request.chatTemplateKwargs {
            for (key, value) in kwargs {
                templateExtras[key] = jsonValueToSendable(value)
            }
        }
        let userInput = UserInput(
            chat: chatMessages, tools: toolSpecs,
            additionalContext: templateExtras)
        let totalImageCount = chatMessages.reduce(0) { $0 + $1.images.count }
        if totalImageCount > 0 {
            await self.log(.info, "engine",
                "stream[iter=\(iteration)]: \(totalImageCount) image(s) attached")
        }
        let params = buildGenerateParameters(request: request, resolved: resolved)

        // Seed for reproducibility (A1 fix). `GenerateParameters` has no
        // seed field — MLX uses a global PRNG — so we set it via
        // `MLXRandom.seed(_:)` right before generation. Only applied
        // when the request explicitly supplies a seed; otherwise MLX's
        // default counter keeps advancing normally so concurrent
        // sessions don't stomp on each other's entropy.
        if let seed = request.seed {
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)))
        }

        // 4. Prepare LMInput inside the model container's actor isolation
        //    (context.processor is non-Sendable), then get an AsyncStream.
        //
        // TTFT definition: wall-clock from the moment we ENTER
        // `vMLXLMCommon.generate` (i.e. just after `processor.prepare` returns
        // and the cache pre-fetch has been recorded) to the first `.chunk(_)`
        // event carrying non-empty text. This deliberately EXCLUDES settings
        // resolution, UserInput construction, `processor.prepare` and the
        // cache pre-fetch so that the reported TTFT matches "prefill +
        // first-decode step" and is comparable to Python `server.py` which
        // starts its TTFT clock at the same point.
        let requestStart = Date()           // used for totalMs end-to-end
        var prefillStart = Date()           // updated inside container.perform
        var firstTokenAt: Date?
        var pendingBuffer = ""
        var inThinkBlock = false
        let toolParser = ToolCallParserRegistry.make(toolParserName)
        let streamingAcc = StreamingAccumulator()

        // Stop-sequence matcher. Built once per request from the union of
        // request.stop + resolved.settings.stopSequences. Empty (no-op)
        // when neither side supplies any. Uses Aho-Corasick so the
        // per-byte cost is O(1) regardless of pattern count.
        let stopUnion: [String] = {
            var out: [String] = []
            if let s = request.stop { out.append(contentsOf: s) }
            // Note: stopSequences is not a GlobalSettings field today —
            // it flows into the matcher via the 4-tier resolver through
            // `RequestOverride.stopSequences` → `request.stop` above.
            // Audit 2026-04-15 initial claim that GlobalSettings had a
            // silent-drop was a false positive.
            return out
        }()
        let stopMatcher = AhoCorasickMatcher(patterns: stopUnion)
        var visibleAccumulator = ""    // running visible content for stop scanning
        var stopHit = false

        // Anthropic-compat `thinking_budget`. Caps the cumulative chars
        // emitted as reasoning. When the cap is hit mid-block we force-close
        // the reasoning segment by injecting `</think>\n` to content and
        // flip `thinkingBudgetExhausted` so any further `.reasoning` deltas
        // get rerouted to visible content. 0 (or nil) disables the cap.
        // We use a 4-chars-per-token approximation because the streaming
        // path operates on decoded text deltas; an exact token count would
        // require re-tokenizing every chunk and is not worth the perf hit.
        let thinkingBudgetTokens = max(0, request.thinkingBudget ?? 0)
        let thinkingBudgetCharCap = thinkingBudgetTokens * 4
        var reasoningCharsEmitted = 0
        var thinkingBudgetExhausted = false

        // Real reasoning parser dispatch. When CapabilityDetector stamped a
        // `reasoningParser` on the loaded model, prefer it over the hand-rolled
        // `splitThinkTags` path so harmony-style Gemma 4 / GPT-OSS streams
        // route their channel markers correctly. If the capability is unset
        // (unknown model) we fall through to `splitThinkTags` which covers the
        // three `<think>`-tag families (Qwen3 / DeepSeek-R1 / Mistral).
        //
        // The parser is fed the streaming (previous, current, delta) triple
        // and returns a ReasoningDelta with optional `.reasoning` / `.content`
        // fields. We keep `streamingAcc.feed(_:)` pointed at the parser's
        // content side so tool-call detection continues to work unchanged.
        let reasoningParserName = caps?.reasoningParser
        let reasoningParser: ReasoningParser? = reasoningParserName
            .flatMap { ReasoningParserRegistry.make($0) }
        // Reset per-request state. `thinkInPrompt` tracks whether the chat
        // template already stamped a `<think>` (Qwen3/MiniMax/Step/NemotronH
        // families do, see `modelStampsThink` above).
        // Round 16 / Rank 11 (§15 NO-REGRESSION): Harmony/GPT-OSS channel
        // markers must only be considered active when the template stamps
        // think tags AND the caller actually requested reasoning. With the
        // old `harmonyActive: modelStampsThink` wiring, a Qwen3/MiniMax/Step
        // model with `enableThinking=false` still ran the harmony branch,
        // swallowing tokens destined for visible content and producing the
        // "reasoning OFF stuck UI" regression documented in
        // `feedback_reasoning_off_ui_stuck.md`.
        let harmonyActive = modelStampsThink && effectiveThinking
        reasoningParser?.resetState(
            thinkInPrompt: modelStampsThink,
            harmonyActive: harmonyActive
        )
        var parserPrevious = ""
        var parserCurrent = ""
        // Latched flag: set true once a reasoning/channel marker has appeared
        // in the stream. Used by the cold-path optimization to skip the full
        // reasoning parser on text-only non-thinking turns (per-token parser
        // scan was O(N²) across decode). See Stream.swift:~910.
        var seenReasoningMarker = false

        // Accumulator for tool calls emitted during this pass. We BUFFER
        // these instead of yielding them directly so the outer tool-loop
        // can execute them server-side and append results before the
        // next generation pass. Each entry is also forwarded to the
        // continuation as a `.toolStatus(.started)` event so the UI can
        // render a "running bash: ls -la" card in real time.
        var collectedToolCalls: [ChatRequest.ToolCall] = []

        // Snapshot the cache coordinator into a local so the `@Sendable`
        // perform closure can see it without re-entering the actor.
        //
        // Cache pre-fetch: BEFORE calling generate, we call
        // `coordinator.fetch(tokens:)` with the tokenized prompt so we can
        // count matched tokens and label the real cache tier
        // ("paged" / "disk" / "miss"). vmlx-swift-lm's TokenIterator ALSO
        // calls fetch internally — that's fine; PagedCacheManager/DiskCache
        // lookups are idempotent and O(tokens). We discard our fetch result
        // (generate() will re-fetch and use it) and only keep the
        // hit-count + tier label for surfacing in Usage.
        //
        // Perf A/B killswitch: `VMLX_DISABLE_CACHE_COORD=1` bypasses the
        // coordinator entirely so the server path uses the same `cache=nil,
        // cacheCoordinator=nil` shape as `vmlx bench-direct`. Isolates
        // whether per-step coordinator work contributes to the Gemma 4
        // decode gap between bench-direct (97 tok/s) and serve (~63 tok/s).
        let coordinator: CacheCoordinator? =
            (ProcessInfo.processInfo.environment["VMLX_DISABLE_CACHE_COORD"] == "1")
            ? nil : self.cacheCoordinator
        struct CachePreFetch: Sendable {
            let matched: Int
            let detail: String
        }
        // Cancellation check BEFORE entering the blocking container.perform
        // hop. If the user clicked stop while we were resolving settings,
        // don't even enter prefill.
        try Task.checkCancellation()
        let performResult: (
            stream: AsyncStream<Generation>,
            cache: CachePreFetch,
            genPromptLen: Int,
            promptTokenIds: [Int]
        ) =
            try await container.perform { (ctx: ModelContext) in
                let lmInput = try await ctx.processor.prepare(input: userInput)

                // Pre-flight OOM protection. MLX Metal allocation
                // failures come back as `fatalError` — not a Swift
                // `throw` — so `do/catch` around the forward pass
                // can't intercept them. The only way to guard the
                // engine against a "500k tokens dumped into a 64k
                // model" crash is a ceiling check BEFORE the forward
                // pass starts. The ceiling lives on `GlobalSettings.maxPromptTokens`
                // (default 256k, configurable) and is 0 to disable.
                let promptTokenCount = lmInput.text.tokens.size
                let ceiling = resolved.settings.maxPromptTokens
                if ceiling > 0 && promptTokenCount > ceiling {
                    // Surface as a thrown error; the outer catch at
                    // line ~92 turns this into a clean SSE error
                    // event and the client sees a 400-style message
                    // instead of the process crashing.
                    throw EngineError.promptTooLong(
                        tokens: promptTokenCount, limit: ceiling)
                }

                // Compute `gen_prompt_len`: the number of trailing tokens
                // in `lmInput.text.tokens` that correspond to the chat
                // template's `add_generation_prompt=true` suffix. Render
                // the same message dict twice — once with the flag, once
                // without — and take the length delta. Both renderings
                // use the same underlying tokenizer, so the delta is a
                // pure tail-suffix count regardless of body length. Used
                // by `CacheCoordinator` to strip the suffix from prefix
                // cache hash keys so thinking-model multi-turn requests
                // can reuse prior-turn prefill state. Mirrors Python
                // `vmlx_engine/prefix_cache.py`. Falls back to 0 on any
                // template error — safe no-op.
                var genPromptLen = 0
                do {
                    let rawMsgs = DefaultMessageGenerator().generate(from: userInput)
                    let withGP = try ctx.tokenizer.applyChatTemplate(
                        messages: rawMsgs,
                        tools: nil,
                        additionalContext: ["add_generation_prompt": true as any Sendable]
                    )
                    let withoutGP = try ctx.tokenizer.applyChatTemplate(
                        messages: rawMsgs,
                        tools: nil,
                        additionalContext: ["add_generation_prompt": false as any Sendable]
                    )
                    if withGP.count > withoutGP.count {
                        genPromptLen = withGP.count - withoutGP.count
                    }
                } catch {
                    genPromptLen = 0
                }

                // Pre-fetch: tokenize is already done. Pull the int array out
                // of the MLXArray and consult the coordinator. `asArray(Int.self)`
                // copies eagerly so we can use it after the MLXArray drops.
                var preFetch = CachePreFetch(matched: 0, detail: "miss")
                if let coord = coordinator {
                    let tokenIds = lmInput.text.tokens.asArray(Int.self)
                    // Audit 2026-04-16: pass isMLLM for VLM pipeline so
                    // SSM companion key uses N-1 boundary (Python parity
                    // `ssm_companion_cache.py:22`). Derived from the
                    // detected model capability — .vision modality is
                    // the MLLM batch generator's domain.
                    let isMLLM = (await self.modelCapabilities?.modality == .vision)
                    // FIX-G-O (2026-04-16): compute mediaSalt here and pass
                    // to fetch() so VL multi-turn can hit prior-turn cache.
                    // Previously mediaSalt was only computed + passed on the
                    // STORE path (via `iterator.mediaSalt` captured in
                    // `cacheStoreAction`), leaving fetch() with nil salt —
                    // so VL request's fetch key was text-only while store
                    // key was text+media, guaranteeing cached=0 on every
                    // multi-turn VL request. Documented in
                    // PERF-VL-MULTITURN-2026-04-16.md candidate #3.
                    let preFetchSalt = computeMediaSalt(for: lmInput)
                    switch coord.fetch(
                        tokens: tokenIds,
                        mediaSalt: preFetchSalt,
                        genPromptLen: genPromptLen,
                        isMLLM: isMLLM
                    ) {
                    case .hit(let matched, _, let tier, _, _, _):
                        preFetch = CachePreFetch(
                            matched: matched,
                            detail: tier.rawValue
                        )
                    case .miss:
                        preFetch = CachePreFetch(matched: 0, detail: "miss")
                    }
                }
                let s = try vMLXLMCommon.generate(
                    input: lmInput,
                    parameters: params,
                    context: ctx,
                    cacheCoordinator: coordinator,
                    genPromptLen: genPromptLen)
                // Capture the full prompt token ids (1D Int array) once
                // here — used below by the SSM re-derive watcher on the
                // last pass to key the clean state into the companion
                // cache without re-crossing the actor boundary.
                let promptIds = lmInput.text.tokens.asArray(Int.self)
                return (s, preFetch, genPromptLen, promptIds)
            }
        let stream = performResult.stream
        let cachePreFetch = performResult.cache
        let capturedGenPromptLen = performResult.genPromptLen
        let capturedPromptIds = performResult.promptTokenIds
        // Start the TTFT clock AFTER prep + pre-fetch. `vMLXLMCommon.generate`
        // builds the AsyncStream synchronously so first-event-arrival is a
        // fair proxy for the first decode step landing.
        prefillStart = Date()

        // Second cancellation gate — if the user clicked stop DURING the
        // prefill blocking window (which we cannot interrupt at the MLX
        // layer), flip the UI immediately and drop the stream. The
        // background iterator task in vmlx-swift-lm will drain and
        // terminate naturally on its next loop tick.
        if Task.isCancelled {
            await self.log(.info, "engine",
                "stream cancelled post-prefill (prefill completed but output dropped)")
            return []
        }

        // Live partial-usage emission state. We emit a `StreamChunk(usage:)`
        // carrying the rolling tok/s + cumulative token count every time we
        // cross a 500ms wall-clock boundary OR every 16 decoded chunks —
        // whichever fires first — so the chat metrics strip stays alive
        // during long generations instead of only painting on the final
        // `.info` chunk.
        var decodedChunkCount = 0
        var lastPartialEmitAt = Date()
        let partialEmitInterval: TimeInterval = 0.5
        let partialEmitEveryNChunks = 16

        // Per-token metrics actor-hop killswitch. `await self.metrics.recordTokenBatch`
        // is a hop to MetricsCollector (a separate actor) executed on every
        // decode tick. Default keeps the per-token recording for live TPS
        // accuracy; `VMLX_DISABLE_PER_TOKEN_METRICS=1` replaces it with a
        // single final batch recording at end-of-stream for A/B measurement.
        let perTokenMetricsDisabled =
            ProcessInfo.processInfo.environment["VMLX_DISABLE_PER_TOKEN_METRICS"] == "1"
        // Elapsed wall-clock accumulator used when per-token recording is
        // disabled. Rolled into ONE `recordTokenBatch` call at .info time.
        var accumulatedDecodeMs: Double = 0
        var accumulatedDecodeTokens = 0

        var lastChunkAt = prefillStart
        for await event in stream {
            if Task.isCancelled { break }
            switch event {
            case .chunk(let text):
                let chunkNow = Date()
                if firstTokenAt == nil { firstTokenAt = chunkNow }

                let tickMs = chunkNow.timeIntervalSince(lastChunkAt) * 1000
                lastChunkAt = chunkNow
                decodedChunkCount += 1
                if perTokenMetricsDisabled {
                    accumulatedDecodeMs += tickMs
                    accumulatedDecodeTokens += 1
                } else {
                    await self.metrics.recordTokenBatch(
                        prefill: false, count: 1, durationMs: tickMs)
                }

                let sinceLast = chunkNow.timeIntervalSince(lastPartialEmitAt)
                if sinceLast >= partialEmitInterval
                    || decodedChunkCount % partialEmitEveryNChunks == 0
                {
                    lastPartialEmitAt = chunkNow
                    let elapsed = chunkNow.timeIntervalSince(firstTokenAt ?? chunkNow)
                    let liveTps = elapsed > 0
                        ? Double(decodedChunkCount) / elapsed
                        : 0
                    let liveTtft = firstTokenAt.map {
                        $0.timeIntervalSince(prefillStart) * 1000
                    }
                    let partialUsage = StreamChunk.Usage(
                        promptTokens: 0,
                        completionTokens: decodedChunkCount,
                        cachedTokens: cachePreFetch.matched,
                        tokensPerSecond: liveTps,
                        promptTokensPerSecond: nil,
                        ttftMs: liveTtft,
                        prefillMs: nil,
                        totalMs: chunkNow.timeIntervalSince(requestStart) * 1000,
                        cacheDetail: cachePreFetch.detail,
                        isPartial: true
                    )
                    continuation.yield(StreamChunk(usage: partialUsage))
                }

                // Two routes for carving the raw decode stream into
                // reasoning vs. content deltas:
                //
                //   A) Real parser dispatch (preferred): when the loaded
                //      model's capabilities stamp a reasoningParser name,
                //      hand each delta to the concrete parser via
                //      `extractReasoningStreaming(previous:current:delta:)`.
                //      This path handles Gemma 4 (<|channel>thought…) and
                //      GPT-OSS (Harmony <|channel|>analysis…) which the
                //      plain `<think>` splitter cannot see.
                //
                //   B) Fallback splitThinkTags: unknown-parser or
                //      legacy-unregistered models still get the rolling
                //      <think>/</think> string splitter — same behavior as
                //      before this change, so nothing regresses.
                //
                // BOTH paths must preserve §15 NO-REGRESSION: when
                // `!effectiveThinking`, any reasoning chunks are routed to
                // visible CONTENT so reasoning-off UIs never go blank. This
                // has regressed 3+ times in the Electron app — see
                // feedback_reasoning_off_ui_stuck.md.
                var splitReasoning = ""
                var splitContent = ""
                // HOT PATH OPT (perf audit 2026-04-16 per partner review):
                // The reasoning parser's `parseAccumulated(current)` scans the
                // full growing string per token = O(N²) total. For text-only
                // non-thinking requests (90%+ of traffic), this burns 30-50%
                // of decode throughput with zero output benefit. Skip when:
                //   - caller disabled thinking (`!effectiveThinking`), AND
                //   - no reasoning-start marker has appeared in the stream
                //
                // Cache `seenReasoningMarker` as a latched flag so we only
                // pay the substring scan once per token (constant work), not
                // a full re-parse. Once latched, fall back to the full parser
                // path for correctness on late-marker streams.
                if !seenReasoningMarker && (
                    text.contains("<think>")
                    || text.contains("<|channel|>")
                    || text.contains("[THINK]")
                ) {
                    seenReasoningMarker = true
                }
                let useFullParser = (reasoningParser != nil)
                    && (effectiveThinking || seenReasoningMarker)
                if useFullParser, let parser = reasoningParser {
                    parserPrevious = parserCurrent
                    parserCurrent += text
                    if let delta = parser.extractReasoningStreaming(
                        previous: parserPrevious,
                        current: parserCurrent,
                        delta: text
                    ) {
                        if let r = delta.reasoning { splitReasoning += r }
                        if let c = delta.content { splitContent += c }
                    }
                } else if reasoningParser != nil {
                    // Cold path: parser installed but no marker + reasoning off.
                    // Route delta straight to content with zero parser work.
                    // Do NOT append to parserCurrent — that would be O(N) waste.
                    // If a marker later appears, `seenReasoningMarker` latches
                    // and the next token routes through the full parser.
                    splitContent = text
                } else {
                    // Hand-rolled fallback path — held back in buffer for
                    // partial-tag safety. Unchanged from the original impl.
                    pendingBuffer += text
                    let split = splitThinkTags(
                        buffer: &pendingBuffer,
                        inThinkBlock: &inThinkBlock)
                    splitReasoning = split.reasoning
                    splitContent = split.content
                }

                if !splitReasoning.isEmpty {
                    if !effectiveThinking || thinkingBudgetExhausted {
                        // §15 reasoning-off OR budget exhausted: reroute to content.
                        continuation.yield(StreamChunk(content: splitReasoning))
                    } else if thinkingBudgetCharCap > 0 {
                        // Budget enforcement: emit up to cap, then force-close.
                        let remaining = thinkingBudgetCharCap - reasoningCharsEmitted
                        if remaining > 0 {
                            let allowed = String(splitReasoning.prefix(remaining))
                            let overflow = String(splitReasoning.dropFirst(allowed.count))
                            continuation.yield(StreamChunk(reasoning: allowed))
                            reasoningCharsEmitted += allowed.count
                            if !overflow.isEmpty {
                                // Force-close the reasoning segment with a
                                // synthetic </think> in the content stream so
                                // downstream parsers (and the chat UI) flip
                                // out of the reasoning state.
                                continuation.yield(StreamChunk(content: "</think>\n"))
                                continuation.yield(StreamChunk(content: overflow))
                                thinkingBudgetExhausted = true
                                inThinkBlock = false
                            }
                        } else {
                            // Already at cap on entry — synthesize close once.
                            continuation.yield(StreamChunk(content: "</think>\n"))
                            continuation.yield(StreamChunk(content: splitReasoning))
                            thinkingBudgetExhausted = true
                            inThinkBlock = false
                        }
                    } else {
                        // P1-STREAM-3: For in-template thinking models (Qwen3,
                        // MiniMax, Step3.5, NemotronH) the reasoning content
                        // is part of the visible-channel stream — a user-set
                        // stop string can legitimately appear inside <think>
                        // and should still halt generation. For Harmony /
                        // GPT-OSS where reasoning is on a distinct channel,
                        // reasoning never overlaps with stop targets and the
                        // scan would be pointless. Gate by the model's
                        // `thinkInTemplate` capability.
                        if !stopMatcher.isEmpty,
                           caps?.thinkInTemplate == true,
                           let m = stopMatcher.firstMatch(in: splitReasoning)
                        {
                            let pat = stopMatcher.patterns[m.patternIndex]
                            let patBytes = pat.utf8.count
                            let cutByte = m.endByteOffset + 1 - patBytes
                            let utf8 = Array(splitReasoning.utf8)
                            if cutByte > 0 {
                                let keep = String(decoding: utf8.prefix(cutByte), as: UTF8.self)
                                continuation.yield(StreamChunk(reasoning: keep))
                                reasoningCharsEmitted += keep.count
                            }
                            stopHit = true
                        } else {
                            continuation.yield(StreamChunk(reasoning: splitReasoning))
                            reasoningCharsEmitted += splitReasoning.count
                        }
                    }
                }
                if !splitContent.isEmpty {
                    if !stopMatcher.isEmpty {
                        // Aho-Corasick stop scan. Append to the running
                        // visible accumulator and look for a match. If
                        // hit, we emit only the slice up to (and not
                        // including) the matched stop string, then break
                        // out of the generation loop. This is the
                        // ONLY place where a stop string can interrupt
                        // generation — reasoning content is excluded
                        // because stop sequences target visible output.
                        let preLen = visibleAccumulator.utf8.count
                        visibleAccumulator += splitContent
                        if let m = stopMatcher.firstMatch(in: visibleAccumulator) {
                            let pat = stopMatcher.patterns[m.patternIndex]
                            let patBytes = pat.utf8.count
                            let matchStartByte = m.endByteOffset + 1 - patBytes
                            // How much of the stop string lies inside the
                            // current delta? Anything before that goes out
                            // unchanged; nothing after it does.
                            let keepBytesInDelta = max(0, matchStartByte - preLen)
                            let utf8 = Array(splitContent.utf8)
                            if keepBytesInDelta > 0 {
                                let keep = String(decoding: utf8.prefix(keepBytesInDelta), as: UTF8.self)
                                continuation.yield(StreamChunk(content: keep))
                            }
                            stopHit = true
                        } else {
                            continuation.yield(StreamChunk(content: splitContent))
                        }
                    } else {
                        continuation.yield(StreamChunk(content: splitContent))
                    }
                }
                if stopHit { break }

                // Tool-call streaming accumulator — detects markers and
                // parses the full tool call when enough bytes have accrued.
                // BUFFER into `collectedToolCalls` — the outer tool-loop
                // executes them between passes.
                //
                // Skip entirely when the request has no tools AND no tool
                // parser installed — the accumulator's `feed()` grows an
                // unbounded buffer and scans 12 tool-call markers every
                // token. On a 200-token response this is O(N²) wasted
                // work (≈1M substring comparisons total) with zero output
                // effect. Tool parser will be nil when neither the loaded
                // model stamped a `toolParser` capability nor the user
                // supplied `request.tools`. 2026-04-16 gemma4 decode
                // investigation — measured kill factor per-token.
                let haveToolsOrParser = toolParser != nil
                    || (mergedTools?.isEmpty == false)
                if haveToolsOrParser {
                    _ = streamingAcc.feed(splitContent)
                }
                if haveToolsOrParser, streamingAcc.buffered, let parser = toolParser {
                    let parsed = parser.parse(streamingAcc.current)
                    if !parsed.isEmpty {
                        for p in parsed {
                            let call = ChatRequest.ToolCall(
                                id: "call_\(UUID().uuidString.prefix(8))",
                                type: "function",
                                function: .init(
                                    name: p.name,
                                    arguments: p.argumentsJSON ?? "{}"))
                            collectedToolCalls.append(call)
                            // Fan out a `started` status so the UI knows
                            // the tool is about to execute.
                            continuation.yield(StreamChunk(
                                toolCalls: [call],
                                toolStatus: StreamChunk.ToolStatus(
                                    toolCallId: call.id,
                                    name: call.function.name,
                                    phase: .started
                                )
                            ))
                        }
                        streamingAcc.reset()
                    }
                }

            case .toolCall(let call):
                // vmlx-swift-lm already parsed a Harmony-style call.
                // `call.function.arguments` is `[String: vMLXLMCommon.JSONValue]`
                // — disambiguate from our own vMLXEngine.JSONValue.
                let mlxArgs: [String: vMLXLMCommon.JSONValue] = call.function.arguments
                let argsJSON = encodeMLXJSONValueDict(mlxArgs) ?? "{}"
                let bufferedCall = ChatRequest.ToolCall(
                    id: "call_\(UUID().uuidString.prefix(8))",
                    type: "function",
                    function: .init(name: call.function.name, arguments: argsJSON))
                collectedToolCalls.append(bufferedCall)
                continuation.yield(StreamChunk(
                    toolCalls: [bufferedCall],
                    toolStatus: StreamChunk.ToolStatus(
                        toolCallId: bufferedCall.id,
                        name: bufferedCall.function.name,
                        phase: .started
                    )
                ))

            case .info(let info):
                // End-of-stream flush for the hand-rolled `splitThinkTags`
                // path. That splitter holds back up to 8 trailing chars in
                // `pendingBuffer` for partial-tag safety (so a `</thi` at
                // chunk boundary isn't emitted as content before the next
                // chunk arrives with `nk>`). When the stream ends those
                // trailing bytes would be silently dropped — which is
                // EXACTLY the `6-7 tokens decoded but content=\"2\"` bug
                // on short Llama/Gemma replies that end on a stop token
                // before the 8-char safety window drains naturally.
                if reasoningParser == nil && !pendingBuffer.isEmpty {
                    let tail = pendingBuffer
                    pendingBuffer = ""
                    if inThinkBlock {
                        if effectiveThinking {
                            continuation.yield(StreamChunk(reasoning: tail))
                        } else {
                            // §15 no-regression — reasoning-off reroutes to content.
                            continuation.yield(StreamChunk(content: tail))
                        }
                    } else {
                        continuation.yield(StreamChunk(content: tail))
                    }
                    _ = streamingAcc.feed(tail)
                }

                // End-of-stream flush for the parser-dispatch path. Some
                // parsers (Gemma 4) buffer the first ~18 characters waiting
                // for a `<|channel>thought` marker before emitting anything.
                // On very short replies (<18 chars) the buffered content is
                // otherwise lost. `finishStreaming(fullText:)` asks the
                // parser to drain its residual; default impl returns nil.
                if let parser = reasoningParser,
                   let residual = parser.finishStreaming(fullText: parserCurrent)
                {
                    if let r = residual.reasoning, !r.isEmpty {
                        if effectiveThinking {
                            continuation.yield(StreamChunk(reasoning: r))
                        } else {
                            continuation.yield(StreamChunk(content: r))
                        }
                    }
                    if let c = residual.content, !c.isEmpty {
                        continuation.yield(StreamChunk(content: c))
                        _ = streamingAcc.feed(c)
                    }
                }

                // Final metrics — populate Usage and emit the finish chunk.
                // TTFT is measured from `prefillStart` (entry to generate())
                // to the first content-bearing `.chunk`. `prefillMs` is the
                // authoritative prompt-processing time from mlx-swift-lm.
                let ttftMs: Double? = firstTokenAt.map {
                    $0.timeIntervalSince(prefillStart) * 1000
                }
                let totalMs = Date().timeIntervalSince(requestStart) * 1000
                // Cache details surfaced from our explicit pre-fetch call
                // above — `cachePreFetch.matched` is the real number of
                // prompt tokens served from the paged/disk tier, and
                // `cachePreFetch.detail` is the tier label. Previously this
                // was hardcoded to 0 / "paged+disk" because mlx-swift-lm
                // doesn't thread the fetch result back through
                // GenerateCompletionInfo.
                let usage = StreamChunk.Usage(
                    promptTokens: info.promptTokenCount,
                    completionTokens: info.generationTokenCount,
                    cachedTokens: cachePreFetch.matched,
                    tokensPerSecond: info.tokensPerSecond,
                    promptTokensPerSecond: info.promptTokensPerSecond,
                    ttftMs: ttftMs,
                    prefillMs: info.promptTime * 1000,
                    totalMs: totalMs,
                    cacheDetail: cachePreFetch.detail,
                    isPartial: false
                )
                // Only record the PREFILL batch here — per-token decode
                // ticks were already recorded inside the `.chunk` branch
                // above, so recording the decode total again would double-
                // count and blow up the rolling window.
                await self.metrics.recordTokenBatch(
                    prefill: true,
                    count: info.promptTokenCount,
                    durationMs: info.promptTime * 1000)
                if perTokenMetricsDisabled && accumulatedDecodeTokens > 0 {
                    // Flush the batched decode tally as a single hop.
                    await self.metrics.recordTokenBatch(
                        prefill: false,
                        count: accumulatedDecodeTokens,
                        durationMs: accumulatedDecodeMs)
                }
                await self.metrics.recordRequest(latencyMs: totalMs)
                await self.log(.info, "engine",
                    "stream done[iter=\(iteration)]: \(info.generationTokenCount) toks, "
                    + "\(String(format: "%.1f", info.tokensPerSecond)) t/s, "
                    + "ttft=\(ttftMs.map { String(format: "%.0fms", $0) } ?? "n/a"), "
                    + "tool_calls=\(collectedToolCalls.count)")
                // Only emit the terminal finishReason when this is the
                // LAST pass (no tool calls to run). Otherwise the outer
                // loop continues and a subsequent pass will emit the
                // real finish chunk.
                if collectedToolCalls.isEmpty {
                    continuation.yield(StreamChunk(
                        finishReason: stopHit ? "stop" : mapStopReason(info.stopReason),
                        usage: usage))
                } else {
                    // Still surface usage mid-loop so the chat metrics
                    // strip shows cumulative tok/s. No finishReason.
                    continuation.yield(StreamChunk(usage: usage))
                }

            }
        }

        // Hybrid + thinking SSM re-derive — only runs on the LAST pass
        // (empty tool calls means the generation terminated naturally)
        // for hybrid-SSM models with `gen_prompt_len > 0` where the
        // normal store path elided the SSM companion due to post-gen
        // contamination. Runs a fresh prompt-only forward pass on the
        // stripped tokens and installs the clean state in the companion
        // cache so the next turn's prefix fetch hits.
        //
        // FIX-G-A (2026-04-16): dispatch to a detached Task so the
        // re-derive runs AFTER the current stream returns, without
        // blocking the ModelContext actor for ~1-2s. Previously ran
        // synchronously inside `container.perform` which stalled the
        // next request while the re-derive finished. The detached Task
        // still acquires the container (sequentially) but the current
        // user's request is already complete. Escape hatch via
        // `VMLX_DISABLE_SSM_RE_DERIVE_ASYNC=1` keeps the old
        // synchronous behavior for debug / A/B.
        //
        // Finding #1 from the deep hybrid audit 2026-04-14 — before
        // this wire-up, `maybeReDeriveSSMState` was unreachable dead
        // code and every hybrid+thinking turn lost the SSM companion.
        if collectedToolCalls.isEmpty,
           let coord = coordinator,
           coord.isHybrid,
           capturedGenPromptLen > 0,
           resolved.settings.enableSSMReDerive
        {
            let syncReDeriveForced =
                ProcessInfo.processInfo.environment["VMLX_DISABLE_SSM_RE_DERIVE_ASYNC"] == "1"
            if syncReDeriveForced {
                await container.perform { (ctx: ModelContext) in
                    maybeReDeriveSSMState(
                        coordinator: coord,
                        model: ctx.model,
                        promptTokenIds: capturedPromptIds,
                        genPromptLen: capturedGenPromptLen,
                        enableSSMReDerive: true)
                }
            } else {
                // Capture inputs into a Sendable closure and fire-and-forget.
                // Task is detached so the Engine actor doesn't wait on it.
                let captureCoord = coord
                let captureIds = capturedPromptIds
                let captureGP = capturedGenPromptLen
                let captureContainer = container
                Task.detached(priority: .utility) {
                    await captureContainer.perform { (ctx: ModelContext) in
                        maybeReDeriveSSMState(
                            coordinator: captureCoord,
                            model: ctx.model,
                            promptTokenIds: captureIds,
                            genPromptLen: captureGP,
                            enableSSMReDerive: true)
                    }
                }
            }
        }

        // Return whatever tool calls the model emitted this pass.
        return collectedToolCalls
    }

    // MARK: - Helpers

    /// Build the Chat.Message array for UserInput.
    ///
    /// When `effectiveThinking == false` AND the request carries no tools,
    /// we append a trailing assistant stub with pre-filled
    /// `<think>\n</think>\n\n` content. The chat template renders this as a
    /// prior assistant turn and the model's next decode starts already
    /// *outside* the think block — i.e. it skips its own reasoning phase
    /// entirely. This mirrors the Python engine's prompt-level injection
    /// in `vmlx_engine/engine/simple.py:445-453`.
    ///
    /// When tools are present we DO NOT inject — the model needs its own
    /// reasoning space to choose a tool. Matches the `server.py` comment
    /// on the Python side.
    ///
    /// §15 NO-REGRESSION: this is ADDITIVE. The stream-time content-router
    /// in `performOneGenerationPass` still routes any residual reasoning
    /// chunks to visible content when `!effectiveThinking`.
    /// Maximum total number of inline images that can be passed to a VLM
    /// processor in a single request. Mirrors the Qwen2.5-VL / Gemma 4 /
    /// mlx-vlm processor caps — exceeding this risks OOM or silent
    /// truncation inside the HF processor. When a multi-turn chat racks
    /// up more images than this, we DROP the OLDEST images (keeping the
    /// most recent N) and log a warning.
    public static let maxTotalImagesPerRequest: Int = 30

    static func buildChatMessages(
        from request: ChatRequest,
        effectiveThinking: Bool,
        modelStampsThink: Bool = false,
        responseFormatInstruction: String? = nil,
        imageMarker: String = "<image>"
    ) async -> [Chat.Message] {
        var out: [Chat.Message] = []
        out.reserveCapacity(request.messages.count + 1)
        // OpenAI `response_format` injection: prepend a system message
        // instructing the model to emit a JSON object (or to follow a
        // schema). This is the same approach the Python engine uses when
        // grammar-biased sampling isn't available — and matches what
        // mainstream tool-calling clients expect from a JSON-mode endpoint.
        if let instruction = responseFormatInstruction, !instruction.isEmpty {
            out.append(Chat.Message(role: .system, content: instruction))
        }
        // Two-pass: first decode every message's text + images, then apply
        // a global image-budget cap across the full conversation. This
        // preserves §VLM-MULTITURN: images from turn 1 + turn 2 BOTH reach
        // the processor on turn 3, so the model can cross-reference them.
        var decoded: [(
            role: Chat.Message.Role,
            text: String,
            images: [UserInput.Image],
            videos: [UserInput.Video]
        )] = []
        decoded.reserveCapacity(request.messages.count)
        for msg in request.messages {
            let role: Chat.Message.Role
            switch msg.role.lowercased() {
            case "system": role = .system
            case "user": role = .user
            case "assistant": role = .assistant
            case "tool": role = .tool
            default: role = .user
            }
            let text = extractText(from: msg.content)
            // Only user (and rarely system) turns carry images in practice.
            // We still run extraction unconditionally so an assistant turn
            // that references a prior image round-trips correctly.
            let images = await extractImages(from: msg.content)
            let videos = await extractVideos(from: msg.content)
            decoded.append((role: role, text: text, images: images, videos: videos))
        }

        // Cap total images across the conversation. If we're over budget,
        // drop from the OLDEST user turns first — the most recent turn
        // almost always needs its images intact for the current question.
        let totalImages = decoded.reduce(0) { $0 + $1.images.count }
        if totalImages > Self.maxTotalImagesPerRequest {
            var toDrop = totalImages - Self.maxTotalImagesPerRequest
            for i in decoded.indices where toDrop > 0 {
                let have = decoded[i].images.count
                if have == 0 { continue }
                let drop = min(have, toDrop)
                decoded[i].images.removeFirst(drop)
                toDrop -= drop
            }
            // Logging: can't hop to the Engine actor from a static helper,
            // so we print. The outer performOneGenerationPass already logs
            // total image count — this cap event is secondary.
            print(
                "[vMLXEngine] multi-turn VL image cap: kept "
                + "\(Self.maxTotalImagesPerRequest) of \(totalImages) images "
                + "(oldest dropped)"
            )
        }

        // VL image marker auto-insertion (perf+correctness audit 2026-04-16).
        // The VLM processor pipeline (both Gemma4 and Qwen2.5-VL) expects the
        // rendered prompt text to contain a special image placeholder token
        // that the chat template / processor expands to N soft-image tokens.
        // Both UIs (OpenAI `image_url` parts + Ollama `images:` array) strip
        // or never emit this marker — so if the user sends ONLY images +
        // caption text, the chat template stamps no image tokens and the
        // model literally responds "I don't see an image". Live repro on
        // Gemma4 E4B: prompt_tokens jumps 32 → 313 after marker insertion
        // and the model correctly identifies the image.
        //
        // Marker tokens (Python v1.3.29 and upstream mlx-vlm agree):
        //   gemma4     → "<|image|>"
        //   gemma3     → "<|image|>"
        //   paligemma  → "<|image|>"
        //   qwen2_vl   → "<image>"
        //   qwen2_5_vl → "<image>"
        //   qwen3_vl   → "<image>"
        //   idefics3   → "<image>"
        //   smolvlm2   → "<image>"
        //   llava*     → "<image>"
        //   pixtral    → "[IMG]"
        //   mistral3   → "[IMG]"
        //   mistral4   → "[IMG]"
        //
        // Family-aware marker selection: each VLM processor does a literal
        // split on its own token, so the WRONG marker means zero image
        // tokens are expanded and the model sees pure text ("I don't see
        // an image"). Caller passes the family-appropriate marker based on
        // `modelCapabilities.family`. Default `<image>` is the most common
        // form (Qwen / Llava / Idefics family) and mostly-tolerant for
        // unknown families since many processors also accept it.
        for d in decoded {
            var renderedText = d.text
            // Check for ANY of the known markers so we don't double-insert
            // if the user already included one in the prompt themselves.
            let hasAnyMarker =
                renderedText.contains("<|image|>")
                || renderedText.contains("<image>")
                || renderedText.contains("[IMG]")
            if !d.images.isEmpty && !hasAnyMarker {
                // One marker per image, prefixed before the text. No newline
                // between marker and text — some chat templates split on
                // whitespace and the newline breaks image-token expansion.
                // Python engine convention: concat directly.
                let markers = String(repeating: imageMarker, count: d.images.count)
                renderedText = markers + renderedText
            }
            // Video marker auto-insert for Qwen-family VLMs (Qwen2.5-VL,
            // Qwen3-VL) that use `<video>` as the video placeholder. Other
            // families don't accept video today — no per-family branch
            // needed yet; Qwen is the only video-capable family we register.
            if !d.videos.isEmpty && !renderedText.contains("<video>") {
                let vMarkers = String(repeating: "<video>", count: d.videos.count)
                renderedText = vMarkers + renderedText
            }
            out.append(Chat.Message(
                role: d.role,
                content: renderedText,
                images: d.images,
                videos: d.videos))
        }
        let hasTools = !(request.tools?.isEmpty ?? true)
        // P1-STREAM-5: when response_format is set (json_object/json_schema),
        // the model is being told to emit a specific structure as its only
        // output. Injecting a `<think></think>` stub before that confuses
        // models like Qwen3 that don't have a separate reasoning channel —
        // the JSON ends up inside the think block. Suppress the stub when
        // a non-text response_format is present; the response_format
        // instruction message handles the "no reasoning" intent on its own.
        let hasStructuredFormat: Bool = {
            guard let rf = request.responseFormat, let type = rf.type else { return false }
            return type == "json_object" || type == "json_schema"
        }()
        // Skip injection when the template already stamps think tags —
        // doubling would confuse the parser (see ModelCapabilities
        // `thinkInTemplate`).
        if !effectiveThinking && !hasTools && !modelStampsThink && !hasStructuredFormat {
            out.append(Chat.Message(role: .assistant, content: "<think>\n</think>\n\n"))
        }
        return out
    }

    /// Build a system-message instruction that tells the model to emit
    /// JSON matching the requested `response_format`. Returns nil when
    /// the format is plain text (or absent).
    ///
    /// `json_object` produces a generic instruction; `json_schema` adds the
    /// schema body verbatim so the model can see field names + types.
    /// Real grammar-biased decoding would be ideal but mlx-swift does not
    /// expose constrained sampling yet, so we fall back to instruction +
    /// post-validation (see `validateResponseFormat`).
    static func responseFormatInstruction(
        from format: ChatRequest.ResponseFormat?
    ) -> String? {
        guard let format = format, let type = format.type else { return nil }
        switch type {
        case "text":
            return nil
        case "json_object":
            return "You must respond with a single valid JSON object. Do not include any prose, markdown code fences, or commentary outside the JSON."
        case "json_schema":
            var lines = [
                "You must respond with a single valid JSON object that strictly conforms to the schema below.",
                "Do not include any prose, markdown code fences, or commentary outside the JSON.",
            ]
            if let schemaSpec = format.jsonSchema {
                if let name = schemaSpec.name {
                    lines.append("Schema name: \(name)")
                }
                if let desc = schemaSpec.description {
                    lines.append("Schema description: \(desc)")
                }
                if let schema = schemaSpec.schema,
                   let data = try? JSONEncoder().encode(schema),
                   let json = String(data: data, encoding: .utf8)
                {
                    lines.append("Schema:")
                    lines.append(json)
                }
            }
            return lines.joined(separator: "\n")
        default:
            return nil
        }
    }

    /// Validate that `text` parses as JSON (and, if the format is
    /// `json_schema`, that the top-level value is an object). Returns nil
    /// on success, or a human-readable reason on failure. Callers convert
    /// the failure into either a 400-style error response or a finish
    /// reason so SDK callers can detect the contract was violated.
    public static func validateResponseFormat(
        _ text: String,
        format: ChatRequest.ResponseFormat?
    ) -> String? {
        guard let format = format, let type = format.type else { return nil }
        if type == "text" { return nil }
        if type != "json_object" && type != "json_schema" { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else {
            return "response is not valid UTF-8"
        }
        do {
            let parsed = try JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
            if type == "json_schema" || type == "json_object" {
                guard parsed is [String: Any] else {
                    return "response_format=\(type) requires a JSON object at the top level"
                }
            }
            return nil
        } catch {
            return "response is not valid JSON: \(error.localizedDescription)"
        }
    }

    /// Flatten multimodal content to text. Image parts are extracted
    /// separately via `extractImages(from:)` and routed through
    /// `Chat.Message.images` → `UserInput.images` → the VLM processor.
    static func extractText(from content: ChatRequest.ContentValue?) -> String {
        switch content {
        case .string(let s): return s
        case .parts(let parts):
            return parts.compactMap { $0.type == "text" ? $0.text : nil }
                .joined(separator: "\n")
        case nil: return ""
        }
    }

    /// Walk a `ChatRequest.Message.content` and pull out every `image_url`
    /// part, decoding each to a `UserInput.Image.ciImage`. Supports data
    /// URLs, file URLs, http(s) URLs and bare base64 (Ollama convention).
    ///
    /// Decoded bytes that don't parse as a valid CIImage are silently
    /// dropped — the VLM processor is given a best-effort list so the
    /// generation can still proceed with whatever images did load.
    static func extractImages(
        from content: ChatRequest.ContentValue?
    ) async -> [UserInput.Image] {
        // Debug: log what shape we got.
        switch content {
        case .none:
            FileHandle.standardError.write(Data("[vmlx][vl][debug] content is nil\n".utf8))
            return []
        case .some(.string):
            // Text-only content is the common case — don't spam the logs.
            return []
        case .some(.parts(let parts)):
            let kinds = parts.map { $0.type }.joined(separator: ",")
            FileHandle.standardError.write(Data(
                "[vmlx][vl][debug] content parts=\(parts.count) types=[\(kinds)]\n".utf8))
            var out: [UserInput.Image] = []
            for part in parts where part.type == "image_url" {
                guard let imageURL = part.imageUrl else {
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl][debug] image_url part missing imageUrl field\n".utf8))
                    continue
                }
                FileHandle.standardError.write(Data(
                    "[vmlx][vl][debug] loading image from: \(imageURL.url.prefix(60))…\n".utf8))
                guard let data = await imageURL.loadImageData() else {
                    let tag = imageURL.url.prefix(80)
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] image URL fetch/decode failed: \(tag)…\n".utf8))
                    continue
                }
                FileHandle.standardError.write(Data(
                    "[vmlx][vl][debug] image bytes=\(data.count)\n".utf8))
                guard let ci = CIImage(data: data) else {
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] CIImage decode failed (\(data.count) bytes) — possibly a non-image format; text-only reply will follow\n".utf8))
                    continue
                }
                FileHandle.standardError.write(Data(
                    "[vmlx][vl][debug] CIImage loaded OK, extent=\(ci.extent)\n".utf8))
                out.append(.ciImage(ci))
            }
            return out
        }
    }

    /// Extract video parts from a `ChatRequest.Message.content` multimodal
    /// payload into `UserInput.Video` values. Each part must have
    /// `type == "video_url"` and a `video_url.url` string; supported
    /// URL shapes include `file://`, `http(s)://`, `data:video/*;base64,`
    /// and bare absolute paths. FIX-G-P (2026-04-16) — adds HTTP-API video
    /// ingestion for Qwen2.5-VL / Qwen3-VL (previously only UserInput-level
    /// callers could supply videos, server path silently dropped them).
    ///
    /// HTTP and data URLs are staged to the system temp dir so
    /// `AVURLAsset` can mmap them. Temp files live for the process
    /// lifetime; callers should clean up via OS-managed temp sweep.
    static func extractVideos(
        from content: ChatRequest.ContentValue?
    ) async -> [UserInput.Video] {
        switch content {
        case .none, .some(.string):
            return []
        case .some(.parts(let parts)):
            var out: [UserInput.Video] = []
            for part in parts where part.type == "video_url" {
                guard let videoURL = part.videoUrl,
                      let localURL = await videoURL.loadVideoLocalURL()
                else {
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] video_url fetch/decode failed\n".utf8))
                    continue
                }
                FileHandle.standardError.write(Data(
                    "[vmlx][vl][debug] video staged at \(localURL.path)\n".utf8))
                out.append(.url(localURL))
            }
            return out
        }
    }

    /// Convert OpenAI-style tools to vmlx-swift-lm ToolSpec dicts. JSON
    /// primitives are all Sendable (String/Int/Double/Bool/NSNumber), but
    /// `JSONSerialization.jsonObject` returns `Any`, so we round-trip via
    /// NSDictionary to satisfy the Sendable marker protocol.
    private func buildToolSpecs(from tools: [ChatRequest.Tool]) -> [ToolSpec] {
        tools.map { t in
            var fn: [String: any Sendable] = [
                "name": t.function.name,
                "description": t.function.description ?? "",
            ]
            // Round-trip the parameters JSON string through `String` —
            // templates only read field names and descriptions, and
            // `Sendable` can't be used as a conditional cast so we store
            // the raw JSON string and let the template layer parse it if
            // it cares. This is the same trade-off the Python side makes.
            if let params = t.function.parameters,
               let data = try? JSONEncoder().encode(params),
               let jsonString = String(data: data, encoding: .utf8) {
                fn["parameters_json"] = jsonString
            }
            return ["type": t.type, "function": fn as [String: any Sendable]] as ToolSpec
        }
    }

    /// Build vMLXLMCommon.GenerateParameters from request + resolved settings.
    private func buildGenerateParameters(
        request: ChatRequest,
        resolved: ResolvedSettings
    ) -> GenerateParameters {
        var params = GenerateParameters()
        params.prefillStepSize = resolved.settings.prefillStepSize
        params.maxTokens = request.maxTokens ?? resolved.maxTokens
        params.temperature = Float(request.temperature ?? resolved.temperature)
        params.topP = Float(request.topP ?? resolved.topP)
        params.topK = request.topK ?? resolved.topK
        params.minP = Float(request.minP ?? resolved.minP)
        params.repetitionPenalty = Float(
            request.repetitionPenalty ?? resolved.repetitionPenalty)

        // Whole-model compiled decode (perf audit 2026-04-16). Enabling
        // `enableCompiledDecode` wraps the entire model forward pass in
        // `compile(shapeless: true)` at decode time — collapses the
        // MoE routing + attention + FFN into ONE Metal dispatch per step
        // (vs N separate dispatches). Matches Inferencer.app's approach
        // which achieves 100 tok/s on Qwen 35B A3B MoE per reference's
        // own SWIFT-PERF-FIXES.md. Requires CompilableKVCache which
        // uses fixed-size buffers so compile() can trace through the
        // forward. Enable via `VMLX_COMPILED_DECODE=1` env for A/B.
        if ProcessInfo.processInfo.environment["VMLX_COMPILED_DECODE"] == "1" {
            params.enableCompiledDecode = true
        }

        // TurboQuant KV-cache compression. Default on for every model
        // (MLX + JANG alike) per user directive. `enableTurboQuant`
        // in GlobalSettings defaults to true, and `turboQuantBits`
        // defaults to 4 (≈3.6x compression, sweet spot from the TQ
        // paper). Hybrid-SSM models are safe because
        // `maybeQuantizeKVCache` only compresses `KVCacheSimple`
        // layers and skips Mamba/Rotating/CacheList — see
        // `KVCache.swift:1666-1685`.
        //
        // We set both `kvMode = .turboQuant(...)` (which takes
        // precedence in `maybeQuantizeKVCache`) AND clear the legacy
        // `kvBits` field so the two paths don't collide.
        // MLA ⊥ TurboQuant. MLA (Multi-head Latent Attention — DeepSeek
        // V2/V3, Mistral 4, GLM-5) keeps a tiny *latent* KV per layer
        // instead of full per-head K/V. The TurboQuant path
        // `maybeQuantizeKVCache` only knows how to compress
        // `KVCacheSimple` (the shape MLA does not produce), and would
        // either silently no-op or hit a fatalError on the layout
        // mismatch. Python's `model_inspector.py` enforces the same
        // exclusion. Skip TQ wiring entirely when the loaded model is
        // MLA — let it use the native MLA cache instead.
        let cacheTypeIsMLA = (self.modelCapabilities?.cacheType == "mla")
        // Runtime override for A/B perf testing. `VMLX_DISABLE_TURBO_QUANT=1`
        // forces TQ KV compression off without rebuilding or mucking with
        // settings files, so we can measure the decode-time cost of the TQ
        // compress+dequant cycle in isolation.
        let tqDisabledViaEnv =
            ProcessInfo.processInfo.environment["VMLX_DISABLE_TURBO_QUANT"] == "1"

        // JANG auto-activation: if the loaded model's `jang_config.json`
        // declares `"turboquant": {"enabled": true, ...}`, activate TQ
        // REGARDLESS of the global `enableTurboQuant` flag. These models
        // are calibrated with specific bit widths per layer role and expect
        // the KV to be compressed at generate-time. The global flag only
        // governs MLX-format models without a calibrated TQ block.
        // Matches Engine.swift:36-41 documented intent.
        //
        // Precedence:
        //   1. MLA model       → skip (layer shape incompatible)
        //   2. env killswitch  → skip
        //   3. JANG calibrated → activate with jang bits
        //   4. global toggle   → activate with global turboQuantBits
        //   5. otherwise       → skip (default OFF post 2026-04-16 audit)
        let jangTQEnabled = (self.loadedJangConfig?.turboquant.enabled == true)
        let shouldActivateTQ = !cacheTypeIsMLA && !tqDisabledViaEnv && (
            jangTQEnabled || resolved.settings.enableTurboQuant
        )
        if shouldActivateTQ {
            let keyBits: Int
            let valueBits: Int
            if let tq = self.loadedJangConfig?.turboquant, tq.enabled {
                keyBits = tq.defaultKeyBits
                valueBits = tq.defaultValueBits
            } else {
                let bits = resolved.settings.turboQuantBits
                keyBits = bits
                valueBits = bits
            }
            params.kvMode = .turboQuant(keyBits: keyBits, valueBits: valueBits)
            params.kvBits = nil
        }
        return params
    }

    /// Split a rolling text buffer on `<think>` / `</think>` tags. Holds
    /// back trailing bytes that might be partial tag openings for the
    /// next delta to complete. Updates `inThinkBlock` state in-place.
    private func splitThinkTags(
        buffer: inout String,
        inThinkBlock: inout Bool
    ) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        while !buffer.isEmpty {
            if inThinkBlock {
                if let end = buffer.range(of: "</think>") {
                    reasoning += buffer[..<end.lowerBound]
                    buffer.removeSubrange(..<end.upperBound)
                    inThinkBlock = false
                } else {
                    // Hold back last 9 chars in case `</think>` splits across deltas.
                    if buffer.count > 9 {
                        reasoning += String(buffer.dropLast(9))
                        buffer = String(buffer.suffix(9))
                    }
                    break
                }
            } else {
                if let start = buffer.range(of: "<think>") {
                    content += buffer[..<start.lowerBound]
                    buffer.removeSubrange(..<start.upperBound)
                    inThinkBlock = true
                } else {
                    // Hold back last 8 chars in case `<think>` splits.
                    if buffer.count > 8 {
                        content += String(buffer.dropLast(8))
                        buffer = String(buffer.suffix(8))
                    }
                    break
                }
            }
        }
        return (reasoning, content)
    }

    /// Testable re-implementation of the reasoning-parser dispatch loop
    /// used by `performOneGenerationPass`. Feeds a list of raw decode
    /// chunks through the provided parser (or the hand-rolled
    /// `splitThinkTags` fallback when `parser == nil`) and returns the
    /// aggregate (reasoning, content) pair AS WOULD BE YIELDED on the
    /// continuation. Preserves §15: when `effectiveThinking == false`,
    /// any reasoning output is routed into content.
    ///
    /// This is a pure function — no engine actor, no model container. It
    /// exercises the same branching logic as the live loop so unit tests
    /// can pin the behavior without spinning up a real model.
    static func dispatchReasoningForTests(
        chunks: [String],
        parser: ReasoningParser?,
        effectiveThinking: Bool,
        thinkInPrompt: Bool = false
    ) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        var pendingBuffer = ""
        var inThinkBlock = false
        var parserPrev = ""
        var parserCur = ""
        parser?.resetState(thinkInPrompt: thinkInPrompt, harmonyActive: thinkInPrompt)
        for text in chunks {
            var r = ""
            var c = ""
            if let p = parser {
                parserPrev = parserCur
                parserCur += text
                if let d = p.extractReasoningStreaming(
                    previous: parserPrev, current: parserCur, delta: text
                ) {
                    if let dr = d.reasoning { r += dr }
                    if let dc = d.content { c += dc }
                }
            } else {
                pendingBuffer += text
                // Same splitter the live path calls. Call via a dummy
                // Engine-free wrapper below.
                let s = Self._staticSplitThinkTags(
                    buffer: &pendingBuffer, inThinkBlock: &inThinkBlock)
                r = s.reasoning
                c = s.content
            }
            if effectiveThinking {
                reasoning += r
            } else {
                // §15: reasoning-off fallthrough.
                content += r
            }
            content += c
        }
        // End-of-stream flush: the splitter holds back up to 8 trailing
        // chars as partial-tag safety. Once we know there are no more
        // chunks coming, drain the remainder.
        if parser == nil && !pendingBuffer.isEmpty {
            if inThinkBlock {
                // Unterminated <think> — treat everything as reasoning.
                if effectiveThinking {
                    reasoning += pendingBuffer
                } else {
                    content += pendingBuffer
                }
            } else {
                content += pendingBuffer
            }
            pendingBuffer = ""
        }
        return (reasoning, content)
    }

    /// Static twin of `splitThinkTags` so `dispatchReasoningForTests`
    /// (which lives in an `Engine` static scope) can reach it without an
    /// Engine instance. Behavior matches the instance method 1:1.
    static func _staticSplitThinkTags(
        buffer: inout String,
        inThinkBlock: inout Bool
    ) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        while !buffer.isEmpty {
            if inThinkBlock {
                if let end = buffer.range(of: "</think>") {
                    reasoning += buffer[..<end.lowerBound]
                    buffer.removeSubrange(..<end.upperBound)
                    inThinkBlock = false
                } else {
                    if buffer.count > 9 {
                        reasoning += String(buffer.dropLast(9))
                        buffer = String(buffer.suffix(9))
                    }
                    break
                }
            } else {
                if let start = buffer.range(of: "<think>") {
                    content += buffer[..<start.lowerBound]
                    buffer.removeSubrange(..<start.upperBound)
                    inThinkBlock = true
                } else {
                    if buffer.count > 8 {
                        content += String(buffer.dropLast(8))
                        buffer = String(buffer.suffix(8))
                    }
                    break
                }
            }
        }
        return (reasoning, content)
    }

    /// Map vmlx-swift-lm stop reason to OpenAI-compatible finish_reason.
    private func mapStopReason(_ info: GenerateStopReason) -> String {
        switch info {
        case .stop: return "stop"
        case .length: return "length"
        case .cancelled: return "cancelled"
        @unknown default: return "stop"
        }
    }

    /// Encode a vMLXLMCommon `[String: JSONValue]` dict to a JSON string.
    /// Fully-qualified to disambiguate from our own `vMLXEngine.JSONValue`.
    private func encodeMLXJSONValueDict(_ dict: [String: vMLXLMCommon.JSONValue]) -> String? {
        guard let data = try? JSONEncoder().encode(dict),
              let s = String(data: data, encoding: .utf8) else {
            return nil
        }
        return s
    }
}

/// Convert our `JSONValue` enum to a `Sendable` value suitable for
/// embedding into `UserInput.additionalContext`. Used to splat
/// `chat_template_kwargs` into the Jinja render context.
func jsonValueToSendable(_ value: JSONValue) -> any Sendable {
    switch value {
    case .null:           return NSNull()
    case .bool(let b):    return b
    case .number(let n):  return n
    case .string(let s):  return s
    case .array(let a):   return a.map { jsonValueToSendable($0) }
    case .object(let o):
        var out: [String: any Sendable] = [:]
        for (k, v) in o { out[k] = jsonValueToSendable(v) }
        return out
    }
}

// MARK: - ToolCallParser compatibility shim
//
// The Swift ToolCallParser registry returns `ParsedToolCall` values. Add a
// lightweight bridge that extracts `.name` and `.argumentsJSON`. The real
// parser entry points live in `ToolCallParser.swift`; if the concrete
// implementations don't yet expose a streaming-friendly entry, this
// placeholder returns empty and the streaming accumulator keeps buffering.

private extension ToolCallParser {
    /// Best-effort streaming parse. Returns an empty array until the
    /// concrete parser exposes a `parseStreaming(_:)` hook — in the
    /// meantime the outer loop keeps buffering and emits the tool call
    /// only when `.toolCall(_)` arrives from vmlx-swift-lm's own parser.
    func parse(_ text: String) -> [(name: String, argumentsJSON: String?)] {
        return []
    }
}
