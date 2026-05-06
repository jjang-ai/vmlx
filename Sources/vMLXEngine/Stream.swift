import Foundation
import AVFoundation
import CoreImage
import MLX
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
        // Iter 144 — task-registration race fix.
        //
        // Capture-after-create pattern: a `_StreamTaskBox` is filled
        // with the real driving Task right after construction so the
        // body can self-identify on first actor hop without racing the
        // external registration Task. The body does its own
        // `setCurrentStreamTask(myself)` AND its own
        // `clearCurrentStreamTaskIfMatches(myself)` on every exit path
        // via a final `defer`, removing the prior pattern of:
        //   (a) a no-op placeholder Task that lingered when the body
        //       returned before external registration ran,
        //   (b) `clearCurrentStreamTask` only on the happy path —
        //       leaking a finished Task ref on early-return paths
        //       (e.g. `EngineError.tooManyQueued`).
        // Match-aware clear ensures a stale defer landing AFTER the
        // next stream's registration cannot null out the new
        // in-flight task.
        let box = _StreamTaskBox()
        return AsyncThrowingStream { continuation in
            let task = Task { [weak self, box] in
                guard let self else {
                    continuation.finish()
                    return
                }
                let me = box.task
                // Register the driving task on the actor so `Engine.stop()`
                // and any future admin endpoints can reach it. Also defer
                // a match-aware clear so the slot is freed on every exit
                // path including early-return rejections, without nulling
                // a sibling stream's registration.
                if let me { await self.setCurrentStreamTask(me) }
                defer {
                    if let me {
                        Task.detached { [weak self] in
                            await self?.clearCurrentStreamTaskIfMatches(me)
                        }
                    }
                }
                // Serialize with any in-flight generation — MLX's shared
                // Metal command queue is NOT concurrency-safe, and a
                // second generate() started while a first one is still
                // encoding trips the `IOGPUMetalCommandBuffer
                // setCurrentCommandEncoder:` assertion and aborts the
                // process. See `GenerationLock.swift`. FIFO — requests
                // queue and drain in arrival order, so users see latency
                // under load but never a server crash.
                // B4 §262: opt-in 503 backpressure. When
                // `VMLX_MAX_QUEUED_REQUESTS` is set, a burst past that
                // threshold gets rejected with `EngineError.tooManyQueued`
                // (→ HTTP 503 + Retry-After hint) instead of blocking
                // indefinitely behind the FIFO lock. Default unbounded
                // — load balancers upstream can still queue, and the
                // serial-fifo scheduler already drains in order.
                if let cap = Int(ProcessInfo.processInfo.environment["VMLX_MAX_QUEUED_REQUESTS"] ?? ""),
                   cap > 0
                {
                    let queued = await self.generationLock.inflightOrQueued
                    if queued >= cap {
                        continuation.finish(throwing:
                            EngineError.tooManyQueued(current: queued, limit: cap))
                        return
                    }
                }
                // Iter 144 — `acquire()` now throws CancellationError if
                // the parent Task is cancelled while queued. Catch +
                // finish early, skipping `release()` since we never
                // owned the lock. The defer-based `clearCurrentStreamTaskIfMatches`
                // still cleans up our task slot.
                do {
                    try await self.generationLock.acquire()
                } catch {
                    continuation.finish()
                    return
                }
                // Post-cancel barrier — if the previous stream cancelled
                // on a JANGTQ-active path, the prior cleanup did a 1s
                // sleep + synchronize() before releasing the lock. That
                // is heuristic; if MXTQ custom-encoder release lagged the
                // sleep budget, the prefill we're about to start could
                // race a still-open compute encoder and trip
                // `_status < MTLCommandBufferStatusCommitted` or
                // `addCompletedHandler: ... after commit call`.
                // Drain again here, on the consuming side, so the
                // hand-off is enforced at BOTH ends of the lock.
                if await self.consumePostCancelBarrierForJangTQ() {
                    MLX.Stream.defaultStream(.gpu).synchronize()
                    MLX.Stream.defaultStream(.cpu).synchronize()
                }

                // §JANGPress — wake any compressed expert tiles before
                // the engine touches them. Both backends are optional;
                // calls are nil-checked, so the burst is microseconds
                // when neither is configured.
                //
                // .mach backend (vm_purgable_control): controller's
                //   willStartInference walks every registered tile and
                //   flips NONVOLATILE. The kernel decompresses on access;
                //   a discarded tile triggers disk-refault.
                //
                // .mmap backend (madvise): no per-turn wake needed —
                //   willNeed/dontNeed is per-acquire/release; the
                //   controller's job is owned by the tier itself.
                //
                // Failsafe property: if both backends are off, this
                // block is two no-ops. Existing behavior is preserved.
                await self.emberController?.willStartInference()
                var wasCancelled = false
                do {
                    // Q3 §299: wrap decode in `MLX.withError` so MLX/Metal
                    // fatal-error trampolines (command-buffer commit
                    // failures, encoder-already-encoding, SIGTRAPs from
                    // `mtl_throw_*`) surface as a Swift throw instead of
                    // a process-wide crash. The block is evaluated
                    // task-locally so other streams in flight keep the
                    // default abort behavior.
                    try await MLX.withError {
                        try await JangPressRouteTelemetry.$controller.withValue(
                            self.emberController
                        ) {
                            try await self.performStreamingGeneration(
                                request: request,
                                continuation: continuation
                            )
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    wasCancelled = true
                    continuation.finish()
                } catch let mlxErr as MLX.MLXError {
                    // Map caught MLX fatal-error to a clean Swift error
                    // with the raw message, instead of letting it bubble
                    // as a generic `.caught(...)`. HTTP maps this to 500.
                    let raw: String = {
                        if case .caught(let s) = mlxErr { return s }
                        return "\(mlxErr)"
                    }()
                    // Iter 144 — scrub developer-local paths and
                    // multi-line C++ stack traces before forwarding to
                    // the API client. MLX's `MLXError.caught` carries
                    // strings like an absolute build-tree path under
                    // `~/.build/checkouts/mlx/...:LINE`, plus the
                    // failure message ("failed to commit command buffer
                    // (kIOSurfaceTimeout)" etc.). The full string is
                    // fine for engine logs; clients get the first line
                    // with absolute paths replaced with "<mlx>".
                    await self.log(.error, "engine", "stream Metal commit failed: \(raw)")
                    let firstLine = raw.split(separator: "\n",
                                              maxSplits: 1,
                                              omittingEmptySubsequences: false)
                                       .first.map(String.init) ?? raw
                    let scrubbed = Self.scrubInternalPaths(firstLine)
                    continuation.finish(throwing: EngineError.metalCommitFailed(scrubbed))
                } catch {
                    await self.log(.error, "engine", "stream failed: \(error)")
                    continuation.finish(throwing: error)
                }
                // Drain any async Metal work still pending from the just-
                // finished generation BEFORE releasing the lock. Three
                // distinct Metal assertions were observed across
                // iterations without this:
                //
                //   1. `_status < MTLCommandBufferStatusCommitted` in
                //      setCurrentCommandEncoder — text-model concurrent
                //      race (fixed by GenerationLock itself).
                //   2. `Completed handler provided after commit call` —
                //      VL-model race where async-eval handler
                //      registration lost to the next gen's commit
                //      (iteration 3 fixed by synchronize()).
                //   3. `A command encoder is already encoding to this
                //      command buffer` — JANGTQ native path, where
                //      MXTQ custom kernels leave an OPEN compute
                //      encoder that `synchronize()` doesn't close
                //      (synchronize only drains committed work, not
                //      uncommitted open encoders).
                //
                // MLX.eval on a dummy scalar forces a commit barrier:
                // any open encoder closes + commits, then blocks for
                // completion. `synchronize()` afterward is belt-and-
                // suspenders for anything scheduled on the default GPU
                // stream that the eval barrier didn't touch directly.
                //
                // We also drain the CPU stream: some MXTQ helper ops
                // (index bookkeeping, scalar setup for the custom
                // kernels) run on the CPU stream, and if they're still
                // in flight when the next waiter commits GPU work that
                // depends on them, Metal trips
                // `tryCoalescingPreviousComputeCommandEncoder: A
                // command encoder is already encoding to this command
                // buffer`.
                //
                // On CancellationError: give MLX's internal async-eval
                // pipeline a small window to propagate cancellation
                // through before we drain. Without this window,
                // synchronize() races the still-open command encoder
                // from the interrupted gen and the NEXT request then
                // trips either
                //   `addCompletedHandler: Completed handler provided
                //    after commit call`, OR
                //   `_status < MTLCommandBufferStatusCommitted at
                //    setCurrentCommandEncoder:`
                // depending on how far the encoder had progressed.
                //
                // 150 ms was the original bisection on Qwen3-0.6B.
                //
                // JANGTQ native models (Qwen3.6-35B-A3B-JANGTQ2 etc.) with
                // custom compute encoders (P3/P15/P17/P18) reproducibly
                // SIGSEGV in the eval+sync barrier BELOW during a single
                // cancel_midstream even when this sleep is extended to
                // 500ms. Root cause is the encoder itself being mid-write
                // when eval fires — not a drain-timing issue. Fix
                // requires either JANGTQ-specific encoder close hooks in
                // the kernel dispatcher OR skipping the eval barrier when
                // the model is JANGTQ-native. Flagged as an open blocker
                // in SWIFT-AUDIT-2026-04-18.md §3 for next session.
                // Cancel-aware drain. The 2026-04-18 root-cause probe
                // showed `MLX.eval` on a scalar reproducibly SIGSEGVs on
                // Qwen3.6-35B-A3B-JANGTQ2 after a cancel — the MXTQ
                // custom compute encoders (P3/P15/P17/P18) are mid-write
                // and the commit barrier trips a Metal invariant. A
                // 2026-04-18 later probe against Qwen3.6-35B-A3B-MXFP4
                // (hybrid SSM, NOT JANGTQ) showed the same crash
                // signature on cancel_midstream — the hybrid SSM
                // companion cache's partial-write state is the broader
                // common cause, not JANGTQ specifically. Widening the
                // gate so ANY cancel path skips the scalar commit
                // barrier and relies on synchronize() for drain. The
                // commit barrier was a belt-and-suspenders
                // graph-flush; synchronize() on both .gpu and .cpu
                // streams already drains the pipeline. Normal
                // (non-cancelled) generation still evals as before.
                let isJangTQPath = await self.isJangTQActivePath()
                if wasCancelled {
                    // Longer drain for JANGTQ — custom compute encoders
                    // need more time to release Metal fences.
                    try? await Task.sleep(
                        nanoseconds: isJangTQPath ? 1_000_000_000 : 250_000_000
                    )
                    // BLOCKER follow-up — flag the next stream to perform
                    // an extra synchronize() barrier before prefill on
                    // JANGTQ. The 1s sleep is heuristic; if MXTQ encoder
                    // release ever lags it (large model, system under
                    // pressure), the next prefill commit could trip a
                    // Metal invariant. Belt-and-suspenders.
                    if isJangTQPath {
                        await self.markPostCancelBarrierPendingForJangTQ()
                    }
                } else {
                    // Non-cancel path: keep the commit barrier to force
                    // graph flush so the next request sees a clean
                    // scheduler state.
                    MLX.eval(MLX.MLXArray(Int32(0)))
                }
                MLX.Stream.defaultStream(.gpu).synchronize()
                MLX.Stream.defaultStream(.cpu).synchronize()
                // Release synchronously (not via a detached Task) so the
                // next FIFO waiter picks up the baton only AFTER MLX has
                // quiesced and the actor state is clean.
                await self.generationLock.release()
                // Happy-path explicit clear before `didFinishInference`
                // — preserves the cleanup ordering pinned by
                // `JangPressReasoningLifecycleContractTests`. The
                // top-of-body `defer { clearCurrentStreamTaskIfMatches }`
                // covers early-return paths (e.g. tooManyQueued) and
                // is a no-op on the happy path because the slot is
                // already nil by the time the defer fires (match-aware).
                await self.clearCurrentStreamTask()
                // §JANGPress — kick the controller's quiesce countdown.
                // Cheap when the controller is nil. When armed this
                // schedules the cold-tile compaction to fire after
                // `quiesceTimeoutMs` of idle (default 30 s), unless
                // another inference arrives first (which cancels the
                // timer in `willStartInference`).
                await self.emberController?.didFinishInference()
            }
            // Iter 144 — fill the box AFTER `task` is constructed. The
            // body reads `box.task` on first actor hop. The 1-cycle gap
            // between `Task { ... }` construction and this assignment is
            // covered by the body's first `await self.setCurrentStreamTask`
            // requiring an actor hop, which is enqueued AFTER the box is
            // filled because the closure body cannot start running
            // without first being scheduled — and scheduling happens at
            // most one cycle later. Replaces the prior detached
            // registration `Task` which raced the body's early-return
            // exits.
            box.task = task
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Strip developer-local absolute paths from MLX error messages
    /// before forwarding them to API clients. The full unscrubbed
    /// message still goes to the engine log.
    static func scrubInternalPaths(_ s: String) -> String {
        // Replace anything that looks like an absolute Unix path under
        // common build-tree roots with a placeholder. Conservative —
        // only swaps obvious artifacts.
        var out = s
        let patterns: [String] = [
            #"/Users/[^/\s]+/(?:[^\s]+/)?\.build/checkouts/mlx[^\s]*"#,
            #"/Users/[^/\s]+/[^\s]*\.swift:\d+"#,
            #"/Users/[^/\s]+/[^\s]*\.cpp:\d+"#,
            #"/Users/[^/\s]+/[^\s]*\.mm:\d+"#,
            #"/Users/[^/\s]+/[^\s]*\.h:\d+"#,
        ]
        for pat in patterns {
            if let re = try? NSRegularExpression(pattern: pat) {
                let range = NSRange(out.startIndex..., in: out)
                out = re.stringByReplacingMatches(
                    in: out, range: range, withTemplate: "<mlx>")
            }
        }
        return out
    }
}

/// Capture-after-create holder for `streamReal`. Filled with the real
/// driving Task right after construction; the Task body reads it on
/// first actor hop. Top-level rather than nested so Swift's
/// `AsyncThrowingStream` initializer overload resolution stays
/// unambiguous.
final class _StreamTaskBox: @unchecked Sendable {
    var task: Task<Void, Never>?
}

extension Engine {

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

        // Iter 143: Smelt mode removed (Eric directive 2026-05-04 —
        // "remove it"). Smelt's partial-expert-loading goal is now
        // covered by JangPress (mmap-based cold-tile compression via
        // safetensors PROT_READ + madvise + Mach purgeable regions).
        // See `Sources/vMLXLMCommon/Cache/JangPress*.swift`. Smelt
        // settings/CLI/load-options were stripped; any old config
        // file referencing smelt is silently ignored at decode time.
        // Default-off matches Python v1.3.36 server.py and §15 contract:
        // when neither the request nor resolved settings specify a value
        // (OpenAI clients that never send enable_thinking), reasoning must
        // route to content so `delta.content` is never empty.
        // Fixes vmlx #67 (streaming empty content) + #6 (think-seed missing).
        //
        // iter-152 §223: OpenAI clients that pass reasoning_effort ∈
        // {"low","medium","high"} but omit enable_thinking get zero
        // reasoning because enable_thinking defaults to false and the
        // Qwen3.5 / MiniMax / Step chat template stamps an empty
        // `<think></think>` block. Live-repro'd 2026-04-20 on
        // Qwen3.5-35B-A3B-4bit: effort=low → reasoning channel empty,
        // content answered directly. The caller clearly wanted reasoning
        // (otherwise they wouldn't have sent effort=non-none). Mirror
        // the Mistral 4 auto-map but model-wide: a non-"none" effort
        // implies enable_thinking=true unless the caller explicitly
        // overrides. Explicit enable_thinking (true or false) wins over
        // the inferred value.
        let effortImpliesThinking: Bool? = request.reasoningEffort.flatMap {
            let v = $0.lowercased()
            if v == "none" || v.isEmpty { return false }
            // §385 — `max` is DeepSeek V4's deep-reasoning level (paired
            // with `high` in jang_config.chat.reasoning.reasoning_effort_levels).
            // Was silently dropping to nil before because only low/medium/high
            // were recognized; DSV4 callers passing effort=max got chat mode.
            if v == "low" || v == "medium" || v == "high" || v == "max" { return true }
            return nil
        }
        var effectiveThinking: Bool = request.enableThinking
            ?? resolved.enableThinking
            ?? effortImpliesThinking
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
        // `modelStampsThink` means "the template owns the think open/close
        // generation prompt." It does NOT mean the rendered prompt is always
        // inside an open think block. Laguna/Qwen-style templates render:
        //
        //   enable_thinking=true  -> <assistant>\n<think>
        //   enable_thinking=false -> <assistant>\n</think>
        //
        // The stream parser must therefore seed "currently inside think"
        // only when thinking is actually enabled. Treating every stamping
        // template as open made Laguna thinking-off/chat-mode responses get
        // classified as reasoning and dropped or displayed as template
        // garbage.
        let promptStartsInThink = modelStampsThink && effectiveThinking

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
        let videoMarker: String = {
            let fam = (caps?.family ?? "").lowercased()
            let mt = (caps?.modelType ?? "").lowercased()
            if fam.contains("qwen") || mt.contains("qwen") {
                return "<|vision_start|><|video_pad|><|vision_end|>"
            }
            return "<video>"
        }()
        // The prompt-level reasoning-off stub is a family-specific hack,
        // not a universal "thinking off" mechanism. It is valid for
        // DeepSeek-style `<think>...</think>` models that do NOT have a
        // template-owned thinking rail. It is invalid for Mistral 3.5 /
        // Mistral 4 (`[MODEL_SETTINGS]` + `[THINK]`), Gemma/Harmony
        // channel models, and every template-owned family where
        // `enable_thinking=false` already renders the right close marker.
        //
        // Live Mistral-Medium-3.5 JANGTQ repro 2026-05-03: the old global
        // injection appended a prior assistant message containing
        // `<think>\n</think>\n\n`; the Mistral renderer treated it as
        // literal assistant history and emitted `<think>...</think></s>`
        // before generation, producing incoherent decode despite a correct
        // `[INST]` prompt. Gate it to DeepSeek-style families only.
        let family = (caps?.family ?? "").lowercased()
        let modelType = (caps?.modelType ?? "").lowercased()
        let parserNameForStub = (caps?.reasoningParser ?? "").lowercased()
        let injectReasoningOffStub =
            !modelStampsThink
            && (
                family.contains("deepseek")
                || modelType.contains("deepseek")
                || parserNameForStub == "deepseek_r1"
            )

        let chatMessages = await Engine.buildChatMessages(
            from: request,
            effectiveThinking: effectiveThinking,
            modelStampsThink: modelStampsThink,
            injectReasoningOffStub: injectReasoningOffStub,
            responseFormatInstruction: Engine.responseFormatInstruction(
                from: request.responseFormat),
            imageMarker: imageMarker,
            videoMarker: videoMarker)

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
        // iter-ralph §229 (H4): wire the ChatSettings.mcpEnabled toggle.
        // Before §229 the toggle was a pure UI draft-state zombie — it
        // round-tripped to disk + back to the popover, but never reached
        // Stream. Users who toggled "MCP: Off" still had every connected
        // MCP server's tool catalog merged into the request. Explicit
        // `false` now suppresses the merge; `nil` (default) keeps the
        // previous always-merge behavior so zero-config users don't
        // silently lose MCP.
        let sessionMCPOff = (resolved.settings.mcpEnabledResolved == false)
        if !mcpMergeBlocked && !userExplicitlyDisabledTools && !sessionMCPOff {
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

        // tool_choice: "none" enforcement (iter-49). OpenAI spec says the
        // model MUST NOT call any tool when tool_choice="none". Even with
        // the stream-side parser catching calls, the tools still reach the
        // chat template and the model happily decides to call them —
        // resulting in a response that violates the contract. The fix is
        // to drop tools from the template render when tool_choice=.none
        // so the model never sees them. MCP merge was already gated on
        // this flag above; this extends the same behavior to user-supplied
        // tools so the contract is honored end-to-end. Live-reproduced on
        // Qwen3-0.6B + `required=["a","b"]` tools.
        // Unwrap Optional<ToolChoice> first — Swift's `case .none =
        // request.toolChoice` matches Optional.none (nil), NOT
        // ChatRequest.ToolChoice.none. Same guard pattern as
        // `mcpMergeBlocked` above.
        if let tc = request.toolChoice, case .none = tc {
            mergedTools = nil
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
        // Kimi-K2.6's bundled chat_template.jinja uses `thinking`, while
        // most other reasoning templates use `enable_thinking`. Send both
        // names so chat-mode requests render the closed `<think></think>`
        // prompt instead of accidentally starting inside an open think block.
        templateExtras["thinking"] = effectiveThinking
        if let effort = request.reasoningEffort {
            templateExtras["reasoning_effort"] = effort
        }
        if let budget = request.thinkingBudget, budget > 0 {
            templateExtras["thinking_budget"] = budget
            // Nemotron-H Omni sidecar templates use `reasoning_budget`
            // for the visible prompt hint. Keep both names wired so the
            // API's Anthropic-compatible `thinking_budget` reaches native
            // HF/JANG templates without per-family request branching.
            templateExtras["reasoning_budget"] = budget
        }
        // OpenAI `chat_template_kwargs` — caller-provided per-request
        // template variables (e.g. assistant_prefix, enable_tools). Merged
        // **last** so callers can override engine-set extras intentionally.
        if let kwargs = request.chatTemplateKwargs {
            for (key, value) in kwargs {
                templateExtras[key] = jsonValueToSendable(value)
            }
        }
        // iter-60: wire `settings.chatTemplate` through as a template
        // override. The `--chat-template path.jinja` CLI flag resolves
        // the file content into `g.chatTemplate`; sessions can override
        // per-session. The TokenizerBridge picks the reserved
        // `__chat_template_override__` key out of additionalContext
        // and delegates to swift-transformers'
        // `applyChatTemplate(messages:chatTemplate:.literal(...))`
        // overload. Non-empty string = override wins; empty = falls
        // through to the upstream tokenizer's built-in template.
        // Replaces the §88 per-request warning — the override now
        // actually does what the flag promised.
        let chatTemplateOverride = resolved.settings.chatTemplate
        if !chatTemplateOverride.isEmpty {
            templateExtras["__chat_template_override__"] = chatTemplateOverride
        } else if modelType == "deepseek_v4",
                  let builtin = caps?.chatTemplate,
                  !builtin.isEmpty
        {
            // DSV4 Flash/Pro bundles intentionally omit
            // tokenizer_config.chat_template and instead ship
            // jang_config.chat.encoder=encoding_dsv4. CapabilityDetector
            // materializes the equivalent built-in Jinja template. Thread it
            // through as a literal override so the generic LLM processor does
            // not fall back to plain joined text. Keep this DSV4-only:
            // other families may expose broken include-wrapper templates
            // (Laguna) that the tokenizer bridge must bypass via native
            // fallback rendering.
            templateExtras["__chat_template_override__"] = builtin
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
        // Seed the hand-rolled splitter only when the rendered prompt leaves
        // generation inside an open think block. Thinking-off templates like
        // Laguna deliberately render `</think>` before generation, so their
        // first model token is visible content and must not be suppressed.
        var inThinkBlock = promptStartsInThink
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

        // iter-152 §222: tool-call marker boundary holdback. When any tool
        // parser or tools: array is attached, hold back the last
        // `maxMarkerLenMinus1` bytes of visible content across deltas. A
        // marker like `<tool_call>` that straddles two deltas (delta 1
        // ends with "<", delta 2 starts with "tool_call>...") would
        // otherwise leak its prefix — the single "<" isn't a marker on
        // delta 1, so feed() doesn't flag buffered, and the "<" yields
        // to the client before the marker is detected. Holdback ensures
        // the client never sees bytes that could be the start of any
        // known marker until enough bytes accrue to either confirm or
        // refute the match.
        //
        // Tracked across the streaming loop: bytes of `streamingAcc.current`
        // already yielded to the client. Held-back bytes sit in
        // `streamingAcc.current[emittedContentBytes:]` and get flushed
        // when the stream ends naturally (no marker, no stop).
        var emittedContentBytes: Int = 0
        let maxMarkerLenMinus1: Int = {
            let m = (toolCallMarkers.map { $0.utf8.count }.max() ?? 1) - 1
            return max(0, m)
        }()

        // Anthropic-compat `thinking_budget`. Caps the cumulative chars
        // emitted as reasoning. When the cap is hit mid-block, flip
        // `thinkingBudgetExhausted` so any further `.reasoning` deltas get
        // rerouted to visible content. 0 (or nil) disables the cap.
        // We use a 4-chars-per-token approximation because the streaming
        // path operates on decoded text deltas; an exact token count would
        // require re-tokenizing every chunk and is not worth the perf hit.
        let thinkingBudgetTokens = max(0, request.thinkingBudget ?? 0)
        let thinkingBudgetCharCap = thinkingBudgetTokens * 4
        var reasoningCharsEmitted = 0
        var thinkingBudgetExhausted = false
        var postBudgetReasoningBuffer = ""
        var postBudgetVisibleStarted = false

        func yieldPostBudgetContent(_ text: String) {
            guard !text.isEmpty else { return }
            if postBudgetVisibleStarted {
                continuation.yield(StreamChunk(content: text))
                return
            }

            postBudgetReasoningBuffer += text
            if let close = postBudgetReasoningBuffer.range(of: "</think>") {
                let visible = String(postBudgetReasoningBuffer[close.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                postBudgetReasoningBuffer = ""
                postBudgetVisibleStarted = true
                if !visible.isEmpty {
                    continuation.yield(StreamChunk(content: visible))
                }
                return
            }
            if let boundary = postBudgetReasoningBuffer.range(of: "\n\n") {
                let visible = String(postBudgetReasoningBuffer[boundary.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                postBudgetReasoningBuffer = ""
                postBudgetVisibleStarted = true
                if !visible.isEmpty {
                    continuation.yield(StreamChunk(content: visible))
                }
                return
            }

            // Last-resort no-close/no-paragraph path. Avoid an indefinitely
            // blank response, but only after buffering enough text that we are
            // unlikely to be cutting a marker or a short reasoning tail.
            if postBudgetReasoningBuffer.count > 1024 {
                let visible = postBudgetReasoningBuffer
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                postBudgetReasoningBuffer = ""
                postBudgetVisibleStarted = true
                if !visible.isEmpty {
                    continuation.yield(StreamChunk(content: visible))
                }
            }
        }

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
        //
        // iter-58: parser selection was `caps?.reasoningParser` only —
        // it ignored `resolved.settings.defaultReasoningParser`. Users
        // with a CLI `--reasoning-parser` flag or SessionConfigForm
        // override got the capability-detector's choice regardless.
        // Match the tool-parser shape from ~line 542:
        //   caps?.toolParser ?? resolved.settings.defaultToolParser
        // so user override wins when non-empty, capability fallback
        // otherwise. Live-verified on Stream startup — empty string
        // from settings falls back cleanly to caps.
        let settingsReasoningParser = resolved.settings.defaultReasoningParser
        let reasoningParserName = (
            !settingsReasoningParser.isEmpty
                ? settingsReasoningParser
                : caps?.reasoningParser
        )
        let reasoningParser: ReasoningParser? = reasoningParserName
            .flatMap { ReasoningParserRegistry.make($0) }
        // Reset per-request state. `thinkInPrompt` tracks whether the chat
        // template already stamped an OPEN `<think>` for this request.
        // Round 16 / Rank 11 (§15 NO-REGRESSION): Harmony/GPT-OSS channel
        // markers must only be considered active when the template stamps
        // think tags AND the caller actually requested reasoning. With the
        // old `harmonyActive: modelStampsThink` wiring, a Qwen3/MiniMax/Step
        // model with `enableThinking=false` still ran the harmony branch,
        // swallowing tokens destined for visible content and producing the
        // "reasoning OFF stuck UI" regression documented in
        // `feedback_reasoning_off_ui_stuck.md`.
        let harmonyActive = promptStartsInThink
        reasoningParser?.resetState(
            thinkInPrompt: promptStartsInThink,
            harmonyActive: harmonyActive
        )
        var parserPrevious = ""
        var parserCurrent = ""
        // Latched flag: set true once a reasoning/channel marker has appeared
        // in the stream. Used by the cold-path optimization to skip the full
        // reasoning parser on text-only non-thinking turns (per-token parser
        // scan was O(N²) across decode). See Stream.swift:~910.
        var seenReasoningMarker = false
        // §241c small sliding window of recent cold-path deltas so the
        // marker-scan can detect multi-token splits like `<`,`|`,`channel`,`>`
        // that never surface as a single delta. Bounded (32 chars) so the
        // cold-path stays O(1) per token.
        var reasoningProbeBuffer = ""

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
            /// Reserved for callers that actually restore from the probe.
            /// The Stream pre-fetch below only builds usage labels; it
            /// releases any paged-cache pins immediately after composing
            /// `detail`, because the real restore happens inside
            /// TokenIterator's own coordinator fetch.
            let pinnedBlocks: [vMLXLMCommon.CacheBlock]
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

                // VL-race barrier (iter-25 fix #4, iter-30 fix #6): for
                // image/video inputs, `processor.prepare` runs the vision
                // encoder which leaves an OPEN Metal compute encoder under
                // the shared GPU stream. The subsequent LLM forward pass
                // then commits its own encoder and Metal trips either
                // `tryCoalescingPreviousComputeCommandEncoder: A command
                // encoder is already encoding…` (smaller VL models; fix
                // landed iter-25) OR `Completed handler provided after
                // commit call` (larger VL models where the text-token eval
                // alone isn't sufficient; found on Qwen3.5-VL-9B iter-29).
                // The end-of-gen drain can't catch this because it's
                // WITHIN a single request, before the LLM forward has
                // even started.
                //
                // iter-25 evaluated `lmInput.text.tokens` to commit the
                // text-side dep chain; iter-30 adds an explicit GPU stream
                // `synchronize()` so ALL pending async vision-encoder work
                // (including the separate image-features tensor, which
                // has its own dep chain not reachable through the text
                // tokens) is drained before the LLM forward starts.
                // Killswitch: `VMLX_DISABLE_VL_RACE_BARRIER=1` restores
                // the pre-iter-25 behavior for A/B debugging.
                if !userInput.images.isEmpty || !userInput.videos.isEmpty,
                   ProcessInfo.processInfo.environment["VMLX_DISABLE_VL_RACE_BARRIER"] != "1"
                {
                    MLX.eval(lmInput.text.tokens)
                    MLX.Stream.defaultStream(.gpu).synchronize()
                }

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
                // iter-122 §197 ROOT CAUSE FIX: the `<think></think>`
                // reasoning-off stub that `buildChatMessages:1930`
                // appends as a trailing assistant message renders into
                // ~13 extra tokens that sit BETWEEN the closed-turn
                // prefix and the real `<|turn>model\n` generation prompt.
                // With the old `withGP-withoutGP` differencing, only the
                // final gp suffix was stripped — the stub tokens stayed
                // inside the cache hash key, so T1's stub-decorated key
                // was a sibling (not a prefix) of T3's, and the memory
                // prefix cache always missed across turns. Fix: render
                // the cache-stable length as messages-WITHOUT-stub and
                // `add_generation_prompt=false`. `genPromptLen` then
                // covers both the stub wrap AND the template's real gp.
                // Live-repro: Gemma-4 T1/T3 both produced hashTokens that
                // contained `<|turn>model\n<think>\n</think>\n\n<turn|>\n`
                // for T1 but `...A1...<|turn>user\n...` for T3 — no
                // overlapping prefix. After fix, both hashTokens strip
                // back to the shared `<bos><|turn>user\nQ1<turn|>\n`.
                var genPromptLen = 0
                do {
                    let rawMsgsFull = DefaultMessageGenerator().generate(from: userInput)
                    var withGPContext = userInput.additionalContext ?? [:]
                    withGPContext["add_generation_prompt"] = true as any Sendable
                    let withGP = try ctx.tokenizer.applyChatTemplate(
                        messages: rawMsgsFull,
                        tools: nil,
                        additionalContext: withGPContext
                    )

                    // Strip the trailing reasoning-off stub from the
                    // cache-stable render. Detection mirrors the inject
                    // site at `buildChatMessages:1930` — last message is
                    // role=assistant with the exact stub literal. Safe
                    // guard: only drop when the last message actually
                    // matches (user-supplied prior-turn assistant messages
                    // must stay intact for multi-turn cache keys).
                    //
                    // iter-ralph §233 (M4): Nemotron-H / Nemotron-Cascade
                    // stamp the stub as `<think></think>` (no inner
                    // newlines) per their chat_template — different from
                    // Qwen3.5's `<think>\n\n</think>\n\n`. Without the
                    // Nemotron variant, multi-turn reasoning-off on
                    // Nemotron missed the prefix cache (§226 B harness
                    // live-confirmed). Recognize both shapes. Any future
                    // family that adds another variant goes here.
                    let stubContents: Set<String> = [
                        "<think>\n</think>\n\n",  // Qwen3.5 family
                        "<think></think>",        // Nemotron-H / Cascade
                        // §388 — DSV4 Flash/Pro chat mode closes the
                        // empty reasoning with a bare `</think>`
                        // (the open `<think>` is already consumed as
                        // part of the `<｜Assistant｜></think>` prompt
                        // tail per the built-in template). Without
                        // this entry, chat-mode DSV4 multi-turn would
                        // skip the gen_prompt_len strip → prefix
                        // cache miss every turn.
                        "</think>",
                    ]
                    var rawMsgsClosed = rawMsgsFull
                    if let last = rawMsgsClosed.last,
                       (last["role"] as? String) == "assistant",
                       let lastContent = last["content"] as? String,
                       stubContents.contains(lastContent)
                    {
                        rawMsgsClosed.removeLast()
                    }
                    var withoutGPContext = userInput.additionalContext ?? [:]
                    withoutGPContext["add_generation_prompt"] = false as any Sendable
                    let withoutGPClosed = try ctx.tokenizer.applyChatTemplate(
                        messages: rawMsgsClosed,
                        tools: nil,
                        additionalContext: withoutGPContext
                    )
                    if withGP.count > withoutGPClosed.count {
                        genPromptLen = withGP.count - withoutGPClosed.count
                    }
                } catch {
                    genPromptLen = 0
                }

                // Pre-fetch: tokenize is already done. Pull the int array out
                // of the MLXArray and consult the coordinator. `asArray(Int.self)`
                // copies eagerly so we can use it after the MLXArray drops.
                //
                // iter-86 §164: the base detail is just the winning tier
                // label ("paged" / "memory" / "disk" / "miss"). That is
                // factually incomplete for hybrid models — users want to
                // know whether the SSM companion ALSO hit (otherwise a
                // paged hit is followed by a full re-derive and the
                // "cached_tokens" number overpromises). Enrich the detail
                // string with an `+ssm(N)` suffix when the companion
                // fetch populated non-nil states so the usage envelope
                // reads honestly.
                var preFetch = CachePreFetch(matched: 0, detail: "miss", pinnedBlocks: [])
                if let coord = coordinator {
                    let tokenIds = lmInput.text.tokens.asArray(Int.self)
                    // Audit 2026-04-16: pass isMLLM for VLM pipeline so
                    // SSM companion key uses N-1 boundary (Python parity
                    // `ssm_companion_cache.py:22`). Derived from the
                    // detected model capability — .vision modality is
                    // the MLLM batch generator's domain.
                    // **iter-71 (§100)** — removed the `isMLLM` flag from the
                    // probe. Previous audit comment claimed VL pipelines use
                    // N-1 SSM companion boundary (Python parity for its
                    // separate MLLM batch generator), but Swift has no
                    // separate MLLM batch path — Evaluate.swift and
                    // BatchEngine.swift both call storeAfterGeneration /
                    // fetch with the default `isMLLM: false`, so
                    // store/fetch consistently use N. The probe's
                    // `matched` return value is the paged tier's
                    // matchedTokens (independent of isMLLM anyway — only
                    // SSM companion boundary differs), so the flag was
                    // inert AND misleading future devs about the
                    // store/fetch parity contract. FIX-G-O's mediaSalt
                    // propagation stays — that was the real VL fetch/store
                    // parity fix (text-only key vs text+media key).
                    let preFetchSalt = computeMediaSalt(for: lmInput)
                    switch coord.fetch(
                        tokens: tokenIds,
                        mediaSalt: preFetchSalt,
                        genPromptLen: genPromptLen
                    ) {
                    case .hit(let matched, _, let tier, let blocks, let ssm, let disk):
                        // iter-86 §164: compose a rich detail string.
                        // Base tier is always present. Add `+ssm(count)`
                        // when companion states came back non-nil — on a
                        // hybrid model a paged hit WITHOUT SSM companion
                        // is still a partial miss (SSM gets re-derived).
                        // Add `+disk` if the primary hit came from paged
                        // but the disk tier ALSO backfilled arrays. Add
                        // `+gp(N)` for gen_prompt_len>0 requests so
                        // thinking-model multi-turn hits are visible.
                        var parts: [String] = [tier.rawValue]
                        if let ssmStates = ssm, !ssmStates.isEmpty {
                            parts.append("ssm(\(ssmStates.count))")
                        }
                        if tier != .disk, disk != nil {
                            parts.append("disk-backfill")
                        }
                        if genPromptLen > 0 {
                            parts.append("gp(\(genPromptLen))")
                        }
                        if !blocks.isEmpty {
                            coord.release(blocks: blocks)
                        }
                        preFetch = CachePreFetch(
                            matched: matched,
                            detail: parts.joined(separator: "+"),
                            pinnedBlocks: []
                        )
                    case .miss:
                        preFetch = CachePreFetch(matched: 0, detail: "miss", pinnedBlocks: [])
                    }
                }
                // iter-122 §197 RESOLVED — multi-turn partial-prefix cache
                // miss. Live trace (VMLX_CACHE_TRACE=1, Gemma-4-e2b-it-4bit)
                // showed the T1 hashTokens diverged from T3's first-23
                // prefix at position 13. Decoded:
                //   T1 stored: `<bos><|turn>user\nQ1<turn|>\n<|turn>model\n
                //               <think>\n</think>\n\n<turn|>\n`
                //   T3 head32:  `<bos><|turn>user\nQ1<turn|>\n<|turn>model\n
                //               A1<turn|>\n<|turn>user\nQ2...`
                // The `<think>\n</think>\n\n` block — 10 intermediate
                // tokens at positions 13-22 of T1's key — is the empty
                // assistant stub that `buildChatMessages:1930` appends
                // when `effectiveThinking=false && !hasTools && !stamps
                // && !structured`. It was stuck IN the cache key because
                // the old `genPromptLen = withGP - withoutGP` differencing
                // saw the stub in BOTH renders, so only the trailing
                // `<|turn>model\n` (3 tokens) got stripped. Fix: compute
                // genPromptLen against messages-WITHOUT-stub render —
                // stub+gp now strips as 16 tokens, T1's key collapses to
                // the 10-token `<bos><|turn>user\nQ1<turn|>\n` prefix
                // which IS a prefix of T3's 36-token key. Live post-fix:
                //   T1 miss (first touch), T2 memory+disk-backfill+gp(16),
                //   T3 memory+disk-backfill+gp(16) — multi-turn prefix
                //   hit restored. See gen-prompt-length fix above in
                //   this same function.
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
        // Iter 143 — wire JangPress embed-tier token-activity tracking.
        // `JangPressEmbedTier.recordTokenActivity` was previously
        // dormant (zero callers); the audit flagged it as a fully
        // built-but-unused feature. Now it sees every prompt's token
        // IDs at prefill time, so the Zipfian frequency map gets
        // populated naturally per request. After enough samples land
        // we trigger `applyZipfianAdvise` once to set MADV_DONTNEED
        // on the cold (1 - hotPercent)% of vocab rows. Fire-and-
        // forget — this is a memory-pressure hint, not a correctness
        // path, and a temporary failure (e.g. zero-shard model) is
        // safely a no-op inside the tier.
        if let embed = self.jangPressEmbed, !capturedPromptIds.isEmpty {
            // recordTokenActivity is bounded by stride sampling +
            // distinct-token cap (see JangPressEmbedTier comments);
            // safe to call inline.
            embed.recordTokenActivity(capturedPromptIds)
            // applyZipfianAdvise iterates ~200k vocab rows issuing
            // madvise() per row → ~50-100ms syscall budget on
            // M-series. Moving it OFF the request hot path so TTFT
            // isn't stalled. Run on a detached low-priority Task so
            // the kernel hint lands soon but doesn't compete with
            // the engine actor for scheduling. Idempotent — multiple
            // concurrent triggers from concurrent requests serialize
            // safely on the JangPressEmbedTier's internal lock.
            let stats = embed.snapshotIfBuilt()
            if stats.distinctTokensSeen >= 256 {
                Task.detached(priority: .background) { [weak embed] in
                    embed?.applyZipfianAdvise()
                }
            }
        }
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

        // iter-64 perf follow-up: per-token metrics recording was doing
        // one actor hop per decoded token (~170× for a 170-token response).
        // The actor hop itself isn't free (~5-10μs each on Apple Silicon)
        // and serializes behind any other actor work. Tray's live TPS
        // display polls at 1 Hz, so per-token fidelity is overkill.
        //
        // New default: batch every `metricsBatchEvery` tokens (8) —
        // preserves tray update smoothness (~10ms batch → 100+ batches/s
        // at 100 tok/s) while reducing actor hops 8×. The env
        // killswitch `VMLX_DISABLE_PER_TOKEN_METRICS=1` is now
        // redundant but kept for backward compat with existing perf
        // scripts; it forces end-of-stream-only recording.
        let perTokenMetricsDisabled =
            ProcessInfo.processInfo.environment["VMLX_DISABLE_PER_TOKEN_METRICS"] == "1"
        let metricsBatchEvery = 8
        // Elapsed wall-clock accumulator used by both the periodic batch
        // flush and the env-gated end-of-stream-only mode.
        var accumulatedDecodeMs: Double = 0
        var accumulatedDecodeTokens = 0


        // Logprobs accumulator — per-token logprob events arrive from the
        // Evaluate.swift token loop and are batched into StreamChunk.logprobs
        // alongside content/reasoning deltas.
        var pendingLogprobs: [TokenLogprob] = []
        let shouldCollectLogprobs = (request.logprobs ?? false)
        var lastChunkAt = prefillStart
        // §163.B4: take-and-clear the pending logprobs so the caller can
        // bundle them into the SAME StreamChunk as the content delta.
        // OpenAI emits logprobs inside `delta.logprobs.content[]` on the
        // same SSE frame as `delta.content`; a standalone frame breaks
        // strict consumers (lm-eval-harness, LangChain logprob chains)
        // that expect 1:1 mapping between logprobs[i] and content[i].
        // `flushLogprobs()` now only runs for end-of-stream drain when
        // there is no further content to attach to.
        func takeLogprobs() -> [TokenLogprob]? {
            guard !pendingLogprobs.isEmpty else { return nil }
            let lps = pendingLogprobs
            pendingLogprobs = []
            return lps
        }
        func flushLogprobs() {
            guard let lps = takeLogprobs() else { return }
            continuation.yield(StreamChunk(logprobs: lps))
        }
        for await event in stream {
            if Task.isCancelled { break }
            switch event {
            case .chunk(let text):
                let chunkNow = Date()
                if firstTokenAt == nil { firstTokenAt = chunkNow }

                let tickMs = chunkNow.timeIntervalSince(lastChunkAt) * 1000
                lastChunkAt = chunkNow
                decodedChunkCount += 1
                // iter-64: always accumulate; batch-flush every N
                // tokens. env killswitch defers to end-of-stream.
                accumulatedDecodeMs += tickMs
                accumulatedDecodeTokens += 1
                if !perTokenMetricsDisabled
                    && accumulatedDecodeTokens >= metricsBatchEvery
                {
                    let flushCount = accumulatedDecodeTokens
                    let flushMs = accumulatedDecodeMs
                    accumulatedDecodeTokens = 0
                    accumulatedDecodeMs = 0
                    await self.metrics.recordTokenBatch(
                        prefill: false, count: flushCount,
                        durationMs: flushMs)
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
                // §241c: markers like `<|channel>` are emitted as 4 separate
                // tokens (`<`, `|`, `channel`, `>`); the per-delta view
                // never crosses the boundary so the latch never fired →
                // Gemma 4 reasoning-off content leaked the literal markers.
                // Track a bounded sliding window of recent deltas and scan
                // over that instead of the raw per-delta chunk.
                if !seenReasoningMarker {
                    reasoningProbeBuffer += text
                    if reasoningProbeBuffer.count > 32 {
                        let keep = reasoningProbeBuffer.count - 32
                        reasoningProbeBuffer.removeFirst(keep)
                    }
                    if reasoningProbeBuffer.contains("<think>")
                        || reasoningProbeBuffer.contains("</think>")
                        || reasoningProbeBuffer.contains("<|channel|>")   // GPT-OSS / Harmony
                        || reasoningProbeBuffer.contains("<|channel>")    // Gemma 4 open
                        || reasoningProbeBuffer.contains("<channel|>")    // Gemma 4 close
                        || reasoningProbeBuffer.contains("[THINK]")
                    {
                        seenReasoningMarker = true
                    }
                }
                // §241d: gemma4 / openai_gptoss emit structural channel
                // markers (`<|channel>thought`, `<|channel|>analysis`,
                // `<channel|>`) that tokenize as 4+ separate tokens. The
                // cold-path optimization (skip parser when no marker yet
                // seen) routes those fragment tokens straight to content
                // BEFORE the sliding-window latch fires. By then the
                // markers are already flushed to the client. For these
                // two families we always use the full parser — the O(N²)
                // scan cost is negligible next to the alternative of
                // leaking the structural tokens as visible content.
                // Template-stamped thinking ON starts inside an open block
                // with no opener in the decoded output; force the parser so
                // it can see the eventual close. Thinking OFF does NOT belong
                // here for Laguna/Qwen-style templates because the rendered
                // prompt already contains `</think>` and generation starts in
                // visible content.
                let alwaysParse = (reasoningParserName == "gemma4"
                    || reasoningParserName == "openai_gptoss"
                    || promptStartsInThink)
                let useFullParser = (reasoningParser != nil)
                    && (effectiveThinking || seenReasoningMarker || alwaysParse)
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
                        // §15 NO-REGRESSION fallthrough vs BLOCKER #3:
                        // the two cases diverge on `modelStampsThink`.
                        //
                        //   • `!modelStampsThink && !effectiveThinking`
                        //     (the §15 case — the model emits free-form
                        //     reasoning that the template did NOT request)
                        //     → route reasoning to visible content so the
                        //     UI never goes blank. See
                        //     `feedback_reasoning_off_ui_stuck.md` —
                        //     regressed 3+ times in the Electron app.
                        //
                        //   • `modelStampsThink && !effectiveThinking`
                        //     (BLOCKER #3 — the chat template stamped
                        //     `<think>` but the user wants reasoning OFF)
                        //     → SUPPRESS the reasoning entirely. The
                        //     parser is tracking the open block from
                        //     `thinkInPrompt=true` and will recognize the
                        //     `</think>` close, flipping any post-close
                        //     bytes into `splitContent` which is emitted
                        //     normally below. Routing this reasoning to
                        //     `content` (the §15 path) leaks structural
                        //     `<think>` block content to the user — the
                        //     observed "loop" symptom.
                        //
                        // Budget-exhausted is independent of stamping —
                        // always reroute the post-cap overflow to content.
                        if modelStampsThink && !effectiveThinking
                            && !thinkingBudgetExhausted
                        {
                            // Drop on the floor — suppression sink.
                        } else if thinkingBudgetExhausted && effectiveThinking {
                            yieldPostBudgetContent(splitReasoning)
                        } else {
                            continuation.yield(StreamChunk(content: splitReasoning))
                        }
                    } else if thinkingBudgetCharCap > 0 {
                        // Budget enforcement: emit up to cap, then force-close.
                        let remaining = thinkingBudgetCharCap - reasoningCharsEmitted
                        if remaining > 0 {
                            let allowed = String(splitReasoning.prefix(remaining))
                            let overflow = String(splitReasoning.dropFirst(allowed.count))
                            continuation.yield(StreamChunk(reasoning: allowed))
                            reasoningCharsEmitted += allowed.count
                            if !overflow.isEmpty {
                                thinkingBudgetExhausted = true
                                inThinkBlock = false
                                yieldPostBudgetContent(overflow)
                            }
                        } else {
                            // Already at cap on entry: overflow is visible
                            // content, but never emit raw structural tags.
                            thinkingBudgetExhausted = true
                            inThinkBlock = false
                            yieldPostBudgetContent(splitReasoning)
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
                // iter-59 tool-call marker bleed fix: pre-feed the accumulator
                // BEFORE emitting any visible content. If adding `splitContent`
                // to the accumulator crosses a tool-call marker, only yield the
                // prefix that sits *before* the marker as content — everything
                // from the marker onwards stays buffered until the parser can
                // parse it (then fires as tool_calls) or the stream ends
                // (flushed as content below). Without this guard the raw
                // `<|python_tag|>{"name":...}` bytes leak to the client before
                // we ever parse them, breaking OpenAI-style tool_calls
                // consumers. Live-caught against Llama-3.2-1B via harness
                // tool_call case.
                let haveToolsOrParser = toolParser != nil
                    || (mergedTools?.isEmpty == false)
                var emittableContent = splitContent
                if !splitContent.isEmpty && !postBudgetReasoningBuffer.isEmpty {
                    postBudgetReasoningBuffer = ""
                    postBudgetVisibleStarted = true
                }
                if haveToolsOrParser && !splitContent.isEmpty {
                    let wasBuffered = streamingAcc.buffered
                    _ = streamingAcc.feed(splitContent)
                    // iter-152 §222: compute target-emit cap against the
                    // entire accumulated buffer rather than the current
                    // delta only. This closes the straddle-leak: bytes that
                    // could be the start of a marker are held back until
                    // the next delta proves (or disproves) the match.
                    let combined = streamingAcc.current
                    let combinedBytes = combined.utf8.count
                    let targetCap: Int
                    if streamingAcc.buffered {
                        if wasBuffered {
                            // Already past the marker boundary — emit
                            // nothing; the buffer will be parsed as a
                            // tool call at stream end.
                            targetCap = emittedContentBytes
                        } else {
                            // Marker landed inside this delta — yield up
                            // to the earliest marker offset.
                            var earliest = combinedBytes
                            for marker in toolCallMarkers {
                                if let r = combined.range(of: marker) {
                                    let off = combined.utf8.distance(
                                        from: combined.utf8.startIndex,
                                        to: r.lowerBound.samePosition(in: combined.utf8)
                                            ?? combined.utf8.startIndex
                                    )
                                    if off < earliest { earliest = off }
                                }
                            }
                            targetCap = earliest
                        }
                    } else {
                        // Not buffered — hold back the last `maxMarkerLenMinus1`
                        // bytes in case the next delta completes a marker
                        // that starts in the tail.
                        targetCap = max(emittedContentBytes,
                                        combinedBytes - maxMarkerLenMinus1)
                    }
                    let emitLen = max(0, targetCap - emittedContentBytes)
                    if emitLen > 0 {
                        let allUtf8 = Array(combined.utf8)
                        let slice = allUtf8[emittedContentBytes..<(emittedContentBytes + emitLen)]
                        emittableContent = String(decoding: slice, as: UTF8.self)
                        emittedContentBytes += emitLen
                    } else {
                        emittableContent = ""
                    }
                }

                // §163.B4: bundle pending logprobs into the SAME chunk
                // as the content delta. OpenAI emits both in the same
                // SSE frame. On a stop-rollback we still attach the
                // logprobs to the partial-content chunk — logprobs are
                // per-sampled-token, regardless of how many chars made
                // it past the stop trim (see §163.B3 note in
                // LOGPROBS-IMPLEMENTATION.md).
                if !emittableContent.isEmpty {
                    let attachedLogprobs = takeLogprobs()
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
                        visibleAccumulator += emittableContent
                        if let m = stopMatcher.firstMatch(in: visibleAccumulator) {
                            let pat = stopMatcher.patterns[m.patternIndex]
                            let patBytes = pat.utf8.count
                            let matchStartByte = m.endByteOffset + 1 - patBytes
                            // How much of the stop string lies inside the
                            // current delta? Anything before that goes out
                            // unchanged; nothing after it does.
                            let keepBytesInDelta = max(0, matchStartByte - preLen)
                            let utf8 = Array(emittableContent.utf8)
                            if keepBytesInDelta > 0 {
                                let keep = String(decoding: utf8.prefix(keepBytesInDelta), as: UTF8.self)
                                continuation.yield(StreamChunk(content: keep, logprobs: attachedLogprobs))
                            } else if let lps = attachedLogprobs {
                                // All content trimmed by stop match but
                                // we still owe the caller the sampled-
                                // token logprobs for this step.
                                continuation.yield(StreamChunk(logprobs: lps))
                            }
                            stopHit = true
                        } else {
                            continuation.yield(StreamChunk(content: emittableContent, logprobs: attachedLogprobs))
                        }
                    } else {
                        continuation.yield(StreamChunk(content: emittableContent, logprobs: attachedLogprobs))
                    }
                } else {
                    // §163.B6: no content this tick → token contributed
                    // purely to reasoning, tool-call JSON, or a
                    // buffered prefix. OpenAI's
                    // `choices[0].logprobs.content` MUST map 1:1 with
                    // the tokens of `message.content`; reasoning/tool
                    // tokens do not belong in that array. Dropping
                    // pendingLogprobs here keeps alignment.
                    //
                    // Known corner: if a tool-call marker turns out NOT
                    // to be a real call (parser rejects and bytes
                    // later flush as content in case .info), those
                    // logprobs are already dropped. Rare and
                    // OpenAI-compat-preserving — flagged in
                    // LOGPROBS-IMPLEMENTATION.md.
                    pendingLogprobs = []
                }
                if stopHit {
                    // Iter 144 — held-back tail (the maxMarkerLenMinus1
                    // bytes withheld for marker-detection in
                    // `haveToolsOrParser && !streamingAcc.buffered`) is
                    // intentionally NOT flushed here. By construction
                    // those bytes are at offsets ≥ `emittedContentBytes`,
                    // which itself is ≥ the stop-match start (the stop
                    // pattern matched WITHIN the just-yielded slice).
                    // So the held-back tail is positionally AFTER the
                    // stop position — dropping it matches stop-string
                    // semantics. A stream-audit reviewer flagged this
                    // as a potential bug at confidence 80; the trace
                    // shows it's correct. Don't re-flag without
                    // re-tracing.
                    break
                }

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
                //
                // (Accumulator was already fed above for the marker-bleed
                // guard; this block now only runs the parse+emit half.)
                if haveToolsOrParser, streamingAcc.buffered, let parser = toolParser {
                    let parsed = parser.parse(streamingAcc.current)
                    // Iter 144 — drop calls with nil argumentsJSON.
                    // Synthesizing `"{}"` previously let the engine
                    // execute structurally-incomplete tool calls (e.g.
                    // `<tool_call>{"name":"bash"}` with no args) up to
                    // 3 times before the repetition detector fires.
                    // For tools requiring arguments this is wasted work
                    // at best, dangerous at worst. Skip the call and
                    // fall through to the content-flush path below.
                    let validParsed = parsed.filter { $0.argumentsJSON != nil }
                    if validParsed.count < parsed.count {
                        await self.log(.warn, "engine",
                            "tool-call parser produced \(parsed.count - validParsed.count) calls with nil arguments — dropped (model emitted malformed JSON, e.g. missing `arguments` key)")
                    }
                    if !validParsed.isEmpty {
                        for p in validParsed {
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
                // Flush any remaining logprobs before the final metrics.
                flushLogprobs()
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

                // iter-59 end-of-stream buffered-tool-call flush: if we held
                // back bytes in the accumulator expecting a tool call (marker
                // was detected) but the parser never produced a call (e.g.,
                // model hit max_tokens mid-call, or JSON ended malformed),
                // release the buffered text as content rather than losing it
                // silently. This keeps the contract that every decoded
                // character lands somewhere visible to the caller.
                if streamingAcc.buffered, let parser = toolParser {
                    let parsed = parser.parse(streamingAcc.current)
                    // Iter 144 — same nil-arguments filter as the
                    // mid-stream parse site above. End-of-stream flush
                    // can also see truncated tool calls (model hit
                    // max_tokens mid-JSON); drop them rather than
                    // synthesize `"{}"` and execute.
                    let validParsed = parsed.filter { $0.argumentsJSON != nil }
                    if validParsed.count < parsed.count {
                        await self.log(.warn, "engine",
                            "end-of-stream tool-call parser dropped \(parsed.count - validParsed.count) calls with nil arguments (truncated JSON)")
                    }
                    if !validParsed.isEmpty {
                        for p in validParsed {
                            let call = ChatRequest.ToolCall(
                                id: "call_\(UUID().uuidString.prefix(8))",
                                type: "function",
                                function: .init(
                                    name: p.name,
                                    arguments: p.argumentsJSON ?? "{}"))
                            collectedToolCalls.append(call)
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
                        emittedContentBytes = 0
                    } else {
                        // Give up on parsing — flush the raw buffered text
                        // from marker-start onward (pre-marker bytes were
                        // already yielded). iter-152 §222: use
                        // emittedContentBytes as the cut point so we don't
                        // double-emit pre-marker content.
                        let allUtf8 = Array(streamingAcc.current.utf8)
                        let start = min(emittedContentBytes, allUtf8.count)
                        if start < allUtf8.count {
                            let tail = String(decoding: allUtf8[start...], as: UTF8.self)
                            continuation.yield(StreamChunk(content: tail))
                        }
                        streamingAcc.reset()
                        emittedContentBytes = 0
                    }
                } else if streamingAcc.buffered {
                    // Parser absent but buffered bytes accumulated via
                    // user-supplied tools — no way to parse, flush from
                    // marker-start onward (same reasoning as above).
                    let allUtf8 = Array(streamingAcc.current.utf8)
                    let start = min(emittedContentBytes, allUtf8.count)
                    if start < allUtf8.count {
                        let tail = String(decoding: allUtf8[start...], as: UTF8.self)
                        continuation.yield(StreamChunk(content: tail))
                    }
                    streamingAcc.reset()
                    emittedContentBytes = 0
                } else if (toolParser != nil || (mergedTools?.isEmpty == false)),
                          emittedContentBytes < streamingAcc.current.utf8.count {
                    // iter-152 §222: flush held-back (non-marker) tail.
                    // When tools/parser are attached we hold back the last
                    // maxMarkerLenMinus1 bytes of each delta; if the stream
                    // ends cleanly with no marker ever landing, those bytes
                    // must be released now.
                    let allUtf8 = Array(streamingAcc.current.utf8)
                    let tail = String(
                        decoding: allUtf8[emittedContentBytes...],
                        as: UTF8.self
                    )
                    if !tail.isEmpty {
                        // Stop-matcher still applies at flush time.
                        if !stopMatcher.isEmpty {
                            let preLen = visibleAccumulator.utf8.count
                            visibleAccumulator += tail
                            if let m = stopMatcher.firstMatch(in: visibleAccumulator) {
                                let pat = stopMatcher.patterns[m.patternIndex]
                                let patBytes = pat.utf8.count
                                let matchStartByte = m.endByteOffset + 1 - patBytes
                                let keepBytes = max(0, matchStartByte - preLen)
                                let tailUtf8 = Array(tail.utf8)
                                if keepBytes > 0, keepBytes <= tailUtf8.count {
                                    let keep = String(
                                        decoding: tailUtf8.prefix(keepBytes),
                                        as: UTF8.self)
                                    continuation.yield(StreamChunk(content: keep))
                                }
                            } else {
                                continuation.yield(StreamChunk(content: tail))
                            }
                        } else {
                            continuation.yield(StreamChunk(content: tail))
                        }
                    }
                    emittedContentBytes = streamingAcc.current.utf8.count
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
                //
                // Iter 144 — VLM caveat. `info.promptTokenCount` is
                // post-prepare (text + expanded image-pad tokens),
                // while `cachePreFetch.matched` is computed against
                // the raw text-token array. For VLM requests with
                // images, the ratio (cachedTokens / promptTokens) is
                // mechanically lower than the user's intuition because
                // image tokens are never cache-eligible but still
                // count in the denominator. Numbers are mathematically
                // consistent (matched ≤ raw text count ≤ expanded
                // count), but operators reading the ratio for VLM
                // workloads should know it's measuring "text-prefix
                // hits" / "total tokens incl. image" not "cache
                // hits" / "cacheable tokens".
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
                // iter-64: flush any remainder accumulated after the
                // last N-token batch boundary (both for perTokenMetricsDisabled
                // path and the normal batched path — end-of-stream always
                // has a sub-batch tail).
                if accumulatedDecodeTokens > 0 {
                    await self.metrics.recordTokenBatch(
                        prefill: false,
                        count: accumulatedDecodeTokens,
                        durationMs: accumulatedDecodeMs)
                    accumulatedDecodeTokens = 0
                    accumulatedDecodeMs = 0
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


            case .logprob(let lp):
                if shouldCollectLogprobs {
                    pendingLogprobs.append(lp)
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
        // 2026-04-18 cancel-SSM-rederive fix — crash diagnosis from a
        // vmlxctl DiagnosticReport showed EXC_BAD_ACCESS in
        // `mlx::core::binary_op_gpu_inplace` during a MLX eval in
        // `reDeriveSSMStates`, triggered via this detached Task after a
        // mid-stream cancel. Reproducer: Qwen3.6-35B-A3B-MXFP4 (hybrid
        // SSM) + harness `cancel_midstream`. The detached re-derive
        // captured pointers into `promptTokenIds` / the ModelContainer's
        // arrays that had already started tearing down on cancel; the
        // follow-up MLX binary op hit a freed MTL::Resource.
        //
        // Fix: skip the re-derive when the generation was cancelled.
        // Cancellation implies the user no longer cares about the SSM
        // state for this turn, so spawning the re-derive is both
        // unnecessary AND unsafe. Also harden the detached branch with
        // an `isCancelled` guard inside the Task so a late-arriving
        // cancellation still short-circuits before touching MLX.
        // §337 — skip re-derive on VL turns. `maybeReDeriveSSMState`
        // constructs a text-only `LMInput` (no image/video) and runs a
        // fresh forward pass; for a VLM request whose actual
        // generation mixed image embeddings into the prompt tokens,
        // the re-derived text-only SSM state doesn't reflect the real
        // state at any token position — restoring it on a future turn
        // would feed garbage into every mamba layer. The VL/hybrid
        // intersection is rare today (Qwen3.5-VL hybrid + thinking
        // being the clearest match), but when it occurs the wasted
        // re-derive is worse than no re-derive. Skip when userInput
        // has any image or video content; the SSM companion for this
        // turn was already elided by `shouldSkipSSMStorage`, so the
        // next VL turn will just re-prefill normally (matches the
        // pre-§317 correctness baseline).
        let isVLTurn =
            !userInput.images.isEmpty || !userInput.videos.isEmpty || !userInput.audios.isEmpty
        if collectedToolCalls.isEmpty,
           !Task.isCancelled,
           !isVLTurn,
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
                // Iter 144 — capture the JangPress controller and rebind
                // the TaskLocal inside the detached re-derive. Without
                // this, `Task.detached` does not inherit the parent
                // Task's TaskLocal values, so any
                // `JangPressRouteTelemetry.recordTopK` issued from the
                // re-derive's forward pass sees `controller = nil` and
                // is silently dropped — undercounting expert activations
                // on hybrid+thinking models.
                let captureEmber = self.emberController
                Task.detached(priority: .utility) {
                    // Late-cancel guard — if the parent Task was cancelled
                    // between spawn and our first resume point, bail before
                    // touching MLX arrays whose Metal resources may be
                    // mid-teardown.
                    if Task.isCancelled { return }
                    await JangPressRouteTelemetry.$controller.withValue(captureEmber) {
                        await captureContainer.perform { (ctx: ModelContext) in
                            if Task.isCancelled { return }
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
        injectReasoningOffStub: Bool = true,
        responseFormatInstruction: String? = nil,
        imageMarker: String = "<image>",
        videoMarker: String = "<video>"
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
            videos: [UserInput.Video],
            audios: [UserInput.Audio]
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
            let audios = await extractAudios(from: msg.content)
            decoded.append((role: role, text: text, images: images, videos: videos, audios: audios))
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
            // Iter 143 — was raw `print()`. Static helper can't hop
            // to Engine actor, but structured stderr + `[vMLX][vlm]`
            // prefix keeps the line grep-able and consistent with the
            // other load/factory/cache stderr lines. The outer
            // performOneGenerationPass still logs total image count
            // via Engine.logs; this cap event is secondary diagnostic.
            if let data = "[vMLX][vlm] multi-turn VL image cap: kept \(Self.maxTotalImagesPerRequest) of \(totalImages) images (oldest dropped)\n".data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
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
            // Iter 144 — added the Qwen-family triple
            // `<|vision_start|><|image_pad|><|vision_end|>` (resolved
            // at line ~822 above as `imageMarker` for Qwen models).
            // A user manually pasting that string in the prompt
            // previously got it doubled, breaking image-token
            // expansion. Audit catch from production review.
            let hasAnyMarker =
                renderedText.contains("<|image|>")
                || renderedText.contains("<image>")
                || renderedText.contains("[IMG]")
                || renderedText.contains("<|image_pad|>")
                || renderedText.contains("<|vision_start|>")
            if !d.images.isEmpty && !hasAnyMarker {
                // One marker per image, prefixed before the text. No newline
                // between marker and text — some chat templates split on
                // whitespace and the newline breaks image-token expansion.
                // Python engine convention: concat directly.
                let markers = String(repeating: imageMarker, count: d.images.count)
                renderedText = markers + renderedText
            }
            // Video marker auto-insert. Qwen3.6/Qwen3-VL bundles need the
            // full `<|vision_start|><|video_pad|><|vision_end|>` triple so
            // `QwenVL.replacePaddingTokens` can expand frame placeholders;
            // Nemotron Omni uses the literal `<video>` token from
            // `config_omni.json`.
            let hasAnyVideoMarker =
                renderedText.contains("<video>")
                || renderedText.contains("<|video_pad|>")
            if !d.videos.isEmpty && !hasAnyVideoMarker {
                let vMarkers = String(repeating: videoMarker, count: d.videos.count)
                renderedText = vMarkers + renderedText
            }
            out.append(Chat.Message(
                role: d.role,
                content: renderedText,
                images: d.images,
                videos: d.videos,
                audios: d.audios))
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
        if injectReasoningOffStub
            && !effectiveThinking && !hasTools && !modelStampsThink && !hasStructuredFormat
        {
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
    /// Iter-25: gated VL debug logger. Writes to stderr only when
    /// `VMLX_VL_DEBUG=1` is set. Computed once per-process into a
    /// static flag so we don't re-read the env on every chunk.
    static let vlDebugEnabled: Bool = {
        ProcessInfo.processInfo.environment["VMLX_VL_DEBUG"] == "1"
    }()

    static func vlDebug(_ msg: @autoclosure () -> String) {
        guard vlDebugEnabled else { return }
        FileHandle.standardError.write(Data("[vmlx][vl][debug] \(msg())\n".utf8))
    }

    static func extractImages(
        from content: ChatRequest.ContentValue?
    ) async -> [UserInput.Image] {
        // Iter-25: gate the VL debug logs on an env flag (default off
        // in production). Pre-fix every text-only request on a
        // VL-capable model wrote "[vmlx][vl][debug] content is nil" to
        // stderr — on a 100 QPS deployment that's 100 lines/sec of
        // noise. `VMLX_VL_DEBUG=1` re-enables them for diagnosis.
        switch content {
        case .none:
            Self.vlDebug("content is nil")
            return []
        case .some(.string):
            // Text-only content is the common case — don't spam the logs.
            return []
        case .some(.parts(let parts)):
            let kinds = parts.map { $0.type }.joined(separator: ",")
            Self.vlDebug("content parts=\(parts.count) types=[\(kinds)]")
            // 2026-04-18 VL memory fix — community users report
            // excessive RAM usage with large image batches. Cap the
            // per-message image count at `maxTotalImagesPerRequest`
            // BEFORE decoding so we never hold N × CIImage+Data in
            // memory when the downstream `buildChatMessages` cap is
            // going to drop them anyway. Previously we decoded every
            // image_url part (retaining CIImage + raw bytes for each)
            // and only dropped the overflow after all CIImages were
            // alive, which was the peak-memory event users hit.
            let imageParts = parts.filter { $0.type == "image_url" }
            let cap = Self.maxTotalImagesPerRequest
            let budgeted = imageParts.prefix(cap)
            if imageParts.count > cap {
                FileHandle.standardError.write(Data(
                    "[vmlx][vl] decode cap reached: keeping \(cap) of \(imageParts.count) images (skipped at parse time)\n".utf8))
            }
            var out: [UserInput.Image] = []
            out.reserveCapacity(budgeted.count)
            for part in budgeted {
                guard let imageURL = part.imageUrl else {
                    Self.vlDebug("image_url part missing imageUrl field")
                    continue
                }
                Self.vlDebug("loading image from: \(imageURL.url.prefix(60))…")
                guard let data = await imageURL.loadImageData() else {
                    // This one stays unconditional — genuine warning
                    // about a failed fetch, users want to see it.
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] image URL fetch/decode failed: \(imageURL.url.prefix(80))…\n".utf8))
                    continue
                }
                Self.vlDebug("image bytes=\(data.count)")
                guard let ci = CIImage(data: data) else {
                    // Warning: unconditional (real failure signal).
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] CIImage decode failed (\(data.count) bytes) — possibly a non-image format; text-only reply will follow\n".utf8))
                    continue
                }
                Self.vlDebug("CIImage loaded OK, extent=\(ci.extent)")
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
                // iter-67: probe the staged file with AVURLAsset before
                // forwarding to the VLM. Garbage-base64 payloads happily
                // decode into bytes → write a .mp4 extension → crash the
                // downstream video processor with a Metal/AVFoundation
                // fault that manifests as HTTP 500 instead of a clean
                // reject+drop. Probing `tracks(withMediaType: .video)`
                // forces the demuxer to parse the file header; failure
                // means we drop this URL and continue (text-only turn).
                // Harness `case_video_url_handling` asserts this path.
                let asset = AVURLAsset(url: localURL)
                let hasVideoTrack: Bool
                do {
                    let tracks = try await asset.loadTracks(withMediaType: .video)
                    hasVideoTrack = !tracks.isEmpty
                } catch {
                    hasVideoTrack = false
                }
                if !hasVideoTrack {
                    FileHandle.standardError.write(Data(
                        "[vmlx][vl] video_url decoded but no video tracks — skipping\n".utf8))
                    // Best-effort temp cleanup; ignore errors.
                    try? FileManager.default.removeItem(at: localURL)
                    continue
                }
                Self.vlDebug("video staged at \(localURL.path)")
                out.append(.url(localURL))
            }
            return out
        }
    }

    /// Extract audio parts from OpenAI-style multimodal content into
    /// `UserInput.Audio` values. Supports vMLX `audio_url` and OpenAI
    /// Responses/chat `input_audio` parts. Non-audio models simply ignore
    /// these values because their processors never read `input.audios`.
    static func extractAudios(
        from content: ChatRequest.ContentValue?
    ) async -> [UserInput.Audio] {
        switch content {
        case .none, .some(.string):
            return []
        case .some(.parts(let parts)):
            var out: [UserInput.Audio] = []
            for part in parts {
                if part.type == "audio_url" {
                    guard let audioURL = part.audioUrl,
                          let localURL = await audioURL.loadAudioLocalURL()
                    else {
                        FileHandle.standardError.write(Data(
                            "[vmlx][vl] audio_url fetch/decode failed\n".utf8))
                        continue
                    }
                    out.append(.url(localURL))
                } else if part.type == "input_audio" {
                    guard let localURL = part.inputAudio?.loadAudioLocalURL() else {
                        FileHandle.standardError.write(Data(
                            "[vmlx][vl] input_audio base64 decode failed\n".utf8))
                        continue
                    }
                    out.append(.url(localURL))
                }
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
        let loadScopedOptions: LoadOptions? =
            (request.sessionId == nil && request.chatId == nil)
            ? self.lastLoadOptions
            : nil

        var params = GenerateParameters()
        params.prefillStepSize =
            loadScopedOptions?.prefillStepSize
            ?? resolved.settings.prefillStepSize

        // §323 KIMI-K2.6-VMLX-INTEGRATION.md §2.6 #6-#8 — always chunk
        // prefill on DSV3-family MLA bundles (kimi_k25, deepseek_v3/v2/v32).
        // Default 2048 trips the Metal watchdog on 100 GB+ MoE because the
        // per-kernel JIT compile dominates cold first-forward. Kimi text
        // parity is 16 tokens; VLM can use 32 in its model-specific prepare.
        let cacheTypeIsMLAForPrefill = (self.modelCapabilities?.cacheType == "mla")
        if cacheTypeIsMLAForPrefill && params.prefillStepSize > 16 {
            params.prefillStepSize = 16
        }
        // §373 + §443 — sampling fallback priority (highest-to-lowest):
        //   1. explicit request field (OpenAI/Anthropic/Ollama/CLI body)
        //   2. user-set session/chat override (via SettingsStore cascade)
        //   3. model's `generation_config.json` (loadedModelDefaults)
        //   4. **family-specific defaults** (FamilySamplingDefaultsRegistry)
        //   5. compiled-in global default (0.7 / 0.9 / etc.)
        //
        // The SettingsStore resolver collapses 1+2+5 into `resolved.*`
        // and exposes the provenance via `resolutionTrace`. When the
        // trace says `.global` we know nobody (request/chat/session)
        // actually set it, so `resolved.temperature` is just the compiled
        // builtin — at that point we consult the model's
        // generation_config.json (tier 3), and if that's silent too, we
        // fall through to the family registry (tier 4). When the trace
        // says `.request/.chat/.session` the user made an explicit
        // choice — keep it. Stops silently overriding user session
        // overrides with generation_config when the user has opinions
        // and stops shipping garbage-token-prone global defaults to
        // families with known-good samplers (Nemotron-H, DeepSeek V3/V4).
        let md = self.loadedModelDefaults
        let trace = resolved.resolutionTrace
        let familyDefaults = FamilySamplingDefaultsRegistry.defaults(
            forFamily: self.modelCapabilities?.family ?? "")
        func pickDouble(_ traceKey: String, _ modelDefault: Double?,
                        _ familyDefault: Float?, _ resolvedValue: Double) -> Double {
            if trace[traceKey] == .global {
                if let m = modelDefault { return m }
                if let f = familyDefault { return Double(f) }
            }
            return resolvedValue
        }
        func pickInt(_ traceKey: String, _ modelDefault: Int?,
                     _ familyDefault: Int?, _ resolvedValue: Int) -> Int {
            if trace[traceKey] == .global {
                if let m = modelDefault { return m }
                if let f = familyDefault { return f }
            }
            return resolvedValue
        }
        params.maxTokens = request.maxTokens
            ?? pickInt("defaultMaxTokens", md.maxTokens, nil, resolved.maxTokens)
        params.temperature = Float(request.temperature
            ?? pickDouble("defaultTemperature", md.temperature,
                          familyDefaults?.temperature, resolved.temperature))
        params.topP = Float(request.topP
            ?? pickDouble("defaultTopP", md.topP,
                          familyDefaults?.topP, resolved.topP))
        params.topK = request.topK
            ?? pickInt("defaultTopK", md.topK,
                       familyDefaults?.topK, resolved.topK)
        params.minP = Float(request.minP ?? resolved.minP)
        params.repetitionPenalty = Float(request.repetitionPenalty
            ?? pickDouble("defaultRepetitionPenalty", md.repetitionPenalty,
                          familyDefaults?.repetitionPenalty, resolved.repetitionPenalty))
        // iter-95 §173: wire OpenAI `frequency_penalty` +
        // `presence_penalty` through to the sampler. Evaluate.swift
        // has supported both since iter-25 (see `FrequencyPenalty`
        // + `PresencePenalty` LogitProcessor types at ~L477/L505)
        // but `buildGenerateParameters` was only setting
        // `repetitionPenalty`. Result: every chat completions
        // request that specified `frequency_penalty: 1.0` or
        // `presence_penalty: 0.8` was accepted (validate() allows
        // [-2, 2]) but the value was silently dropped — users
        // saw "penalty had no effect" with no diagnostic. Fix:
        // forward the clamped Double into the Float fields only
        // when non-nil + non-zero so the zero-cost path stays
        // identical to before for the 99% case that doesn't set
        // them.
        if let fp = request.frequencyPenalty, fp != 0 {
            params.frequencyPenalty = Float(fp)
        }
        if let pp = request.presencePenalty, pp != 0 {
            params.presencePenalty = Float(pp)
        }
        // Iter 143: wire OpenAI `logit_bias` to the sampler. Replaces
        // the iter-96 §174 FIXME (accepted-but-unwired). OpenAI sends
        // bias as a `[token_id_string: number]` map; we parse the
        // string keys to Int and drop unparseable entries silently
        // (some clients ship empty strings or the literal "default"
        // key — neither of which a Swift Int initializer can recover).
        // Range is documented as [-100, 100]; we don't hard-clamp so
        // power users sending [-200, 200] for stronger biases still get
        // them. The LogitBiasContext handles out-of-vocab IDs by
        // ignoring them (see Evaluate.swift LogitBiasContext.process).
        if let rawBias = request.logitBias, !rawBias.isEmpty {
            var parsed: [Int: Float] = [:]
            for (key, value) in rawBias {
                if let id = Int(key) {
                    parsed[id] = Float(value)
                }
            }
            if !parsed.isEmpty {
                params.logitBias = parsed
            }
        }
        // iter-64: sampler seed. Samplers in Evaluate.swift hold
        // private RandomState instances that ignore the global
        // MLXRandom.seed() call earlier in stream(); this plumbs the
        // user-supplied seed through to them. Without this, two
        // requests with the same `seed` produced different output.
        if let seed = request.seed {
            params.samplerSeed = UInt64(bitPattern: Int64(seed))
        }

        // Logprobs: wire request fields into GenerateParameters so
        // LogprobsCollector is activated inside the TokenIterator.
        params.logprobs = request.logprobs ?? false
        params.topLogprobs = request.topLogprobs ?? 0

        let env = ProcessInfo.processInfo.environment
        let caps = self.modelCapabilities
        let family = (caps?.family ?? "").lowercased()
        let modelType = (caps?.modelType ?? "").lowercased()
        let modality = caps?.modality ?? .unknown
        let requestHasMedia = request.messages.contains { message in
            guard case .parts(let parts) = message.content else { return false }
            return parts.contains { part in
                let kind = part.type.lowercased()
                return kind != "text"
                    || part.imageUrl != nil
                    || part.videoUrl != nil
                    || part.audioUrl != nil
                    || part.inputAudio != nil
            }
        }
        if (family.contains("laguna") || modelType.contains("laguna")),
           let rp = params.repetitionPenalty,
           rp != 0,
           rp != 1.0 {
            let lagunaRepContext = Int(
                env["VMLX_LAGUNA_REPETITION_CONTEXT_SIZE"] ?? "256"
            ) ?? 256
            params.repetitionContextSize = max(
                params.repetitionContextSize,
                max(20, lagunaRepContext))
        }
        let slidingMode = (
            loadScopedOptions?.slidingWindowMode
            ?? resolved.settings.slidingWindowMode
        ).lowercased()

        // Whole-model compiled decode (perf audit 2026-04-16 and SWA
        // follow-up). Enabling `enableCompiledDecode` wraps the entire
        // decode forward pass in MLX compile(), collapsing MoE routing,
        // attention, and FFN work into a single traced graph. Leaving this
        // behind an env var made SWA/MoE models look "correct" but run at
        // the 20-60 tok/s uncompiled floor. Apply a conservative family
        // policy for model classes whose cache topology is known to be
        // compile-safe after promotion.
        //
        // DSV4 is deliberately excluded: its HSA/CSA/SWA long-context path
        // uses DSV4LayerCache pool-state semantics and must not be flattened
        // into the generic SWA policy.
        //
        // Laguna is now included in the bounded-SWA compile-first policy,
        // but only on the speed profile where default KV TurboQuant is
        // suppressed below. Live repro 2026-05-03:
        //
        //   VMLX_COMPILED_DECODE=1 VMLX_DISABLE_TURBO_QUANT=1
        //     → rotating=40, stable.
        //
        //   VMLX_COMPILED_DECODE=1 with KV TQ on a long prompt
        //     → tq=10, SIGTRAP in LagunaMoE argPartition trace.
        //
        // So production default is "bounded rotating compile" for speed,
        // while `VMLX_FORCE_TURBO_QUANT=1` remains a diagnostics-only
        // memory-mode escape hatch.
        let compileFirstFamilies: Set<String> = [
            "laguna",
            "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_next",
            "nemotron_h",
            "minimax", "minimax_m2_5",
            "mistral3", "ministral3", "mistral4",
            "gemma3_text", "gemma4_text",
            "mimo_v2_flash", "baichuan_m1", "afmoe", "gpt_oss",
        ]
        let boundedSwaFamilies: Set<String> = [
            "laguna",
            "mistral3", "ministral3", "mistral4",
            "gemma3_text", "gemma4_text",
            "pixtral",
        ]
        let isDSV4 = family == "deepseek_v4" || modelType == "deepseek_v4"
        let textOnlyVLMCompileFamilies: Set<String> = [
            "gemma4", "gemma4_text",
        ]
        let textOnlyVLMCompileEligible =
            modality == .vision
            && !requestHasMedia
            && (textOnlyVLMCompileFamilies.contains(family)
                || textOnlyVLMCompileFamilies.contains(modelType))
        let compileEligibleModality =
            modality == .text || textOnlyVLMCompileEligible
        let isCompileFirstFamily =
            !isDSV4
            && compileEligibleModality
            && (compileFirstFamilies.contains(family)
                || modelType == "ministral3")
        let isBoundedSWACompileFamily =
            boundedSwaFamilies.contains(family)
            || modelType == "laguna"
            || modelType == "ministral3"
        let nativeSlidingWindow = caps?.slidingWindow
        let configuredSlidingWindow = max(
            512,
            loadScopedOptions?.slidingWindowSize
                ?? resolved.settings.slidingWindowSize)
        let effectiveSWACompileWindow: Int = {
            switch slidingMode {
            case "bounded":
                return configuredSlidingWindow
            case "auto":
                // `auto` means honor `config.json::sliding_window`, not the
                // global UI fallback. This matters for Laguna
                // (`sliding_window=512`): compiling a 16k rotating state
                // erased the intended SWA fast path and left decode near the
                // uncompiled floor.
                return max(512, nativeSlidingWindow ?? configuredSlidingWindow)
            default:
                return configuredSlidingWindow
            }
        }()

        let compiledDecodeEnv = env["VMLX_COMPILED_DECODE"]
        if compiledDecodeEnv == "1" {
            params.enableCompiledDecode = true
        } else if compiledDecodeEnv == "0" {
            params.enableCompiledDecode = false
        } else if isCompileFirstFamily
                    && !(isBoundedSWACompileFamily && slidingMode == "long") {
            params.enableCompiledDecode = true
        }

        if params.enableCompiledDecode {
            params.compiledMaxCacheLength = max(
                params.compiledMaxCacheLength ?? 0, effectiveSWACompileWindow)

            // Full-attention layers in Laguna/Gemma/Mistral/Pixtral can
            // otherwise allocate their full declared context (Laguna is
            // 131072). For the compile-first speed profile, cap those
            // layers to the model-native SWA window in `auto`, or to the
            // operator's explicit size in `bounded`. Do not cap Mistral 3.5
            // / ministral3 production bundles that declare `sliding_window:
            // null`; they have no SWA layers and should keep full-context
            // semantics unless the operator explicitly chose `bounded`.
            if isBoundedSWACompileFamily
                && slidingMode != "long"
                && (nativeSlidingWindow != nil || slidingMode == "bounded")
            {
                params.maxKVSize = effectiveSWACompileWindow
            }
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
        let tqDisabledViaEnv = env["VMLX_DISABLE_TURBO_QUANT"] == "1"
        let tqForcedViaEnv = env["VMLX_FORCE_TURBO_QUANT"] == "1"

        // JANG auto-activation: if the loaded model's `jang_config.json`
        // declares `"turboquant": {"enabled": true, ...}`, activate TQ
        // REGARDLESS of the global `enableTurboQuant` flag. These models
        // are calibrated with specific bit widths per layer role and expect
        // the KV to be compressed at generate-time. The global flag only
        // governs MLX-format models without a calibrated TQ block.
        // Matches Engine.swift:36-41 documented intent.
        //
        // 2026-05-01 STATUS: the auto-activation path is **dormant in
        // practice** — every JANGTQ bundle currently in the wild omits
        // the `turboquant` block (verified across 10+ bundles spanning
        // Kimi K2.6, MiniMax M2.7, Qwen 3.5/3.6, DeepSeek V4, Nemotron-
        // Omni, Laguna, Mistral 4 — see JangLoader.swift parser comment).
        // TQ KV stays OFF for JANGTQ bundles unless the user explicitly
        // toggles `enableTurboQuant`. This is consistent with the
        // 2026-04-16 perf audit (default-on TQ KV cost 25-40 % on
        // MoE/hybrid). The wiring is preserved so a future jang_tools
        // release can opt a model in without a Swift-side change.
        //
        // Precedence:
        //   1. MLA model       → skip (layer shape incompatible)
        //   2. env killswitch  → skip
        //   3. JANG calibrated → activate with jang bits  (DORMANT — no
        //                        bundle currently emits the block)
        //   4. global toggle   → activate with global turboQuantBits
        //                        (the live opt-in path)
        //   5. otherwise       → skip (default OFF post 2026-04-16 audit)
        let jangTQEnabled = (self.loadedJangConfig?.turboquant.enabled == true)
        let enableTurboQuant =
            loadScopedOptions?.enableTurboQuant
            ?? resolved.settings.enableTurboQuant
        let compileFirstSuppressesTQ =
            params.enableCompiledDecode
            && isCompileFirstFamily
            && !tqForcedViaEnv
            && !jangTQEnabled
        let shouldActivateTQ =
            !cacheTypeIsMLA
            && !tqDisabledViaEnv
            && !compileFirstSuppressesTQ
            && (jangTQEnabled || enableTurboQuant)
        // Iter 143 — Eric directive 2026-05-04: TQ default-on for ALL
        // models. The three skip paths (MLA / env-killswitch /
        // compile-first) each have a load-bearing justification:
        //  • MLA: latent KV is already compressed natively; logged
        //    once at Engine.swift load time.
        //  • env-killswitch: operator-explicit, no log needed.
        //  • compile-first: the bounded-SWA fast path on
        //    Laguna/Qwen3.5/Gemma/Mistral runs without the per-step
        //    TQ compress/dequant cycle (25–40% faster per the
        //    2026-04-16 audit). Override via VMLX_FORCE_TURBO_QUANT=1.
        //    Per-request log not added here because
        //    `buildGenerateParameters` is non-async (logging requires
        //    actor hop). The load-time log line at
        //    Engine.swift:2581 already documents the family-level
        //    decision; CompiledDecodeMonitor inserts a one-shot info
        //    line in `applyCompiledDecode` when compile is enabled.
        if shouldActivateTQ {
            let keyBits: Int
            let valueBits: Int
            if let tq = self.loadedJangConfig?.turboquant, tq.enabled {
                keyBits = tq.defaultKeyBits
                valueBits = tq.defaultValueBits
            } else {
                let bits =
                    loadScopedOptions?.turboQuantBits
                    ?? resolved.settings.turboQuantBits
                keyBits = bits
                valueBits = bits
            }
            params.kvMode = .turboQuant(keyBits: keyBits, valueBits: valueBits)
            params.kvBits = nil
        }
        if env["VMLX_COMPILED_DECODE_DEBUG"] == "1" {
            let line = "[vmlx][generation-params] family=\(family) modelType=\(modelType) " +
                "compiled=\(params.enableCompiledDecode) nativeSlidingWindow=\(nativeSlidingWindow.map(String.init) ?? "nil") " +
                "effectiveSWACompileWindow=\(effectiveSWACompileWindow) maxKVSize=\(params.maxKVSize.map(String.init) ?? "nil") " +
                "compiledMaxCacheLength=\(params.compiledMaxCacheLength.map(String.init) ?? "nil") " +
                "turboQuantActive=\(shouldActivateTQ) turboQuantSuppressed=\(compileFirstSuppressesTQ)\n"
            try? FileHandle.standardError.write(contentsOf: Data(line.utf8))
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
        // §344 — parser-dispatch path also drains a residual via
        // `finishStreaming(fullText:)`. Gemma 4's parser holds back up to 18
        // chars as a `<|channel>thought` marker window, and on very short
        // replies (<18 chars) those bytes are otherwise lost. Mirrors the
        // live Stream loop at Stream.swift:1705.
        if let parser,
           let residual = parser.finishStreaming(fullText: parserCur)
        {
            if let r = residual.reasoning, !r.isEmpty {
                if effectiveThinking { reasoning += r } else { content += r }
            }
            if let c = residual.content, !c.isEmpty {
                content += c
            }
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
    /// Best-effort streaming parse. Returns empty so the streaming
    /// accumulator keeps buffering until vmlx-swift-lm's native
    /// `.toolCall(_)` event emits the parsed call. Wiring this to
    /// `extractToolCalls` was attempted in iter-22 but duplicated Qwen/
    /// Hermes calls (native + shim both fired with same content). The
    /// correct path is a per-token streaming parser with an end-of-call
    /// delimiter — left for a future iteration.
    func parse(_ text: String) -> [(name: String, argumentsJSON: String?)] {
        return []
    }
}
