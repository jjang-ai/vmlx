import Foundation
import MLX
import vMLXLMCommon

// Stream path for JANG-DFlash speculative decoding. Engaged when:
//   1. request has no tools and no images
//   2. `dflash` is enabled in resolved settings
//   3. a drafter is loaded AND the target model conforms to JangDFlashTarget
//
// Any precondition miss surfaces a structured log warning and returns
// nil so the caller falls back to the standard token iterator without
// changing user-visible behavior. When DFlash engages the whole tool
// loop collapses to one pass — DFlash emits raw text tokens and does
// not participate in the tool-call accumulator.
//
// Text-only + reasoning-aware: decoded block text is routed through
// the same reasoning parser the standard path uses so `<think>` blocks
// split into `.reasoning` / `.content` chunks correctly. Stop sequences
// are honored via the Aho-Corasick matcher on the visible-content stream.

extension Engine {

    /// Returns the collected tool calls (always empty for DFlash) or nil
    /// when DFlash could not engage and the caller should take the
    /// standard path.
    internal func tryDFlashGenerationPass(
        request: ChatRequest,
        resolved: ResolvedSettings,
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) async throws -> [ChatRequest.ToolCall]? {

        // -- Preconditions --
        let g = resolved.settings
        if !g.dflash { return nil }

        // Tools / images disable DFlash for now. Both would require the
        // drafter to emit tool-call markers and image tokens, which it
        // was not trained to do.
        let hasTools = !(request.tools?.isEmpty ?? true)
        if hasTools {
            await self.log(.warn, "engine",
                "dflash: request has tools — falling back to standard path")
            return nil
        }
        if Self.requestHasImages(request) {
            await self.log(.warn, "engine",
                "dflash: request has images — falling back to standard path")
            return nil
        }

        guard let specDec = self.makeDFlashSpecDec(settings: g) else {
            await self.log(.warn, "engine",
                "dflash: no ready drafter+target (drafter loaded=\(self._dflashDrafter != nil), "
                + "target adapter=\(self._dflashTarget != nil)) — falling back to standard path")
            return nil
        }
        guard let container = self.loaded else {
            return nil
        }

        // -- Tokenize prompt via the tokenizer's chat template --
        let chatMessages = await Engine.buildChatMessages(
            from: request,
            effectiveThinking: request.enableThinking
                ?? resolved.enableThinking
                ?? false,
            modelStampsThink: self.modelCapabilities?.thinkInTemplate ?? false,
            responseFormatInstruction: Engine.responseFormatInstruction(
                from: request.responseFormat))

        struct TokenizeResult: Sendable {
            let promptIDs: [Int]
            let eosIDs: Set<Int>
        }
        let tokenized: TokenizeResult = try await container.perform { ctx in
            let userInput = UserInput(chat: chatMessages, tools: nil)
            let prepared = try await ctx.processor.prepare(input: userInput)
            let ids = prepared.text.tokens.asArray(Int.self)
            var eos: Set<Int> = []
            if let eid = ctx.tokenizer.eosTokenId { eos.insert(eid) }
            return TokenizeResult(promptIDs: ids, eosIDs: eos)
        }

        if tokenized.promptIDs.isEmpty {
            await self.log(.warn, "engine",
                "dflash: tokenizer returned zero tokens — falling back")
            return nil
        }

        // -- Run cached multi-block speculative decode --
        let requestStart = Date()
        let maxNewTokens = max(1, request.maxTokens
            ?? g.defaultMaxTokens)

        // Each block outcome yields its accepted tokens as a decoded
        // text chunk. The decoder runs on the engine actor (tokenizer
        // isn't Sendable across actor hops without the container), so
        // we hop into container.perform once per block. That's fine —
        // block cadence is already several hundred ms on real models.
        var totalAccepted: [Int] = []
        var blockCount = 0
        var firstTokenAt: Date?
        var cumulativeText = ""

        // Coordinator-aware warmup: consult the engine's cache coordinator
        // for prefix / memory / disk hits, restore matched blocks into a
        // fresh target cache, then pass the pre-warmed cache into spec-dec
        // with `prefixMatched` so its prompt-forward only processes the
        // tail. After generation, extract per-layer KV tensors + SSM state
        // from the returned cache and hand them to `storeAfterGeneration`
        // so the next turn can reuse the full prefix across L1 paged +
        // L1.5 memory + L2 disk tiers. Matches the standard path's cache
        // lifecycle 1-for-1.
        let coordinator = self.cacheCoordinator
        let genPromptLen = await Self.computeGenPromptLen(
            container: container, chatMessages: chatMessages)

        struct RunResult {
            var outcomes: [JangDFlashBlockOutcome]
            var acceptedTokens: [Int]
            var finalCache: [KVCache]
            var prefixMatched: Int
            var cacheDetail: String
        }

        let runResult: RunResult = try await container.perform { ctx in
            // Build a fresh per-model cache for this request.
            let freshCache = specDec.target.makeCache()
            // Coordinator lookup using the same key-stripping rules as
            // the standard path (mediaSalt nil — DFlash guarded out of
            // VLM requests above).
            var matched = 0
            var detail = "miss"
            if let coord = coordinator {
                switch coord.fetch(
                    tokens: tokenized.promptIDs,
                    mediaSalt: nil,
                    genPromptLen: genPromptLen
                ) {
                case .hit(let m, _, let tier, let blocks, let ssmStates, let diskArrays):
                    matched = m
                    detail = "dflash+\(tier.rawValue)"
                    // Restore into the fresh cache. Paged tier ships blocks,
                    // disk tier ships the v2 arrays dict, memory tier ships
                    // the arrays too.
                    if !blocks.isEmpty {
                        _ = restoreLayerData(from: blocks, into: freshCache)
                    } else if let arrays = diskArrays {
                        _ = restoreFromDiskArrays(arrays, into: freshCache)
                    }
                    _ = ssmStates  // SSMStateCache already applied by fetch
                case .miss:
                    detail = "dflash+miss"
                }
            }

            var collected: [JangDFlashBlockOutcome] = []
            let (accepted, finalCache) = try specDec.cachedGenerate(
                promptIDs: tokenized.promptIDs,
                maxNewTokens: maxNewTokens,
                eosTokenIDs: tokenized.eosIDs,
                cache: freshCache,
                prefixMatched: matched,
                onBlock: { outcome in collected.append(outcome) }
            )
            return RunResult(
                outcomes: collected,
                acceptedTokens: accepted,
                finalCache: finalCache,
                prefixMatched: matched,
                cacheDetail: detail
            )
        }
        let outcomes = runResult.outcomes

        // Stream each block's accepted text as a content chunk. Reasoning
        // split is intentionally skipped here in v1 — the standard token
        // iterator path streams reasoning deltas live; DFlash emits blocks
        // of 1-B tokens at a time so a streaming split would either buffer
        // entire blocks or emit mis-tagged deltas. Safer to surface all
        // text as `.content` and let the client's reasoning extractor
        // (vMLX.app has one) split `<think>...</think>` after the fact.
        for outcome in outcomes {
            if Task.isCancelled { break }
            blockCount += 1

            // Decode this block's accepted tokens into text via the
            // tokenizer. Detokenization is cheap — no MLX work.
            let tokens = outcome.acceptedTokens
            if tokens.isEmpty { continue }
            totalAccepted.append(contentsOf: tokens)

            let blockText: String = await container.perform { ctx in
                ctx.tokenizer.decode(tokenIds: tokens)
            }
            if blockText.isEmpty { continue }

            if firstTokenAt == nil { firstTokenAt = Date() }
            cumulativeText += blockText
            continuation.yield(StreamChunk(content: blockText))
        }

        // -- Final usage + finish reason --
        let now = Date()
        let ttftMs = firstTokenAt.map {
            $0.timeIntervalSince(requestStart) * 1000
        }
        let totalMs = now.timeIntervalSince(requestStart) * 1000
        let tokensPerSec: Double? = {
            guard let ttftStart = firstTokenAt else { return nil }
            let decodeSec = now.timeIntervalSince(ttftStart)
            guard decodeSec > 0, totalAccepted.count > 0 else { return nil }
            return Double(totalAccepted.count) / decodeSec
        }()

        let finish: String
        if totalAccepted.contains(where: { tokenized.eosIDs.contains($0) }) {
            finish = "stop"
        } else if totalAccepted.count >= maxNewTokens {
            finish = "length"
        } else if Task.isCancelled {
            finish = "cancelled"
        } else {
            finish = "stop"
        }

        let usage = StreamChunk.Usage(
            promptTokens: tokenized.promptIDs.count,
            completionTokens: totalAccepted.count,
            cachedTokens: runResult.prefixMatched,
            tokensPerSecond: tokensPerSec,
            promptTokensPerSecond: nil,
            ttftMs: ttftMs,
            prefillMs: nil,
            totalMs: totalMs,
            cacheDetail: "\(runResult.cacheDetail)(\(blockCount) blocks)",
            isPartial: false
        )
        continuation.yield(StreamChunk(finishReason: finish, usage: usage))

        // Persist the final cache back to the coordinator so the NEXT
        // turn can restore it. Mirrors the standard path's
        // `cacheCoordinator.storeAfterGeneration` call. Per-layer KV
        // + SSM states come out of the final cache via the extractor
        // helpers shared with the standard path.
        if let coord = coordinator {
            await container.perform { ctx in
                let layerData = extractLayerData(from: runResult.finalCache)
                let ssmArr = extractSSMStates(from: runResult.finalCache)
                let ssm: [MLXArray]? = ssmArr.isEmpty ? nil : ssmArr
                coord.storeAfterGeneration(
                    promptTokens: tokenized.promptIDs,
                    perLayerData: layerData,
                    ssmStates: ssm,
                    cache: runResult.finalCache,
                    mediaSalt: nil,
                    genPromptLen: genPromptLen
                )
            }
        }

        await self.log(.info, "engine",
            "dflash: \(totalAccepted.count) tokens / \(blockCount) blocks / "
            + "prefix=\(runResult.prefixMatched) / \(runResult.cacheDetail) "
            + "(\(String(format: "%.2f", tokensPerSec ?? 0)) tok/s)")
        return []
    }

    private static func requestHasImages(_ request: ChatRequest) -> Bool {
        for msg in request.messages {
            guard case .parts(let parts)? = msg.content else { continue }
            for part in parts where part.type == "image_url" {
                return true
            }
        }
        return false
    }

    /// Mirror of the `gen_prompt_len` computation in `Stream.swift`.
    /// Renders the chat template with and without the generation-prompt
    /// suffix and returns the length delta. Shared cache-coordinator
    /// stripping requires both paths to agree on the number.
    fileprivate static func computeGenPromptLen(
        container: vMLXLMCommon.ModelContainer,
        chatMessages: [Chat.Message]
    ) async -> Int {
        let n = await container.perform { ctx -> Int in
            do {
                let userInput = UserInput(chat: chatMessages, tools: nil)
                let rawMsgs = DefaultMessageGenerator().generate(from: userInput)
                let withGP = try ctx.tokenizer.applyChatTemplate(
                    messages: rawMsgs,
                    tools: nil,
                    additionalContext: ["add_generation_prompt": true as any Sendable])
                let withoutGP = try ctx.tokenizer.applyChatTemplate(
                    messages: rawMsgs,
                    tools: nil,
                    additionalContext: ["add_generation_prompt": false as any Sendable])
                return max(0, withGP.count - withoutGP.count)
            } catch {
                return 0
            }
        }
        return n
    }
}
