// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import os

// MARK: - BatchEngine

/// Continuous batching inference engine for mlx-swift-lm.
///
/// `BatchEngine` processes multiple generation requests simultaneously by batching
/// their decode steps through a single model forward pass. This provides significantly
/// higher throughput than serial single-sequence generation when serving multiple
/// concurrent requests.
///
/// ## Architecture
///
/// The engine follows the continuous batching pattern used by production inference
/// servers (vLLM, TGI):
///
/// 1. **Request submission** — Callers submit requests via ``submit(input:parameters:)``
///    and receive an `AsyncStream<BatchGeneration>` that yields tokens as they are generated.
///
/// 2. **Scheduling loop** — A background task runs the engine loop:
///    - Admits pending requests from the wait queue into active slots
///    - Processes prefill chunks for newly admitted requests (one chunk per iteration)
///    - Batches all decode-phase slots into a single `[B, 1]` forward pass
///    - Samples tokens independently per sequence using each request's own parameters
///    - Detects completion (EOS, max tokens) and cleans up finished slots
///
/// 3. **Cache management** — Each sequence owns its own `[KVCache]` array (B=1).
///    During batched decode, per-layer ``BatchKVCache`` wrappers present these as
///    a single `[B, H, L, D]` cache to the model.
///
/// ## Usage
///
/// ```swift
/// // Load model normally
/// let modelContext = try await ModelFactory.shared.load(...)
///
/// // Create engine — uses existing GenerateParameters per-request
/// let engine = BatchEngine(context: modelContext, maxBatchSize: 8)
///
/// // Submit requests (from different async contexts, e.g., HTTP handlers)
/// let stream = await engine.submit(input: lmInput, parameters: generateParams)
/// for await event in stream {
///     switch event {
///     case .token(let id):
///         // Feed to NaiveStreamingDetokenizer
///         detokenizer.append(token: id)
///     case .info(let completionInfo):
///         print(completionInfo.summary())
///     }
/// }
/// ```
///
/// ## Thread Safety
///
/// `BatchEngine` is an `actor` — all state is automatically isolated. The model
/// is only accessed from the engine's scheduling loop, ensuring single-threaded
/// model access without explicit locking.
///
/// ## Compatibility
///
/// - All input parameters come from the existing ``GenerateParameters`` struct.
///   No new configuration types are forced on callers.
/// - The engine uses the model's `callAsFunction` and `newCache` methods directly.
///   No model code changes are required.
/// - Existing single-sequence ``TokenIterator`` and ``generate()`` APIs are unaffected.
///
/// ## Extensibility
///
/// The slot cache type is `[KVCache]` (protocol-typed). Future cache implementations
/// (TurboQuant, paged caches, hybrid SSM) can be used as slot caches without changing
/// the engine core.
public actor BatchEngine {

    // MARK: - Configuration

    /// Maximum number of sequences decoded simultaneously in one batch.
    /// Additional requests are queued until a slot opens.
    public let maxBatchSize: Int

    /// Number of iterations between GPU memory cache purges.
    /// Matches the 256-token interval used by ``TokenIterator``.
    public let memoryPurgeInterval: Int

    // MARK: - State

    /// The loaded model context (model, tokenizer, config, processor).
    private let context: ModelContext

    /// Optional cache coordinator for multi-tier KV caching.
    /// When present, the engine will attempt to fetch cached state before prefill
    /// and store cache state after generation completes.
    private let cacheCoordinator: CacheCoordinator?

    /// Logger for cache-related diagnostics.
    private static let logger = Logger(subsystem: "vmlx", category: "BatchEngine")

    /// Set of token IDs that signal end of generation for this model.
    private let stopTokenIDs: Set<Int>

    /// Requests waiting to be admitted into active slots.
    private var waitQueue: [BatchPendingRequest] = []

    /// Active generation slots (max `maxBatchSize`).
    private var activeSlots: [BatchSlot] = []

    /// Background scheduling loop task handle.
    private var loopTask: Task<Void, Never>?

    /// Total decode steps since last memory purge.
    private var stepsSinceMemoryPurge: Int = 0

    // MARK: - Initialization

    /// Create a new continuous batching engine.
    ///
    /// - Parameters:
    ///   - context: The loaded model context from ``ModelFactory``.
    ///   - maxBatchSize: Maximum concurrent sequences. Defaults to 8.
    ///     Higher values increase throughput but use more memory.
    ///   - memoryPurgeInterval: Steps between GPU memory cache purges. Defaults to 256.
    ///   - cacheCoordinator: Optional multi-tier cache coordinator. When provided,
    ///     the engine will attempt cache lookups before prefill and store cache state
    ///     after generation completes. Defaults to nil.
    public init(
        context: ModelContext,
        maxBatchSize: Int = 8,
        memoryPurgeInterval: Int = 256,
        cacheCoordinator: CacheCoordinator? = nil
    ) {
        self.context = context
        self.maxBatchSize = maxBatchSize
        self.memoryPurgeInterval = memoryPurgeInterval
        self.cacheCoordinator = cacheCoordinator

        // Build stop token set from model config + tokenizer.
        // Matches the logic in Evaluate.swift's buildStopTokenIds plus unknownTokenId.
        var stops = context.configuration.eosTokenIds
        if let tokenizerEOS = context.tokenizer.eosTokenId {
            stops.insert(tokenizerEOS)
        }
        if let unknownID = context.tokenizer.unknownTokenId {
            stops.insert(unknownID)
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stops.insert(id)
            }
        }
        self.stopTokenIDs = stops
    }

    // MARK: - Public API

    /// Submit a generation request, returning raw token events.
    ///
    /// This is the low-level API. For text output, use ``generate(input:parameters:)``
    /// which handles detokenization automatically.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (from `UserInputProcessor.prepare()`).
    ///   - parameters: Generation parameters for this request.
    /// - Returns: A tuple of `(requestID, stream)`. The stream yields token IDs
    ///   and completion info. Use the ID with ``cancel(_:)`` to stop early.
    @discardableResult
    public func submit(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        genPromptLen: Int = 0
    ) -> (id: BatchRequestID, stream: AsyncStream<BatchGeneration>) {
        let (stream, continuation) = AsyncStream<BatchGeneration>.makeStream()
        let request = BatchPendingRequest(
            input: input,
            parameters: parameters,
            continuation: continuation,
            genPromptLen: genPromptLen
        )
        waitQueue.append(request)
        ensureLoopRunning()
        return (request.id, stream)
    }

    /// Generate text from prepared input — drop-in replacement for `ModelContainer.generate()`.
    ///
    /// Returns the same `AsyncStream<Generation>` type as the existing single-sequence
    /// API, with `.chunk(String)` for decoded text and `.info(GenerateCompletionInfo)`
    /// for completion metrics. Handles detokenization internally.
    ///
    /// ## Example
    /// ```swift
    /// let engine = BatchEngine(context: modelContext)
    /// let input = try await modelContext.processor.prepare(input: userInput)
    /// let stream = await engine.generate(input: input, parameters: params)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text, terminator: "")
    ///     case .info(let info): print("\n\(info.summary())")
    ///     case .toolCall: break
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared model input.
    ///   - parameters: Generation parameters for this request.
    /// - Returns: An `AsyncStream<Generation>` yielding text chunks and completion info.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) -> AsyncStream<Generation> {
        let tokenizer = context.tokenizer
        let (_, tokenStream) = submit(input: input, parameters: parameters)

        return AsyncStream<Generation> { continuation in
            Task {
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                for await event in tokenStream {
                    switch event {
                    case .token(let id):
                        detokenizer.append(token: id)
                        while let text = detokenizer.next() {
                            continuation.yield(.chunk(text))
                        }
                    case .info(let info):
                        // Flush remaining text from detokenizer
                        detokenizer.startNewSegment()
                        continuation.yield(.info(info))
                    }
                }
                continuation.finish()
            }
        }
    }

    /// Cancel a specific request by ID.
    ///
    /// If the request is still in the wait queue, it is removed immediately.
    /// If it is actively generating, it is marked as finished and its stream
    /// is closed with a `.cancelled` stop reason.
    ///
    /// - Parameter id: The request ID returned by ``submit(input:parameters:)``.
    public func cancel(_ id: BatchRequestID) {
        // Check wait queue first
        if let idx = waitQueue.firstIndex(where: { $0.id == id }) {
            let request = waitQueue.remove(at: idx)
            request.continuation.yield(.info(GenerateCompletionInfo(
                promptTokenCount: request.input.text.tokens.size,
                generationTokenCount: 0,
                promptTime: 0,
                generationTime: 0,
                stopReason: .cancelled
            )))
            request.continuation.finish()
            return
        }

        // Check active slots
        if let idx = activeSlots.firstIndex(where: { $0.id == id }) {
            var slot = activeSlots[idx]
            finishSlot(slot, reason: .cancelled)
            slot.isFinished = true
            activeSlots[idx] = slot
        }
    }

    /// Shut down the engine, finishing all active streams.
    ///
    /// Pending requests receive a `.info` with `.cancelled` stop reason.
    /// Active slots are allowed to complete their current step before finishing.
    public func shutdown() {
        loopTask?.cancel()
        loopTask = nil

        // Finish all pending requests
        for request in waitQueue {
            request.continuation.yield(.info(GenerateCompletionInfo(
                promptTokenCount: request.input.text.tokens.size,
                generationTokenCount: 0,
                promptTime: 0,
                generationTime: 0,
                stopReason: .cancelled
            )))
            request.continuation.finish()
        }
        waitQueue.removeAll()

        // Finish all active slots
        for slot in activeSlots {
            finishSlot(slot, reason: .cancelled)
        }
        activeSlots.removeAll()
    }

    /// The number of requests currently waiting in the queue.
    public var pendingCount: Int { waitQueue.count }

    /// The number of sequences currently being generated.
    public var activeCount: Int { activeSlots.count }

    /// Whether the engine is currently running (has active or pending work).
    public var isRunning: Bool { loopTask != nil }

    // MARK: - Scheduling Loop

    /// Start the background scheduling loop if not already running.
    private func ensureLoopRunning() {
        guard loopTask == nil else { return }
        loopTask = Task {
            await self.schedulingLoop()
        }
    }

    /// Main scheduling loop. Runs until all work is complete.
    private func schedulingLoop() async {
        while !Task.isCancelled {
            // Exit when no work remains
            if waitQueue.isEmpty && activeSlots.isEmpty {
                break
            }

            // 1. Admit new requests from wait queue
            admitPendingRequests()

            // 2. Run one scheduling step
            step()

            // 3. Remove finished slots
            activeSlots.removeAll { $0.isFinished }

            // 4. Periodic memory cleanup
            stepsSinceMemoryPurge += 1
            if stepsSinceMemoryPurge >= memoryPurgeInterval {
                Memory.clearCache()
                stepsSinceMemoryPurge = 0
            }

            // 5. Yield to allow submit() calls and stream consumers to run
            await Task.yield()
        }

        loopTask = nil
    }

    // MARK: - Admission

    /// Move requests from the wait queue into active slots up to `maxBatchSize`.
    private func admitPendingRequests() {
        while activeSlots.count < maxBatchSize && !waitQueue.isEmpty {
            let request = waitQueue.removeFirst()
            let cache = context.model.newCache(parameters: request.parameters)
            let slot = BatchSlot(from: request, cache: cache, stopTokenIDs: stopTokenIDs)
            activeSlots.append(slot)
        }
    }

    // MARK: - Step Logic

    /// Run one scheduling step: prefill pending slots, then batch-decode active slots.
    private func step() {
        // Phase 1: Process one prefill chunk per slot that's still prefilling.
        // Prefill is done sequentially per slot (each chunk is large, batching
        // prefill chunks of different lengths wastes compute on padding).
        for i in activeSlots.indices where activeSlots[i].phase == .prefill {
            stepPrefill(slotIndex: i)
        }

        // Phase 2: Batch-decode all slots that are in decode phase.
        let decodeIndices = activeSlots.indices.filter { activeSlots[$0].phase == .decode }
        if !decodeIndices.isEmpty {
            stepBatchDecode(slotIndices: decodeIndices)
        }
    }

    // MARK: - Prefill

    /// Run the full prefill for a slot using the model's `prepare()` method.
    ///
    /// This delegates to `model.prepare()` which handles:
    /// - **LLM models**: Chunked prefill of the prompt in `prefillStepSize` chunks
    /// - **VLM models**: Vision tower processing, `maskedScatter` of image embeddings,
    ///   and full prompt processing including multimodal fusion
    ///
    /// After prefill, samples the first decode token and transitions the slot to `.decode`.
    private func stepPrefill(slotIndex: Int) {
        var slot = activeSlots[slotIndex]

        // Check multi-tier cache for a prefix match before running full prefill.
        // On cache hit, restore KV state and only prefill remaining tokens.
        var inputForPrepare = slot.originalInput
        let hasRotatingCache = slot.cache.contains { $0 is RotatingKVCache }
        // VLM inputs are now supported via `mediaSalt` (pixel fingerprint
        // mixed into the cache-coordinator key) so "same text + same image"
        // hits while "same text + different image" misses. RotatingKVCache
        // is still skipped because its sliding-window semantics are
        // incompatible with partial restore. Port of reference `ed989f2`.
        if let coordinator = cacheCoordinator, !hasRotatingCache {
            let tokenIds = slot.originalInput.text.tokens.asArray(Int.self)
            let mediaSalt = computeMediaSalt(for: slot.originalInput)
            let result = coordinator.fetch(
                tokens: tokenIds,
                mediaSalt: mediaSalt,
                genPromptLen: slot.genPromptLen)
            if case .hit(_, let remaining, let detail, let blocks, let ssmStates, let diskArrays) = result {
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: slot.cache)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: slot.cache)
                        }
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit for slot \(slot.id): restored \(restoredTokens) tokens, prefilling \(remaining.count) remaining"
                        )
                    }
                }

                // Disk cache restore (blocks are empty, arrays are present).
                //
                // v2 disk format includes per-layer `.mamba` entries —
                // `restoreFromDiskArrays` writes them directly into MambaCache
                // via `restoreMambaLayer`. Calling `restoreSSMStates(ssm,...)`
                // here would double-write and crash on hybrid VL JANG.
                // See Evaluate.swift for the full root-cause comment.
                if let diskArrays, !restored {
                    let diskRestored = restoreFromDiskArrays(diskArrays, into: slot.cache)
                    if diskRestored > 0 {
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit for slot \(slot.id): restored \(diskRestored) tokens from disk, prefilling \(remaining.count) remaining"
                        )
                    }
                }
                _ = ssmStates  // silence unused — paged branch above uses it

                if restored {
                    if remaining.isEmpty {
                        // Full cache hit — feed last token to seed decode.
                        // Shape MUST be 2D `[1, 1]` not 1D `[1]` because VL
                        // model `prepare(_:cache:)` paths call
                        // `inputs.dim(1)` for sequence length and a 1D input
                        // fatal-traps with `SmallVector out of range`. See
                        // Evaluate.swift for the full root-cause comment.
                        // Construct via 1D + .newAxis to dodge a Swift type
                        // inference ambiguity around the nested Int32 literal.
                        // Same fix applied in Evaluate.swift's twin branch.
                        let lastToken = MLXArray([Int32(tokenIds.last!)])[.newAxis]
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: lastToken),
                            image: nil, video: nil)
                    } else {
                        let remainingArray = MLXArray(remaining.map { Int32($0) })[.newAxis]
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: remainingArray),
                            image: nil, video: nil)
                    }
                }
            }
        }

        // Prefill: either full input (cache miss) or remaining tokens (cache hit).
        let prepareResult: PrepareResult
        do {
            prepareResult = try context.model.prepare(
                inputForPrepare, cache: slot.cache, windowSize: slot.prefillStepSize)
        } catch {
            // Prefill failed (e.g., invalid input) — finish with cancellation
            finishSlot(slot, reason: .cancelled)
            slot.isFinished = true
            activeSlots[slotIndex] = slot
            return
        }

        // Seed the processor with the full prompt tokens.
        let promptTokens = slot.originalInput.text.tokens
        slot.processor?.prompt(promptTokens)

        // Extract the first generated token from the prepare result
        let firstToken: MLXArray
        switch prepareResult {
        case .tokens(let remainingText):
            // LLM path: prepare() consumed all but the last chunk, returned remaining tokens.
            // Run the last chunk through the model to get logits for the first decode token.
            let result = context.model(
                remainingText[text: .newAxis], cache: slot.cache, state: nil)
            MLX.eval(slot.cache)
            let logits = result.logits[0 ..< 1, -1, 0...]
            firstToken = slot.sampleToken(from: logits)

        case .logits(let result):
            // VLM path: prepare() already ran the full prompt and returned logits directly.
            let logits = result.logits[0 ..< 1, -1, 0...]
            firstToken = slot.sampleToken(from: logits)
        }

        let tokenID = firstToken.item(Int.self)

        slot.phase = .decode
        slot.decodeStartTime = Date()
        slot.pendingTokens = MLXArray([Int32]()) // clear

        // Apply TurboQuant / affine KV compression now that prefill is done.
        // This is the BatchEngine analog of the `maybeQuantizeKVCache` calls
        // in `Evaluate.swift`. Without it, the Mac default batched path
        // silently ignores `enableTurboQuant` and runs on plain KVCacheSimple.
        // `maybeQuantizeKVCache` is a no-op for Rotating/Mamba/CacheList
        // layers and for caches whose offset is below the TQ floor, so it's
        // safe for hybrid SSM and sliding-window VL models.
        maybeQuantizeKVCache(
            cache: &slot.cache,
            kvBits: slot.kvBits,
            kvMode: slot.kvMode
        )

        // Check EOS on first generated token before yielding
        if stopTokenIDs.contains(tokenID) {
            finishSlot(slot, reason: .stop)
            slot.isFinished = true
        } else {
            slot.continuation.yield(.token(tokenID))
            slot.generatedTokenCount += 1
            slot.nextToken = firstToken

            if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
                finishSlot(slot, reason: .length)
                slot.isFinished = true
            }
        }

        activeSlots[slotIndex] = slot
    }

    // MARK: - Batched Decode

    /// Run one batched decode step across all decode-phase slots.
    ///
    /// Constructs `[B, 1]` input from each slot's next token, builds per-layer
    /// ``BatchKVCache`` wrappers, runs one model forward pass, then samples
    /// independently per sequence.
    private func stepBatchDecode(slotIndices: [Int]) {
        let B = slotIndices.count

        // Build batched input: [B, 1]
        let tokenArrays = slotIndices.map { activeSlots[$0].nextToken! }
        let batchTokens = stacked(tokenArrays).reshaped(B, 1)

        // Build per-layer batched cache wrappers.
        // Each cache type gets its appropriate batch wrapper:
        // - KVCacheSimple/RotatingKVCache → BatchKVCache (split/pad/stack)
        // - ArraysCache/MambaCache → BatchArraysCache (merge along batch dim)
        // - CacheList → BatchCacheList (wraps each sub-cache appropriately)
        let numLayers = activeSlots[slotIndices[0]].cache.count
        var layerCaches = [KVCache]()
        var batchArraysCaches = [BatchArraysCache]()  // track for splitBack
        var batchCacheLists = [BatchCacheList]()       // track for splitBack
        layerCaches.reserveCapacity(numLayers)

        for layer in 0 ..< numLayers {
            let slotCachesForLayer = slotIndices.map { activeSlots[$0].cache[layer] }
            let representative = slotCachesForLayer[0]

            if let _ = representative as? CacheList {
                let cacheLists = slotCachesForLayer.map { $0 as! CacheList }
                let batchCL = BatchCacheList(slotCacheLists: cacheLists)
                layerCaches.append(batchCL)
                batchCacheLists.append(batchCL)
            } else if let _ = representative as? ArraysCache {
                let arraysCaches = slotCachesForLayer.map { $0 as! ArraysCache }
                let batchAC = BatchArraysCache(slotCaches: arraysCaches)
                layerCaches.append(batchAC)
                batchArraysCaches.append(batchAC)
            } else {
                layerCaches.append(BatchKVCache(slotCaches: slotCachesForLayer))
            }
        }

        // Run batched forward pass
        let result = context.model(
            LMInput.Text(tokens: batchTokens),
            cache: layerCaches,
            state: nil
        )
        // result.logits shape: [B, 1, vocabSize]

        // Synchronize — we need to read token values for EOS checking
        MLX.eval(result.logits)

        // Split SSM states back to per-sequence caches
        for batchAC in batchArraysCaches {
            batchAC.splitBack()
        }
        for batchCL in batchCacheLists {
            batchCL.splitBack()
        }

        // Sample per sequence and route results
        for (batchIdx, slotIdx) in slotIndices.enumerated() {
            var slot = activeSlots[slotIdx]

            // Extract this sequence's logits as [1, vocabSize] (2D) for processor compatibility.
            let logits = result.logits[batchIdx ..< batchIdx + 1, 0, 0...]
            let token = slot.sampleToken(from: logits)
            let tokenID = token.item(Int.self)

            // Check stop conditions BEFORE yielding — don't emit EOS tokens to callers.
            // This matches TokenIterator behavior where the stop token is never surfaced.
            if stopTokenIDs.contains(tokenID) {
                finishSlot(slot, reason: .stop)
                slot.isFinished = true
            } else {
                slot.continuation.yield(.token(tokenID))
                slot.generatedTokenCount += 1
                slot.nextToken = token

                if let maxTokens = slot.maxTokens, slot.generatedTokenCount >= maxTokens {
                    finishSlot(slot, reason: .length)
                    slot.isFinished = true
                }
            }

            activeSlots[slotIdx] = slot
        }
    }

    // MARK: - Completion

    /// Finish a slot by yielding completion info and closing its stream.
    ///
    /// When a cache coordinator is present and the slot completed normally
    /// (not cancelled), stores the prompt tokens for future cache reuse.
    private func finishSlot(_ slot: BatchSlot, reason: GenerateStopReason) {
        let now = Date()
        let prefillTime = (slot.decodeStartTime ?? now).timeIntervalSince(slot.prefillStartTime)
        let decodeTime = slot.decodeStartTime.map { now.timeIntervalSince($0) } ?? 0

        // Store prompt cache state for completed (non-cancelled) generations.
        // Extract real KV data and SSM states from the slot's cache.
        //
        // GUARD: skip the store entirely when the slot has any RotatingKVCache
        // layers. Rotating caches (sliding-window attention) compress to
        // window_size and the FETCH path already refuses them at
        // BatchEngine.swift:396. Storing them is pure waste — extractLayerData
        // walks every layer (KV + SSM), serializes 100s of MB to disk, and
        // every subsequent fetch silently throws it all away. On long-context
        // hybrid models this caused both the steady-state cratering and the
        // rolling pool-pressure that bled into adjacent slots. Symmetric with
        // the fetch guard — fix #1 from the user's batch.
        let hasRotatingCache = slot.cache.contains { $0 is RotatingKVCache }
        if reason != .cancelled, !hasRotatingCache, let coordinator = cacheCoordinator {
            let promptTokens = slot.originalInput.text.tokens.asArray(Int.self)
            let perLayerData = extractLayerData(from: slot.cache)
            let ssmStates: [MLXArray]? = coordinator.isHybrid
                ? extractSSMStates(from: slot.cache) : nil
            // Compute and pass `mediaSalt` so VLM requests don't poison
            // the text-only cache key bucket. Evaluate.swift's
            // `generateLoopTask` twin path already does this via
            // `iterator.mediaSalt`; BatchEngine was silently defaulting
            // to `nil` and cross-hitting text-only entries on image
            // inputs. Cache deep audit 2026-04-14 finding #3.
            let mediaSalt = computeMediaSalt(for: slot.originalInput)
            coordinator.storeAfterGeneration(
                promptTokens: promptTokens,
                perLayerData: perLayerData,
                ssmStates: ssmStates,
                cache: slot.cache,
                mediaSalt: mediaSalt,
                genPromptLen: slot.genPromptLen
            )
            Self.logger.debug(
                "Stored cache entry for slot \(slot.id): \(promptTokens.count) prompt tokens"
            )
        }

        // Long-context pool-pressure relief. The global memoryPurgeInterval
        // (256 decode steps) is too coarse when a single long request
        // allocates several GB of activations before releasing the pool
        // back to the allocator — without a boundary purge, that pressure
        // bleeds into the next slot and craters its decode throughput.
        //
        // Short chats skip the extra C call; only requests whose prompt
        // was big enough to actually pressure the pool trigger it. Port
        // of reference `ed989f2` (4096-token threshold; previously
        // unconditional here which cost ~100µs per short chat request
        // for no benefit).
        let longContextPurgeThreshold = 4096
        if slot.promptTokenCount >= longContextPurgeThreshold {
            MLX.Memory.clearCache()
            stepsSinceMemoryPurge = 0
        }

        slot.continuation.yield(.info(GenerateCompletionInfo(
            promptTokenCount: slot.promptTokenCount,
            generationTokenCount: slot.generatedTokenCount,
            promptTime: prefillTime,
            generationTime: decodeTime,
            stopReason: reason
        )))
        slot.continuation.finish()
    }
}
