// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

// MARK: - Slot Phase

/// The generation phase of an active slot in the batch engine.
enum SlotPhase {
    /// Still processing the prompt in chunks. The slot is not yet part of the
    /// batched decode step.
    case prefill

    /// Prompt fully processed, generating tokens one at a time as part of
    /// the batched decode step.
    case decode
}

// MARK: - Active Slot

/// An active generation slot managed by ``BatchEngine``.
///
/// Each slot represents one in-flight request. It owns its KV cache, sampler,
/// processor, and token history. Slots transition from `.prefill` → `.decode`
/// and eventually become finished when they hit a stop token or max token limit.
struct BatchSlot {
    /// Unique identifier matching the original request.
    let id: BatchRequestID

    /// Continuation for streaming tokens back to the caller.
    let continuation: AsyncStream<BatchGeneration>.Continuation

    /// Per-request sampler (temperature, topP, etc. from the request's `GenerateParameters`).
    let sampler: LogitSampler

    /// Per-request logit processor (repetition penalty, etc.).
    var processor: LogitProcessor?

    /// Set of token IDs that signal end of generation.
    let stopTokenIDs: Set<Int>

    /// Maximum tokens to generate for this request. `nil` = unlimited.
    let maxTokens: Int?

    /// Per-layer KV caches for this sequence (B=1 each).
    ///
    /// Mutable so that `maybeQuantizeKVCache` can swap `KVCacheSimple` layers
    /// for `TurboQuantKVCache` (or `QuantizedKVCache`) after prefill — this
    /// is how the default Mac batched path actually honors
    /// `enableTurboQuant`.
    var cache: [KVCache]

    /// Legacy affine KV quantization bits (nil when not set).
    let kvBits: Int?

    /// KV compression mode (e.g. `.turboQuant(keyBits:valueBits:)`).
    let kvMode: KVQuantizationMode

    /// The original full input, preserved for VLM `prepare()` which needs image data.
    /// Also used for seeding the logit processor with the full prompt tokens.
    let originalInput: LMInput

    /// Remaining prompt tokens to process during prefill.
    var pendingTokens: MLXArray

    /// The next token to feed into the model (last sampled token during decode).
    var nextToken: MLXArray?

    /// Current generation phase.
    var phase: SlotPhase = .prefill

    /// Number of tokens generated so far (decode phase only).
    var generatedTokenCount: Int = 0

    /// Number of prompt tokens (for completion info).
    let promptTokenCount: Int

    /// Timestamp when prefill started (for metrics).
    let prefillStartTime: Date

    /// Timestamp when decode started (for metrics).
    var decodeStartTime: Date?

    /// Whether this slot has finished generating.
    var isFinished: Bool = false

    /// Prefill step size for this request.
    let prefillStepSize: Int

    /// Chat-template generation-prompt suffix length, inherited from
    /// the pending request. See `CacheCoordinator.fetch(genPromptLen:)`.
    let genPromptLen: Int

    // MARK: - Sampling

    /// Sample a token from logits, applying processor and sampler.
    ///
    /// Returns the sampled token as an `MLXArray` scalar.
    mutating func sampleToken(from logits: MLXArray) -> MLXArray {
        var logits = logits
        if var proc = processor {
            logits = proc.process(logits: logits)
            let token = sampler.sample(logits: logits)
            proc.didSample(token: token)
            self.processor = proc
            return token
        }
        return sampler.sample(logits: logits)
    }
}

// MARK: - Slot Construction

extension BatchSlot {
    /// Create a slot from a pending request.
    ///
    /// - Parameters:
    ///   - request: The pending request to activate.
    ///   - cache: Per-layer KV caches allocated by the model's `newCache()`.
    ///   - stopTokenIDs: Set of EOS token IDs for this model.
    init(from request: BatchPendingRequest, cache: [KVCache], stopTokenIDs: Set<Int>) {
        self.id = request.id
        self.continuation = request.continuation
        self.sampler = request.parameters.sampler()
        self.processor = request.parameters.processor()
        self.stopTokenIDs = stopTokenIDs
        self.maxTokens = request.parameters.maxTokens
        self.cache = cache
        self.originalInput = request.input
        self.pendingTokens = request.input.text.tokens
        self.nextToken = nil
        self.promptTokenCount = request.input.text.tokens.size
        self.prefillStartTime = Date()
        self.prefillStepSize = request.parameters.prefillStepSize
        self.kvBits = request.parameters.kvBits
        self.kvMode = request.parameters.kvMode
        self.genPromptLen = request.genPromptLen
    }
}
