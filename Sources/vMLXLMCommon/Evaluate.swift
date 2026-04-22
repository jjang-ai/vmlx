// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import os

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor {

    /// Called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// Called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// Called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.

/// KV cache quantization/compression mode.
///
/// Controls how the KV cache is compressed during inference:
///
/// ```swift
/// // No compression (default, same as today)
/// var params = GenerateParameters()
///
/// // Affine quantization (existing path, unchanged)
/// var params = GenerateParameters(kvBits: 4, kvGroupSize: 64)
///
/// // TurboQuant compression (Hadamard + Lloyd-Max + QJL)
/// var params = GenerateParameters()
/// params.kvMode = .turboQuant(keyBits: 3, valueBits: 3)
/// ```
public enum KVQuantizationMode: Sendable, Equatable {
    /// No cache compression (float16, default)
    case none

    /// Affine quantization (existing QuantizedKVCache path)
    case affine(bits: Int, groupSize: Int = 64)

    /// TurboQuant compression: randomized Hadamard rotation + Lloyd-Max optimal
    /// codebook quantization + QJL residual correction for keys.
    /// Achieves 4.7-5.0x compression with zero generation speed overhead.
    ///
    /// - Parameters:
    ///   - keyBits: Total bits per key element (default 3). Split as (b-1) codebook + 1 QJL.
    ///   - valueBits: Total bits per value element (default 3). All bits go to codebook.
    case turboQuant(keyBits: Int = 3, valueBits: Int = 3)
}

public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize: Int

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// KV cache quantization/compression mode.
    ///
    /// When set to a value other than `.none`, this takes precedence over `kvBits`/`kvGroupSize`.
    /// The legacy `kvBits`/`kvGroupSize` fields continue to work for backward compatibility.
    public var kvMode: KVQuantizationMode = .none

    public var enableCompiledDecode: Bool = false
    public var compiledMaxCacheLength: Int? = nil

    /// Sampling temperature
    public var temperature: Float

    /// Top-p sampling
    public var topP: Float

    /// Top-k sampling (0 disables)
    public var topK: Int

    /// Min-p sampling threshold relative to the highest probability token (0 disables)
    public var minP: Float

    /// Penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// Number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// additive penalty for tokens that appear in recent context
    public var presencePenalty: Float?

    /// number of tokens to consider for presence penalty
    public var presenceContextSize: Int

    /// additive penalty that scales with token frequency in recent context
    public var frequencyPenalty: Float?

    /// number of tokens to consider for frequency penalty
    public var frequencyContextSize: Int

    /// Optional seed for the sampler's private `MLXRandom.RandomState`.
    /// Added as a stored-only property (NOT wired through `init`) so
    /// build targets that link `GenerateParameters` don't need a relink.
    /// Callers mutate via direct assignment: `p.samplerSeed = 42`.
    public var samplerSeed: UInt64? = nil

    /// When true, capture per-token log probabilities during generation.
    /// The `LogprobsCollector` processor is automatically created by
    /// `logprobsProcessor()` when this is true.
    public var logprobs: Bool = false

    /// Number of top log probabilities to capture per token position.
    /// Only meaningful when `logprobs == true`. Range: 0–20.
    public var topLogprobs: Int = 0

    /// When true, echo prompt text in the response and capture prompt
    /// token logprobs (when combined with `logprobs`). This is the
    /// legacy `/v1/completions` `echo` parameter. Normal chat
    /// completions never set this, so there is zero overhead on the
    /// standard path.
    public var echo: Bool = false

    /// Number of top prompt token logprobs to capture per position
    /// during prefill. Works independently of `echo` — prompt logprobs
    /// are captured even when `echo` is false. Range: 0–20.
    /// Only meaningful for legacy `/v1/completions`. Normal chat
    /// completions never set this, ensuring zero overhead.
    public var promptLogprobs: Int = 0

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        kvMode: KVQuantizationMode = .none,
        enableCompiledDecode: Bool = false,
        compiledMaxCacheLength: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.kvMode = kvMode
        self.enableCompiledDecode = enableCompiledDecode
        self.compiledMaxCacheLength = compiledMaxCacheLength
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefillStepSize = prefillStepSize
    }

    public func sampler() -> LogitSampler {
        let usesTopP = topP > 0 && topP < 1
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(
                temperature: temperature, topP: topP, topK: topK, minP: minP,
                seed: samplerSeed)
        } else {
            return CategoricalSampler(temperature: temperature, seed: samplerSeed)
        }
    }

    public func processor() -> LogitProcessor? {
        let repetitionContext: RepetitionContext?
        if let repetitionPenalty, repetitionPenalty != 0, repetitionContextSize > 0 {
            repetitionContext = RepetitionContext(
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize
            )
        } else {
            repetitionContext = nil
        }

        let presenceContext: PresencePenaltyContext?
        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            presenceContext = PresencePenaltyContext(
                presencePenalty: presencePenalty,
                presenceContextSize: presenceContextSize
            )
        } else {
            presenceContext = nil
        }

        let frequencyContext: FrequencyPenaltyContext?
        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            frequencyContext = FrequencyPenaltyContext(
                frequencyPenalty: frequencyPenalty,
                frequencyContextSize: frequencyContextSize
            )
        } else {
            frequencyContext = nil
        }

        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil {
            return nil
        }

        return PenaltyProcessor(
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext
        )
    }

    /// Returns a `LogprobsCollector` when `logprobs == true`, nil otherwise.
    /// The collector captures per-token log probabilities during generation.
    /// Callers should compose this with any penalty processor via
    /// `CompositeLogitProcessor` if both are needed.
    public func logprobsProcessor() -> LogprobsCollector? {
        guard logprobs else { return nil }
        return LogprobsCollector(topLogprobs: topLogprobs)
    }
}

// MARK: - Logprobs collection

/// Per-token log probability data captured during generation.
/// Carries the selected token's logprob and optionally the top-N
/// alternative tokens at each position.
public struct TokenLogprob: Sendable {
    /// The token string (detokenized).
    public var token: String
    /// Log probability of the selected token.
    public var logprob: Float
    /// Byte offset of this token in the generated output.
    public var byteOffset: Int?
    /// Top-N alternative tokens with their log probabilities.
    public var topLogprobs: [TopTokenLogprob]

    public init(token: String, logprob: Float, byteOffset: Int? = nil,
                topLogprobs: [TopTokenLogprob] = []) {
        self.token = token
        self.logprob = logprob
        self.byteOffset = byteOffset
        self.topLogprobs = topLogprobs
    }
}

/// A single alternative token and its log probability.
public struct TopTokenLogprob: Sendable {
    public var token: String
    public var logprob: Float

    public init(token: String, logprob: Float) {
        self.token = token
        self.logprob = logprob
    }
}

/// A `LogitProcessor` that captures per-token log probabilities.
///
/// After each sampling step, this processor records the log probability
/// of the selected token and (optionally) the top-N alternatives.
/// The collected data is retrieved via `collectedLogprobs` after
/// generation completes.
///
/// Logprobs are computed *after* penalty processing (repetition,
/// presence, frequency) so they reflect the actual sampling distribution.
public struct LogprobsCollector: LogitProcessor {
    let topLogprobs: Int
    public private(set) var collectedLogprobs: [TokenLogprob] = []

    public init(topLogprobs: Int = 0) {
        self.topLogprobs = topLogprobs
    }

    public mutating func prompt(_ prompt: MLXArray) {
        collectedLogprobs = []
    }

    public func process(logits: MLXArray) -> MLXArray {
        // No-op: logprobs are captured after sampling in `didSample`.
        return logits
    }

    public mutating func didSample(token: MLXArray) {
        // This is called with the sampled token *after* the sampler has
        // made its choice. We don't have the logits here — those are
        // consumed inside the sampler. Instead, logprob capture happens
        // inside the token loop via `captureLogprobs(logits:sampledToken:)`.
    }

    /// Capture log probabilities for a single token step.
    /// Called from the token loop *after* sampling but before the token
    /// is committed. Computes log-softmax of the full logit vector to
    /// extract the chosen token's probability and top-N alternatives.
    ///
    /// - Parameters:
    ///   - logits: raw logits BEFORE sampling (post-penalty).
    ///   - sampledToken: the token ID selected by the sampler.
    ///   - tokenizer: used to decode token IDs back to strings.
    public mutating func capture(
        logits: MLXArray,
        sampledToken: Int,
        tokenizer: Tokenizer
    ) {
        var lp = logits
        if lp.dtype == .bfloat16 {
            lp = lp.asType(.float32)
        }
        let logProbs = logSoftmax(lp)
        let sampledLogprob = logProbs[0..., sampledToken].item(Float.self)

        let tokenStr = tokenizer.decode(tokenIds: [sampledToken])

        var topAlts: [TopTokenLogprob] = []
        if topLogprobs > 0 {
            let vocabSize = logProbs.dim(-1)
            let n = min(topLogprobs, vocabSize)
            // Use argPartition (O(V)) instead of argSort (O(V log V))
            // to extract top-N entries. Negate logProbs so the k largest
            // original values become the k smallest negated values,
            // landing at positions [0, n) after partition.
            let flatLogProbs = logProbs.reshaped(-1)
            let negated = -flatLogProbs
            let partIndices = argPartition(negated, kth: n - 1, axis: -1)
            let topIdx = partIndices[0..<n].asArray(Int.self)

            // Build the top_logprobs array with chosen token first,
            // then alternatives sorted by descending logprob.
            // This satisfies the OpenAI contract where top_logprobs[0]
            // is always the chosen/generated token.
            var chosenEntry: TopTokenLogprob?
            var alternatives: [TopTokenLogprob] = []
            alternatives.reserveCapacity(n)
            for idx in topIdx {
                let lpVal = flatLogProbs[idx].item(Float.self)
                let tokStr = tokenizer.decode(tokenIds: [idx])
                if idx == sampledToken {
                    chosenEntry = TopTokenLogprob(token: tokStr, logprob: lpVal)
                } else {
                    alternatives.append(TopTokenLogprob(token: tokStr, logprob: lpVal))
                }
            }

            // If the chosen token wasn't in the top-N partition, create
            // its entry from the already-extracted logprob.
            if chosenEntry == nil {
                chosenEntry = TopTokenLogprob(token: tokenStr, logprob: sampledLogprob)
            }

            // Sort alternatives descending by logprob, keep at most n-1.
            alternatives.sort { $0.logprob > $1.logprob }
            if alternatives.count > n - 1 {
                alternatives = Array(alternatives.prefix(n - 1))
            }

            // Chosen token first, then sorted alternatives.
            topAlts = [chosenEntry!] + alternatives
        }

        collectedLogprobs.append(TokenLogprob(
            token: tokenStr,
            logprob: sampledLogprob,
            topLogprobs: topAlts
        ))
    }

    /// Reset collected state for reuse.
    public mutating func reset() {
        collectedLogprobs = []
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`
/// to sample the logits.
///
/// Filters are applied in the same order as Python mlx-lm: top_p → min_p → top_k.
/// Each filter operates on the full vocabulary in original token order, masking
/// rejected tokens with `-inf`. This matches the composable filter chain in
/// `mlx_lm.sample_utils.make_sampler`.
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    public init(
        temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0,
        seed: UInt64? = nil
    ) {
        self.temp = MLXArray(temperature)
        if topP > 0 && topP < 1 {
            self.topP = MLXArray(topP)
        } else {
            self.topP = nil
        }
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        // lint-ok: negInf is only consumed by `sample()` which first
        // upcasts logits to fp32 (see line 287), so fp32 negInf is the
        // correct dtype here. See SWIFT-NO-REGRESSION-CHECKLIST §27.
        self.negInf = MLXArray(-Float.infinity)
        // iter-64: honor caller-supplied seed. Prior code ignored
        // MLXRandom.seed() set by Stream.swift because RandomState is
        // per-instance — caught via case_seed_reproducibility harness.
        self.randomState = seed.map(MLXRandom.RandomState.init(seed:))
            ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            // Apply filters in Python mlx-lm order: top_p → min_p → top_k.
            if let topP {
                logprobs = applyTopP(logprobs, topP: topP)
            }
            if let minP {
                logprobs = applyMinP(logprobs, minP: minP)
            }
            if let topK {
                logprobs = applyTopK(logprobs, topK: topK)
            }

            return categorical(logprobs * (1 / temp))
        }
    }

    /// Keep tokens whose cumulative probability exceeds `1 - topP` (nucleus sampling).
    /// Matches `apply_top_p` from `mlx_lm/sample_utils.py`.
    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)

        // Mask low-probability tail in sorted order, scatter back to original vocab order.
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    /// Keep tokens with probability >= maxProb * minP.
    /// Matches `apply_min_p` from `mlx_lm/sample_utils.py`.
    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        // threshold in log-space: log(maxProb * minP) = maxLogprob + log(minP)
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)
        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    /// Keep only the top-k highest-probability tokens.
    /// Mirrors `apply_top_k` from `mlx_lm/sample_utils.py`.
    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }
        // O(V) partition on negated logprobs so top-k land at [0, topK).
        // Indices at [topK, V) are the tokens to mask out.
        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, seed: UInt64? = nil) {
        self.temp = MLXArray(temperature)
        // iter-64: honor caller-supplied seed (see TopPSampler comment).
        self.randomState = seed.map(MLXRandom.RandomState.init(seed:))
            ?? MLXRandom.RandomState()
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// GPU-resident ring buffer of recent token IDs.
///
/// Shared by penalty processors to avoid duplicating ring buffer logic.
/// Uses `MLX.where` mask operations for GPU-only updates (no CPU←GPU sync),
/// preserving `asyncEval()` pipelining in `TokenIterator`.
struct TokenRing {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    /// The valid portion of the ring (all of it once full), or `nil` if empty.
    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    /// Bulk-load from a prompt. Keeps the last `capacity` tokens.
    ///
    /// Accepts 1D `[T]` or 2D `[B, T]` / `[1, T]` prompts — we always operate
    /// on the flattened 1D token sequence. Using `prompt.dim(0)` directly was a
    /// bug: for 2D prompts it returned the batch dim (e.g. 1) instead of the
    /// token count, so `buffer` ended up with the wrong length (padding was
    /// computed from batch size, not token count) and every subsequent
    /// `append`'s `MLX.where` broadcast would fail.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let flat = prompt.reshaped(-1).asType(.int32)
        let n = flat.size
        if n <= capacity {
            if n < capacity {
                let padding = MLXArray.zeros([capacity - n], type: Int32.self)
                buffer = concatenated([flat, padding])
            } else {
                buffer = flat
            }
            count = n
            writeIndex = n % capacity
        } else {
            buffer = flat[(-capacity)...]
            count = capacity
            writeIndex = 0
        }
    }

    /// Append a single token using GPU-only mask write (no CPU←GPU sync).
    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
}

/// Processor that implements a `repetitionPenalty`.
public struct RepetitionContext: LogitProcessor {
    private var ring: TokenRing
    let repetitionPenalty: Float

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        var selectedLogits = logits[0..., indices]

        selectedLogits = MLX.where(
            selectedLogits .< 0, selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty)

        logits[0..., indices] = selectedLogits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive presence penalty to tokens in a recent context window.
///
/// The penalty is applied once per unique token via scatter-write (writing the
/// same value to the same index multiple times is idempotent).
public struct PresencePenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let presencePenalty: Float

    public init(presencePenalty: Float, presenceContextSize: Int) {
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else { return logits }
        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive frequency penalty to tokens in a recent context window.
///
/// Frequency counting is performed on GPU via `scatter_add` to build a histogram
/// of token occurrences, avoiding CPU←GPU synchronization.
public struct FrequencyPenaltyContext: LogitProcessor {
    private var ring: TokenRing
    let frequencyPenalty: Float

    public init(frequencyPenalty: Float, frequencyContextSize: Int) {
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else { return logits }

        let vocabSize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabSize], type: Float32.self)
            .at[validTokens.asType(.int32)].add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating public func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that composes penalty processors in Python mlx-lm order.
public struct PenaltyProcessor: LogitProcessor {
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?

    public init(
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?
    ) {
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
    }

    mutating public func prompt(_ prompt: MLXArray) {
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
    }
}

/// Common properties shared by token-generating iterators.
protocol TokenIteratorProtocol: Sequence, IteratorProtocol where Element == Int {
    var maxTokens: Int? { get }
    var tokenCount: Int { get }
    var promptPrefillTime: TimeInterval { get }
    /// Per-token log probabilities collected during generation.
    /// Empty when logprobs were not requested.
    var collectedLogprobs: [TokenLogprob] { get }
    mutating func popCollectedLogprobs() -> [TokenLogprob]
    /// Prompt token logprobs captured during prefill when echo or
    /// prompt_logprobs was requested. `nil` on the normal path.
    var promptLogprobsResult: [TokenLogprob]? { get }
    /// Pop and return prompt logprobs, clearing the stored value.
    /// Returns nil if not applicable.
    mutating func popPromptLogprobs() -> [TokenLogprob]?
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: generateParameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: TokenIteratorProtocol {

    private static let logger = Logger(subsystem: "vmlx", category: "TokenIterator")

    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    /// Optional logprob collector — created when `GenerateParameters.logprobs`
    /// is true. Captures per-token log probabilities during generation.
    var logprobsCollector: LogprobsCollector?

    /// Tokenizer reference for logprob token decoding. Only set when
    /// `logprobsCollector` is active to avoid unnecessary retains.
    let logprobsTokenizer: Tokenizer?

    /// Prompt token logprobs captured during prefill when `echo == true`
    /// or `promptLogprobs > 0`. `nil` when neither is requested (zero
    /// overhead on normal path). First entry has `logprob` set to
    /// `.nan` to indicate null (no prior context for position 0).
    var promptLogprobsResult: [TokenLogprob]?

    /// Whether prompt logprob capture is needed. Computed once from
    /// GenerateParameters at init time; gates the expensive logSoftmax
    /// in prepare() to ensure zero overhead on the normal path.
    let needsPromptLogprobs: Bool

    /// Top-K for prompt logprobs (from GenerateParameters.promptLogprobs).
    /// Only meaningful when `needsPromptLogprobs == true`.
    let promptLogprobsTopK: Int

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let kvMode: KVQuantizationMode

    private var compiledForward: (@Sendable ([MLXArray]) -> [MLXArray])?

    // Multi-tier cache coordinator (skeleton integration)
    let cacheCoordinator: CacheCoordinator?

    /// Prompt token IDs captured at init for cache store after generation.
    let promptTokenIds: [Int]

    /// Stable fingerprint of any VLM image/video content in the input.
    /// `nil` for text-only inputs. Mixed into cache-coordinator keys so
    /// VLM multi-turn conversations can cache-hit on identical media,
    /// and won't collide with text-only entries. See `computeMediaSalt`.
    let mediaSalt: String?

    /// Number of trailing tokens in `promptTokenIds` that belong to the
    /// chat template's `add_generation_prompt=true` suffix. Stripped
    /// before hashing so multi-turn thinking-model requests can hit
    /// prior prefill state. `0` for base-model / continuation requests.
    /// Mirrors Python `vmlx_engine/prefix_cache.py:gen_prompt_len`.
    let genPromptLen: Int

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.logprobsCollector = parameters.logprobsProcessor()
        self.logprobsTokenizer = nil
        self.promptLogprobsResult = nil
        self.needsPromptLogprobs = false
        self.promptLogprobsTopK = 0
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvMode = parameters.kvMode

        self.cacheCoordinator = nil
        self.promptTokenIds = []
        self.mediaSalt = nil
        self.genPromptLen = 0

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    ///   - cacheCoordinator: optional multi-tier cache coordinator for prefix reuse
    ///   - tokenizer: optional tokenizer for logprob token decoding
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        cacheCoordinator: CacheCoordinator? = nil,
        genPromptLen: Int = 0,
        tokenizer: Tokenizer? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.cacheCoordinator = cacheCoordinator
        self.genPromptLen = genPromptLen

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.logprobsCollector = parameters.logprobsProcessor()
        // Tokenizer is needed for both generation logprobs AND prompt
        // logprobs — ensure it's retained when either is active.
        let needsTokenizer = self.logprobsCollector != nil
            || parameters.promptLogprobs > 0
            || (parameters.echo && parameters.logprobs)
        self.logprobsTokenizer = needsTokenizer ? tokenizer : nil

        // Prompt logprob capture: gated behind explicit logprob requests
        // to ensure zero overhead on the normal path. echo alone does
        // NOT trigger logprob capture — it only echoes the prompt text.
        // Prompt logprobs are captured when:
        //   - promptLogprobs > 0 (standalone prompt logprob parameter)
        //   - echo == true AND logprobs == true (combined echo+logprobs)
        let wantsPromptLogprobs = parameters.promptLogprobs > 0
            || (parameters.echo && parameters.logprobs)
        self.needsPromptLogprobs = wantsPromptLogprobs
        // When promptLogprobs is explicitly set, use it for top-K.
        // Otherwise fall back to topLogprobs (used with echo+logprobs).
        self.promptLogprobsTopK = parameters.promptLogprobs > 0
            ? parameters.promptLogprobs
            : parameters.topLogprobs
        self.promptLogprobsResult = nil

        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvMode = parameters.kvMode

        // Capture prompt token IDs for cache store after generation.
        let tokenCount = input.text.tokens.size
        if tokenCount > 0 {
            self.promptTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        } else {
            self.promptTokenIds = []
        }

        // Compute a stable fingerprint of any image/video content once at
        // init, so both the pre-prepare fetch below and the post-generation
        // store see the same salt. Text-only inputs get nil here, which
        // preserves the exact pre-existing text-only cache hashing.
        self.mediaSalt = computeMediaSalt(for: input)

        // Multi-tier cache: attempt prefix fetch before prepare.
        // On cache hit, restore KV state and only prefill remaining tokens.
        //
        // VLM inputs (image/video) are now supported: the mediaSalt computed
        // above is mixed into the cache keys by the coordinator, so "same
        // text prefix + same image" hits while "same text + different image"
        // misses. Previously any image/video bypassed the cache entirely,
        // wasting a full vision-tower encode and prefill on every turn.
        var inputForPrepare = input
        // SLIDING-1: legacy `!hasRotatingCache` fetch guard removed.
        // TQDiskSerializer v2 round-trips RotatingKVCache via the
        // `.rotating` LayerKind, so sliding-window layers now participate
        // in the same fetch/restore path as standard KV. Reference
        // commit `bf942a8`.
        //
        // PROMPT-LOGPROBS: skip cache fetch when prompt logprobs are
        // requested (echo+logprobs or prompt_logprobs > 0). On a cache
        // hit, only the uncached remainder (or just the last token on
        // full hit) is prefilled, so logits for cached prompt positions
        // are never computed and prompt logprobs would be missing.
        // Bypassing the cache ensures a full prefill with complete
        // logits, at the cost of re-computing cached prefix KV state.
        // This is acceptable because prompt-logprob requests already
        // incur the expensive logSoftmax over [1, seq_len, vocab].
        if let coordinator = cacheCoordinator, !promptTokenIds.isEmpty, !needsPromptLogprobs {
            let result = coordinator.fetch(
                tokens: promptTokenIds, mediaSalt: mediaSalt,
                genPromptLen: genPromptLen)
            switch result {
            case .hit(_, let remainingTokens, let detail, let blocks, let ssmStates, let diskArrays):
                var restored = false
                if !blocks.isEmpty {
                    let restoredTokens = restoreLayerData(from: blocks, into: self.cache)
                    if restoredTokens > 0 {
                        if let ssm = ssmStates {
                            restoreSSMStates(ssm, into: self.cache)
                        }
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit: restored \(restoredTokens) tokens, prefilling \(remainingTokens.count) remaining"
                        )
                    }
                }

                // Disk cache restore (blocks are empty, arrays are present).
                //
                // The v2 disk format includes per-layer `.mamba` entries that
                // `restoreFromDiskArrays` writes directly into MambaCache
                // instances via `restoreMambaLayer`. The bundled `ssmStates`
                // parameter is the SAME data folded into a flat array for
                // legacy in-memory companion cache reuse — calling
                // `restoreSSMStates(ssm, into:)` here would DOUBLE-WRITE the
                // Mamba layers (once per-layer, once via the legacy
                // ellipsis-indexed path) and the second write fataled with
                // `SmallVector out of range` on Qwen3.5-VL-4B-JANG hybrid.
                //
                // Skip the legacy SSM restore in the disk branch — v2's
                // per-layer .mamba handling is sufficient. The paged-cache
                // branch above still does the legacy restore because its
                // `blocks` payload doesn't carry per-layer Mamba state.
                if let diskArrays, !restored {
                    let diskRestored = restoreFromDiskArrays(diskArrays, into: self.cache)
                    if diskRestored > 0 {
                        restored = true
                        Self.logger.info(
                            "Cache \(detail.rawValue) hit: restored \(diskRestored) tokens from disk, prefilling \(remainingTokens.count) remaining"
                        )
                    }
                }
                _ = ssmStates  // silence unused — used by paged branch only

                if restored {
                    if remainingTokens.isEmpty {
                        // Full cache hit — feed just the last token to seed decode.
                        // prepare() needs at least 1 token to produce initial logits.
                        //
                        // SHAPE: 2D `[1, 1]` (batch=1, seq=1), NOT 1D `[1]`.
                        // VL model `prepare(_:cache:)` paths (e.g. Qwen35
                        // VL languageModel) call `inputs.dim(1)` to get
                        // sequence length. A 1D input fatal-traps with
                        // `SmallVector out of range. at array.cpp:335`
                        // because dim(1) overflows the shape SmallVector.
                        // The crash was reachable on every disk-cache hit
                        // for any VL model — including Qwen3.5-VL hybrid.
                        // Construct via 1D + .newAxis to dodge a type-inference
                        // ambiguity around the nested Int32 literal — Swift
                        // resolves the outer `[[...]]` to `Array<Array<Int>>`
                        // and rejects Int32. Matches the [.newAxis] pattern
                        // used a few lines below for the multi-token branch.
                        let lastToken = MLXArray([Int32(promptTokenIds.last!)])[.newAxis]
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: lastToken),
                            image: nil, video: nil)
                    } else {
                        let remainingArray = MLXArray(remainingTokens.map { Int32($0) })[.newAxis]
                        inputForPrepare = LMInput(
                            text: LMInput.Text(tokens: remainingArray),
                            image: nil, video: nil)
                    }
                }
            case .miss:
                let count = promptTokenIds.count
                Self.logger.debug("Cache miss for \(count) prompt tokens")
            }
        }

        // Prefill: either full input (cache miss) or remaining tokens (cache hit).
        self.promptPrefillTime = try measure {
            try prepare(input: inputForPrepare, windowSize: parameters.prefillStepSize)
        }

        if parameters.enableCompiledDecode {
            try setupCompiledDecode(
                maxCacheLength: parameters.compiledMaxCacheLength ?? 4096)
        }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.logprobsCollector = nil
        self.logprobsTokenizer = nil
        self.promptLogprobsResult = nil
        self.needsPromptLogprobs = false
        self.promptLogprobsTopK = 0
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0
        self.kvMode = .none

        self.cacheCoordinator = nil
        self.promptTokenIds = []
        self.mediaSalt = nil
        self.genPromptLen = 0

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: prefillStepSize)
        }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        // Capture prompt token IDs for logprob indexing. Must be done
        // BEFORE model.prepare() consumes the input.
        let inputTokenIds: [Int]
        if needsPromptLogprobs {
            inputTokenIds = input.text.tokens.reshaped(-1).asArray(Int.self)
        } else {
            inputTokenIds = []
        }

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)

        case .logits(let result):
            // Capture prompt logprobs from the full prefill logits when
            // echo or prompt_logprobs is requested. GATED behind
            // needsPromptLogprobs to ensure ZERO overhead on normal path.
            //
            // The logits tensor is [1, seq_len, vocab]. We compute
            // logSoftmax once over the entire tensor (batched), then
            // index out per-position logprobs for the actual prompt
            // tokens. This is O(seq_len * vocab) for logSoftmax +
            // O(seq_len) for indexing — no per-token loops.
            if needsPromptLogprobs && !inputTokenIds.isEmpty,
               let tokenizer = logprobsTokenizer {
                var logits = result.logits
                if logits.dtype == .bfloat16 {
                    logits = logits.asType(.float32)
                }
                let logProbs = logSoftmax(logits)

                // logProbs shape: [1, seq_len, vocab]
                let seqLen = inputTokenIds.count
                var captured: [TokenLogprob] = []
                captured.reserveCapacity(seqLen)

                for i in 0..<seqLen {
                    let tokenId = inputTokenIds[i]
                    let tokenStr = tokenizer.decode(tokenIds: [tokenId])

                    // Position 0 has no prior context → null logprob.
                    // We use NaN as sentinel; the route handler maps it
                    // to JSON null.
                    if i == 0 {
                        var topAlts: [TopTokenLogprob] = []
                        if promptLogprobsTopK > 0 {
                            topAlts = Self.extractTopK(
                                logProbs: logProbs, position: i,
                                topK: promptLogprobsTopK,
                                chosenTokenId: tokenId, chosenLogprob: .nan,
                                tokenizer: tokenizer
                            )
                        }
                        captured.append(TokenLogprob(
                            token: tokenStr,
                            logprob: .nan,
                            topLogprobs: topAlts
                        ))
                        continue
                    }

                    let logProb = logProbs[0..., i, tokenId].item(Float.self)

                    var topAlts: [TopTokenLogprob] = []
                    if promptLogprobsTopK > 0 {
                        topAlts = Self.extractTopK(
                            logProbs: logProbs, position: i,
                            topK: promptLogprobsTopK,
                            chosenTokenId: tokenId, chosenLogprob: logProb,
                            tokenizer: tokenizer
                        )
                    }
                    captured.append(TokenLogprob(
                        token: tokenStr,
                        logprob: logProb,
                        topLogprobs: topAlts
                    ))
                }
                self.promptLogprobsResult = captured
            }

            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)
        }


    }

    /// Extract top-K logprobs at a given position using argPartition
    /// (O(V)) instead of argSort (O(V log V)). Returns entries with
    /// the chosen token first, followed by alternatives sorted by
    /// descending logprob. Total entries ≤ topK.
    private static func extractTopK(
        logProbs: MLXArray,
        position: Int,
        topK: Int,
        chosenTokenId: Int,
        chosenLogprob: Float,
        tokenizer: Tokenizer
    ) -> [TopTokenLogprob] {
        // Slice out [1, 1, vocab] → reshape to [vocab]
        let posLogProbs = logProbs[0..., position, 0...].reshaped(-1)
        let vocabSize = posLogProbs.dim(0)
        let k = Swift.min(topK, vocabSize)

        // argPartition gives us the k smallest values (we want largest).
        // Use negated logProbs so argPartition(negated, kth) gives indices
        // of the k largest logprobs.
        let negated = -posLogProbs
        let partitioned = argPartition(negated, kth: k - 1)
        let topIndices = partitioned[0..<k].asArray(Int.self)

        // Build entries: chosen token first, then sorted alternatives.
        var chosenEntry: TopTokenLogprob?
        var alternatives: [TopTokenLogprob] = []
        alternatives.reserveCapacity(k)
        for idx in topIndices {
            let lpVal = posLogProbs[idx].item(Float.self)
            let tokStr = tokenizer.decode(tokenIds: [idx])
            if idx == chosenTokenId {
                chosenEntry = TopTokenLogprob(token: tokStr, logprob: lpVal)
            } else {
                alternatives.append(TopTokenLogprob(token: tokStr, logprob: lpVal))
            }
        }

        if chosenEntry == nil {
            let tokStr = tokenizer.decode(tokenIds: [chosenTokenId])
            chosenEntry = TopTokenLogprob(token: tokStr, logprob: chosenLogprob)
        }

        // Sort alternatives descending, keep at most k-1.
        alternatives.sort { $0.logprob > $1.logprob }
        if alternatives.count > k - 1 {
            alternatives = Array(alternatives.prefix(k - 1))
        }

        return [chosenEntry!] + alternatives
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]

        if var processor {
            logits = processor.process(logits: logits)
            let y = sampler.sample(logits: logits)
            // Capture logprobs from post-penalty logits.
            if var collector = logprobsCollector,
               let tokenizer = logprobsTokenizer {
                let tokenId = y.item(Int.self)
                collector.capture(logits: logits, sampledToken: tokenId, tokenizer: tokenizer)
                self.logprobsCollector = collector
            }
            processor.didSample(token: y)
            self.processor = processor
            return y
        }

        let y = sampler.sample(logits: logits)
        // Capture logprobs from raw logits (no penalty processor).
        if var collector = logprobsCollector,
           let tokenizer = logprobsTokenizer {
            let tokenId = y.item(Int.self)
            collector.capture(logits: logits, sampledToken: tokenId, tokenizer: tokenizer)
            self.logprobsCollector = collector
        }
        return y
    }

    // Whether cache quantization is needed (skip the function call entirely when not)
    var needsCacheQuantization: Bool { kvBits != nil || kvMode != .none }

    mutating func setupCompiledDecode(maxCacheLength: Int) throws {
        guard HardwareInfo.isCompiledDecodeSupported else { return }
        // Compiled decode requires no auxiliary state — models with state (e.g. vision
        // encoder cross-attention) use the uncompiled path.
        guard state == nil else { return }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // KVCacheSimple → CompilableKVCache (static buffer, compile-traceable).
        // ArraysCache/MambaCache — NOT compile-safe, bail.
        // RotatingKVCache, QuantizedKVCache — bail.
        // Only compile if ALL caches are KVCacheSimple.
        for i in 0..<cache.count {
            if cache[i] is KVCacheSimple {
                continue
            } else {
                return
            }
        }

        let capturedModel = model
        let cacheRef = cache

        self.compiledForward = compile(
            inputs: cacheRef, outputs: cacheRef
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: cacheRef.isEmpty ? nil : cacheRef,
                state: nil)
            return [result.logits]
        }
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        if self.compiledForward != nil {
            let input = previous.tokens
            let result = self.compiledForward!([input])

            if result.count > 0 {
                self.state = nil
                if needsCacheQuantization {
                    maybeQuantizeKVCache(
                        cache: &cache, kvBits: kvBits,
                        kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart,
                        kvMode: kvMode)
                }
                return convertToToken(logits: result[0])
            }
            self.compiledForward = nil
        }

        // Models expect [B, L] input. If the caller passed 1D tokens [L], add a batch
        // axis. If they passed 2D [B, L] already (some VLM bench/test paths), use as-is —
        // adding another newAxis would produce 3D and break QuantizedLinear matmul on
        // pure-LLM model paths (Llama, Mistral, Phi, etc).
        let stepInput: LMInput.Text =
            previous.tokens.ndim == 1 ? previous[text: .newAxis] : previous
        let result = model(
            stepInput, cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        if needsCacheQuantization {
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart,
                kvMode: kvMode
            )
        }

        return convertToToken(logits: result.logits)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        let previousY = y

        let token = step(previous: previousY)
        y = .init(tokens: token)

        asyncEval(token)

        tokenCount += 1

        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }

        return previousY.tokens.item(Int.self)
    }

    public var collectedLogprobs: [TokenLogprob] {
        logprobsCollector?.collectedLogprobs ?? []
    }

    public mutating func popCollectedLogprobs() -> [TokenLogprob] {
        guard var collector = logprobsCollector else { return [] }
        let result = collector.collectedLogprobs
        collector.reset()
        self.logprobsCollector = collector
        return result
    }

    public mutating func popPromptLogprobs() -> [TokenLogprob]? {
        let result = promptLogprobsResult
        promptLogprobsResult = nil
        return result
    }
}
///
/// This is typically used via a call to ``generate(input:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)``
/// returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let mainModel: LanguageModel
/// let draftModel: LanguageModel
///
/// let iterator = try SpeculativeTokenIterator(
///     input: input, mainModel: mainModel, draftModel: draftModel,
///     parameters: generateParameters, numDraftTokens: 2)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `speculative_generate_step()` from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
public struct SpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    var draftY: LMInput.Text

    let mainModel: any LanguageModel
    let draftModel: any LanguageModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var draftCache: [KVCache]
    let quantizeKVCache: (inout [KVCache]) -> Void

    var processor: LogitProcessor?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?
    let numDraftTokens: Int

    // Buffer of accepted tokens from the current speculation round
    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    /// Initialize a `SpeculativeTokenIterator` with the given input.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - mainModel: the main (verifier) ``LanguageModel``
    ///   - draftModel: the draft ``LanguageModel`` (must share the same tokenizer)
    ///   - mainCache: optional ``KVCache`` for the main model
    ///   - draftCache: optional ``KVCache`` for the draft model
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: number of tokens the draft model proposes per round
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int
    ) throws {
        self.y = input.text
        self.draftY = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                kvMode: parameters.kvMode
            )
        }

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Prefill both main and draft models with the prompt, priming caches for generation
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        // Prefill main model
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
        }

        // Prefill draft model, don't call didSample here -- processor tracks main model's accepted sequence only
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            draftY = tokens
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            draftY = .init(tokens: token)
            asyncEval(draftY.tokens)
        }
    }

    /// Run one round of speculative decoding: draft, verify, accept/reject
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else {
            return
        }

        // Draft generation: autoregressive loop with draft model
        var draftProcessor = processor  // Copy to discard later
        var draftTokens = [MLXArray]()
        for _ in 0 ..< numDraft {
            let draftResult = draftModel(draftY[text: .newAxis], cache: draftCache, state: nil)
            var draftLogits = draftResult.logits[0..., -1, 0...]
            draftLogits = draftProcessor?.process(logits: draftLogits) ?? draftLogits
            let draftToken = sampler.sample(logits: draftLogits)
            draftProcessor?.didSample(token: draftToken)
            asyncEval(draftToken)
            draftTokens.append(draftToken)
            draftY = .init(tokens: draftToken)
        }

        // Verification: main model processes proposals in one pass
        let verifyTokens = [y.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let mainResult = mainModel(verifyInput[text: .newAxis], cache: mainCache, state: mainState)
        let mainLogits = mainResult.logits
        mainState = mainResult.state

        let mainTokens: MLXArray
        if var verifyProcessor = processor {
            // Process each position sequentially so that the processor sees tokens sampled at earlier positions
            var sampled = [MLXArray]()
            for i in 0 ..< (numDraft + 1) {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                sampled.append(token)
            }
            mainTokens = concatenated(sampled)
        } else {
            // Batch-sample all verify tokens from main model in one operation
            let verifyLogits = mainLogits[0..., verifyStart..., 0...].squeezed(axis: 0)
            mainTokens = sampler.sample(logits: verifyLogits)
        }

        // Compare and accept proposed tokens
        eval(mainTokens, draftTokens)
        let mainTokensList = mainTokens.asArray(Int.self)
        let draftTokensList = concatenated(draftTokens).asArray(Int.self)
        var accepted = 0
        for i in 0 ..< numDraft {
            guard mainTokensList[i] == draftTokensList[i] else {
                break
            }

            processor?.didSample(token: draftTokens[i])
            pendingTokens.append(mainTokensList[i])
            accepted += 1
        }

        // Always emit the main model's token at position `accepted`
        // (either the correction token or the bonus token if all drafts matched)
        let finalToken = mainTokens[accepted ... accepted]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokensList[accepted])

        // Rewind caches for rejected tokens
        trimPromptCache(mainCache, numTokens: numDraft - accepted)
        trimPromptCache(draftCache, numTokens: Swift.max(numDraft - accepted - 1, 0))

        // Apply dynamic cache quantization after rewind
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        // Set y/draftY for the next round
        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)

        // If all draft tokens were accepted, the draft model hasn't processed
        // the last accepted draft token yet. Feed it through to keep caches in sync.
        if accepted == numDraft {
            draftY = .init(
                tokens: concatenated([
                    draftTokens[numDraft - 1].reshaped([1]),
                    finalToken,
                ])
            )
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Drain the pending buffer first
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        // Run a new speculation round
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    public var collectedLogprobs: [TokenLogprob] { [] }

    public mutating func popCollectedLogprobs() -> [TokenLogprob] { [] }

    /// Prompt logprobs not supported for speculative decoding.
    public var promptLogprobsResult: [TokenLogprob]? { nil }

    public mutating func popPromptLogprobs() -> [TokenLogprob]? { nil }
}
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokenIds: The array of generated token IDs.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokenIds: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokenIds = tokenIds
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    @available(*, deprecated, renamed: "init(inputText:tokenIds:output:promptTime:generateTime:)")
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.init(
            inputText: inputText, tokenIds: tokens, output: output, promptTime: promptTime,
            generateTime: generateTime)
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    /// The token IDs of the input prompt.
    public var promptTokenIds: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    @available(*, deprecated, renamed: "promptTokenIds")
    public var promptTokens: [Int] { promptTokenIds }

    /// Generated token IDs
    public let tokenIds: [Int]

    @available(*, deprecated, renamed: "tokenIds")
    public var tokens: [Int] { tokenIds }

    /// Output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokenIds.count }

    /// Time to process the prompt (generate the first token)
    public let promptTime: TimeInterval

    /// Time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokenIds.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in deprecated callback-based generate functions.
public enum GenerateDisposition: Sendable {
    /// Keep producing tokens until an EOS token is produced
    case more

    /// Stop producing tokens, e.g. a token limit has been hit
    case stop
}

private struct SynchronousGenerationLoopResult {
    let generatedTokenIds: [Int]
    let promptTime: TimeInterval
    let generateTime: TimeInterval
    let promptPrefillTime: TimeInterval
    let stopReason: GenerateStopReason
}

private func buildStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer
) -> Set<Int> {
    // Build complete EOS token set from all sources.
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

private func runSynchronousGenerationLoop(
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    let stopTokenIds = buildStopTokenIds(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )

    var generatedTokenIds = [Int]()
    var iterator = iterator
    var stopReason: GenerateStopReason?

    while let token = iterator.next() {
        // Compute the timing for the prompt.
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens.
        if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
            stopReason = .stop
            break
        }

        generatedTokenIds.append(token)

        if didGenerate(token, generatedTokenIds) == .stop {
            stopReason = .cancelled
            break
        }
    }

    // If the iterator ends naturally, the max-token limit was reached.
    // Use the outer-loop-local `generatedTokenIds.count` (authoritative
    // decode count for this pass) rather than `iterator.tokenCount`,
    // which in batched / async flows may be reset or shadowed by the
    // iterator's own internal accounting and cause `.length` to
    // misclassify as `.cancelled`. Mirrors the async-loop fix in the
    // generateTask path below. Session 2026-04-14 deep audit #21.
    if stopReason == nil {
        if let maxTokens = iterator.maxTokens,
           generatedTokenIds.count >= maxTokens
        {
            stopReason = .length
        } else {
            stopReason = .cancelled
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    Stream().synchronize()

    return SynchronousGenerationLoopResult(
        generatedTokenIds: generatedTokenIds,
        promptTime: promptTime,
        generateTime: generateTime,
        promptPrefillTime: iterator.promptPrefillTime,
        stopReason: stopReason ?? .cancelled
    )
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognize special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { _, generatedTokens in
        didGenerate(generatedTokens)
    }

    return GenerateResult(
        inputText: input.text, tokenIds: result.generatedTokenIds,
        output: context.tokenizer.decode(tokenIds: result.generatedTokenIds),
        promptTime: result.promptTime + result.promptPrefillTime,
        generateTime: result.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    let result = runSynchronousGenerationLoop(
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    ) { token, _ in
        didGenerate(token)
    }

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: result.generatedTokenIds.count,
        promptTime: result.promptTime + result.promptPrefillTime,
        generationTime: result.generateTime,
        stopReason: result.stopReason
    )
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent text chunks, tool calls, or summary
/// completion information.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  This is typically OK for
/// one-shot calls, but for "chat session" type calls consider using
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:)``
/// so that the end of the generation task can be observed.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated text chunks (`.chunk`),
///   tool calls (`.toolCall`), and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: generateParameters, context: context)
///
/// // Process the stream asynchronously to handle text chunks and completion info.
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil,
    genPromptLen: Int = 0
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator, genPromptLen: genPromptLen,
        tokenizer: context.tokenizer)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Generates text and tool calls asynchronously using speculative decoding with a draft model.
///
/// This function uses a smaller draft model to propose tokens that are verified in batch
/// by the main model, potentially accelerating generation. The resulting stream yields
/// decoded text chunks, tool calls, and completion information. It has the same output as the
/// non-speculative ``generate(input:cache:parameters:context:wiredMemoryTicket:)``.
///
/// Both models must share the same tokenizer.
///
/// ### Example Usage:
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let mainContext: ModelContext
/// let draftModel: LanguageModel
///
/// let lmInput = try mainContext.processor.prepare(input: input)
///
/// let stream = try generate(
///     input: lmInput, parameters: generateParameters,
///     context: mainContext, draftModel: draftModel)
///
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values.
/// - Throws: An error if the iterator initialization fails.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            format: context.configuration.toolCallFormat ?? .json
        )
    )
    return stream
}

@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Low-level token generation using a ``TokenIterator``, returning an
/// `AsyncStream<Generation>` and a `Task`.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  Callers can await
/// the `task` to observe when the use of the parameters is complete.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens and tool-call format)
///   - tokenizer: tokenizer (for EOS id, unknown token id, and detokenization)
///   - iterator: token iterator
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values and a `Task`
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    // Capture cache coordinator state and extract KV data before consuming the iterator.
    let cacheStoreAction: (@Sendable () -> Void)? = {
        guard let coordinator = iterator.cacheCoordinator,
              !iterator.promptTokenIds.isEmpty else { return nil }
        let promptTokenIds = iterator.promptTokenIds
        let capturedMediaSalt = iterator.mediaSalt
        let capturedGenPromptLen = iterator.genPromptLen
        let rawCache = iterator.cache
        let perLayerData = extractLayerData(from: rawCache)
        let ssmStates: [MLXArray]? = coordinator.isHybrid
            ? extractSSMStates(from: rawCache) : nil
        // MLXArray is not Sendable but is safe after eval; suppress the diagnostic.
        nonisolated(unsafe) let layerCapture = perLayerData
        nonisolated(unsafe) let ssmCapture = ssmStates
        nonisolated(unsafe) let cacheCapture = rawCache
        return {
            coordinator.storeAfterGeneration(
                promptTokens: promptTokenIds,
                perLayerData: layerCapture,
                ssmStates: ssmCapture,
                cache: cacheCapture,
                mediaSalt: capturedMediaSalt,
                genPromptLen: capturedGenPromptLen
            )
        }
    }()

    return generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: tokenizer,
            format: modelConfiguration.toolCallFormat ?? .json
        ),
        cacheStoreAction: cacheStoreAction
    )
}

/// Generates raw token IDs asynchronously using the provided language model input, parameters, and context.
///
/// This is similar to `generate(input:cache:parameters:context:)`, but yields raw token IDs instead of decoded text/tool calls.
/// This is useful for downstream parsers that need access to token IDs directly (e.g. Harmony parsing).
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
///   - cacheCoordinator: Optional multi-tier cache coordinator for prefix reuse.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator, tokenizer: context.tokenizer)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generates raw token IDs asynchronously using speculative decoding with a draft model.
///
/// This is similar to `generate(input:parameters:context:draftModel:draftCache:numDraftTokens:wiredMemoryTicket:)`,
/// but yields raw token IDs instead of decoded text/tool calls.
///
/// Both models must share the same tokenizer.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
/// - Throws: An error if the iterator initialization fails.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: RawTokenLoopHandler()
    )
    return stream
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
///
/// Prefer this overload if you want to be able to observe when the underlying generation work is finished
/// (especially if the consumer terminates the stream early).
///
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values and a `Task`.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
///   - cacheCoordinator: Optional multi-tier cache coordinator for prefix reuse.
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    cacheCoordinator: CacheCoordinator? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters,
        cacheCoordinator: cacheCoordinator)
    return generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
/// `AsyncStream<TokenGeneration>` and a `Task`.
///
/// This is useful for parsers that need access to the token IDs directly (e.g. Harmony parsing)
/// without detokenization or tool-call parsing.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens)
///   - tokenizer: tokenizer (for EOS id and unknown token id)
///   - iterator: token iterator
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits token IDs and a final `.info`, plus a `Task`.
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    // Capture cache coordinator state and extract KV data before consuming the iterator.
    let cacheStoreAction: (@Sendable () -> Void)? = {
        guard let coordinator = iterator.cacheCoordinator,
              !iterator.promptTokenIds.isEmpty else { return nil }
        let promptTokenIds = iterator.promptTokenIds
        let capturedMediaSalt = iterator.mediaSalt
        let capturedGenPromptLen = iterator.genPromptLen
        let rawCache = iterator.cache
        let perLayerData = extractLayerData(from: rawCache)
        let ssmStates: [MLXArray]? = coordinator.isHybrid
            ? extractSSMStates(from: rawCache) : nil
        nonisolated(unsafe) let layerCapture = perLayerData
        nonisolated(unsafe) let ssmCapture = ssmStates
        nonisolated(unsafe) let cacheCapture = rawCache
        return {
            coordinator.storeAfterGeneration(
                promptTokens: promptTokenIds,
                perLayerData: layerCapture,
                ssmStates: ssmCapture,
                cache: cacheCapture,
                mediaSalt: capturedMediaSalt,
                genPromptLen: capturedGenPromptLen
            )
        }
    }()

    return generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler(),
        cacheStoreAction: cacheStoreAction
    )
}

private func generateLoopTask<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    handler: consuming Handler,
    cacheStoreAction: (@Sendable () -> Void)? = nil
) -> (AsyncStream<Handler.Output>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<Handler.Output>.makeStream()

    let iterator = SendableBox(iterator)
    let handler = SendableBox(handler)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        let performIteration = {
            let iterator = iterator.consume()
            var handler = handler.consume()

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0
            var stopReason: GenerateStopReason?

            let stopTokenIds = buildStopTokenIds(
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer
            )

            var iter = iterator
            var didEmitPromptLogprobs = false
            while let token = iter.next() {
                // Check for cancellation on every loop iteration.
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }

                if promptTime == 0 {
                    let now = Date.timeIntervalSinceReferenceDate
                    promptTime = now - start
                    start = now
                }

                // Emit prompt logprobs once, right after the first next()
                // call completes prefill. These are captured during
                // TokenIterator.prepare() and stored on the iterator.
                // Only Generation output supports logprobs.
                if !didEmitPromptLogprobs && Handler.Output.self == Generation.self {
                    didEmitPromptLogprobs = true
                    if let promptLps = iter.popPromptLogprobs() {
                        for lp in promptLps {
                            if case .terminated = continuation.yield(Generation.logprob(lp) as! Handler.Output) {
                                stopReason = .cancelled
                                break
                            }
                        }
                    }
                    if stopReason == .cancelled { break }
                }

                // Check for end-of-sequence tokens
                if token == tokenizer.unknownTokenId || stopTokenIds.contains(token) {
                    if includeStopToken {
                        tokenCount += 1
                        if !handler.onStopToken(token, emit: continuation.yield) {
                            stopReason = .cancelled
                            break
                        }
                    }
                    stopReason = .stop
                    break
                }

                tokenCount += 1
                if !handler.onToken(token, emit: continuation.yield) {
                    stopReason = .cancelled
                    break
                }
                // Emit per-token logprob events when logprobs were requested.
                // Only Generation (text) output supports logprobs — TokenGeneration
                // does not have a .logprob case.
                if Handler.Output.self == Generation.self {
                    for lp in iter.popCollectedLogprobs() {
                        if case .terminated = continuation.yield(Generation.logprob(lp) as! Handler.Output) {
                            stopReason = .cancelled
                            break
                        }
                    }
                }

                if stopReason == .cancelled { break }

            }
            if stopReason == nil {
                // Check the `maxTokens` reached condition BEFORE `Task.isCancelled`.
                // Reaching max_tokens is a natural end-of-stream — the iterator
                // returned nil because it hit the limit, not because of external
                // cancellation. A late-arriving Task cancellation would
                // otherwise mask the real `.length` result. Use the OUTER
                // `tokenCount` (decoded-in-this-loop), NOT `iterator.tokenCount`
                // which in batched flows may be reset or shadowed by the
                // iterator's own internal accounting.
                if let maxTokens = iter.maxTokens, tokenCount >= maxTokens {
                    stopReason = .length
                } else if Task.isCancelled {
                    stopReason = .cancelled
                } else {
                    stopReason = .cancelled
                }
            }

            handler.onGenerationEnd(emit: continuation.yield)

            // Emit any remaining prompt logprobs when the loop exited
            // before they could be emitted (e.g. max_tokens: 0 with echo).
            // Only applies to Generation output (not TokenGeneration).
            if Handler.Output.self == Generation.self,
               let promptLps = iter.popPromptLogprobs() {
                for lp in promptLps {
                    if case .terminated = continuation.yield(Generation.logprob(lp) as! Handler.Output) {
                        break
                    }
                }
            }

            // Multi-tier cache: store prompt state after generation completes.
            // SKIP the store on cancelled turns — a user-stop mid-generation
            // leaves SSM state post-position-mismatched and KV layers with
            // partial decode tokens, so caching them would poison the next
            // turn's prefix hit. BatchEngine already guards this at
            // `BatchEngine.finishSlot` line 648; the sync generateLoopTask
            // path was missing the symmetric guard. Cache deep audit
            // 2026-04-14 finding #3 (hybrid SSM section).
            if case .cancelled = stopReason ?? .cancelled {
                // user-stop / timeout / client-disconnect: don't poison cache
            } else if let cacheStoreAction = cacheStoreAction {
                cacheStoreAction()
            }

            let now = Date.timeIntervalSinceReferenceDate
            let generateTime = now - start

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: promptTime + iter.promptPrefillTime,
                generationTime: generateTime,
                stopReason: stopReason ?? .cancelled
            )
            _ = continuation.yield(handler.infoEvent(info))

            // Synchronize with the stream to ensure tasks are completed
            Stream().synchronize()

            // Finalize the stream
            continuation.finish()
        }

        if let ticket = wiredMemoryTicket {
            await WiredMemoryTicket.withWiredLimit(ticket) {
                performIteration()
            }
        } else {
            performIteration()
        }
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return (stream, task)
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}

// MARK: - Generation structs

/// Reason why token generation stopped.
public enum GenerateStopReason: Sendable {
    /// Generation stopped because an EOS/unknown stop token was encountered.
    case stop

    /// Generation stopped because the configured max token limit was reached.
    case length

    /// Generation stopped due to explicit task cancellation or early stream termination.
    case cancelled
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// Reason generation stopped.
    public let stopReason: GenerateStopReason

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: GenerateStopReason = .stop
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
        self.stopReason = stopReason
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.toolCall`: A tool call parsed from the generated output.
/// - `.info`: Metadata and performance statistics about the generation process.
public enum Generation: Sendable {
    /// A generated text chunk as a String.
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Per-token log probability data. Emitted once per generated token
    /// when `GenerateParameters.logprobs == true`.
    case logprob(TokenLogprob)

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        case .toolCall: nil
        case .logprob: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        case .toolCall: nil
        case .logprob: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        case .logprob: nil
        }
    }

    /// Token log probability data, or nil.
    public var logprob: TokenLogprob? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall: nil
        case .logprob(let lp): lp
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}

/// Represents the different stages or outputs of raw-token generation.
///
/// This mirrors `Generation`, but yields raw token IDs instead of decoded text/tool calls.
public enum TokenGeneration: Sendable {
    /// A generated token ID.
    case token(Int)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Token ID or nil
    public var token: Int? {
        switch self {
        case .token(let token): token
        case .info: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .token: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [TokenGeneration]?, _ element: TokenGeneration)
        -> [TokenGeneration]
    {
        (batch ?? []) + [element]
    }
}

// MARK: - TokenLoopHandlers

private protocol TokenLoopHandler: Sendable {
    associatedtype Output

    /// Return false to stop the loop early.
    mutating func onToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called only when includeStopToken == true and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    ) -> Bool

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> AsyncStream<Output>.Continuation.YieldResult
    )

    func infoEvent(_ info: GenerateCompletionInfo) -> Output
}

private struct TextToolTokenLoopHandler: TokenLoopHandler, @unchecked Sendable {
    typealias Output = Generation

    var detokenizer: NaiveStreamingDetokenizer
    let toolCallProcessor: ToolCallProcessor

    init(tokenizer: Tokenizer, format: ToolCallFormat) {
        detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        toolCallProcessor = ToolCallProcessor(format: format)
    }

    mutating func onToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        detokenizer.append(token: token)
        if let chunk = detokenizer.next() {
            // Process chunk through the tool call processor.
            if let textToYield = toolCallProcessor.processChunk(chunk) {
                if case .terminated = emit(.chunk(textToYield)) {
                    return false
                }
            }

            // Check if we have a complete tool call.
            if let toolCall = toolCallProcessor.toolCalls.popLast() {
                if case .terminated = emit(.toolCall(toolCall)) {
                    return false
                }
            }
        }

        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) -> Bool {
        true
    }

    mutating func onGenerationEnd(
        emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult
    ) {
        toolCallProcessor.processEOS()

        for toolCall in toolCallProcessor.toolCalls {
            if case .terminated = emit(.toolCall(toolCall)) {
                break
            }
        }
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        .info(info)
    }
}

private struct RawTokenLoopHandler: TokenLoopHandler {
    typealias Output = TokenGeneration

    mutating func onToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onStopToken(
        _ token: Int,
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) -> Bool {
        if case .terminated = emit(.token(token)) {
            return false
        }
        return true
    }

    mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> AsyncStream<TokenGeneration>.Continuation.YieldResult
    ) {}

    func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}
