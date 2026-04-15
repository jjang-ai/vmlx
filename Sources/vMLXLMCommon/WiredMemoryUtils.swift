// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Result of a wired memory measurement pass.
public struct WiredMemoryMeasurement: Sendable {
    /// Total bytes for model weights (`nbytes` sum).
    public let weightBytes: Int
    /// Total bytes for KV caches after prefill.
    public let kvBytes: Int
    /// Estimated transient workspace bytes (prefill peak minus weights + KV).
    public let workspaceBytes: Int
    /// Peak active memory observed during prefill.
    public let peakActiveBytes: Int
    /// Number of tokens used during the prefill measurement.
    public let tokenCount: Int
    /// Prefill step size used for the measurement.
    public let prefillStepSize: Int

    /// Combined budget suggestion (weights + KV + workspace).
    public var totalBytes: Int {
        max(0, weightBytes) + max(0, kvBytes) + max(0, workspaceBytes)
    }
}

/// Helpers for deriving wired memory budgets from real runtime measurements.
public enum WiredMemoryUtils {
    /// Produce a token ID array of exactly `count` tokens using the given tokenizer.
    ///
    /// This does not attempt to generate semantically meaningful text; it only ensures
    /// a valid token sequence of the requested length for memory sizing purposes.
    private static func makeTokenIds(
        count: Int,
        tokenizer: Tokenizer,
        seedText: String = " hello"
    ) -> [Int] {
        guard count > 0 else { return [] }

        let pad = tokenizer.eosTokenId ?? tokenizer.unknownTokenId ?? 0
        var tokens: [Int] = []

        var chunk = seedText
        while tokens.count < count {
            let newTokens = tokenizer.encode(text: chunk)
            if newTokens.isEmpty {
                tokens.append(pad)
            } else {
                tokens.append(contentsOf: newTokens)
            }
            if tokens.count < count {
                chunk += seedText
            }
        }

        if tokens.count > count {
            tokens = Array(tokens.prefix(count))
        }

        if tokens.count < count {
            tokens.append(contentsOf: repeatElement(pad, count: count - tokens.count))
        }

        return tokens
    }

    /// Create a minimal `LMInput` with exactly `count` tokens.
    ///
    /// - Note: This is intended for text-only models. Multimodal models should
    ///   supply a fully prepared `LMInput` via their processor instead.
    private static func makeTokenInput(
        count: Int,
        tokenizer: Tokenizer,
        seedText: String = " hello"
    ) -> LMInput {
        let tokenIds = makeTokenIds(count: count, tokenizer: tokenizer, seedText: seedText)
        return LMInput(tokens: MLXArray(tokenIds))
    }

    /// Run a prefill-only pass to populate caches for the given input.
    ///
    /// This mirrors the prefill path used by `TokenIterator` without generating
    /// additional tokens. It forces evaluation to ensure allocations are realized.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (text-only or multimodal).
    ///   - model: The language model to prefill.
    ///   - parameters: Generation parameters that control prefill behavior.
    /// - Returns: The cache array after prefill, suitable for KV sizing.
    private static func prefillOnly(
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> [KVCache] {
        var cache = model.newCache(parameters: parameters)

        switch try model.prepare(input, cache: cache, windowSize: parameters.prefillStepSize) {
        case .tokens(let tokens):
            let result = model(
                tokens[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
            eval(result.logits)
        case .logits(let result):
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
            eval(result.logits)
        }

        return cache
    }

    /// Measure weights, KV cache, and prefill workspace for the given model context.
    ///
    /// This is a diagnostic helper intended to **measure** real memory usage
    /// rather than assume it. The returned values are best used to construct
    /// a ticket budget or to compare against manual estimates.
    ///
    /// - Important: `Memory.peakMemory` is global. For accurate results, run
    ///   this in isolation (no concurrent inference).
    ///
    /// - Note: This uses the tokenizer directly to build a synthetic prompt and
    ///   is best suited for text-only models. For multimodal models, use the
    ///   overload that accepts a prepared `LMInput`.
    public static func tune(
        context: ModelContext,
        tokenCount: Int,
        parameters: GenerateParameters,
        seedText: String = " hello",
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let weights = context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        let input = makeTokenInput(
            count: tokenCount,
            tokenizer: context.tokenizer,
            seedText: seedText
        )

        let startActive = Memory.activeMemory
        if resetPeakMemory {
            Memory.peakMemory = 0
        }

        let cache = try prefillOnly(input: input, model: context.model, parameters: parameters)
        let cacheArrays = cache.flatMap { $0.state }
        if !cacheArrays.isEmpty {
            eval(cacheArrays)
        }

        let kvBytes = cacheArrays.reduce(0) { $0 + $1.nbytes }
        let peakActive = max(Memory.peakMemory, startActive)
        let workspace = max(0, peakActive - weights - kvBytes)

        return WiredMemoryMeasurement(
            weightBytes: weights,
            kvBytes: kvBytes,
            workspaceBytes: workspace,
            peakActiveBytes: peakActive,
            tokenCount: tokenCount,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    /// Measure weights, KV cache, and prefill workspace using a prepared input.
    ///
    /// This overload is recommended for multimodal models because it accepts a
    /// fully prepared `LMInput` (e.g. with image/video tensors already embedded).
    ///
    /// - Parameters:
    ///   - input: Prepared model input from the model's processor.
    ///   - context: The loaded model context.
    ///   - parameters: Generation parameters that control prefill behavior.
    ///   - resetPeakMemory: If true, resets `Memory.peakMemory` before measuring.
    /// - Returns: A measurement snapshot for weights, KV, and workspace.
    public static func tune(
        input: LMInput,
        context: ModelContext,
        parameters: GenerateParameters,
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let weights = context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }

        let startActive = Memory.activeMemory
        if resetPeakMemory {
            Memory.peakMemory = 0
        }

        let cache = try prefillOnly(input: input, model: context.model, parameters: parameters)
        let cacheArrays = cache.flatMap { $0.state }
        if !cacheArrays.isEmpty {
            eval(cacheArrays)
        }

        let kvBytes = cacheArrays.reduce(0) { $0 + $1.nbytes }
        let peakActive = max(Memory.peakMemory, startActive)
        let workspace = max(0, peakActive - weights - kvBytes)

        return WiredMemoryMeasurement(
            weightBytes: weights,
            kvBytes: kvBytes,
            workspaceBytes: workspace,
            peakActiveBytes: peakActive,
            tokenCount: input.text.tokens.size,
            prefillStepSize: parameters.prefillStepSize
        )
    }

    /// Measure weights, KV cache, and prefill workspace using a user input.
    ///
    /// This is a convenience wrapper that runs the model's processor to build a
    /// prepared `LMInput`, then delegates to the `tune(input:context:parameters:)`
    /// overload. It is especially useful for VLMs where images or videos are part
    /// of the input and significantly affect memory usage.
    ///
    /// - Parameters:
    ///   - userInput: High-level input (text/images/video) to be prepared.
    ///   - context: The loaded model context.
    ///   - parameters: Generation parameters that control prefill behavior.
    ///   - resetPeakMemory: If true, resets `Memory.peakMemory` before measuring.
    /// - Returns: A measurement snapshot for weights, KV, and workspace.
    public static func tune(
        userInput: UserInput,
        context: ModelContext,
        parameters: GenerateParameters,
        resetPeakMemory: Bool = true
    ) async throws -> WiredMemoryMeasurement {
        let prepared = try await context.processor.prepare(input: userInput)
        return try await tune(
            input: prepared,
            context: context,
            parameters: parameters,
            resetPeakMemory: resetPeakMemory
        )
    }
}
