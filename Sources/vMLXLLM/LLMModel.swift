// Copyright © 2024 Apple Inc.

import MLX
import vMLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// Walks the prompt in chunks until a small number of tokens remain,
    /// which are then fed into the `TokenIterator`.
    ///
    /// **Rank handling** (fixes upstream mlx-swift-examples bug 5b26831):
    /// Callers pass tokens as 1D `[T]` OR 2D `[1, T]` depending on path —
    /// `TokenIterator.prepare()` and the batched engine both hand in 2D
    /// `[1, T]`. The old chunked slicing code used
    ///
    ///     let input = y[.newAxis, ..<prefillStepSize]
    ///     y = y[prefillStepSize...]
    ///
    /// which is correct for 1D input but silently corrupts 2D input:
    /// variadic subscript applies `.newAxis` first, so the `..<N`
    /// subscript lands on the *batch* axis instead of the token axis,
    /// the chunk is not sliced at all, and `y[N...]` then slices the
    /// batch axis (size 1) from N → produces an empty tensor. The next
    /// forward pass then crashes with `[reshape] Cannot infer the shape
    /// of an empty array`.
    ///
    /// Fix: flatten tokens to 1D at entry, perform the chunked walk on
    /// the flat view (dimension-independent), then reshape back to the
    /// original rank on exit so the downstream `TokenIterator` sees the
    /// same shape it passed in.
    ///
    /// Regression tests: Qwen3.5-35B-A3B JANG_2S-TEXT (LLM path) at
    /// prompt_len 1394 + prefillStep 512. VLM Qwen35 has its own
    /// `prepare()` so that path never hit the bug; the LLM-only variant
    /// (vision_config stripped) was the only crash surface.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        let originalShape = input.text.tokens.shape

        // Flatten to 1D for dimension-independent slicing. If the input
        // was already 1D, `reshape([-1])` is a no-op. For 2D `[1, T]`
        // it produces `[T]`; for any other rank it flattens to `[total]`.
        let flat = input.text.tokens.reshaped([-1])
        var y = LMInput.Text(tokens: flat, mask: input.text.mask)

        while y.tokens.size > prefillStepSize {
            // `[.newAxis, ..<N]` now correctly chunks the flat token
            // array: `.newAxis` lifts to `[1, size]`, then `..<N` slices
            // the token axis. Batched callers that need [B, T] come
            // through the VLM / batched-engine prepare() paths and do
            // NOT reach this default extension.
            let chunk = y[.newAxis, ..<prefillStepSize]
            _ = self(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
            MLX.eval(cache)
            y = y[prefillStepSize...]
            Memory.clearCache()
        }

        // Reshape the remaining tail back to the caller's original rank
        // so the downstream TokenIterator sees a consistent shape. For
        // 1D callers this is a no-op; for 2D `[1, T]` callers we restore
        // the leading batch axis.
        let tailReshaped: MLXArray
        if originalShape.count >= 2 {
            // Preserve all non-token leading axes (almost always `[1]`).
            let leading = Array(originalShape.dropLast())
            tailReshaped = y.tokens.reshaped(leading + [y.tokens.size])
        } else {
            tailReshaped = y.tokens
        }
        return .tokens(LMInput.Text(tokens: tailReshaped, mask: y.mask))
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
