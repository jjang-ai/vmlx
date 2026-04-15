// SPDX-License-Identifier: Apache-2.0
//
// Shared chunked-prefill helper for VLM `prepare(_:cache:windowSize:)`.
// Before this, every VLM model invoked its language model once on the
// full input embedding tensor, e.g.
//
//     let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)
//     return .logits(result)
//
// For large-image prompts (100k+ token embeddings) this blew past the
// Metal single-buffer cap on large MoE models. `LLMModel`'s default
// `prepare` already chunks by `prefillStepSize` for text-only paths;
// this helper gives VLM paths the same loop without duplicating it
// in every model file.
//
// Fixes the VLM half of vmlx #50/#51 — the LLM side was already handled
// by `LLMModel.prepare`.

import Foundation
import MLX

/// Run a VLM's language-model forward pass over an input embedding in
/// `prefillStepSize`-sized chunks along the sequence dimension (axis 1).
///
/// For N chunks:
/// - Chunks 0..N-2 are called for side effect (cache update) and their
///   result is discarded. `MLX.eval(cache)` + `MLX.GPU.clearCache()`
///   run between chunks to match Python mlx-lm / `LLMModel.prepare`
///   behavior, which is critical for MoE models on Apple Silicon.
/// - The final chunk's result is returned so the caller can produce
///   `.logits(result)` for the generate loop's first token sample.
///
/// If `prefillStepSize <= 0` or the embedding's seq dimension is already
/// ≤ `prefillStepSize`, the step is invoked exactly once with the full
/// embedding, matching the pre-chunking behavior.
///
/// - Parameters:
///   - inputEmbedding: The full `[B, T, D]` input embedding (post vision
///     fusion). Only axis 1 (`T`) is sliced.
///   - cache: The model's per-layer KV cache. Passed to `MLX.eval` between
///     chunks so each pass materializes before the next slice is dispatched.
///   - prefillStepSize: The chunk size along the sequence dimension. Use
///     `windowSize ?? 512` in your VLM `prepare` to match `LLMModel.prepare`'s
///     default.
///   - step: Closure that invokes the VLM's language model on a single
///     chunk of the embedding. Returns whatever the VLM's language model
///     returns (`LMOutput`, `MLXArray`, etc. — generic).
/// - Returns: The result of calling `step` on the *final* chunk.
@discardableResult
public func chunkedPrefillEmbedding<Result>(
    inputEmbedding: MLXArray,
    cache: [KVCache],
    prefillStepSize: Int,
    step: (MLXArray) -> Result
) -> Result {
    let T = inputEmbedding.dim(1)
    guard prefillStepSize > 0, T > prefillStepSize else {
        return step(inputEmbedding)
    }

    var offset = 0
    while offset + prefillStepSize < T {
        let end = offset + prefillStepSize
        let chunk = inputEmbedding[0..., offset ..< end, 0...]
        _ = step(chunk)
        MLX.eval(cache)
        offset = end
        MLX.GPU.clearCache()
    }

    // Final chunk: keep the result so the caller can sample from it.
    let finalChunk = inputEmbedding[0..., offset..., 0...]
    return step(finalChunk)
}
