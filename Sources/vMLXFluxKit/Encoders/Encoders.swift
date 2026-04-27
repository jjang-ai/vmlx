import Foundation
@preconcurrency import MLX

// MARK: - Text encoder protocol
//
// Common surface for the three text encoders Track 1 ports: T5-XXL (used
// by FLUX.1 Schnell/Dev + FLUX.1 Fill/Kontext + FIBO), CLIP-L (FLUX.1
// pooled conditioning), and Qwen2-VL-7B (Qwen-Image + Qwen-Image-Edit
// + FLUX.2 Klein single-encoder mode).
//
// Z-Image uses its own Qwen2.5-style encoder with a custom RoPE; that
// lives at `vMLXFluxModels/ZImage/ZImage.swift` since it's
// model-specific. The three exports here are the cross-model encoders.
//
// Every encoder takes a tokenized prompt as `[Int]` token IDs (caller
// runs the swift-transformers tokenizer first) and returns a
// `(B, seqLen, hiddenDim)` MLXArray ready to feed into the DiT.
//
// Status (2026-04-27, Track 1):
//   - Module trees declare the layer shapes faithfully (matching mflux's
//     Python construction). Forward passes are ported but rely on the
//     DiT to load weights via `WeightLoader.load(from:)` then apply
//     them via `Module.update(parameters:)`.
//   - Tokenization is delegated to swift-transformers' `AutoTokenizer`
//     loaded from the `tokenizer/` or `text_encoder*/` HuggingFace
//     subdirectories of the model snapshot.
//   - `isPlaceholder: true` stays on every model in the registry until
//     a runtime smoke test (env `VMLX_SWIFT_TEST_WEIGHTS`) verifies the
//     encoder + DiT + VAE roundtrip produces non-noise pixels.

/// Common surface for FLUX-family text encoders.
///
/// `encode(_:)` takes pre-tokenized integer IDs and produces hidden
/// states. The DiT consumer is responsible for projecting these into
/// its own embedding dim via the `context_embedder` / `txt_in` linear.
///
/// Track 2's `EditOps.swift` has a `MockTextEncoder` that conforms to
/// this same protocol via Track 1 ownership of `Encoders.swift`. Both
/// the real encoders below and the mock in EditOps share `TextEncoder`.
public protocol TextEncoder: Sendable {
    /// Hidden size of the encoder's output. Used by DiTs to size their
    /// `txt_in` / `context_embedder` projection.
    var hiddenSize: Int { get }

    /// Maximum sequence length supported. T5-XXL uses 256/512, CLIP-L
    /// uses 77, Qwen2-VL-7B uses 1024+ depending on the variant.
    var maxSeqLen: Int { get }

    /// Encode a tokenized prompt to hidden states. The default batch
    /// size is 1 — multi-image generation just calls multiple times.
    /// Returns a `(1, S, hiddenSize)` MLXArray where `S <= maxSeqLen`.
    func encode(tokenIds: [Int]) -> MLXArray
}

/// Pooled-output addendum for CLIP-L. FLUX.1 takes the pooled output of
/// the CLIP encoder (the `[EOS]` position's projected hidden state) as
/// its 768-dim conditioning vector. Non-pooled encoders (T5, Qwen2-VL)
/// don't conform.
public protocol PooledTextEncoder: TextEncoder {
    /// Hidden size of the pooled output (typically 768 for CLIP-L).
    var pooledSize: Int { get }
    /// Encode and also return the pooled output `(1, pooledSize)`.
    func encodePooled(tokenIds: [Int]) -> (hidden: MLXArray, pooled: MLXArray)
}

/// Errors specific to text encoder construction.
public enum EncoderError: Error, CustomStringConvertible {
    case missingWeights(name: String)
    case invalidConfig(reason: String)
    case unsupportedShape(name: String, expected: String, actual: String)

    public var description: String {
        switch self {
        case .missingWeights(let n): return "missing encoder weights: \(n)"
        case .invalidConfig(let r):  return "invalid encoder config: \(r)"
        case .unsupportedShape(let n, let e, let a):
            return "encoder \(n) shape mismatch: expected \(e), got \(a)"
        }
    }
}
