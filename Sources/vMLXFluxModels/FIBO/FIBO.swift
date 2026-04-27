import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// MARK: - FIBO
//
// Python source: /tmp/mflux-ref/src/mflux/models/fibo/variants/txt2img/fibo.py
//
// Architecture:
//   - Text encoder: SmolLM3-3B (a 3B param Llama-style decoder used as
//     a unidirectional encoder). NOT T5, NOT CLIP, NOT Qwen. Track 1's
//     three encoders don't fit; SmolLM3 is its own port.
//   - DiT: Flux-style transformer with FIBO-specific block counts.
//   - VAE: Flux-family AutoencoderKL.
//
// Status: SmolLM3-3B is NOT in Track 1's spec'd encoder list (T5-XXL,
// CLIP-L, Qwen2-VL-7B). Per spec rule #3 (no model substitution), FIBO
// is intentionally NOT swapped to a T5-based stand-in. Stays
// `notImplemented` until SmolLM3-3B is ported in a follow-up. The
// registry surface (`fibo` model name) remains so the UI doesn't lose
// the entry.

public final class FIBO: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "fibo",
            displayName: "FIBO",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 3.5,
            isPlaceholder: true,
            loader: { path, quant in
                _ = FIBO._register
                return try FIBO(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "FIBO requires the SmolLM3-3B text encoder which Track 1 doesn't own (Track 1 ports T5-XXL, CLIP-L, Qwen2-VL-7B). FIBO will land in a follow-up after SmolLM3-3B is ported. Reference: mflux/models/fibo/model/fibo_text_encoder/smol_lm3_3b_text_encoder.py."))
        }
    }
}
