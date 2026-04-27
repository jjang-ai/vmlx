import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import vMLXFluxKit

// MARK: - QwenImageEdit (Track 2 port)
//
// Alibaba's Qwen-Image-Edit. Joint conditioning via Qwen2-VL-7B (which
// processes the source image AND the prompt as a single multimodal
// sequence). The DiT then steers a fresh noise latent toward the
// resulting embeddings.
//
// Python source: `mflux/qwen_image/qwen_image_edit.py`. Key delta from
// Flux1 Kontext: the encoder is a single VL model that fuses image +
// text into one (B, N, D=3584) sequence, whereas Kontext keeps
// image-latent + T5 text in separate streams that get concatenated.
// Functionally similar — different feature space.
//
// Architecture sketch:
//
//     (source_image, prompt) ──► Qwen2-VL-7B ──► (B, N, 3584)
//     init_noise             ──► (B, 16, h, w)
//
//     for step in 0..<num_steps:
//         vec      = time_emb(step)
//         velocity = dit(noise_patched, qwen_vl_tokens, vec)
//         noise    = scheduler.step(noise, velocity, step)
//
//     output = VAE.decode(noise)
//
// Qwen-Image-Edit is full-precision only in mflux (~54GB VRAM). JANGTQ
// quant via the bundle's `jang_config.json` should bring this under
// 32GB; routed through `JangBridge` like every other Flux family port.
//
// Smoke test: similar to Kontext — edit cat→dog, structure preserved
// (correlation > 0.6) AND prompt-driven content delta detectable.

public final class QwenImageEditEditor: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image-edit",
            displayName: "Qwen-Image-Edit",
            kind: .imageEdit,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            supportsLoRA: false,
            isPlaceholder: true,
            loader: { path, quant in
                _ = QwenImageEditEditor._register
                return try QwenImageEditEditor(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    private var weights: LoadedWeights?

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
    }

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = try await self.runEdit(request) { step, total in
                        continuation.yield(.step(step: step, total: total, etaSeconds: nil))
                    }
                    continuation.yield(.completed(url: url, seed: request.seed ?? 0))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func runEdit(
        _ request: ImageEditRequest,
        progress: (Int, Int) -> Void
    ) async throws -> URL {
        if weights == nil {
            self.weights = try WeightLoader.load(from: modelPath)
        }

        // 1. Load source image at the requested working resolution.
        let width = request.width ?? 1024
        let height = request.height ?? 1024
        let srcImage = try await MainActor.run {
            try EditOps.loadImageRGB(url: request.sourceImage, width: width, height: height)
        }

        // 2. Encode (image + prompt) via Qwen2-VL-7B. Track 1 owns the
        //    real encoder. Until it lands we use MockTextEncoder with
        //    hiddenDim 3584 (Qwen2-VL-7B's hidden size). The image-half
        //    of the fusion is a no-op in the mock — real Qwen2-VL packs
        //    visual tokens into the same sequence via vision tower +
        //    merger projection.
        let mockEncoder = MockTextEncoder(hiddenSize: 3584, maxSeqLen: 512)
        let qwenTokens = mockEncoder.encodeFromString(prompt: request.prompt)
        let pooled = mockEncoder.encodePooledFromString(prompt: request.prompt)

        // 3. Encode source → latent (STUB until VAE encoder lands). For
        //    Qwen-Image-Edit the source is also fed to the DiT via the
        //    image stream conditioning, similar to Kontext.
        let srcLatent = EditOps.encodeToLatentStub(srcImage)
        _ = srcLatent

        // 4. Init noise.
        var noise = LatentSpace.initialNoise(
            width: width, height: height,
            layout: .spatial(channels: 16),
            batchSize: 1, seed: request.seed
        )

        // 5. Scheduler loop.
        let initStep = EditOps.initStepFromStrength(
            strength: request.strength, numSteps: request.steps
        )
        let scheduler = FlowMatchEulerScheduler(
            steps: request.steps,
            imageSeqLen: (height / 8) * (width / 8)
        )
        for step in initStep..<scheduler.steps {
            progress(step + 1, scheduler.steps)
            // Real impl:
            //   let velocity = dit.callAsFunction(
            //       imgPatched: patchify(noise, patchSize: 2, inChannels: 16),
            //       txt: qwenTokens,
            //       pooledClip: pooled,
            //       timestep: MLXArray(scheduler.timesteps[step]))
            //   noise = scheduler.step(latent: noise, velocity: velocity, stepIndex: step)
            _ = qwenTokens
            _ = pooled
            _ = noise
        }

        // 6. Decode. Stub: write source image until VAE decoder is fully
        //    wired with checkpoint weights. `isPlaceholder: true` warns.
        return try await MainActor.run {
            try ImageIO.writePNG(
                (srcImage + MLXArray(Float(1))) * MLXArray(Float(0.5)),
                outputDir: request.outputDir,
                prefix: "qwen-image-edit"
            )
        }
    }
}
