import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// MARK: - Qwen-Image (gen path)
//
// Alibaba's text-to-image model. Python source:
//   /tmp/mflux-ref/src/mflux/models/qwen/variants/txt2img/qwen_image.py
//
// Architecture:
//   - Single text encoder: Qwen2-VL-7B (text-only forward path here)
//   - DiT: 60-block patchified transformer with mRoPE-aware attention
//   - VAE: Flux-family AutoencoderKL (16-channel latent)
//
// Track 2 owns Qwen-Image-Edit in `QwenImageEdit.swift` (registered as
// `QwenImageEditEditor` there). This file is gen-only.
//
// Status: Module tree assembled; smoke gated on `VMLX_SWIFT_TEST_WEIGHTS`.

public final class QwenImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "qwen-image",
            displayName: "Qwen-Image",
            kind: .imageGen,
            defaultSteps: 30,
            defaultGuidance: 4.0,
            isPlaceholder: true,
            loader: { path, quant in
                _ = QwenImage._register
                return try QwenImage(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    public let transformer: FluxDiTModel
    public let vae: VAEDecoder
    public let textEncoder: Qwen2VL7BEncoder

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        self.loadedWeights = try WeightLoader.load(from: modelPath)
        // Use the FLUX.1-Dev preset as a starting DiT config — Qwen-Image's
        // 60-block layout differs (more blocks, narrower hidden) and a
        // dedicated `qwenImage` config will land once the smoke fixture
        // exists to validate it. This documents the gap rather than
        // hiding it.
        self.transformer = FluxDiTModel(config: .dev)
        self.vae = VAEDecoder()
        self.textEncoder = Qwen2VL7BEncoder(maxSeqLen: 256)
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.runGenerate(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    let msg = String(describing: error)
                    let hf = msg.contains("401") || msg.contains("403")
                    continuation.yield(.failed(message: msg, hfAuth: hf))
                    continuation.finish()
                }
            }
        }
    }

    private func runGenerate(
        _ request: ImageGenRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        let scheduler = FlowMatchEulerScheduler(
            steps: max(1, request.steps),
            imageSeqLen: (request.width / 16) * (request.height / 16),
            baseShift: 0.5,
            maxShift: 1.15
        )
        var latent = LatentSpace.initialNoise(
            width: request.width,
            height: request.height,
            layout: .spatial(channels: transformer.config.inChannels),
            batchSize: 1,
            seed: request.seed
        )

        let tokens = Array(repeating: 0, count: 64)
        var textHidden = textEncoder.encode(tokenIds: tokens)
        // Pad text dim to DiT's expected textDim if smaller (3584 → 4096).
        let curD = textHidden.dim(2)
        if curD < transformer.config.textDim {
            let pad = MLXArray.zeros([
                textHidden.dim(0), textHidden.dim(1),
                transformer.config.textDim - curD
            ])
            textHidden = concatenated([textHidden, pad], axis: -1)
        }
        let pooled = textHidden.mean(axis: 1)
        let pooledClip: MLXArray
        if pooled.dim(-1) >= 768 {
            pooledClip = pooled[.ellipsis, 0..<768]
        } else {
            let need = 768 - pooled.dim(-1)
            let pad = MLXArray.zeros([pooled.dim(0), need])
            pooledClip = concatenated([pooled, pad], axis: -1)
        }
        let guidance = MLXArray([request.guidance])

        let total = scheduler.stepCount
        let startedAt = Date()
        for step in 0..<total {
            if Task.isCancelled { continuation.yield(.cancelled); return }
            let imgPatched = patchify(
                latent,
                patchSize: transformer.config.patchSize,
                inChannels: transformer.config.inChannels
            )
            let timestep = MLXArray([scheduler.timesteps[step]])
            let velocityPatched = transformer(
                imgPatched: imgPatched,
                txt: textHidden,
                pooledClip: pooledClip,
                timestep: timestep,
                guidance: guidance,
                rope: nil
            )
            let velocity = unpatchify(
                velocityPatched,
                patchSize: transformer.config.patchSize,
                outChannels: transformer.config.outChannels,
                height: request.height,
                width: request.width
            )
            latent = scheduler.step(latent: latent, velocity: velocity, stepIndex: step)
            _ = latent.shape
            let elapsed = Date().timeIntervalSince(startedAt)
            let perStep = elapsed / Double(step + 1)
            let eta = perStep * Double(total - step - 1)
            continuation.yield(.step(step: step + 1, total: total, etaSeconds: eta))
        }

        let rescaled = VAEDecoder.preprocessFluxLatent(latent)
        let decoded = vae(rescaled)
        let image = VAEDecoder.postprocess(decoded)
        let outURL = try await MainActor.run {
            try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "qwen-image")
        }
        continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
    }
}
