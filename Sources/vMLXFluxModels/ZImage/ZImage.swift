import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

// Z-Image-Turbo — single-encoder turbo model, ~2B params, 4-8 steps.
// Python source: `mflux.models.z_image.variants.z_image.ZImage`.
//
// STATUS: end-to-end wiring + scheduler + weight loading + PNG output.
// The transformer velocity predictor is a PLACEHOLDER (returns a scaled
// noise field) until the DiT port lands. With this placeholder the UI
// gets real progress events, step counts match, and a valid PNG lands
// on disk. Replacing the velocity predictor with the real transformer
// forward pass is a localized change.

public final class ZImage: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "z-image-turbo",
            displayName: "Z-Image Turbo",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,
            supportsLoRA: false,
            loader: { path, quant in
                _ = ZImage._register
                return try ZImage(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    public let transformer: FluxDiTModel
    public let vae: VAEDecoder

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        // Eagerly load weights via the WeightLoader so we surface any
        // JANG config / missing-shard errors at `.load` time rather than
        // on the first generate call.
        self.loadedWeights = try WeightLoader.load(from: modelPath)

        // Z-Image Turbo — ~2B params, single-encoder. Uses the
        // `zImageTurbo` preset which is Flux-style but narrower and
        // shallower. Real hyperparams will come from parsing the
        // checkpoint's `config.json` when the per-model loader lands.
        self.transformer = FluxDiTModel(config: .zImageTurbo)
        // Flux-family VAE — shared with Flux1/Flux2/FIBO/Qwen.
        self.vae = VAEDecoder()
    }

    public func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    let message = String(describing: error)
                    let hfAuth = message.contains("401") || message.contains("403")
                    continuation.yield(.failed(message: message, hfAuth: hfAuth))
                    continuation.finish()
                }
            }
        }
    }

    private func performGenerate(
        _ request: ImageGenRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        // 1. Build the scheduler at the requested step count.
        let scheduler = FlowMatchEulerScheduler(
            steps: request.steps,
            imageSeqLen: (request.width / 8) * (request.height / 8),
            baseShift: 0.5,
            maxShift: 1.15
        )

        // 2. Allocate the initial noise latent. The Flux-family VAE
        // uses 16-channel latents (NOT 4 like SD/SDXL), and the DiT
        // transformer's `imgIn: Linear(patch²·16, dim)` hard-codes
        // that channel count. Sourcing from `transformer.config.inChannels`
        // keeps the two in sync and prevents a shape mismatch at
        // patchify time.
        var latent = LatentSpace.initialNoise(
            width: request.width,
            height: request.height,
            layout: .spatial(channels: transformer.config.inChannels),
            batchSize: 1,
            seed: request.seed
        )

        // 3. Text conditioning — REAL text encoders (T5-XXL + CLIP-L)
        // are a follow-up. For now we feed zero tensors of the right
        // shape so the transformer's `txtIn` / `vectorIn0` projections
        // receive the correct dtypes. Output is coherent noise until
        // real encoders land.
        let nTxt = 256
        let txtEmb = MLXArray.zeros([1, nTxt, transformer.config.textDim])
        let pooledClip = MLXArray.zeros([1, 768])

        // 4. Sampling loop — real FluxDiT forward pass per step.
        let total = scheduler.stepCount
        let startedAt = Date()
        for step in 0..<total {
            if Task.isCancelled {
                continuation.yield(.cancelled)
                return
            }
            // Patchify (B, 16, H/8, W/8) → (B, N, patch²·16).
            let imgPatched = patchify(
                latent,
                patchSize: transformer.config.patchSize,
                inChannels: transformer.config.inChannels
            )
            let timestep = MLXArray([scheduler.timesteps[step]])
            // Real transformer forward.
            let velocityPatched = transformer(
                imgPatched: imgPatched,
                txt: txtEmb,
                pooledClip: pooledClip,
                timestep: timestep,
                guidance: nil,
                rope: nil
            )
            // Unpatchify back to spatial (B, 16, H/8, W/8).
            let velocity = unpatchify(
                velocityPatched,
                patchSize: transformer.config.patchSize,
                outChannels: transformer.config.outChannels,
                height: request.height,
                width: request.width
            )
            latent = scheduler.step(
                latent: latent,
                velocity: velocity,
                stepIndex: step
            )
            // MLX is lazy — reading the shape forces materialization.
            _ = latent.shape

            let elapsed = Date().timeIntervalSince(startedAt)
            let perStep = elapsed / Double(step + 1)
            let eta = perStep * Double(total - step - 1)
            continuation.yield(.step(step: step + 1, total: total, etaSeconds: eta))
        }

        // 5. Decode latent → pixel space via the real Flux VAE decoder.
        // Flux applies a scale/shift before the decoder.
        let rescaled = VAEDecoder.preprocessFluxLatent(latent)
        let decoded = vae(rescaled)
        let image = VAEDecoder.postprocess(decoded)

        // 5. Write the PNG. `ImageIO.writePNG` is @MainActor so the
        // actor hop happens here.
        let outURL = try await MainActor.run {
            try ImageIO.writePNG(
                image,
                outputDir: request.outputDir,
                prefix: "z-image"
            )
        }
        let seed = request.seed ?? 0
        continuation.yield(.completed(url: outURL, seed: seed))
    }

    // MARK: - Placeholder math (to be replaced by real transformer + VAE)

    /// Placeholder velocity predictor. Returns a scaled noise field that
    /// drives the scheduler to a stable-but-meaningless converged state
    /// so the end-to-end plumbing is testable today.
    private func velocityPlaceholder(
        latent: MLXArray,
        stepIndex: Int,
        scheduler: FlowMatchEulerScheduler
    ) -> MLXArray {
        let noise = MLXRandom.normal(latent.shape)
        let sigmaDelta = scheduler.sigmas[stepIndex + 1] - scheduler.sigmas[stepIndex]
        return noise * MLXArray(Float(sigmaDelta * 0.1))
    }

    /// Placeholder VAE decoder. Converts a (B, 4, H/8, W/8) latent into a
    /// (B, 3, H, W) image by averaging channel pairs, tanh-squashing to
    /// [0, 1], and nearest-neighbor upsampling. The output is a blocky
    /// gradient — not the prompt, but a valid PNG that exercises every
    /// pipeline stage. Replaced by the real VAE decoder when weights land.
    private func vaeDecodePlaceholder(
        latent: MLXArray,
        targetWidth: Int,
        targetHeight: Int
    ) -> MLXArray {
        let b = latent.dim(0)
        let h = latent.dim(2)
        let w = latent.dim(3)
        let r = mean(latent[0..<b, 0..<2, 0..<h, 0..<w], axis: 1, keepDims: true)
        let g = mean(latent[0..<b, 1..<3, 0..<h, 0..<w], axis: 1, keepDims: true)
        let bl = mean(latent[0..<b, 2..<4, 0..<h, 0..<w], axis: 1, keepDims: true)
        let rgb = concatenated([r, g, bl], axis: 1)

        let squashed = (MLX.tanh(rgb) + MLXArray(Float(1))) * MLXArray(Float(0.5))
        let upsampled = upsampleNearest(squashed, factor: 8)
        return cropToSize(upsampled, height: targetHeight, width: targetWidth)
    }

    /// Nearest-neighbor upsample by an integer factor. (B, C, H, W) →
    /// (B, C, H*f, W*f). Uses `MLX.repeated(_:count:axis:)` along H and W.
    private func upsampleNearest(_ x: MLXArray, factor: Int) -> MLXArray {
        // Repeat along H (axis 2) then W (axis 3). Each repeat expands
        // that dimension by `factor` via nearest-neighbor duplication.
        let expandedH = repeated(x, count: factor, axis: 2)
        let expandedW = repeated(expandedH, count: factor, axis: 3)
        return expandedW
    }

    /// Crop an image tensor to exact (height, width). Takes top-left crop
    /// if the input is larger; smaller-than-target cases fall through
    /// since upsampleNearest already pads up to `factor * latentDim`.
    private func cropToSize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let currentH = x.dim(2)
        let currentW = x.dim(3)
        if currentH == height && currentW == width { return x }
        let h = min(currentH, height)
        let w = min(currentW, width)
        return x[0 ..< x.dim(0), 0 ..< x.dim(1), 0 ..< h, 0 ..< w]
    }
}
