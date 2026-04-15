import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

// MARK: - WAN (Wan 2.1 / Wan 2.2) — Apple Silicon video generation
//
// End-to-end scaffold: scheduler → noise latent → WanDiT forward →
// WanVAEDecoder → MP4. The transformer velocity predictor and the VAE
// are module-complete (FP32 weights would load directly via
// WeightLoader) but ship initialized to random weights, so current
// output is a random noise MP4 — real weights decode into real video
// once we wire a safetensors loader for Wan's specific file layout.
//
// Architecture flow:
//   1. text → T5-XXL → (B, N_txt, 4096)     [STUB — currently zero tensor]
//   2. noise latent: (1, 16, T/4, H/8, W/8) via LatentSpace
//   3. patchify video latent: (1, N_vid, patch_t*patch_h*patch_w*16)
//   4. For each scheduler step:
//        velocity = WanDiTModel(video_patched, txt, t)
//        latent = scheduler.step(latent, velocity, i)
//        yield progress event
//   5. unpatchify latent back to (1, 16, T/4, H/8, W/8)
//   6. WanVAEDecoder(latent) → (1, 3, T, H, W) in [-1, 1]
//   7. Postprocess → [0, 1]
//   8. Write MP4 via WanVideoIO
//   9. Yield .completed
//
// This is the REAL pipeline — the only placeholders are the
// un-loaded weights inside WanDiTModel + WanVAEDecoder.

public final class WANModel: VideoGenerator, @unchecked Sendable {
    /// Wan 2.1 — the original 1.3B / 14B variants.
    public static let _registerWan21: Void = {
        ModelRegistry.register(ModelEntry(
            name: "wan-2.1",
            displayName: "Wan 2.1",
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            loader: { path, quant in
                _ = WANModel._registerWan21
                return try WANModel(modelPath: path, quantize: quant, version: .wan21)
            }
        ))
    }()

    /// Wan 2.2 — second generation with higher resolution defaults.
    public static let _registerWan22: Void = {
        ModelRegistry.register(ModelEntry(
            name: "wan-2.2",
            displayName: "Wan 2.2",
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            loader: { path, quant in
                _ = WANModel._registerWan22
                return try WANModel(modelPath: path, quantize: quant, version: .wan22)
            }
        ))
    }()

    public enum Version: Sendable { case wan21, wan22 }

    public let modelPath: URL
    public let quantize: Int?
    public let version: Version
    public let config: WanDiTConfig
    public let transformer: WanDiTModel
    public let vae: WanVAEDecoder
    public let loadedWeights: LoadedWeights

    public init(modelPath: URL, quantize: Int?, version: Version) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        self.version = version

        // Pick the right config for the version. 1.3B vs 14B is decided
        // at load time by inspecting the checkpoint file sizes; for now
        // we default to the larger 14B topology.
        self.config = {
            switch version {
            case .wan21: return .wan21_14B
            case .wan22: return .wan22
            }
        }()

        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }

        // Module construction. Real safetensors → module tree application
        // plugs in once the Wan-specific key mapping is written; for now
        // the modules initialize with random weights from their Linear
        // constructors so the forward pass compiles and runs.
        self.transformer = WanDiTModel(config: config)
        self.vae = WanVAEDecoder()

        // Eagerly scan the weights dir so JANG config / missing-shard
        // errors surface at `.load` time, same as ZImage.
        self.loadedWeights = try WeightLoader.load(from: modelPath)
    }

    public func generate(_ request: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    let msg = String(describing: error)
                    let hfAuth = msg.contains("401") || msg.contains("403")
                    continuation.yield(.failed(message: msg, hfAuth: hfAuth))
                    continuation.finish()
                }
            }
        }
    }

    private func performGenerate(
        _ request: VideoGenRequest,
        continuation: AsyncThrowingStream<VideoGenEvent, Error>.Continuation
    ) async throws {
        // 1. Scheduler. Video image-seq-len is (T/4) × (H/8 × W/8) patches,
        // typically much larger than still-image so the shift is saturated
        // at maxShift.
        let patchedSpatial = (request.width / 8 / config.patchSizeH)
            * (request.height / 8 / config.patchSizeW)
        let patchedTemporal = (request.numFrames / 4 / config.patchSizeT)
        let videoSeqLen = max(256, patchedSpatial * patchedTemporal)
        let scheduler = FlowMatchEulerScheduler(
            steps: request.steps,
            imageSeqLen: videoSeqLen,
            baseShift: 0.5,
            maxShift: 1.15
        )

        // 2. Noise latent: (1, 16, T/4, H/8, W/8). LatentSpace only
        // supports 2D layouts, so we allocate a flat (1, seq, 16)
        // and reshape.
        let tLatent = request.numFrames / 4
        let hLatent = request.height / 8
        let wLatent = request.width / 8
        if let seed = request.seed {
            MLXRandom.seed(seed)
        }
        var latent5D = MLXRandom.normal([1, 16, tLatent, hLatent, wLatent])

        // 3. Patchify: (B, 16, T, H, W) → (B, N_vid, patch_t*patch_h*patch_w*16).
        // For patch (1, 2, 2): reshape + transpose to collect each 1×2×2×16
        // block into one vector.
        func patchify(_ x: MLXArray) -> MLXArray {
            let pT = config.patchSizeT
            let pH = config.patchSizeH
            let pW = config.patchSizeW
            let b = x.dim(0)
            let c = x.dim(1)
            let t = x.dim(2)
            let h = x.dim(3)
            let w = x.dim(4)
            // (B, C, T, H, W) → (B, C, T/pT, pT, H/pH, pH, W/pW, pW)
            let r = x.reshaped([b, c, t / pT, pT, h / pH, pH, w / pW, pW])
            // → (B, T/pT, H/pH, W/pW, pT, pH, pW, C)
            let p = r.transposed(0, 2, 4, 6, 3, 5, 7, 1)
            // → (B, N, pT*pH*pW*C)
            let n = (t / pT) * (h / pH) * (w / pW)
            return p.reshaped([b, n, pT * pH * pW * c])
        }
        func unpatchify(_ x: MLXArray, t: Int, h: Int, w: Int) -> MLXArray {
            let pT = config.patchSizeT
            let pH = config.patchSizeH
            let pW = config.patchSizeW
            let b = x.dim(0)
            let c = 16
            // (B, N, pT*pH*pW*C) → (B, T/pT, H/pH, W/pW, pT, pH, pW, C)
            let r = x.reshaped([b, t / pT, h / pH, w / pW, pT, pH, pW, c])
            // → (B, C, T/pT, pT, H/pH, pH, W/pW, pW)
            let p = r.transposed(0, 7, 1, 4, 2, 5, 3, 6)
            return p.reshaped([b, c, t, h, w])
        }

        // 4. Text encoding. REAL T5-XXL Swift port is future work; for
        // now feed a zero tensor of the right shape. Wan uses ~256
        // max tokens for the text stream.
        let nTxt = 256
        let txt = MLXArray.zeros([1, nTxt, config.textDim])

        // 5. Sampling loop.
        let total = scheduler.stepCount
        let startedAt = Date()
        for step in 0..<total {
            if Task.isCancelled {
                continuation.yield(.cancelled)
                return
            }
            let patched = patchify(latent5D)
            let timestep = MLXArray([scheduler.timesteps[step]])
            let velocityPatched = transformer(
                videoPatched: patched,
                txt: txt,
                timestep: timestep
            )
            let velocity5D = unpatchify(velocityPatched, t: tLatent, h: hLatent, w: wLatent)
            // Euler step in 5D space — the scheduler only cares about
            // shape, not dimensionality.
            let sigmaCurrent = scheduler.sigmas[step]
            let sigmaNext = scheduler.sigmas[step + 1]
            let delta = sigmaNext - sigmaCurrent
            latent5D = latent5D + velocity5D * MLXArray(Float(delta))
            _ = latent5D.shape  // force eval

            let elapsed = Date().timeIntervalSince(startedAt)
            let perStep = elapsed / Double(step + 1)
            let eta = perStep * Double(total - step - 1)
            continuation.yield(.step(step: step + 1, total: total, etaSeconds: eta))
        }

        // 6. VAE decode.
        let rescaled = WanVAEDecoder.preprocessLatent(latent5D)
        let decoded = vae(rescaled)   // (1, 3, T, H, W) in ~[-1, 1]
        let processed = WanVAEDecoder.postprocess(decoded)

        // 7. Drop batch dim and write MP4 + frame PNG sequence.
        let video = processed[0]   // (3, T, H, W)
        try FileManager.default.createDirectory(
            at: request.outputDir, withIntermediateDirectories: true)
        let mp4URL = request.outputDir.appendingPathComponent(
            "wan-\(UUID().uuidString.prefix(8)).mp4"
        )
        try await MainActor.run {
            try WanVideoIO.writeMP4(video, outputURL: mp4URL, fps: request.fps)
        }

        let seed = request.seed ?? 0
        continuation.yield(.completed(
            url: mp4URL,
            seed: seed,
            fps: request.fps,
            frameCount: request.numFrames
        ))
    }
}

/// Force-register the video models. Call once at app launch.
public enum vMLXFluxVideo {
    public static func registerAll() {
        _ = WANModel._registerWan21
        _ = WANModel._registerWan22
    }
}
