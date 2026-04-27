import Foundation
@preconcurrency import MLX
import MLXRandom
import vMLXFluxKit

// MARK: - WAN (Wan 2.1 / Wan 2.2) — Apple Silicon video generation
//
// End-to-end pipeline:
//   1. text → UMT5 encoder → (B, text_len, 4096)
//   2. text → text_embedding MLP (in WanDiTModel) → (B, text_len, dim)
//   3. noise latent: (1, 16, T/4, H/8, W/8)
//   4. patchify video latent: (1, N, patch_t*patch_h*patch_w*16)
//   5. for each step: pick high vs low-noise expert via WanMoE,
//      compute time conditioning, run transformer + 3D RoPE, Euler step
//   6. unpatchify back to (1, 16, T/4, H/8, W/8)
//   7. WanVAEDecoder → (1, 3, T, H, W) in [-1, 1]
//   8. Postprocess → [0, 1] → MP4 via WanVideoIO
//
// STATUS: Real architecture wired end-to-end; weights shipped with
// random init until proven against on-disk checkpoints. Every variant
// registers as `isPlaceholder: true` per spec rule 5 — flips to false
// in the same commit that proves consecutive-frame correlation > 0.5.

public final class WANModel: VideoGenerator, @unchecked Sendable {

    // MARK: Variants

    public enum Variant: String, Sendable, CaseIterable {
        case wan21_1_3B = "wan2.1-t2v-1.3b"
        case wan21_14B = "wan2.1-t2v-14b"
        case wan22_t2v_14B = "wan2.2-t2v-14b"
        case wan22_ti2v_5B = "wan2.2-ti2v-5b"
        case wan22_i2v_14B = "wan2.2-i2v-14b"

        public var displayName: String {
            switch self {
            case .wan21_1_3B:    return "Wan 2.1 T2V 1.3B"
            case .wan21_14B:     return "Wan 2.1 T2V 14B"
            case .wan22_t2v_14B: return "Wan 2.2 T2V 14B (MoE)"
            case .wan22_ti2v_5B: return "Wan 2.2 TI2V 5B"
            case .wan22_i2v_14B: return "Wan 2.2 I2V 14B (MoE)"
            }
        }

        public var ditConfig: WanDiTConfig {
            switch self {
            case .wan21_1_3B:    return .wan21_1_3B
            case .wan21_14B:     return .wan21_14B
            case .wan22_t2v_14B: return .wan22_t2v_14B
            case .wan22_ti2v_5B: return .wan22_ti2v_5B
            case .wan22_i2v_14B: return .wan22_i2v_14B
            }
        }

        /// I2V variants need an init image as conditioning.
        public var requiresInputImage: Bool {
            self == .wan22_i2v_14B
        }

        /// True for variants we couldn't yet validate end-to-end against
        /// real weights (every variant today). Flips to false when the
        /// smoke test confirms consecutive-frame correlation > 0.5 on a
        /// real checkpoint.
        public var isPlaceholder: Bool { true }
    }

    // MARK: Registry — 5 variants

    public static let _registerWan21_1_3B: Void = {
        ModelRegistry.register(ModelEntry(
            name: Variant.wan21_1_3B.rawValue,
            displayName: Variant.wan21_1_3B.displayName,
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            isPlaceholder: Variant.wan21_1_3B.isPlaceholder,
            loader: { path, quant in
                _ = WANModel._registerWan21_1_3B
                return try WANModel(modelPath: path, quantize: quant, variant: .wan21_1_3B)
            }
        ))
    }()

    public static let _registerWan21_14B: Void = {
        ModelRegistry.register(ModelEntry(
            name: Variant.wan21_14B.rawValue,
            displayName: Variant.wan21_14B.displayName,
            kind: .videoGen,
            defaultSteps: 50,
            defaultGuidance: 5.0,
            isPlaceholder: Variant.wan21_14B.isPlaceholder,
            loader: { path, quant in
                _ = WANModel._registerWan21_14B
                return try WANModel(modelPath: path, quantize: quant, variant: .wan21_14B)
            }
        ))
    }()

    public static let _registerWan22_t2v_14B: Void = {
        ModelRegistry.register(ModelEntry(
            name: Variant.wan22_t2v_14B.rawValue,
            displayName: Variant.wan22_t2v_14B.displayName,
            kind: .videoGen,
            defaultSteps: 40,
            defaultGuidance: 4.0,
            isPlaceholder: Variant.wan22_t2v_14B.isPlaceholder,
            loader: { path, quant in
                _ = WANModel._registerWan22_t2v_14B
                return try WANModel(modelPath: path, quantize: quant, variant: .wan22_t2v_14B)
            }
        ))
    }()

    public static let _registerWan22_ti2v_5B: Void = {
        ModelRegistry.register(ModelEntry(
            name: Variant.wan22_ti2v_5B.rawValue,
            displayName: Variant.wan22_ti2v_5B.displayName,
            kind: .videoGen,
            defaultSteps: 40,
            defaultGuidance: 5.0,
            isPlaceholder: Variant.wan22_ti2v_5B.isPlaceholder,
            loader: { path, quant in
                _ = WANModel._registerWan22_ti2v_5B
                return try WANModel(modelPath: path, quantize: quant, variant: .wan22_ti2v_5B)
            }
        ))
    }()

    public static let _registerWan22_i2v_14B: Void = {
        ModelRegistry.register(ModelEntry(
            name: Variant.wan22_i2v_14B.rawValue,
            displayName: Variant.wan22_i2v_14B.displayName,
            kind: .videoGen,
            defaultSteps: 40,
            defaultGuidance: 3.5,
            isPlaceholder: Variant.wan22_i2v_14B.isPlaceholder,
            loader: { path, quant in
                _ = WANModel._registerWan22_i2v_14B
                return try WANModel(modelPath: path, quantize: quant, variant: .wan22_i2v_14B)
            }
        ))
    }()

    // MARK: Stored state

    public let modelPath: URL
    public let quantize: Int?
    public let variant: Variant
    public let config: WanDiTConfig

    /// Single transformer (Wan 2.1 + Wan 2.2 TI2V-5B) OR low-noise
    /// transformer (Wan 2.2 dual variants).
    public let transformerLow: WanDiTModel
    /// High-noise transformer for dual-model variants. Nil for single-model.
    public let transformerHigh: WanDiTModel?

    public let textEncoder: UMT5Encoder
    public let vae: WanVAEDecoder
    public let moe: WanMoE
    public let loadedWeights: LoadedWeights

    public init(modelPath: URL, quantize: Int?, variant: Variant) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        self.variant = variant
        self.config = variant.ditConfig

        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }

        self.transformerLow = WanDiTModel(config: config)
        self.transformerHigh = config.dualModel ? WanDiTModel(config: config) : nil
        self.textEncoder = UMT5Encoder(config: .umt5XXL, sharedPos: false)
        self.vae = WanVAEDecoder()
        self.moe = WanMoE.fromConfig(
            dualModel: config.dualModel, boundary: config.boundary
        )

        // Eagerly enumerate weights so JANG-config + missing-shard
        // failures surface at .load time. The actual `applyWeights` call
        // happens lazily on first forward — keeps load time fast for the
        // UI's "added to library" flow.
        self.loadedWeights = try WeightLoader.load(from: modelPath)
    }

    // MARK: Public API

    public func generate(_ request: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(request, inputImage: nil, continuation: continuation)
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

    /// I2V entry — accepts an init image URL alongside the text prompt.
    /// Used by Wan 2.2 I2V-14B and TI2V-5B.
    public func generate(
        _ request: VideoGenRequest, inputImage: URL?
    ) -> AsyncThrowingStream<VideoGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(
                        request, inputImage: inputImage, continuation: continuation
                    )
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

    // MARK: Internal generate

    private func performGenerate(
        _ request: VideoGenRequest,
        inputImage: URL?,
        continuation: AsyncThrowingStream<VideoGenEvent, Error>.Continuation
    ) async throws {
        // 1. Scheduler.
        let patchedSpatial = (request.width / 8 / config.patchSizeH)
            * (request.height / 8 / config.patchSizeW)
        let patchedTemporal = max(1, (request.numFrames / 4 / config.patchSizeT))
        let videoSeqLen = max(256, patchedSpatial * patchedTemporal)
        let scheduler = FlowMatchEulerScheduler(
            steps: request.steps,
            imageSeqLen: videoSeqLen,
            baseShift: 0.5,
            maxShift: 1.15
        )

        // 2. Latent. (1, in_dim, T/4, H/8, W/8). For TI2V-5B we use
        // vae_stride=(4,16,16) so spatial divides by 16, but the user-
        // facing API still expects width/height divisible by 8 — the
        // patchify step handles the rest internally.
        let tLatent = max(1, request.numFrames / 4)
        let hLatent = request.height / 8
        let wLatent = request.width / 8
        if let seed = request.seed { MLXRandom.seed(seed) }
        var latent5D = MLXRandom.normal([
            1, config.inChannels, tLatent, hLatent, wLatent
        ])

        // 3. Image conditioning (I2V variants). The init image is decoded
        // through `WanVideoIO.decodeInputImage` and channel-concatenated
        // into the latent (Python `WanModel.__call__` `y` argument). The
        // final concat happens *inside* the patchify step.
        var imageCond: MLXArray? = nil
        if variant.requiresInputImage {
            guard let img = inputImage else {
                throw FluxError.invalidRequest(
                    "\(variant.displayName) requires an input image"
                )
            }
            imageCond = try await MainActor.run {
                try WanVideoIO.decodeInputImage(
                    at: img,
                    width: request.width,
                    height: request.height,
                    numFrames: request.numFrames,
                    inChannels: config.inChannels - 16  // 36-16=20 for I2V-14B
                )
            }
        }

        // 4. Patchify.
        func patchify(_ x: MLXArray) -> MLXArray {
            let pT = config.patchSizeT, pH = config.patchSizeH, pW = config.patchSizeW
            let b = x.dim(0), c = x.dim(1), t = x.dim(2), h = x.dim(3), w = x.dim(4)
            let r = x.reshaped([b, c, t / pT, pT, h / pH, pH, w / pW, pW])
            let p = r.transposed(0, 2, 4, 6, 3, 5, 7, 1)
            let n = (t / pT) * (h / pH) * (w / pW)
            return p.reshaped([b, n, pT * pH * pW * c])
        }
        func unpatchify(_ x: MLXArray, t: Int, h: Int, w: Int) -> MLXArray {
            let pT = config.patchSizeT, pH = config.patchSizeH, pW = config.patchSizeW
            let b = x.dim(0)
            let c = config.outChannels
            let r = x.reshaped([b, t / pT, h / pH, w / pW, pT, pH, pW, c])
            let p = r.transposed(0, 7, 1, 4, 2, 5, 3, 6)
            return p.reshaped([b, c, t, h, w])
        }

        // 5. Text encoding via UMT5. Real tokenizer/embedding pipeline
        // requires a SentencePiece tokenizer for UMT5 — until that's
        // bundled, feed a zero-id batch of length 512 so the architecture
        // exercises every layer. Production: run UMT5 tokenizer here.
        let textIds = MLXArray.zeros([1, config.textLen]).asType(.int32)
        let textHidden4096 = textEncoder(ids: textIds)        // (1, 512, 4096)

        // 6. Pre-project text in BOTH transformers (low + optional high).
        // Cheap call — just two Linear + GELU.
        let txtLow = transformerLow.encodeText(textHidden4096)
        let txtHigh = transformerHigh?.encodeText(textHidden4096)

        // 7. Precompute 3D RoPE for the constant grid.
        let headDim = config.dim / config.numHeads
        let rope3D = WanRoPE3D(
            headDim: headDim,
            t: tLatent / config.patchSizeT,
            h: hLatent / config.patchSizeH,
            w: wLatent / config.patchSizeW
        )

        // 8. Sampling loop.
        let total = scheduler.stepCount
        let startedAt = Date()
        for step in 0..<total {
            if Task.isCancelled {
                continuation.yield(.cancelled)
                return
            }

            // Channel-concat image conditioning if I2V.
            let latentForStep: MLXArray
            if let cond = imageCond {
                latentForStep = concatenated([latent5D, cond], axis: 1)
            } else {
                latentForStep = latent5D
            }

            let patched = patchify(latentForStep)
            let timestep = MLXArray([scheduler.timesteps[step]])

            // Pick expert via WanMoE.
            let stepFraction = Float(step) / Float(max(1, total))
            let useHigh = moe.useHighNoise(stepFraction: stepFraction)
            let activeTransformer: WanDiTModel = useHigh
                ? (transformerHigh ?? transformerLow)
                : transformerLow
            let activeTxt: MLXArray = useHigh
                ? (txtHigh ?? txtLow)
                : txtLow

            // Time conditioning (per-transformer because the time MLP
            // weights differ between high/low experts).
            let (eHead, e0) = activeTransformer.timeConditioning(timestep: timestep)
            let velPatched = activeTransformer.forward(
                videoPatched: patched,
                textHidden: activeTxt,
                e0: e0,
                eHead: eHead,
                rope3D: rope3D
            )

            // Out has out_channels (= 16 for T2V/I2V, 48 for TI2V-5B).
            let velocity5D = unpatchify(
                velPatched, t: tLatent, h: hLatent, w: wLatent
            )

            // For I2V we predict velocity for the noise channels only.
            let velocityNoise: MLXArray
            if imageCond != nil {
                velocityNoise = velocity5D[
                    0..., 0 ..< 16, 0..., 0..., 0...
                ]
            } else {
                velocityNoise = velocity5D
            }

            let sigmaCurrent = scheduler.sigmas[step]
            let sigmaNext = scheduler.sigmas[step + 1]
            let delta = sigmaNext - sigmaCurrent
            latent5D = latent5D + velocityNoise * MLXArray(Float(delta))
            _ = latent5D.shape

            let elapsed = Date().timeIntervalSince(startedAt)
            let perStep = elapsed / Double(step + 1)
            let eta = perStep * Double(total - step - 1)
            continuation.yield(.step(step: step + 1, total: total, etaSeconds: eta))
        }

        // 9. VAE decode.
        let rescaled = WanVAEDecoder.preprocessLatent(latent5D)
        let decoded = vae(rescaled)
        let processed = WanVAEDecoder.postprocess(decoded)

        // 10. Drop batch dim, write MP4.
        let video = processed[0]
        try FileManager.default.createDirectory(
            at: request.outputDir, withIntermediateDirectories: true
        )
        let mp4URL = request.outputDir.appendingPathComponent(
            "wan-\(variant.rawValue)-\(UUID().uuidString.prefix(8)).mp4"
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

// MARK: - Module init helper

/// Force-register the video models. Call once at app launch.
public enum vMLXFluxVideo {
    public static func registerAll() {
        _ = WANModel._registerWan21_1_3B
        _ = WANModel._registerWan21_14B
        _ = WANModel._registerWan22_t2v_14B
        _ = WANModel._registerWan22_ti2v_5B
        _ = WANModel._registerWan22_i2v_14B
    }
}
