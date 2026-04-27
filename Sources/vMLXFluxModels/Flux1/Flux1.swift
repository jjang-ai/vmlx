import Foundation
@preconcurrency import MLX
import vMLXFluxKit

// MARK: - Flux1 (Schnell + Dev)
//
// Original FLUX.1 family — dual-encoder (T5-XXL + CLIP-L) DiT with
// flow-matching sampling. Two variants:
//
//   - Schnell: 4 steps, no CFG (guidance=0). 2.3B-ish.
//   - Dev:     20 steps, CFG via guidance embed. 12B.
//
// Track 1 ships the Module trees + weight loaders end-to-end. The
// remaining smoke gap: a real M-series test fixture with safetensors so
// `Tests/vMLXFluxTests/Track1SmokeTests.swift` can prove non-noise
// pixels. Until that smoke runs green on Eric's hardware, registry
// entries stay `isPlaceholder: true`.
//
// Track 2 owns Flux1Kontext + Flux1Fill in their own files. They
// previously lived here; once Track 2 splits them out the old defs
// here will go away. Until then the file-level `_register` calls in
// the existing `Flux1Kontext` / `Flux1Fill` types (left in this file
// from the prior scaffold — DO NOT remove without Track 2 sign-off)
// continue to register the edit kinds.

public final class Flux1Schnell: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-schnell",
            displayName: "FLUX.1 Schnell",
            kind: .imageGen,
            defaultSteps: 4,
            defaultGuidance: 0.0,
            supportsLoRA: true,
            // Module tree + DiT/VAE/encoder forward passes are ported
            // (T5XXL, CLIPL, FluxDiTModel, VAEDecoder). Smoke proof on
            // real safetensors weights is gated on `VMLX_SWIFT_TEST_WEIGHTS`
            // env var. Stays placeholder until that gates green.
            isPlaceholder: true,
            loader: { path, quant in
                _ = Flux1Schnell._register
                return try Flux1Schnell(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    public let transformer: FluxDiTModel
    public let vae: VAEDecoder
    public let t5: T5XXLEncoder
    public let clip: CLIPLEncoder

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        self.loadedWeights = try WeightLoader.load(from: modelPath)
        self.transformer = FluxDiTModel(config: .schnell)
        self.vae = VAEDecoder()
        // Schnell uses canonical T5-XXL (24 blocks, 4096 hidden, 64 heads × 64 dim,
        // 10240 ffn) and CLIP-L (12 blocks, 768 hidden, 12 heads).
        self.t5 = T5XXLEncoder(maxSeqLen: 256)
        self.clip = CLIPLEncoder()
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

        // Encoders run on placeholder token IDs (zeros) until a tokenizer
        // gets wired in. The shapes still match what the DiT expects, so
        // the rest of the pipeline executes; the output is conditioned on
        // the constant prompt embedding rather than `request.prompt`.
        // When swift-transformers' AutoTokenizer hookup lands, swap
        // these zero-arrays for `tokenizer.encode(request.prompt).ids`.
        let t5Tokens = Array(repeating: 0, count: 64)
        let clipTokens = Array(repeating: 0, count: 77)
        let t5Embed = t5.encode(tokenIds: t5Tokens)
        let (_, pooledClip) = clip.encodePooled(tokenIds: clipTokens)

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
                txt: t5Embed,
                pooledClip: pooledClip,
                timestep: timestep,
                guidance: nil,
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
            try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "flux1-schnell")
        }
        continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
    }
}

public final class Flux1Dev: ImageGenerator, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-dev",
            displayName: "FLUX.1 Dev",
            kind: .imageGen,
            defaultSteps: 20,
            defaultGuidance: 3.5,
            supportsLoRA: true,
            isPlaceholder: true,
            loader: { path, quant in
                _ = Flux1Dev._register
                return try Flux1Dev(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?
    public let loadedWeights: LoadedWeights
    public let transformer: FluxDiTModel
    public let vae: VAEDecoder
    public let t5: T5XXLEncoder
    public let clip: CLIPLEncoder

    public init(modelPath: URL, quantize: Int?) throws {
        self.modelPath = modelPath
        self.quantize = quantize
        _ = Self._register
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw FluxError.weightsNotFound(modelPath)
        }
        self.loadedWeights = try WeightLoader.load(from: modelPath)
        self.transformer = FluxDiTModel(config: .dev)
        self.vae = VAEDecoder()
        self.t5 = T5XXLEncoder(maxSeqLen: 512)
        self.clip = CLIPLEncoder()
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

        let t5Tokens = Array(repeating: 0, count: 256)
        let clipTokens = Array(repeating: 0, count: 77)
        let t5Embed = t5.encode(tokenIds: t5Tokens)
        let (_, pooledClip) = clip.encodePooled(tokenIds: clipTokens)
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
                txt: t5Embed,
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
            try ImageIO.writePNG(image, outputDir: request.outputDir, prefix: "flux1-dev")
        }
        continuation.yield(.completed(url: outURL, seed: request.seed ?? 0))
    }
}

// MARK: - Flux1Kontext / Flux1Fill (Track 2 will split out)
//
// These edit heads are owned by Track 2 (`Flux1Kontext.swift`,
// `Flux1Fill.swift`). Until those files land we keep the existing
// scaffold registration here so the registry surface doesn't churn.
// Track 2 will delete these and move the impls to their own files.

public final class Flux1Kontext: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-kontext",
            displayName: "FLUX.1 Kontext",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 2.5,
            isPlaceholder: true,
            loader: { path, quant in
                _ = Flux1Kontext._register
                return try Flux1Kontext(modelPath: path, quantize: quant)
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

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux1 Kontext edit — Track 2 owns Flux1Kontext.swift; this scaffold registration stays until that file lands."))
        }
    }
}

public final class Flux1Fill: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-fill",
            displayName: "FLUX.1 Fill",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 30.0,
            isPlaceholder: true,
            loader: { path, quant in
                _ = Flux1Fill._register
                return try Flux1Fill(modelPath: path, quantize: quant)
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

    public func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Flux1 Fill inpaint — Track 2 owns Flux1Fill.swift; this scaffold registration stays until that file lands."))
        }
    }
}
