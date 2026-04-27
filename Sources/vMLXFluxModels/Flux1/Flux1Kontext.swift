import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import vMLXFluxKit

// MARK: - Flux1Kontext (Track 2 port)
//
// Prompt-only edit head. No mask: takes a `(source_image, prompt)` pair
// and lets the FLUX.1 transformer steer the source image toward the
// prompt via source-latent conditioning concatenated to the text stream.
//
// Python source: `mflux/flux/transformer.py` — when a model loads with
// `kontext=True`, the transformer's joint stream consumes the pre-encoded
// source image latent as additional "text" tokens. The image stream is
// initialized from Gaussian noise as in plain FLUX.1; the source latent
// only enters via cross-attention with the text+source-image tokens.
//
// Architecture sketch (matching mflux):
//
//     prompt           ──► T5-XXL    ──► (B, N_t5,   4096)   ┐
//     prompt           ──► CLIP-L    ──► (B, 768)            │ joint stream
//     source_image     ──► VAE.enc   ──► (B, 16, H/8, W/8)   │  → txt + src
//     source_image     ──► patchify  ──► (B, N_src, 64)      ┘
//     init_noise       ──►              (B, 16, H/8, W/8)
//     init_noise       ──► patchify  ──► (B, N_img, 64)      ─► img stream
//
//     for step in 0..<num_steps:
//         vec = time_emb(step) + pooled_clip + guidance_emb
//         img_tokens = imgIn(img_patched)
//         txt_tokens = txtIn(t5_tokens) ++ srcIn(src_img_tokens)
//         (img, txt) = double_blocks(img, txt, vec)
//         merged     = single_blocks(concat(txt, img), vec)
//         velocity   = final_layer(merged[:, N_txt+N_src:])
//         img_patched = scheduler.step(img_patched, velocity, step)
//
//     output_image = VAE.decode(unpatchify(img_patched))
//
// Smoke test: edit cat → dog, structure preservation > 0.6 correlation.
//
// THIS FILE owns the registry key `flux1-kontext`. The original stub in
// `Flux1.swift` (Track-1 territory) re-registers with the same key but is
// expected to be removed when Track 1 lands the real Flux1 forward path.
// Until then, last-write-wins on `ModelRegistry.entries` so whichever of
// the two `_register` statics fires last is what callers see. The
// per-class `_register` triggers via `init()` on first instantiation and
// via the umbrella `vMLXFluxModels.registerAll()`. Two registrations of
// the same key are idempotent — neither fights the other.

public final class Flux1KontextEditor: ImageEditor, @unchecked Sendable {
    public static let _register: Void = {
        ModelRegistry.register(ModelEntry(
            name: "flux1-kontext",
            displayName: "FLUX.1 Kontext",
            kind: .imageEdit,
            defaultSteps: 28,
            defaultGuidance: 2.5,
            supportsLoRA: true,
            // Keep `isPlaceholder: true` until end-to-end smoke test
            // passes a real cat→dog edit on weights from disk. Flips to
            // false in the same commit that proves real output.
            isPlaceholder: true,
            loader: { path, quant in
                _ = Flux1KontextEditor._register
                return try Flux1KontextEditor(modelPath: path, quantize: quant)
            }
        ))
    }()

    public let modelPath: URL
    public let quantize: Int?

    /// Loaded weights. Lazy because construction is sync-throws but we
    /// want to allow the model to register in the registry without
    /// touching disk.
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

    // MARK: - End-to-end edit
    //
    // Wired up against `MockTextEncoder` until Track 1 ships
    // `vMLXFluxKit.Encoders.{T5XXL,CLIPL}`. Search `MockTextEncoder` to
    // find the swap points.

    private func runEdit(
        _ request: ImageEditRequest,
        progress: (Int, Int) -> Void
    ) async throws -> URL {
        // 1. Load weights once.
        if weights == nil {
            self.weights = try WeightLoader.load(from: modelPath)
        }

        // 2. Decide working resolution. Match source unless overridden.
        let srcImage = try await MainActor.run {
            try EditOps.loadImageRGB(
                url: request.sourceImage,
                width: request.width ?? 1024,
                height: request.height ?? 1024
            )
        }
        let width = srcImage.dim(3)
        let height = srcImage.dim(2)

        // 3. Encode source → latent (STUB until VAE encoder lands).
        let srcLatent = EditOps.encodeToLatentStub(srcImage)

        // 4. Encode prompt via the TEMPORARY mock. Real T5+CLIP from
        //    Track 1 swaps in here:
        //        let t5    = T5XXL(weights: ...)
        //        let clipL = CLIPL(weights: ...)
        //        let txtTokens  = t5.encode(tokenIds: tokenizer.encode(request.prompt))
        //        let (_, pooledClip) = clipL.encodePooled(
        //            tokenIds: tokenizer.encode(request.prompt))
        let mockEncoder = MockTextEncoder(hiddenSize: 4096, maxSeqLen: 256)
        let txtTokens = mockEncoder.encodeFromString(prompt: request.prompt)
        let pooledClip = MockTextEncoder(hiddenSize: 768, maxSeqLen: 256)
            .encodePooledFromString(prompt: request.prompt)

        // 5. Patchify source latent → src tokens. Channel dim must match
        //    `txtTokens.dim(2)` (4096) for concat. The real impl
        //    projects via a learned `srcIn` linear; we approximate with
        //    a deterministic tile.
        let srcPatched = patchify(
            srcLatent, patchSize: 2, inChannels: 16
        )
        let srcTokens = projectSourceTokensStub(
            srcPatched, targetDim: txtTokens.dim(2))

        // 6. Concat into joint conditioning sequence.
        let conditioning = EditOps.concatKontextConditioning(
            txtTokens: txtTokens, srcImgTokens: srcTokens
        )

        // 7. Initialize image-stream noise.
        let noise = LatentSpace.initialNoise(
            width: width, height: height,
            layout: .spatial(channels: 16),
            batchSize: 1, seed: request.seed
        )

        // 8. Run the scheduler. The real per-step velocity prediction
        //    invokes `FluxDiTModel`. Until weights are wired, we return
        //    the source image — `isPlaceholder: true` advertises this.
        //    `progress` callback runs once per step so the UI's
        //    streaming bar still ticks.
        let initStep = EditOps.initStepFromStrength(
            strength: request.strength, numSteps: request.steps
        )
        let scheduler = FlowMatchEulerScheduler(
            steps: request.steps,
            imageSeqLen: noise.dim(0) > 0 ? noise.dim(2) * noise.dim(3) : 4096
        )
        for step in initStep..<scheduler.steps {
            progress(step + 1, scheduler.steps)
            // Real impl:
            //   let v = dit.callAsFunction(
            //       imgPatched: latent, txt: conditioning,
            //       pooledClip: pooledClip, timestep: ts, guidance: g
            //   )
            //   latent = scheduler.step(latent: latent, velocity: v, stepIndex: step)
            _ = conditioning
            _ = pooledClip
            _ = scheduler
        }

        // 9. Decode. Until VAE decoder is wired with real weights, we
        //    write the source image as the output (placeholder bytes).
        return try await MainActor.run {
            try ImageIO.writePNG(
                ((srcImage + MLXArray(Float(1))) * MLXArray(Float(0.5))),
                outputDir: request.outputDir,
                prefix: "flux1-kontext"
            )
        }
    }

    /// **STUB** — project (B, N_src, 64) source tokens to the same
    /// channel dim as the text tokens so they can be concatenated into
    /// the joint stream. Real impl uses the checkpoint's `kontext_proj`
    /// learned linear. Stub: tile + zero-pad to `targetDim`.
    private func projectSourceTokensStub(
        _ srcPatched: MLXArray, targetDim: Int
    ) -> MLXArray {
        let b = srcPatched.dim(0)
        let n = srcPatched.dim(1)
        let inDim = srcPatched.dim(2)
        if inDim == targetDim {
            return srcPatched
        }
        let tilesNeeded = (targetDim + inDim - 1) / inDim
        var tiled = srcPatched
        for _ in 1..<tilesNeeded {
            tiled = MLX.concatenated([tiled, srcPatched], axis: 2)
        }
        // Trim to exactly targetDim.
        return tiled[0..<b, 0..<n, 0..<targetDim]
    }
}
