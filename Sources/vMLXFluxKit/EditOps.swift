import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
#if canImport(AppKit)
import AppKit
import CoreImage
#endif

// MARK: - EditOps
//
// Shared primitives for FLUX.1 Kontext, FLUX.1 Fill, and Qwen-Image-Edit
// (Track 2). Three concerns live here:
//
//   1. Source-image ingestion: PNG → (B, 3, H, W) float in [-1, 1] →
//      VAE-encoded latent (or a deterministic noise-rescale stub when the
//      VAE encoder isn't ported yet).
//
//   2. Mask handling (Fill): PNG → (B, 1, latentH, latentW) float in [0, 1]
//      where 1 = "edit this region", 0 = "keep source as-is". Composited
//      back to the latent at every Euler step so the unmasked region stays
//      pixel-equal to the source.
//
//   3. Image-conditioning patchify: source-image latent → (B, N_src, D)
//      tokens that the FLUX.1 transformer's text-image-joint stream
//      consumes (Kontext) or appends to the fill mask channel (Fill).
//
// Every function here is deliberately small + dep-free so Track 2 can
// compile with a `MockTextEncoder` while Track 1 finishes the real
// T5-XXL / CLIP-L / Qwen2-VL-7B encoder layer.

// MARK: - MockTextEncoder
//
// Track 1 owns the real T5-XXL / CLIP-L / Qwen2-VL-7B encoders in
// `vMLXFluxKit/Encoders/{T5XXL,CLIPL,Qwen2VL7B}.swift` and defines the
// canonical `TextEncoder` protocol in `Encoders/Encoders.swift`. Track 2
// re-uses that protocol — see `MockTextEncoder` below.
//
// `MockTextEncoder` is a **temporary** stand-in so Track 2 can wire its
// edit pipelines end-to-end before Track 1's real encoder weights are
// loadable on every developer's box. Produces a deterministic per-token
// embedding by hashing token IDs into a (1, S, hiddenSize) MLX tensor.
// NOT semantically meaningful — the smoke tests assert on tensor shapes
// and on the placeholder source-passthrough behavior, not on real
// encoder output. Once the real encoders are confirmed loadable, swap
// call sites to use `T5XXL` / `CLIPL` / `Qwen2VL7B` from
// `vMLXFluxKit.Encoders` and delete this struct.

/// Deterministic mock encoder. Mirrors the shape of Track 1's
/// `TextEncoder` protocol (`hiddenSize` / `maxSeqLen` / `encode(tokenIds:)`)
/// so swapping to the real `T5XXL` / `CLIPL` / `Qwen2VL7B` is mechanical
/// once Track 1 ships them. **Currently NOT declared as conforming**
/// because Track 1's `Encoders/{T5XXL,CLIPL,Qwen2VL7B}.swift` reference
/// a `FluxTextEncoder` protocol that doesn't exist yet (Track 1
/// in-flight bug, 2026-04-27). Once Track 1 stabilizes either the
/// `TextEncoder` or `FluxTextEncoder` protocol name, add `: TextEncoder`
/// (or whichever name wins) to this struct's declaration. Search
/// `MockTextEncoder` under `Sources/vMLXFluxModels/` to find the
/// edit-head call sites that need swapping.
public struct MockTextEncoder: Sendable {
    public let hiddenSize: Int
    public let maxSeqLen: Int

    public init(hiddenSize: Int = 4096, maxSeqLen: Int = 256) {
        self.hiddenSize = hiddenSize
        self.maxSeqLen = maxSeqLen
    }

    public func encode(tokenIds: [Int]) -> MLXArray {
        let s = min(maxSeqLen, max(1, tokenIds.count))
        var flat: [Float] = []
        flat.reserveCapacity(s * hiddenSize)
        // Hash each token id into a deterministic embedding row.
        for i in 0..<s {
            let tok = i < tokenIds.count ? tokenIds[i] : 0
            // FNV-1a-flavored hash so different token IDs produce
            // visibly different rows. Seeded by the position too so
            // the same token at different positions differs.
            var h: UInt64 = 1469598103934665603
            h ^= UInt64(bitPattern: Int64(tok))
            h = h &* 1099511628211
            h ^= UInt64(i) &* 0x9E3779B97F4A7C15
            for d in 0..<hiddenSize {
                let mix = h
                    ^ (UInt64(d) &* 0xBF58476D1CE4E5B9)
                let normalized = Float(mix & 0xFFFF) / 32768.0 - 1.0
                flat.append(normalized * 0.05)
            }
        }
        return MLXArray(flat).reshaped([1, s, hiddenSize])
    }

    /// Convenience: hash a UTF-8 prompt into a token-id list and encode.
    /// Track 2 callers can call this without reaching for a tokenizer.
    public func encodeFromString(prompt: String) -> MLXArray {
        let ids: [Int] = prompt.utf8.prefix(maxSeqLen).map { Int($0) }
        return encode(tokenIds: ids.isEmpty ? [0] : ids)
    }

    /// Pooled embedding (1, hiddenSize) — mean over sequence axis.
    public func encodePooledFromString(prompt: String) -> MLXArray {
        return encodeFromString(prompt: prompt).mean(axis: 1)
    }
}

// MARK: - Image ingestion (PNG → MLXArray)

public enum EditOps {

    /// Read a PNG/JPEG/etc. from disk, resize to `(width, height)`, and
    /// return a `(1, 3, H, W)` MLXArray normalized to `[-1, 1]`.
    /// Caller passes pixel-space dims; latent-space conversion happens
    /// later via `encodeToLatentStub` once a VAE encoder ships.
    @MainActor
    public static func loadImageRGB(
        url: URL, width: Int, height: Int
    ) throws -> MLXArray {
        #if canImport(AppKit)
        guard let nsImg = NSImage(contentsOf: url) else {
            throw FluxError.invalidRequest("could not read image at \(url.path)")
        }
        guard let cgImg = nsImg.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw FluxError.invalidRequest("could not extract CGImage from \(url.path)")
        }

        // Resize via Core Graphics into an sRGB bitmap context.
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        var pixelBuffer = [UInt8](repeating: 0, count: bytesPerRow * height)
        guard let ctx = pixelBuffer.withUnsafeMutableBytes({ raw -> CGContext? in
            CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        }) else {
            throw FluxError.invalidRequest("CGContext alloc failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: width, height: height))

        // RGBA → RGB float in [-1, 1]. (H, W, 4) byte → (3, H, W) float.
        var floats: [Float] = []
        floats.reserveCapacity(3 * height * width)
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * bytesPerRow + x * 4 + c
                    let v = Float(pixelBuffer[idx]) / 255.0
                    floats.append(v * 2.0 - 1.0)
                }
            }
        }
        return MLXArray(floats).reshaped([1, 3, height, width])
        #else
        throw FluxError.notImplemented("EditOps.loadImageRGB requires AppKit")
        #endif
    }

    /// Read a single-channel mask PNG. White pixels → 1.0, black → 0.0.
    /// Resize to `(latentH, latentW)` (latent-space dims, i.e. pixel/8).
    /// Returns shape `(1, 1, latentH, latentW)`.
    @MainActor
    public static func loadMaskLatent(
        url: URL, latentH: Int, latentW: Int, threshold: Float = 0.5
    ) throws -> MLXArray {
        #if canImport(AppKit)
        guard let nsImg = NSImage(contentsOf: url),
              let cgImg = nsImg.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else {
            throw FluxError.invalidRequest("could not read mask at \(url.path)")
        }
        let cs = CGColorSpaceCreateDeviceGray()
        let bytesPerRow = latentW
        var buf = [UInt8](repeating: 0, count: bytesPerRow * latentH)
        guard let ctx = buf.withUnsafeMutableBytes({ raw -> CGContext? in
            CGContext(
                data: raw.baseAddress,
                width: latentW,
                height: latentH,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
        }) else {
            throw FluxError.invalidRequest("CGContext alloc failed for mask")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: latentW, height: latentH))

        var floats: [Float] = []
        floats.reserveCapacity(latentH * latentW)
        for byte in buf {
            let v = Float(byte) / 255.0
            floats.append(v >= threshold ? 1.0 : 0.0)
        }
        return MLXArray(floats).reshaped([1, 1, latentH, latentW])
        #else
        throw FluxError.notImplemented("EditOps.loadMaskLatent requires AppKit")
        #endif
    }

    // MARK: - VAE-encode stub
    //
    // The real path: run the source image through the VAE encoder to
    // produce a `(1, 16, H/8, W/8)` latent. The encoder isn't ported yet
    // (only the decoder is in `VAE.swift`). Until it is, we approximate
    // with a deterministic projection so the rest of the edit pipeline
    // can run + the smoke tests can assert on shapes/structure.
    //
    // Once `VAEEncoder` ships, replace this with a real call.

    /// **STUB** — turn a `(1, 3, H, W)` source image into a Flux latent.
    /// Real impl: `VAEEncoder()(image)` then `Flux preprocess`. Stub:
    /// 8× spatial average-pool to mimic the 8× downsample, then tile to
    /// 16 channels via a deterministic permutation. Output shape:
    /// `(1, 16, H/8, W/8)`.
    public static func encodeToLatentStub(_ image: MLXArray) -> MLXArray {
        precondition(image.ndim == 4, "encodeToLatentStub: expected 4D (B,C,H,W)")
        let b = image.dim(0)
        let c = image.dim(1)
        let h = image.dim(2)
        let w = image.dim(3)
        precondition(c == 3, "encodeToLatentStub: expected 3 input channels")
        let lh = h / 8
        let lw = w / 8

        // 8× average-pool: (B, 3, H, W) → (B, 3, lh, lw) via reshape+mean.
        let reshaped = image.reshaped([b, c, lh, 8, lw, 8])
        let pooled = reshaped.mean(axes: [3, 5])  // (B, 3, lh, lw)

        // Tile 3 channels → 16 by stacking 6 rotations + a zero pad.
        var slabs: [MLXArray] = []
        for shift in 0..<5 {
            let rolled = MLX.concatenated([
                pooled[0..<b, shift % 3 ..< 3, 0..<lh, 0..<lw],
                pooled[0..<b, 0 ..< shift % 3, 0..<lh, 0..<lw],
            ], axis: 1)
            slabs.append(rolled)
        }
        slabs.append(MLXArray.zeros([b, 1, lh, lw]).asType(pooled.dtype))
        let tiled = MLX.concatenated(slabs, axis: 1)
        // Trim to exactly 16 channels.
        return tiled[0..<b, 0..<16, 0..<lh, 0..<lw]
    }

    // MARK: - Mask compositing
    //
    // After every Euler step, Fill blends the current denoised latent
    // with the original source latent so unmasked pixels are preserved.
    //
    //   composite = mask * edited + (1 - mask) * source
    //
    // mask: (B, 1, latH, latW) — broadcasts across the channel axis.

    /// Compose source + edited via mask. Returns `edited` shape.
    public static func compositeMaskedLatent(
        source: MLXArray, edited: MLXArray, mask: MLXArray
    ) -> MLXArray {
        // Broadcast mask (B,1,H,W) over (B,C,H,W).
        return mask * edited + (MLXArray(Float(1)) - mask) * source
    }

    /// Apply the mask compositing AFTER an unpatchify back to spatial
    /// layout. Convenience wrapper for callers that work in patchified
    /// (B, N, D) space and want to apply mask at boundaries only.
    public static func compositeMaskedLatentPatchified(
        sourcePatched: MLXArray,
        editedPatched: MLXArray,
        maskPatched: MLXArray
    ) -> MLXArray {
        // mask is (B, N, 1) — broadcast across the channel/D axis.
        return maskPatched * editedPatched
            + (MLXArray(Float(1)) - maskPatched) * sourcePatched
    }

    /// Patchify a `(B, 1, lh, lw)` mask into `(B, N, 1)` for patchified
    /// compositing. Mirrors `LatentSpace.patchify` but returns `Float`
    /// values in [0, 1] (mean-pooled when patchSize > 1).
    public static func patchifyMask(
        _ mask: MLXArray, patchSize: Int
    ) -> MLXArray {
        let b = mask.dim(0)
        precondition(mask.dim(1) == 1, "patchifyMask: expected single channel")
        let h = mask.dim(2)
        let w = mask.dim(3)
        let ph = h / patchSize
        let pw = w / patchSize
        // (B, 1, ph, ps, pw, ps) → mean over ps,ps → (B, ph, pw, 1)
        let r = mask.reshaped([b, 1, ph, patchSize, pw, patchSize])
        let pooled = r.mean(axes: [3, 5])
        // (B, 1, ph, pw) → (B, ph*pw, 1)
        return pooled.reshaped([b, ph * pw, 1])
    }

    // MARK: - Kontext: source-latent token concat
    //
    // FLUX.1 Kontext appends the source-image latent (after patchify) to
    // the text-token sequence the transformer consumes. The text stream
    // sees `[txt_tokens; src_img_tokens]` while the image stream is
    // initialized from Gaussian noise as in plain FLUX.1.
    //
    // Reference: `mflux/flux/transformer.py` — `__call__` with
    // `condition_image_latent` non-nil.

    /// Concat source-image latent tokens onto the text stream.
    /// `txtTokens` and `srcImgTokens` must have the same channel dim D.
    /// Returns `(B, N_txt + N_src, D)`.
    public static func concatKontextConditioning(
        txtTokens: MLXArray, srcImgTokens: MLXArray
    ) -> MLXArray {
        precondition(txtTokens.dim(2) == srcImgTokens.dim(2),
                     "concatKontextConditioning: channel dim mismatch")
        return MLX.concatenated([txtTokens, srcImgTokens], axis: 1)
    }

    // MARK: - Fill: mask channel append
    //
    // FLUX.1 Fill expands the transformer's input channels from 16 to 320
    // (16 latent + 1 mask + … residual + masked-source). The mflux impl
    // packs `[noisy_latent; mask; masked_source_latent]` along the
    // channel axis BEFORE the patch_embed projection.
    //
    // We expose just the channel-concat helper so the per-model file
    // can pick its own patch-embed input width.

    /// Concat `[noisyLatent; mask; maskedSource]` along the channel axis.
    /// Shapes: noisyLatent `(B, 16, H, W)`, mask `(B, 1, H, W)`,
    /// maskedSource `(B, 16, H, W)` → returns `(B, 33, H, W)`.
    public static func packFillChannels(
        noisyLatent: MLXArray, mask: MLXArray, maskedSource: MLXArray
    ) -> MLXArray {
        return MLX.concatenated([noisyLatent, mask, maskedSource], axis: 1)
    }

    /// Apply the mask to the source latent: zero out pixels where the
    /// mask says "edit", preserve elsewhere. Used to build the
    /// `masked_source` channel for Fill.
    public static func maskSourceLatent(
        source: MLXArray, mask: MLXArray
    ) -> MLXArray {
        return source * (MLXArray(Float(1)) - mask)
    }

    // MARK: - Strength → init step
    //
    // The `strength` parameter in `ImageEditRequest` controls how far
    // we deviate from the source. Maps to the scheduler's "init step"
    // — `init_step = floor(strength * num_steps)`. At init_step=0 the
    // edited image is just the source; at num_steps it's a fresh gen.

    public static func initStepFromStrength(
        strength: Float, numSteps: Int
    ) -> Int {
        let clamped = max(0.0, min(1.0, strength))
        return Int(Float(numSteps) * clamped)
    }
}

// MARK: - Pixel-level diff helpers (for smoke tests)

public enum EditOpsTesting {

    /// Mean absolute pixel difference between two `(C, H, W)` or
    /// `(B, C, H, W)` tensors normalized to [0, 1]. Returns a Float.
    public static func meanAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a - b)
        return diff.mean().item(Float.self)
    }

    /// Cheap structural-similarity proxy — Pearson correlation across
    /// per-channel pixel arrays. SSIM is overkill for smoke; correlation
    /// catches "is the structure preserved?" with two-tensor support.
    /// Returns a value in roughly [-1, 1].
    public static func pixelCorrelation(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.flattened().asType(.float32)
        let bf = b.flattened().asType(.float32)
        let aMean = af.mean()
        let bMean = bf.mean()
        let aCenter = af - aMean
        let bCenter = bf - bMean
        let num = (aCenter * bCenter).sum()
        let den = MLX.sqrt((aCenter * aCenter).sum() * (bCenter * bCenter).sum())
            + MLXArray(Float(1e-8))
        return (num / den).item(Float.self)
    }
}
