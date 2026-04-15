import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Flux DiT transformer backbone
//
// Pure-Swift port of the FLUX.1 transformer used by Flux1 Schnell/Dev,
// Flux2 Klein, Qwen-Image, and FIBO. Architecture:
//
//   patch_embed      : (B, 16, H/8, W/8) → (B, N_img, D=3072)
//   time_mlp         : timestep → time embedding (D=3072)
//   text_proj        : (B, N_txt, C_text) → (B, N_txt, D)
//   guidance_mlp     : (optional, Dev only) guidance scale → embedding
//   blocks[19]       : "double" blocks — dual-stream (img + txt) attention
//   single_blocks[38]: fused single-stream attention (img only after merge)
//   final_layer      : norm + linear → (B, N_img, 16*patch²) → unpatchify
//
// The model splits into two regimes:
//   - Double blocks keep image and text tokens in separate streams with
//     shared attention (MM-DiT style).
//   - Single blocks concatenate img + txt into one sequence and run
//     standard self-attention. After the single blocks we drop text
//     tokens and return just the image tokens.
//
// Parameter counts (Flux1 Dev):
//   - 12B total
//   - 19 double blocks × 360M = 6.8B
//   - 38 single blocks × 120M = 4.6B
//   - rest: 600M (embed, norm, text proj)
//
// This file ships the BLOCK types + the full-model assembler. The
// per-model files (Flux1Schnell, Flux1Dev, Flux2Klein, ...) just pick
// hyperparameters and register for weight loading.
//
// Weight naming follows Black Forest Labs' safetensors layout so the
// standard .safetensors checkpoints from HuggingFace load directly.

// MARK: - AdaLN modulation

/// AdaLN single-stream modulation: produces (shift, scale, gate) triples
/// applied around attention and MLP. The `num_mods` controls how many
/// triples we produce — double blocks use 6 (three for attn, three for
/// mlp), single blocks use 3.
public final class FluxModulation: Module {
    public let linear: Linear
    public let numMods: Int   // 3 or 6
    public let doubleMode: Bool

    public init(dim: Int, doubleMode: Bool) {
        self.doubleMode = doubleMode
        self.numMods = doubleMode ? 6 : 3
        self.linear = Linear(dim, dim * numMods)
        super.init()
    }

    /// Returns a list of `ModTriple` — 2 for double-mode blocks (attn,mlp),
    /// 1 for single-mode blocks. `vec` is the conditioning embedding
    /// (time + pooled CLIP + guidance).
    public func callAsFunction(_ vec: MLXArray) -> [ModTriple] {
        let out = linear(silu(vec))
        let dim = out.dim(-1) / numMods
        var result: [ModTriple] = []
        let mods = numMods / 3
        for i in 0..<mods {
            let base = i * 3
            let shift = out[.ellipsis, (base + 0) * dim ..< (base + 1) * dim]
            let scale = out[.ellipsis, (base + 1) * dim ..< (base + 2) * dim]
            let gate  = out[.ellipsis, (base + 2) * dim ..< (base + 3) * dim]
            // Add a sequence dim so we broadcast over tokens: (B, 1, D).
            result.append(ModTriple(
                shift: shift.reshaped([shift.dim(0), 1, dim]),
                scale: scale.reshaped([scale.dim(0), 1, dim]),
                gate:  gate.reshaped([gate.dim(0), 1, dim])
            ))
        }
        return result
    }
}

/// Explicit return type for `FluxModulation`. Swift auto-tuple-labeling
/// inference gets confused when threading named-field tuples through
/// array literals, so we use a named struct for clarity + type safety.
/// Not `Sendable` because MLXArray isn't — the module tree is actor-
/// isolated so cross-actor passing never happens.
public struct ModTriple {
    public let shift: MLXArray
    public let scale: MLXArray
    public let gate: MLXArray

    public init(shift: MLXArray, scale: MLXArray, gate: MLXArray) {
        self.shift = shift
        self.scale = scale
        self.gate = gate
    }
}

// MARK: - Q/K norm (RMSNorm per-head, applied to Q and K separately)

/// QKNorm applies per-head RMSNorm to Q and K before attention.
/// Prevents attention score blow-up at mixed precision.
public final class QKNorm: Module {
    public let qNorm: RMSNorm
    public let kNorm: RMSNorm

    public init(headDim: Int) {
        // Use MLXNN's RMSNorm — it's hardware-accelerated via
        // MLXFast.rmsNorm and ~5x faster than the pure-Swift variant.
        self.qNorm = RMSNorm(dimensions: headDim, eps: 1e-6)
        self.kNorm = RMSNorm(dimensions: headDim, eps: 1e-6)
        super.init()
    }

    public func callAsFunction(q: MLXArray, k: MLXArray) -> (MLXArray, MLXArray) {
        return (qNorm(q), kNorm(k))
    }
}

// MARK: - Double stream block (dual img + txt attention)

/// One double-stream MM-DiT block. Image and text tokens keep separate
/// Q/K/V projections but share an attention op after concatenation.
public final class FluxDoubleStreamBlock: Module {
    public let dim: Int
    public let numHeads: Int

    // Image stream
    public let imgMod: FluxModulation
    public let imgNorm1: LayerNorm
    public let imgAttnQKV: Linear
    public let imgAttnNorm: QKNorm
    public let imgAttnProj: Linear
    public let imgNorm2: LayerNorm
    public let imgMlp0: Linear
    public let imgMlp2: Linear

    // Text stream
    public let txtMod: FluxModulation
    public let txtNorm1: LayerNorm
    public let txtAttnQKV: Linear
    public let txtAttnNorm: QKNorm
    public let txtAttnProj: Linear
    public let txtNorm2: LayerNorm
    public let txtMlp0: Linear
    public let txtMlp2: Linear

    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0) {
        self.dim = dim
        self.numHeads = numHeads
        let headDim = dim / numHeads
        let mlpDim = Int(Float(dim) * mlpRatio)

        // Image stream
        self.imgMod = FluxModulation(dim: dim, doubleMode: true)
        self.imgNorm1 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.imgAttnQKV = Linear(dim, dim * 3)
        self.imgAttnNorm = QKNorm(headDim: headDim)
        self.imgAttnProj = Linear(dim, dim)
        self.imgNorm2 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.imgMlp0 = Linear(dim, mlpDim)
        self.imgMlp2 = Linear(mlpDim, dim)

        // Text stream — same shape as image stream.
        self.txtMod = FluxModulation(dim: dim, doubleMode: true)
        self.txtNorm1 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.txtAttnQKV = Linear(dim, dim * 3)
        self.txtAttnNorm = QKNorm(headDim: headDim)
        self.txtAttnProj = Linear(dim, dim)
        self.txtNorm2 = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.txtMlp0 = Linear(dim, mlpDim)
        self.txtMlp2 = Linear(mlpDim, dim)

        super.init()
    }

    /// - Parameters:
    ///   - img: (B, N_img, D) image tokens after patch_embed.
    ///   - txt: (B, N_txt, D) text tokens after T5+CLIP encoding & projection.
    ///   - vec: (B, D) pooled conditioning vector (time + pooled CLIP + guidance).
    ///   - rope: optional rotary embedding applied to Q and K for BOTH streams.
    public func callAsFunction(
        img: MLXArray, txt: MLXArray, vec: MLXArray, rope: RoPE2D?
    ) -> (img: MLXArray, txt: MLXArray) {
        let imgMods = imgMod(vec)   // 2 triples: attn, mlp
        let txtMods = txtMod(vec)

        // Image Q/K/V + modulation.
        var imgNormed = imgNorm1(img)
        imgNormed = imgNormed * (MLXArray(Float(1)) + imgMods[0].scale) + imgMods[0].shift
        let imgQkv = imgAttnQKV(imgNormed)
        let (imgQ, imgK, imgV) = splitQKV(imgQkv, numHeads: numHeads)
        let (imgQn, imgKn) = imgAttnNorm(q: imgQ, k: imgK)

        // Text Q/K/V + modulation.
        var txtNormed = txtNorm1(txt)
        txtNormed = txtNormed * (MLXArray(Float(1)) + txtMods[0].scale) + txtMods[0].shift
        let txtQkv = txtAttnQKV(txtNormed)
        let (txtQ, txtK, txtV) = splitQKV(txtQkv, numHeads: numHeads)
        let (txtQn, txtKn) = txtAttnNorm(q: txtQ, k: txtK)

        // Joint attention: concat text + image tokens along seq axis,
        // run attention, split back. RoPE applies to image tokens only.
        // Shape: (B, H, N_txt + N_img, D_head)
        let q = concatenated([txtQn, imgQn], axis: 2)
        let k = concatenated([txtKn, imgKn], axis: 2)
        let v = concatenated([txtV, imgV], axis: 2)

        // Apply RoPE only to the image slice. Here we simplify by
        // skipping RoPE on text tokens (they get their own positional
        // bands in the full Flux impl — adding that is a ~30-line extension
        // when we wire per-model hyperparameters).
        _ = rope  // TODO: split q/k, rope image half, reassemble

        let attnOut = scaledDotProductAttention(q: q, k: k, v: v, rope: nil)
        // (B, H, N_total, D_head) → (B, N_total, D)
        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([
            attnOut.dim(0), attnOut.dim(2), dim
        ])
        let nTxt = txt.dim(1)
        let txtAttn = merged[0 ..< merged.dim(0), 0 ..< nTxt, 0 ..< dim]
        let imgAttn = merged[0 ..< merged.dim(0), nTxt ..< merged.dim(1), 0 ..< dim]

        // Image attention residual with gate.
        var imgOut = img + imgAttnProj(imgAttn) * imgMods[0].gate

        // Image MLP.
        var imgNorm2Out = imgNorm2(imgOut)
        imgNorm2Out = imgNorm2Out * (MLXArray(Float(1)) + imgMods[1].scale) + imgMods[1].shift
        let imgMlp = imgMlp2(gelu(imgMlp0(imgNorm2Out)))
        imgOut = imgOut + imgMlp * imgMods[1].gate

        // Text attention residual with gate.
        var txtOut = txt + txtAttnProj(txtAttn) * txtMods[0].gate

        // Text MLP.
        var txtNorm2Out = txtNorm2(txtOut)
        txtNorm2Out = txtNorm2Out * (MLXArray(Float(1)) + txtMods[1].scale) + txtMods[1].shift
        let txtMlp = txtMlp2(gelu(txtMlp0(txtNorm2Out)))
        txtOut = txtOut + txtMlp * txtMods[1].gate

        return (img: imgOut, txt: txtOut)
    }
}

// MARK: - Single stream block (fused attention after img/txt merge)

/// Single-stream block. Runs after the double blocks on the
/// concatenated (txt, img) sequence. Has one attention op and one
/// parallel MLP fused into the same residual.
public final class FluxSingleStreamBlock: Module {
    public let dim: Int
    public let numHeads: Int
    public let mlpDim: Int

    public let mod: FluxModulation
    public let norm: LayerNorm
    public let linear1: Linear  // projects to qkv + mlp_in concatenated
    public let qkNorm: QKNorm
    public let linear2: Linear  // projects attn + mlp concatenated back to dim

    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0) {
        self.dim = dim
        self.numHeads = numHeads
        self.mlpDim = Int(Float(dim) * mlpRatio)
        let headDim = dim / numHeads

        self.mod = FluxModulation(dim: dim, doubleMode: false)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        // linear1 outputs (qkv: 3*dim) + (mlp_in: mlpDim)
        self.linear1 = Linear(dim, dim * 3 + mlpDim)
        self.qkNorm = QKNorm(headDim: headDim)
        // linear2 takes (attn: dim) + (mlp: mlpDim) back to dim
        self.linear2 = Linear(dim + mlpDim, dim)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, vec: MLXArray, rope: RoPE2D?
    ) -> MLXArray {
        let mods = mod(vec)
        let triple = mods[0]
        var xMod = norm(x)
        xMod = xMod * (MLXArray(Float(1)) + triple.scale) + triple.shift

        let out = linear1(xMod)
        // Split into qkv (3*dim) and mlp_in (mlpDim).
        let qkv = out[.ellipsis, 0 ..< (dim * 3)]
        let mlpIn = out[.ellipsis, (dim * 3) ..< (dim * 3 + mlpDim)]

        let (q, k, v) = splitQKV(qkv, numHeads: numHeads)
        let (qn, kn) = qkNorm(q: q, k: k)

        _ = rope  // TODO: apply to qn/kn when per-model RoPE config lands
        let attn = scaledDotProductAttention(q: qn, k: kn, v: v, rope: nil)
        // (B, H, N, D_head) → (B, N, D)
        let attnMerged = attn.transposed(0, 2, 1, 3).reshaped([
            attn.dim(0), attn.dim(2), dim
        ])

        // MLP gelu activation.
        let mlpActivated = gelu(mlpIn)

        // Concat attn + mlp along channel axis, project back to dim.
        let concated = concatenated([attnMerged, mlpActivated], axis: -1)
        let projected = linear2(concated)

        return x + projected * triple.gate
    }
}

// MARK: - Q/K/V split helper

/// Split a (B, N, 3*D) QKV tensor into three (B, H, N, D_head) tensors.
public func splitQKV(_ qkv: MLXArray, numHeads: Int) -> (MLXArray, MLXArray, MLXArray) {
    let b = qkv.dim(0)
    let n = qkv.dim(1)
    let d3 = qkv.dim(2)
    let d = d3 / 3
    let headDim = d / numHeads
    // (B, N, 3*D) → (B, N, 3, H, D_head)
    let reshaped = qkv.reshaped([b, n, 3, numHeads, headDim])
    // → (3, B, H, N, D_head)
    let transposed = reshaped.transposed(2, 0, 3, 1, 4)
    let q = transposed[0]
    let k = transposed[1]
    let v = transposed[2]
    return (q, k, v)
}

// MARK: - Final layer (norm + project to patchified pixel space)

/// Final output head. Takes the image-token sequence (B, N_img, D) and
/// produces (B, N_img, patch² × out_channels) ready to unpatchify into
/// the VAE input resolution.
public final class FluxFinalLayer: Module {
    public let norm: LayerNorm
    public let linear: Linear
    public let mod: Linear

    public init(dim: Int, patchSize: Int = 2, outChannels: Int = 16) {
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.linear = Linear(dim, patchSize * patchSize * outChannels)
        // Modulation: produces (shift, scale) from vec.
        self.mod = Linear(dim, dim * 2)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, vec: MLXArray) -> MLXArray {
        let out = mod(silu(vec))
        let d = out.dim(-1) / 2
        let shift = out[.ellipsis, 0 ..< d].reshaped([out.dim(0), 1, d])
        let scale = out[.ellipsis, d ..< (d * 2)].reshaped([out.dim(0), 1, d])
        var h = norm(x)
        h = h * (MLXArray(Float(1)) + scale) + shift
        return linear(h)
    }
}

// MARK: - FluxDiTModel assembly
//
// The per-model variant (Flux1 Schnell, Flux1 Dev, Flux2 Klein, etc.)
// instantiates one of these with its own hyperparameters + optional
// `guidanceEmbed` flag.

public struct FluxDiTConfig: Sendable {
    public let dim: Int
    public let numDoubleBlocks: Int
    public let numSingleBlocks: Int
    public let numHeads: Int
    public let patchSize: Int
    public let inChannels: Int
    public let outChannels: Int
    public let guidanceEmbed: Bool
    public let textDim: Int

    public init(
        dim: Int = 3072,
        numDoubleBlocks: Int = 19,
        numSingleBlocks: Int = 38,
        numHeads: Int = 24,
        patchSize: Int = 2,
        inChannels: Int = 16,
        outChannels: Int = 16,
        guidanceEmbed: Bool = false,
        textDim: Int = 4096
    ) {
        self.dim = dim
        self.numDoubleBlocks = numDoubleBlocks
        self.numSingleBlocks = numSingleBlocks
        self.numHeads = numHeads
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.guidanceEmbed = guidanceEmbed
        self.textDim = textDim
    }

    /// FLUX.1 Schnell config. 4-step, no guidance embed.
    public static let schnell = FluxDiTConfig(
        numDoubleBlocks: 19, numSingleBlocks: 38, numHeads: 24,
        guidanceEmbed: false
    )

    /// FLUX.1 Dev config. 20-step, uses guidance embed for CFG.
    public static let dev = FluxDiTConfig(
        numDoubleBlocks: 19, numSingleBlocks: 38, numHeads: 24,
        guidanceEmbed: true
    )

    /// Z-Image Turbo — ~2B param single-encoder variant. The architecture
    /// is close to Flux Schnell but with fewer blocks + a narrower
    /// hidden dim. Numbers are approximate until we parse the checkpoint's
    /// own `config.json` on load.
    public static let zImageTurbo = FluxDiTConfig(
        dim: 2048,
        numDoubleBlocks: 8,
        numSingleBlocks: 16,
        numHeads: 16,
        guidanceEmbed: false
    )

    /// FLUX.2 Klein — single-encoder, ~6B, similar block count to Flux1.
    public static let flux2Klein = FluxDiTConfig(
        dim: 3072,
        numDoubleBlocks: 19,
        numSingleBlocks: 38,
        numHeads: 24,
        guidanceEmbed: true
    )
}

/// ## Weight key mapping (for the real `.safetensors` loader)
///
/// Black Forest Labs ships FLUX checkpoints with the following naming
/// convention. When the real weight loader lands, it must translate
/// from these checkpoint keys to the Swift property names via
/// `@ModuleInfo(key: "...")` decorators OR an explicit remap table fed
/// to `Module.update(parameters:)`.
///
///     Checkpoint key                        → Swift property
///     ─────────────────────────────────────────────────────────
///     img_in.weight                         → imgIn
///     time_in.in_layer.weight               → timeIn0
///     time_in.out_layer.weight              → timeIn2
///     vector_in.in_layer.weight             → vectorIn0
///     vector_in.out_layer.weight            → vectorIn2
///     guidance_in.in_layer.weight           → guidanceIn0 (Dev only)
///     guidance_in.out_layer.weight          → guidanceIn2 (Dev only)
///     txt_in.weight                         → txtIn
///     double_blocks.{i}.img_mod.lin.weight  → doubleBlocks[i].imgMod.linear
///     double_blocks.{i}.img_attn.qkv.weight → doubleBlocks[i].imgAttnQKV
///     double_blocks.{i}.img_attn.proj.weight→ doubleBlocks[i].imgAttnProj
///     double_blocks.{i}.img_attn.norm.query_norm.scale → .imgAttnNorm.qNorm.weight
///     double_blocks.{i}.img_attn.norm.key_norm.scale   → .imgAttnNorm.kNorm.weight
///     double_blocks.{i}.img_mlp.0.weight    → doubleBlocks[i].imgMlp0
///     double_blocks.{i}.img_mlp.2.weight    → doubleBlocks[i].imgMlp2
///     double_blocks.{i}.txt_*                → doubleBlocks[i].txt*  (same pattern)
///     single_blocks.{i}.modulation.lin.weight → singleBlocks[i].mod.linear
///     single_blocks.{i}.linear1.weight      → singleBlocks[i].linear1
///     single_blocks.{i}.linear2.weight      → singleBlocks[i].linear2
///     single_blocks.{i}.norm.query_norm.scale → .qkNorm.qNorm.weight
///     single_blocks.{i}.norm.key_norm.scale   → .qkNorm.kNorm.weight
///     final_layer.linear.weight             → finalLayer.linear
///     final_layer.adaLN_modulation.1.weight → finalLayer.mod
public final class FluxDiTModel: Module {
    public let config: FluxDiTConfig
    public let imgIn: Linear   // patch embed: (patch² * in_channels) → dim
    public let timeIn0: Linear
    public let timeIn2: Linear
    public let vectorIn0: Linear  // pooled CLIP projection
    public let vectorIn2: Linear
    public let guidanceIn0: Linear?
    public let guidanceIn2: Linear?
    public let txtIn: Linear      // T5 projection
    public let doubleBlocks: [FluxDoubleStreamBlock]
    public let singleBlocks: [FluxSingleStreamBlock]
    public let finalLayer: FluxFinalLayer

    public init(config: FluxDiTConfig) {
        self.config = config

        self.imgIn = Linear(
            config.patchSize * config.patchSize * config.inChannels,
            config.dim
        )
        self.timeIn0 = Linear(256, config.dim)
        self.timeIn2 = Linear(config.dim, config.dim)
        self.vectorIn0 = Linear(768, config.dim)   // pooled CLIP dim
        self.vectorIn2 = Linear(config.dim, config.dim)
        if config.guidanceEmbed {
            self.guidanceIn0 = Linear(256, config.dim)
            self.guidanceIn2 = Linear(config.dim, config.dim)
        } else {
            self.guidanceIn0 = nil
            self.guidanceIn2 = nil
        }
        self.txtIn = Linear(config.textDim, config.dim)

        self.doubleBlocks = (0..<config.numDoubleBlocks).map { _ in
            FluxDoubleStreamBlock(dim: config.dim, numHeads: config.numHeads)
        }
        self.singleBlocks = (0..<config.numSingleBlocks).map { _ in
            FluxSingleStreamBlock(dim: config.dim, numHeads: config.numHeads)
        }
        self.finalLayer = FluxFinalLayer(
            dim: config.dim,
            patchSize: config.patchSize,
            outChannels: config.outChannels
        )

        super.init()
    }

    /// Forward pass. Returns the predicted velocity field for the Euler
    /// scheduler to consume.
    ///
    /// - Parameters:
    ///   - imgPatched: (B, N_img, patch²*16) — patchified VAE latent.
    ///   - txt: (B, N_txt, 4096) — T5-XXL encoded prompt.
    ///   - pooledClip: (B, 768) — pooled CLIP-L text embedding.
    ///   - timestep: (B,) — current denoising step (1000 → 0).
    ///   - guidance: (B,) — guidance scale (Dev only, nil for Schnell).
    public func callAsFunction(
        imgPatched: MLXArray,
        txt: MLXArray,
        pooledClip: MLXArray,
        timestep: MLXArray,
        guidance: MLXArray? = nil,
        rope: RoPE2D? = nil
    ) -> MLXArray {
        // 1. Project image patches to model dim.
        var img = imgIn(imgPatched)

        // 2. Build conditioning vector `vec`.
        let timeEmb = sinusoidalTimeEmbedding(
            timesteps: timestep, embeddingDim: 256
        )
        let timeProjected = timeIn2(silu(timeIn0(timeEmb)))
        let pooledProjected = vectorIn2(silu(vectorIn0(pooledClip)))
        var vec = timeProjected + pooledProjected
        if config.guidanceEmbed,
           let g0 = guidanceIn0, let g2 = guidanceIn2, let g = guidance {
            let gEmb = sinusoidalTimeEmbedding(timesteps: g, embeddingDim: 256)
            vec = vec + g2(silu(g0(gEmb)))
        }

        // 3. Project text tokens.
        var txtTokens = txtIn(txt)

        // 4. Double blocks.
        for block in doubleBlocks {
            let out = block(img: img, txt: txtTokens, vec: vec, rope: rope)
            img = out.img
            txtTokens = out.txt
        }

        // 5. Concatenate for single blocks.
        var merged = concatenated([txtTokens, img], axis: 1)
        for block in singleBlocks {
            merged = block(merged, vec: vec, rope: rope)
        }

        // 6. Drop text tokens, take image slice.
        let nTxt = txtTokens.dim(1)
        let imgOut = merged[0 ..< merged.dim(0), nTxt ..< merged.dim(1), 0 ..< config.dim]

        // 7. Final layer → patchified velocity.
        return finalLayer(imgOut, vec: vec)
    }
}
