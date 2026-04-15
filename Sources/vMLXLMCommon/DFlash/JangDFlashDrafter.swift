//
//  JangDFlashDrafter.swift
//  vMLXLMCommon / DFlash
//
//  5-layer block-diffusion drafter with KV injection of target hidden
//  states. One-step denoising at inference time — emits B parallel
//  candidate distributions in a single forward pass.
//
//  Architecture mirrors `jang_tools.dflash.drafter.JangDFlashDrafter`
//  (PyTorch). Keep parameter keys in sync so PT → MLX safetensors round
//  trip works via the existing JangLoader.
//
//  References:
//    - DFlash arXiv 2602.06036 (KV-injection conditioning, 1-step decode)
//    - BD3-LM arXiv 2503.09573 (masked absorbing-state kernel)
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Attention with KV injection

final class JangDFlashAttention: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "wk_ctx") var wkCtx: Linear
    @ModuleInfo(key: "wv_ctx") var wvCtx: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let cfg: JangDFlashConfig
    let scale: Float
    let rope: RoPE

    init(_ cfg: JangDFlashConfig) {
        self.cfg = cfg
        self.scale = pow(Float(cfg.headDim), -0.5)
        self.rope = RoPE(dimensions: cfg.headDim, traditional: false, base: cfg.ropeTheta)

        let qOut = cfg.numHeads * cfg.headDim
        let kvOut = cfg.numKVHeads * cfg.headDim
        self._wq.wrappedValue = Linear(cfg.hiddenDim, qOut, bias: false)
        self._wk.wrappedValue = Linear(cfg.hiddenDim, kvOut, bias: false)
        self._wv.wrappedValue = Linear(cfg.hiddenDim, kvOut, bias: false)
        self._wo.wrappedValue = Linear(qOut, cfg.hiddenDim, bias: false)
        self._wkCtx.wrappedValue = Linear(cfg.hiddenDim, kvOut, bias: false)
        self._wvCtx.wrappedValue = Linear(cfg.hiddenDim, kvOut, bias: false)

        // Normalize the full (heads flattened) K/Q — matches Olmo3/Qwen3
        // pattern in this codebase. The head dim is implicit in how we
        // reshape afterwards.
        self._qNorm.wrappedValue = RMSNorm(dimensions: qOut, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: kvOut, eps: cfg.rmsNormEps)

        super.init()
    }

    /// `x` is the block hidden-state stream, shape `[B, L, hidden]`.
    /// `hCtxKV` is the injected context, shape `[B, T_ctx, hidden]`.
    ///
    /// The block positions attend bidirectionally to the entire context
    /// and causally within themselves. The context carries its own
    /// positional information from the fusion MLP — we apply RoPE only
    /// to the block's own Q/K.
    func callAsFunction(_ x: MLXArray, hCtxKV: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let Tctx = hCtxKV.dim(1)

        // Block-side Q/K/V.
        var queries = qNorm(wq(x))
        var keys = kNorm(wk(x))
        let values = wv(x)

        queries = queries.reshaped(B, L, cfg.numHeads, cfg.headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, cfg.numKVHeads, cfg.headDim).transposed(0, 2, 1, 3)
        let valuesHeaded = values.reshaped(B, L, cfg.numKVHeads, cfg.headDim)
            .transposed(0, 2, 1, 3)

        // RoPE on block positions only.
        queries = rope(queries)
        keys = rope(keys)

        // Context K/V from a separate projection pair — input is the
        // fused target hidden, not the block's own embeddings, so it
        // needs its own weights.
        var kCtx = kNorm(wkCtx(hCtxKV))
        let vCtx = wvCtx(hCtxKV)
        kCtx = kCtx.reshaped(B, Tctx, cfg.numKVHeads, cfg.headDim).transposed(0, 2, 1, 3)
        let vCtxHeaded = vCtx.reshaped(B, Tctx, cfg.numKVHeads, cfg.headDim)
            .transposed(0, 2, 1, 3)

        // Concatenate on the time axis, context first. Shape
        // `[B, numKVHeads, Tctx + L, headDim]`.
        let kFull = concatenated([kCtx, keys], axis: 2)
        let vFull = concatenated([vCtxHeaded, valuesHeaded], axis: 2)

        // Mask: block row i can attend to all context positions AND to
        // block positions ≤ i (causal within block). Shape `[L, Tctx + L]`,
        // additive bias (0 for visible, -inf for masked).
        //
        // Build as two sub-blocks and concat:
        //   ctx_part: zeros[L, Tctx]        — block sees all context
        //   causal_part: triu[L, L] with -inf strictly above diag — block
        //                sees only itself and earlier block positions
        let ctxPart = MLXArray.zeros([L, Tctx], dtype: .float32)
        let causalRaw = MLXArray.full([L, L], values: MLXArray(Float(-1e9)))
        // Upper triangle (strictly above diagonal) gets the -inf mass.
        // The lower triangle + diagonal stays at 0.
        let causalMask = triu(causalRaw, k: 1)
        let blockMask = concatenated([ctxPart, causalMask], axis: 1)

        let out = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: kFull,
            values: vFull,
            scale: scale,
            mask: .array(blockMask)
        )
        // SDPA returns `[B, numHeads, L, headDim]`.
        let flat = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(flat)
    }
}

// MARK: - SwiGLU FFN

final class JangDFlashFFN: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(_ cfg: JangDFlashConfig) {
        self._w1.wrappedValue = Linear(cfg.hiddenDim, cfg.ffnDim, bias: false)
        self._w2.wrappedValue = Linear(cfg.ffnDim, cfg.hiddenDim, bias: false)
        self._w3.wrappedValue = Linear(cfg.hiddenDim, cfg.ffnDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Transformer block

final class JangDFlashBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: JangDFlashAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: JangDFlashFFN

    init(_ cfg: JangDFlashConfig) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenDim, eps: cfg.rmsNormEps)
        self._attn.wrappedValue = JangDFlashAttention(cfg)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenDim, eps: cfg.rmsNormEps)
        self._ffn.wrappedValue = JangDFlashFFN(cfg)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, hCtxKV: MLXArray) -> MLXArray {
        var y = x + attn(attnNorm(x), hCtxKV: hCtxKV)
        y = y + ffn(ffnNorm(y))
        return y
    }
}

// MARK: - Fusion MLP (tap → hidden_dim)

/// The fusion MLP takes the raw concatenation of `numTapLayers` hidden
/// states from the target model and produces the drafter's per-token
/// context feature `h_ctx`.
///
/// The shape contract is:
///   input:  `[B, T_ctx, tapDim]`  (tap_dim = numTapLayers * targetHidden)
///   output: `[B, T_ctx, hiddenDim]`
final class JangDFlashFusion: Module {
    @ModuleInfo(key: "0") var proj0: Linear
    // index 1 is the SiLU activation — no parameters
    @ModuleInfo(key: "2") var proj1: Linear
    @ModuleInfo(key: "3") var outNorm: RMSNorm

    init(_ cfg: JangDFlashConfig) {
        self._proj0.wrappedValue = Linear(cfg.tapDim, cfg.hiddenDim * 2, bias: false)
        self._proj1.wrappedValue = Linear(cfg.hiddenDim * 2, cfg.hiddenDim, bias: false)
        self._outNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenDim, eps: cfg.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        outNorm(proj1(silu(proj0(h))))
    }
}

// MARK: - Drafter

public final class JangDFlashDrafter: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "fusion_mlp") var fusionMLP: JangDFlashFusion
    @ModuleInfo(key: "layers") var layers: [JangDFlashBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let cfg: JangDFlashConfig

    public init(_ cfg: JangDFlashConfig) {
        self.cfg = cfg
        self._embed.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize + 1, dimensions: cfg.hiddenDim)
        self._fusionMLP.wrappedValue = JangDFlashFusion(cfg)
        self._layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in JangDFlashBlock(cfg) }
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenDim, eps: cfg.rmsNormEps)
        self._lmHead.wrappedValue = Linear(cfg.hiddenDim, cfg.vocabSize, bias: false)
        super.init()
    }

    /// One-step denoising forward. `blockIDs` is `[B, L]` (anchor at
    /// position 0, MASK at positions 1..L-1). Supply exactly one of
    /// `hTaps` (raw tap concatenation — runs through the fusion MLP)
    /// or `hCtxKV` (pre-fused, skips the fusion MLP).
    public func callAsFunction(
        _ blockIDs: MLXArray,
        hTaps: MLXArray? = nil,
        hCtxKV: MLXArray? = nil
    ) -> MLXArray {
        precondition(
            (hTaps == nil) != (hCtxKV == nil),
            "JangDFlashDrafter: provide exactly one of hTaps or hCtxKV"
        )
        let fused: MLXArray
        if let hCtxKV {
            fused = hCtxKV
        } else {
            fused = fusionMLP(hTaps!)
        }
        var x = embed(blockIDs)
        for layer in layers {
            x = layer(x, hCtxKV: fused)
        }
        return lmHead(norm(x))
    }
}
