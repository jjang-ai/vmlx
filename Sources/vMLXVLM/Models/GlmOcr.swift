//
//  GlmOcr.swift
//  mlx-swift-lm
//
//  Created by Sachin Desai on 2/15/26.
//
//  port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/glm_ocr
//

import CoreImage
import Foundation
import MLX
import vMLXLMCommon
import MLXNN

// MARK: - Language

private enum Language {

    // MARK: M-RoPE helpers

    /// Interleaved rotate_half: takes even/odd indices instead of splitting in half.
    /// x[..., 0::2] and x[..., 1::2] interleaved.
    static func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
        let lastDim = x.dim(-1)
        let x1 = x[.ellipsis, stride(from: 0, to: lastDim, by: 2)]
        let x2 = x[.ellipsis, stride(from: 1, to: lastDim, by: 2)]
        let neg = -x2
        // Interleave [-x2, x1]: stack on new last axis then flatten last two dims
        let stacked = MLX.stacked([neg, x1], axis: -1)
        return stacked.reshaped(x.shape)
    }

    /// repeat_interleave: [a,b,c] with repeats=2 -> [a,a,b,b,c,c] along axis
    static func repeatInterleave(_ x: MLXArray, repeats: Int, axis: Int) -> MLXArray {
        let resolvedAxis = axis >= 0 ? axis : x.ndim + axis
        let expanded = expandedDimensions(x, axis: resolvedAxis + 1)
        var tileShape = [Int](repeating: 1, count: expanded.ndim)
        tileShape[resolvedAxis + 1] = repeats
        let t = tiled(expanded, repetitions: tileShape)
        var newShape = x.shape
        newShape[resolvedAxis] *= repeats
        return t.reshaped(newShape)
    }

    /// Apply rotary position embedding (language model style - interleaved).
    static func applyRotaryPosEmb(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        // cos, sin: (batch, seq, headDim) -> (batch, 1, seq, headDim)
        var cos = cos[0..., .newAxis, 0..., 0...]
        var sin = sin[0..., .newAxis, 0..., 0...]

        // Take first half of frequencies, repeat-interleave to full headDim
        let halfDim = cos.dim(-1) / 2
        cos = repeatInterleave(cos[.ellipsis, ..<halfDim], repeats: 2, axis: -1)
        sin = repeatInterleave(sin[.ellipsis, ..<halfDim], repeats: 2, axis: -1)

        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cos) + (rotateHalfInterleaved(qRot) * sin)
        let kEmbed = (kRot * cos) + (rotateHalfInterleaved(kRot) * sin)

        return (
            concatenated([qEmbed, qPass], axis: -1),
            concatenated([kEmbed, kPass], axis: -1)
        )
    }

    // MARK: M-RoPE Rotary Embedding

    fileprivate class GlmOcrRotaryEmbedding {
        let mropeSplitIndices: [Int]
        let invFreq: MLXArray
        let attentionScaling: Float

        init(_ config: GlmOcrConfiguration.TextConfiguration) {
            // Pre-compute cumulative split indices from mropeSection (config-derived, never changes)
            var indices = [Int]()
            var cumsum = 0
            for s in config.ropeParameters.mropeSection.dropLast() {
                cumsum += s
                indices.append(cumsum)
            }
            self.mropeSplitIndices = indices

            let dim = Int(Float(config.headDim) * config.ropeParameters.partialRotaryFactor)
            let base = config.ropeParameters.ropeTheta
            self.attentionScaling = 1.0

            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            self.invFreq = 1.0 / pow(base, p)
        }

        /// Apply M-RoPE: select different frequency dimensions for T, H, W.
        func applyMrope(_ freqs: MLXArray) -> MLXArray {
            let chunks = split(freqs, indices: mropeSplitIndices, axis: -1)
            // Select chunk[i % 3] from the temporal dimension (axis 0)
            let selected = chunks.enumerated().map { i, chunk in
                chunk[i % 3]
            }
            return concatenated(selected, axis: -1)
        }

        func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            // positionIds: (3, batch, seq)
            let batchSize = positionIds.dim(1)

            // inv_freq: (dim/2,) -> (1, 1, dim/2, 1)
            var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
            // Broadcast to (3, batch, dim/2, 1)
            invFreqExpanded = broadcast(
                invFreqExpanded,
                to: [3, batchSize, invFreq.dim(0), 1])

            // position_ids: (3, batch, seq) -> (3, batch, 1, seq)
            let positionIdsExpanded = positionIds[0..., 0..., .newAxis, 0...].asType(.float32)

            // freqs = matmul -> (3, batch, dim/2, seq) -> transpose -> (3, batch, seq, dim/2)
            let freqs = matmul(invFreqExpanded, positionIdsExpanded).transposed(0, 1, 3, 2)

            // Apply M-RoPE section selection -> (batch, seq, dim/2)
            let mropeFreqs = applyMrope(freqs)

            // Double frequencies and compute cos/sin
            let emb = concatenated([mropeFreqs, mropeFreqs], axis: -1)
            let cos = MLX.cos(emb) * attentionScaling
            let sin = MLX.sin(emb) * attentionScaling

            return (cos.asType(x.dtype), sin.asType(x.dtype))
        }
    }

    // MARK: Attention

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        public init(_ args: GlmOcrConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = args.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray)
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // MARK: MLP

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gateUpProj.wrappedValue = Linear(dimensions, hiddenDimensions * 2, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let x = gateUpProj(x)
            let parts = split(x, parts: 2, axis: -1)
            return down(silu(parts[0]) * parts[1])
        }
    }

    // MARK: Decoder Layer

    fileprivate class GlmOcrDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm

        public init(_ args: GlmOcrConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postSelfAttnLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray)
        ) -> MLXArray {
            var r = x
            var h = attention(
                inputLayerNorm(x), mask: mask, cache: cache,
                positionEmbeddings: positionEmbeddings)
            h = postSelfAttnLayerNorm(h)
            h = r + h
            r = h
            h = postAttentionLayerNorm(h)
            h = mlp(h)
            h = postMlpLayerNorm(h)
            h = r + h
            return h
        }
    }

    // MARK: Text Model

    fileprivate class GlmOcrTextModel: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [GlmOcrDecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmb: GlmOcrRotaryEmbedding

        public init(_ args: GlmOcrConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in GlmOcrDecoderLayer(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.rotaryEmb = GlmOcrRotaryEmbedding(args)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            // Compute position_ids if not provided (autoregressive generation)
            var posIds: MLXArray
            if let positionIds {
                posIds = positionIds
            } else {
                let offset = cache?.first?.offset ?? 0
                let seqLen = h.dim(h.ndim - 2)
                let positions = MLXArray(Int32(offset) ..< Int32(offset + seqLen))
                    .expandedDimensions(axis: 0)
                posIds = tiled(positions, repetitions: [3, 1, 1])
            }

            let positionEmbeddings = rotaryEmb(h, positionIds: posIds)
            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h, mask: mask, cache: cache?[i],
                    positionEmbeddings: positionEmbeddings)
            }

            return norm(h)
        }
    }

    // MARK: Language Model

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: GlmOcrTextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var _positionIds: MLXArray?
        var _ropeDeltas: MLXArray?

        public init(_ args: GlmOcrConfiguration.TextConfiguration) {
            self.model = GlmOcrTextModel(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var positionIds: MLXArray? = nil

            let cacheOffset: Int
            if let cache = cache, let first = cache.first {
                cacheOffset = first.offset
            } else {
                cacheOffset = 0
            }

            if let storedPositionIds = _positionIds {
                let seqLen: Int
                if let inputEmbedding {
                    seqLen = inputEmbedding.dim(inputEmbedding.ndim - 2)
                } else if let inputs {
                    seqLen = inputs.dim(inputs.ndim - 1)
                } else {
                    seqLen = 0
                }

                let storedLen = storedPositionIds.dim(2)
                if cacheOffset + seqLen <= storedLen {
                    // Prefill: use stored M-RoPE position IDs
                    positionIds =
                        storedPositionIds[
                            0..., 0..., cacheOffset ..< (cacheOffset + seqLen)]
                } else {
                    // Autoregressive: compute sequential positions using rope_deltas
                    let delta = _ropeDeltas ?? MLXArray(Int32(0))
                    let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
                    var posArrays = [MLXArray]()
                    for _ in 0 ..< 3 {
                        let pos = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                            .expandedDimensions(axis: 0)
                        let tiledPos = tiled(pos, repetitions: [batchSize, 1])
                        posArrays.append((tiledPos + delta).expandedDimensions(axis: 0))
                    }
                    positionIds = concatenated(posArrays, axis: 0)
                }
            }

            var out = model(
                inputs, cache: cache, inputEmbedding: inputEmbedding,
                positionIds: positionIds)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {

    static fileprivate func applyRotaryPosEmbVision(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cosVal = MLX.cos(freqs)
        var sinVal = MLX.sin(freqs)

        // freqs: (seq, dim/2) -> expand to (seq, 1, dim) for broadcasting with (seq, heads, dim)
        cosVal = expandedDimensions(cosVal, axis: 1)
        cosVal = tiled(cosVal, repetitions: [1, 1, 2])

        sinVal = expandedDimensions(sinVal, axis: 1)
        sinVal = tiled(sinVal, repetitions: [1, 1, 2])

        let output = (tensor * cosVal) + (QwenVL.rotateHalf(tensor) * sinVal)
        return output.asType(tensor.dtype)
    }

    fileprivate class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let embedDim: Int

        init(_ config: GlmOcrConfiguration.VisionConfiguration) {
            self.patchSize = config.patchSize
            self.temporalPatchSize = config.temporalPatchSize
            self.inChannels = config.inChannels
            self.embedDim = config.hiddenSize

            let kernelSize = IntOrTriple(
                [config.temporalPatchSize, config.patchSize, config.patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: config.inChannels,
                outputChannels: config.hiddenSize,
                kernelSize: kernelSize,
                stride: kernelSize,
                bias: true
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var h = hiddenStates.reshaped(
                -1, inChannels, temporalPatchSize, patchSize, patchSize
            ).movedAxis(source: 1, destination: 4)

            h = proj(h)
            h = h.reshaped(-1, embedDim)
            return h
        }
    }

    fileprivate class Attention: Module {

        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        public init(_ config: GlmOcrConfiguration.VisionConfiguration) {
            self.numHeads = config.numHeads
            self.headDim = config.hiddenSize / config.numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize * 3, bias: true)
            self._proj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: true)
            self._qNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            self._kNorm.wrappedValue = RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkvOut = qkv(x)
            let qkvReshaped = qkvOut.reshaped(sequenceLength, 3, numHeads, -1)
                .transposed(1, 0, 2, 3)
            let parts = split(qkvReshaped, parts: 3, axis: 0)
            var q = parts[0].squeezed(axis: 0)
            var k = parts[1].squeezed(axis: 0)
            let v = parts[2].squeezed(axis: 0)

            q = qNorm(q)
            k = kNorm(k)

            q = applyRotaryPosEmbVision(q, freqs: rotaryPositionEmbedding)
            k = applyRotaryPosEmbVision(k, freqs: rotaryPositionEmbedding)

            // Reshape for attention: (seq, heads, dim) -> (1, heads, seq, dim)
            let qT = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kT = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vT = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

            // Per-image attention using cu_seqlens
            var attnOutputs = [MLXArray]()
            for i in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[i]
                let end = cuSeqlens[i + 1]
                let qChunk = qT[0..., 0..., start ..< end, 0...]
                let kChunk = kT[0..., 0..., start ..< end, 0...]
                let vChunk = vT[0..., 0..., start ..< end, 0...]
                let output = MLXFast.scaledDotProductAttention(
                    queries: qChunk, keys: kChunk, values: vChunk,
                    scale: scale, mask: .none)
                attnOutputs.append(output)
            }

            let attnOutput = concatenated(attnOutputs, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)

            return proj(attnOutput)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(_ config: GlmOcrConfiguration.VisionConfiguration) {
            self._gate.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            self._up.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            self._down.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: true)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class GlmOcrVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: GlmOcrConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, cuSeqlens: [Int], rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate class PatchMerger: Module, UnaryLayer {

        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "post_projection_norm") var postProjectionNorm: LayerNorm
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dim: Int, contextDim: Int) {
            self._proj.wrappedValue = Linear(dim, dim, bias: false)
            self._postProjectionNorm.wrappedValue = LayerNorm(dimensions: dim)
            self._gate.wrappedValue = Linear(dim, contextDim, bias: false)
            self._up.wrappedValue = Linear(dim, contextDim, bias: false)
            self._down.wrappedValue = Linear(contextDim, dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = proj(x)
            h = gelu(postProjectionNorm(h))
            return down(silu(gate(h)) * up(h))
        }
    }

    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let rotaryPosEmb: QwenVL.VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [GlmOcrVisionBlock]
        @ModuleInfo var downsample: Conv2d
        @ModuleInfo var merger: PatchMerger
        @ModuleInfo(key: "post_layernorm") var postLayernorm: RMSNorm

        let spatialMergeSize: Int

        public init(_ config: GlmOcrConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize

            self._patchEmbed.wrappedValue = PatchEmbed(config)

            let headDim = config.hiddenSize / config.numHeads
            self.rotaryPosEmb = QwenVL.VisionRotaryEmbedding(
                dimensions: headDim / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                GlmOcrVisionBlock(config)
            }

            self._downsample.wrappedValue = Conv2d(
                inputChannels: config.hiddenSize,
                outputChannels: config.outHiddenSize,
                kernelSize: IntOrPair(config.spatialMergeSize),
                stride: IntOrPair(config.spatialMergeSize),
                bias: true)

            self._merger.wrappedValue = PatchMerger(
                dim: config.outHiddenSize,
                contextDim: config.outHiddenSize * config.inChannels)

            self._postLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPosEmbFull = rotaryPosEmb(sequenceLength: maxFrameSize)[indices]

            return rotaryPosEmbFull.reshaped(indices.dim(0), -1)
        }

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(hiddenStates)
            let rotaryPosEmbedding = rotaryPositionEmbedding(frames)

            // Compute cu_seqlens from frames
            var cuSeqlens = [0]
            var cumsum = 0
            for frame in frames {
                for _ in 0 ..< frame.t {
                    cumsum += frame.h * frame.w
                    cuSeqlens.append(cumsum)
                }
            }

            for block in blocks {
                hiddenStates = block(
                    hiddenStates, cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPosEmbedding)
            }

            hiddenStates = postLayernorm(hiddenStates)

            // Spatial merge via Conv2d downsample
            let hiddenDim = hiddenStates.dim(-1)
            hiddenStates = hiddenStates.reshaped(
                -1, spatialMergeSize, spatialMergeSize, hiddenDim)
            hiddenStates = downsample(hiddenStates).reshaped(
                -1, downsample.weight.dim(0))

            hiddenStates = merger(hiddenStates)
            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim == 4 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
                return outChannels >= kH && outChannels >= kW && kH == kW
            } else if array.ndim == 5 {
                let (outChannels, kH, kW) = (array.dim(0), array.dim(2), array.dim(3))
                return outChannels >= kH && outChannels >= kW && kH == kW
            }
            return false
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    continue
                } else if k.contains("patch_embed.proj.weight")
                    || k.contains("downsample.weight")
                {
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        if v.ndim == 5 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                        } else if v.ndim == 4 {
                            sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                        } else {
                            sanitizedWeights[k] = v
                        }
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Processor

/// GlmOcr VLM `UserInputProcessor`.
public struct GlmOcrProcessor: UserInputProcessor {
    private let config: GlmOcrProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: GlmOcrProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages = images.map { image in
            preprocess(image: image, resizedSize: resizedSize).asMLXArray()
        }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    /// Replace single image placeholder tokens with the correct count based on grid dimensions.
    private func replacePaddingTokens(
        in promptTokens: [Int], frames: [THW]
    ) throws -> [Int] {
        let paddingToken = "<|image|>"
        let placeholderTokens = tokenizer.encode(
            text: "<|begin_of_image|>\(paddingToken)<|end_of_image|>")
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens (\(placeholderRanges.count)) does not match number of frames (\(frames.count))"
            )
        }
        let mergeLength = config.mergeSize * config.mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|begin_of_image|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|end_of_image|>"
            )
        }
        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = GlmOcrMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        // Text-only input
        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        // Process images
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [try $0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })
            if let imageFrames = processedImage?.frames {
                promptTokens = try replacePaddingTokens(
                    in: promptTokens, frames: imageFrames)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage)
    }
}

// MARK: - Model

/// GlmOcr VLM
public class GlmOcr: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: GlmOcrConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: GlmOcrConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    /// Compute 3D M-RoPE position IDs for prefill with images.
    /// Returns position_ids (3, batch, seq) and rope_deltas.
    private func getRopeIndex(
        inputIds: MLXArray, imageGridThw: [THW]?
    ) -> (MLXArray, MLXArray) {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(1)
        let spatialMergeSize = config.visionConfiguration.spatialMergeSize
        let imageTokenId = config.baseConfiguration.imageTokenId

        guard let imageGridThw, !imageGridThw.isEmpty else {
            // Text only: sequential positions tiled 3x
            let positions = MLXArray(0 ..< Int32(seqLength)).expandedDimensions(axis: 0)
            let positionIds = tiled(
                broadcast(positions, to: [batchSize, seqLength]).expandedDimensions(axis: 0),
                repetitions: [3, 1, 1])
            let deltas = MLXArray(Int32(0))
            return (positionIds, deltas)
        }

        // Build position_ids per batch element using 3 separate dimension arrays
        precondition(batchSize == 1, "GlmOcr getRopeIndex only supports batchSize == 1")
        let positionIds = zeros([3, batchSize, seqLength], type: Int32.self)
        var imageIndex = 0
        var mropePositionDelta: Int = 0

        for batchIdx in 0 ..< batchSize {
            let inputTokens: [Int32] = inputIds[batchIdx].asArray(Int32.self)

            // Keep 3 separate arrays for T, H, W dimensions
            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(seqLength)
            dimH.reserveCapacity(seqLength)
            dimW.reserveCapacity(seqLength)
            var st = 0
            var lastMax: Int32 = -1

            // Append sequential text positions to all 3 dimensions
            let appendTextPositions = { (count: Int) in
                guard count > 0 else { return }
                let base: Int32 = lastMax + 1
                for j in 0 ..< count {
                    let pos = base + Int32(j)
                    dimT.append(pos)
                    dimH.append(pos)
                    dimW.append(pos)
                }
                lastMax = base + Int32(count) - 1
            }

            // Process images: find each image token and build 3D positions
            while imageIndex < imageGridThw.count {
                guard let ed = inputTokens[st...].firstIndex(of: Int32(imageTokenId)) else {
                    break
                }

                let frame = imageGridThw[imageIndex]
                let llmGridT = frame.t
                let llmGridH = frame.h / spatialMergeSize
                let llmGridW = frame.w / spatialMergeSize
                imageIndex += 1

                // Text before this image
                appendTextPositions(ed - st)

                // Image tokens: 3D spatial positions
                let imgOffset: Int32 = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(Int32(t) + imgOffset)
                            dimH.append(Int32(h) + imgOffset)
                            dimW.append(Int32(w) + imgOffset)
                        }
                    }
                }
                let tMax = Int32(llmGridT - 1) + imgOffset
                let hMax = Int32(llmGridH - 1) + imgOffset
                let wMax = Int32(llmGridW - 1) + imgOffset
                lastMax = max(tMax, max(hMax, wMax))

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Remaining text after last image
            appendTextPositions(inputTokens.count - st)

            positionIds[0, batchIdx] = MLXArray(dimT)
            positionIds[1, batchIdx] = MLXArray(dimH)
            positionIds[2, batchIdx] = MLXArray(dimW)

            mropePositionDelta = Int(lastMax) + 1 - inputTokens.count
        }

        let deltas = MLXArray(Int32(mropePositionDelta))
        return (positionIds, deltas)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            languageModel._positionIds = nil
            languageModel._ropeDeltas = nil
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        var hiddenStates = self.visionModel(pixelValues, frames: frames)

        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        let merged = QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: hiddenStates,
            imageTokenId: config.baseConfiguration.imageTokenId,
            videoTokenId: config.baseConfiguration.videoTokenId)

        // Pre-compute M-RoPE position IDs
        let (positionIds, ropeDeltas) = getRopeIndex(
            inputIds: inputIds, imageGridThw: frames)
        languageModel._positionIds = positionIds
        languageModel._ropeDeltas = ropeDeltas

        return merged
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var allPixels: MLXArray?
        var allFrames: [THW] = []

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        // Chunked prefill — see Qwen25VL / vmlx #50/#51.
        let result = chunkedPrefillEmbedding(
            inputEmbedding: inputEmbeddings,
            cache: cache,
            prefillStepSize: windowSize ?? 512
        ) { chunk in
            languageModel(nil, cache: cache, inputEmbedding: chunk)
        }

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Step 1: Transform keys from HuggingFace format to internal format
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var k = key

            // Map visual -> vision_tower
            if k.contains("visual") && !k.contains("vision_tower") {
                k = k.replacingOccurrences(of: "model.", with: "")
                k = k.replacingOccurrences(of: "visual", with: "vision_tower")
            }

            // Map model.language_model -> language_model.model
            if k.contains("model.language_model") {
                k = k.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            // Map lm_head -> language_model.lm_head
            if k.contains("lm_head") && !k.hasPrefix("language_model") {
                k = k.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            // The checkpoint includes a "next-n" prediction head at the layer index
            // equal to hiddenLayers (one past the last real decoder layer). Skip it.
            if k.contains("layers.\(config.textConfiguration.hiddenLayers).") {
                continue
            }

            transformed[k] = value
        }

        // Step 2: Sanitize vision weights (conv weight transposes)
        return visionModel.sanitize(weights: transformed)
    }
}

// MARK: - Configuration

/// Configuration for ``GlmOcr``
public struct GlmOcrConfiguration: Codable, Sendable {

    public struct RopeParameters: Codable, Sendable {
        public let mropeSection: [Int]
        private let _partialRotaryFactor: Float?
        public var partialRotaryFactor: Float { _partialRotaryFactor ?? 1.0 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case _partialRotaryFactor = "partial_rotary_factor"
            case _ropeTheta = "rope_theta"
        }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vocabularySize: Int
        public let ropeParameters: RopeParameters
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
        public var ropeTheta: Float { ropeParameters.ropeTheta }
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case ropeParameters = "rope_parameters"
            case _rmsNormEps = "rms_norm_eps"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let outHiddenSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        private let _inChannels: Int?
        public var inChannels: Int { _inChannels ?? 3 }
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case outHiddenSize = "out_hidden_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case _inChannels = "in_channels"
            case _rmsNormEps = "rms_norm_eps"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        private let _vocabularySize: Int?
        private let _imageTokenId: Int?
        private let _videoTokenId: Int?
        private let _imageStartTokenId: Int?
        private let _hiddenSize: Int?

        public var vocabularySize: Int { _vocabularySize ?? 59392 }
        public var imageTokenId: Int { _imageTokenId ?? 59280 }
        public var videoTokenId: Int { _videoTokenId ?? 59281 }
        public var imageStartTokenId: Int { _imageStartTokenId ?? 59256 }
        public var hiddenSize: Int { _hiddenSize ?? 1536 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _vocabularySize = "vocab_size"
            case _imageTokenId = "image_token_id"
            case _videoTokenId = "video_token_id"
            case _imageStartTokenId = "image_start_token_id"
            case _hiddenSize = "hidden_size"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.textConfiguration = try container.decode(
            TextConfiguration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // BaseConfiguration overlaid at top level (fields may be absent)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``GlmOcrProcessor``
public struct GlmOcrProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let shortestEdge: Int
        public let longestEdge: Int

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let size: Size

    public var minPixels: Int { size.shortestEdge }
    public var maxPixels: Int { size.longestEdge }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case size
    }
}

// MARK: - Message Generator

/// Message Generator for GlmOcr
public struct GlmOcrMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> vMLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                + message.images.map { _ in
                    ["type": "image"]
                },
        ]
    }
}
