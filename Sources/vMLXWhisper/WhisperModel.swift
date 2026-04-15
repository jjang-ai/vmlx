// Copyright © 2026 vMLX. Native MLX-Swift port of the Whisper
// encoder-decoder transformer — matches mlx-examples/whisper's Python
// `whisper/whisper.py` layout so safetensors checkpoints from
// `mlx-community/whisper-*` load without a rename table.
//
// Weight-key map (mlx-whisper Python → this file):
//
//   encoder.conv1.(weight|bias)      → encoder.conv1
//   encoder.conv2.(weight|bias)      → encoder.conv2
//   encoder.positional_embedding     → encoder.positionalEmbedding (buffer)
//   encoder.blocks.N.attn_ln         → encoder.blocks[N].attnLn
//   encoder.blocks.N.attn.query      → encoder.blocks[N].attn.query
//   encoder.blocks.N.attn.key        → encoder.blocks[N].attn.key
//   encoder.blocks.N.attn.value      → encoder.blocks[N].attn.value
//   encoder.blocks.N.attn.out        → encoder.blocks[N].attn.out
//   encoder.blocks.N.mlp_ln          → encoder.blocks[N].mlpLn
//   encoder.blocks.N.mlp.0           → encoder.blocks[N].mlp1
//   encoder.blocks.N.mlp.2           → encoder.blocks[N].mlp2
//   encoder.ln_post                  → encoder.lnPost
//
// Decoder mirrors encoder with the addition of cross_attn(_ln).
//
// Because mlx-whisper stores MLP as `nn.Sequential(Linear, GELU,
// Linear)`, its safetensors keys are suffixed `.mlp.0.weight` and
// `.mlp.2.weight`. We translate those in `sanitizeWeights(_:)` below.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Attention

/// Whisper self/cross attention block. Mirrors mlx-whisper's
/// `MultiHeadAttention` (not to be confused with MLXNN's version) — the
/// projections are named `query`, `key`, `value`, `out` so safetensors
/// keys map directly.
final class WhisperAttention: Module {
    let nHead: Int
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(nState: Int, nHead: Int) {
        self.nHead = nHead
        // Whisper uses bias on query/value/out, and NO bias on key.
        // (OpenAI whisper, see whisper/model.py.)
        self._query.wrappedValue = Linear(nState, nState, bias: true)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState, bias: true)
        self._out.wrappedValue = Linear(nState, nState, bias: true)
    }

    /// Self-attention / cross-attention. When `xa == nil`, acts as self-
    /// attention on `x`. When `xa != nil`, keys/values come from `xa`.
    /// `mask` is an additive mask broadcastable to the attention shape.
    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let q = query(x)
        let source = xa ?? x
        let k = key(source)
        let v = value(source)

        let (b, nQ, _) = (q.dim(0), q.dim(1), q.dim(2))
        let nK = k.dim(1)
        let headDim = q.dim(-1) / nHead

        let qH = q.reshaped([b, nQ, nHead, headDim]).transposed(0, 2, 1, 3)
        let kH = k.reshaped([b, nK, nHead, headDim]).transposed(0, 2, 1, 3)
        let vH = v.reshaped([b, nK, nHead, headDim]).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(headDim))
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask { .array(mask) } else { .none }

        let attn = MLXFast.scaledDotProductAttention(
            queries: qH, keys: kH, values: vH, scale: scale, mask: maskMode)

        let merged = attn.transposed(0, 2, 1, 3).reshaped([b, nQ, nHead * headDim])
        return out(merged)
    }
}

// MARK: - Residual block

final class ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: WhisperAttention
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm

    @ModuleInfo(key: "cross_attn") var crossAttn: WhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttnLn: LayerNorm?

    // mlx-whisper stores the MLP as a two-linear Sequential, so the
    // checkpoint keys are `.mlp.0` and `.mlp.2`. We declare them with
    // the literal key names via `sanitizeWeights`.
    @ModuleInfo(key: "mlp1") var mlp1: Linear
    @ModuleInfo(key: "mlp2") var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(nState: Int, nHead: Int, crossAttention: Bool) {
        self._attn.wrappedValue = WhisperAttention(nState: nState, nHead: nHead)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)
        if crossAttention {
            self._crossAttn.wrappedValue =
                WhisperAttention(nState: nState, nHead: nHead)
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
        }
        let nMlp = nState * 4
        self._mlp1.wrappedValue = Linear(nState, nMlp, bias: true)
        self._mlp2.wrappedValue = Linear(nMlp, nState, bias: true)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x + attn(attnLn(x), mask: mask)
        if let crossAttn, let crossAttnLn, let xa {
            h = h + crossAttn(crossAttnLn(h), xa: xa)
        }
        let mlpOut = mlp2(gelu(mlp1(mlpLn(h))))
        return h + mlpOut
    }
}

// MARK: - Audio encoder

public final class AudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "blocks") var blocks: [ResidualAttentionBlock]

    // Sinusoidal positional embedding is a non-trainable buffer in the
    // original model and is stored in the checkpoint as a plain tensor
    // under the key `encoder.positional_embedding`. We expose it as a
    // Module parameter so `update(parameters:)` will populate it.
    @ParameterInfo(key: "positional_embedding")
    var positionalEmbedding: MLXArray

    init(nMels: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels, outputChannels: nState,
            kernelSize: 3, stride: 1, padding: 1, bias: true)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState,
            kernelSize: 3, stride: 2, padding: 1, bias: true)
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            ResidualAttentionBlock(
                nState: nState, nHead: nHead, crossAttention: false)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: nState)
        // Whisper's encoder positional_embedding is a *non-trainable
        // sinusoidal buffer*, not a saved checkpoint tensor. mlx-whisper
        // Python builds it at construction time (`sinusoids(nCtx, nState)`)
        // and so must we — npz/safetensors dumps never include it.
        self._positionalEmbedding.wrappedValue =
            Self.sinusoids(length: nCtx, channels: nState)
    }

    /// Port of mlx-whisper/audio.py `sinusoids(length, channels)`.
    /// Returns a `[length, channels]` float32 array of fixed sinusoidal
    /// position encodings matching OpenAI's reference whisper.
    static func sinusoids(
        length: Int, channels: Int, maxTimescale: Float = 10_000
    ) -> MLXArray {
        precondition(channels % 2 == 0, "whisper sinusoid channels must be even")
        let half = channels / 2
        let logTimescaleIncrement = log(maxTimescale) / Float(half - 1)
        var invTimescales = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            invTimescales[i] = exp(-logTimescaleIncrement * Float(i))
        }
        var values = [Float](repeating: 0, count: length * channels)
        for t in 0 ..< length {
            for i in 0 ..< half {
                let scaled = Float(t) * invTimescales[i]
                values[t * channels + i] = sin(scaled)
                values[t * channels + half + i] = cos(scaled)
            }
        }
        return MLXArray(values, [length, channels])
    }

    /// `mel` shape `[B, nMels, T]`. Returns `[B, nCtx, nState]`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // MLX-Swift Conv1d expects NLC (channels last), so we transpose
        // the (B, nMels, T) input to (B, T, nMels).
        let melNLC = mel.transposed(0, 2, 1)
        var x = gelu(conv1(melNLC))
        x = gelu(conv2(x))
        // After conv2 (stride 2), sequence length == nAudioCtx.
        x = x + positionalEmbedding.asType(x.dtype)
        for block in blocks {
            x = block(x)
        }
        return lnPost(x)
    }
}

// MARK: - Text decoder

public final class TextDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    @ParameterInfo(key: "positional_embedding")
    var positionalEmbedding: MLXArray

    let nTextCtx: Int

    init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int) {
        self.nTextCtx = nCtx
        self._tokenEmbedding.wrappedValue =
            Embedding(embeddingCount: nVocab, dimensions: nState)
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            ResidualAttentionBlock(
                nState: nState, nHead: nHead, crossAttention: true)
        }
        self._ln.wrappedValue = LayerNorm(dimensions: nState)
        self._positionalEmbedding.wrappedValue =
            MLXArray.zeros([nCtx, nState])
    }

    /// `tokens` shape `[B, T]`, `audioFeatures` shape `[B, nAudioCtx, nState]`.
    /// `offset` is the position offset for the text positional embedding
    /// when decoding one token at a time against accumulated prefix.
    public func callAsFunction(
        _ tokens: MLXArray,
        audioFeatures: MLXArray,
        offset: Int = 0
    ) -> MLXArray {
        let t = tokens.dim(1)
        let posSlice = positionalEmbedding[offset ..< (offset + t)]
        var x = tokenEmbedding(tokens) + posSlice.asType(tokenEmbedding(tokens).dtype)

        // Additive causal mask. Must share dtype with x so that
        // MLXFast.scaledDotProductAttention's mask promotion check passes
        // (e.g. fp16 weights → fp16 mask).
        let mask = Self.causalMask(t, dtype: x.dtype)

        for block in blocks {
            x = block(x, xa: audioFeatures, mask: mask)
        }
        x = ln(x)
        // Weight tying: logits = x @ token_embedding.weight.T
        let logits = matmul(x, tokenEmbedding.weight.transposed())
        return logits
    }

    static func causalMask(_ n: Int, dtype: DType = .float16) -> MLXArray {
        if n <= 1 { return MLXArray.zeros([1, 1], dtype: dtype) }
        let indices = MLXArray(0 ..< n)
        var mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
        // Use a finite large-negative value that is representable in fp16
        // (-1e9 overflows to -inf in fp16, which then breaks SDPA's mask
        // dtype promotion check). -1e4 is the standard whisper choice.
        mask = mask.asType(dtype) * MLXArray(-1e4, dtype: dtype)
        return mask
    }
}

// MARK: - Top-level Whisper

public final class Whisper: Module {
    public let config: WhisperConfig
    @ModuleInfo(key: "encoder") public var encoder: AudioEncoder
    @ModuleInfo(key: "decoder") public var decoder: TextDecoder

    public init(_ config: WhisperConfig) {
        self.config = config
        self._encoder.wrappedValue = AudioEncoder(
            nMels: config.nMels,
            nCtx: config.nAudioCtx,
            nState: config.nAudioState,
            nHead: config.nAudioHead,
            nLayer: config.nAudioLayer)
        self._decoder.wrappedValue = TextDecoder(
            nVocab: config.nVocab,
            nCtx: config.nTextCtx,
            nState: config.nTextState,
            nHead: config.nTextHead,
            nLayer: config.nTextLayer)
    }

    /// Rewrite mlx-whisper's Sequential MLP keys so they match our
    /// ResidualAttentionBlock field names. `mlp.0.*` → `mlp1.*`,
    /// `mlp.2.*` → `mlp2.*`. Also coerce safetensors int buffers
    /// (positional_embedding) into float32 if needed.
    public static func sanitizeWeights(_ weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var k = key
            // .mlp.0. / .mlp.2. live inside residual blocks; translate
            // to flattened `.mlp1.` / `.mlp2.` module keys.
            if k.contains(".mlp.0.") {
                k = k.replacingOccurrences(of: ".mlp.0.", with: ".mlp1.")
            } else if k.contains(".mlp.2.") {
                k = k.replacingOccurrences(of: ".mlp.2.", with: ".mlp2.")
            }
            out[k] = value
        }
        return out
    }
}
