// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TurboQuant — arXiv:2504.19874 (Google DeepMind)

import Foundation
import MLX
import MLXRandom

/// TurboQuant encoder/decoder — per-coordinate scalar quantization with QJL correction.
///
/// ## Full Pipeline (Keys, b bits total)
///
/// ```
/// float16 keys [B, H, T, D]
///     │
///     ├─ Extract sink tokens (first 4) at full precision
///     │
///     ├─ Step 1: Normalize to unit sphere
///     │   k_unit = k / ||k||,  store ||k|| as vectorNorms
///     │
///     ├─ Step 2: Randomized Hadamard rotation
///     │   k_rot = H * diag(signs) * k_unit
///     │   (spreads energy uniformly → single codebook works for all dims)
///     │
///     ├─ Step 3: Lloyd-Max quantization with (b-1) bits
///     │   indices = quantize(k_rot, codebook_{b-1})
///     │   k_mse = dequantize(indices, codebook_{b-1})
///     │
///     ├─ Step 4: QJL residual correction (1 bit)
///     │   residual = k_rot - k_mse
///     │   qjl_signs = sign(residual @ S^T)    — 1 bit per dim
///     │   store ||residual|| as residualNorms
///     │
///     └─ Step 5: Pack everything
///         indices → packed uint32 (10 values/word for 2-bit)
///         qjl_signs → packed uint32 (32 signs/word)
///         norms → float16
/// ```
///
/// ## Full Pipeline (Values, b bits total)
///
/// Same as keys but WITHOUT Step 4 (QJL). All b bits go to codebook indices.
/// Values don't need QJL because they're linearly combined in attention
/// (no exponential amplification through softmax).
///
/// ## Decode Pipeline
///
/// ```
/// packed data
///     │
///     ├─ Unpack indices and signs
///     ├─ Codebook lookup: k_mse = codebook[indices]
///     ├─ QJL correction: k_corr = k_mse + sqrt(π/2)/d * ||r|| * (signs @ S)
///     ├─ Inverse Hadamard: k_unit = diag(signs) * H * k_corr
///     ├─ Scale by norms: k = k_unit * ||k||
///     └─ Prepend sink tokens
///     → float16 keys [B, H, T, D]  (fed directly to attention)
/// ```
public struct TQEncoder: Sendable {

    /// Default number of sink tokens preserved at full precision.
    /// Sink tokens (BOS, system prompt start) receive disproportionate attention
    /// from all subsequent tokens. Compressing them degrades quality measurably.
    public static let defaultSinkTokens = 4

    // MARK: - Precomputed State

    /// Precomputed encoder state for a given (dim, keyBits, valueBits, seed).
    ///
    /// Create once per model configuration, reuse across all encode/decode calls.
    /// Contains:
    /// - Random sign vector for Hadamard rotation (deterministic from seed)
    /// - Lloyd-Max codebooks for keys and values (cached by dim+bits)
    /// - QJL projection matrix S (deterministic from seed+1000)
    ///
    /// All fields are deterministic from (dim, keyBits, valueBits, seed),
    /// so the state never needs to be serialized — just recreate from params.
    public struct EncoderState: @unchecked Sendable {
        public let dim: Int
        public let keyBits: Int
        public let valueBits: Int
        public let seed: Int

        public let rotationSigns: MLXArray
        public let keyCodebook: [Float]
        public let keyIndexBits: Int
        public let valueCodebook: [Float]
        public let valueIndexBits: Int
        public let qjlS: MLXArray

        public init(dim: Int, keyBits: Int = 3, valueBits: Int = 3, seed: Int = 42) {
            self.dim = dim
            self.keyBits = keyBits
            self.valueBits = valueBits
            self.seed = seed

            self.rotationSigns = TQHadamard.generateRandomSigns(dim: dim, seed: seed)

            // Keys: (b-1) bits for MSE codebook, 1 bit for QJL
            let kMseBits = max(keyBits - 1, 1)
            self.keyCodebook = TQCodebook.computeCodebook(dim: dim, bits: kMseBits)
            self.keyIndexBits = kMseBits

            // Values: all b bits for MSE codebook (no QJL)
            self.valueCodebook = TQCodebook.computeCodebook(dim: dim, bits: valueBits)
            self.valueIndexBits = valueBits

            // QJL projection matrix (offset seed to avoid correlation with rotation signs)
            self.qjlS = TQQJL.generateProjection(dim: dim, seed: seed + 1000)
        }
    }

    // MARK: - Encode Keys

    /// Compress float keys to TurboQuant format.
    ///
    /// - Parameters:
    ///   - keys: Float16/32 key tensor, shape [batch, heads, tokens, head_dim]
    ///   - state: Precomputed encoder state
    ///   - sinkTokens: Leading tokens to preserve at full precision (default 4)
    /// - Returns: EncodedKeys with packed indices, QJL signs, and norms
    public static func encodeKeys(
        _ keys: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let origShape = keys.shape
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        // Extract sink tokens at full precision
        let sinkData: MLXArray?
        let compressKeys: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = keys[.ellipsis, 0..<sinkTokens, 0...]
            compressKeys = keys[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressKeys = keys
        }

        let compressShape = compressKeys.shape

        // Step 1: Normalize to unit sphere, store vector norms
        let vectorNorms = (compressKeys * compressKeys).sum(axis: -1, keepDims: true).sqrt()
        let keysUnit = compressKeys / (vectorNorms + 1e-8)

        // Step 2: Randomized Hadamard rotation
        let keysRotated = TQHadamard.hadamardRotate(keysUnit, signs: state.rotationSigns)

        // Step 3: Lloyd-Max quantization with (b-1) bits
        let flatRotated = keysRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.keyCodebook)
        let mseDequant = TQCodebook.dequantizeScalar(mseIndices, codebook: state.keyCodebook)

        // Step 4: QJL 1-bit correction on residual
        let residual = flatRotated - mseDequant
        let projected = matmul(residual, state.qjlS.transposed())
        let qjlSigns = which(projected .>= 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))
        let residualNorms = (residual * residual).sum(axis: -1, keepDims: true).sqrt()

        // Step 5: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped([-1]), bits: state.keyIndexBits)
        let packedQJL = TQBitPack.packSigns(qjlSigns.reshaped([-1]))

        return EncodedKeys(
            indicesPacked: packedIndices,
            qjlPacked: packedQJL,
            residualNorms: residualNorms
                .reshaped(Array(compressShape.dropLast()) + [1]).asType(.float16),
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.keyIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Keys

    /// Decompress keys from TurboQuant format back to float16.
    ///
    /// Reverses the encode pipeline:
    /// unpack → codebook lookup → QJL correction → inverse Hadamard → scale by norms
    public static func decodeKeys(_ encoded: EncodedKeys, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Unpack indices and QJL signs
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatQJL = TQBitPack.unpackSigns(
            encoded.qjlPacked, nElements: nElements
        ).reshaped([-1, dim])
        let flatResNorms = encoded.residualNorms.asType(.float32).reshaped([-1, 1])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Codebook lookup
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.keyCodebook)

        // QJL correction: r_hat = sqrt(pi/2) / d * ||r|| * (signs @ S)
        let qjlScale = Float(Foundation.sqrt(Double.pi / 2.0)) / Float(dim)
        let qjlDequant = MLXArray(qjlScale) * flatResNorms * matmul(flatQJL, state.qjlS)

        // Combine MSE + QJL, inverse Hadamard rotation
        let reconstructedRotated = (mseDequant + qjlDequant).reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Scale by original vector norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Encode Values

    /// Compress float values to TurboQuant format (MSE only, no QJL).
    ///
    /// Same pipeline as keys but simpler — all b bits go to the codebook,
    /// no residual correction needed.
    public static func encodeValues(
        _ values: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let origShape = values.shape
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        let sinkData: MLXArray?
        let compressValues: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = values[.ellipsis, 0..<sinkTokens, 0...]
            compressValues = values[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressValues = values
        }

        let compressShape = compressValues.shape

        // Step 1: Normalize
        let vectorNorms = (compressValues * compressValues).sum(axis: -1, keepDims: true).sqrt()
        let valuesUnit = compressValues / (vectorNorms + 1e-8)

        // Step 2: Hadamard rotate
        let valuesRotated = TQHadamard.hadamardRotate(valuesUnit, signs: state.rotationSigns)

        // Step 3: Lloyd-Max quantization (full b bits)
        let flatRotated = valuesRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.valueCodebook)

        // Step 4: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped([-1]), bits: state.valueIndexBits)

        return EncodedValues(
            indicesPacked: packedIndices,
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.valueIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Values

    /// Decompress values from TurboQuant format back to float16.
    public static func decodeValues(_ encoded: EncodedValues, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Unpack
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Codebook lookup
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.valueCodebook)

        // Inverse Hadamard
        let reconstructedRotated = mseDequant.reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Scale by norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }
}
