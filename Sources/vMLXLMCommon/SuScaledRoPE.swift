import Foundation
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py

/// Su Scaled Rotary Position Embedding.
public class SuScaledRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let originalMaxPositionEmbeddings: Int
    let _freqs: MLXArray
    let _scale: Float

    public init(
        dimensions: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131072,
        originalMaxPositionEmbeddings: Int = 4096,
        shortFactor: [Float] = [1.0],
        longFactor: [Float] = [1.0],
        shortMScale: Float? = nil,
        longMScale: Float? = nil
    ) {
        // Note: per python source shortFactor and shortMScale are unused.
        // https://github.com/ml-explore/mlx-lm/pull/707

        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        let freqs = pow(base, (arange(0, dimensions, step: 2, dtype: .float32) / dimensions))
        self._freqs = MLXArray(longFactor) * freqs

        func defaultScale(_ factor: Float) -> Float {
            sqrt(1 + log(factor) / log(Float(originalMaxPositionEmbeddings)))
        }

        let factor = Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        self._scale = longMScale ?? (factor < 1 ? 1 : defaultScale(factor))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // "copy" of x as we are going to write through it and don't want to update
        // through the reference
        // https://github.com/ml-explore/mlx-swift/issues/364
        let x = x[0..., .ellipsis]
        x[.ellipsis, 0 ..< dimensions] *= _scale
        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let x = x[0..., .ellipsis]
        x[.ellipsis, 0 ..< dimensions] *= _scale
        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }
}

/// Backward compatibility alias.
@available(*, deprecated, renamed: "SuScaledRoPE")
public typealias SuScaledRotaryEmbedding = SuScaledRoPE
