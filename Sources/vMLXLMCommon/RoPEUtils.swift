//
//  RoPEUtils.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/8/11.
//

import Foundation
import MLX
import MLXNN

/// Iter 143 — emit a stderr warning when a RoPE config is malformed,
/// in lieu of crashing. Once LogStore is reachable from vMLXLMCommon
/// without a circular dep, replace these with structured log lines.
@inline(__always)
private func ropeWarn(_ message: String) {
    if let data = "[vMLX][rope] WARN: \(message)\n".data(using: .utf8) {
        try? FileHandle.standardError.write(contentsOf: data)
    }
}

public class Llama3RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        // Iter 143 — graceful fallback. Was: `fatalError("Llama3RoPE
        // requires scaling_config")` which crashed engine load on any
        // Llama-3 community quant whose `rope_scaling` block was
        // truncated or absent. Now we log + use the standard
        // (factor=1.0) Llama 3 defaults so the model loads with
        // standard-context behavior; long-context inference may be
        // suboptimal but the engine stays up.
        let scalingConfig: [String: StringOrNumber] = scalingConfig ?? [:]
        if scalingConfig.isEmpty {
            ropeWarn(
                "Llama3RoPE: scaling_config absent — using factor=1.0/8192 defaults. "
                + "Long-context inference may underperform until rope_scaling is restored.")
        }

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192.0

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

}

public class YarnRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let traditional: Bool

    private let _mscale: Float
    private let _freqs: MLXArray

    /// Public accessor for the wavelength tensor (post-YaRN correction).
    /// Used by DSV4 attention to compute INVERSE rope by passing -freqs to
    /// MLXFast.RoPE. Without inverse rope, DSV4 output is gibberish (verified
    /// in jang-tools Python tests 2026-04-24, see research/DSV4-RUNTIME-ARCHITECTURE.md §29).
    public var freqs: MLXArray { _freqs }
    public var mscale: Float { _mscale }

    public init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            return Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            if scale <= 1 {
                return 1.0
            }
            return 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal {
                maxVal += 0.001
            }

            let linearFunc = (MLXArray(0 ..< dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        self._mscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
            / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / dimensions)
        let freqInter =
            scalingFactor
            * pow(
                base,
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                    / dimensions)

        let (low, high) = yarnFindCorrectionRange()
        let freqMask =
            1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // "copy" of x as we are going to write through it and don't want to update
        // through the reference
        // https://github.com/ml-explore/mlx-swift/issues/364
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

}

private let yarnTypes: Set = ["yarn", "deepseek_yarn", "telechat3-yarn"]

public typealias RoPELayer = OffsetLayer & ArrayOffsetLayer

public func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> RoPELayer {
    let ropeType: String = {
        if let config = scalingConfig,
            let typeValue = config["type"] ?? config["rope_type"],
            case .string(let s) = typeValue
        {
            return s
        }
        return "default"
    }()

    if ropeType == "default" || ropeType == "linear" {
        let scale: Float
        if ropeType == "linear", let factor = scalingConfig?["factor"]?.asFloat() {
            scale = 1 / factor
        } else {
            scale = 1.0
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)
    } else if ropeType == "llama3" {
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )
    } else if yarnTypes.contains(ropeType) {
        let factor = scalingConfig?["factor"]?.asFloat() ?? 32.0
        let origMax = scalingConfig?["original_max_position_embeddings"]?.asInt() ?? 4096
        let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32.0
        let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1.0
        let mscale = scalingConfig?["mscale"]?.asFloat() ?? 1.0
        let mscaleAllDim = scalingConfig?["mscale_all_dim"]?.asFloat() ?? 0.0

        return YarnRoPE(
            dimensions: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: factor,
            originalMaxPositionEmbeddings: origMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )
    } else if ropeType == "longrope" {
        // Iter 143 — graceful fallback. Was 4 fatalErrors that crashed
        // engine load on any Phi-3-class community quant whose
        // `rope_scaling` block was truncated. Now: missing fields
        // produce a warn + fallback to plain RoPE (standard context
        // length); long-context inference will be suboptimal but the
        // engine stays up.
        guard let config = scalingConfig,
              let origMax = config["original_max_position_embeddings"]?.asInt(),
              let shortFactor = config["short_factor"]?.asFloats(),
              let longFactor = config["long_factor"]?.asFloats()
        else {
            let missing: [String] = {
                guard let config = scalingConfig else {
                    return ["scaling_config (entire block)"]
                }
                var m: [String] = []
                if config["original_max_position_embeddings"]?.asInt() == nil {
                    m.append("original_max_position_embeddings")
                }
                if config["short_factor"]?.asFloats() == nil { m.append("short_factor") }
                if config["long_factor"]?.asFloats() == nil { m.append("long_factor") }
                return m
            }()
            ropeWarn(
                "longrope: missing fields \(missing.joined(separator: ",")) — "
                + "falling back to standard RoPE; long-context inference may be wrong "
                + "for this bundle until rope_scaling is restored.")
            return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
        }

        return SuScaledRoPE(
            dimensions: dims,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131072,
            originalMaxPositionEmbeddings: origMax,
            shortFactor: shortFactor,
            longFactor: longFactor
        )
    } else if ropeType == "mrope" {
        // MRoPE returns basic RoPE here. The actual multi-modal rotary embedding logic
        // (applying different embeddings per modality) is handled in the attention layer
        // of multimodal models like Qwen2VL, not in the RoPE module itself.
        if let config = scalingConfig,
            let mropeSection = config["mrope_section"]?.asInts()
        {
            precondition(
                mropeSection.count == 3,
                "MRoPE currently only supports 3 sections, got \(mropeSection.count)"
            )
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
    } else {
        // Iter 143 — graceful fallback for unsupported rope_type. Was
        // a fatalError that crashed engine load on any model with a
        // future/exotic rope_type the Swift port doesn't yet
        // recognize (Phi-4 variants, Qwen exp configs). Plain RoPE
        // works for prompts up to the model's natural context.
        ropeWarn(
            "Unsupported RoPE type '\(ropeType)' — falling back to plain RoPE. "
            + "Add a handler in initializeRope to support this type properly.")
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
    }
}
