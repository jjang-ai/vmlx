//
//  JangDFlashConfig.swift
//  vMLXLMCommon / DFlash
//

import Foundation

/// Configuration for the JANG-DFlash block-diffusion drafter.
///
/// Mirrors `jang_tools.dflash.config.JangDFlashConfig` (Python). Field names
/// and defaults MUST stay in sync so a single checkpoint round-trips between
/// the PyTorch trainer and the MLX-Swift inference runtime.
public struct JangDFlashConfig: Codable, Sendable, Equatable {
    public var vocabSize: Int = 200064
    public var hiddenDim: Int = 1536
    public var numLayers: Int = 5
    public var numHeads: Int = 12
    public var numKVHeads: Int = 4
    public var ffnDim: Int = 4096
    public var blockSize: Int = 16
    /// ID of the MASK token appended to the vocabulary. The drafter's
    /// embedding table has `vocabSize + 1` rows; the extra row at
    /// `vocabSize` is the MASK token.
    public var maskTokenId: Int = 200064
    /// Dimension of the raw tap concatenation fed to the fusion MLP —
    /// `numTapLayers × targetHiddenDim`. For MiniMax-JANG_2L with 5 tap
    /// layers and hidden 3072 this is `5 * 3072 = 15360`.
    public var tapDim: Int = 15360
    public var headDim: Int = 128
    public var ropeTheta: Float = 10000.0
    public var rmsNormEps: Float = 1e-6

    public init() {}

    public init(
        vocabSize: Int = 200064,
        hiddenDim: Int = 1536,
        numLayers: Int = 5,
        numHeads: Int = 12,
        numKVHeads: Int = 4,
        ffnDim: Int = 4096,
        blockSize: Int = 16,
        maskTokenId: Int = 200064,
        tapDim: Int = 15360,
        headDim: Int = 128,
        ropeTheta: Float = 10000.0,
        rmsNormEps: Float = 1e-6
    ) {
        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.ffnDim = ffnDim
        self.blockSize = blockSize
        self.maskTokenId = maskTokenId
        self.tapDim = tapDim
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
    }
}
