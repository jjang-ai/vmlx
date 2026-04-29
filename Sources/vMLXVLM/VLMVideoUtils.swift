// VLMVideoUtils.swift
// Shared video preprocessing primitives reused across Qwen 2/2.5/3/3.5/3.6 VL,
// Kimi VL, NemotronHOmni, and any future video-capable VLM in vmlx-swift-lm.
//
// Built on top of `MediaProcessing.asCIImageSequence` (the existing async
// frame extraction) to keep the decode path uniform across all models.
//
// What's shared:
//   • Frame extraction → `MediaProcessing.asCIImageSequence`
//   • Bicubic resize + RGB Float32 + custom mean/std normalize
//   • T-frame channel stacking (used by NemotronHOmni; compatible with any
//     model that stacks consecutive frames into the channel dim before a
//     temporal patch embedder, e.g. video_embedder layouts in InternVL,
//     Nemotron-Omni, Qwen 3.6 VL)
//   • Cosine-similarity-based token retention (EVS — applicable to any
//     VLM that wants to drop redundant inter-frame tokens)
//
// What's NOT shared (per-model specific):
//   • The actual ViT body (RADIO vs Qwen vs Kimi)
//   • The tile-tagging text format (Nemotron uses NVLM 1-D, Qwen uses
//     `<|video_pad|>`-style markers)
//   • Frame-rate / sampling policies (model-specific defaults)

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import AVFoundation
import MLX

private let sharedVideoCIContext = CIContext(options: [
    .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any
])

// MARK: - CLIP / SigLIP mean/std presets (used by Nemotron / Qwen 3.6 / Kimi)

/// OpenAI CLIP normalization (Nemotron RADIO, Qwen-VL old)
public let CLIP_NORM_MEAN: [Float] = [0.48145466, 0.4578275, 0.40821073]
public let CLIP_NORM_STD: [Float] = [0.26862954, 0.26130258, 0.27577711]

/// SigLIP normalization (used by some Qwen 2.5+ VL variants)
public let SIGLIP_NORM_MEAN: [Float] = [0.5, 0.5, 0.5]
public let SIGLIP_NORM_STD: [Float] = [0.5, 0.5, 0.5]

// MARK: - Generic uniform frame extractor

/// Extract roughly `targetFrames` frames uniformly from a video.
///
/// Wraps `MediaProcessing.asCIImageSequence` with a sample-rate computed
/// from total duration. Models that prefer fps-based sampling can call
/// `MediaProcessing.asCIImageSequence(samplesPerSecond:)` directly.
@available(macOS 14.0, *)
public func vlmExtractFramesUniform(
    url: URL,
    targetFrames: Int = 32
) async throws -> [CIImage] {
    let asset = AVURLAsset(url: url)
    let duration = try await asset.load(.duration)
    let durationSeconds = duration.seconds
    guard durationSeconds.isFinite, durationSeconds > 0 else {
        throw NSError(
            domain: "VLMVideoUtils", code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Invalid video duration"])
    }
    let samplesPerSecond = max(1, Int(round(Double(targetFrames) / durationSeconds)))
    return try await MediaProcessing.asCIImageSequence(
        asset, samplesPerSecond: samplesPerSecond,
    )
}

// MARK: - Resize + normalize

/// Bicubic-resize a CIImage to (target, target) and normalize via the given
/// mean/std. Returns a contiguous (3*target*target,) Float32 buffer in
/// (3, H, W) row-major order — matches PyTorch tensor layout.
@available(macOS 14.0, *)
public func vlmResizeAndNormalize(
    _ image: CIImage,
    target: Int,
    mean: [Float] = CLIP_NORM_MEAN,
    std: [Float] = CLIP_NORM_STD
) -> [Float] {
    let extent = image.extent
    let scaleX = CGFloat(target) / extent.width
    let scaleY = CGFloat(target) / extent.height
    let resized = image
        .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        .cropped(to: CGRect(x: 0, y: 0, width: target, height: target))

    // Render to RGBA8, strip alpha, normalize, transpose HWC → CHW.
    let bytesPerRow = 4 * target
    var buffer = [UInt8](repeating: 0, count: 4 * target * target)
    let cgContext = CGContext(
        data: &buffer, width: target, height: target,
        bitsPerComponent: 8, bytesPerRow: bytesPerRow,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue,
    )!
    if let cg = sharedVideoCIContext.createCGImage(
        resized,
        from: CGRect(x: 0, y: 0, width: target, height: target),
    ) {
        cgContext.draw(cg, in: CGRect(x: 0, y: 0, width: target, height: target))
    }

    var out = [Float](repeating: 0, count: 3 * target * target)
    let plane = target * target
    for y in 0 ..< target {
        for x in 0 ..< target {
            let i = (y * target + x) * 4
            let r = Float(buffer[i]) / 255.0
            let g = Float(buffer[i + 1]) / 255.0
            let b = Float(buffer[i + 2]) / 255.0
            let pix = y * target + x
            out[0 * plane + pix] = (r - mean[0]) / std[0]
            out[1 * plane + pix] = (g - mean[1]) / std[1]
            out[2 * plane + pix] = (b - mean[2]) / std[2]
        }
    }
    return out
}

// MARK: - T-frame channel stacking

/// Build a (nGroups, T*3, H, W) MLXArray from a list of CIImages, where
/// each group of T consecutive frames is stacked into the channel dim.
///
/// Used by Nemotron-3-Nano-Omni's RADIO video_embedder (T=2). Other VLMs
/// can use this with their own T parameter (e.g. T=1 = no stacking, just
/// per-frame embedding).
///
/// Frames are right-padded with the last frame to a multiple of T.
@available(macOS 14.0, *)
public func vlmStackFramesIntoChannels(
    _ frames: [CIImage],
    imageSize: Int = 512,
    temporalPatchDim T: Int = 2,
    mean: [Float] = CLIP_NORM_MEAN,
    std: [Float] = CLIP_NORM_STD
) -> MLXArray {
    var fs = frames
    while fs.count % T != 0 {
        fs.append(fs.last!)
    }
    let n = fs.count
    let groups = n / T
    let H = imageSize, W = imageSize

    var stacked = [Float](repeating: 0, count: n * 3 * H * W)
    let perFrame = 3 * H * W
    for (i, f) in fs.enumerated() {
        let resized = vlmResizeAndNormalize(f, target: imageSize, mean: mean, std: std)
        stacked.replaceSubrange(i * perFrame ..< (i + 1) * perFrame, with: resized)
    }
    let pixelValues = MLXArray(stacked, [n, 3, H, W])
    return pixelValues.reshaped([groups, T * 3, H, W])
}

// MARK: - EVS (Efficient Video Sampling)

/// Drop a fraction of redundant tokens between consecutive temporal groups,
/// based on cosine similarity at matching spatial positions. Mirrors
/// `compute_evs_retention_mask` from the Python `nemotron_omni.video_processor`.
///
/// Generic enough to apply to ANY model whose video tokens have the layout
/// (n_groups, tokens_per_group, hidden) — e.g. NemotronHOmni, but also
/// Qwen 3.6 VL and Kimi VL after their respective patch-merge stages.
///
/// - Parameters:
///   - feats: (nGroups, tokensPerGroup, hidden) MLXArray
///   - pruningRate: fraction of tokens to drop. Source default is 0.7
///     (drop 70%).
///   - keepFirstFrame: always keep all tokens of the first temporal group
///     (matches the source's `dissimilarity = [255, …]` first-row trick).
///     Defaults to true.
///
/// Returns `(1, kept_count, hidden)` MLXArray with the pruned tokens.
@available(macOS 14.0, *)
public func vlmApplyEVS(
    _ feats: MLXArray,
    pruningRate: Float = 0.7,
    keepFirstFrame: Bool = true
) -> MLXArray {
    let nGroups = feats.dim(0)
    let tokensPerGroup = feats.dim(1)
    let hidden = feats.dim(2)

    if nGroups < 2 {
        return feats
    }

    // Cosine similarity between consecutive groups, per token position
    let g0 = feats[0 ..< (nGroups - 1)]
    let g1 = feats[1 ..< nGroups]
    let dot = (g0 * g1).sum(axis: -1)
    let n0 = MLX.sqrt((g0 * g0).sum(axis: -1) + 1e-8)
    let n1 = MLX.sqrt((g1 * g1).sum(axis: -1) + 1e-8)
    let cos = dot / (n0 * n1)

    let totalTokens = nGroups * tokensPerGroup
    let dropTarget = Int(Float(totalTokens) * pruningRate)

    let cosFlat = cos.reshaped([(nGroups - 1) * tokensPerGroup])
    let cosArray = cosFlat.asArray(Float.self)
    let sortedIdx = (0 ..< cosArray.count).sorted { cosArray[$0] > cosArray[$1] }

    var keep = [Bool](repeating: true, count: totalTokens)
    var dropped = 0
    for relIdx in sortedIdx {
        if dropped >= dropTarget { break }
        let group = 1 + relIdx / tokensPerGroup
        let tokIn = relIdx % tokensPerGroup
        let absIdx = group * tokensPerGroup + tokIn
        if keepFirstFrame && group == 0 { continue }
        if keep[absIdx] {
            keep[absIdx] = false
            dropped += 1
        }
    }

    let keptIdx = (0 ..< totalTokens).filter { keep[$0] }
    if keptIdx.isEmpty {
        return feats[0 ..< 1]
    }
    let flat = feats.reshaped([totalTokens, hidden])
    let idxArr = MLXArray(keptIdx.map { Int32($0) })
    let gathered = flat.take(idxArr, axis: 0)
    return gathered.reshaped([1, keptIdx.count, hidden])
}
