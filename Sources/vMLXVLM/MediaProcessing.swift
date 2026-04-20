// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreImage.CIFilterBuiltins
import MLX
import vMLXLMCommon

public typealias VideoFrame = UserInput.VideoFrame

public struct ProcessedFrames {
    public let frames: [MLXArray]
    public let timestamps: [CMTime]
    public let totalDuration: CMTime
}

private let context = CIContext()

/// Collection of methods for processing media (images, video, etc.).
///
/// A typical image preparation pipeline might look like this:
///
/// ```swift
/// var image: CIImage
/// image = MediaProcessing.inSRGBToneCurveSpace(image)
///
/// // Apply user instructions
/// image = MediaProcessing.apply(image, processing: processing)
///
/// image = MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
/// image = MediaProcessing.normalize(
///     image, mean: config.imageMeanTuple, std: config.imageStdTuple)
///
/// return MediaProcessing.asMLXArray(image)
/// ```
///
/// This is the responsibility of the `UserInputProcessor`.
public enum MediaProcessing {

    /// VLM media processing is normally done without regard to the colorspace. Many,
    /// though not all, images are stored in sRGB and this will be the implicit colorspace
    /// used. This converts to a colorspace with an sRGB tone curve, though not necessarily
    /// sRGB primaries, etc.
    ///
    /// See ``inLinearToneCurveSpace(_:)``
    public static func inSRGBToneCurveSpace(_ image: CIImage) -> CIImage {
        let filter = CIFilter.linearToSRGBToneCurve()
        filter.inputImage = image
        return filter.outputImage!
    }

    /// Inverse of ``inSRGBToneCurveSpace(_:)`` (for completeness).
    public static func inLinearToneCurveSpace(_ image: CIImage) -> CIImage {
        let filter = CIFilter.sRGBToneCurveToLinear()
        filter.inputImage = image
        return filter.outputImage!
    }

    /// Compute the best fit size of one size in another (respecting aspect ratio).
    public static func bestFit(_ size: CGSize, in other: CGSize) -> CGSize {
        let scale = bestFitScale(size, in: other)
        return CGSize(width: round(size.width * scale), height: round(size.height * scale))
    }

    /// Compute the best fit scale of one size in another (respecting aspect ratio).
    public static func bestFitScale(_ size: CGSize, in other: CGSize) -> CGFloat {
        min(other.width / size.width, other.height / size.height)
    }

    static public func aspectRatioForResample(_ image: CIImage, size: CGSize) -> Float {
        let inputAspectRatio = image.extent.width / image.extent.height
        let desiredAspectRatio = size.width / size.height
        return Float(1 / inputAspectRatio * desiredAspectRatio)
    }

    /// Resample the image using Lanczos interpolation.
    static public func resampleLanczos(_ image: CIImage, to size: CGSize) -> CIImage {
        // Create a bicubic scale filter

        let yScale = size.height / image.extent.height
        let xScale = size.width / image.extent.width

        let filter = CIFilter.lanczosScaleTransform()
        filter.inputImage = image
        filter.scale = Float(yScale)
        filter.aspectRatio = Float(xScale / yScale)
        let scaledImage = filter.outputImage!

        // Create a rect with the exact dimensions we want
        let exactRect = CGRect(
            x: 0,
            y: 0,
            width: size.width,
            height: size.height
        )

        // Crop to ensure exact dimensions
        return scaledImage.cropped(to: exactRect)
    }

    /// Resample the image using bicubic interpolation.
    /// - Parameters:
    ///   - image: The image to resample
    ///   - size: The target size
    /// - Returns: The resampled image
    public static func resampleBicubic(_ image: CIImage, to size: CGSize) -> CIImage {
        // Create a bicubic scale filter

        let yScale = size.height / image.extent.height
        let xScale = size.width / image.extent.width

        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image
        filter.scale = Float(yScale)
        filter.aspectRatio = Float(xScale / yScale)
        let scaledImage = filter.outputImage!

        // Create a rect with the exact dimensions we want
        let exactRect = CGRect(
            x: 0,
            y: 0,
            width: size.width,
            height: size.height
        )

        // Crop to ensure exact dimensions
        return scaledImage.cropped(to: exactRect)
    }

    /// Normalize the image using the given mean and standard deviation parameters.
    public static func normalize(
        _ image: CIImage, mean: (CGFloat, CGFloat, CGFloat), std: (CGFloat, CGFloat, CGFloat)
    ) -> CIImage {
        let filter = CIFilter.colorMatrix()
        filter.inputImage = image

        // This should match
        // https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
        //
        // output[channel] = (input[channel] - mean[channel]) / std[channel]
        //
        // The CI filter computes input * factor + bias so we want to do:
        // input * 1 / std - mean / std

        filter.rVector = .init(x: 1 / std.0, y: 0, z: 0, w: 0)
        filter.gVector = .init(x: 0, y: 1 / std.1, z: 0, w: 0)
        filter.bVector = .init(x: 0, y: 0, z: 1 / std.2, w: 0)

        filter.aVector = .init(x: 0, y: 0, z: 0, w: 1)
        filter.biasVector = .init(x: -mean.0 / std.0, y: -mean.1 / std.1, z: -mean.2 / std.2, w: 0)

        return filter.outputImage!
    }

    /// Convert the CIImage into a planar 3 channel MLXArray `[1, C, H, W]`
    /// - Parameters:
    ///   - image: The image to convert
    ///   - colorSpace: Optional color space for rendering
    /// - Returns: The MLXArray representation of the image
    public static func asMLXArray(_ image: CIImage, colorSpace: CGColorSpace? = nil) -> MLXArray {
        let size = image.extent.size
        let w = Int(size.width.rounded())
        let h = Int(size.height.rounded())

        // probably not strictly necessary, but this is what happens in
        // e.g. image_processing_siglip in transformers (float32)
        let format = CIFormat.RGBAf
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * 4
        let bytesPerRow = w * bytesPerPixel

        var data = Data(count: w * h * bytesPerPixel)
        data.withUnsafeMutableBytes { ptr in
            context.render(
                image, toBitmap: ptr.baseAddress!, rowBytes: bytesPerRow, bounds: image.extent,
                format: format, colorSpace: colorSpace)
            context.clearCaches()
        }

        var array = MLXArray(data, [h, w, 4], type: Float32.self)

        // Drop 4th channel
        array = array[0..., 0..., ..<3]

        // Convert to 1, C, H, W
        array = array.reshaped(1, h, w, 3).transposed(0, 3, 1, 2)

        return array
    }

    /// Return `true` if the size is smaller or equal to the size of the `extent`.
    public static func rectSmallerOrEqual(_ extent: CGRect, size: CGSize) -> Bool {
        return extent.width <= size.width && extent.height <= size.height
    }

    /// Given an `extent` and a target `size` produce the `CGRect` that will be a center crop.
    public static func centerCrop(_ extent: CGRect, size: CGSize) -> CGRect {
        let targetWidth = min(extent.width, size.width)
        let targetHeight = min(extent.height, size.height)

        return CGRect(
            x: (extent.maxX - targetWidth) / 2,
            y: (extent.maxY - targetHeight) / 2,
            width: targetWidth, height: targetHeight
        )
    }

    /// Given an `image` and a target `size` produce the `CIImage` that will be a center crop.
    public static func centerCrop(_ image: CIImage, size: CGSize) -> CIImage {
        let extent = image.extent
        if rectSmallerOrEqual(extent, size: size) {
            return image
        }

        let crop = centerCrop(extent, size: size)
        return
            image
            .cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
    }

    /// Given a `size` and a target `shortestEdge` compute a new size
    /// that respects the aspect ratio of the original `size` and is
    /// constrained by the `shortestEdge`.
    public static func fitIn(_ size: CGSize, shortestEdge: Int) -> CGSize {
        let floatShortestEdge = CGFloat(shortestEdge)

        let (short, long) =
            size.width <= size.height ? (size.width, size.height) : (size.height, size.width)
        let newShort = floatShortestEdge
        let newLong = floatShortestEdge * long / short

        return size.width <= size.height
            ? CGSize(width: newShort, height: newLong) : CGSize(width: newLong, height: newShort)
    }

    /// Given a `size` and a target `longestEdge` compute a new size
    /// that respects the aspect ratio of the original `size` and is
    /// constrained by the `longestEdge`.
    public static func fitIn(_ size: CGSize, longestEdge: Int) -> CGSize {
        let floatLongestEdge = CGFloat(longestEdge)

        var (newShort, newLong) =
            size.width <= size.height ? (size.width, size.height) : (size.height, size.width)

        if newLong > floatLongestEdge {
            newLong = floatLongestEdge
            newShort = floatLongestEdge * newShort / newLong
        }

        return size.width <= size.height
            ? CGSize(width: newShort, height: newLong) : CGSize(width: newLong, height: newShort)
    }

    /// Enlarge an image by padding to square size.
    /// The original image is centered inside the square.
    public static func padToSquare(_ image: CIImage, backgroundColor: CIColor = .black) -> CIImage {
        let rect = image.extent.integral
        let (w, h) = (rect.width, rect.height)
        let side = max(w, h)

        let background = CIImage(color: backgroundColor).cropped(
            to: CGRect(x: 0, y: 0, width: side, height: side))

        let tx = (side - w) * 0.5 - rect.origin.x
        let ty = (side - h) * 0.5 - rect.origin.y
        let centered = image.transformed(by: CGAffineTransform(translationX: tx, y: ty))

        return centered.composited(over: background)
    }

    /// Apply `UserInput.Processing`, if needed, to the image.
    public static func apply(_ image: CIImage, processing: UserInput.Processing?) -> CIImage {
        var image = image

        if let resize = processing?.resize {
            let scale = bestFitScale(image.extent.size, in: resize)
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        return image
    }

    public static func asCIImageSequence(_ asset: AVAsset, samplesPerSecond: Int) async throws
        -> [CIImage]
    {
        // Use AVAssetImageGenerator to extract frames
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        // Calculate the time values we want to sample
        guard let duration = try? await asset.load(.duration) else {
            throw NSError(
                domain: "MediaProcessing", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load the asset's duration"])
        }

        let durationInSeconds = duration.seconds
        let samplesPerSecond = Double(samplesPerSecond)
        let totalFramesToSample = durationInSeconds * samplesPerSecond
        let durationTimeValue = duration.value
        let sampledTimeValues = MLXArray.linspace(
            0, durationTimeValue, count: Int(totalFramesToSample)
        ).asArray(Int64.self)

        // Construct a CMTime using the sampled CMTimeValue's and the asset's timescale
        let timescale = duration.timescale
        let sampledTimes = sampledTimeValues.map { CMTime(value: $0, timescale: timescale) }

        // Collect the frames
        var ciImages: [CIImage] = []
        for await result in generator.images(for: sampledTimes) {
            switch result {
            case .success(requestedTime: _, let image, actualTime: _):
                let ciImage = CIImage(
                    cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                ciImages.append(ciImage)
            case .failure(requestedTime: _, _):
                break
            }
        }

        return ciImages
    }

    private static func validateAsset(_ asset: AVAsset) async throws {
        let tracks = try await asset.loadTracks(withMediaType: .video)

        guard !tracks.isEmpty,
            let videoTrack = tracks.first
        else { throw VLMError.noVideoTrackFound }

        let isDecodable = try await videoTrack.load(.isDecodable)

        if !isDecodable {
            throw VLMError.videoNotDecodable
        }
    }

    static public func asProcessedSequence(
        _ video: UserInput.Video,
        samplesPerSecond: Int,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 },
    ) async throws -> ProcessedFrames {
        return try await asProcessedSequence(
            video,
            targetFPS: { _ in Double(samplesPerSecond) },
            maxFrames: Int.max,
            frameProcessing: frameProcessing
        )
    }

    static public func asProcessedSequence(
        _ video: UserInput.Video,
        targetFPS: (CMTime) -> Double,
        maxFrames: Int = Int.max,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 }
    ) async throws -> ProcessedFrames {

        switch video
        {
        case .avAsset(let asset):
            try await Self.validateAsset(asset)
            return try await _asProcessedSequence(
                asset, maxFrames: maxFrames, targetFPS: targetFPS, frameProcessing: frameProcessing)

        case .url(let url):
            let asset = AVAsset(url: url)
            try await Self.validateAsset(asset)
            return try await _asProcessedSequence(
                asset, maxFrames: maxFrames, targetFPS: targetFPS, frameProcessing: frameProcessing)

        case .frames(let videoFrames):
            return try await _asProcessedSequence(
                videoFrames, targetFPS: targetFPS, frameProcessing: frameProcessing)
        }
    }

    @available(
        *, deprecated, message: "Use MediaProcessing.asProcessedSequence() with the Video directly"
    )
    static public func asProcessedSequence(
        _ asset: AVAsset, maxFrames: Int, targetFPS: (CMTime) -> Double,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 }
    ) async throws -> ProcessedFrames {
        return try await Self._asProcessedSequence(
            asset, maxFrames: maxFrames, targetFPS: targetFPS, frameProcessing: frameProcessing)
    }

    @available(
        *, deprecated, message: "Use MediaProcessing.asProcessedSequence() with the Video directly"
    )
    static public func asProcessedSequence(
        _ asset: AVAsset, samplesPerSecond: Int,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 }
    ) async throws -> ProcessedFrames {
        return try await _asProcessedSequence(
            asset, maxFrames: Int.max, targetFPS: { _ in Double(samplesPerSecond) },
            frameProcessing: frameProcessing)
    }

    static private func _asProcessedSequence(
        _ asset: AVAsset, maxFrames: Int, targetFPS: (CMTime) -> Double,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 }
    ) async throws -> ProcessedFrames {
        // Use AVAssetImageGenerator to extract frames
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        guard let duration = try? await asset.load(.duration) else {
            throw NSError(
                domain: "MediaProcessing", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load the asset's duration"])
        }
        let fps = targetFPS(duration)
        // Note: the round was not present in `asCIImageSequence`, so we may now be passing 1 more frame to Qwen depending on video duration.
        let estimatedFrames = Int(round(fps * duration.seconds))
        let desiredFrames = min(estimatedFrames, maxFrames)
        let finalFrameCount = max(desiredFrames, 1)

        let sampledTimeValues = MLXArray.linspace(
            0, duration.value, count: Int(finalFrameCount)
        ).asArray(Int64.self)

        // Construct a CMTime using the sampled CMTimeValue's and the asset's timescale
        let timescale = duration.timescale
        let sampledTimes = sampledTimeValues.map { CMTime(value: $0, timescale: timescale) }

        // 2026-04-18 VL+video memory fix — community users report
        // swap-storming when processing videos. The old path retained
        // every sampled CIImage in `ciImages` AND every post-decoded
        // fp32 MLXArray in `framesAsArrays` simultaneously, so peak
        // resident set was ≈ 2× `frameCount × frameBytes`. For a 60 s
        // 1080p clip at 2 FPS that's ~3.6 GB per video request; a 4K
        // clip or a multi-video message balloons further. Converting
        // each CIImage to an MLXArray inside the loop and dropping the
        // CIImage ref immediately halves peak memory. Identical output
        // shape/values — the final `ProcessedFrames.frames` list is
        // constructed in the same order as before.
        var frameArrays: [MLXArray] = []
        var timestamps: [CMTime] = []

        for await result in generator.images(for: sampledTimes) {
            switch result {
            case .success(requestedTime: _, let image, actualTime: let actual):
                let ciImage = CIImage(
                    cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                let frame = try frameProcessing(.init(frame: ciImage, timeStamp: actual))
                // Eager-convert to MLXArray so the CIImage + its CGImage
                // backing can drop refcount at the end of this iteration.
                frameArrays.append(frame.frame.asMLXArray())
                timestamps.append(frame.timeStamp)
            case .failure(requestedTime: _, _):
                break
            }
        }

        return ProcessedFrames(
            frames: frameArrays,
            timestamps: timestamps,
            totalDuration: duration
        )
    }

    static private func _asProcessedSequence(
        _ videoFrames: [VideoFrame],
        targetFPS: (CMTime) -> Double,
        frameProcessing: (VideoFrame) throws -> VideoFrame = { $0 }
    ) async throws -> ProcessedFrames {

        precondition(videoFrames.isEmpty == false)

        let startTime = videoFrames.first?.timeStamp ?? .zero
        let endTime = videoFrames.last?.timeStamp ?? .zero
        let timeRangeOfVideoFrames = CMTimeRange(start: startTime, end: endTime)

        let duration = timeRangeOfVideoFrames.duration

        let fps = targetFPS(duration)
        // Note: the round was not present in `asCIImageSequence`, so we may now be passing 1 more frame to Qwen depending on video duration.
        let estimatedFrames = Int(round(fps * duration.seconds))
        let desiredFrames = min(estimatedFrames, videoFrames.count)
        let finalFrameCount = max(desiredFrames, 1)

        let sampledTimeValues = MLXArray.linspace(
            0, duration.value, count: Int(finalFrameCount)
        ).asArray(Int64.self)

        // Construct a CMTime using the sampled CMTimeValue's and the asset's timescale
        let timescale = duration.timescale

        // Same memory fix as the AVAsset overload above: eager-convert
        // each sampled frame to an MLXArray and drop the CIImage ref in
        // the same iteration. See the longer comment there for context.
        var frameArrays: [MLXArray] = []
        var timestamps: [CMTime] = []

        // See https://github.com/ml-explore/mlx-swift-lm/pull/64#discussion_r2713532157
        // for rationalle for the follwing timing code

        var frameIndex = videoFrames.startIndex
        for value in sampledTimeValues {
            let targetTime = CMTime(value: value, timescale: timescale)

            // find the last frame <= the targetTime
            var targetIndex: Int?
            while frameIndex < videoFrames.endIndex {
                if videoFrames[frameIndex].timeStamp > targetTime {
                    break
                } else {
                    targetIndex = frameIndex
                    frameIndex += 1
                }
            }

            if let targetIndex {
                let videoFrame = videoFrames[targetIndex]
                let frame = try frameProcessing(
                    .init(frame: videoFrame.frame, timeStamp: videoFrame.timeStamp))
                frameArrays.append(frame.frame.asMLXArray())
                timestamps.append(frame.timeStamp)
            }
        }

        return ProcessedFrames(
            frames: frameArrays,
            timestamps: timestamps,
            totalDuration: duration
        )
    }
}

// MARK: - Convenience

extension CIImage {
    public enum ResamplingMethod {
        case bicubic
        case lanczos
    }

    public func resampled(to size: CGSize, method: ResamplingMethod = .bicubic) -> CIImage {
        switch method {
        case .bicubic:
            return MediaProcessing.resampleBicubic(self, to: size)
        case .lanczos:
            return MediaProcessing.resampleLanczos(self, to: size)
        }
    }

    public func toSRGB() -> CIImage {
        return MediaProcessing.inSRGBToneCurveSpace(self)
    }

    public func toLinear() -> CIImage {
        return MediaProcessing.inLinearToneCurveSpace(self)
    }

    public func normalized(mean: (CGFloat, CGFloat, CGFloat), std: (CGFloat, CGFloat, CGFloat))
        -> CIImage
    {
        return MediaProcessing.normalize(self, mean: mean, std: std)
    }

    public func paddingToSquare(backgroundColor color: CIColor = .black) -> CIImage {
        MediaProcessing.padToSquare(self, backgroundColor: color)
    }

    public func asMLXArray(colorSpace: CGColorSpace? = nil) -> MLXArray {
        return MediaProcessing.asMLXArray(self, colorSpace: colorSpace)
    }
}
