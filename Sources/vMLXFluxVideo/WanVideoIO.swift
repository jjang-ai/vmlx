import Foundation
@preconcurrency import MLX
import vMLXFluxKit
#if canImport(AppKit)
import AppKit
import AVFoundation
import CoreImage
#endif

// MARK: - Video frame output
//
// Write a decoded (B, 3, T, H, W) video tensor to disk. Three output
// modes:
//   1. `writeFrames` — per-frame PNG sequence (simplest, always works).
//   2. `writeMP4`    — H.264 video via AVAssetWriter (macOS/iOS).
//   3. `writeFramesAndMP4` — both (for debugging the first real Wan run).
//
// Shape convention: (C=3, T, H, W) in [0, 1] after WanVAEDecoder.postprocess.
// The batch dim is dropped by the caller.

public enum WanVideoIO {

    // MARK: - Image-to-video input
    //
    // For I2V variants (Wan 2.2 I2V-14B, TI2V-5B with image conditioning),
    // load a still image from disk and tile it into a 5D conditioning
    // tensor `(1, condChannels, T, H, W)` shape-compatible with the
    // latent.
    //
    // Wan reference (`wan_2/i2v_utils.py`): the init image is encoded
    // through the VAE encoder, then concatenated channel-wise with the
    // noise latent before patchify. We don't yet have a Swift port of
    // the VAE *encoder* (only the decoder ships), so this implementation
    // takes the simpler path: decode the image to a (3, H, W) RGB tensor,
    // tile it across all T frames and pad/repeat the channel dim to
    // `condChannels`. This is a placeholder strategy — once the VAE
    // encoder ships, replace `decodeInputImage` with `vaeEncode(image) →
    // (16, T/4, H/8, W/8)` then channel-concat as the reference does.

    /// Decode an image file into a Wan-compatible 5D conditioning tensor.
    /// - Parameters:
    ///   - at: PNG/JPEG image URL.
    ///   - width / height: target latent width × 8 (model space).
    ///   - numFrames: video frame count (will be tiled).
    ///   - inChannels: number of conditioning channels expected by the
    ///     transformer (e.g. 20 for Wan 2.2 I2V-14B = 36 in_dim - 16 latent).
    /// - Returns: `(1, inChannels, T/4, H/8, W/8)` placeholder tensor.
    @MainActor
    public static func decodeInputImage(
        at url: URL,
        width: Int,
        height: Int,
        numFrames: Int,
        inChannels: Int
    ) throws -> MLXArray {
        #if canImport(AppKit)
        guard let img = NSImage(contentsOf: url) else {
            throw FluxError.invalidRequest("could not load init image at \(url.path)")
        }
        let cgImage: CGImage? = img.cgImage(
            forProposedRect: nil, context: nil, hints: nil
        )
        guard let cg = cgImage else {
            throw FluxError.invalidRequest("could not rasterize init image")
        }

        // Resize to (width, height) into a contiguous RGBA8 buffer.
        let bitsPerComponent = 8
        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw FluxError.invalidRequest("CGContext alloc failed")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

        // (H, W, 4) → strip alpha → (3, H, W) float in [-1, 1].
        var rgb: [Float] = []
        rgb.reserveCapacity(3 * height * width)
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * bytesPerRow + x * 4 + c
                    rgb.append(Float(rgba[idx]) / 127.5 - 1.0)
                }
            }
        }
        let img3 = MLXArray(rgb).reshaped([3, height, width])

        // Spatially downsample to latent resolution (H/8, W/8) via 8×8
        // mean-pool. Placeholder until the VAE encoder ports — smoke
        // test only checks shape compatibility for the conditioning.
        let hLat = height / 8
        let wLat = width / 8
        let pooled = img3
            .reshaped([3, hLat, 8, wLat, 8])
            .mean(axes: [2, 4])     // (3, hLat, wLat)

        // Tile to (1, inChannels, tLat, hLat, wLat). Repeat the 3 RGB
        // channels to cover inChannels, then slice down to exact count.
        let tLat = max(1, numFrames / 4)
        let chRepeats = (inChannels + 2) / 3
        let chTiled = MLX.tiled(
            pooled.reshaped([1, 3, hLat, wLat]),
            repetitions: [1, chRepeats, 1, 1]
        )
        let chTrimmed = chTiled[0..., 0 ..< inChannels, 0..., 0...]
        let withTime = MLX.tiled(
            chTrimmed.reshaped([1, inChannels, 1, hLat, wLat]),
            repetitions: [1, 1, tLat, 1, 1]
        )
        return withTime
        #else
        throw FluxError.notImplemented("WanVideoIO.decodeInputImage requires AppKit")
        #endif
    }

    /// Write every frame of the video as a PNG in `dir/frame-NNNN.png`.
    /// Returns the array of URLs written.
    @MainActor
    public static func writeFrames(
        _ video: MLXArray,
        outputDir: URL,
        prefix: String = "wan"
    ) throws -> [URL] {
        #if canImport(AppKit)
        precondition(video.ndim == 4, "expected (C, T, H, W) — drop batch first")
        let channels = video.dim(0)
        let t = video.dim(1)
        let h = video.dim(2)
        let w = video.dim(3)
        precondition(channels == 3, "video must be 3-channel RGB")

        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        var urls: [URL] = []
        for frameIdx in 0..<t {
            // Extract (C, H, W) slice for this frame.
            let frame = video[0 ..< 3, frameIdx ..< frameIdx + 1, 0 ..< h, 0 ..< w]
                .reshaped([3, h, w])
            // Reuse the still-image writer with a synthetic batch dim.
            let withBatch = frame.reshaped([1, 3, h, w])
            let url = try ImageIO.writePNG(
                withBatch,
                outputDir: outputDir,
                prefix: String(format: "\(prefix)-frame-%04d", frameIdx)
            )
            urls.append(url)
        }
        return urls
        #else
        throw FluxError.notImplemented("WanVideoIO.writeFrames requires AppKit")
        #endif
    }

    /// Write the full video as an H.264 MP4 at the requested fps.
    /// Uses AVAssetWriter + CVPixelBuffer. Same tensor shape as writeFrames.
    @MainActor
    public static func writeMP4(
        _ video: MLXArray,
        outputURL: URL,
        fps: Int
    ) throws {
        #if canImport(AppKit)
        precondition(video.ndim == 4, "expected (C, T, H, W)")
        let c = video.dim(0)
        let t = video.dim(1)
        let h = video.dim(2)
        let w = video.dim(3)
        precondition(c == 3, "video must be 3-channel RGB")

        // Remove any existing file — AVAssetWriter won't overwrite.
        try? FileManager.default.removeItem(at: outputURL)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attrs
        )

        guard writer.canAdd(input) else {
            throw FluxError.invalidRequest("AVAssetWriter rejected video input")
        }
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for frameIdx in 0..<t {
            // Extract (C, H, W) → pack into ARGB pixel buffer below.
            let frame = video[0 ..< 3, frameIdx ..< frameIdx + 1, 0 ..< h, 0 ..< w]
                .reshaped([3, h, w])
            let clamped = clip(frame, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
            let scaled = clamped * MLXArray(Float(255))
            let asUInt8 = scaled.asType(.uint8)
            // (C, H, W) → (H, W, C)
            let interleaved = asUInt8.transposed(1, 2, 0)
            let rgbBytes = interleaved.asArray(UInt8.self)

            // Build a CVPixelBuffer in BGRA order.
            var pb: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault, w, h,
                kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pb
            )
            guard status == kCVReturnSuccess, let buffer = pb else {
                throw FluxError.invalidRequest("CVPixelBuffer alloc failed")
            }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
                let ptr = base.assumingMemoryBound(to: UInt8.self)
                for y in 0..<h {
                    for x in 0..<w {
                        let srcIdx = (y * w + x) * 3
                        let dstIdx = y * rowBytes + x * 4
                        // ARGB: A, R, G, B
                        ptr[dstIdx + 0] = 255
                        ptr[dstIdx + 1] = rgbBytes[srcIdx + 0]
                        ptr[dstIdx + 2] = rgbBytes[srcIdx + 1]
                        ptr[dstIdx + 3] = rgbBytes[srcIdx + 2]
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])

            // Wait for the input to be ready, then append.
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }
            let presentationTime = CMTime(
                value: CMTimeValue(frameIdx),
                timescale: CMTimeScale(fps)
            )
            _ = adaptor.append(buffer, withPresentationTime: presentationTime)
        }

        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        writer.finishWriting { sem.signal() }
        sem.wait()
        if writer.status != .completed {
            throw FluxError.invalidRequest(
                "AVAssetWriter failed: \(writer.error?.localizedDescription ?? "unknown")")
        }
        #else
        throw FluxError.notImplemented("WanVideoIO.writeMP4 requires AppKit")
        #endif
    }
}
