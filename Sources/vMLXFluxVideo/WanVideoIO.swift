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
