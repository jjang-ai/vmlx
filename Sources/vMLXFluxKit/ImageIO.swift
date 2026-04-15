import Foundation
@preconcurrency import MLX
#if canImport(AppKit)
import AppKit
import CoreImage
#endif

// MARK: - ImageIO
//
// Write an MLX tensor to a PNG on disk. Used by every model as the
// final step of `generate()` / `edit()` / `upscale()`. Isolated here so
// the VAE decode path has a single function to call at the end of
// sampling.
//
// Expected input shape: (B, C=3, H, W) float in [0, 1]. Values outside
// that range are clamped.

public enum ImageIO {

    /// Save an image tensor to `dir/<prefix>-<uuid>.png`.
    /// Returns the URL of the written file.
    @MainActor
    public static func writePNG(
        _ tensor: MLXArray,
        outputDir: URL,
        prefix: String = "vmlx"
    ) throws -> URL {
        #if canImport(AppKit)
        guard tensor.ndim == 4 || tensor.ndim == 3 else {
            throw FluxError.invalidRequest(
                "image tensor must be (B,C,H,W) or (C,H,W), got ndim=\(tensor.ndim)")
        }
        // Squeeze batch dim if present.
        let single: MLXArray
        if tensor.ndim == 4 {
            single = tensor[0]
        } else {
            single = tensor
        }
        let channels = single.dim(0)
        let height = single.dim(1)
        let width = single.dim(2)

        guard channels == 3 || channels == 1 else {
            throw FluxError.invalidRequest(
                "image tensor must have 1 or 3 channels, got \(channels)")
        }

        // Clamp to [0, 1], scale to [0, 255], cast to uint8.
        let clamped = clip(single, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
        let scaled = clamped * MLXArray(Float(255))
        let rounded = MLX.round(scaled)
        let asUInt8 = rounded.asType(.uint8)

        // (C, H, W) → (H, W, C) for pixel buffer interpretation.
        let interleaved = asUInt8.transposed(1, 2, 0)
        let bytes = interleaved.asArray(UInt8.self)

        // Build an NSBitmapImageRep and serialize.
        let bitsPerPixel = channels == 3 ? 24 : 8
        let bytesPerRow = width * channels
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: channels,
            hasAlpha: false,
            isPlanar: false,
            colorSpaceName: channels == 3 ? .calibratedRGB : .calibratedWhite,
            bytesPerRow: bytesPerRow,
            bitsPerPixel: bitsPerPixel
        ) else {
            throw FluxError.invalidRequest("failed to create NSBitmapImageRep")
        }
        // Copy pixel data into the rep.
        if let ptr = rep.bitmapData {
            bytes.withUnsafeBufferPointer { src in
                if let base = src.baseAddress {
                    ptr.update(from: base, count: bytes.count)
                }
            }
        }
        guard let pngData = rep.representation(using: .png, properties: [:]) else {
            throw FluxError.invalidRequest("failed to PNG-encode image")
        }

        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)
        let filename = "\(prefix)-\(UUID().uuidString).png"
        let url = outputDir.appendingPathComponent(filename)
        try pngData.write(to: url)
        return url
        #else
        throw FluxError.notImplemented("ImageIO.writePNG requires AppKit")
        #endif
    }
}
