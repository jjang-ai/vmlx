import Foundation

// MARK: - Core protocols
//
// Every concrete model (Flux1, Flux2Klein, ZImage, Qwen, SeedVR2, …)
// conforms to `FluxModel` plus one-or-more capability protocols:
//
//   ImageGenerator — text → image
//   ImageEditor    — (image, prompt[, mask]) → image
//   ImageUpscaler  — low-res image → high-res image
//   VideoGenerator — text → video (future)
//
// Lives in vMLXFluxKit so every model implementation depends on it
// through a single target edge.

/// Marker protocol for anything that can be loaded by `FluxEngine`.
public protocol FluxModel: Sendable {}

/// Text-to-image generator.
public protocol ImageGenerator: FluxModel {
    func generate(_ request: ImageGenRequest) -> AsyncThrowingStream<ImageGenEvent, Error>
}

/// Image-to-image editor (inpaint, outpaint, controlnet-like).
public protocol ImageEditor: FluxModel {
    func edit(_ request: ImageEditRequest) -> AsyncThrowingStream<ImageGenEvent, Error>
}

/// Super-resolution / upscale (SeedVR2).
public protocol ImageUpscaler: FluxModel {
    func upscale(_ request: UpscaleRequest) -> AsyncThrowingStream<ImageGenEvent, Error>
}

/// Text-to-video (Apple WAN 2.1/2.2, future).
public protocol VideoGenerator: FluxModel {
    func generate(_ request: VideoGenRequest) -> AsyncThrowingStream<VideoGenEvent, Error>
}

// MARK: - Kind (routed by FluxEngine)

public enum ModelKind: String, Sendable, Codable {
    case imageGen
    case imageEdit
    case imageUpscale
    case videoGen
}

// MARK: - Errors

public enum FluxError: Error, CustomStringConvertible {
    case unknownModel(String)
    case notLoaded
    case wrongModelKind(expected: String, actual: String)
    case weightsNotFound(URL)
    case notImplemented(String)
    case invalidRequest(String)

    public var description: String {
        switch self {
        case .unknownModel(let n): return "unknown model: \(n)"
        case .notLoaded:           return "no model loaded — call FluxEngine.load first"
        case .wrongModelKind(let e, let a):
            return "wrong model kind: expected \(e), got \(a)"
        case .weightsNotFound(let u): return "weights not found at \(u.path)"
        case .notImplemented(let s): return "not implemented: \(s)"
        case .invalidRequest(let s): return "invalid request: \(s)"
        }
    }
}
