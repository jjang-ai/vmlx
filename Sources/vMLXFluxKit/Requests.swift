import Foundation

// MARK: - ImageGenRequest

public struct ImageGenRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var numImages: Int
    public var outputDir: URL
    public var outputFormat: ImageFormat

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 20,
        guidance: Float = 3.5,
        seed: UInt64? = nil,
        numImages: Int = 1,
        outputDir: URL,
        outputFormat: ImageFormat = .png
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.numImages = numImages
        self.outputDir = outputDir
        self.outputFormat = outputFormat
    }
}

// MARK: - ImageEditRequest

public struct ImageEditRequest: Sendable {
    public var prompt: String
    public var sourceImage: URL
    public var mask: URL?          // optional PNG mask — white=edit, black=keep
    public var strength: Float     // 0..1 — how much to deviate from source
    public var width: Int?         // nil → match source
    public var height: Int?
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var outputDir: URL
    public var outputFormat: ImageFormat

    public init(
        prompt: String,
        sourceImage: URL,
        mask: URL? = nil,
        strength: Float = 0.75,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int = 20,
        guidance: Float = 3.5,
        seed: UInt64? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png
    ) {
        self.prompt = prompt
        self.sourceImage = sourceImage
        self.mask = mask
        self.strength = strength
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputDir = outputDir
        self.outputFormat = outputFormat
    }
}

// MARK: - UpscaleRequest (SeedVR2)

public struct UpscaleRequest: Sendable {
    public var sourceImage: URL
    public var scale: Int          // 2, 4
    public var steps: Int
    public var seed: UInt64?
    public var outputDir: URL
    public var outputFormat: ImageFormat

    public init(
        sourceImage: URL,
        scale: Int = 4,
        steps: Int = 10,
        seed: UInt64? = nil,
        outputDir: URL,
        outputFormat: ImageFormat = .png
    ) {
        self.sourceImage = sourceImage
        self.scale = scale
        self.steps = steps
        self.seed = seed
        self.outputDir = outputDir
        self.outputFormat = outputFormat
    }
}

// MARK: - VideoGenRequest (future)

public struct VideoGenRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var outputDir: URL

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 1280,
        height: Int = 720,
        numFrames: Int = 121,   // WAN 2.1 default: 121 frames @ 24fps ≈ 5s
        fps: Int = 24,
        steps: Int = 50,
        guidance: Float = 5.0,
        seed: UInt64? = nil,
        outputDir: URL
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputDir = outputDir
    }
}

// MARK: - Events

/// Streamed progress event for gen / edit / upscale.
public enum ImageGenEvent: Sendable {
    /// Incremental progress. `step` is 1-indexed, `total` is the full step count.
    case step(step: Int, total: Int, etaSeconds: Double?)
    /// Optional in-progress preview (partially denoised latent decoded to a
    /// small PNG). Not every model emits previews — this fires at most
    /// every N steps if the scheduler supports it.
    case preview(pngData: Data, step: Int)
    /// Generation finished. `url` points to the saved image on disk.
    case completed(url: URL, seed: UInt64)
    /// Fatal error. `hfAuth=true` signals a HuggingFace 401/403 so the UI
    /// can show a "Add HF token" CTA instead of a generic error banner.
    case failed(message: String, hfAuth: Bool)
    /// User or system cancelled the job.
    case cancelled
}

/// Streamed progress event for video gen (future).
public enum VideoGenEvent: Sendable {
    case step(step: Int, total: Int, etaSeconds: Double?)
    case preview(pngData: Data, frame: Int)
    case completed(url: URL, seed: UInt64, fps: Int, frameCount: Int)
    case failed(message: String, hfAuth: Bool)
    case cancelled
}

// MARK: - Formats

public enum ImageFormat: String, Sendable, Codable {
    case png
    case jpeg
    case webp
}
