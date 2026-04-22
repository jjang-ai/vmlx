import Foundation
@_exported import vMLXFluxKit
@_exported import vMLXFluxModels
@_exported import vMLXFluxVideo

/// Top-level facade for vmlx-flux. One import, one actor, one API.
///
/// The engine loads a FluxModel on demand via `load(_:)`, then dispatches
/// generation / edit / upscale / video requests to the right backend. It
/// streams progress events as they arrive from the scheduler so the
/// calling app (vMLX) can render live step counters + partial previews.
///
/// Threading: actor-isolated. MLX ops are not thread-safe across the
/// same allocator, and the generation loop has a persistent latent buffer,
/// so every entry point goes through the actor's executor.
public actor FluxEngine {

    // MARK: - State

    /// Currently loaded model (nil if no model is resident).
    public private(set) var loaded: LoadedModel?

    /// Cached model registry lookup — `ModelRegistry.lookup(name:)`
    /// is O(1) but we keep a local copy for logging.
    public private(set) var lastLoadedName: String?

    // MARK: - Init

    public init() {}

    // MARK: - Load / unload

    /// Load a model by canonical name. Mirrors the Python mflux
    /// `ModelConfig.from_name(_:)` resolution with the `SUPPORTED_MODELS`
    /// dict from `vmlx_engine/image_gen.py`.
    ///
    /// Parameters:
    /// - `name`: canonical model key — `"flux1-schnell"`, `"flux2-klein"`,
    ///   `"z-image-turbo"`, `"qwen-image"`, `"qwen-image-edit"`,
    ///   `"flux1-kontext"`, `"flux1-fill"`, `"seedvr2"`, `"fibo"`.
    /// - `modelPath`: local directory containing weights. We NEVER
    ///   silently download from HuggingFace — the caller (vMLX
    ///   `DownloadManager`) must stage the weights first.
    /// - `quantize`: bit width (4, 8) or nil for full precision.
    public func load(
        name: String,
        modelPath: URL,
        quantize: Int? = nil
    ) async throws {
        guard let entry = ModelRegistry.lookup(name: name) else {
            throw FluxError.unknownModel(name)
        }
        let model = try await entry.loader(modelPath, quantize)
        self.loaded = LoadedModel(
            name: name,
            kind: entry.kind,
            model: model,
            isPlaceholder: entry.isPlaceholder
        )
        self.lastLoadedName = name
    }

    /// I1/I3 §311 — true when the currently-resident image model ships a
    /// placeholder transformer (e.g. noise velocity predictor). Callers
    /// stamp a warning on /v1/images/generations responses so users
    /// don't mistake noise bytes for a real generation failure.
    public func loadedIsPlaceholder() -> Bool {
        loaded?.isPlaceholder ?? false
    }

    /// Unload the current model and release weights.
    public func unload() {
        self.loaded = nil
    }

    // MARK: - Generate (text-to-image)

    /// Run a text-to-image generation. Returns an AsyncThrowingStream of
    /// progress events — the final `.completed(url:)` carries the
    /// written image path.
    public func generate(
        _ request: ImageGenRequest
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performGenerate(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performGenerate(
        _ request: ImageGenRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        guard let loaded else { throw FluxError.notLoaded }
        guard let generator = loaded.model as? ImageGenerator else {
            throw FluxError.wrongModelKind(
                expected: "ImageGenerator",
                actual: String(describing: type(of: loaded.model))
            )
        }
        for try await event in generator.generate(request) {
            continuation.yield(event)
        }
    }

    // MARK: - Edit (image-to-image)

    /// Run an image edit request (Kontext / Fill / Qwen-Image-Edit).
    public func edit(
        _ request: ImageEditRequest
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performEdit(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performEdit(
        _ request: ImageEditRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        guard let loaded else { throw FluxError.notLoaded }
        guard let editor = loaded.model as? ImageEditor else {
            throw FluxError.wrongModelKind(
                expected: "ImageEditor",
                actual: String(describing: type(of: loaded.model))
            )
        }
        for try await event in editor.edit(request) {
            continuation.yield(event)
        }
    }

    // MARK: - Upscale (SeedVR2)

    /// Run an upscale request. Emits the same progress events as gen/edit.
    public func upscale(
        _ request: UpscaleRequest
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.performUpscale(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performUpscale(
        _ request: UpscaleRequest,
        continuation: AsyncThrowingStream<ImageGenEvent, Error>.Continuation
    ) async throws {
        guard let loaded else { throw FluxError.notLoaded }
        guard let upscaler = loaded.model as? ImageUpscaler else {
            throw FluxError.wrongModelKind(
                expected: "ImageUpscaler",
                actual: String(describing: type(of: loaded.model))
            )
        }
        for try await event in upscaler.upscale(request) {
            continuation.yield(event)
        }
    }

    // MARK: - Video (future — Apple WAN 2.x)

    /// Video generation stub — scaffolded for Apple WAN models but not
    /// implemented yet. See `vMLXFluxVideo/WANModel.swift`.
    public func generateVideo(
        _ request: VideoGenRequest
    ) -> AsyncThrowingStream<VideoGenEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: FluxError.notImplemented(
                "Video generation — WAN models scaffolded but not implemented"))
        }
    }
}

/// A loaded model + its canonical name + kind (for fast dispatch).
public struct LoadedModel: Sendable {
    public let name: String
    public let kind: ModelKind
    public let model: any FluxModel
    /// I1/I3 §311 — mirrors ModelEntry.isPlaceholder so callers outside
    /// the Flux package can check without re-looking-up the registry.
    public let isPlaceholder: Bool

    public init(name: String, kind: ModelKind, model: any FluxModel,
                isPlaceholder: Bool = false) {
        self.name = name
        self.kind = kind
        self.model = model
        self.isPlaceholder = isPlaceholder
    }
}
