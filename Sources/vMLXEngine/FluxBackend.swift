import Foundation
import vMLXFlux

// MARK: - vMLX ↔ vmlx-flux bridge
//
// Wires the vMLX `Engine` actor's image generation surface to the
// vmlx-flux `FluxEngine` actor. The app's `ImageScreen` already calls
// `Engine.generateImage(prompt:model:settings:)` and subscribes to
// `imageGenStream(jobId:)` — this file replaces the `notImplemented`
// stubs with real `FluxEngine` dispatch.
//
// Lifecycle: a single `FluxEngine` is created lazily on the first image
// call and held for the Engine actor's lifetime. Switching image models
// is handled by `FluxEngine.load(name:modelPath:quantize:)` — one
// resident model at a time, matching the current SwiftUI UX.
//
// Stream semantics: `FluxEngine` returns
// `AsyncThrowingStream<ImageGenEvent, Error>`. We bridge each
// `ImageGenEvent` to the Engine's own `ImageGenEvent` (same shape, in
// the vMLXEngine namespace) and store the active job's stream so
// `imageGenStream(jobId:)` can tee off of it.

extension Engine {

    /// Lazy-loaded flux backend. Created on first call.
    internal func getOrCreateFluxBackend() -> FluxEngine {
        if let existing = fluxBackend as? FluxEngine { return existing }
        let e = FluxEngine()
        fluxBackend = e
        // Make sure the model registry is populated on first use.
        vMLXFluxModels.registerAll()
        vMLXFluxVideo.registerAll()
        return e
    }

    // MARK: - Typed image generation (called from ImageScreen)

    /// Run a text-to-image generation end-to-end and return the final URL.
    /// The UI subscribes to `imageGenStream(jobId:)` for progress — we
    /// kick off the actual generation here and feed the bridged events
    /// into the job registry before returning.
    public func generateImage(
        prompt: String,
        model: String,
        settings: ImageGenSettings
    ) async throws -> URL {
        let flux = getOrCreateFluxBackend()

        // Ensure the requested model is loaded. If a different model is
        // currently resident, swap. `lastLoadedName` is an actor accessor.
        let lastName = await flux.lastLoadedName
        if lastName != model {
            guard let libraryEntry = await self.modelLibrary.entries().first(where: {
                $0.displayName == model || $0.id == model
            }) else {
                throw EngineError.notImplemented(
                    "FluxBackend — no model entry for '\(model)'. "
                    + "Stage it via DownloadManager first."
                )
            }
            try await flux.load(
                name: model.lowercased(),
                modelPath: libraryEntry.canonicalPath,
                quantize: nil
            )
        }

        // vMLXEngine.ImageGenSettings has a flat Int seed (-1 = random) and
        // no outputDir field; resolve the output dir from a scratch path
        // under NSTemporaryDirectory for now and flow the UUID-named file
        // back to the caller via the .completed event.
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vmlx-flux-out", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        let seed: UInt64? = settings.seed >= 0
            ? UInt64(bitPattern: Int64(settings.seed))
            : nil

        let request = vMLXFlux.ImageGenRequest(
            prompt: prompt,
            width: settings.width,
            height: settings.height,
            steps: settings.steps,
            guidance: Float(settings.guidance),
            seed: seed,
            numImages: settings.numImages,
            outputDir: outputDir
        )

        // Drain the stream until completion, feeding bridged events into
        // the per-job event channel so the UI gets live updates.
        let jobId = UUID()
        let bridge = FluxJobBridge(jobId: jobId)
        registerFluxJob(bridge)

        var finalURL: URL? = nil
        var seedReturned: UInt64 = 0
        do {
            for try await evt in await flux.generate(request) {
                bridge.yield(bridgeEvent(evt))
                if case .completed(let url, let seed) = evt {
                    finalURL = url
                    seedReturned = seed
                }
                if case .failed(let msg, _) = evt {
                    throw EngineError.notImplemented("FluxBackend: \(msg)")
                }
            }
        } catch {
            bridge.finish(throwing: error)
            unregisterFluxJob(jobId)
            throw error
        }
        bridge.finish()
        unregisterFluxJob(jobId)
        _ = seedReturned
        guard let url = finalURL else {
            throw EngineError.notImplemented("FluxBackend — no output URL")
        }
        return url
    }

    /// Run an image edit end-to-end. Mirrors `generateImage` but dispatches
    /// through `FluxEngine.edit` which routes to whichever concrete editor
    /// the loaded model exposes (`Flux1Fill`, `Flux2KleinEdit`, `QwenImageEdit`).
    ///
    /// Writes `source` and optional `mask` to PNGs under a scratch directory,
    /// builds an `ImageEditRequest`, drains the bridge stream, and returns
    /// the final output URL. Progress events are fed into `fluxJobs` under
    /// the returned job id — callers can subscribe via `imageGenStream`.
    public func editImage(
        prompt: String,
        model: String,
        source: Data,
        mask: Data?,
        strength: Double,
        settings: ImageGenSettings
    ) async throws -> URL {
        let flux = getOrCreateFluxBackend()

        // Ensure the requested model is loaded — same pattern as generateImage.
        let lastName = await flux.lastLoadedName
        if lastName != model {
            guard let libraryEntry = await self.modelLibrary.entries().first(where: {
                $0.displayName == model || $0.id == model
            }) else {
                throw EngineError.notImplemented(
                    "FluxBackend.editImage — no model entry for '\(model)'. "
                    + "Stage it via DownloadManager first."
                )
            }
            try await flux.load(
                name: model.lowercased(),
                modelPath: libraryEntry.canonicalPath,
                quantize: nil
            )
        }

        // Scratch dir for source/mask/output, matching generateImage's layout.
        let outputDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vmlx-flux-out", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        // Dump inputs to disk — vmlx-flux editors take URL handles so they
        // can memory-map / reopen on the model actor. Use random filenames
        // so concurrent edits don't race on the same path.
        let stem = UUID().uuidString.prefix(8)
        let sourceURL = outputDir.appendingPathComponent("src-\(stem).png")
        try source.write(to: sourceURL, options: .atomic)

        var maskURL: URL? = nil
        if let mask {
            let u = outputDir.appendingPathComponent("mask-\(stem).png")
            try mask.write(to: u, options: .atomic)
            maskURL = u
        }

        let seed: UInt64? = settings.seed >= 0
            ? UInt64(bitPattern: Int64(settings.seed))
            : nil

        let request = vMLXFlux.ImageEditRequest(
            prompt: prompt,
            sourceImage: sourceURL,
            mask: maskURL,
            strength: Float(strength),
            width: settings.width > 0 ? settings.width : nil,
            height: settings.height > 0 ? settings.height : nil,
            steps: settings.steps,
            guidance: Float(settings.guidance),
            seed: seed,
            outputDir: outputDir
        )

        // Drain the edit stream, feeding events into the per-job bridge
        // so the UI gets live updates. Symmetric to generateImage.
        let jobId = UUID()
        let bridge = FluxJobBridge(jobId: jobId)
        registerFluxJob(bridge)

        var finalURL: URL? = nil
        do {
            for try await evt in await flux.edit(request) {
                bridge.yield(bridgeEvent(evt))
                if case .completed(let url, _) = evt {
                    finalURL = url
                }
                if case .failed(let msg, _) = evt {
                    throw EngineError.notImplemented("FluxBackend.editImage: \(msg)")
                }
            }
        } catch {
            bridge.finish(throwing: error)
            unregisterFluxJob(jobId)
            try? FileManager.default.removeItem(at: sourceURL)
            if let maskURL { try? FileManager.default.removeItem(at: maskURL) }
            throw error
        }
        bridge.finish()
        unregisterFluxJob(jobId)

        // Best-effort cleanup of the temporary inputs. The output PNG
        // lives under the same `outputDir` and is the caller's return value.
        try? FileManager.default.removeItem(at: sourceURL)
        if let maskURL { try? FileManager.default.removeItem(at: maskURL) }

        guard let url = finalURL else {
            throw EngineError.notImplemented("FluxBackend.editImage — no output URL")
        }
        return url
    }

    /// Subscribe to progress events for a specific job.
    public nonisolated func imageGenStream(
        jobId: UUID
    ) -> AsyncThrowingStream<ImageGenEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                guard let bridge = await self.fluxJobs[jobId] else {
                    continuation.finish()
                    return
                }
                for await e in bridge.subscribe() {
                    continuation.yield(e)
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Job registry helpers

    private func registerFluxJob(_ bridge: FluxJobBridge) {
        fluxJobs[bridge.jobId] = bridge
    }

    private func unregisterFluxJob(_ id: UUID) {
        fluxJobs[id] = nil
    }

    /// Convert a vmlx-flux ImageGenEvent to the vMLXEngine ImageGenEvent
    /// the app layer already knows about. Note: the two enums have
    /// overlapping names but different shapes — vMLXEngine's `.step` bundles
    /// the preview into the same case while vmlx-flux has a separate
    /// `.preview` case. We merge them here.
    private nonisolated func bridgeEvent(_ e: vMLXFlux.ImageGenEvent) -> ImageGenEvent {
        switch e {
        case .step(let step, let total, _):
            return .step(step: step, total: total, preview: nil)
        case .preview(let data, let step):
            // Surface as a step event with preview bytes; total unknown here
            // so use step for both. The UI re-derives progress from its own
            // counter between bridge events.
            return .step(step: step, total: step, preview: data)
        case .completed(let url, _):
            return .completed(url: url)
        case .failed(let msg, let hfAuth):
            return .failed(message: msg, hfAuth: hfAuth)
        case .cancelled:
            return .cancelled
        }
    }
}

// MARK: - FluxJobBridge
//
// Per-job fan-out. A single `FluxEngine.generate()` stream feeds here,
// and any number of UI subscribers tee off via `subscribe()`. When the
// job finishes, all subscriptions finish too.

final class FluxJobBridge: @unchecked Sendable {
    let jobId: UUID
    private let lock = NSLock()
    private var continuations: [UUID: AsyncStream<ImageGenEvent>.Continuation] = [:]
    private var finished = false
    private var terminalError: Error?

    init(jobId: UUID) {
        self.jobId = jobId
    }

    func subscribe() -> AsyncStream<ImageGenEvent> {
        AsyncStream { cont in
            lock.lock()
            let id = UUID()
            if finished {
                cont.finish()
                lock.unlock()
                return
            }
            continuations[id] = cont
            lock.unlock()
            cont.onTermination = { [weak self] _ in
                self?.lock.lock()
                self?.continuations[id] = nil
                self?.lock.unlock()
            }
        }
    }

    func yield(_ event: ImageGenEvent) {
        lock.lock()
        let conts = continuations
        lock.unlock()
        for (_, c) in conts { c.yield(event) }
    }

    func finish(throwing error: Error? = nil) {
        lock.lock()
        finished = true
        terminalError = error
        let conts = continuations
        continuations.removeAll()
        lock.unlock()
        for (_, c) in conts { c.finish() }
    }
}
