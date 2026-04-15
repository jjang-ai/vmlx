// SPDX-License-Identifier: Apache-2.0
//
// Typed image-generation request + event types consumed by the Swift Image
// screen and eventually emitted by the mflux adapter. Kept separate from
// `Engine.swift` so the view layer can `import vMLXEngine` and reference
// these without pulling in the larger Engine actor surface at source-read
// time.
//
// Python parity references:
//   vmlx_engine/server.py:3429 create_image
//   vmlx_engine/server.py:3632 create_image_edit
//   vmlx_engine/image/*            (mflux routing)
//   panel/src/renderer/src/components/image/ImageSettings.tsx

import Foundation

/// Per-request image-gen settings. Mirrors the Electron ImageSettings
/// component field-for-field. Every field has a sensible default so the UI
/// can construct one with `ImageGenSettings()` and override only what the
/// user explicitly changed.
public struct ImageGenSettings: Sendable, Codable, Equatable {
    public var steps: Int
    public var guidance: Double
    public var width: Int
    public var height: Int
    public var seed: Int             // -1 = random
    public var numImages: Int
    public var scheduler: String
    public var strength: Double      // edit-mode img2img strength (0..1)

    public init(
        steps: Int = 4,
        guidance: Double = 3.5,
        width: Int = 1024,
        height: Int = 1024,
        seed: Int = -1,
        numImages: Int = 1,
        scheduler: String = "default",
        strength: Double = 0.75
    ) {
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height
        self.seed = seed
        self.numImages = numImages
        self.scheduler = scheduler
        self.strength = strength
    }

    /// Construct from the image-related fields of a `GlobalSettings` snapshot.
    /// Used by the Image screen to seed defaults from the persisted store.
    public static func fromGlobal(_ g: GlobalSettings) -> ImageGenSettings {
        ImageGenSettings(
            steps: g.imageDefaultSteps,
            guidance: g.imageDefaultGuidance,
            width: g.imageDefaultWidth,
            height: g.imageDefaultHeight,
            seed: g.imageDefaultSeed,
            numImages: g.imageDefaultNumImages,
            scheduler: g.imageDefaultScheduler,
            strength: g.imageDefaultStrength
        )
    }
}

/// Live events emitted by `Engine.imageGenStream(jobId:)`. The UI subscribes
/// and drives the progress bar / step counter / partial-preview view.
public enum ImageGenEvent: Sendable {
    /// A generation step has completed. `step` is 1-indexed, `total` is the
    /// step count scheduled for the run (from `ImageGenSettings.steps`).
    case step(step: Int, total: Int, preview: Data?)
    /// The generation succeeded; output was written to `url`.
    case completed(url: URL)
    /// Generation failed. `hfAuth` is true when the failure was a 401/403
    /// against a gated Hugging Face repo so the UI can show the "gated
    /// model" banner instead of a generic error.
    case failed(message: String, hfAuth: Bool)
    /// User cancelled the job via Stop.
    case cancelled
}
