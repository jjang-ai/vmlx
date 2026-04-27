// Umbrella file for vMLXFluxModels. Re-exports the per-model public
// types so `import vMLXFluxModels` gives callers everything at once.
//
// Registration is done via `_register` statics inside each model file.
// This file just touches each type so the static initializer runs.

import Foundation

/// Force-register every model in the registry. Call from app launch or
/// the FluxEngine init to ensure all models are discoverable via
/// `ModelRegistry.lookup(_:)` before the first `load()` call.
public enum vMLXFluxModels {
    public static func registerAll() {
        _ = Flux1Schnell._register
        _ = Flux1Dev._register
        _ = Flux1Kontext._register
        _ = Flux1Fill._register
        _ = Flux2Klein._register
        _ = Flux2KleinEdit._register
        _ = ZImage._register
        _ = QwenImage._register
        // QwenImageEdit is owned by Track 2; registered below via
        // `QwenImageEditEditor`. The legacy `QwenImageEdit` class was
        // removed from `QwenImage.swift` by Track 1 since Track 1's
        // ownership is gen-only.
        _ = FIBO._register
        _ = Bria._register
        _ = SeedVR2._register
        // Track 2 edit ports — register LAST so they win the registry
        // last-write semantics over the temporary stubs in Flux1.swift /
        // QwenImage.swift. Once Track 1 removes those stubs, the V2
        // suffix on these classes can be dropped and the originals
        // restored to canonical names.
        _ = Flux1KontextEditor._register
        _ = Flux1FillEditor._register
        _ = QwenImageEditEditor._register
    }
}
