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
        _ = QwenImageEdit._register
        _ = FIBO._register
        _ = SeedVR2._register
    }
}
