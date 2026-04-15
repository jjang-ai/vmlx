import Foundation
import MLX
import vMLXLMCommon
import vMLXLLM

// JANG-DFlash (block-diffusion drafter + DDTree speculative decoding)
// engine integration. The drafter weights, target adapter, and spec-dec
// driver all live on the Engine actor so stream invocations can reach
// them without re-loading per request.
//
// Surface:
//   - `loadDFlashDrafter(from:config:)` — load a safetensors checkpoint
//     and bind a target adapter for the currently-loaded model
//   - `unloadDFlashDrafter()` — free the drafter + adapter
//   - `dflashIsReady()` — check whether a spec-dec driver can be built
//   - `makeDFlashSpecDec(settings:)` — build a driver configured with
//     user settings (tap layers, block size, top-k, num paths)
//
// The heavy work — forward-with-taps, drafter block forward, DDTree
// beam, tree-attention verify — all lives in JangDFlashSpecDec. This
// file only glues that to the Engine actor's lifecycle.

extension Engine {

    // MARK: - Drafter lifecycle

    /// Load a JANG-DFlash drafter checkpoint from `url`. Replaces any
    /// previously-loaded drafter. On success, `dflashIsReady()` flips
    /// to true for target models that conform to JangDFlashTarget.
    ///
    /// The drafter config is read from an accompanying `dflash.json`
    /// next to the safetensors file if `config` is nil. When both are
    /// missing we fall back to defaults from `JangDFlashConfig()` —
    /// that matches the hardcoded shape the MiniMax-M2.7 drafter was
    /// trained with and is correct for most checkpoints in the wild.
    public func loadDFlashDrafter(
        from url: URL,
        config: JangDFlashConfig? = nil
    ) async throws {
        let resolvedConfig: JangDFlashConfig
        if let config { resolvedConfig = config }
        else if let fromDisk = try? Self.readDFlashConfig(nextTo: url) {
            resolvedConfig = fromDisk
        } else {
            resolvedConfig = JangDFlashConfig()
        }

        // Drafter ↔ target shape validation. `tapDim` must equal
        // `num_tap_layers × target_hidden_dim`. A mismatch means the
        // drafter was trained for a different target model — loading
        // would silent-corrupt on every request. Refuse at load time
        // with a clear error. Validation is deferred when no target
        // adapter is bound (user may stage the drafter before loading
        // a compatible model).
        let g = await self.settings.global()
        if self._dflashTarget != nil {
            let expectedTapDim =
                Self.parseTapLayers(g.dflashTapLayers).count * g.dflashTargetHiddenDim
            if expectedTapDim > 0 && resolvedConfig.tapDim != expectedTapDim {
                throw EngineError.notImplemented(
                    "DFlash drafter tapDim \(resolvedConfig.tapDim) does not match "
                    + "target (num_tap_layers × target_hidden_dim = \(expectedTapDim)). "
                    + "Drafter and target must be trained together.")
            }
        } else {
            await self.log(.warn, "engine",
                "dflash drafter loaded without a bound target — shape "
                + "check deferred until a compatible model is loaded.")
        }

        let drafter = try JangDFlashLoader.loadNew(
            config: resolvedConfig,
            from: url,
            castToBF16: true
        )
        self._dflashDrafter = drafter
        self._dflashDrafterURL = url
        self._dflashDrafterConfig = resolvedConfig
        await self.log(.info, "engine",
            "dflash drafter loaded from \(url.lastPathComponent) "
            + "(blockSize=\(resolvedConfig.blockSize), layers=\(resolvedConfig.numLayers), "
            + "tapDim=\(resolvedConfig.tapDim))")
    }

    /// Auto-load the drafter from the persisted settings path, called
    /// from `Engine.load` completion so the drafter survives app
    /// relaunch. Silent no-op when settings don't have a path or the
    /// drafter is already loaded.
    internal func autoLoadDFlashDrafterIfConfigured() async {
        let g = await self.settings.global()
        guard g.dflash else { return }
        guard !g.dflashDrafterPath.isEmpty else {
            await self.log(.warn, "engine",
                "dflash enabled but `dflashDrafterPath` is empty — skipping auto-load.")
            return
        }
        if self._dflashDrafter != nil { return }
        let url = URL(fileURLWithPath: g.dflashDrafterPath)
        do {
            try await self.loadDFlashDrafter(from: url)
        } catch {
            await self.log(.warn, "engine",
                "dflash auto-load failed: \(error) — falling back to standard path.")
        }
    }

    /// Drop any loaded DFlash drafter. Safe to call when nothing is loaded.
    public func unloadDFlashDrafter() async {
        if self._dflashDrafter != nil {
            await self.log(.info, "engine", "dflash drafter unloaded")
        }
        self._dflashDrafter = nil
        self._dflashDrafterURL = nil
        self._dflashDrafterConfig = nil
    }

    /// True when a drafter is loaded AND the currently-loaded target
    /// model exposes the `JangDFlashTarget` hook (MiniMax family only
    /// at v1). False when either side is missing — callers fall back
    /// to the standard token iterator.
    public func dflashIsReady() -> Bool {
        self._dflashDrafter != nil && self._dflashTargetAdapter() != nil
    }

    /// Path of the currently-loaded drafter, if any. Used by the UI /
    /// admin `/status` route to surface what's active.
    public func dflashDrafterPath() -> URL? { self._dflashDrafterURL }

    // MARK: - Driver construction

    /// Build a `JangDFlashSpecDec` driver from the currently-loaded
    /// drafter and the currently-loaded target model, configured with
    /// user settings. Returns nil when preconditions aren't met.
    ///
    /// Parameters consumed from `settings`:
    /// `dflashBlockSize`, `dflashTopK`, `dflashNumPaths`, `dflashTapLayers`
    /// (comma-separated int list), `dflashTargetHiddenDim`.
    public func makeDFlashSpecDec(
        settings: GlobalSettings
    ) -> JangDFlashSpecDec? {
        guard let drafter = self._dflashDrafter as? JangDFlashDrafter,
              let target = self._dflashTargetAdapter()
        else {
            return nil
        }
        var cfg = JangDFlashSpecConfig()
        if settings.dflashBlockSize > 0 {
            cfg.blockSize = settings.dflashBlockSize
        }
        if settings.dflashTopK > 0 {
            cfg.topK = settings.dflashTopK
        }
        if settings.dflashNumPaths > 0 {
            cfg.numPaths = settings.dflashNumPaths
        }
        if settings.dflashTargetHiddenDim > 0 {
            cfg.targetHiddenDim = settings.dflashTargetHiddenDim
        }
        let parsed = Self.parseTapLayers(settings.dflashTapLayers)
        if !parsed.isEmpty {
            cfg.tapLayers = Set(parsed)
        }
        return JangDFlashSpecDec(target: target, drafter: drafter, cfg: cfg)
    }

    // MARK: - Internal helpers

    /// Adapter over the currently-loaded target model. MiniMax is the
    /// only family that currently conforms. Returns nil for every other
    /// architecture so callers cleanly skip DFlash on unsupported models.
    private func _dflashTargetAdapter() -> (any JangDFlashTarget)? {
        guard let container = self.loaded else { return nil }
        // ModelContainer.perform is the only legal access path, but we
        // want a synchronous adapter handle here. Cache an adapter by
        // walking the underlying model type during load via
        // `setDFlashTargetAdapter(for:)`. For now, the adapter is
        // produced on demand by the caller who already holds a
        // ModelContext — see `setDFlashTargetAdapter`.
        _ = container
        return self._dflashTarget as? (any JangDFlashTarget)
    }

    /// Register a target adapter produced by the model-specific load
    /// path. Called from `Engine.load` (or the chat/batched loader
    /// variants) after the target model is in-hand. The Model-specific
    /// code is the only place with enough type info to build the right
    /// adapter (e.g. `MiniMaxDFlashTarget`), so it hands the adapter
    /// back to the engine through this setter.
    internal func setDFlashTargetAdapter(_ adapter: (any JangDFlashTarget)?) {
        self._dflashTarget = adapter
    }

    /// Inspects the freshly-loaded container and binds a target adapter
    /// when the model conforms. Called from `Engine.load` right after
    /// `setLoaded` + Flash MoE apply. Silent no-op for non-conforming
    /// models so the unload-path doesn't need a matching unbind call.
    internal func bindDFlashTargetIfEligible(
        container: vMLXLMCommon.ModelContainer
    ) async {
        let adapter: (any JangDFlashTarget)? = await container.perform { ctx in
            // MiniMax is the only family with a registered JangDFlashTarget
            // adapter today (MiniMaxDFlashTarget). Add more `if let ... as?`
            // branches here as more model families pick up the hooks.
            if let mm = ctx.model as? MiniMaxModel {
                return MiniMaxDFlashTarget(mm) as (any JangDFlashTarget)
            }
            if let m4 = ctx.model as? Mistral4Model {
                return Mistral4DFlashTarget(m4) as (any JangDFlashTarget)
            }
            if let d3 = ctx.model as? DeepseekV3Model {
                return DeepseekV3DFlashTarget(d3) as (any JangDFlashTarget)
            }
            return nil
        }
        self._dflashTarget = adapter
        if adapter != nil {
            await self.log(.info, "engine",
                "dflash target adapter bound (model supports speculative decoding)")
        }
    }

    private static func readDFlashConfig(nextTo url: URL) throws -> JangDFlashConfig {
        let sibling = url.deletingLastPathComponent().appendingPathComponent("dflash.json")
        let data = try Data(contentsOf: sibling)
        return try JSONDecoder().decode(JangDFlashConfig.self, from: data)
    }

    internal static func parseTapLayers(_ csv: String) -> [Int] {
        csv.split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
}
