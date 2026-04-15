// Copyright © 2025 JANG. All rights reserved.
//
// Engine-level LoRA / DoRA runtime hot-swap surface.
//
// Lets callers load an adapter directory into the currently-loaded
// chat model without restarting the engine, swap between adapters
// across requests, or fuse the adapter into the base weights for
// permanent installation. Mirrors Python vmlx_engine's `/v1/adapters`
// REST surface, exposed here as actor-isolated `Engine.loadAdapter`,
// `unloadAdapter`, `fuseAdapter`, and `listAdapter`, plus a matching
// route group in `vMLXServer/Routes/AdapterRoutes.swift`.
//
// Design constraints:
//   - Only one adapter at a time is active on the chat model. Attempt
//     to load a second adapter implicitly unloads the first — this
//     matches Python's single-slot semantics and avoids the matrix
//     explosion of composing rank-8 LoRA layers.
//   - Fused adapters cannot be unloaded (fusion is destructive). The
//     caller must reload the base model. `unloadAdapter()` after a
//     `fuseAdapter()` returns an error.
//   - All model mutation happens inside `container.perform { ... }`
//     so it is actor-isolated with the forward pass — no partial-
//     load races where a request hits half-replaced layers.

import Foundation
import vMLXLMCommon

extension Engine {

    // MARK: - Types

    /// Describes the currently active adapter, if any. `nil` means no
    /// adapter is loaded and the model is running with the base
    /// weights as-shipped.
    public struct ActiveAdapterInfo: Sendable, Codable {
        /// Absolute path to the adapter directory that was last loaded.
        public let path: String
        /// Configuration read from `adapter_config.json`.
        public let fineTuneType: String
        public let rank: Int
        public let scale: Float
        public let numLayers: Int
        /// Whether this adapter has been fused into the base weights.
        /// Fused adapters survive model reloads only if the user also
        /// persists the fused weights; otherwise they are lost.
        public let fused: Bool

        public init(
            path: String, fineTuneType: String, rank: Int, scale: Float,
            numLayers: Int, fused: Bool
        ) {
            self.path = path
            self.fineTuneType = fineTuneType
            self.rank = rank
            self.scale = scale
            self.numLayers = numLayers
            self.fused = fused
        }
    }

    // MARK: - Public API

    /// Load a LoRA/DoRA adapter directory into the currently-loaded chat model.
    ///
    /// If another adapter is already active it is unloaded first.
    /// The adapter directory must contain `adapter_config.json` and
    /// `adapters.safetensors` in the format produced by `mlx-lm` / `jang_tools`
    /// LoRA trainers.
    ///
    /// - Parameter directory: Filesystem path to the adapter directory.
    /// - Throws: `EngineError.notLoaded` when no chat model is loaded,
    ///   `ModelAdapterError.incompatibleModelType` when the loaded
    ///   model does not conform to `LoRAModel` (protocol defined in
    ///   `vMLXLMCommon/Adapters/LoRA/LoRAModel.swift`), or a decode/IO
    ///   error when the directory is malformed.
    public func loadAdapter(directory: URL) async throws {
        guard let container = self.loaded else {
            throw EngineError.notLoaded
        }
        // Resolve absolutely and fail fast if the adapter files are missing.
        let resolved = directory.standardizedFileURL
        let configPath = resolved.appendingPathComponent("adapter_config.json")
        let weightsPath = resolved.appendingPathComponent("adapters.safetensors")
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            throw EngineError.adapterMissingFile(configPath.path)
        }
        guard FileManager.default.fileExists(atPath: weightsPath.path) else {
            throw EngineError.adapterMissingFile(weightsPath.path)
        }

        // If a previous adapter is already active on this container,
        // unload it first. Fused adapters block this path — the caller
        // must reload the base model.
        if let previous = self.activeAdapter {
            if previous.fused {
                throw EngineError.adapterAlreadyFused
            }
            try await container.perform { ctx in
                let container = try LoRAContainer.from(
                    directory: URL(fileURLWithPath: previous.path))
                container.unload(from: ctx.model)
            }
            self.activeAdapter = nil
        }

        // Load the new adapter, freezing the base and installing LoRA layers.
        let lora = try LoRAContainer.from(directory: resolved)
        try await container.perform { ctx in
            try lora.load(into: ctx.model)
        }

        let info = ActiveAdapterInfo(
            path: resolved.path,
            fineTuneType: lora.configuration.fineTuneType.rawValue,
            rank: lora.configuration.loraParameters.rank,
            scale: lora.configuration.loraParameters.scale,
            numLayers: lora.configuration.numLayers,
            fused: false
        )
        self.activeAdapter = info
        await logs.append(
            .info, category: "adapters",
            "Loaded LoRA adapter from \(resolved.lastPathComponent) " +
            "(rank=\(info.rank), scale=\(info.scale), layers=\(info.numLayers))"
        )
    }

    /// Unload the currently active adapter, restoring the base weights.
    ///
    /// No-op when no adapter is active. Returns an error when the
    /// active adapter was previously fused via `fuseAdapter` — fusion
    /// is destructive and can only be undone by reloading the model.
    public func unloadAdapter() async throws {
        guard let info = self.activeAdapter else { return }
        if info.fused {
            throw EngineError.adapterAlreadyFused
        }
        guard let container = self.loaded else {
            throw EngineError.notLoaded
        }
        let lora = try LoRAContainer.from(
            directory: URL(fileURLWithPath: info.path))
        await container.perform { ctx in
            lora.unload(from: ctx.model)
        }
        self.activeAdapter = nil
        await logs.append(
            .info, category: "adapters",
            "Unloaded LoRA adapter (\(info.path))"
        )
    }

    /// Permanently fuse the currently active adapter into the base weights.
    ///
    /// After fusion, `unloadAdapter` will fail and the only way to get
    /// back to the original base weights is to reload the model from
    /// disk. Useful for production deployment where a stable
    /// fine-tuned model is desired with no per-request LoRA overhead.
    public func fuseAdapter() async throws {
        guard let info = self.activeAdapter else {
            throw EngineError.adapterNotLoaded
        }
        if info.fused {
            return   // already fused, idempotent
        }
        guard let container = self.loaded else {
            throw EngineError.notLoaded
        }
        let lora = try LoRAContainer.from(
            directory: URL(fileURLWithPath: info.path))
        try await container.perform { ctx in
            try lora.fuse(with: ctx.model)
        }
        self.activeAdapter = ActiveAdapterInfo(
            path: info.path,
            fineTuneType: info.fineTuneType,
            rank: info.rank,
            scale: info.scale,
            numLayers: info.numLayers,
            fused: true
        )
        await logs.append(
            .info, category: "adapters",
            "Fused LoRA adapter into base weights (\(info.path))"
        )
    }

    /// Return info about the currently active adapter, or `nil` when
    /// the model is running with its unmodified base weights.
    public func listAdapter() -> ActiveAdapterInfo? {
        self.activeAdapter
    }

    /// Internal storage — mutated only from this extension. Exposed as
    /// an actor-isolated field on `Engine` (declared as stored property
    /// in the main type).
    internal var activeAdapter: ActiveAdapterInfo? {
        get { _activeAdapter }
        set { _activeAdapter = newValue }
    }
}
