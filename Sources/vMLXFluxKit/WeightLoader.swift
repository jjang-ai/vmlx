import Foundation
@preconcurrency import MLX
import MLXNN
@preconcurrency import vMLXLMCommon

// MARK: - WeightLoader
//
// Loads safetensors weight shards from a local model directory and
// applies vMLX's JANG-aware remapping if the model has a `jang_config.json`.
//
// Flow (JANG-less path):
//   1. Read `model.safetensors.index.json` to enumerate all shards.
//   2. Open each shard via `MLX.loadArrays(url:)`.
//   3. Merge into a single `[String: MLXArray]` dict keyed by weight name.
//
// Flow (JANG path):
//   1. Parse `jang_config.json` via `JangLoader` (reused from vmlx-swift-lm).
//   2. Enumerate shards as above.
//   3. Apply per-layer quantization metadata before use — the
//      `QuantizedLinear` modules in the model module tree check the
//      `jangConfig.quantization.bitWidthsUsed[layer_idx]` to pick
//      the right decode path.
//
// For now this file is the SCAFFOLD — weight loading returns a dict,
// per-layer JANG remapping plugs in when the first real model port
// (Flux1Schnell) has a concrete module tree to apply it to.

public struct LoadedWeights: Sendable {
    public let weights: [String: MLXArray]
    public let jangConfig: vMLXLMCommon.JangConfig?

    public init(weights: [String: MLXArray], jangConfig: vMLXLMCommon.JangConfig? = nil) {
        self.weights = weights
        self.jangConfig = jangConfig
    }
}

public enum WeightLoader {

    /// Load all safetensors shards from a model directory and return a
    /// merged dict. If the directory contains `jang_config.json`, the
    /// parsed config is returned alongside so the caller can apply
    /// per-layer quantization during module construction.
    public static func load(from directory: URL) throws -> LoadedWeights {
        // Detect JANG first so the caller gets the config regardless of
        // the shard layout.
        let jang = try JangBridge.detect(at: directory)

        // Enumerate safetensors shards.
        let shards = try enumerateShards(in: directory)
        guard !shards.isEmpty else {
            throw FluxError.weightsNotFound(directory)
        }

        // Merge all shards into a single dict.
        var merged: [String: MLXArray] = [:]
        for shard in shards {
            let arrays = try MLX.loadArrays(url: shard)
            for (k, v) in arrays {
                merged[k] = v
            }
        }

        return LoadedWeights(weights: merged, jangConfig: jang.config)
    }

    /// Enumerate .safetensors files in the directory. Prefers the
    /// `model.safetensors.index.json` manifest when present (so we load
    /// shards in deterministic order), falling back to a sorted glob.
    /// If neither is found at the top level, falls through to the
    /// diffusion-layout subdirs (`transformer/`, `text_encoder/`, `vae/`)
    /// — Z-Image / Flux snapshots ship weights per-component that way
    /// and this loader needs to see all of them in one merge.
    private static func enumerateShards(in directory: URL) throws -> [URL] {
        let fm = FileManager.default
        // Try top-level index first.
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = obj["weight_map"] as? [String: String] {
                let shardNames = Set(weightMap.values)
                return shardNames
                    .sorted()
                    .map { directory.appendingPathComponent($0) }
            }
        }
        // Top-level glob.
        let topLevel = (try? fm.contentsOfDirectory(atPath: directory.path)) ?? []
        let topShards = topLevel
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
            .map { directory.appendingPathComponent($0) }
        if !topShards.isEmpty {
            return topShards
        }
        // Diffusion-layout subdir fallback. Z-Image / Flux variants store
        // shards under `transformer/`, `text_encoder/`, and `vae/` —
        // collect all three so the merged weight dict carries every
        // component the model builder needs. Prefer each subdir's own
        // `model.safetensors.index.json` when present for deterministic
        // order, otherwise enumerate.
        var collected: [URL] = []
        for sub in ["transformer", "text_encoder", "text_encoder_2", "vae", "scheduler"] {
            let subURL = directory.appendingPathComponent(sub)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: subURL.path, isDirectory: &isDir),
                  isDir.boolValue
            else { continue }
            let subIndex = subURL.appendingPathComponent("model.safetensors.index.json")
            if fm.fileExists(atPath: subIndex.path),
               let data = try? Data(contentsOf: subIndex),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = obj["weight_map"] as? [String: String]
            {
                let names = Set(weightMap.values).sorted()
                collected.append(contentsOf: names.map { subURL.appendingPathComponent($0) })
                continue
            }
            let entries = (try? fm.contentsOfDirectory(atPath: subURL.path)) ?? []
            let shards = entries
                .filter { $0.hasSuffix(".safetensors") }
                .sorted()
                .map { subURL.appendingPathComponent($0) }
            collected.append(contentsOf: shards)
        }
        return collected
    }
}
