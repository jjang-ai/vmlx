// SPDX-License-Identifier: Apache-2.0
//
// ZImageDiTWeightRemap — translate HuggingFace Z-Image transformer
// safetensors keys to the Swift module tree.
//
// Reference (ground truth):
//   /tmp/mflux-ref/src/mflux/models/z_image/weights/z_image_weight_mapping.py
//   ZImageWeightMapping.get_transformer_mapping()
//
// Z-Image's transformer key namespace differs from FLUX:
//   - flat under the safetensors root (no `model.` prefix, no
//     `transformer/` prefix once we've already loaded the subdir)
//   - top-level modules: `t_embedder`, `cap_embedder`, `x_pad_token`,
//     `cap_pad_token`, `all_x_embedder.2-1`, `all_final_layer.2-1`
//   - per-layer towers: `noise_refiner.{i}`, `context_refiner.{i}`,
//     `layers.{i}` (the main DiT stack)
//
// The MFLUX mapping is largely identity for the on-disk keys with
// these tweaks:
//   - `t_embedder.mlp.0.*`  → `t_embedder.linear1.*`
//   - `t_embedder.mlp.2.*`  → `t_embedder.linear2.*`
//   - `all_final_layer.2-1.adaLN_modulation.1.*`
//                           → `all_final_layer.2-1.adaLN_modulation.0.*`
//
// All other transformer keys pass through unchanged.

import Foundation
@preconcurrency import MLX

public enum ZImageDiTWeightRemap {

    /// 1:1 key rename from HF safetensors → Swift module tree. Mirrors
    /// `ZImageWeightMapping.get_transformer_mapping()`.
    public static func remapKey(_ key: String) -> String {
        // t_embedder.mlp.{0,2} → linear{1,2}
        if key.hasPrefix("t_embedder.mlp.0.") {
            return "t_embedder.linear1." + String(key.dropFirst("t_embedder.mlp.0.".count))
        }
        if key.hasPrefix("t_embedder.mlp.2.") {
            return "t_embedder.linear2." + String(key.dropFirst("t_embedder.mlp.2.".count))
        }
        // all_final_layer.2-1.adaLN_modulation.1.* → ...adaLN_modulation.0.*
        // (mflux flattens the `nn.Sequential(SiLU(), Linear(...))` into a
        // bare Linear under index 0.)
        let finalAdaPrefix = "all_final_layer.2-1.adaLN_modulation.1."
        if key.hasPrefix(finalAdaPrefix) {
            return "all_final_layer.2-1.adaLN_modulation.0."
                + String(key.dropFirst(finalAdaPrefix.count))
        }
        return key
    }

    /// Apply `remapKey` over a full `[String: MLXArray]` weight dict.
    public static func remap(_ all: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(all.count)
        for (k, v) in all {
            out[remapKey(k)] = v
        }
        return out
    }
}
