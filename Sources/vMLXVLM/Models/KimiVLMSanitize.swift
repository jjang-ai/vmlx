//
//  KimiVLMSanitize.swift
//  vMLXVLM
//
//  Load-time weight-key renames for Kimi K2.6 VL bundles.
//
//  Per `research/KIMI-K2.6-VMLX-INTEGRATION.md §2.2.1` / §2.6 #14:
//  the on-disk Kimi VL bundle stores the multimodal projector as
//
//      mm_projector.pre_norm.{weight,bias}
//      mm_projector.proj.0.{weight,bias}
//      mm_projector.proj.2.{weight,bias}
//
//  because the upstream weights came from a PyTorch
//  `nn.Sequential(LayerNorm, Linear, GELU, Linear)`. vMLX / mlx_vlm's
//  canonical VLM layout wants discrete named submodules, so the keys
//  are renamed at load time to
//
//      multi_modal_projector.pre_norm.{weight,bias}
//      multi_modal_projector.linear_1.{weight,bias}
//      multi_modal_projector.linear_2.{weight,bias}
//
//  This is the Swift mirror of `_hydrate_jangtq_model`'s Kimi VL
//  branch in `jang-tools/jang_tools/load_jangtq.py`. Pure key-
//  rewriting — no tensor modification — so it is safe to run on both
//  affine and JANGTQ (mxtq) Kimi VL bundles. The function is
//  idempotent: a bundle that already uses the canonical layout is
//  returned unchanged.
//
//  K5 §325. KimiVLM wrapper + MoonViT port (K3/K4) land in separate
//  files — this sanitize helper is safe to ship ahead of them so
//  that when the full VL wrapper arrives it just calls
//  `renameKimiMMProjectorKeys` at the top of its `sanitize(_:)`.
//
//  Created by Jinho Jang (eric@jangq.ai).
//

import Foundation
import MLX

/// Rewrite on-disk `mm_projector.*` keys to the canonical
/// `multi_modal_projector.*` layout that `KimiVLMModel.multiModalProjector`
/// will bind to.
///
/// Mapping (mirrors jang_tools `load_jangtq.py` Kimi VL branch):
///
/// | On-disk                      | vMLX canonical                           |
/// | ---------------------------- | ---------------------------------------- |
/// | `mm_projector.pre_norm.*`    | `multi_modal_projector.pre_norm.*`       |
/// | `mm_projector.proj.0.*`      | `multi_modal_projector.linear_1.*`       |
/// | `mm_projector.proj.2.*`      | `multi_modal_projector.linear_2.*`       |
///
/// - Parameter weights: the raw weight dict as returned by
///   `WeightLoader.load`. Keys matching the on-disk Kimi layout are
///   replaced in-place; all other keys pass through unchanged.
/// - Returns: the sanitized dict. Call site should assign the return
///   value back — the function does not mutate the argument.
public func renameKimiMMProjectorKeys(
    _ weights: [String: MLXArray]
) -> [String: MLXArray] {
    var out = weights
    let mappings: [(String, String)] = [
        ("mm_projector.pre_norm.", "multi_modal_projector.pre_norm."),
        ("mm_projector.proj.0.",   "multi_modal_projector.linear_1."),
        ("mm_projector.proj.2.",   "multi_modal_projector.linear_2."),
    ]
    for key in weights.keys {
        for (from, to) in mappings where key.hasPrefix(from) {
            let newKey = to + key.dropFirst(from.count)
            if out[newKey] == nil, let v = out.removeValue(forKey: key) {
                out[newKey] = v
            }
            break
        }
    }
    return out
}
