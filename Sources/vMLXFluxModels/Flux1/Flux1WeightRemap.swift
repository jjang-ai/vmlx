// SPDX-License-Identifier: Apache-2.0
//
// Flux1WeightRemap — translate FLUX.1 transformer safetensors keys
// onto the Swift `FluxDiTModel` module tree's parameter keys.
//
// Two source-of-truth checkpoint formats exist for FLUX.1:
//
//   1. BFL upstream (canonical) — single `flux1-schnell.safetensors` /
//      `flux1-dev.safetensors` shipped at the snapshot root by
//      black-forest-labs/FLUX.1-{schnell,dev}. Uses fused QKV linears
//      (`double_blocks.{i}.img_attn.qkv.weight`). This is the layout
//      `Sources/vMLXFluxKit/FluxDiT.swift`'s module tree models, so
//      remap is mostly a snake_case → camelCase rename.
//
//   2. HF Diffusers (`transformer/diffusion_pytorch_model-*.safetensors`)
//      — split `to_q`/`to_k`/`to_v` linears, separate `attn.norm_q` /
//      `norm_k` etc. This format requires per-block QKV concatenation
//      to fit the Swift module tree's fused QKV. The remap below emits
//      the right *target* keys for the BFL-shaped output but leaves
//      QKV fusion as a follow-up; callers who only ship diffusers-
//      format weights must run a fuse pass after the rename.
//
// Reference for the diffusers→BFL pattern:
//   /tmp/mflux-ref/src/mflux/models/flux/weights/flux_weight_mapping.py
//   /tmp/mflux-ref/src/mflux/models/flux/model/flux_transformer/
//
// Target Swift parameter-key conventions (from FluxDiT.swift @ line 440):
//
//   img_in.weight                              → imgIn.weight
//   time_in.in_layer.weight                    → timeIn0.weight
//   time_in.out_layer.weight                   → timeIn2.weight
//   vector_in.in_layer.weight                  → vectorIn0.weight
//   vector_in.out_layer.weight                 → vectorIn2.weight
//   guidance_in.in_layer.weight                → guidanceIn0.weight (Dev only)
//   guidance_in.out_layer.weight               → guidanceIn2.weight
//   txt_in.weight                              → txtIn.weight
//   double_blocks.{i}.img_mod.lin.weight       → doubleBlocks.{i}.imgMod.linear.weight
//   double_blocks.{i}.img_attn.qkv.weight      → doubleBlocks.{i}.imgAttnQKV.weight
//   double_blocks.{i}.img_attn.proj.weight     → doubleBlocks.{i}.imgAttnProj.weight
//   double_blocks.{i}.img_attn.norm.query_norm.scale
//                                              → doubleBlocks.{i}.imgAttnNorm.qNorm.weight
//   double_blocks.{i}.img_attn.norm.key_norm.scale
//                                              → doubleBlocks.{i}.imgAttnNorm.kNorm.weight
//   double_blocks.{i}.img_mlp.0.weight         → doubleBlocks.{i}.imgMlp0.weight
//   double_blocks.{i}.img_mlp.2.weight         → doubleBlocks.{i}.imgMlp2.weight
//   (txt_mod / txt_attn / txt_mlp follow the same pattern)
//   single_blocks.{i}.modulation.lin.weight    → singleBlocks.{i}.mod.linear.weight
//   single_blocks.{i}.linear1.weight           → singleBlocks.{i}.linear1.weight
//   single_blocks.{i}.linear2.weight           → singleBlocks.{i}.linear2.weight
//   single_blocks.{i}.norm.query_norm.scale    → singleBlocks.{i}.qkNorm.qNorm.weight
//   single_blocks.{i}.norm.key_norm.scale      → singleBlocks.{i}.qkNorm.kNorm.weight
//   final_layer.linear.weight                  → finalLayer.linear.weight
//   final_layer.adaLN_modulation.1.weight      → finalLayer.mod.weight

import Foundation
@preconcurrency import MLX

public enum Flux1WeightRemap {

    /// Apply all key renames in sequence. Returns the rewritten key.
    /// Pure function so unit tests can assert exact transforms.
    public static func remapKey(_ key: String) -> String {
        var k = key

        // Strip the `model.diffusion_model.` prefix some HF mirrors add
        // when shipping the BFL checkpoint inside a wrapper module.
        if k.hasPrefix("model.diffusion_model.") {
            k = String(k.dropFirst("model.diffusion_model.".count))
        }

        // ── Diffusers → BFL: block-prefix renames ────────────────────
        // diffusers calls the dual-stream stack `transformer_blocks.*`
        // and the single-stream stack `single_transformer_blocks.*`.
        // BFL (and Swift) call them `double_blocks` / `single_blocks`.
        if k.hasPrefix("transformer_blocks.") {
            k = "double_blocks." + k.dropFirst("transformer_blocks.".count)
        }
        if k.hasPrefix("single_transformer_blocks.") {
            k = "single_blocks." + k.dropFirst("single_transformer_blocks.".count)
        }

        // ── Top-level entry / time embed renames ─────────────────────
        // FLUX.1 BFL uses underscored names; Swift module tree uses
        // camelCase property identifiers, which is what
        // `Module.parameters()` keys on. Map each BFL leaf to its
        // Swift counterpart.
        let topLevel: [(String, String)] = [
            ("img_in.weight",                             "imgIn.weight"),
            ("img_in.bias",                               "imgIn.bias"),
            ("time_in.in_layer.weight",                   "timeIn0.weight"),
            ("time_in.in_layer.bias",                     "timeIn0.bias"),
            ("time_in.out_layer.weight",                  "timeIn2.weight"),
            ("time_in.out_layer.bias",                    "timeIn2.bias"),
            ("vector_in.in_layer.weight",                 "vectorIn0.weight"),
            ("vector_in.in_layer.bias",                   "vectorIn0.bias"),
            ("vector_in.out_layer.weight",                "vectorIn2.weight"),
            ("vector_in.out_layer.bias",                  "vectorIn2.bias"),
            ("guidance_in.in_layer.weight",               "guidanceIn0.weight"),
            ("guidance_in.in_layer.bias",                 "guidanceIn0.bias"),
            ("guidance_in.out_layer.weight",              "guidanceIn2.weight"),
            ("guidance_in.out_layer.bias",                "guidanceIn2.bias"),
            ("txt_in.weight",                             "txtIn.weight"),
            ("txt_in.bias",                               "txtIn.bias"),
            ("final_layer.linear.weight",                 "finalLayer.linear.weight"),
            ("final_layer.linear.bias",                   "finalLayer.linear.bias"),
            // BFL ships final modulation under `adaLN_modulation.1.*`
            // (the Linear inside a 2-element Sequential whose `[0]` is
            // a SiLU). Swift collapses the SiLU into the forward and
            // names the Linear `mod`.
            ("final_layer.adaLN_modulation.1.weight",     "finalLayer.mod.weight"),
            ("final_layer.adaLN_modulation.1.bias",       "finalLayer.mod.bias"),
        ]
        for (from, to) in topLevel where k == from {
            return to
        }

        // ── Per-block patterns (double_blocks.{i}.*) ─────────────────
        if let renamed = remapDoubleBlockKey(k) { return renamed }
        if let renamed = remapSingleBlockKey(k) { return renamed }

        // Unknown key — return unchanged so callers can detect orphans.
        return k
    }

    /// Apply `remapKey` to every entry in a checkpoint dict. Last-write
    /// wins on collision (shouldn't happen with a clean BFL or
    /// diffusers checkpoint but it's better to ship a deterministic
    /// merge than crash).
    public static func remap(_ all: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(all.count)
        for (k, v) in all {
            out[remapKey(k)] = v
        }
        return out
    }

    /// Diff the remapped checkpoint against the Swift module's parameter
    /// keys. Returns:
    ///   - `missing`: keys the Swift module needs but the checkpoint
    ///     didn't supply (these become loader errors / random init).
    ///   - `extra`: keys the checkpoint had that the Swift module
    ///     doesn't expose (these get silently dropped by `update`).
    public static func diff(
        remapped: [String: MLXArray],
        modelKeys: Set<String>
    ) -> (missing: [String], extra: [String]) {
        let provided = Set(remapped.keys)
        let missing = modelKeys.subtracting(provided).sorted()
        let extra = provided.subtracting(modelKeys).sorted()
        return (missing, extra)
    }

    // MARK: - Per-block dispatch

    private static func remapDoubleBlockKey(_ k: String) -> String? {
        // Split `double_blocks.{i}.<rest>` once.
        let prefix = "double_blocks."
        guard k.hasPrefix(prefix) else { return nil }
        let afterPrefix = k.dropFirst(prefix.count)
        guard let dot = afterPrefix.firstIndex(of: ".") else { return nil }
        let idx = String(afterPrefix[..<dot])
        let rest = String(afterPrefix[afterPrefix.index(after: dot)...])

        // BFL leaf → Swift leaf, expressed as exact-match table.
        let leafTable: [(String, String)] = [
            // img modulation
            ("img_mod.lin.weight",                          "imgMod.linear.weight"),
            ("img_mod.lin.bias",                            "imgMod.linear.bias"),
            // img attention
            ("img_attn.qkv.weight",                         "imgAttnQKV.weight"),
            ("img_attn.qkv.bias",                           "imgAttnQKV.bias"),
            ("img_attn.proj.weight",                        "imgAttnProj.weight"),
            ("img_attn.proj.bias",                          "imgAttnProj.bias"),
            ("img_attn.norm.query_norm.scale",              "imgAttnNorm.qNorm.weight"),
            ("img_attn.norm.key_norm.scale",                "imgAttnNorm.kNorm.weight"),
            // img mlp
            ("img_mlp.0.weight",                            "imgMlp0.weight"),
            ("img_mlp.0.bias",                              "imgMlp0.bias"),
            ("img_mlp.2.weight",                            "imgMlp2.weight"),
            ("img_mlp.2.bias",                              "imgMlp2.bias"),
            // txt modulation
            ("txt_mod.lin.weight",                          "txtMod.linear.weight"),
            ("txt_mod.lin.bias",                            "txtMod.linear.bias"),
            // txt attention
            ("txt_attn.qkv.weight",                         "txtAttnQKV.weight"),
            ("txt_attn.qkv.bias",                           "txtAttnQKV.bias"),
            ("txt_attn.proj.weight",                        "txtAttnProj.weight"),
            ("txt_attn.proj.bias",                          "txtAttnProj.bias"),
            ("txt_attn.norm.query_norm.scale",              "txtAttnNorm.qNorm.weight"),
            ("txt_attn.norm.key_norm.scale",                "txtAttnNorm.kNorm.weight"),
            // txt mlp
            ("txt_mlp.0.weight",                            "txtMlp0.weight"),
            ("txt_mlp.0.bias",                              "txtMlp0.bias"),
            ("txt_mlp.2.weight",                            "txtMlp2.weight"),
            ("txt_mlp.2.bias",                              "txtMlp2.bias"),
        ]
        for (from, to) in leafTable where rest == from {
            return "doubleBlocks.\(idx).\(to)"
        }
        return nil
    }

    private static func remapSingleBlockKey(_ k: String) -> String? {
        let prefix = "single_blocks."
        guard k.hasPrefix(prefix) else { return nil }
        let afterPrefix = k.dropFirst(prefix.count)
        guard let dot = afterPrefix.firstIndex(of: ".") else { return nil }
        let idx = String(afterPrefix[..<dot])
        let rest = String(afterPrefix[afterPrefix.index(after: dot)...])

        let leafTable: [(String, String)] = [
            ("modulation.lin.weight",                       "mod.linear.weight"),
            ("modulation.lin.bias",                         "mod.linear.bias"),
            ("linear1.weight",                              "linear1.weight"),
            ("linear1.bias",                                "linear1.bias"),
            ("linear2.weight",                              "linear2.weight"),
            ("linear2.bias",                                "linear2.bias"),
            ("norm.query_norm.scale",                       "qkNorm.qNorm.weight"),
            ("norm.key_norm.scale",                         "qkNorm.kNorm.weight"),
        ]
        for (from, to) in leafTable where rest == from {
            return "singleBlocks.\(idx).\(to)"
        }
        return nil
    }
}
