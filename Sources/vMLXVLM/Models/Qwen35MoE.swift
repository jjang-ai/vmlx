//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5_moe
//

import MLX

public final class Qwen35MoE: Qwen35 {
    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var remapped = [String: MLXArray]()
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            remapped[key] = value
        }

        // §434 — HF-canonical format fold-in. Raw Hugging Face Qwen3.5-MoE
        // (and Holo3) ship per-expert split weights:
        //   mlp.experts.{N}.gate_proj.weight  shape [I, H]
        //   mlp.experts.{N}.up_proj.weight    shape [I, H]
        //   mlp.experts.{N}.down_proj.weight  shape [H, I]
        // The MLX-converted format consolidates them into:
        //   mlp.experts.gate_up_proj  shape [E, 2I, H]
        //   mlp.experts.down_proj     shape [E, H, I]
        // Detect per-expert keys and stack them so the SparseMoeBlock
        // sees a tensor of the expected rank. Without this, raw-HF
        // Holo3 / Qwen3.5-MoE bundles fail with `unhandledKeys: experts`.
        let numExperts = config.textConfiguration.numExperts
        for layer in 0 ..< config.textConfiguration.hiddenLayers {
            let prefixes = [
                "model.language_model.layers.\(layer).mlp",
                "language_model.model.layers.\(layer).mlp",
                "model.layers.\(layer).mlp",
            ]
            for prefix in prefixes {
                let firstGateKey = "\(prefix).experts.0.gate_proj.weight"
                guard remapped[firstGateKey] != nil else { continue }
                var gates: [MLXArray] = []
                var ups: [MLXArray] = []
                var downs: [MLXArray] = []
                gates.reserveCapacity(numExperts)
                ups.reserveCapacity(numExperts)
                downs.reserveCapacity(numExperts)
                var ok = true
                for e in 0 ..< numExperts {
                    let gK = "\(prefix).experts.\(e).gate_proj.weight"
                    let uK = "\(prefix).experts.\(e).up_proj.weight"
                    let dK = "\(prefix).experts.\(e).down_proj.weight"
                    guard let g = remapped.removeValue(forKey: gK),
                          let u = remapped.removeValue(forKey: uK),
                          let d = remapped.removeValue(forKey: dK)
                    else { ok = false; break }
                    gates.append(g)
                    ups.append(u)
                    downs.append(d)
                }
                if ok {
                    let stackedGate = stacked(gates, axis: 0)  // [E, I, H]
                    let stackedUp = stacked(ups, axis: 0)      // [E, I, H]
                    let stackedDown = stacked(downs, axis: 0)  // [E, H, I]
                    let gateUp = concatenated([stackedGate, stackedUp], axis: 1) // [E, 2I, H]
                    remapped["\(prefix).experts.gate_up_proj"] = gateUp
                    remapped["\(prefix).experts.down_proj"] = stackedDown
                }
            }
        }

        for layer in 0 ..< config.textConfiguration.hiddenLayers {
            let prefixes = [
                "model.language_model.layers.\(layer).mlp",
                "language_model.model.layers.\(layer).mlp",
            ]

            for prefix in prefixes {
                let gateUpKey = "\(prefix).experts.gate_up_proj"
                if let gateUp = remapped.removeValue(forKey: gateUpKey) {
                    let mid = gateUp.dim(-2) / 2
                    remapped["\(prefix).switch_mlp.gate_proj.weight"] =
                        gateUp[
                            .ellipsis, ..<mid, 0...]
                    remapped["\(prefix).switch_mlp.up_proj.weight"] =
                        gateUp[
                            .ellipsis, mid..., 0...]

                    let downProjKey = "\(prefix).experts.down_proj"
                    if let downProj = remapped.removeValue(forKey: downProjKey) {
                        remapped["\(prefix).switch_mlp.down_proj.weight"] = downProj
                    }
                }
            }
        }

        return super.sanitize(weights: remapped)
    }
}
