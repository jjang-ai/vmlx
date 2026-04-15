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
