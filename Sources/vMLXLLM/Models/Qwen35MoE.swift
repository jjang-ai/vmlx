//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import vMLXLMCommon
import MLXNN

public struct Qwen35Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Qwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen35TextConfiguration(from: decoder)
        }
    }
}

public class Qwen35MoEModel: Qwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            newWeights[key] = value
        }

        for l in 0 ..< languageModel.configuration.hiddenLayers {
            let prefix = "language_model.model.layers.\(l).mlp"
            let gateUpKey = "\(prefix).experts.gate_up_proj"
            if let gateUp = newWeights[gateUpKey] {
                newWeights[gateUpKey] = nil
                let mid = gateUp.dim(-2) / 2
                newWeights["\(prefix).switch_mlp.gate_proj.weight"] =
                    gateUp[.ellipsis, ..<mid, 0...]
                newWeights["\(prefix).switch_mlp.up_proj.weight"] =
                    gateUp[.ellipsis, mid..., 0...]
                if let downProj = newWeights["\(prefix).experts.down_proj"] {
                    newWeights["\(prefix).experts.down_proj"] = nil
                    newWeights["\(prefix).switch_mlp.down_proj.weight"] = downProj
                }
            }
        }

        return languageModel.sanitize(weights: newWeights)
    }
}
