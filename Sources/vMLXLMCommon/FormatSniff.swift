// SPDX-License-Identifier: Apache-2.0
//
// FormatSniff — shared `weight_format == "mxtq"` sniffer used by the
// LLM and VLM factories to route JANGTQ-quantised checkpoints to their
// `*JANGTQModel` class.
//
// Before this helper the sniff was open-coded in four separate factory
// entries (`qwen3_5_moe` LLM + VLM, `minimax_m2`, `glm4_moe`) and drifted:
// the Qwen VLM path correctly handled nested-under-`text_config` AND
// top-level keys, but the MiniMax and GLM LLM paths only checked top-
// level, and none were case-insensitive. A checkpoint shipping
// `{"weight_format": "MXTQ"}` at top-level or under `text_config`
// silently fell through to the affine expander, blew up when the
// model tried to run `TurboQuantSwitchGLU` over non-existent
// `.tq_packed` tensors, and surfaced as a mysterious loader crash.
//
// One canonical sniffer keeps the four factory entries in lock-step.

import Foundation

public enum FormatSniff {

    /// JSON config shape: `weight_format` may live at the top level
    /// (plain text-only JANGTQ) OR nested under `text_config` (VLM
    /// wrappers where the JANGTQ-quantised language model sits inside
    /// a VLM model_type). Both are read, both are compared
    /// case-insensitively.
    private struct FormatCheck: Codable {
        let weightFormat: String?
        let textConfig: TextConfigCheck?
        enum CodingKeys: String, CodingKey {
            case weightFormat = "weight_format"
            case textConfig = "text_config"
        }
        struct TextConfigCheck: Codable {
            let weightFormat: String?
            enum CodingKeys: String, CodingKey {
                case weightFormat = "weight_format"
            }
        }
    }

    /// Returns `true` iff the raw `config.json` bytes declare
    /// `weight_format` as `"mxtq"` (case-insensitive, at either level).
    /// Returns `false` on decode error, missing field, or a different
    /// weight format — callers should fall through to the affine
    /// loader path.
    public static func isMXTQ(from data: Data) -> Bool {
        guard let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: data)
        else { return false }
        let top = check.weightFormat?.lowercased()
        let nested = check.textConfig?.weightFormat?.lowercased()
        return top == "mxtq" || nested == "mxtq"
    }

    public static func hasVisionConfig(from data: Data) -> Bool {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return json["vision_config"] != nil
    }
}
