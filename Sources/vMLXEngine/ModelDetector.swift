import Foundation

/// Three-tier model detection mirroring `vmlx_engine/model_configs.py::detect_model`:
///   1. text_config.model_type   (VLM wrapper case)
///   2. config.json model_type
///   3. name regex fallback
///
/// Plus jang_config.has_vision authoritative VLM flag (see is_mllm_model in mllm.py).
public struct ModelDetector {

    public enum Modality: String, Sendable { case text, vision }

    public struct Detected: Sendable {
        public var modelType: String
        public var modality: Modality
        public var hasJANG: Bool
        public var hasMXTQ: Bool
    }

    public static func detect(at directory: URL) throws -> Detected {
        let cfgURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfgURL) else {
            throw EngineError.modelNotFound(cfgURL)
        }
        let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]

        // Load the sidecar jang_config.json (separate file, not nested) —
        // this is where `has_vision` actually lives. Mirrors
        // api/utils.py::is_mllm_model resolution order.
        let jangURL = directory.appendingPathComponent("jang_config.json")
        let jangJSON: [String: Any] = {
            guard let d = try? Data(contentsOf: jangURL),
                  let j = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
            else { return [:] }
            return j
        }()
        let jangHasVision: Bool? = {
            if let hv = jangJSON["has_vision"] as? Bool { return hv }
            if let arch = jangJSON["architecture"] as? [String: Any],
               let hv = arch["has_vision"] as? Bool { return hv }
            // JANGTQ Omni bundles do not always stamp `has_vision`.
            // Nemotron-3-Nano-Omni currently carries
            // `jang_config.modality = "omni"` plus a `source_model`
            // modality, and `config.json` remains the text-only
            // `model_type = nemotron_h`. Treat any explicit multimodal
            // modality as authoritative vision routing so Engine.load
            // reaches VLMModelFactory instead of the LLM-only factory.
            if let m = (jangJSON["modality"] as? String)?.lowercased(),
               ["vision", "vl", "mllm", "multimodal", "omni"].contains(m) {
                return true
            }
            if let caps = jangJSON["capabilities"] as? [String: Any],
               let m = (caps["modality"] as? String)?.lowercased(),
               ["vision", "vl", "mllm", "multimodal", "omni"].contains(m) {
                return true
            }
            if let src = jangJSON["source_model"] as? [String: Any],
               let m = (src["modality"] as? String)?.lowercased(),
               ["vision", "vl", "mllm", "multimodal", "omni"].contains(m) {
                return true
            }
            return nil
        }()

        // Resolve model_type — prefer text_config.model_type for VLM
        // wrappers (mistral3/qwen3_vl) so the downstream family dispatch
        // sees the language-model arch name.
        let mt: String = {
            if let text = json["text_config"] as? [String: Any],
               let s = text["model_type"] as? String { return s }
            return (json["model_type"] as? String) ?? "unknown"
        }()
        let hasOmniSidecar = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("config_omni.json").path)

        // Tier 1: jang_config.has_vision is AUTHORITATIVE when set (may be
        // false for text-only JANG built from a VLM wrapper arch).
        if let hv = jangHasVision {
            return Detected(modelType: mt,
                            modality: hv ? .vision : .text,
                            hasJANG: true,
                            hasMXTQ: hasMXTQ(json, jangJSON: jangJSON))
        }

        // Nemotron-H Omni bundles use a text `config.json` plus
        // `config_omni.json` for the RADIO/Parakeet wrapper. If an older
        // bundle is missing `jang_config.modality`, the sidecar is still
        // enough to route through the VLM factory. Keep `modelType` as
        // `nemotron_h`; VLMModelFactory registers that alias for the Omni
        // wrapper because config.json dispatch still sees the LLM type.
        if mt == "nemotron_h", hasOmniSidecar {
            return Detected(modelType: mt,
                            modality: .vision,
                            hasJANG: hasJANG(json, jangJSON: jangJSON),
                            hasMXTQ: hasMXTQ(json, jangJSON: jangJSON))
        }

        // Tier 2: HF text_config wrapper ⇒ vision.
        if json["text_config"] is [String: Any] {
            return Detected(modelType: mt,
                            modality: .vision,
                            hasJANG: hasJANG(json, jangJSON: jangJSON),
                            hasMXTQ: hasMXTQ(json, jangJSON: jangJSON))
        }

        // Tier 3: top-level vision_config.
        let hasVision = json["vision_config"] != nil
        return Detected(modelType: mt,
                        modality: hasVision ? .vision : .text,
                        hasJANG: hasJANG(json, jangJSON: jangJSON),
                        hasMXTQ: hasMXTQ(json, jangJSON: jangJSON))
    }

    private static func hasJANG(_ json: [String: Any], jangJSON: [String: Any] = [:]) -> Bool {
        !jangJSON.isEmpty || json["jang_config"] != nil || json["jang"] != nil
    }

    private static func hasMXTQ(_ json: [String: Any], jangJSON: [String: Any]) -> Bool {
        if (json["weight_format"] as? String)?.lowercased() == "mxtq" {
            return true
        }
        if (jangJSON["weight_format"] as? String)?.lowercased() == "mxtq" {
            return true
        }
        if let q = json["quantization"] as? [String: Any],
           let method = q["method"] as? String, method.lowercased().contains("mxtq") {
            return true
        }
        if let q = jangJSON["quantization"] as? [String: Any],
           let method = q["method"] as? String, method.lowercased().contains("mxtq") {
            return true
        }
        return false
    }
}
