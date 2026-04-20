// Pure, testable helper that computes the `capabilities` array Ollama
// clients (GitHub Copilot, Open WebUI, llama-cpp GUI shells) read out of
// `/api/show` to decide which models belong in their model picker /
// tools picker / vision picker.
//
// Extracted from `Sources/vMLXServer/Routes/OllamaRoutes.swift` on
// 2026-04-18 so the family-set classification can be exercised by
// `swift test` without a live HTTP server. Mirrors Python
// `vmlx_engine/api/ollama_adapter.py::model_capabilities`.
//
// The capability set is *additive* — every model has `completion`; a
// VL model adds `vision`; an embedding model adds `embeddings`; a model
// whose family has a registered tool parser adds `tools`; a thinking
// family adds `thinking`. Image + rerank modalities stay bare.

import Foundation

public enum OllamaCapabilities {

    /// Families with a registered tool-call parser. Kept in sync with the
    /// ToolCallParser registry in vMLXEngine/Parsers/ToolCallParser.swift.
    /// We match by family-name *prefix* (case-insensitive) so `qwen3_5_moe`
    /// and `qwen3_6_vl` both get `tools` via the `qwen` entry.
    public static let toolFamilies: [String] = [
        "qwen", "llama", "mistral", "gemma", "deepseek",
        "granite", "hermes", "glm", "minimax", "nemotron",
        "functionary", "xlam", "kimi",
    ]

    /// Families where the assistant emits `<think>…</think>` or an
    /// equivalent reasoning segment that clients may want to fold out of
    /// the main bubble. Kept separately from `toolFamilies` because not
    /// every tool-capable model thinks (e.g. Llama 3 has tools but no
    /// reasoning) and vice-versa.
    public static let thinkingFamilies: [String] = [
        "qwen3", "qwen3_5", "qwen3_6", "minimax", "step",
        "nemotron_h", "nemotron", "gemma4", "deepseek",
        "mistral4", "glm47", "glm-5",
    ]

    /// Compute the capability array for a single model entry. The result
    /// is ordered so `completion` is always first (some Ollama clients
    /// sort lexically), followed by modality tags, then tool / thinking.
    public static func capabilities(
        family: String,
        modality: ModelLibrary.Modality
    ) -> [String] {
        var caps: [String] = ["completion"]

        switch modality {
        case .vision: caps.append("vision")
        case .embedding: caps.append("embeddings")
        case .image, .rerank, .text, .unknown: break
        }

        let fam = family.lowercased()
        if toolFamilies.contains(where: { fam.hasPrefix($0) }) {
            caps.append("tools")
        }
        if thinkingFamilies.contains(where: { fam.hasPrefix($0) }) {
            caps.append("thinking")
        }
        return caps
    }
}
