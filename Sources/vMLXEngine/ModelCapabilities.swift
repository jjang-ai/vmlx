// SPDX-License-Identifier: Apache-2.0
// Automatic model capability bundle. Built by `CapabilityDetector` from
// jang_config.json + config.json. Replaces name-heuristic parser selection
// and user-toggled "I know this is a reasoning model" settings.
//
// Detection tiers (see `CapabilityDetector.detect`):
//   gold   — JANG pipeline stamped `capabilities` object in jang_config.json
//   silver — curated allowlist indexed by HF `model_type`
//   bronze — vanilla HF heuristic (logged as a warning so users can promote)

import Foundation

public struct ModelCapabilities: Sendable, Codable, Equatable, Hashable {

    public enum Modality: String, Sendable, Codable {
        case text
        case vision
        case embedding
        case image
        case rerank
        case unknown
    }

    public enum Source: String, Sendable, Codable {
        case jangStamped       // gold — jang_config.json.capabilities object
        case modelTypeTable    // silver — curated table matched model_type
        case fallback          // bronze — heuristic guess, warn in logs
    }

    public var family: String
    public var modality: Modality
    public var modelType: String
    public var reasoningParser: String?
    public var toolParser: String?
    public var thinkInTemplate: Bool
    public var cacheType: String            // "kv" | "mamba" | "hybrid" | "mla"
    public var supportsTools: Bool
    public var supportsThinking: Bool
    public var quantBits: Int?
    public var isJANG: Bool
    public var isMXTQ: Bool
    public var chatTemplate: String?
    public var detectionSource: Source

    public init(
        family: String,
        modality: Modality,
        modelType: String,
        reasoningParser: String? = nil,
        toolParser: String? = nil,
        thinkInTemplate: Bool = false,
        cacheType: String = "kv",
        supportsTools: Bool = false,
        supportsThinking: Bool = false,
        quantBits: Int? = nil,
        isJANG: Bool = false,
        isMXTQ: Bool = false,
        chatTemplate: String? = nil,
        detectionSource: Source = .fallback
    ) {
        self.family = family
        self.modality = modality
        self.modelType = modelType
        self.reasoningParser = reasoningParser
        self.toolParser = toolParser
        self.thinkInTemplate = thinkInTemplate
        self.cacheType = cacheType
        self.supportsTools = supportsTools
        self.supportsThinking = supportsThinking
        self.quantBits = quantBits
        self.isJANG = isJANG
        self.isMXTQ = isMXTQ
        self.chatTemplate = chatTemplate
        self.detectionSource = detectionSource
    }

    /// Empty bronze-tier sentinel for unknown directories.
    public static let unknown = ModelCapabilities(
        family: "unknown",
        modality: .unknown,
        modelType: "unknown",
        detectionSource: .fallback
    )
}

// MARK: - Curated silver-tier table
//
// Ported from `vmlx_engine/model_configs.py::register_all`. Every entry that
// carries a tool/reasoning parser or non-default cache type is represented.
// Entries with NO parser info (chatglm base, gemma/gemma2, cohere, internlm,
// exaone, olmo, llava, idefics, etc.) are included with nil parsers so the
// silver tier still correctly reports modality + cache type without falling
// through to bronze heuristics.
//
// Priority mirrors the Python value — when multiple rows match a model_type,
// the one with the LOWEST numeric priority wins (matches Python lookup
// semantics). Since our `model_type → [row]` lookup is typically 1:1 this is
// a defensive tiebreak; the only multi-entry collision is glm_moe_dsa etc.
// which we resolve by priority ordering in the lookup function.

public struct ModelTypeTableEntry: Sendable {
    public let family: String
    public let modelTypes: [String]
    public let cacheType: String
    public let toolParser: String?
    public let reasoningParser: String?
    public let thinkInTemplate: Bool
    public let isMLLM: Bool
    public let priority: Int

    public init(
        family: String,
        modelTypes: [String],
        cacheType: String = "kv",
        toolParser: String? = nil,
        reasoningParser: String? = nil,
        thinkInTemplate: Bool = false,
        isMLLM: Bool = false,
        priority: Int = 20
    ) {
        self.family = family
        self.modelTypes = modelTypes
        self.cacheType = cacheType
        self.toolParser = toolParser
        self.reasoningParser = reasoningParser
        self.thinkInTemplate = thinkInTemplate
        self.isMLLM = isMLLM
        self.priority = priority
    }
}

public enum ModelTypeTable {

    /// Full allowlist — see `model_configs.py` for the source of truth.
    public static let entries: [ModelTypeTableEntry] = [
        // ── Qwen ──
        .init(family: "qwen3_5", modelTypes: ["qwen3_5"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 4),
        .init(family: "qwen3_5_moe", modelTypes: ["qwen3_5_moe"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 4),
        .init(family: "qwen3", modelTypes: ["qwen3"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 10),
        .init(family: "qwen3_moe", modelTypes: ["qwen3_moe"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 5),
        .init(family: "qwen3_vl", modelTypes: ["qwen3_vl", "qwen3_vl_moe"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, isMLLM: true, priority: 5),
        // qwen3_next is HYBRID (attention + GatedDelta SSM layers), not pure
        // mamba — matches Python `_HYBRID_MODEL_TYPES` in
        // vmlx_engine/utils/ssm_companion_cache.py. CacheCoordinator must flip
        // setHybrid(true) so the SSMStateCache companion path activates.
        .init(family: "qwen3_next", modelTypes: ["qwen3_next"],
              cacheType: "hybrid",
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 1),
        .init(family: "qwen2", modelTypes: ["qwen2", "qwen2_moe", "qwen"],
              toolParser: "qwen", priority: 20),
        .init(family: "qwen2_vl", modelTypes: ["qwen2_vl", "qwen2_5_vl"],
              toolParser: "qwen", isMLLM: true, priority: 10),
        .init(family: "qwen_mamba", modelTypes: ["qwen_mamba"],
              cacheType: "mamba", toolParser: "qwen", priority: 5),

        // ── Llama ──
        .init(family: "llama4", modelTypes: ["llama4"],
              toolParser: "llama", priority: 5),
        .init(family: "llama", modelTypes: ["llama"],
              toolParser: "llama", priority: 20),

        // ── Mistral ──
        .init(family: "devstral", modelTypes: ["devstral"],
              toolParser: "mistral", priority: 5),
        .init(family: "codestral", modelTypes: ["codestral"],
              toolParser: "mistral", priority: 5),
        .init(family: "pixtral", modelTypes: ["pixtral"],
              toolParser: "mistral", isMLLM: true, priority: 5),
        .init(family: "mistral", modelTypes: ["mistral", "mixtral"],
              toolParser: "mistral", priority: 20),
        // Mistral 4 — MLA attention; cacheType="mla"
        .init(family: "mistral4", modelTypes: ["mistral4"],
              cacheType: "mla",
              toolParser: "mistral", reasoningParser: "mistral",
              thinkInTemplate: false, priority: 30),
        .init(family: "mistral3", modelTypes: ["mistral3"],
              toolParser: "mistral", thinkInTemplate: false,
              isMLLM: true, priority: 10),

        // ── DeepSeek ──
        .init(family: "deepseek_vl",
              modelTypes: ["deepseek_vl", "deepseek_vl2", "deepseek_vl_v2"],
              toolParser: "deepseek", isMLLM: true, priority: 5),
        .init(family: "deepseek",
              modelTypes: ["deepseek_v2", "deepseek_v3", "deepseek2", "deepseek"],
              cacheType: "mla",
              toolParser: "deepseek", reasoningParser: "deepseek_r1",
              priority: 20),
        .init(family: "glm5", modelTypes: ["glm_moe_dsa"],
              cacheType: "mla",
              toolParser: "deepseek", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 20),

        // ── GLM / GPT-OSS (Harmony channel) ──
        .init(family: "gpt_oss", modelTypes: ["gpt_oss"],
              toolParser: "glm47", reasoningParser: "openai_gptoss",
              priority: 3),
        .init(family: "glm4_moe", modelTypes: ["glm4_moe", "glm4_moe_lite"],
              toolParser: "glm47", reasoningParser: "openai_gptoss",
              priority: 3),
        // glm_z1 has no unique model_type (shares glm4) — skipped from table;
        // users needing it must stamp JANG or fall back to chatglm entry.
        .init(family: "chatglm", modelTypes: ["chatglm", "glm4", "glm"],
              toolParser: "glm47", priority: 20),

        // ── StepFun ──
        .init(family: "step_vl", modelTypes: ["step1v"],
              toolParser: "step3p5", reasoningParser: "qwen3",
              thinkInTemplate: true, isMLLM: true, priority: 5),
        .init(family: "step", modelTypes: ["step3p5", "step"],
              toolParser: "step3p5", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 10),

        // ── Gemma ──
        .init(family: "gemma4", modelTypes: ["gemma4"],
              toolParser: "gemma4", reasoningParser: "gemma4",
              isMLLM: true, priority: 5),
        .init(family: "gemma4_text", modelTypes: ["gemma4_text"],
              toolParser: "gemma4", reasoningParser: "gemma4",
              priority: 4),
        // Gemma 3 shares Gemma 4's tool format (F-G4 2026-04-15):
        // the prior "hermes" stamp silently dropped tool calls because
        // Gemma 3 emits the same `<tool_call>...` envelope gemma4 parses.
        .init(family: "gemma3", modelTypes: ["gemma3"],
              toolParser: "gemma4", reasoningParser: "deepseek_r1",
              isMLLM: true, priority: 10),
        .init(family: "gemma3_text", modelTypes: ["gemma3_text"],
              toolParser: "gemma4", reasoningParser: "deepseek_r1",
              priority: 8),
        .init(family: "gemma", modelTypes: ["gemma", "gemma2"],
              priority: 30),
        .init(family: "paligemma", modelTypes: ["paligemma", "paligemma2"],
              isMLLM: true, priority: 15),

        // ── Phi ──
        .init(family: "phi4_reasoning", modelTypes: ["phi4_reasoning"],
              toolParser: "hermes", reasoningParser: "deepseek_r1",
              priority: 2),
        .init(family: "phi4_multimodal", modelTypes: ["phi4mm"],
              isMLLM: true, priority: 2),
        .init(family: "phi4", modelTypes: ["phi4", "phi4flash"],
              toolParser: "hermes", priority: 10),
        .init(family: "phi3_v", modelTypes: ["phi3v"],
              toolParser: "llama", isMLLM: true, priority: 8),
        .init(family: "phi3", modelTypes: ["phi3", "phi3small", "phi"],
              toolParser: "llama", priority: 20),

        // ── Hermes ──
        .init(family: "hermes", modelTypes: ["hermes"],
              toolParser: "hermes", priority: 30),

        // ── Nemotron ──
        .init(family: "nemotron", modelTypes: ["nemotron"],
              toolParser: "nemotron", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 10),
        .init(family: "nemotron_h", modelTypes: ["nemotron_h"],
              cacheType: "hybrid",
              toolParser: "nemotron", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 10),

        // ── Cohere ──
        .init(family: "cohere", modelTypes: ["cohere", "cohere2"],
              priority: 20),

        // ── IBM Granite ──
        .init(family: "granite", modelTypes: ["granite", "granite_moe"],
              toolParser: "granite", priority: 20),

        // ── MiniMax ──
        .init(family: "minimax",
              modelTypes: ["minimax", "minimax_m2", "minimax_m2_5"],
              toolParser: "minimax", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 20),

        // ── Kimi / Moonshot ──
        .init(family: "kimi", modelTypes: ["kimi_k2"],
              toolParser: "kimi", reasoningParser: "deepseek_r1",
              priority: 20),

        // ── Misc LLMs ──
        .init(family: "internlm",
              modelTypes: ["internlm", "internlm2", "internlm3"], priority: 20),
        .init(family: "exaone", modelTypes: ["exaone", "exaone3"], priority: 20),
        .init(family: "olmo", modelTypes: ["olmo", "olmo2"], priority: 20),
        .init(family: "starcoder", modelTypes: ["starcoder2"], priority: 30),
        .init(family: "stablelm", modelTypes: ["stablelm"], priority: 30),
        .init(family: "baichuan", modelTypes: ["baichuan"], priority: 30),

        // ── VLM / MLLM ──
        .init(family: "llava", modelTypes: ["llava", "llava_next"],
              isMLLM: true, priority: 20),
        .init(family: "idefics", modelTypes: ["idefics2", "idefics3"],
              isMLLM: true, priority: 15),
        .init(family: "cogvlm", modelTypes: ["cogvlm", "cogvlm2"],
              isMLLM: true, priority: 20),
        .init(family: "florence", modelTypes: ["florence2"],
              isMLLM: true, priority: 20),
        .init(family: "got_ocr", modelTypes: ["got_ocr2"],
              isMLLM: true, priority: 15),
        .init(family: "molmo", modelTypes: ["molmo"],
              isMLLM: true, priority: 20),
        .init(family: "minicpm_v", modelTypes: ["minicpmv"],
              isMLLM: true, priority: 20),
        .init(family: "smolvlm", modelTypes: ["smolvlm"],
              isMLLM: true, priority: 20),
        .init(family: "internvl", modelTypes: ["internvl_chat"],
              isMLLM: true, priority: 15),
        .init(family: "internlm_xcomposer", modelTypes: ["internlm_xcomposer2"],
              isMLLM: true, priority: 8),

        // ── SSM / Mamba ──
        .init(family: "falcon_mamba", modelTypes: ["falcon_mamba"],
              cacheType: "mamba", priority: 5),
        .init(family: "mamba",
              modelTypes: ["mamba", "mamba2", "codestral_mamba"],
              cacheType: "mamba", priority: 30),
        .init(family: "rwkv", modelTypes: ["rwkv", "rwkv5", "rwkv6"],
              cacheType: "mamba", priority: 30),

        // ── Hybrid SSM ──
        .init(family: "jamba", modelTypes: ["jamba"],
              cacheType: "hybrid", priority: 10),
    ]

    /// Look up the best matching entry for a given HF `model_type`. Among
    /// all rows whose `modelTypes` contains the lookup key, the one with the
    /// lowest `priority` value wins (mirrors Python `lookup()` ordering).
    public static func lookup(modelType: String) -> ModelTypeTableEntry? {
        let key = modelType.lowercased()
        let matches = entries.filter { $0.modelTypes.contains(where: { $0.lowercased() == key }) }
        return matches.min(by: { $0.priority < $1.priority })
    }
}
