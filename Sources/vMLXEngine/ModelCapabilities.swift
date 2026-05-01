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
        case templateSniff     // bronze+ — unknown model_type, but chat_template
                               // carried a family signature (e.g. `[TOOL_CALLS]`,
                               // `<|channel|>analysis`, `<|tool_call>`). Beats
                               // plain heuristic guessing — less risky than
                               // bronze because the evidence came from the
                               // model's own template string.
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
        .init(family: "qwen3_5_moe", modelTypes: ["qwen3_5_moe", "qwen3_5_moe_text"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 4),
        // Laguna (poolside) — 33B/3B agentic-coding MoE. 40 hybrid SWA+full
        // layers with per-layer head count, 256 routed experts top-8 + 1
        // shared, dual RoPE (full=YaRN / SWA=default). Text-only. Qwen2-
        // flavored tokenizer (vocab 100352, eos `<|im_end|>`), so the
        // qwen tool parser + qwen3 reasoning parser are the right fallback.
        // The actual loader lives in jang_tools.laguna.runtime — we
        // currently route the load through the Python loader; the Swift
        // factory entry below is for capability + cache-type detection.
        .init(family: "laguna", modelTypes: ["laguna"],
              toolParser: "qwen", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 10),
        // `qwen3_5_moe_text` is the inner `text_config.model_type` name
        // Qwen 3.6 VLM wrappers use. `resolveModelType` prefers
        // text_config over top-level, so without this alias a Qwen 3.6
        // VLM's text path would fall through to bronze and lose the
        // qwen/qwen3 parser assignment. Audit 2026-04-16 per JANG
        // capability stamp rollout item #5.
        .init(family: "qwen3_5_text", modelTypes: ["qwen3_5_text", "qwen3_5_text_text"],
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
        // Mistral-Medium-3.5-128B — outer wrapper is `mistral3` (registered
        // above with PIXTRAL vision); inner text decoder type is
        // `ministral3` (dense GQA 96/8 with 88 layers, hidden 12288, 256K
        // YaRN). Distinct from legacy mistral / mistral4. Loaded via
        // `load_mistral3` Python loader; Swift port pending.
        .init(family: "mistral3", modelTypes: ["ministral3"],
              toolParser: "mistral", thinkInTemplate: false,
              priority: 10),

        // ── DeepSeek ──
        .init(family: "deepseek_vl",
              modelTypes: ["deepseek_vl", "deepseek_vl2", "deepseek_vl_v2"],
              toolParser: "deepseek", isMLLM: true, priority: 5),
        .init(family: "deepseek",
              modelTypes: ["deepseek_v2", "deepseek_v3", "deepseek2", "deepseek"],
              cacheType: "mla",
              toolParser: "deepseek", reasoningParser: "deepseek_r1",
              priority: 20),
        // DeepSeek V4 (Flash 284B / Pro 1.6T) — mHC + hybrid CSA/HCA
        // attention + sqrtsoftplus routing + 3 hash layers + DSML tool
        // envelope. MLA latent head_dim=512. Full port lives in
        // vMLXLLM/Models/DeepseekV4JANGTQ.swift. Native DSV4 reasoning
        // distinguishes "chat" (prompt ends `</think>`, empty reasoning)
        // from "thinking" (prompt ends `<think>`, model fills) with an
        // optional `reasoning_effort ∈ {"high","max"}` hint. Our stream
        // wiring maps `reasoning_effort=none` → chat mode and anything
        // else → thinking mode via the existing enable_thinking rail.
        // Tool parser is `dsml` (｜DSML｜-framed XML-ish blocks).
        .init(family: "deepseek_v4", modelTypes: ["deepseek_v4"],
              cacheType: "mla",
              toolParser: "dsml", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 20),
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
        .init(family: "gemma3n", modelTypes: ["gemma3n"],
              cacheType: "hybrid",
              toolParser: "gemma4", reasoningParser: "gemma4",
              priority: 6),
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

        // ── XLaM (tool-calling LLM family) + Functionary ──
        // F-G11 2026-04-15: silver rows so these don't fall through to
        // the bronze heuristic. XLaM ships its own envelope format;
        // Functionary uses a distinct `<|function_call|>` preamble.
        .init(family: "xlam", modelTypes: ["xlam"],
              toolParser: "xlam", priority: 10),
        .init(family: "functionary", modelTypes: ["functionary"],
              toolParser: "functionary", priority: 10),

        // ── Nemotron ──
        .init(family: "nemotron", modelTypes: ["nemotron"],
              toolParser: "nemotron", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 10),
        .init(family: "nemotron_h", modelTypes: ["nemotron_h"],
              cacheType: "hybrid",
              toolParser: "nemotron", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 10),
        // NemotronH-Omni multimodal (vision + audio + video). Inherits the
        // nemotron_h hybrid cache + reasoning/tool parsers. `isMLLM: true`
        // routes the engine through the VLM dispatch path so image/audio
        // inputs reach the NemotronHOmniProcessor instead of being dropped
        // by the text-only LLM path.
        .init(family: "nemotron_h_omni", modelTypes: ["nemotron_h_omni"],
              cacheType: "hybrid",
              toolParser: "nemotron", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, isMLLM: true, priority: 10),

        // ── Cohere ──
        .init(family: "cohere", modelTypes: ["cohere", "cohere2"],
              priority: 20),

        // ── IBM Granite ──
        .init(family: "granite",
              modelTypes: ["granite", "granite_moe", "granitemoe"],
              toolParser: "granite", priority: 20),
        .init(family: "granite_moe_hybrid",
              modelTypes: ["granite_moe_hybrid", "granitemoehybrid"],
              cacheType: "hybrid",
              toolParser: "granite", priority: 5),

        // ── MiniMax ──
        .init(family: "minimax",
              modelTypes: ["minimax", "minimax_m2", "minimax_m2_5"],
              toolParser: "minimax", reasoningParser: "qwen3",
              thinkInTemplate: true, priority: 20),
        // Kimi K2.6 family — DeepSeek V3 MLA architecture. Chat template
        // unconditionally appends `<think>` to the assistant prefix
        // (KIMI-K2.6-VMLX-INTEGRATION.md §2.6 #16). thinkInTemplate=true
        // keeps the UI showing the thinking toggle + lets clients set
        // `enable_thinking=false` via reasoning_effort=none when they
        // want short-answer output. Same policy as MiniMax / Nemotron.
        .init(family: "kimi_k25",
              modelTypes: ["kimi_k25", "deepseek_v2", "deepseek_v32"],
              cacheType: "mla",
              toolParser: "kimi", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 10),

        // ── Kimi / Moonshot (older K2) ──
        .init(family: "kimi", modelTypes: ["kimi_k2"],
              toolParser: "kimi", reasoningParser: "deepseek_r1",
              thinkInTemplate: true, priority: 20),

        // ── Misc LLMs ──
        // Audit 2026-04-16: silver rows for families registered in
        // LLMTypeRegistry but falling through to bronze heuristic with
        // wrong defaults (no tool parser, maybe wrong cache policy).
        .init(family: "smollm3", modelTypes: ["smollm3"], priority: 15),
        .init(family: "nanochat", modelTypes: ["nanochat"], priority: 15),
        .init(family: "afmoe", modelTypes: ["afmoe"], priority: 15),
        .init(family: "bitnet", modelTypes: ["bitnet"], priority: 15),
        .init(family: "llama4_text",
              modelTypes: ["llama4_text"],
              toolParser: "llama", priority: 5),
        .init(family: "ernie4_5",
              modelTypes: ["ernie4_5", "ernie4_5_moe"],
              priority: 15),
        .init(family: "exaone_moe",
              modelTypes: ["exaone_moe"],
              priority: 15),
        .init(family: "internlm",
              modelTypes: ["internlm", "internlm2", "internlm3"], priority: 20),
        .init(family: "exaone", modelTypes: ["exaone", "exaone3"], priority: 20),
        .init(family: "olmo", modelTypes: ["olmo", "olmo2", "olmo3"], priority: 20),
        // F-G12 2026-04-15: explicit silver rows for MoE variants.
        .init(family: "olmoe", modelTypes: ["olmoe"],
              priority: 10),
        .init(family: "bailing_moe", modelTypes: ["bailing_moe"],
              priority: 10),
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
        .init(family: "baichuan_m1", modelTypes: ["baichuan_m1"],
              cacheType: "hybrid", priority: 5),
        .init(family: "mimo_v2_flash", modelTypes: ["mimo_v2_flash"],
              cacheType: "hybrid", priority: 5),
        .init(family: "lfm2", modelTypes: ["lfm2"],
              cacheType: "hybrid", priority: 5),
        .init(family: "lfm2_moe", modelTypes: ["lfm2_moe"],
              cacheType: "hybrid", priority: 5),
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
