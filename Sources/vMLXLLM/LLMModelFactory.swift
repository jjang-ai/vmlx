// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon

/// Creates a function that decodes configuration data and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(from:configuration:progressHandler:)``.
public enum LLMTypeRegistry {

    // Split into functions to help the compiler type-check the large model registry
    private static func coreModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "mistral": create(LlamaConfiguration.self, LlamaModel.init),
            "llama": create(LlamaConfiguration.self, LlamaModel.init),
            "phi": create(PhiConfiguration.self, PhiModel.init),
            "phi3": create(Phi3Configuration.self, Phi3Model.init),
            "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
            "gemma": create(GemmaConfiguration.self, GemmaModel.init),
            "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
            "gemma3": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3_text": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3n": create(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
            "gemma4": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            "gemma4_text": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
            "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
            "qwen3": create(Qwen3Configuration.self, Qwen3Model.init),
            "qwen3_moe": create(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
            "qwen3_next": create(Qwen3NextConfiguration.self, Qwen3NextModel.init),
            "qwen3_5": create(Qwen35Configuration.self, Qwen35Model.init),
            "qwen3_5_moe": { data in
                // Sniff weight_format at the top level OR nested under
                // text_config. JANGTQ-quantized checkpoints declare
                // `weight_format: "mxtq"` — route to the JANGTQ model so
                // the routed-expert MoE projections run through
                // TurboQuantSwitchGLU. Affine MoE checkpoints (no
                // weight_format key) fall through to Qwen35MoEModel.
                //
                // Comparison is case-insensitive — `jang_tools` writes
                // lowercase but third-party converters may emit "MXTQ".
                if FormatSniff.isMXTQ(from: data) {
                    let config = try JSONDecoder.json5().decode(
                        Qwen35JANGTQConfiguration.self, from: data)
                    return Qwen35JANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
                return Qwen35MoEModel(config)
            },
            "qwen3_5_text": create(Qwen35TextConfiguration.self, Qwen35TextModel.init),
        ]
    }

    private static func extendedModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "mistral4": create(Mistral4Configuration.self, Mistral4Model.init),
            "minicpm": create(MiniCPMConfiguration.self, MiniCPMModel.init),
            "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
            "cohere": create(CohereConfiguration.self, CohereModel.init),
            "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
            "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
            "deepseek_v3": { data in
                try Self.makeDeepseekV3OrJANGTQ(family: "deepseek_v3", data: data)
            },
            // Audit 2026-04-16 parity: deepseek_v2 + deepseek_v32 + kimi_k25
            // all share the DeepSeek V3 MLA architecture. Python `mlx_lm`
            // has dedicated files for each (deepseek_v2.py, deepseek_v32.py,
            // kimi_k25.py) but the Swift V3 model class handles the same
            // weights. Registering aliases prevents hard load failures
            // on real HF configs that declare any of these model_types.
            //
            // §317 — JANGTQ (mxtq) bundles on any of these families are
            // NOT YET supported natively in Swift (DeepseekV3JANGTQModel
            // port pending — see jang/research/KIMI-K2.6-VMLX-INTEGRATION.md
            // §2.2.5). Detect mxtq via weight_format sniff + throw a
            // clean error pointing at the jang-tools Path A conversion
            // so users aren't left with an affine loader silently
            // mangling the 2-bit codebook weights.
            "deepseek_v2": { data in
                try Self.makeDeepseekV3OrJANGTQ(family: "deepseek_v2", data: data)
            },
            "deepseek_v32": { data in
                try Self.makeDeepseekV3OrJANGTQ(family: "deepseek_v32", data: data)
            },
            // DeepSeek V4 (Flash 284B / Pro 1.6T). §385 registration.
            // Single model class handles both JANG (affine everywhere) and
            // JANGTQ (mxtq routed + 8-bit non-routed) bundles — the loader
            // auto-detects quant bits per module via `_fix_quantized_bits`.
            // Config decoder accepts both variants because
            // DeepseekV4JANGTQConfiguration.jangtqRoutedBits defaults to 2
            // and is only consulted when the bundle actually declares
            // `routed_expert_bits` / `mxtq_seed` keys.
            "deepseek_v4": { data in
                var config = try JSONDecoder.json5().decode(
                    DeepseekV4JANGTQConfiguration.self, from: data)
                // §389 — fold nested HF `quantization.{bits,group_size}`
                // into the JANG fields. JANG_2L bundles ship
                // `quantization: {group_size:32, bits:2}` and would
                // otherwise inherit the default group_size=64 → garbage
                // 2-bit decode.
                config.resolveQuantOverrides()
                return DeepseekV4JANGTQModel(config)
            },
            "kimi_k25": { data in
                try Self.makeDeepseekV3OrJANGTQ(family: "kimi_k25", data: data)
            },
            // GLM-5.1 smoke-test alias (Ralph iter 12, S02 quick-win).
            // GLM-5.1 declares model_type=glm_moe_dsa with an MLA attention
            // block (kv_lora_rank > 0) very close to DeepSeek V3 — the
            // model structure is compatible, but the HF config uses the
            // newer `rope_parameters: {rope_theta, rope_type}` unified
            // field instead of the legacy `rope_theta` scalar expected
            // by DeepseekV3Configuration. Patch the Data before decode
            // so the factory doesn't throw. MoE-specific details (shared
            // expert count, noaux_tc scoring, MTP layers) may diverge
            // subtly from DeepSeek V3's — audit iter-12+ for drift.
            // Full GlmMoeDsa.swift / GlmMoeDsaJANGTQ.swift ports are S03/S04.
            "glm_moe_dsa": { data in
                var dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
                if dict["rope_theta"] == nil,
                   let rp = dict["rope_parameters"] as? [String: Any],
                   let rt = rp["rope_theta"] {
                    dict["rope_theta"] = rt
                }
                let patched = try JSONSerialization.data(withJSONObject: dict)
                // Affine GLM-5.1 path — JANGTQ sniff deferred to S04.
                // When FormatSniff.isMXTQ returns true here a future
                // commit must route to GlmMoeDsaJANGTQModel (not yet
                // implemented); today the JANGTQ bundle still decodes as
                // an affine DeepseekV3Model skeleton which the loader
                // subsequently fails to weight-match — intentional: we
                // want the failure to surface at weight-load time, not
                // at factory dispatch.
                let config = try JSONDecoder.json5().decode(
                    DeepseekV3Configuration.self, from: patched)
                return DeepseekV3Model(config)
            },
            // MiniMax M2.5 shares M2's architecture — reuses same model class.
            "minimax_m2_5": create(MiniMaxConfiguration.self, MiniMaxModel.init),
            "granite": create(GraniteConfiguration.self, GraniteModel.init),
            "granitemoehybrid": create(
                GraniteMoeHybridConfiguration.self, GraniteMoeHybridModel.init),
            "mimo": create(MiMoConfiguration.self, MiMoModel.init),
            "mimo_v2_flash": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
            "minimax": create(MiniMaxConfiguration.self, MiniMaxModel.init),
            "minimax_m2": { data in
                // Peek at weight_format (top-level OR nested text_config,
                // case-insensitive) — "mxtq" routes to the JANGTQ variant.
                // Previously only checked top-level exact-case; VLM
                // wrappers that nest minimax_m2 under `text_config` +
                // uppercase `"MXTQ"` fell through to affine.
                if FormatSniff.isMXTQ(from: data) {
                    let config = try JSONDecoder.json5().decode(
                        MiniMaxJANGTQConfiguration.self, from: data)
                    return MiniMaxJANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data)
                return MiniMaxModel(config)
            },
            "glm4": create(GLM4Configuration.self, GLM4Model.init),
            "glm4_moe": { data in
                // Sniff weight_format (top-level OR nested text_config,
                // case-insensitive). JANGTQ-quantized GLM 5.1 declares
                // `weight_format: "mxtq"` and routes to the JANGTQ model
                // so routed-expert MoE projections run through
                // TurboQuantSwitchGLU. Affine GLM 4 / 5.1 fall through.
                if FormatSniff.isMXTQ(from: data) {
                    let config = try JSONDecoder.json5().decode(
                        GLM4MoEJANGTQConfiguration.self, from: data)
                    return GLM4MoEJANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(GLM4MoEConfiguration.self, from: data)
                return GLM4MoEModel(config)
            },
            "glm4_moe_lite": create(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
            "acereason": create(Qwen2Configuration.self, Qwen2Model.init),
            "falcon_h1": create(FalconH1Configuration.self, FalconH1Model.init),
            "bitnet": create(BitnetConfiguration.self, BitnetModel.init),
            "smollm3": create(SmolLM3Configuration.self, SmolLM3Model.init),
            "ernie4_5": create(Ernie45Configuration.self, Ernie45Model.init),
            // Audit 2026-04-16 registry parity: ERNIE 4.5 MoE shares the
            // same config/model class as the dense variant — aliased to
            // prevent hard-load failure on HF checkpoints that declare
            // `model_type: ernie4_5_moe`.
            "ernie4_5_moe": create(Ernie45Configuration.self, Ernie45Model.init),
            "lfm2": create(LFM2Configuration.self, LFM2Model.init),
        ]
    }

    private static func additionalModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "baichuan_m1": create(BaichuanM1Configuration.self, BaichuanM1Model.init),
            "exaone4": create(Exaone4Configuration.self, Exaone4Model.init),
            "exaone_moe": create(Exaone4Configuration.self, Exaone4Model.init),
            "gpt_oss": create(GPTOSSConfiguration.self, GPTOSSModel.init),
            "lille-130m": create(Lille130mConfiguration.self, Lille130mModel.init),
            "olmoe": create(OlmoEConfiguration.self, OlmoEModel.init),
            "olmo2": create(Olmo2Configuration.self, Olmo2Model.init),
            "olmo3": create(Olmo3Configuration.self, Olmo3Model.init),
            "bailing_moe": create(BailingMoeConfiguration.self, BailingMoeModel.init),
            "lfm2_moe": create(LFM2MoEConfiguration.self, LFM2MoEModel.init),
            "nanochat": create(NanoChatConfiguration.self, NanoChatModel.init),
            "nemotron_h": create(NemotronHConfiguration.self, NemotronHModel.init),
            "afmoe": create(AfMoEConfiguration.self, AfMoEModel.init),
            "jamba": create(JambaConfiguration.self, JambaModel.init),
            "mistral3": { data in
                // Mistral3 VLM may wrap Mistral4 text decoder — check text_config.model_type
                struct TextConfigCheck: Codable {
                    let textConfig: TextModelType?
                    struct TextModelType: Codable {
                        let modelType: String?
                        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
                    }
                    enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
                }
                if let check = try? JSONDecoder.json5().decode(TextConfigCheck.self, from: data),
                    check.textConfig?.modelType == "mistral4"
                {
                    let config = try JSONDecoder.json5().decode(Mistral4Configuration.self, from: data)
                    return Mistral4Model(config)
                }
                let config = try JSONDecoder.json5().decode(Mistral3TextConfiguration.self, from: data)
                return Mistral3TextModel(config)
            },
            "apertus": create(ApertusConfiguration.self, ApertusModel.init),
            "llama4": create(Llama4Configuration.self, Llama4Model.init),
            // Llama 4 text-only variant reuses the same Llama4 class.
            "llama4_text": create(Llama4Configuration.self, Llama4Model.init),
        ]
    }

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(
        creators: coreModels().merging(extendedModels()) { a, _ in a }
            .merging(additionalModels()) { a, _ in a }
    )

    /// §317 — make a DeepseekV3 model OR throw a clean error when the
    /// bundle declares `weight_format: "mxtq"` (JANGTQ). Kimi K2.6
    /// REAP-30-JANGTQ_1L and DSV3.2 JANGTQ checkpoints use this
    /// format; until DeepseekV3JANGTQModel ships natively in Swift,
    /// the affine loader would silently mangle the 2-bit codebook
    /// weights. Refusing with an actionable error is honest.
    ///
    /// Path A conversion from the jang-tools repo:
    ///   python -m jang_tools.convert_jangtq_to_affine \
    ///     --in  /path/Kimi-K2.6-REAP-30-JANGTQ_1L \
    ///     --out /path/Kimi-K2.6-REAP-30-JANG_1L \
    ///     --bits 2
    /// The affine output (~233 GB vs 191 GB JANGTQ) loads natively
    /// with the existing DeepseekV3Model + QuantizedSwitchLinear.
    /// Dispatch for the four DeepSeek V3 MLA family model_types
    /// (`deepseek_v3`, `deepseek_v2`, `deepseek_v32`, `kimi_k25`).
    /// JANGTQ (mxtq weight_format) bundles route to
    /// `DeepseekV3JANGTQModel` — routed-expert projections run through
    /// `TurboQuantSwitchGLU`. Affine bundles fall through to the
    /// existing `DeepseekV3Model` skeleton. §318 — Kimi K2.6 native.
    fileprivate static func makeDeepseekV3OrJANGTQ(
        family: String, data: Data
    ) throws -> any LanguageModel {
        if FormatSniff.isMXTQ(from: data) {
            let config = try JSONDecoder.json5().decode(
                DeepseekV3JANGTQConfiguration.self, from: data)
            return DeepseekV3JANGTQModel(config)
        }
        let config = try JSONDecoder.json5().decode(
            DeepseekV3Configuration.self, from: data)
        return DeepseekV3Model(config)
    }
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The Python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = LLMRegistry(modelConfigurations: all())

    static public let smolLM_135M_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    static public let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    static public let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    static public let deepSeekR1_7B_4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    static public let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    static public let phi3_5_4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let phi3_5MoE = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_9b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_2b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma3_1B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-1b-it-qat-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E4B_it_lm_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E2B_it_lm_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E4B_it_lm_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3n_E2B_it_lm_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        defaultPrompt: "why is the sky blue?"
    )

    static public let qwen2_5_7b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen2_5_1_5b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_0_6b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-0.6B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_1_7b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-1.7B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_4b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-4B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_8b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-8B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3MoE_30b_a3b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-30B-A3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    static public let llama3_1_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_1B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_3B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let deepseek_r1_4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let granite3_3_2b_4bit = ModelConfiguration(
        id: "mlx-community/granite-3.3-2b-instruct-4bit",
        defaultPrompt: ""
    )

    static public let mimo_7b_sft_4bit = ModelConfiguration(
        id: "mlx-community/MiMo-7B-SFT-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let glm4_9b_4bit = ModelConfiguration(
        id: "mlx-community/GLM-4-9B-0414-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .glm4
    )

    static public let acereason_7b_4bit = ModelConfiguration(
        id: "mlx-community/AceReason-Nemotron-7B-4bit",
        defaultPrompt: ""
    )

    static public let bitnet_b1_58_2b_4t_4bit = ModelConfiguration(
        id: "mlx-community/bitnet-b1.58-2B-4T-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let baichuan_m1_14b_instruct_4bit = ModelConfiguration(
        id: "mlx-community/Baichuan-M1-14B-Instruct-4bit-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let smollm3_3b_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM3-3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let ernie_45_0_3BPT_bf16_ft = ModelConfiguration(
        id: "mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let lfm2_1_2b_4bit = ModelConfiguration(
        id: "mlx-community/LFM2-1.2B-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .lfm2
    )

    static public let exaone_4_0_1_2b_4bit = ModelConfiguration(
        id: "mlx-community/exaone-4.0-1.2b-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let lille_130m_bf16 = ModelConfiguration(
        id: "mlx-community/lille-130m-instruct-bf16",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let olmoe_1b_7b_0125_instruct_4bit = ModelConfiguration(
        id: "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let olmo_2_1124_7B_Instruct_4bit = ModelConfiguration(
        id: "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let ling_mini_2_2bit = ModelConfiguration(
        id: "mlx-community/Ling-mini-2.0-2bit-DWQ",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let granite_4_0_h_tiny_4bit_dwq = ModelConfiguration(
        id: "mlx-community/Granite-4.0-H-Tiny-4bit-DWQ",
        defaultPrompt: ""
    )

    static public let lfm2_8b_a1b_3bit_mlx = ModelConfiguration(
        id: "mlx-community/LFM2-8B-A1B-3bit-MLX",
        defaultPrompt: "",
        toolCallFormat: .lfm2
    )

    static public let nanochat_d20_mlx = ModelConfiguration(
        id: "dnakov/nanochat-d20-mlx",
        defaultPrompt: ""
    )

    static public let gpt_oss_20b_MXFP4_Q8 = ModelConfiguration(
        id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let jamba_3b = ModelConfiguration(
        id: "mlx-community/AI21-Jamba-Reasoning-3B-bf16",
        defaultPrompt: ""
    )

    static public let gemma4_27b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-27b-it-4bit",
        defaultPrompt: "Explain quantum computing briefly.",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma4_12b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-12b-it-4bit",
        defaultPrompt: "What is the meaning of life?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma4_27b_it_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-27b-it-qat-4bit",
        defaultPrompt: "Explain quantum computing briefly.",
        extraEOSTokens: ["<end_of_turn>"]
    )

    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
            deepSeekR1_7B_4bit,
            gemma2bQuantized,
            gemma_2_2b_it_4bit,
            gemma_2_9b_it_4bit,
            gemma3_1B_qat_4bit,
            gemma3n_E4B_it_lm_bf16,
            gemma3n_E2B_it_lm_bf16,
            gemma3n_E4B_it_lm_4bit,
            gemma3n_E2B_it_lm_4bit,
            granite3_3_2b_4bit,
            granite_4_0_h_tiny_4bit_dwq,
            llama3_1_8B_4bit,
            llama3_2_1B_4bit,
            llama3_2_3B_4bit,
            llama3_8B_4bit,
            mistral7B4bit,
            mistralNeMo4bit,
            openelm270m4bit,
            phi3_5MoE,
            phi3_5_4bit,
            phi4bit,
            qwen205b4bit,
            qwen2_5_7b,
            qwen2_5_1_5b,
            qwen3_0_6b_4bit,
            qwen3_1_7b_4bit,
            qwen3_4b_4bit,
            qwen3_8b_4bit,
            qwen3MoE_30b_a3b_4bit,
            smolLM_135M_4bit,
            deepseek_r1_4bit,
            mimo_7b_sft_4bit,
            glm4_9b_4bit,
            acereason_7b_4bit,
            bitnet_b1_58_2b_4t_4bit,
            smollm3_3b_4bit,
            ernie_45_0_3BPT_bf16_ft,
            lfm2_1_2b_4bit,
            baichuan_m1_14b_instruct_4bit,
            exaone_4_0_1_2b_4bit,
            lille_130m_bf16,
            olmoe_1b_7b_0125_instruct_4bit,
            olmo_2_1124_7B_Instruct_4bit,
            ling_mini_2_2bit,
            lfm2_8b_a1b_3bit_mlx,
            nanochat_d20_mlx,
            gpt_oss_20b_MXFP4_Q8,
            jamba_3b,
            gemma4_27b_it_4bit,
            gemma4_12b_it_4bit,
            gemma4_27b_it_qat_4bit,
        ]
    }

}

@available(*, deprecated, renamed: "LLMRegistry", message: "Please use LLMRegistry directly.")
public typealias ModelRegistry = LLMRegistry

private struct LLMUserInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)

            return LMInput(tokens: MLXArray(promptTokens))
        } catch TokenizerError.missingChatTemplate {
            print(
                "No chat template was included or provided, so converting messages to simple text format. This is not optimal for model performance, so applications should provide a chat template if none is included with the model."
            )
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama3_8B_4bit)
/// ```
public final class LLMModelFactory: ModelFactory {

    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let modelDirectory = configuration.modelDirectory

        // Load config.json once and decode for both base config and model-specific config
        let configurationURL = modelDirectory.appending(component: "config.json")
        var configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // JANGTQ: merge `weight_format`, `mxtq_bits`, `mxtq_seed` from
        // jang_config.json into config.json so per-type creator closures
        // can dispatch on them (e.g. minimax_m2 → MiniMaxJANGTQModel).
        let jangConfigURL = modelDirectory.appending(component: "jang_config.json")
        if let jangData = try? Data(contentsOf: jangConfigURL),
            var configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
            let jangDict = try? JSONSerialization.jsonObject(with: jangData) as? [String: Any]
        {
            for key in ["weight_format", "mxtq_seed"] {
                if configDict[key] == nil, let v = jangDict[key] {
                    configDict[key] = v
                }
            }
            // mxtq_bits is a dict {attention, routed_expert, ...} — pull the
            // routed_expert bit width out as the scalar the Swift config wants.
            if configDict["mxtq_bits"] == nil,
                let bitsMap = jangDict["mxtq_bits"] as? [String: Any],
                let routed = bitsMap["routed_expert"] as? Int
            {
                configDict["mxtq_bits"] = routed
            }
            if let merged = try? JSONSerialization.data(withJSONObject: configDict) {
                configData = merged
            }
        }

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // Determine effective model type for the LLM factory.
        // VLM configs may wrap a different text decoder (e.g., mistral3 VLM wraps mistral4 text).
        // If text_config.model_type exists and differs from the top-level, prefer it when
        // it's a registered type — it means the text decoder is a different architecture.
        struct TextConfigModelType: Codable {
            let modelType: String?
            enum CodingKeys: String, CodingKey { case modelType = "model_type" }
        }
        struct TextConfigWrapper: Codable {
            let textConfig: TextConfigModelType?
            enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
        }
        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch {
            // Top-level model_type failed (e.g. "mistral3" is a VLM type not in LLM registry,
            // or the config couldn't be decoded for that type).
            // Try text_config.model_type as fallback (e.g. "mistral4" text decoder).
            if let wrapper = try? JSONDecoder.json5().decode(TextConfigWrapper.self, from: configData),
                let textModelType = wrapper.textConfig?.modelType,
                textModelType != baseConfig.modelType
            {
                do {
                    model = try await typeRegistry.createModel(
                        configuration: configData, modelType: textModelType)
                } catch let innerError as DecodingError {
                    throw ModelFactoryError.configurationDecodingError(
                        configurationURL.lastPathComponent, configuration.name, innerError)
                }
            } else if let decodingError = error as? DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, decodingError)
            } else {
                throw error
            }
        }

        // Load EOS token IDs from config.json, with optional override from generation_config.json
        var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        if let generationData = try? Data(contentsOf: generationConfigURL),
            let generationConfig = try? JSONDecoder.json5().decode(
                GenerationConfigFile.self, from: generationData),
            let genEosIds = generationConfig.eosTokenIds?.values
        {
            eosTokenIds = Set(genEosIds)  // Override per Python mlx-lm behavior
        }

        // Build a ModelConfiguration with loaded EOS token IDs and tool call format
        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds
        if mutableConfiguration.toolCallFormat == nil {
            mutableConfiguration.toolCallFormat = ToolCallFormat.infer(from: baseConfig.modelType)
        }

        // Detect JANG model — if jang_config.json exists, load it for per-layer quantization.
        // Standard MLX models skip this entirely (jangConfig stays nil).
        let jangConfig: JangConfig?
        if JangLoader.isJangModel(at: modelDirectory) {
            jangConfig = try JangLoader.loadConfig(at: modelDirectory)
        } else {
            jangConfig = nil
        }

        // Load tokenizer and weights in parallel
        async let tokenizerTask = tokenizerLoader.load(
            from: configuration.tokenizerDirectory)

        // §400 — when both jangConfig AND config.json's perLayerQuantization
        // are present, forward BOTH. Load.swift now merges them: config.json
        // entries (which are explicit and unambiguous) win on the layers they
        // cover, JANG inference fills the rest. The previous behavior (drop
        // perLayerQuantization when jangConfig was present) silently routed
        // bundles whose `jang_config.json` lacks an explicit `quantization`
        // block (e.g. DSV4 Flash JANGTQ ships only `{"weight_format":"bf16"}`)
        // through default-disambiguation in `inferBitWidthAndGroupSize`,
        // picking the wrong (bits, group_size) pair when multiple satisfy the
        // packed-shape equation. Concrete failure: embed=[V,1024]/scales=[V,128]
        // → JANG defaults pick (bits=4, gs=64) → output dim 8192 instead of
        // the bundle's actual (bits=8, gs=32) → 4096. Doubled hidden state
        // collapses fn matmul to a 0-d scalar → mHC sinkhorn crashes.
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization,
            jangConfig: jangConfig)

        let tokenizer = try await tokenizerTask

        let messageGenerator =
            if let model = model as? LLMModel {
                model.messageGenerator(tokenizer: tokenizer)
            } else {
                DefaultMessageGenerator()
            }

        // Build a ModelConfiguration for the ModelContext
        let tokenizerSource: TokenizerSource? =
            configuration.tokenizerDirectory == modelDirectory
            ? nil
            : .directory(configuration.tokenizerDirectory)
        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: tokenizerSource,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: mutableConfiguration.extraEOSTokens,
            eosTokenIds: mutableConfiguration.eosTokenIds,
            toolCallFormat: mutableConfiguration.toolCallFormat)

        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer, configuration: modelConfig,
            messageGenerator: messageGenerator)

        return .init(
            configuration: modelConfig, model: model, processor: processor,
            tokenizer: tokenizer, jangConfig: jangConfig)
    }

}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any vMLXLMCommon.ModelFactory)? {
        LLMModelFactory.shared
    }
}
