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
            "qwen3_5_moe": create(Qwen35Configuration.self, Qwen35MoEModel.init),
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
            "deepseek_v3": create(DeepseekV3Configuration.self, DeepseekV3Model.init),
            "granite": create(GraniteConfiguration.self, GraniteModel.init),
            "granitemoehybrid": create(
                GraniteMoeHybridConfiguration.self, GraniteMoeHybridModel.init),
            "mimo": create(MiMoConfiguration.self, MiMoModel.init),
            "mimo_v2_flash": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
            "minimax": create(MiniMaxConfiguration.self, MiniMaxModel.init),
            "minimax_m2": { data in
                // Peek at weight_format — "mxtq" routes to the JANGTQ variant.
                struct FormatCheck: Codable {
                    let weightFormat: String?
                    enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
                }
                if let check = try? JSONDecoder.json5().decode(FormatCheck.self, from: data),
                    check.weightFormat == "mxtq"
                {
                    let config = try JSONDecoder.json5().decode(
                        MiniMaxJANGTQConfiguration.self, from: data)
                    return MiniMaxJANGTQModel(config)
                }
                let config = try JSONDecoder.json5().decode(MiniMaxConfiguration.self, from: data)
                return MiniMaxModel(config)
            },
            "glm4": create(GLM4Configuration.self, GLM4Model.init),
            "glm4_moe": create(GLM4MoEConfiguration.self, GLM4MoEModel.init),
            "glm4_moe_lite": create(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
            "acereason": create(Qwen2Configuration.self, Qwen2Model.init),
            "falcon_h1": create(FalconH1Configuration.self, FalconH1Model.init),
            "bitnet": create(BitnetConfiguration.self, BitnetModel.init),
            "smollm3": create(SmolLM3Configuration.self, SmolLM3Model.init),
            "ernie4_5": create(Ernie45Configuration.self, Ernie45Model.init),
            "lfm2": create(LFM2Configuration.self, LFM2Model.init),
        ]
    }

    private static func additionalModels() -> [String: (Data) throws -> any LanguageModel] {
        [
            "baichuan_m1": create(BaichuanM1Configuration.self, BaichuanM1Model.init),
            "exaone4": create(Exaone4Configuration.self, Exaone4Model.init),
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
            "jamba_3b": create(JambaConfiguration.self, JambaModel.init),
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
        ]
    }

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(
        creators: coreModels().merging(extendedModels()) { a, _ in a }
            .merging(additionalModels()) { a, _ in a }
    )
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

        // When JANG, skip config.json's perLayerQuantization — JANG infers correct
        // per-layer bits from tensor shapes. This avoids creating QuantizedLinear at
        // the wrong bit width (which can't be re-quantized later).
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: jangConfig != nil ? nil : baseConfig.perLayerQuantization,
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
