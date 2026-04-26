// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import vMLXLMCommon

public enum VLMError: LocalizedError, Equatable {
    case imageRequired
    case maskRequired
    case singleImageAllowed
    case singleVideoAllowed
    case singleMediaTypeAllowed
    case imageProcessingFailure(String)
    case processing(String)
    case noVideoTrackFound
    case videoNotDecodable

    public var errorDescription: String? {
        switch self {
        case .imageRequired:
            return String(localized: "An image is required for this operation.")
        case .maskRequired:
            return String(localized: "An image mask is required for this operation.")
        case .singleImageAllowed:
            return String(localized: "Only a single image is allowed for this operation.")
        case .singleVideoAllowed:
            return String(localized: "Only a single video is allowed for this operation.")
        case .singleMediaTypeAllowed:
            return String(
                localized:
                    "Only a single media type (image or video) is allowed for this operation.")
        case .imageProcessingFailure(let details):
            return String(localized: "Failed to process the image: \(details)")
        case .processing(let details):
            return String(localized: "Processing error: \(details)")
        case .noVideoTrackFound:
            return String(localized: "Video file has no video tracks.")
        case .videoNotDecodable:
            return String(localized: "Video file not decodable.")
        }
    }
}

public struct BaseProcessorConfiguration: Codable, Sendable {
    public let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }
}

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return modelInit(configuration)
    }
}

private func create<C: Codable, P>(
    _ configurationType: C.Type,
    _ processorInit:
        @escaping (
            C,
            any Tokenizer
        ) -> P
) -> (Data, any Tokenizer) throws -> P {
    { data, tokenizer in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        return processorInit(configuration, tokenizer)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(from:configuration:progressHandler:)``.
public enum VLMTypeRegistry {

    /// The set of model type strings supported by the VLM factory.
    /// Use this to check if a model_type from config.json is a known VLM architecture.
    public static let supportedModelTypes: Set<String> = Set(_creators.keys)

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry = .init(creators: _creators)

    nonisolated(unsafe) private static let _creators: [String: (Data) throws -> any LanguageModel] = [
        "paligemma": create(PaliGemmaConfiguration.self, PaliGemma.init),
        "qwen2_vl": create(Qwen2VLConfiguration.self, Qwen2VL.init),
        "qwen2_5_vl": create(Qwen25VLConfiguration.self, Qwen25VL.init),
        "qwen3_vl": create(Qwen3VLConfiguration.self, Qwen3VL.init),
        "qwen3_5": create(Qwen35Configuration.self, Qwen35.init),
        "qwen3_5_moe": { data in
            // Sniff weight_format (top-level OR nested text_config,
            // case-insensitive). JANGTQ-quantized Qwen3.5 / 3.6 VLMs
            // declare `weight_format: "mxtq"` and route to
            // Qwen35MoEJANGTQ so the routed-expert MoE runs through
            // TurboQuantSwitchGLU. Affine VLMs fall through to the
            // standard Qwen35MoE class.
            //
            // Uses the shared `vMLXLLM.FormatSniff.isMXTQ` so the four
            // factory entries (LLM qwen3_5_moe / minimax_m2 / glm4_moe
            // + this VLM entry) stay in lock-step.
            if FormatSniff.isMXTQ(from: data) {
                let config = try JSONDecoder.json5().decode(
                    Qwen35MoEJANGTQConfiguration.self, from: data)
                return Qwen35MoEJANGTQ(config)
            }
            let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data)
            return Qwen35MoE(config)
        },
        "idefics3": create(Idefics3Configuration.self, Idefics3.init),
        "gemma3": create(Gemma3Configuration.self, Gemma3.init),
        "smolvlm": create(SmolVLM2Configuration.self, SmolVLM2.init),
        // TODO: see if we can make it work with fastvlm rather than llava_qwen2
        "fastvlm": create(FastVLMConfiguration.self, FastVLM.init),
        "llava_qwen2": create(FastVLMConfiguration.self, FastVLM.init),
        "pixtral": create(PixtralConfiguration.self, PixtralVLM.init),
        "mistral3": { data in
            // Mistral3 VLM may wrap Mistral4 text decoder — check text_config.model_type
            struct TextCheck: Codable {
                let textConfig: TextType?
                struct TextType: Codable {
                    let modelType: String?
                    enum CodingKeys: String, CodingKey { case modelType = "model_type" }
                }
                enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
            }
            if let check = try? JSONDecoder.json5().decode(TextCheck.self, from: data),
                check.textConfig?.modelType == "mistral4"
            {
                let config = try JSONDecoder.json5().decode(Mistral4VLMConfiguration.self, from: data)
                return Mistral4VLM(config)
            }
            let config = try JSONDecoder.json5().decode(Mistral3VLMConfiguration.self, from: data)
            return Mistral3VLM(config)
        },
        "lfm2_vl": create(LFM2VLConfiguration.self, LFM2VL.init),
        "lfm2-vl": create(LFM2VLConfiguration.self, LFM2VL.init),
        "glm_ocr": create(GlmOcrConfiguration.self, GlmOcr.init),
        "gemma4": create(Gemma4Configuration.self, Gemma4.init),
    ]
}

public enum VLMProcessorTypeRegistry {

    /// Shared instance with default processor types.
    public static let shared: ProcessorTypeRegistry = .init(creators: [
        "PaliGemmaProcessor": create(
            PaliGemmaProcessorConfiguration.self, PaliGemmaProcessor.init),
        "Qwen2VLProcessor": create(
            Qwen2VLProcessorConfiguration.self, Qwen2VLProcessor.init),
        "Qwen2_5_VLProcessor": create(
            Qwen25VLProcessorConfiguration.self, Qwen25VLProcessor.init),
        "Qwen3VLProcessor": create(
            Qwen3VLProcessorConfiguration.self, Qwen3VLProcessor.init),
        "Idefics3Processor": create(
            Idefics3ProcessorConfiguration.self, Idefics3Processor.init),
        "Gemma3Processor": create(
            Gemma3ProcessorConfiguration.self, Gemma3Processor.init),
        "SmolVLMProcessor": create(
            SmolVLMProcessorConfiguration.self, SmolVLMProcessor.init),
        "FastVLMProcessor": create(
            FastVLMProcessorConfiguration.self, FastVLMProcessor.init),
        "PixtralProcessor": create(
            PixtralProcessorConfiguration.self, PixtralProcessor.init),
        "Mistral3Processor": create(
            Mistral3VLMProcessorConfiguration.self, Mistral3VLMProcessor.init),
        "Lfm2VlProcessor": create(
            LFM2VLProcessorConfiguration.self, LFM2VLProcessor.init),
        "Glm46VProcessor": create(
            GlmOcrProcessorConfiguration.self, GlmOcrProcessor.init),
        "Gemma4Processor": create(
            Gemma4ProcessorConfiguration.self, Gemma4Processor.init),
    ])
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class VLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared: VLMRegistry = .init(modelConfigurations: all())

    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2VL2BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen2_5VL3BInstruct4Bit = ModelConfiguration(
        id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct4Bit = ModelConfiguration(
        id: "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3VL4BInstruct8Bit = ModelConfiguration(
        id: "mlx-community/Qwen3-VL-4B-Instruct-8bit",
        defaultPrompt: "Write a haiku about Swift programming"
    )

    static public let smolvlminstruct4bit = ModelConfiguration(
        id: "mlx-community/SmolVLM-Instruct-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let lfm2_5_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2.5-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let lfm2_vl_1_6B_4bit = ModelConfiguration(
        id: "mlx-community/LFM2-VL-1.6B-4bit",
        defaultPrompt: ""
    )

    static public let mistral3_3B_Instruct_4bit = ModelConfiguration(
        id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        defaultPrompt: ""
    )

    static public let gemma3_4B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-4b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_12B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-12b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let gemma3_27B_qat_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-27b-it-qat-4bit",
        defaultPrompt: "Describe the image in English",
        extraEOSTokens: ["<end_of_turn>"]
    )

    static public let smolvlm = ModelConfiguration(
        id: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
        defaultPrompt:
            "What is the main action or notable event happening in this segment? Describe it in one brief sentence."
    )

    static public let fastvlm = ModelConfiguration(
        id: "mlx-community/FastVLM-0.5B-bf16",
        defaultPrompt: "Describe this image in detail."
    )

    static public let qwen3_5_27B_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.5-27B-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public let qwen3_5_35B_A3B_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.5-35B-A3B-4bit",
        defaultPrompt: "Describe the image in English"
    )

    static public func all() -> [ModelConfiguration] {
        [
            paligemma3bMix448_8bit,
            qwen2VL2BInstruct4Bit,
            qwen2_5VL3BInstruct4Bit,
            qwen3VL4BInstruct4Bit,
            qwen3VL4BInstruct8Bit,
            smolvlminstruct4bit,
            gemma3_4B_qat_4bit,
            gemma3_12B_qat_4bit,
            gemma3_27B_qat_4bit,
            smolvlm,
            fastvlm,
        ]
    }

}

@available(*, deprecated, renamed: "VLMRegistry", message: "Please use VLMRegistry directly.")
public typealias ModelRegistry = VLMRegistry

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await VLMModelFactory.shared.loadContainer(
///     configuration: VLMRegistry.paligemma3bMix4488bit)
/// ```
public final class VLMModelFactory: ModelFactory {

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = VLMModelFactory(
        typeRegistry: VLMTypeRegistry.shared, processorRegistry: VLMProcessorTypeRegistry.shared,
        modelRegistry: VLMRegistry.shared)

    /// registry of model type, e.g. configuration value `paligemma` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of input processor type, e.g. configuration value `PaliGemmaProcessor` -> configuration and init methods
    public let processorRegistry: ProcessorTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/paligemma-3b-mix-448-8bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> sending ModelContext {
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
        // jang_config.json into config.json so the VLM factory sniff
        // (FormatSniff.isMXTQ) sees the declaration even on bundles
        // that keep the JANGTQ flag OUT of config.json. Mirrors the
        // identical merge in LLMModelFactory._load — kept symmetric so
        // a VLM bundle with `weight_format` ONLY in jang_config.json
        // (e.g. MiniMax-M2.7-JANGTQ-CRACK-style layouts when a VLM
        // variant ships) routes to Qwen35MoEJANGTQ instead of silently
        // falling through to the affine Qwen35MoE class.
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
            // iter-ralph §238 DIAG: log what landed in the merged config
            // so we can confirm JANGTQ VLM bundles route to the TurboQuant
            // class instead of silently falling through to affine.
            if ProcessInfo.processInfo.environment["VMLX_JANG_TRACE"] == "1" {
                let wf = (configDict["weight_format"] as? String) ?? "(none)"
                let bits = (configDict["mxtq_bits"] as? Int) ?? -1
                let seed = (configDict["mxtq_seed"] as? Int) ?? -1
                FileHandle.standardError.write(Data(
                    "[jang-merge VLM] weight_format=\(wf) mxtq_bits=\(bits) mxtq_seed=\(seed)\n".utf8))
            }
        } else if ProcessInfo.processInfo.environment["VMLX_JANG_TRACE"] == "1" {
            FileHandle.standardError.write(Data(
                "[jang-merge VLM] skipped — no jang_config.json or JSON parse failed\n".utf8))
        }

        // §424 (2026-04-25) — VLM-side shape-authoritative routed-bits override.
        // Mirrors the §421 chain in `LLMModelFactory._load`. Qwen3.6 / Holo3
        // bundles have `model_type: qwen3_5_moe` with a `vision_config`, so
        // they route through the VLM factory instead of the LLM factory.
        // Without this block, broken bundles (no `mxtq_bits`,
        // `routed_expert_bits` absent at every layer) silently picked the
        // wrong codebook → "2+2 2+2" loop. JangLoader.peekRoutedBitsFromSafetensors
        // reads the `tq_packed` tensor headers and infers bits from
        // packed_cols + candidate in_features. injectRoutedBits propagates
        // the resolved value into top-level mxtq_bits + routed_expert_bits +
        // text_config.mxtq_bits so any downstream decoder finds it.
        if let inferredBits = JangLoader.peekRoutedBitsFromSafetensors(
            modelDirectory: modelDirectory, configData: configData
        ) {
            let currentBits: Int? = {
                guard let dict = (try? JSONSerialization.jsonObject(with: configData))
                        as? [String: Any]
                else { return nil }
                if let i = dict["mxtq_bits"] as? Int { return i }
                if let m = dict["mxtq_bits"] as? [String: Any],
                   let r = (m["routed_expert"] ?? m["shared_expert"]) as? Int { return r }
                if let tc = dict["text_config"] as? [String: Any],
                   let i = tc["mxtq_bits"] as? Int { return i }
                return nil
            }()
            // §425 — always inject (idempotent). injectRoutedBits skips
            // top-level fields that are already correct, but propagates
            // into text_config.mxtq_bits + routed_expert_bits which may
            // still be nil even when top-level matches.
            let line: String
            if let c = currentBits, c == inferredBits {
                line = "[§424] mxtq_bits already correct at top (=\(c)); ensuring text_config + routed_expert_bits also set\n"
            } else if let c = currentBits {
                line = "[§424] mxtq_bits override: declared \(c) → shape-authoritative \(inferredBits)\n"
            } else {
                line = "[§424] mxtq_bits inferred from safetensors shape: \(inferredBits) (config field absent)\n"
            }
            FileHandle.standardError.write(Data(line.utf8))
            configData = JangLoader.injectRoutedBits(into: configData, bits: inferredBits)
        }

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
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

        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds

        // Auto-detect tool call format from model type if not explicitly set
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

        // Load tokenizer from model directory (or alternate tokenizer repo),
        // processor config, and weights in parallel using async let.
        // Note: loadProcessorConfig does synchronous I/O but is marked async to enable
        // parallel scheduling. This may briefly block a cooperative thread pool thread,
        // but the config file is small and model loading is not a high-concurrency path.
        async let tokenizerTask = tokenizerLoader.load(
            from: configuration.tokenizerDirectory)
        async let processorConfigTask = loadProcessorConfig(from: modelDirectory)

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: jangConfig != nil ? nil : baseConfig.perLayerQuantization,
            jangConfig: jangConfig)

        let tokenizer = try await tokenizerTask
        let processorConfigData: Data
        let baseProcessorConfig: BaseProcessorConfiguration
        do {
            (processorConfigData, baseProcessorConfig) = try await processorConfigTask
        } catch let error as ProcessorConfigError {
            if let decodingError = error.underlying as? DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    error.filename, configuration.name, decodingError)
            }
            throw ModelFactoryError.configurationFileError(
                error.filename, configuration.name, error.underlying)
        }

        // Override processor type based on model type for models that need special handling
        // Mistral3 models ship with "PixtralProcessor" in their config but need Mistral3Processor
        // to handle spatial merging correctly
        let processorTypeOverrides: [String: String] = [
            "mistral3": "Mistral3Processor"
        ]
        let processorType =
            processorTypeOverrides[baseConfig.modelType] ?? baseProcessorConfig.processorClass

        let processor = try await processorRegistry.createModel(
            configuration: processorConfigData,
            processorType: processorType, tokenizer: tokenizer)

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

        return .init(
            configuration: modelConfig, model: model, processor: processor,
            tokenizer: tokenizer, jangConfig: jangConfig)
    }

}

/// Error wrapper that includes the filename for better error messages.
private struct ProcessorConfigError: Error {
    let filename: String
    let underlying: Error
}

/// Loads processor configuration, preferring preprocessor_config.json over processor_config.json.
/// Marked async to enable parallel scheduling via async let, though the underlying I/O is synchronous.
/// Throws ProcessorConfigError wrapping any underlying error with the filename.
private func loadProcessorConfig(from modelDirectory: URL) async throws -> (
    Data, BaseProcessorConfiguration
) {
    let processorConfigURL = modelDirectory.appending(component: "processor_config.json")
    let preprocessorConfigURL = modelDirectory.appending(component: "preprocessor_config.json")
    let url =
        FileManager.default.fileExists(atPath: preprocessorConfigURL.path)
        ? preprocessorConfigURL
        : processorConfigURL
    do {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder.json5().decode(BaseProcessorConfiguration.self, from: data)
        return (data, config)
    } catch {
        throw ProcessorConfigError(filename: url.lastPathComponent, underlying: error)
    }
}

public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    public static func modelFactory() -> (any vMLXLMCommon.ModelFactory)? {
        VLMModelFactory.shared
    }
}
