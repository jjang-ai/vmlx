// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
// JANG format support for mlx-swift-lm

import Foundation
import MLX

// MARK: - Config File Names

/// Primary JANG config file name.
public let jangConfigFileName = "jang_config.json"

/// Legacy config file names to search for (fallback only).
public let jangConfigFileNames = [
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
]

// MARK: - JANG Config Structs

/// Quantization settings from jang_config.json `quantization` block.
public struct JangQuantization: Sendable, Equatable {
    public let method: String
    public let profile: String
    public let targetBits: Float
    public let actualBits: Float
    public let blockSize: Int
    public let bitWidthsUsed: [Int]
    public let quantizationScheme: String
    public let quantizationBackend: String

    public init(
        method: String = "jang-importance",
        profile: String = "JANG_2S",
        targetBits: Float = 2.5,
        actualBits: Float = 2.85,
        blockSize: Int = 64,
        bitWidthsUsed: [Int] = [2, 4, 6],
        quantizationScheme: String = "asymmetric",
        quantizationBackend: String = "mx.quantize"
    ) {
        self.method = method
        self.profile = profile
        self.targetBits = targetBits
        self.actualBits = actualBits
        self.blockSize = blockSize
        self.bitWidthsUsed = bitWidthsUsed
        self.quantizationScheme = quantizationScheme
        self.quantizationBackend = quantizationBackend
    }
}

/// Source model info from jang_config.json `source_model` block.
public struct JangSourceModel: Sendable, Equatable {
    public let name: String
    public let dtype: String
    public let parameters: String

    public init(name: String = "", dtype: String = "bfloat16", parameters: String = "0") {
        self.name = name
        self.dtype = dtype
        self.parameters = parameters
    }

    public var parameterCount: Int { Int(parameters) ?? 0 }
}

/// Architecture info from jang_config.json `architecture` block.
public struct JangArchitecture: Sendable, Equatable {
    public let type: String
    public let attention: String
    public let hasVision: Bool
    public let hasSSM: Bool
    public let hasMoE: Bool

    public init(
        type: String = "transformer",
        attention: String = "gqa",
        hasVision: Bool = false,
        hasSSM: Bool = false,
        hasMoE: Bool = false
    ) {
        self.type = type
        self.attention = attention
        self.hasVision = hasVision
        self.hasSSM = hasSSM
        self.hasMoE = hasMoE
    }
}

/// TurboQuant KV cache settings declared by a JANG model.
///
/// Mirrors the `turboquant` block produced by `jang_tools`:
/// ```json
/// "turboquant": {
///   "enabled": true,
///   "default_key_bits": 3,
///   "default_value_bits": 3,
///   "critical_key_bits": 4,
///   "critical_value_bits": 4,
///   "critical_layers": [0, 1, 2, -3, -2, -1],
///   "seed": 42
/// }
/// ```
/// When present, the engine should override its global TurboQuant defaults
/// with these values so the cache matches the bit budget the model was
/// calibrated for. `critical_layers` is advisory — vMLX currently applies a
/// uniform bit width across all compressible layers and uses the default
/// bits from this block (critical layer awareness is a future refinement).
public struct JangTurboQuant: Sendable, Equatable {
    public let enabled: Bool
    public let defaultKeyBits: Int
    public let defaultValueBits: Int
    public let criticalKeyBits: Int
    public let criticalValueBits: Int
    public let criticalLayers: [Int]
    public let seed: Int

    public init(
        enabled: Bool = false,
        defaultKeyBits: Int = 4,
        defaultValueBits: Int = 4,
        criticalKeyBits: Int = 4,
        criticalValueBits: Int = 4,
        criticalLayers: [Int] = [],
        seed: Int = 42
    ) {
        self.enabled = enabled
        self.defaultKeyBits = defaultKeyBits
        self.defaultValueBits = defaultValueBits
        self.criticalKeyBits = criticalKeyBits
        self.criticalValueBits = criticalValueBits
        self.criticalLayers = criticalLayers
        self.seed = seed
    }
}

/// Runtime info from jang_config.json `runtime` block.
public struct JangRuntime: Sendable, Equatable {
    public let totalWeightBytes: Int
    public let totalWeightGB: Float

    public init(totalWeightBytes: Int = 0, totalWeightGB: Float = 0) {
        self.totalWeightBytes = totalWeightBytes
        self.totalWeightGB = totalWeightGB
    }
}

/// Parsed JANG model configuration from jang_config.json.
public struct JangConfig: Sendable {
    public let format: String
    public let formatVersion: String
    public var isV2: Bool { formatVersion.hasPrefix("2") }
    public let quantization: JangQuantization
    public let sourceModel: JangSourceModel
    public let architecture: JangArchitecture
    public let runtime: JangRuntime
    public let turboquant: JangTurboQuant
    /// Top-level MXTQ seed from jang_config.json (or model config.json).
    /// Surfaced here so generic loader paths (runtime sidecar sizing,
    /// pre-allocation, non-MiniMax JANGTQ models that don't round-trip
    /// through their own ModelArgs) can reach it without a second JSON
    /// parse. Model args structs that already decode `mxtq_seed` from
    /// CodingKeys continue to use their own path — this field is a
    /// loader-side fallback.
    public let mxtqSeed: Int?
    /// Top-level MXTQ bits map (`shared_expert` / `routed_expert` / …).
    /// Same reasoning as `mxtqSeed`.
    public let mxtqBits: [String: Int]?
    /// Top-level `weight_format` field. When `"mxtq"` the bundle ships
    /// `.tq_packed` / `.tq_norms` tensors that MUST stay in that form
    /// for the native TurboQuant kernels (P3/P17). The MXTQ→affine
    /// expander in `Load.swift` would corrupt these, so the loader
    /// branches on this to stay out of the way. Matches Python
    /// `jang_tools/load_jangtq.py` which uses the same config field.
    /// Values seen: "mxtq" (JANGTQ native), absent / other (affine JANG).
    public let weightFormat: String?

    public init(
        format: String = "jang",
        formatVersion: String = "2.0",
        quantization: JangQuantization = JangQuantization(),
        sourceModel: JangSourceModel = JangSourceModel(),
        architecture: JangArchitecture = JangArchitecture(),
        runtime: JangRuntime = JangRuntime(),
        turboquant: JangTurboQuant = JangTurboQuant(),
        mxtqSeed: Int? = nil,
        mxtqBits: [String: Int]? = nil,
        weightFormat: String? = nil
    ) {
        self.format = format
        self.formatVersion = formatVersion
        self.quantization = quantization
        self.sourceModel = sourceModel
        self.architecture = architecture
        self.runtime = runtime
        self.weightFormat = weightFormat
        self.turboquant = turboquant
        self.mxtqSeed = mxtqSeed
        self.mxtqBits = mxtqBits
    }
}

// MARK: - JANG Loader

/// JANG model loader — detects, parses config, and infers per-layer quantization.
// DIAG (2026-04-25): mutable global for first-call trace; not thread-safe but
// JangLoader.inferBitWidthAndGroupSize is called serially from one place.
nonisolated(unsafe) var _inferDiagFired = false

public struct JangLoader: Sendable {

    /// Safetensors files that are auxiliary to a JANGTQ bundle and must not
    /// be treated as model weights. In particular, `jangtq_stacked.safetensors`
    /// is a converter/debug artifact that duplicates routed experts and can be
    /// tens of GB; loading or header-counting it beside prestacked main shards
    /// corrupts key selection and can double memory pressure.
    public static func shouldSkipModelSafetensorsFile(_ url: URL) -> Bool {
        shouldSkipModelSafetensorsFile(url.lastPathComponent)
    }

    public static func shouldSkipModelSafetensorsFile(_ name: String) -> Bool {
        name == "jangtq_runtime.safetensors"
            || name == "jangtq_stacked.safetensors"
    }

    /// Parse HuggingFace/MLX safetensors index metadata when present.
    ///
    /// Rebundled JANGTQ directories can contain stale duplicate tensor
    /// entries in older shard headers while `model.safetensors.index.json`
    /// points at the canonical copy. Header sniffers and weight loading must
    /// honor the index or a stale non-index shard can overwrite the intended
    /// tensor or make a clean bundle look mixed/inconsistent.
    public static func safetensorsIndexWeightMap(at modelDirectory: URL) -> [String: String]? {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String],
              !weightMap.isEmpty
        else { return nil }
        return weightMap
    }

    public static func indexedSafetensorsFiles(
        in modelDirectory: URL,
        weightMap: [String: String]
    ) -> [(url: URL, indexFileName: String)] {
        Array(Set(weightMap.values))
            .filter { $0.hasSuffix(".safetensors") && !shouldSkipModelSafetensorsFile($0) }
            .sorted()
            .map { (modelDirectory.appendingPathComponent($0), $0) }
    }

    public static func indexAllowsSafetensorKey(
        _ key: String,
        in indexFileName: String?,
        weightMap: [String: String]?
    ) -> Bool {
        guard let weightMap, let indexFileName else { return true }
        return weightMap[key] == indexFileName
    }

    public static func safetensorsHeaderScanTargets(
        in modelDirectory: URL
    ) -> (targets: [(url: URL, indexFileName: String?)], weightMap: [String: String]?) {
        if let weightMap = safetensorsIndexWeightMap(at: modelDirectory) {
            let targets = indexedSafetensorsFiles(in: modelDirectory, weightMap: weightMap)
                .map { (url: $0.url, indexFileName: Optional($0.indexFileName)) }
            return (targets, weightMap)
        }

        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else { return ([], nil) }

        let targets = files
            .filter {
                $0.pathExtension == "safetensors"
                    && !shouldSkipModelSafetensorsFile($0)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { (url: $0, indexFileName: Optional<String>.none) }
        return (targets, nil)
    }

    /// Check if a model directory contains a JANG model.
    public static func isJangModel(at path: URL) -> Bool {
        findConfigPath(at: path) != nil
    }

    /// Find the JANG config file in a model directory.
    public static func findConfigPath(at modelPath: URL) -> URL? {
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        // .jangspec bundles built before the Plan 6 builder update only place
        // jang_config.json under target/. Fall back to the bundle layout so
        // those still load without rebuilding the bundle.
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent("target")
                .appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        return nil
    }

    /// Load and parse the JANG config from a model directory. Errors
    /// are wrapped with enough context for the user-facing banner to
    /// point at the exact file and cause rather than "Failed to parse
    /// JSON", which left users with no idea which of potentially many
    /// config files was malformed (A3 fix).
    public static func loadConfig(at modelPath: URL) throws -> JangConfig {
        guard let configURL = findConfigPath(at: modelPath) else {
            throw JangLoaderError.configNotFound(modelPath.path)
        }

        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw JangLoaderError.invalidConfig(
                "Cannot read \(configURL.lastPathComponent) at \(configURL.path): \(error.localizedDescription)"
            )
        }

        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data)
        } catch let parseError as NSError {
            // NSError on JSONSerialization carries line/column in userInfo
            // under NSDebugDescription and a readable message in
            // localizedDescription. Surface both so the banner is useful.
            let detail = (parseError.userInfo["NSDebugDescription"] as? String)
                ?? parseError.localizedDescription
            throw JangLoaderError.invalidConfig(
                "\(configURL.lastPathComponent) is not valid JSON: \(detail)"
            )
        }

        guard let json = raw as? [String: Any] else {
            throw JangLoaderError.invalidConfig(
                "\(configURL.lastPathComponent) must be a JSON object at the root"
            )
        }

        return try parseConfig(from: json)
    }

    /// Parse a JangConfig from a raw JSON dictionary.
    public static func parseConfig(from json: [String: Any]) throws -> JangConfig {
        let format = json["format"] as? String ?? "jang"
        let formatVersion = json["format_version"] as? String ?? "2.0"

        let quantization: JangQuantization
        if let qDict = json["quantization"] as? [String: Any] {
            quantization = JangQuantization(
                method: qDict["method"] as? String ?? "jang-importance",
                profile: qDict["profile"] as? String ?? "JANG_2S",
                targetBits: floatValue(qDict["target_bits"]) ?? 2.5,
                actualBits: floatValue(qDict["actual_bits"]) ?? 2.5,
                blockSize: qDict["block_size"] as? Int ?? 64,
                bitWidthsUsed: qDict["bit_widths_used"] as? [Int] ?? [],
                quantizationScheme: qDict["quantization_scheme"] as? String ?? "asymmetric",
                quantizationBackend: qDict["quantization_backend"] as? String ?? "mx.quantize"
            )
        } else {
            quantization = JangQuantization()
        }

        let sourceModel: JangSourceModel
        if let smDict = json["source_model"] as? [String: Any] {
            let params: String
            if let s = smDict["parameters"] as? String {
                params = s
            } else if let n = smDict["parameters"] as? Int {
                params = String(n)
            } else {
                params = "0"
            }
            sourceModel = JangSourceModel(
                name: smDict["name"] as? String ?? "",
                dtype: smDict["dtype"] as? String ?? "bfloat16",
                parameters: params
            )
        } else {
            sourceModel = JangSourceModel()
        }

        let architecture: JangArchitecture
        if let aDict = json["architecture"] as? [String: Any] {
            architecture = JangArchitecture(
                type: aDict["type"] as? String ?? "transformer",
                attention: aDict["attention"] as? String ?? "gqa",
                hasVision: aDict["has_vision"] as? Bool ?? false,
                hasSSM: aDict["has_ssm"] as? Bool ?? false,
                hasMoE: aDict["has_moe"] as? Bool ?? false
            )
        } else {
            architecture = JangArchitecture()
        }

        let runtime: JangRuntime
        if let rDict = json["runtime"] as? [String: Any] {
            runtime = JangRuntime(
                totalWeightBytes: rDict["total_weight_bytes"] as? Int ?? 0,
                totalWeightGB: floatValue(rDict["total_weight_gb"]) ?? 0
            )
        } else {
            runtime = JangRuntime()
        }

        // 2026-05-01 STATUS-OF-FIELD audit: NO JANGTQ bundle currently in
        // the wild emits a `turboquant` block. Every JANGTQ-CRACK / JANGTQ4 /
        // -Small-JANGTQ / -Med-JANGTQ bundle on Eric's drive omits the
        // field — verified across 10+ bundles spanning Kimi K2.6, MiniMax
        // M2.7, Qwen 3.5/3.6, DeepSeek V4, Nemotron-Omni, Laguna, Mistral 4.
        //
        // The auto-activation path that reads this block downstream
        // (Stream.swift:2755 `loadedJangConfig?.turboquant.enabled`) is
        // therefore dormant in practice — TQ KV cache stays OFF for every
        // JANGTQ bundle UNLESS the user explicitly enables `enableTurboQuant`
        // in settings. This is consistent with the 2026-04-16 perf audit
        // ruling that default-on TQ KV taxed MoE/hybrid models 25-40%.
        //
        // The block parser is kept active so a future jang_tools release
        // can opt a model into per-bundle calibrated TQ bits without a
        // Swift-side change. The default `JangTurboQuant()` enabled=false
        // ensures the absence of the block does NOT silently flip TQ on.
        let turboquant: JangTurboQuant
        if let tDict = json["turboquant"] as? [String: Any] {
            turboquant = JangTurboQuant(
                enabled: tDict["enabled"] as? Bool ?? false,
                defaultKeyBits: tDict["default_key_bits"] as? Int ?? 4,
                defaultValueBits: tDict["default_value_bits"] as? Int ?? 4,
                criticalKeyBits: tDict["critical_key_bits"] as? Int ?? 4,
                criticalValueBits: tDict["critical_value_bits"] as? Int ?? 4,
                criticalLayers: tDict["critical_layers"] as? [Int] ?? [],
                seed: tDict["seed"] as? Int ?? 42
            )
        } else {
            // No `turboquant` block — fall through to the canonical default
            // (`enabled=false`). Any bundle that wants TQ must emit the
            // block explicitly. Per the user-toggle path below, the global
            // `enableTurboQuant` setting still controls TQ for all bundles.
            turboquant = JangTurboQuant()
        }

        // Top-level mxtq_seed / mxtq_bits — Python's JANGTQ loader reads
        // these straight off the root of jang_config.json (or the model's
        // own config.json for standalone JANGTQ quants). Accept mxtq_bits
        // as either a flat Int (legacy uniform-bit dumps) or a dict
        // {shared_expert, routed_expert, ...} (modern per-role maps).
        let mxtqSeed = json["mxtq_seed"] as? Int
        let mxtqBits: [String: Int]?
        if let dict = json["mxtq_bits"] as? [String: Int] {
            mxtqBits = dict
        } else if let bits = json["mxtq_bits"] as? Int {
            mxtqBits = ["routed_expert": bits]
        } else {
            mxtqBits = nil
        }

        // Top-level weight_format — when set to "mxtq", the bundle keeps
        // .tq_packed/.tq_norms tensors for native TurboQuant consumption
        // (P3/P17 Metal kernels), and the MXTQ→affine expander must NOT
        // run. This replaces the stale sidecar-file heuristic in Load.swift
        // which only fired when a jangtq_runtime.safetensors was present
        // (that sidecar is optional — Python regenerates signs at runtime
        // from mxtqSeed). Matches `jang_tools.load_jangtq` weightFormat
        // gating. 2026-04-16 fix for MiniMax-M2.7-JANGTQ-CRACK load.
        let weightFormat = json["weight_format"] as? String

        return JangConfig(
            format: format,
            formatVersion: formatVersion,
            quantization: quantization,
            sourceModel: sourceModel,
            architecture: architecture,
            runtime: runtime,
            turboquant: turboquant,
            mxtqSeed: mxtqSeed,
            mxtqBits: mxtqBits,
            weightFormat: weightFormat
        )
    }

    // MARK: - Per-Layer Bit Width Inference

    /// Infer per-layer quantization from loaded JANG weights.
    ///
    /// JANG v2 stores different tensors at different bit widths. The bit width is
    /// inferred from tensor shapes: `actual_bits = (weight.shape[-1] * 32) / (scales.shape[-1] * group_size)`
    ///
    /// Returns a `BaseConfiguration.PerLayerQuantization` that the existing
    /// `loadWeights()` quantization path can use directly.
    public static func inferPerLayerQuantization(
        weights: [String: MLXArray],
        jangConfig: JangConfig,
        hiddenSizeHint: Int? = nil,
        validInDims: Set<Int> = [],
        declaredDefaultQuantization: BaseConfiguration.Quantization? = nil
    ) -> BaseConfiguration.PerLayerQuantization {
        // JANGTQ bundles use two independent bit namespaces:
        //   - `mxtq_bits` / `routed_expert_bits` for tq_packed routed experts.
        //   - `config.json::quantization` for affine dense/router weights.
        //
        // Some modern bundles (Laguna, Nemotron Omni, Qwen 3.6) have a
        // sparse `jang_config.json` with no `quantization` block; falling
        // back to JangQuantization's legacy [2,4,6] default makes the affine
        // inferer compare 8-bit dense weights against "declared 2-bit" and
        // can make MoE gate dequantization skip the real 8-bit candidate.
        // When the caller has already decoded config.json's top-level
        // affine quantization, treat that as the declared default.
        let groupSize = declaredDefaultQuantization?.groupSize
            ?? jangConfig.quantization.blockSize
        var perLayer = [String: BaseConfiguration.QuantizationOption]()

        // Find the default (most common) bit width from jang_config
        let defaultBits = declaredDefaultQuantization?.bits
            ?? (jangConfig.quantization.bitWidthsUsed.min() ?? 4)
        let bitWidthsUsed = Array(Set(
            jangConfig.quantization.bitWidthsUsed + [defaultBits]
        )).sorted()

        // Group weight keys by their base path (strip .weight/.scales/.biases suffix)
        var quantizedLayers = Set<String>()
        for key in weights.keys {
            if key.hasSuffix(".scales") {
                let basePath = String(key.dropLast(".scales".count))
                quantizedLayers.insert(basePath)
            }
        }

        // §410 — shape is authoritative. Always emit a per-layer
        // override, even when the inferred (bits, gs) happens to match
        // the JANG-declared default. Reason: the downstream loader
        // walks `quantization(layer:)` lookups, falling through to
        // `top-level.quantization` only when the layer has no entry.
        // If we omit "matches default" entries, an incorrectly-set
        // top-level default (e.g. bad config declaring bits=2 but
        // storing 8-bit attention on disk) silently routes through
        // the wrong dequant. By writing the inferred value at every
        // quantized layer's path, the loader's lookup is shape-driven
        // regardless of what jang_config or config.json declared.
        var disagreementCount = 0
        var sampleDeclared: (Int, Int)? = nil
        var sampleInferred: (Int, Int)? = nil
        // Deterministic traversal matters for diagnostics. Set iteration
        // order made the sample "(bits, gs)" in the config-metadata warning
        // vary across identical DSV4 loads (2/32, 4/32, 6/32 depending on
        // hash order). Per-layer overrides were still applied, but the log
        // looked like the loader was guessing. Sort by path so repeated runs
        // produce stable evidence.
        for basePath in quantizedLayers.sorted() {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else {
                continue
            }

            // 2026-05-01 PARITY FIX — for embed_tokens, lm_head, and
            // any tensor whose `in_dim` should equal `hidden_size`,
            // pass the architecture hint so the inferer can disambiguate
            // between (8, 32) and (4, 64) shape collisions. Bundles
            // like Gemma-4-26B-A4B JANG_4M-CRACK with hidden=2816 need
            // (4, 64) for embed [262144, 352] / scales [..., 44]; the
            // bits-first default would pick (8, 32) producing
            // in_dim=1408 and crashing the next layernorm.
            let pd = weightArray.shape.last ?? 0
            let ng = scalesArray.shape.last ?? 1
            let (bits, inferredGroupSize): (Int, Int)
            // 2026-05-01 — narrow architecture-aware disambiguation.
            // Apply ONLY to tensor paths whose in_dim is provably
            // hidden_size: embed_tokens, lm_head, and switch_mlp /
            // experts down_proj (which projects intermediate→hidden).
            // For everything else, fall back to bits-first which is
            // correct for the vast majority of bundles. Wider hints
            // regressed DSV4 MLA paths because too many in_dims
            // matched + the wrong pair was picked.
            let isHiddenAnchor =
                basePath.hasSuffix("embed_tokens")
                || basePath.hasSuffix("embed")
                || basePath.hasSuffix("lm_head")
            let isHiddenInputProjection =
                basePath.hasSuffix(".linear_attn.in_proj_qkv")
                || basePath.hasSuffix(".linear_attn.in_proj_z")
                || basePath.hasSuffix(".linear_attn.in_proj_a")
                || basePath.hasSuffix(".linear_attn.in_proj_b")
                || basePath.hasSuffix(".self_attn.q_proj")
                || basePath.hasSuffix(".self_attn.k_proj")
                || basePath.hasSuffix(".self_attn.v_proj")
                || basePath.hasSuffix(".attn.q_proj")
                || basePath.hasSuffix(".attn.k_proj")
                || basePath.hasSuffix(".attn.v_proj")
                || basePath.hasSuffix(".mlp.gate_proj")
                || basePath.hasSuffix(".mlp.up_proj")
            // For switch_mlp.down_proj (and similar MoE down projections),
            // the in_dim is moe_intermediate_size, the out_dim is hidden_size.
            // The OUT shape is `weight.shape[0]` (or shape[-2] for 3D MoE).
            // We can't easily get out_dim → hidden_size relation here
            // without more model context, so leave MoE down_proj to the
            // validInDims path when available.
            let isSwitchMLPDown =
                basePath.hasSuffix("switch_mlp.down_proj")
                || basePath.hasSuffix("switch_glu.down_proj")
                || basePath.contains("switch_mlp.up_proj")
                || basePath.contains("switch_mlp.gate_proj")
                || basePath.contains("switch_glu.up_proj")
                || basePath.contains("switch_glu.gate_proj")

            func inferFromUniqueValidInDim() -> (bits: Int, groupSize: Int)? {
                guard !validInDims.isEmpty else { return nil }
                let preferred: [(Int, Int)] = [
                    (8, 32), (8, 64), (8, 128),
                    (4, 32), (4, 64), (4, 128),
                    (2, 32), (2, 64), (2, 128),
                    (3, 32), (6, 32),
                    (5, 32), (5, 64), (3, 64), (6, 64),
                ]
                var matches: [(bits: Int, groupSize: Int, inDim: Int)] = []
                for (b, gs) in preferred {
                    guard b > 0, (pd * 32) % b == 0 else { continue }
                    let inDim = (pd * 32) / b
                    guard validInDims.contains(inDim), inDim % ng == 0 else { continue }
                    let impliedGs = inDim / ng
                    if impliedGs == gs {
                        matches.append((b, gs, inDim))
                    }
                }
                let uniqueInDims = Set(matches.map(\.inDim))
                guard uniqueInDims.count == 1, let first = matches.first else {
                    return nil
                }
                return (first.bits, first.groupSize)
            }

            if (isHiddenAnchor || isHiddenInputProjection), let hSize = hiddenSizeHint, hSize > 0 {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    packedDim: pd, numGroups: ng,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed,
                    expectedInDim: hSize)
            } else if let picked = inferFromUniqueValidInDim() {
                (bits, inferredGroupSize) = picked
            } else if isSwitchMLPDown && !validInDims.isEmpty {
                // Try each valid in_dim against the bits-first
                // preferred order. The first pair that matches is
                // accepted. If none match, fall through to bits-first.
                let preferred: [(Int, Int)] = [
                    (8, 32), (8, 64), (8, 128),
                    (4, 32), (4, 64), (4, 128),
                    (2, 32), (2, 64), (2, 128),
                ]
                var picked: (Int, Int)? = nil
                for (b, gs) in preferred {
                    guard (pd * 32) % b == 0 else { continue }
                    let inDim = (pd * 32) / b
                    guard validInDims.contains(inDim) else { continue }
                    guard inDim % ng == 0 else { continue }
                    let impliedGs = inDim / ng
                    if impliedGs == gs { picked = (b, gs); break }
                }
                if let p = picked {
                    (bits, inferredGroupSize) = p
                } else {
                    (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                        weight: weightArray, scales: scalesArray,
                        knownGroupSize: groupSize,
                        bitWidthsUsed: bitWidthsUsed)
                }
            } else {
                (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                    weight: weightArray, scales: scalesArray,
                    knownGroupSize: groupSize,
                    bitWidthsUsed: bitWidthsUsed)
            }

            // §410 — track disagreement against the JANG-declared default
            // for the user-facing log line below. We compare against
            // `(defaultBits, groupSize)` because that's what the loader
            // would have used for any layer absent from per-layer
            // overrides under the OLD inferer.
            if bits != defaultBits || inferredGroupSize != groupSize {
                disagreementCount += 1
                if sampleDeclared == nil {
                    sampleDeclared = (defaultBits, groupSize)
                    sampleInferred = (bits, inferredGroupSize)
                }
            }

            perLayer[basePath] = .quantize(
                BaseConfiguration.Quantization(
                    groupSize: inferredGroupSize, bits: bits))
        }

        // Mirror Python's user-facing log line:
        //   "⚠ config-metadata BUG detected, patched in-memory:
        //    top_bits 2 → 8, +312 per-module overrides (312 mismatches)"
        // Visible in vmlxctl serve / chat output so users on legacy
        // bundles see the runtime save when it fires. No spam on
        // clean configs (disagreementCount == 0).
        if disagreementCount > 0,
           let dec = sampleDeclared, let inf = sampleInferred
        {
            let plural = disagreementCount == 1 ? "" : "s"
            let line = (
                "⚠ [JangLoader] config-metadata BUG detected, "
                + "patched in-memory: declared (bits=\(dec.0), gs=\(dec.1)) "
                + "→ shape-inferred (bits=\(inf.0), gs=\(inf.1)), "
                + "\(disagreementCount) per-layer override\(plural) "
                + "applied. Set VMLX_REPAIR_BAD_JANG_CONFIG=1 to also "
                + "patch config.json on disk.\n"
            )
            FileHandle.standardError.write(Data(line.utf8))
        }

        // Layers without .scales are unquantized (norms, biases) — they don't need entries
        // The default quantization covers all layers not in perLayer

        return BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(groupSize: groupSize, bits: defaultBits),
            perLayerQuantization: perLayer
        )
    }

    /// Infer bit width from weight and scales tensor shapes using a fixed group size.
    public static func inferBitWidth(
        weight: MLXArray, scales: MLXArray, groupSize: Int
    ) -> Int {
        inferBitWidthAndGroupSize(weight: weight, scales: scales, knownGroupSize: groupSize).bits
    }

    /// Infer BOTH bit width and group size from weight and scales tensor shapes.
    ///
    /// A JANG quantized tensor has:
    ///   weight.shape[-1] = (in_dim * bits) / 32   (packed into uint32)
    ///   scales.shape[-1] = in_dim / groupSize     (one scale per group per row)
    ///
    /// From these two equations:
    ///   in_dim = scales.shape[-1] * groupSize
    ///   bits   = weight.shape[-1] * 32 / in_dim
    ///
    /// With knownGroupSize this is a direct calculation. Without it, the answer
    /// is not unique from shapes alone — multiple (bits, groupSize) pairs can
    /// produce the same packed shape. In that case we require the provided
    /// `bitWidthsUsed` from the JANG config to disambiguate, preferring
    /// higher bits first (JANG CRITICAL tier uses the highest bits).
    public static func inferBitWidthAndGroupSize(
        weight: MLXArray, scales: MLXArray, knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = []
    ) -> (bits: Int, groupSize: Int) {
        return inferBitWidthAndGroupSize(
            packedDim: weight.shape.last ?? 0,
            numGroups: scales.shape.last ?? 1,
            knownGroupSize: knownGroupSize,
            bitWidthsUsed: bitWidthsUsed)
    }

    /// Pure-Int overload — testable without MLX Metal init. The
    /// MLXArray version above delegates to this. Mirror Python
    /// `vmlx_engine/utils/jang_loader.py:_pre_fix_bits_from_shard`
    /// exactly: group-size candidates first (config block_size → 64
    /// → 128), then derive bits from `(packedDim * 32) / in_dim`.
    public static func inferBitWidthAndGroupSize(
        packedDim: Int, numGroups: Int,
        knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = []
    ) -> (bits: Int, groupSize: Int) {
        // §411 — first-call shape trace, gated behind `VMLX_LOAD_DIAG=1`.
        // Useful for diagnosing path-resolution mismatches between
        // shape inference and the loader's leaf-path lookup; off by
        // default so production loads stay quiet.
        if !_inferDiagFired,
           ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1"
        {
            _inferDiagFired = true
            let msg = "[infer-diag] packedDim=\(packedDim) numGroups=\(numGroups) knownGS=\(String(describing: knownGroupSize)) bitsUsed=\(bitWidthsUsed)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }

        guard packedDim > 0 && numGroups > 0 else { return (4, knownGroupSize ?? 64) }

        // §410 — shape-authoritative bit-width inference.
        //
        // The shape equation `weight_cols * 32 = bits * num_groups * group_size`
        // is genuinely ambiguous when num_groups divides packedDim*32
        // by multiple bit widths. For R = packed/num_groups = 8, the
        // pairs (8, 32), (4, 64), (2, 128) ALL match.
        //
        // Bits-first (preferred order below) is the correct default
        // for MOST bundles: the JANGTQ converter emits CRITICAL-tier
        // attention/embed/lm_head at the highest bits available
        // (typically 8-bit gs=32 for hidden=4096 models like DSV4,
        // MiniMax M2.7, Qwen3.6, Nemotron-H Omni MXFP4 etc.).
        //
        // BUT: for bundles whose hidden_size produces an
        // (bits=8, gs=32) in_dim that is HALF the actual hidden_size
        // — e.g., Gemma-4-26B-A4B with hidden=2816 — the bits-first
        // path picks (8, 32) and mis-classifies the layer. The
        // CORRECT pair is (4, 64) producing in_dim=2816.
        //
        // Disambiguation: when we have an `expectedInDim` hint from
        // the model's config (e.g., hidden_size for embed/o_proj
        // input dim), prefer pairs whose computed in_dim matches.
        // Without the hint, fall back to bits-first preferred order.
        // The `inferPerLayerQuantization` caller passes the hint
        // when known.
        let preferred: [(Int, Int)] = [
            (8, 32), (8, 64), (8, 128),
            (4, 32), (4, 64), (4, 128),
            (2, 32), (2, 64), (2, 128),
            (3, 32), (6, 32),
            (5, 32), (5, 64), (3, 64), (6, 64),
        ]
        for (bits, gs) in preferred {
            guard (packedDim * 32) % bits == 0 else { continue }
            let inDim = (packedDim * 32) / bits
            guard inDim > 0, inDim % numGroups == 0 else { continue }
            let impliedGs = inDim / numGroups
            if impliedGs == gs {
                return (bits, gs)
            }
        }

        // Tail safety
        let validBits = [2, 3, 4, 5, 6, 8]
        let candidates = bitWidthsUsed.isEmpty
            ? validBits.sorted(by: >)
            : bitWidthsUsed.sorted(by: >)
        for bits in candidates {
            guard bits > 0, (packedDim * 32) % bits == 0 else { continue }
            let inDim = (packedDim * 32) / bits
            guard inDim > 0, inDim % numGroups == 0 else { continue }
            let gs = inDim / numGroups
            return (bits, gs)
        }

        return (4, knownGroupSize ?? 64)
    }

    /// Architecture-aware disambiguation overload — when the caller
    /// knows the expected in_dim for this tensor (e.g., hidden_size
    /// for embed_tokens / lm_head / o_proj output), the inferer
    /// prefers `(bits, gs)` pairs whose computed in_dim matches.
    /// Falls back to bits-first preferred order when no pair matches
    /// or `expectedInDim` is nil.
    ///
    /// Resolves the Gemma-4-26B-A4B JANG_4M-CRACK regression where
    /// embed_tokens [262144, 352] / scales [..., 44] satisfies BOTH
    /// (8, 32) → in_dim=1408 AND (4, 64) → in_dim=2816, and the
    /// bits-first default picked the wrong one (1408 → embed output
    /// halved → next layernorm crashed with weight=2816 vs x.last=1408).
    public static func inferBitWidthAndGroupSize(
        packedDim: Int, numGroups: Int,
        knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = [],
        expectedInDim: Int
    ) -> (bits: Int, groupSize: Int) {
        guard packedDim > 0 && numGroups > 0 && expectedInDim > 0 else {
            return inferBitWidthAndGroupSize(
                packedDim: packedDim, numGroups: numGroups,
                knownGroupSize: knownGroupSize, bitWidthsUsed: bitWidthsUsed)
        }
        // Only consider pairs where in_dim == expectedInDim.
        let preferred: [(Int, Int)] = [
            (8, 32), (8, 64), (8, 128),
            (4, 32), (4, 64), (4, 128),
            (2, 32), (2, 64), (2, 128),
            (3, 32), (6, 32),
        ]
        for (bits, gs) in preferred {
            guard (packedDim * 32) % bits == 0 else { continue }
            let inDim = (packedDim * 32) / bits
            guard inDim == expectedInDim else { continue }
            guard inDim % numGroups == 0 else { continue }
            let impliedGs = inDim / numGroups
            if impliedGs == gs {
                return (bits, gs)
            }
        }
        // No pair matches expected in_dim → fall back to bits-first.
        return inferBitWidthAndGroupSize(
            packedDim: packedDim, numGroups: numGroups,
            knownGroupSize: knownGroupSize, bitWidthsUsed: bitWidthsUsed)
    }

    // MARK: - MoE Gate Dequantization

    /// Dequantize MoE gate/router weights from quantized uint32 to float.
    ///
    /// JANG quantizes MoE gate weights at CRITICAL tier (highest available bits)
    /// for routing precision, but the model expects them as plain float Linear
    /// (not QuantizedLinear). This function detects gate weights that have
    /// .scales/.biases companions and dequantizes them in-place.
    ///
    /// Gate patterns matched:
    /// - `.gate.weight` (not `.gate_proj.weight`) — Nemotron, MiniMax
    /// - `.mlp.gate.weight` — Qwen3.5 MoE, general MoE
    /// - `.mixer.gate.weight` — Nemotron-H
    /// - `.router.proj.weight` — Gemma4 (already handled separately)
    public static func dequantizeMoEGates(
        weights: inout [String: MLXArray], groupSize: Int, bitWidthsUsed: [Int] = []
    ) {
        // Find gate weight keys that have .scales companion (meaning they're quantized)
        var gateBasePaths = Set<String>()

        for key in weights.keys {
            // Match gate patterns but NOT gate_proj (which is an expert MLP weight)
            if key.hasSuffix(".gate.scales") && !key.contains("gate_proj") && !key.contains("gate_up") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
            // Also match shared_expert_gate (Qwen3.5 MoE)
            if key.hasSuffix(".shared_expert_gate.scales") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
        }

        for basePath in gateBasePaths {
            guard let gateWeight = weights[basePath + ".weight"],
                let gateScales = weights[basePath + ".scales"]
            else { continue }

            let gateBiases = weights[basePath + ".biases"]

            let packedDim = gateWeight.shape.last ?? 0
            let numGroups = gateScales.shape.last ?? 1

            // Infer bits using bitWidthsUsed (highest first — gates are CRITICAL tier).
            // Shape-only inference is ambiguous: multiple (bits, gs) produce the same
            // packed shapes. Using the known bit widths resolves the ambiguity.
            // For each candidate bits, compute inDim = packedDim * 32 / bits and check
            // that numGroups divides it evenly.
            var inferredBits = 4
            var inferredGroupSize = groupSize

            let candidates = bitWidthsUsed.isEmpty
                ? [8, 6, 5, 4, 3, 2]
                : bitWidthsUsed.sorted(by: >)

            for bits in candidates {
                guard bits > 0 && (packedDim * 32) % bits == 0 else { continue }
                let inDim = (packedDim * 32) / bits
                guard numGroups > 0 && inDim % numGroups == 0 else { continue }
                let gs = inDim / numGroups
                // Verify round-trip: packing inDim at this bits produces packedDim
                if (inDim * bits + 31) / 32 == packedDim || inDim * bits / 32 == packedDim {
                    inferredBits = bits
                    inferredGroupSize = gs
                    break
                }
            }

            // Dequantize to float32 for routing precision
            let dequantized = MLX.dequantized(
                gateWeight, scales: gateScales, biases: gateBiases,
                groupSize: inferredGroupSize, bits: inferredBits)

            // Replace quantized gate with float version, remove scales/biases
            weights[basePath + ".weight"] = dequantized.asType(.float32)
            weights.removeValue(forKey: basePath + ".scales")
            weights.removeValue(forKey: basePath + ".biases")
        }
    }

    // MARK: - V1 Format Support

    /// Check if a model directory contains v1 format JANG weights.
    public static func hasV1Weights(at modelPath: URL) -> Bool {
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: modelPath, includingPropertiesForKeys: nil)
        else { return false }
        return files.contains {
            $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
        }
    }

    /// Load JANG v1 format weights (legacy uint8 → uint32 repacking).
    public static func loadV1Weights(at modelPath: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let files =
            try fm.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
            }

        guard !files.isEmpty else {
            throw JangLoaderError.loadFailed(
                "No .jang.safetensors files found at \(modelPath.path)")
        }

        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let (weights, _) = try loadArraysAndMetadata(url: file)
            for (key, array) in weights {
                if array.dtype == .uint8 {
                    allWeights[key] = repackUint8ToUint32(array)
                } else {
                    allWeights[key] = array
                }
            }
        }
        return allWeights
    }

    /// Repack a uint8 array to uint32 by packing groups of 4 bytes (little-endian).
    private static func repackUint8ToUint32(_ array: MLXArray) -> MLXArray {
        let shape = array.shape
        let lastDim = shape.last ?? 0
        guard lastDim % 4 == 0 else { return array.asType(.uint32) }

        var newShape = shape
        newShape[newShape.count - 1] = lastDim / 4
        newShape.append(4)

        let reshaped = array.reshaped(newShape)
        let b0 = reshaped[0..., 0].asType(.uint32)
        let b1 = reshaped[0..., 1].asType(.uint32) << 8
        let b2 = reshaped[0..., 2].asType(.uint32) << 16
        let b3 = reshaped[0..., 3].asType(.uint32) << 24
        return b0 | b1 | b2 | b3
    }

    // MARK: - Helpers

    private static func floatValue(_ value: Any?) -> Float? {
        if let d = value as? Double { return Float(d) }
        if let f = value as? Float { return f }
        if let i = value as? Int { return Float(i) }
        return nil
    }

    // MARK: - §421 JANGTQ shape-authoritative routed-bits resolution

    /// §421 — Read safetensors HEADERS (not tensor data) in
    /// `modelDirectory`, find any tensor whose key path contains
    /// `tq_packed`, and return the routed-expert bit width implied by
    /// that tensor's last-dim packing.
    ///
    /// The shape relationship is exact:
    ///     packed_cols == ceil(in_features * bits / 32)
    /// where `in_features` is one of the dimensions declared by
    /// config.json (`hidden_size`, `moe_intermediate_size`,
    /// `intermediate_size`, optionally nested under `text_config`).
    /// We try every candidate `in_features` against the legal set
    /// {2, 3, 4, 6, 8} and accept the unique pairing that satisfies
    /// the equation.
    ///
    /// **Why this exists.** Some JANGTQ bundles (Qwen3.6-A3B-JANGTQ4
    /// pre-2026-04-25-patch, Kimi-K2.6-Small-JANGTQ, MiniMax-M2.7-JANGTQ)
    /// ship config.json with neither `mxtq_bits` nor
    /// `routed_expert_bits`. The downstream config-decode chain falls
    /// back to top-level `quantization.bits` — but that field describes
    /// AFFINE non-routed bits, NOT routed-expert bits. When those two
    /// differ (e.g. 8-bit affine attention with 4-bit routed experts),
    /// constructing `TurboQuantSwitchLinear` with the wrong bits value
    /// allocates the wrong `_packed` shape — load either fails with
    /// a generic shape-mismatch error or, if shapes happen to align,
    /// silently produces garbage at inference because the kernel reads
    /// the wrong codebook + bit-shifts off the wrong stride.
    ///
    /// This function returns the SHAPE-AUTHORITATIVE answer from on-disk
    /// data. Callers in `LLMModelFactory` use it to OVERRIDE the
    /// `mxtq_bits` field in `configData` before the model is constructed,
    /// so `TurboQuantSwitchLinear` is built at the right bits from the
    /// start. Returns nil when:
    ///   • The bundle has no `tq_packed` tensors (not a JANGTQ bundle)
    ///   • Multiple `tq_packed` tensors disagree on bits (bundle is
    ///     internally inconsistent — let the explicit shape-mismatch at
    ///     load time surface it)
    ///   • No (in_features, bits) pairing matches (config dimensions are
    ///     missing or the bundle uses non-standard packing)
    public static func peekRoutedBitsFromSafetensors(
        modelDirectory: URL,
        configData: Data
    ) -> Int? {
        let candidateInFeatures = collectInFeaturesCandidates(from: configData)
        let namedDims = collectNamedInFeatures(from: configData)
        guard !candidateInFeatures.isEmpty else { return nil }

        let scan = safetensorsHeaderScanTargets(in: modelDirectory)
        guard !scan.targets.isEmpty else { return nil }

        var votes: [Int: Int] = [:]
        var unresolvedTQPacked = 0
        for target in scan.targets {
            // Read header: 8-byte little-endian uint64 = JSON length, then
            // that many UTF-8 bytes of JSON metadata. We never read the
            // tensor payload — header-only is O(filename count), not
            // O(parameter count).
            guard let headerJSON = try? readSafetensorsHeader(at: target.url) else { continue }
            for (key, entry) in headerJSON {
                guard indexAllowsSafetensorKey(
                    key,
                    in: target.indexFileName,
                    weightMap: scan.weightMap
                ) else { continue }
                guard key.contains("tq_packed"),
                      let dict = entry as? [String: Any],
                      let shape = dict["shape"] as? [Int],
                      let packedCols = shape.last,
                      packedCols > 0
                else { continue }
                let exactCandidates = exactInFeaturesForTQPackedKey(key, dims: namedDims)
                let bits = exactCandidates.isEmpty
                    ? solveBitsForPackedCols(packedCols, candidates: candidateInFeatures)
                    : solveBitsForPackedCols(packedCols, candidates: exactCandidates)
                if let bits {
                    votes[bits, default: 0] += 1
                } else {
                    unresolvedTQPacked += 1
                }
            }
        }

        guard unresolvedTQPacked == 0 else {
            return nil   // mixed/ambiguous bundle; caller must use a
                         // richer family-specific plan or explicit metadata.
        }
        guard votes.count == 1, let bits = votes.keys.first else {
            return nil   // no match OR disagreement — caller falls
                         // through to existing config-driven resolution
        }
        return bits
    }

    /// Header-only mixed projection-bit sniffer for prestacked/per-expert
    /// JANGTQ SwiGLU experts.
    ///
    /// MiniMax JANGTQ_K stores `gate_proj`/`up_proj` at 2-bit and
    /// `down_proj` at 4-bit. The scalar `peekRoutedBitsFromSafetensors`
    /// intentionally returns nil for that mixed bundle, because flattening
    /// the profile to a single width is wrong. This richer probe solves each
    /// projection independently from its `tq_packed` last dimension:
    ///
    ///   gate_proj / w1  input = hidden_size
    ///   up_proj   / w3  input = hidden_size
    ///   down_proj / w2  input = intermediate/moe_intermediate_size
    ///
    /// It reads only safetensors JSON headers and returns nil unless all
    /// three projections are present and internally consistent.
    public static func peekRoutedProjectionBitsFromSafetensors(
        modelDirectory: URL,
        configData: Data
    ) -> [String: Int]? {
        let namedDims = collectNamedInFeatures(from: configData)
        guard !namedDims.hidden.isEmpty,
              !namedDims.expertIntermediate.isEmpty
        else { return nil }

        let scan = safetensorsHeaderScanTargets(in: modelDirectory)
        guard !scan.targets.isEmpty else { return nil }

        var votesByProjection: [String: [Int: Int]] = [:]
        var unresolved = 0
        for target in scan.targets {
            guard let headerJSON = try? readSafetensorsHeader(at: target.url) else { continue }
            for (key, entry) in headerJSON {
                guard indexAllowsSafetensorKey(
                    key,
                    in: target.indexFileName,
                    weightMap: scan.weightMap
                ) else { continue }
                guard key.contains("tq_packed"),
                      let projection = routedProjectionRole(forTQPackedKey: key),
                      let dict = entry as? [String: Any],
                      let shape = dict["shape"] as? [Int],
                      let packedCols = shape.last,
                      packedCols > 0
                else { continue }

                let exactCandidates = exactInFeaturesForTQPackedKey(key, dims: namedDims)
                let bits = exactCandidates.isEmpty
                    ? nil
                    : solveBitsForPackedCols(packedCols, candidates: exactCandidates)
                guard let bits else {
                    unresolved += 1
                    continue
                }
                votesByProjection[projection, default: [:]][bits, default: 0] += 1
            }
        }

        guard unresolved == 0 else { return nil }
        var out: [String: Int] = [:]
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            guard let votes = votesByProjection[projection],
                  votes.count == 1,
                  let bits = votes.keys.first
            else {
                return nil
            }
            out[projection] = bits
        }
        return out
    }

    /// Header-only DSV4 mixed-bit routed-expert plan sniffer.
    ///
    /// DeepSeek-V4 JANGTQ2-family bundles can be "mostly 2-bit" while
    /// protecting hash-routed / high-sensitivity layers at 4-bit. The
    /// scalar `mxtq_bits` / `routed_expert_bits` metadata cannot express
    /// that. The converter still writes per-tensor `.tq_bits`, but the
    /// Swift module graph must know the bit width BEFORE model.update()
    /// because `TurboQuantSwitchGLU` chooses its codebook + bit-unpack
    /// stride at construction time.
    ///
    /// This peeks only safetensors JSON headers and infers a layer → bits
    /// map from raw DSV4 tensor names:
    ///
    ///   layers.<L>.ffn.experts.<E>.w1.tq_packed              in_features=hidden
    ///   layers.<L>.ffn.experts.<E>.w3.tq_packed              in_features=hidden
    ///   layers.<L>.ffn.experts.<E>.w2.tq_packed              in_features=moe_intermediate
    ///   layers.<L>.(ffn|mlp).switch_mlp.gate_proj.tq_packed  in_features=hidden
    ///   layers.<L>.(ffn|mlp).switch_mlp.up_proj.tq_packed    in_features=hidden
    ///   layers.<L>.(ffn|mlp).switch_mlp.down_proj.tq_packed  in_features=moe_intermediate
    ///
    /// Returns nil if no DSV4 routed tensors are found or if any layer is
    /// internally inconsistent.
    public static func peekDSV4RoutedBitsByLayerFromSafetensors(
        modelDirectory: URL,
        configData: Data
    ) -> [Int: Int]? {
        guard let dims = collectDSV4RoutedDims(from: configData) else { return nil }
        let scan = safetensorsHeaderScanTargets(in: modelDirectory)
        guard !scan.targets.isEmpty else { return nil }

        let perExpertPattern =
            #"^layers\.(\d+)\.ffn\.experts\.\d+\.(w[123]|gate_proj|up_proj|down_proj)\.tq_packed$"#
        let stackedPattern =
            #"^layers\.(\d+)\.(?:ffn|mlp)\.switch_mlp\.(gate_proj|up_proj|down_proj)\.tq_packed$"#
        guard let perExpertRegex = try? NSRegularExpression(pattern: perExpertPattern),
              let stackedRegex = try? NSRegularExpression(pattern: stackedPattern)
        else { return nil }

        var votesByLayer: [Int: [Int: Int]] = [:]
        for target in scan.targets {
            guard let headerJSON = try? readSafetensorsHeader(at: target.url) else { continue }
            for (key, entry) in headerJSON {
                guard indexAllowsSafetensorKey(
                    key,
                    in: target.indexFileName,
                    weightMap: scan.weightMap
                ) else { continue }
                let nsRange = NSRange(key.startIndex..<key.endIndex, in: key)
                let match = perExpertRegex.firstMatch(in: key, range: nsRange)
                    ?? stackedRegex.firstMatch(in: key, range: nsRange)
                guard let match,
                      match.numberOfRanges == 3,
                      let layerRange = Range(match.range(at: 1), in: key),
                      let projRange = Range(match.range(at: 2), in: key),
                      let layer = Int(key[layerRange]),
                      let dict = entry as? [String: Any],
                      let shape = dict["shape"] as? [Int],
                      let packedCols = shape.last
                else { continue }

                let proj = String(key[projRange])
                let inFeatures = (proj == "w2" || proj == "down_proj")
                    ? dims.moeIntermediateSize
                    : dims.hiddenSize
                guard let bits = solveBitsForPackedCols(
                    packedCols,
                    exactInFeatures: inFeatures
                ) else {
                    return nil
                }
                votesByLayer[layer, default: [:]][bits, default: 0] += 1
            }
        }

        guard !votesByLayer.isEmpty else { return nil }
        var out: [Int: Int] = [:]
        for (layer, votes) in votesByLayer {
            guard votes.count == 1, let bits = votes.keys.first else {
                return nil
            }
            out[layer] = bits
        }
        return out
    }

    /// Inject a DSV4 per-layer routed-expert bit plan into config.json.
    /// JSON object keys are strings so the receiving decoder can parse
    /// both ordinary JSON and Swift's Dictionary<Int, Int> variants.
    public static func injectRoutedBitsByLayer(
        into configData: Data,
        bitsByLayer: [Int: Int]
    ) -> Data {
        guard !bitsByLayer.isEmpty,
              var dict = (try? JSONSerialization.jsonObject(with: configData))
                as? [String: Any]
        else { return configData }

        var encoded: [String: Int] = [:]
        for (layer, bits) in bitsByLayer.sorted(by: { $0.key < $1.key }) {
            encoded[String(layer)] = bits
        }
        dict["routed_expert_bits_by_layer"] = encoded

        guard let mutated = try? JSONSerialization.data(withJSONObject: dict)
        else { return configData }
        return mutated
    }

    /// Inject a per-projection routed-expert bit plan without flattening it
    /// into `mxtq_bits`. Mixed profiles have no single correct scalar.
    public static func injectRoutedProjectionBits(
        into configData: Data,
        bitsByProjection: [String: Int]
    ) -> Data {
        let normalized = [
            "gate_proj": bitsByProjection["gate_proj"] ?? bitsByProjection["gate"] ?? bitsByProjection["w1"],
            "up_proj": bitsByProjection["up_proj"] ?? bitsByProjection["up"] ?? bitsByProjection["w3"],
            "down_proj": bitsByProjection["down_proj"] ?? bitsByProjection["down"] ?? bitsByProjection["w2"],
        ].compactMapValues { $0 }
        guard normalized.keys.contains("gate_proj"),
              normalized.keys.contains("up_proj"),
              normalized.keys.contains("down_proj"),
              var dict = (try? JSONSerialization.jsonObject(with: configData))
                as? [String: Any]
        else { return configData }

        dict["routed_expert_projection_bits"] = normalized
        dict["mxtq_projection_bits"] = normalized

        if var textConfig = dict["text_config"] as? [String: Any] {
            textConfig["routed_expert_projection_bits"] = normalized
            textConfig["mxtq_projection_bits"] = normalized
            dict["text_config"] = textConfig
        }

        guard let mutated = try? JSONSerialization.data(withJSONObject: dict)
        else { return configData }
        return mutated
    }

    /// §421 — Inject `mxtq_bits` into a config.json byte buffer at the
    /// top level. Used by JANGTQ factory closures to apply the
    /// shape-authoritative override before `JSONDecoder.decode(...)`
    /// runs. Preserves all other fields exactly. Returns the original
    /// `configData` unchanged if the JSON cannot be parsed (callers
    /// will surface that downstream).
    ///
    /// Behavior choices:
    ///   • Top-level field, not nested under `text_config`. The
    ///     Qwen35JANGTQTextConfiguration / DeepseekV4JANGTQConfiguration
    ///     decoders BOTH accept `mxtq_bits` at either level (top-level
    ///     wins via the §346 dict/Int dual-form parser).
    ///   • Always overwrites — if the bundle declared a wrong value,
    ///     shape-authoritative wins. The on-disk packing is ground
    ///     truth; the metadata field is convention.
    public static func injectRoutedBits(
        into configData: Data,
        bits: Int
    ) -> Data {
        guard var dict = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any]
        else { return configData }
        // §425 — make all writes idempotent (only fill when missing OR
        // when the existing value is a dict, which means the merge step
        // hasn't yet flattened the per-role dict to the routed-expert Int).
        let topMxtqExisting = dict["mxtq_bits"]
        let needsTopMxtq = (topMxtqExisting == nil) || (topMxtqExisting is [String: Any]) ||
            ((topMxtqExisting as? Int) != nil && (topMxtqExisting as? Int) != bits)
        if needsTopMxtq {
            dict["mxtq_bits"] = bits
        }
        // §423 (2026-04-25) — also mirror into BOTH `routed_expert_bits`
        // (DSV4 convention) AND nested `text_config.mxtq_bits`
        // (Qwen3.6/Holo3 convention). The nested-config gotcha bit on
        // Qwen3.6-A3B-JANGTQ4: outer config had `mxtq_bits` but
        // `Qwen35JANGTQTextConfiguration.init` reads ONLY from
        // `text_config` when present, so top-level injection was
        // invisible. Setting all three locations is idempotent (we
        // never overwrite a bundle-declared value other than the
        // top-level, which we already overwrote above) and eliminates
        // the nested-config trap for any future model class that
        // delegates to a `text_config` decoder.
        if dict["routed_expert_bits"] == nil {
            dict["routed_expert_bits"] = bits
        }
        if var textConfig = dict["text_config"] as? [String: Any] {
            if textConfig["mxtq_bits"] == nil {
                textConfig["mxtq_bits"] = bits
            }
            dict["text_config"] = textConfig
        }
        guard let mutated = try? JSONSerialization.data(withJSONObject: dict)
        else { return configData }
        return mutated
    }

    /// Read a single safetensors file's JSON header. Public for
    /// regression tests; loaders should call peekRoutedBitsFromSafetensors
    /// instead.
    public static func readSafetensorsHeader(at url: URL) throws -> [String: Any] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let lenBytes = handle.readData(ofLength: 8)
        guard lenBytes.count == 8 else {
            throw JangLoaderError.loadFailed(
                "safetensors header truncated: \(url.lastPathComponent)")
        }
        // Little-endian uint64. Header lengths are always < 2^32 in
        // practice; cast guard prevents overflow on 32-bit platforms.
        var n: UInt64 = 0
        for i in 0..<8 {
            n |= UInt64(lenBytes[i]) << (8 * i)
        }
        guard n > 0, n < 64 * 1024 * 1024 else {
            throw JangLoaderError.loadFailed(
                "safetensors header length suspicious (\(n)): "
              + url.lastPathComponent)
        }
        let json = handle.readData(ofLength: Int(n))
        guard json.count == Int(n) else {
            throw JangLoaderError.loadFailed(
                "safetensors header read short: \(url.lastPathComponent)")
        }
        guard let parsed = try JSONSerialization.jsonObject(with: json)
            as? [String: Any]
        else {
            throw JangLoaderError.loadFailed(
                "safetensors header is not a JSON object: "
              + url.lastPathComponent)
        }
        return parsed
    }

    /// Extract candidate `in_features` values from a config.json byte
    /// buffer. Reads `hidden_size`, `intermediate_size`,
    /// `moe_intermediate_size`, `shared_expert_intermediate_size`, and
    /// the same fields nested under `text_config`. Filters to >0.
    private static func collectInFeaturesCandidates(from configData: Data) -> [Int] {
        guard let dict = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any]
        else { return [] }
        var values = Set<Int>()
        let keys = [
            "hidden_size", "intermediate_size", "moe_intermediate_size",
            "shared_expert_intermediate_size", "head_dim",
        ]
        for source in [dict, (dict["text_config"] as? [String: Any]) ?? [:]] {
            for k in keys {
                if let v = source[k] as? Int, v > 0 {
                    values.insert(v)
                }
            }
        }
        return Array(values).sorted()
    }

    private struct NamedInFeatureDims {
        var hidden: [Int]
        var expertIntermediate: [Int]
    }

    /// Extract named dimensions so known expert projection keys can be solved
    /// against their exact input dimension instead of the broad candidate set.
    ///
    /// This matters for Laguna/Nemotron-style bundles where, for example,
    /// `gate_up_proj.tq_packed` with packed_cols=128 can be explained by
    /// hidden_size=2048 at 2-bit OR moe_intermediate_size=512 at 8-bit. The
    /// tensor name disambiguates the input side: gate/up/fc1/w1/w3 read the
    /// hidden state, while down/fc2/w2 read the expert intermediate.
    private static func collectNamedInFeatures(from configData: Data) -> NamedInFeatureDims {
        guard let dict = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any]
        else { return NamedInFeatureDims(hidden: [], expertIntermediate: []) }

        var hidden = Set<Int>()
        var expertIntermediate = Set<Int>()
        let sources = [dict, (dict["text_config"] as? [String: Any]) ?? [:]]
        for source in sources {
            if let v = source["hidden_size"] as? Int, v > 0 {
                hidden.insert(v)
            }
            for key in [
                "moe_intermediate_size",
                "intermediate_size",
                "expert_intermediate_size",
                "ffn_hidden_size",
                "n_inner",
            ] {
                if let v = source[key] as? Int, v > 0 {
                    expertIntermediate.insert(v)
                }
            }
        }

        return NamedInFeatureDims(
            hidden: Array(hidden).sorted(),
            expertIntermediate: Array(expertIntermediate).sorted())
    }

    private static func exactInFeaturesForTQPackedKey(
        _ key: String,
        dims: NamedInFeatureDims
    ) -> [Int] {
        let lower = key.lowercased()
        if lower.hasSuffix("gate_up_proj.tq_packed")
            || lower.hasSuffix("gate_proj.tq_packed")
            || lower.hasSuffix("up_proj.tq_packed")
            || lower.hasSuffix("w1.tq_packed")
            || lower.hasSuffix("w3.tq_packed")
            || lower.hasSuffix("fc1.tq_packed")
        {
            return dims.hidden
        }
        if lower.hasSuffix("down_proj.tq_packed")
            || lower.hasSuffix("w2.tq_packed")
            || lower.hasSuffix("fc2.tq_packed")
        {
            return dims.expertIntermediate
        }
        return []
    }

    private static func routedProjectionRole(forTQPackedKey key: String) -> String? {
        let lower = key.lowercased()
        if lower.hasSuffix("gate_proj.tq_packed")
            || lower.hasSuffix("w1.tq_packed")
        {
            return "gate_proj"
        }
        if lower.hasSuffix("up_proj.tq_packed")
            || lower.hasSuffix("w3.tq_packed")
        {
            return "up_proj"
        }
        if lower.hasSuffix("down_proj.tq_packed")
            || lower.hasSuffix("w2.tq_packed")
        {
            return "down_proj"
        }
        return nil
    }

    private static func collectDSV4RoutedDims(
        from configData: Data
    ) -> (hiddenSize: Int, moeIntermediateSize: Int)? {
        guard let dict = (try? JSONSerialization.jsonObject(with: configData))
            as? [String: Any]
        else { return nil }
        let source = (dict["text_config"] as? [String: Any]) ?? dict
        guard let hidden = source["hidden_size"] as? Int, hidden > 0,
              let moe = source["moe_intermediate_size"] as? Int, moe > 0
        else { return nil }
        return (hidden, moe)
    }

    /// Given `packed_cols` from a tq_packed tensor's last dim, return
    /// the unique `bits` value (in {2,3,4,6,8}) for which there exists
    /// a candidate `in_features` such that
    ///     packed_cols == ceil(in_features * bits / 32) == (in + 32/bits - 1) / (32/bits)
    /// Returns nil if no pairing matches OR multiple match (ambiguous).
    private static func solveBitsForPackedCols(
        _ packedCols: Int,
        candidates: [Int]
    ) -> Int? {
        var matched: Set<Int> = []
        for inFeatures in candidates where inFeatures > 0 {
            for bits in [2, 3, 4, 6, 8] {
                let valsPerU32 = 32 / bits
                let expected = (inFeatures + valsPerU32 - 1) / valsPerU32
                if expected == packedCols {
                    matched.insert(bits)
                    break  // first bits hit per inFeatures wins
                }
            }
        }
        return matched.count == 1 ? matched.first : nil
    }

    private static func solveBitsForPackedCols(
        _ packedCols: Int,
        exactInFeatures: Int
    ) -> Int? {
        guard packedCols > 0, exactInFeatures > 0 else { return nil }
        var matched: Set<Int> = []
        for bits in [2, 3, 4, 6, 8] {
            let valsPerU32 = 32 / bits
            let expected = (exactInFeatures + valsPerU32 - 1) / valsPerU32
            if expected == packedCols {
                matched.insert(bits)
            }
        }
        return matched.count == 1 ? matched.first : nil
    }

    // MARK: - §421b JANGTQ shape-authoritative validator (post-load)


    /// §421 — Inspect post-sanitize weight tensors for a JANGTQ bundle and
    /// verify every `tq_packed` shape is internally consistent with a single
    /// `(bits, in_features)` choice. Returns a struct describing the inferred
    /// routed-expert bit width and any inconsistencies.
    ///
    /// **Why this exists.** Some JANGTQ bundles (Qwen3.6-A3B-JANGTQ4,
    /// Kimi-K2.6-Small-JANGTQ at audit time 2026-04-25) shipped without
    /// `mxtq_bits` / `routed_expert_bits` fields, so the config-time
    /// resolution chain falls back to top-level `quantization.bits` —
    /// which describes affine non-routed bits, NOT routed-expert bits.
    /// When those two differ (8-bit affine attention with 4-bit routed
    /// experts) the loader picked 8 → TurboQuantSwitchLinear allocated
    /// the wrong-sized `_packed` array → either hard shape-mismatch on
    /// `model.update` OR silent-garbage output if the kernel happened
    /// to consume mismatched packing without throwing.
    ///
    /// `inferRoutedBitsFromWeights` is the **shape-authoritative** answer:
    /// `tq_packed.shape == [E, out_features, packed_cols]` where
    /// `packed_cols == ceil(in_features * bits / 32)`. Given a candidate
    /// `in_features` from config, we solve for bits and validate that
    /// it lies in the legal set {2, 3, 4, 6, 8}. Run this against EVERY
    /// `*.tq_packed` tensor; all must agree.
    ///
    /// Invocation pattern:
    /// ```swift
    /// let report = JangLoader.inferRoutedBitsFromWeights(
    ///     weights: weights,
    ///     candidateInFeatures: [hiddenSize, moeIntermediateSize, intermediateSize]
    /// )
    /// if let bits = report.inferredBits, bits != configuredBits {
    ///     // log warning, the bundle metadata is incomplete or wrong
    /// }
    /// ```
    public struct JANGTQShapeReport: Sendable {
        /// Bit width that ALL inspected `tq_packed` tensors agree on, or
        /// nil if no tensors were found / shapes were inconsistent.
        public let inferredBits: Int?

        /// Per-tensor diagnoses. Empty when bundle has no `tq_packed`
        /// tensors (i.e. not a JANGTQ bundle, or a misnamed one).
        public let perTensor: [(key: String, packedCols: Int, inferredBits: Int?, inFeaturesUsed: Int?)]

        /// Human-readable warning strings. Empty when everything aligns.
        /// Otherwise contains one line per anomaly:
        /// - "bundle has tq_packed but no candidate in_features matched"
        /// - "tq_packed tensors disagree on bits: X vs Y"
        /// - "bundle bits=4 but config declared bits=8"
        public let warnings: [String]
    }

    /// §421 — Walk post-sanitize weights, find every `tq_packed` tensor,
    /// and infer the routed-expert bit width from on-disk shape.
    ///
    /// `candidateInFeatures` lists every plausible in_features value
    /// (typically `[hiddenSize, moeIntermediateSize]` for MoE, sometimes
    /// also `intermediateSize` for shared-expert paths). The function
    /// tries each candidate against each tensor and accepts the bits
    /// value that yields a legal {2,3,4,6,8} for that pairing.
    ///
    /// Robust against:
    ///   • per-expert layout `experts.<E>.w<N>.tq_packed`  shape (out, P)
    ///   • stacked layout    `switch_mlp.X_proj.tq_packed` shape (E, out, P)
    ///   • any future flat layout where the LAST dim is packed_cols
    public static func inferRoutedBitsFromWeights(
        weights: [String: MLXArray],
        candidateInFeatures: [Int],
        configuredBits: Int? = nil
    ) -> JANGTQShapeReport {
        // Legal routed-expert bit widths: {2, 3, 4, 6, 8}. Inlined into
        // the per-tensor solver below (no need for a separate set).
        var warnings: [String] = []
        var perTensor: [(String, Int, Int?, Int?)] = []
        var votes: [Int: Int] = [:]   // bits → count of agreeing tensors

        for (key, arr) in weights where key.hasSuffix("tq_packed")
            || key.contains(".tq_packed") {
            let shape = arr.shape
            guard let packedCols = shape.last else {
                perTensor.append((key, 0, nil, nil))
                continue
            }

            // Try each candidate in_features. For a match we need:
            //   bits = packedCols * 32 / inFeatures   AND   packedCols == ceil(inFeatures * bits / 32)
            // The integer-rounding form keeps us honest on edge sizes.
            var found: (bits: Int, inFeatures: Int)?
            for inFeatures in candidateInFeatures where inFeatures > 0 {
                for bits in [2, 3, 4, 6, 8] {
                    let valsPerU32 = 32 / bits
                    let expected = (inFeatures + valsPerU32 - 1) / valsPerU32
                    if expected == packedCols {
                        found = (bits, inFeatures)
                        break
                    }
                }
                if found != nil { break }
            }

            if let f = found {
                perTensor.append((key, packedCols, f.bits, f.inFeatures))
                votes[f.bits, default: 0] += 1
            } else {
                perTensor.append((key, packedCols, nil, nil))
                warnings.append(
                    "[§421] tq_packed tensor \(key) shape=\(shape) does not match "
                    + "any (bits, in_features) pairing for in_features ∈ "
                    + "\(candidateInFeatures) at bits ∈ {2,3,4,6,8}. "
                    + "Bundle may use a non-standard packing or candidates list "
                    + "is incomplete.")
            }
        }

        let inferredBits: Int?
        switch votes.count {
        case 0:
            inferredBits = nil
        case 1:
            inferredBits = votes.keys.first
        default:
            inferredBits = nil
            // Pick the single bits if one tensor disagrees vs the majority,
            // but still warn loudly.
            let sorted = votes.sorted { $0.value > $1.value }
            warnings.append(
                "[§421] tq_packed tensors disagree on inferred bits: "
                + "\(sorted.map { "\($0.key)→\($0.value) tensors" }.joined(separator: ", "))"
                + ". Bundle is malformed — different experts/layers packed at "
                + "different bit widths. Affected keys: "
                + perTensor.compactMap { $0.2 != sorted.first?.key ? $0.0 : nil }
                    .prefix(3).joined(separator: ", "))
        }

        if let inferred = inferredBits, let configured = configuredBits, inferred != configured {
            warnings.append(
                "[§421] config-declared mxtq_bits=\(configured) but bundle's "
                + "tq_packed shapes imply bits=\(inferred). "
                + "Routed-expert TurboQuantSwitchLinear was constructed at the "
                + "wrong bit width — this would have produced silent-garbage "
                + "output. Either (a) add `mxtq_bits: \(inferred)` to the "
                + "bundle's config.json, or (b) re-quantize with a JANGTQ "
                + "tool that emits the correct metadata. The shape-authoritative "
                + "answer is \(inferred); trust that over the config field.")
        }

        return JANGTQShapeReport(
            inferredBits: inferredBits,
            perTensor: perTensor,
            warnings: warnings
        )
    }
}

// MARK: - Errors

public enum JangLoaderError: Error, LocalizedError, CustomStringConvertible, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case unsupportedVersion(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let path): return "JANG config not found at: \(path)"
        case .invalidConfig(let msg): return "Invalid JANG config: \(msg)"
        case .unsupportedVersion(let ver): return "Unsupported JANG version: \(ver)"
        case .loadFailed(let msg): return "JANG load failed: \(msg)"
        }
    }

    // Engine.swift reports load failures via `await fail("\(error)")`
    // which produces the enum-case form unless we also conform to
    // `CustomStringConvertible`. Without this, Load.swift's carefully-
    // crafted HTML-error-page message, empty-weights message, and
    // JANGTQ misconfig message all surfaced to the user as
    // `loadFailed("Shard …")` rather than the actual diagnostic.
    // **iter-66 (§95)**.
    public var description: String { errorDescription ?? "\(Self.self)" }
}
