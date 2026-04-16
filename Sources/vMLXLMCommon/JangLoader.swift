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
public struct JangLoader: Sendable {

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
        jangConfig: JangConfig
    ) -> BaseConfiguration.PerLayerQuantization {
        let groupSize = jangConfig.quantization.blockSize
        var perLayer = [String: BaseConfiguration.QuantizationOption]()

        // Find the default (most common) bit width from jang_config
        let defaultBits = jangConfig.quantization.bitWidthsUsed.min() ?? 4

        // Group weight keys by their base path (strip .weight/.scales/.biases suffix)
        var quantizedLayers = Set<String>()
        for key in weights.keys {
            if key.hasSuffix(".scales") {
                let basePath = String(key.dropLast(".scales".count))
                quantizedLayers.insert(basePath)
            }
        }

        // For each quantized layer, infer the actual bit width.
        // First try with the JANG block_size as group_size. If that doesn't produce
        // a valid integer bit width, fall back to inferring both bits and group_size
        // from shapes (handles gates with different group_size like 64 vs 128).
        for basePath in quantizedLayers {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else {
                continue
            }

            // Try with known group_size first; pass bitWidthsUsed so the
            // fallback can disambiguate layers whose group_size differs
            // from the body (e.g., MoE gates).
            let (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                weight: weightArray, scales: scalesArray,
                knownGroupSize: groupSize,
                bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)

            if bits != defaultBits || inferredGroupSize != groupSize {
                perLayer[basePath] = .quantize(
                    BaseConfiguration.Quantization(groupSize: inferredGroupSize, bits: bits))
            }
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
        let packedDim = weight.shape.last ?? 0
        let numGroups = scales.shape.last ?? 1

        guard packedDim > 0 && numGroups > 0 else { return (4, knownGroupSize ?? 64) }

        let validBits = [2, 3, 4, 5, 6, 8]

        // Primary path: knownGroupSize gives an unambiguous answer.
        // bits must divide (packedDim * 32) exactly.
        if let gs = knownGroupSize, gs > 0 {
            let inDim = numGroups * gs
            if inDim > 0 && (packedDim * 32) % inDim == 0 {
                let bits = (packedDim * 32) / inDim
                if validBits.contains(bits) {
                    return (bits, gs)
                }
            }
        }

        // Fallback: the provided knownGroupSize is wrong for this tensor (e.g.,
        // MoE gates use a different group_size than body layers). Search for
        // (bits, groupSize) pairs that produce an exact round-trip. Prefer
        // higher bits first to favor CRITICAL tier layers (which JANG stores
        // at the highest available bit width).
        //
        // For any candidate `bits`, the implied group size must be an integer:
        //   in_dim = (packedDim * 32) / bits    (must be exact)
        //   gs     = in_dim / numGroups         (must be exact)
        let candidates = bitWidthsUsed.isEmpty
            ? validBits.sorted(by: >)  // [8, 6, 5, 4, 3, 2]
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
}

// MARK: - Errors

public enum JangLoaderError: Error, LocalizedError, Sendable {
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
}
