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
        for basePath in quantizedLayers {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else {
                continue
            }

            // Pass bitWidthsUsed for the tail-safety fallback path; the
            // primary preference-ordered enumeration in
            // `inferBitWidthAndGroupSize` is shape-only and ignores
            // knownGroupSize.
            let (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                weight: weightArray, scales: scalesArray,
                knownGroupSize: groupSize,
                bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)

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
        let packedDim = weight.shape.last ?? 0
        let numGroups = scales.shape.last ?? 1
        // §411 — first-call shape trace, gated behind `VMLX_LOAD_DIAG=1`.
        // Useful for diagnosing path-resolution mismatches between
        // shape inference and the loader's leaf-path lookup; off by
        // default so production loads stay quiet.
        if !_inferDiagFired,
           ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1"
        {
            _inferDiagFired = true
            let msg = "[infer-diag] weight.shape=\(weight.shape) scales.shape=\(scales.shape) packedDim=\(packedDim) numGroups=\(numGroups) knownGS=\(String(describing: knownGroupSize)) bitsUsed=\(bitWidthsUsed)\n"
            if let data = msg.data(using: .utf8) {
                try? FileHandle.standardError.write(contentsOf: data)
            }
        }

        guard packedDim > 0 && numGroups > 0 else { return (4, knownGroupSize ?? 64) }

        // §410 — shape-authoritative bit-width inference (replaces the
        // earlier `knownGroupSize-first` path).
        //
        // The packed-shape equation is:
        //   weight.shape[-1] * 32 = bits * num_groups * group_size
        //   ⇒ bits * group_size = 32 * (packedDim / numGroups)
        //
        // For typical R = packedDim / numGroups = 8, the equation is
        // satisfied by THREE plausible pairs: (8, 32), (4, 64), (2, 128).
        // The earlier code locked `group_size` to whatever
        // `jang_config.quantization.block_size` declared (defaulting to
        // 64) and solved only for `bits` — which silently picked
        // (4, 64) for an attention/embed/head layer that the JANGTQ
        // converter actually emitted at (8, 32). Result: load succeeds
        // with the wrong bit width, embed output dim doubles, attention
        // matmul dimensions don't match, downstream tensors collapse to
        // 0-d scalars, model "loads" but generates garbage or crashes.
        //
        // The fix: enumerate (bits, group_size) pairs in the preference
        // order the JANGTQ converter actually emits — 8-bit before 4-bit
        // before 2-bit, smaller group_size before larger — and pick the
        // FIRST pair whose implied shape matches the on-disk tensor.
        // This makes shape authoritative regardless of what
        // `jang_config.json` (or any future converter bug) declares.
        //
        // Works for THREE config classes:
        //   - BAD config (bits=2 declared everywhere, mixed storage on
        //     disk): inferred shape correctly identifies 8-bit
        //     attention / embed / lm_head, ignores the 2-bit declaration.
        //   - GOOD config (per-layer overrides match disk): inferred
        //     shape agrees, no-op.
        //   - FUTURE config (new converter bug): inferred shape stays
        //     authoritative, model still loads correctly.
        //
        // Preference order (matches the JANGTQ converter classify rules
        // verified against 522/522 + 312/312 + 187/187 trusted bundles):
        //   8-bit attention / embed / lm_head / shared experts at
        //   group_size 32 first (most common, gs=64/128 fallback for
        //   wider hidden), then 4-bit (routed experts in 4-bit
        //   profile), 2-bit (routed experts in 2-bit profile),
        //   then 3/6-bit edge cases. (5-bit + odd group_sizes also
        //   tried last for completeness.)
        let preferred: [(Int, Int)] = [
            (8, 32), (8, 64), (8, 128),
            (4, 32), (4, 64), (4, 128),
            (2, 32), (2, 64), (2, 128),
            (3, 32), (6, 32),
            (5, 32), (5, 64), (3, 64), (6, 64),
        ]
        for (bits, gs) in preferred {
            // Implied in_dim = bits * num_groups * group_size / 32
            // packedDim must equal in_dim * bits / 32 = num_groups * group_size * bits^2 / (32 * 32)
            // Simpler: check bits * group_size == 32 * packedDim / numGroups
            // and packedDim * 32 % bits == 0 + the resulting in_dim divides cleanly.
            guard (packedDim * 32) % bits == 0 else { continue }
            let inDim = (packedDim * 32) / bits
            guard inDim > 0, inDim % numGroups == 0 else { continue }
            let impliedGs = inDim / numGroups
            if impliedGs == gs {
                return (bits, gs)
            }
        }

        // Tail safety: if no preferred pair matches, fall back to the
        // bitWidthsUsed-scoped search (preserves prior behavior for any
        // odd CRITICAL-tier layout the converter might emit).
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

    // MARK: - §421 JANGTQ shape-authoritative validator

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
