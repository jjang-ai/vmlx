// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and updates the model with the weights.
///
/// When a JANG model is detected (via `jangConfig`), per-layer bit widths are
/// inferred from tensor shapes automatically. Standard MLX models are unaffected.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    jangConfig: JangConfig? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // Resolve symlinks (mlxstudio uses symlinked model directories)
    let modelDirectory = modelDirectory.resolvingSymlinksInPath()

    // JANGTQ-native detection: presence of the runtime sidecar means the
    // bundle ships tq_packed/tq_norms tensors that should be consumed RAW
    // by TurboQuantSwitchGLU — we must NOT run the MXTQ→affine expander,
    // the per-layer quant inference, or the MoE-gate dequant.
    let jangtqSidecarURL = modelDirectory.appendingPathComponent("jangtq_runtime.safetensors")
    let isJANGTQNative = FileManager.default.fileExists(atPath: jangtqSidecarURL.path)

    // .jangspec bundle: per-expert blobs + hot-core safetensors + flat index.
    // Read everything via JangSpecBundleLoader, which produces a {key: MLXArray}
    // dict identical in shape to the standard safetensors enumeration so the
    // rest of this function (sanitize, MXTQ dequant, per-layer quant inference,
    // model.update) runs unchanged.
    if JangSpecBundleLoader.isBundle(at: modelDirectory) {
        weights = try JangSpecBundleLoader.loadWeights(from: modelDirectory)
    } else if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                // Skip the JANGTQ sidecar — it contains runtime signs/codebook
                // arrays that go into JANGTQRuntimeCache, not module params.
                if url.lastPathComponent == "jangtq_runtime.safetensors" {
                    continue
                }
                // Defensive pre-check: Cloudflare / HuggingFace sometimes return
                // an HTML error page instead of the LFS blob when the download
                // rate-limits or auth fails. The resulting "shard" is a few
                // hundred bytes of `<!DOCTYPE HTML>...`, which MLX rejects with
                // the opaque "Invalid json header length" error. Catch this
                // specific mode here and surface a user-actionable message.
                // Live-triggered by MiniMax-M2.7-JANGTQ-CRACK shards 01 + 61
                // on 2026-04-16 (partial download — the HTML error pages were
                // saved instead of the real 1 GB LFS blobs).
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
                let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
                if size > 0 && size < 10_000 {
                    let head: Data? = {
                        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
                        defer { try? handle.close() }
                        return handle.readData(ofLength: 32)
                    }()
                    if let h = head,
                       let s = String(data: h, encoding: .utf8)?.lowercased(),
                       s.hasPrefix("<!doctype html") || s.hasPrefix("<html")
                    {
                        throw JangLoaderError.loadFailed(
                            "Shard \(url.lastPathComponent) is a \(size)-byte HTML "
                          + "error page, not a safetensors blob. The download "
                          + "from HuggingFace failed — re-fetch this file. "
                          + "(Common cause: Cloudflare / auth returned HTML "
                          + "when the LFS blob request was rate-limited.)")
                    }
                }
                let (w, m) = try loadArraysAndMetadata(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
                if metadata.isEmpty {
                    metadata = m
                }
            }
        }
    }

    // JANG MXTQ (TurboQuant-packed) dequantization.
    // Detects `.tq_packed` keys and rewrites them into affine
    // (.weight/.scales/.biases) triplets BEFORE per-model sanitize so that
    // downstream key-remap / expert-rename logic sees the final parameter
    // layout. No-op for non-MXTQ JANG models.
    if let jangConfig, !isJANGTQNative {
        do {
            _ = try dequantizeJangMXTQ(
                weights: &weights,
                jangConfig: jangConfig,
                mxtqSeed: jangConfig.mxtqSeed ?? 42,
                mxtqBits: jangConfig.mxtqBits
            )
        } catch {
            print("[loadWeights] MXTQ dequant failed: \(error)")
            throw error
        }
    }

    // JANGTQ native: load the signs/codebook sidecar into the runtime cache
    // before model.update() so TurboQuantSwitchGLU has everything it needs
    // on first forward.
    if isJANGTQNative {
        do {
            try JANGTQRuntimeCache.shared.loadSidecar(from: jangtqSidecarURL)
        } catch {
            print("[loadWeights] JANGTQ sidecar load failed: \(error)")
            throw error
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // JANG: dequantize MoE gate weights from quantized uint32 → float.
    // Gates are stored at 8-bit (CRITICAL tier) but may have different group_size
    // than the body. Dequantizing resolves ambiguous bit/group_size inference.
    if let jangConfig {
        JangLoader.dequantizeMoEGates(
            weights: &weights, groupSize: jangConfig.quantization.blockSize,
            bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)
    }

    // Determine quantization: JANG models infer per-layer bit widths from tensor shapes.
    // Standard MLX models use the quantization from config.json as before.
    let effectivePerLayerQuantization: BaseConfiguration.PerLayerQuantization?
    if let jangConfig {
        // Safe for JANGTQ-native: infer only walks `.scales` keys, so it picks
        // up the affine 8-bit attention / embed / lm_head and ignores the
        // tq_packed expert projections.
        effectivePerLayerQuantization = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig)
    } else if let perLayerQuantization {
        // Remap perLayerQuantization keys to match sanitized weight paths.
        // Config.json uses VLM-prefixed keys like "language_model.model.layers.0..."
        // LLM sanitize strips to "model.layers.0..." but VLM keeps "language_model.model.layers.0..."
        // Keep BOTH original and stripped keys so it works for both paths.
        var remappedPerLayer = perLayerQuantization.perLayerQuantization
        for (key, value) in perLayerQuantization.perLayerQuantization {
            if key.hasPrefix("language_model.model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            } else if key.hasPrefix("language_model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            }
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: perLayerQuantization.quantization,
            perLayerQuantization: remappedPerLayer
        )
    } else {
        effectivePerLayerQuantization = nil
    }

    // quantize if needed
    if quantization != nil || effectivePerLayerQuantization != nil {
        // Inline quantize with error logging instead of try! crash
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            guard weights["\(path).scales"] != nil else { return nil }
            let tup: (groupSize: Int, bits: Int, mode: QuantizationMode)?
            if let effectivePerLayerQuantization {
                tup = effectivePerLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                tup = quantization?.asTuple
            }
            guard let (gs, b, mode) = tup else { return nil }

            // MXFP4/MXFP8: quantizeSingle creates QuantizedLinear with dummy biases
            // from MLX.quantized(), but MX formats don't use biases. Create the module
            // directly with nil biases to avoid "biases must be null" at inference time.
            if (mode == .mxfp4 || mode == .mxfp8), m is Linear {
                let linear = m as! Linear
                let (qW, scales, _) = MLX.quantized(linear.weight, groupSize: gs, bits: b)
                return (path, QuantizedLinear(
                    weight: qW, bias: linear.bias, scales: scales, biases: nil,
                    groupSize: gs, bits: b, mode: mode))
            }

            if let q = quantizeSingle(layer: m, groupSize: gs, bits: b, mode: mode) {
                return (path, q)
            }
            return nil
        }
        do {
            try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
        } catch {
            print("[loadWeights] quantize model.update failed: \(error)")
            for (path, mod) in updates.prefix(5) {
                print("  update path: \(path) → \(type(of: mod))")
            }
            throw error
        }
    }

    // apply the loaded weights
    // Use .noUnusedKeys instead of .all — MXFP4/MXFP8 quantized layers don't have .biases
    // in the weight files, but QuantizedLinear's optional .biases property gets initialized
    // by the quantize step. Strict .all verification would fail on the missing keys.
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Convert all float16/float32 parameters to bfloat16 to prevent AsType cascades.
    // float16 causes AsType when mixed with internal float32 ops (softmax, RMSNorm).
    // bfloat16 shares float32's exponent range, so promotion is cheaper/eliminated.
    // Check model.parameters() (not the original weights dict) because the quantize step
    // above may have created QuantizedLinear modules with float32 scales from MLX.quantized().
    // JANG models with format:mlx have bfloat16 weights but the quantizer can still produce
    // float32 scales, causing 1000+ AsType ops if not converted.
    // JANGTQ: Python baseline runs with fp16 norms — convertToBFloat16 would
    // cast the TurboQuant norms dtype and the MLXFast kernel's inferred
    // signature, causing the gate/up/down projections to produce nonsense
    // (verified on MiniMax M2.7 JANGTQ_2L). Skip the bf16 cascade fix when
    // a JANGTQ sidecar is in play — JANGTQ dispatches are already fp32
    // internally, so there's no fp16↔fp32 ping-pong to collapse.
    if !isJANGTQNative {
        let allParams = model.parameters().flattened().map { $0.1 }
        let hasNonBFloat16 = allParams.contains { (arr: MLXArray) in
            arr.dtype == .float16 || arr.dtype == .float32
        }
        if hasNonBFloat16 {
            convertToBFloat16(model: model)
        }
    }

    eval(model)
}

/// Convert float16/float32 model parameters to bfloat16 for MoE performance.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing ~50% speed regression for MoE models where gate routing runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
/// Quantization scales/biases are ALSO converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
private func convertToBFloat16(model: Module) {
    var converted = [String: MLXArray]()
    for (key, array) in model.parameters().flattened() {
        if array.dtype == .float16 || array.dtype == .float32 {
            converted[key] = array.asType(.bfloat16)
        }
    }
    if !converted.isEmpty {
        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            print("[convertToBFloat16] model.update failed: \(error)")
        }
    }
}
