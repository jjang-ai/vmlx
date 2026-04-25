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
// DIAG (2026-04-25): mutable wrapper for diag counter, since loadWeights is
// a free function. Remove after path resolution diagnosis.
fileprivate final class _MutInt { var value: Int = 0 }
fileprivate let _loadDiagCounter = _MutInt()

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

    // JANGTQ-native detection: two signals, either one flips the bundle
    // into the native path (tq_packed/tq_norms stay RAW, no MXTQ→affine
    // expansion, no per-layer quant inference, no MoE-gate dequant).
    //  1. `jang_config.json -> weight_format == "mxtq"` — the canonical
    //     signal matching Python `jang_tools.load_jangtq` detection.
    //     Preferred because it doesn't require the optional sidecar.
    //  2. `jangtq_runtime.safetensors` sidecar present — legacy signal
    //     retained for bundles that still ship a sidecar.
    // Before 2026-04-16 we only checked (2), which silently misrouted
    // MiniMax-M2.7-JANGTQ-CRACK (no sidecar — signs regenerate at
    // runtime from mxtq_seed) through the affine expander, which then
    // blew up because the expansion didn't match the TQ-native model
    // shape. See `jang_tools/load_jangtq.py` in the jang-tools repo
    // for the Python reference loader.
    let jangtqSidecarURL = modelDirectory.appendingPathComponent("jangtq_runtime.safetensors")
    // Case-insensitive compare — jang-tools writes `"mxtq"` (lowercase)
    // but bundles coming from third-party converters or hand-edited
    // `jang_config.json` may use `"MXTQ"` or mixed case. A case-sensitive
    // exact match silently fell through to the affine expander.
    let isJANGTQNative =
        (jangConfig?.weightFormat?.lowercased() == "mxtq")
        || FileManager.default.fileExists(atPath: jangtqSidecarURL.path)

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

    // Empty-weight guard. A bundle with zero safetensors shards (typical
    // symptom: HuggingFace snapshot dir where only `config.json` +
    // tokenizer were fetched and the .safetensors blobs never downloaded)
    // used to silently fall through `model.update(verify: [.noUnusedKeys])`
    // because `noUnusedKeys` only checks that the incoming dict has no
    // extras — it does NOT check that every module parameter got a value.
    // The model then ran the MLX graph evaluator on its freshly-initialized
    // random parameters, which hangs Metal on anything MoE-sized (e.g.
    // Gemma 4 26B-A4B: the load bar pins at "100% Ready" while the backend
    // silently stalls on a 26B random-weight graph evaluation).
    //
    // Throw a clear, user-actionable message instead. Also runs for the
    // JangSpec / JANG-v1 branches above because they populate the same
    // `weights` dict — an empty JangSpec bundle fails here too.
    if weights.isEmpty {
        throw JangLoaderError.loadFailed(
            "No safetensors weights found in \(modelDirectory.lastPathComponent). "
          + "The bundle contains config/tokenizer files but no weight shards "
          + "— likely a partial HuggingFace download. Re-fetch the model "
          + "(DownloadStatusBar → pull again, or `vmlx pull <repo>`) and "
          + "retry. Expected: `model-0000X-of-0000Y.safetensors` or a "
          + "`.jangspec` bundle directory."
        )
    }

    // JANGTQ native-path validation. When `jang_config.weight_format ==
    // "mxtq"` (or the sidecar is present) we're committed to the
    // TurboQuant-native code path — we skipped the affine expander and
    // will run `TurboQuantSwitchGLU` over the loaded weights. If the
    // bundle ACTUALLY ships no `.tq_packed` tensors the expander would
    // have rebuilt them anyway, but the native path trusts them to be
    // there and crashes deep in `model.update` with a parameter-count
    // mismatch that points at the wrong layer. Catch it here with an
    // actionable message instead.
    //
    // The validation is cheap — we already have `weights.keys` enumerated —
    // and it fires only on the native path, so affine-JANG load is
    // unaffected.
    if isJANGTQNative {
        let hasTQPacked = weights.keys.contains { $0.hasSuffix(".tq_packed") }
        if !hasTQPacked {
            throw JangLoaderError.loadFailed(
                "Bundle declares weight_format=\"mxtq\" (or ships "
              + "jangtq_runtime.safetensors) but contains no .tq_packed "
              + "tensors. This is a misconfigured JANGTQ bundle — either "
              + "(a) set weight_format to the affine value if the bundle "
              + "is not TurboQuant-native, (b) re-run jang_tools' "
              + "convert_qwen35_jangtq / convert_glm4_jangtq / "
              + "convert_minimax_jangtq to regenerate .tq_packed shards, "
              + "or (c) delete jangtq_runtime.safetensors if this is an "
              + "affine JANG bundle. See the jang-tools repo for "
              + "the Python reference loader.")
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

    // JANGTQ native: optionally load signs/codebook sidecar into the runtime
    // cache before model.update(). Sidecar is OPTIONAL per Python
    // `load_jangtq.py` — when absent, `TurboQuantSwitchLinear` regenerates
    // sign sequences at runtime from `mxtq_seed` via `NumPyPCG64`. Skip
    // gracefully when not present instead of throwing (previously this
    // only ran when the file existed, but we now also enter this branch
    // from the jang_config.weight_format path which may not have a sidecar).
    if isJANGTQNative,
       FileManager.default.fileExists(atPath: jangtqSidecarURL.path)
    {
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
        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig)
        // §411 — opt-in on-disk bundle repair. No-op unless
        // `VMLX_REPAIR_BAD_JANG_CONFIG=1` is set. The runtime path-
        // variant fix below makes the loader correct WITHOUT modifying
        // the bundle; the on-disk repair is just for users who want
        // their config.json to also reflect the truth (helps third-
        // party tooling that reads the metadata).
        JangConfigRepair.repairIfBad(
            at: modelDirectory,
            inferredPerLayer: inferred.perLayerQuantization
        )
        // §400 — bundle disambiguation. JANG `inferBitWidthAndGroupSize` is
        // mathematically ambiguous when (bits, group_size) pairs both satisfy
        // the packed-shape equation. For example:
        //   weight=[V, 1024], scales=[V, 128] satisfies BOTH
        //     (bits=4, gs=64): inDim=8192
        //     (bits=8, gs=32): inDim=4096
        // When `jang_config.json` lacks an explicit quantization block (the
        // DSV4 Flash JANGTQ bundle ships `{"weight_format":"bf16"}` with no
        // bit-widths) the JANG path falls back to defaults `[2,4,6]` and
        // picks bits=4 — silently doubling the embed/head/attention output
        // dim. The bundle's authoritative answer lives in
        // `config.json::quantization` (top-level + per-layer overrides like
        // `"embed": {bits:8, group_size:32}`). When that block is present,
        // overlay it onto JANG inference: per-layer entries from config.json
        // win, JANG fills the gaps.
        if let perLayerQuantization {
            // §410 — shape-inferred entries are authoritative. The JANG
            // inferer (`inferPerLayerQuantization`) emits a per-layer
            // override at EVERY quantized leaf (no "matches default"
            // gate, per §410 in JangLoader.swift), so `inferred` already
            // contains the correct (bits, gs) for every layer that has
            // .scales on disk. config.json's per-layer block is treated
            // as a *backstop* — only used when the shape inferer didn't
            // emit an entry (purely unquantized layers, layers loaded
            // from a sidecar etc.) and broadcast across path variants
            // ("attn" ↔ "self_attn", `model.` prefix, language_model
            // strip) so leaf-path lookup resolves on either naming
            // convention.
            //
            // History: before §410 the merge ran the other direction —
            // config.json overrode shape inference. That worked when
            // config.json was correct but BROKE for bundles where
            // `jang_config.json` declared a uniform bits=2/gs=64 default
            // and the actual safetensors stored 8-bit/gs=32 attention
            // (the "BAD config" class). The shape-authoritative
            // direction works for both BAD and GOOD configs and stays
            // robust against future converter bugs.
            var merged = inferred.perLayerQuantization
            func variants(_ k: String) -> [String] {
                var out = [k]
                if k.contains(".attn.") || k.hasSuffix(".attn") {
                    out.append(k.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                    if k.hasSuffix(".attn") {
                        out.append(String(k.dropLast(".attn".count)) + ".self_attn")
                    }
                }
                return out
            }
            // ── §411 fix (2026-04-25): also expand `model.X` prefix variants
            // for SHAPE-INFERRED entries. The earlier §410 code only added
            // `model.X` variants for config.json-declared overrides, leaving
            // shape-inferred keys without the prefix. Symptom: leaf module
            // path is `model.embed` (after Swift sanitize prefixes with
            // `model.`), but inferred dict has just `embed`. Loader looks up
            // `model.embed`, gets nil, falls back to top-level → wrong bits
            // (e.g. picks 4-bit instead of 8-bit) → embed output dim doubles
            // → hcCollapse matmul crash with shape (B, L, 32768) × (16384, 24).
            // Diagnosed via stderr trace showing h.shape=[1, 5, 4, 8192] when
            // expected [1, 5, 4, 4096]. Fix: clone every shape-inferred entry
            // under `model.<key>` too.
            for (key, value) in inferred.perLayerQuantization {
                let modelPrefixed = "model.\(key)"
                if merged[modelPrefixed] == nil { merged[modelPrefixed] = value }
                // Also expand attn.→self_attn. variants for the model-prefixed form
                for v in variants(modelPrefixed) {
                    if merged[v] == nil { merged[v] = value }
                }
            }
            for (key, value) in perLayerQuantization.perLayerQuantization {
                for v in variants(key) {
                    // Backstop only — don't overwrite shape-inferred entries.
                    if merged[v] == nil { merged[v] = value }
                    let modelPrefixed = "model.\(v)"
                    if merged[modelPrefixed] == nil { merged[modelPrefixed] = value }
                }
                if key.hasPrefix("language_model.model.") {
                    let stripped = String(key.dropFirst("language_model.".count))
                    for v in variants(stripped) {
                        if merged[v] == nil { merged[v] = value }
                    }
                } else if key.hasPrefix("language_model.") {
                    let stripped = String(key.dropFirst("language_model.".count))
                    for v in variants(stripped) {
                        if merged[v] == nil { merged[v] = value }
                    }
                }
            }
            // Top-level (default) quantization: still prefer config.json's
            // explicit declaration over JANG's heuristic when present —
            // the top-level acts only as the fallback for paths absent
            // from per-layer; per-layer entries always win.
            effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
                quantization: perLayerQuantization.quantization ?? inferred.quantization,
                perLayerQuantization: merged
            )
            // §411 — diag dump gated behind `VMLX_LOAD_DIAG=1`. Off by
            // default so production loads stay quiet; flip on to debug
            // path-resolution mismatches between shape-inferred entries
            // and the loader's leaf-path lookup keys.
            if ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1" {
                let probeKeys = ["embed", "model.embed", "lm_head", "model.lm_head",
                                 "layers.0.attn.wq_a", "layers.0.self_attn.wq_a",
                                 "model.layers.0.attn.wq_a", "model.layers.0.self_attn.wq_a"]
                for k in probeKeys {
                    let v = merged[k]
                    let s = "[merge-diag] merged[\(k)] = \(v.map { "\($0)" } ?? "NIL")\n"
                    if let data = s.data(using: .utf8) {
                        try? FileHandle.standardError.write(contentsOf: data)
                    }
                }
                let topQ = perLayerQuantization.quantization ?? inferred.quantization
                let s = "[merge-diag] top-level quantization = \(topQ.map { "(b=\($0.bits), gs=\($0.groupSize))" } ?? "NIL"); merged_count=\(merged.count); inferred_count=\(inferred.perLayerQuantization.count); explicit_count=\(perLayerQuantization.perLayerQuantization.count)\n"
                if let data = s.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: data)
                }
            }
        } else {
            // §411 fix (2026-04-25): when config.json has no perLayerQuantization
            // block, the inferred dict alone is used — but its keys don't have
            // the `model.` prefix that the leafModule path lookup uses. Add
            // `model.X` variants so the lookup hits.
            var merged = inferred.perLayerQuantization
            func variants(_ k: String) -> [String] {
                var out = [k]
                if k.contains(".attn.") || k.hasSuffix(".attn") {
                    out.append(k.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                    if k.hasSuffix(".attn") {
                        out.append(String(k.dropLast(".attn".count)) + ".self_attn")
                    }
                }
                return out
            }
            for (key, value) in inferred.perLayerQuantization {
                for v in variants(key) {
                    if merged[v] == nil { merged[v] = value }
                }
                let modelPrefixed = "model.\(key)"
                if merged[modelPrefixed] == nil { merged[modelPrefixed] = value }
                for v in variants(modelPrefixed) {
                    if merged[v] == nil { merged[v] = value }
                }
            }
            effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
                quantization: inferred.quantization,
                perLayerQuantization: merged
            )
            if ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1" {
                let dbg = "[merge-diag-else] inferred_count=\(inferred.perLayerQuantization.count) merged_count=\(merged.count) sample_keys=\(Array(merged.keys).prefix(5))\n"
                if let data = dbg.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: data)
                }
            }
        }
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

    // §411 — diag entry summary, gated. Off by default.
    if ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1" {
        let leafCount = model.leafModules().flattened().count
        let quantHas = quantization != nil
        let effHas = effectivePerLayerQuantization != nil
        let scalesCount = weights.keys.filter { $0.hasSuffix(".scales") }.count
        let msg = "[load-diag entry] quantization=\(quantHas) effPerLayer=\(effHas) leafModules=\(leafCount) weight_scales_keys=\(scalesCount) isJANGTQNative=\(isJANGTQNative)\n"
        if let data = msg.data(using: .utf8) {
            try? FileHandle.standardError.write(contentsOf: data)
        }
    }

    // quantize if needed
    if quantization != nil || effectivePerLayerQuantization != nil {
        // Inline quantize with error logging instead of try! crash
        let loadDiag = ProcessInfo.processInfo.environment["VMLX_LOAD_DIAG"] == "1"
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            guard weights["\(path).scales"] != nil else { return nil }
            let tup: (groupSize: Int, bits: Int, mode: QuantizationMode)?
            if let effectivePerLayerQuantization {
                tup = effectivePerLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                tup = quantization?.asTuple
            }
            // §411 — per-leaf trace, gated. Limited to 12 entries on
            // embed/lm_head/early-layer paths so the trace is useful
            // without flooding stderr on big models.
            if loadDiag, _loadDiagCounter.value < 12 && (path.contains("embed") || path.contains("lm_head") || path.contains("layers.0") || path.contains("layers.1.")) {
                _loadDiagCounter.value += 1
                let direct = effectivePerLayerQuantization?.perLayerQuantization[path]
                let directDesc: String
                switch direct {
                case .none: directDesc = "MISS"
                case .some(.skip): directDesc = "SKIP"
                case .some(.quantize(let q)): directDesc = "(b=\(q.bits), gs=\(q.groupSize))"
                }
                let topQ = effectivePerLayerQuantization?.quantization
                let topDesc = topQ.map { "(b=\($0.bits), gs=\($0.groupSize))" } ?? "NIL"
                let msg = "[load-diag #\(_loadDiagCounter.value)] path=\(path) direct_lookup=\(directDesc) top=\(topDesc) tup=\(tup.map { "(b=\($0.bits), gs=\($0.groupSize), m=\($0.mode))" } ?? "NIL") module=\(type(of: m))\n"
                if let data = msg.data(using: .utf8) {
                    try? FileHandle.standardError.write(contentsOf: data)
                }
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
    //
    // 2026-04-18 memory fix — community users on a 3L MiniMax-M2.7-JANGTQ
    // report ~2x RAM at load vs Python, swapping 50 GB and hanging on
    // first response. Root cause: holding `weights` + `parameters`
    // (NestedDictionary) alive through the graph-eval step below
    // doubles the MLXArray refcount for every tensor. The Python
    // loader drops its equivalent dict via refcount decay on scope
    // exit; Swift needs the explicit clear plus an inline temporary
    // for `parameters` so it dies after model.update returns. No cost
    // on the success path; only helps giant JANGTQ bundles.
    try model.update(parameters: ModuleParameters.unflattened(weights),
                     verify: [.noUnusedKeys])
    weights.removeAll(keepingCapacity: false)

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
