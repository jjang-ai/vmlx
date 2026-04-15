// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - KV Cache Extraction

/// Extract per-layer KV tensors from a model's cache array.
///
/// Returns per-layer `(keys, values)` tuples. SSM/MambaCache layers return `nil`.
/// Used to populate ``CacheBlock/cacheData`` for paged cache storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: An array of optional `(keys, values)` tuples, one per layer.
public func extractLayerData(from cache: [any KVCache]) -> [(keys: MLXArray, values: MLXArray)?] {
    cache.map { layer in
        if let simple = layer as? KVCacheSimple {
            let state = simple.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let quantized = layer as? QuantizedKVCache {
            // QuantizedKVCache stores quantized tuples, not raw KV.
            // Dequantize back to float keys/values for cache block storage.
            let unquantized = quantized.toUnquantized()
            let state = unquantized.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let tq = layer as? TurboQuantKVCache {
            // TurboQuantKVCache.state returns float KV in both fill and compressed phases.
            // In fill phase: returns raw float keys/values.
            // In compressed phase: returns unified (decompressed prefix + float window).
            let state = tq.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let cacheList = layer as? CacheList {
            // CacheList: check sub-caches for KV data.
            // Sub-cache[0] is typically MambaCache, Sub-cache[1] is KVCacheSimple.
            // We only extract the KV part; SSM state is handled separately.
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    let state = simple.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
                if let quantized = cacheList[i] as? QuantizedKVCache {
                    let unquantized = quantized.toUnquantized()
                    let state = unquantized.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    let state = tq.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
            }
            return nil
        }
        // MambaCache, ArraysCache, RotatingKVCache — no KV extraction
        return nil
    }
}

// MARK: - KV Cache Restoration

/// Restore per-layer KV tensors from cached blocks into a model's cache array.
///
/// Blocks only contain KV-bearing layers (SSM/RotatingKVCache layers are filtered
/// during storage). This function maps block layer indices to the KV-bearing
/// cache layers, skipping non-KV layers.
///
/// - Parameters:
///   - blocks: The cache blocks to restore from, ordered by sequence position.
///   - cache: The model's per-layer cache array to restore into.
/// - Returns: The total number of tokens restored across all blocks.
@discardableResult
public func restoreLayerData(from blocks: [CacheBlock], into cache: [any KVCache]) -> Int {
    guard let firstBlock = blocks.first, let firstData = firstBlock.cacheData else { return 0 }
    let numBlockLayers = firstData.count

    // Build mapping: block layer index → cache layer index
    // Only KVCacheSimple, QuantizedKVCache, TurboQuantKVCache, and CacheList-with-KV layers are KV-bearing
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            // Check if any sub-cache is KV-bearing
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Block layers should match KV-bearing cache layers
    guard numBlockLayers == kvCacheIndices.count else { return 0 }

    for (blockLayerIdx, cacheLayerIdx) in kvCacheIndices.enumerated() {
        var keySlices: [MLXArray] = []
        var valueSlices: [MLXArray] = []

        for block in blocks {
            guard let data = block.cacheData, blockLayerIdx < data.count,
                  let kv = data[blockLayerIdx] else { continue }
            keySlices.append(kv.keys)
            valueSlices.append(kv.values)
        }

        guard !keySlices.isEmpty else { continue }

        var restoredKeys = keySlices.count == 1 ? keySlices[0] : concatenated(keySlices, axis: 2)
        var restoredValues = valueSlices.count == 1 ? valueSlices[0] : concatenated(valueSlices, axis: 2)

        // Ensure restored KV matches bfloat16 (prevents dtype mismatch from stale
        // disk cache entries created before the universal bfloat16 conversion)
        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if let simple = cache[cacheLayerIdx] as? KVCacheSimple {
            simple.state = [restoredKeys, restoredValues]
        } else if let quantizedCache = cache[cacheLayerIdx] as? QuantizedKVCache {
            let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
            if let biases = qKeys.biases { stateArrays.append(biases) }
            stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
            if let biases = qValues.biases { stateArrays.append(biases) }
            quantizedCache.state = stateArrays
            quantizedCache.offset = restoredKeys.dim(2)
        } else if let tq = cache[cacheLayerIdx] as? TurboQuantKVCache {
            // Setting state transitions TQ to fill phase with the restored float KV.
            // The model will re-compress during the next generation cycle if needed.
            tq.state = [restoredKeys, restoredValues]
        } else if let cacheList = cache[cacheLayerIdx] as? CacheList {
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    simple.state = [restoredKeys, restoredValues]
                    break
                }
                if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                    let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                    if let biases = qKeys.biases { stateArrays.append(biases) }
                    stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                    if let biases = qValues.biases { stateArrays.append(biases) }
                    quantizedCache.state = stateArrays
                    quantizedCache.offset = restoredKeys.dim(2)
                    break
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    tq.state = [restoredKeys, restoredValues]
                    break
                }
            }
        }
    }

    let totalTokens = blocks.reduce(0) { $0 + $1.tokenCount }
    return totalTokens
}

// MARK: - SSM State Extraction

/// Extract SSM (MambaCache/ArraysCache) states from a model's cache array.
///
/// Returns the state arrays from each SSM layer. Non-SSM layers are skipped.
/// Used to populate ``SSMStateCache`` for hybrid model companion storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: All SSM state arrays, flattened across layers.
public func extractSSMStates(from cache: [any KVCache]) -> [MLXArray] {
    var states: [MLXArray] = []
    for layer in cache {
        if let mamba = layer as? MambaCache {
            // MambaCache.state returns [conv_state, hidden_state]
            states.append(contentsOf: mamba.state)
        } else if let arrays = layer as? ArraysCache {
            states.append(contentsOf: arrays.state)
        } else if let cacheList = layer as? CacheList {
            // Extract SSM sub-cache from composite layers
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    states.append(contentsOf: mamba.state)
                } else if let arrays = cacheList[i] as? ArraysCache {
                    states.append(contentsOf: arrays.state)
                }
            }
        }
    }
    return states
}

// MARK: - SSM State Restoration

/// Restore SSM states into a model's cache array.
///
/// The `states` array should match the output order of ``extractSSMStates(from:)``.
/// Each MambaCache consumes 2 state arrays (conv state + hidden state).
///
/// - Parameters:
///   - states: The SSM state arrays to restore.
///   - cache: The model's per-layer cache array to restore into.
public func restoreSSMStates(_ states: [MLXArray], into cache: [any KVCache]) {
    var stateIdx = 0
    for layer in cache {
        if let mamba = layer as? MambaCache {
            let existingCount = mamba.state.count
            if existingCount == 0 {
                // Fresh cache — MambaCache always has 2 slots (conv + hidden)
                let slotCount = 2
                if stateIdx + slotCount <= states.count {
                    mamba.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                        .map { $0[.ellipsis] }
                    stateIdx += slotCount
                }
            } else if stateIdx + existingCount <= states.count {
                mamba.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                    .map { $0[.ellipsis] }
                stateIdx += existingCount
            }
        } else if let arrays = layer as? ArraysCache {
            // ArraysCache (non-Mamba variant) — restore however many slots it has
            let existingCount = arrays.state.count
            if existingCount > 0, stateIdx + existingCount <= states.count {
                arrays.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                    .map { $0[.ellipsis] }
                stateIdx += existingCount
            }
        } else if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    let slotCount = 2
                    if stateIdx + slotCount <= states.count {
                        mamba.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                            .map { $0[.ellipsis] }
                        stateIdx += slotCount
                    }
                } else if let arrays = cacheList[i] as? ArraysCache {
                    let existingCount = arrays.state.count
                    if existingCount > 0, stateIdx + existingCount <= states.count {
                        arrays.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                            .map { $0[.ellipsis] }
                        stateIdx += existingCount
                    }
                }
            }
        }
    }
}

// MARK: - Disk Cache KV Restoration

/// Restore KV state (and, for hybrid models, Mamba SSM state) from a disk
/// cache arrays dictionary back into the model's per-layer cache array.
///
/// ## Format handling
///
/// Two on-disk formats are supported:
///
/// - **Version 2 (current)** — produced by `TQDiskSerializer.serialize`.
///   Each layer has an explicit `__layer_kind_{i}__` tag so mixed-kind
///   caches (hybrid attention + Mamba) round-trip correctly. Restored into
///   `cache[i]` by real cache position.
///
/// - **Version 1 (legacy block format)** — old entries using
///   `b{block}_l{layer}_keys/values`. Handled via the KV-only restoration
///   path. Hybrid models can't restore from v1 entries (the Mamba layers
///   were effectively corrupt), so they get a silent 0-token miss and
///   re-prefill. Acceptable — v1 entries expire naturally as new v2 entries
///   overwrite them.
///
/// - Parameters:
///   - arrays: The disk cache dictionary loaded via `DiskCache.fetch()`.
///   - cache: The model's per-layer KV cache array to restore into.
/// - Returns: The total number of tokens restored, measured from the first
///   attention layer's key tensor sequence dim, or `0` if nothing matched.
@discardableResult
public func restoreFromDiskArrays(_ arrays: [String: MLXArray], into cache: [any KVCache]) -> Int {
    let version = TQDiskSerializer.formatVersion(of: arrays)
    if version >= 2 {
        return restoreFromV2Arrays(arrays, into: cache)
    }
    return restoreFromLegacyArrays(arrays, into: cache)
}

/// Restore a format v2 disk dictionary into the model cache.
///
/// The serializer tags each layer with an authoritative `LayerKind`, so this
/// path can restore attention layers, Mamba SSM layers, and skipped layers
/// independently and by real cache index.
@discardableResult
private func restoreFromV2Arrays(
    _ arrays: [String: MLXArray],
    into cache: [any KVCache]
) -> Int {
    let indexed = TQDiskSerializer.deserializeIndexed(arrays)
    guard !indexed.isEmpty else { return 0 }

    var totalTokens = 0

    for entry in indexed {
        let i = entry.index
        guard i < cache.count else { continue }

        switch entry.data {
        case .standard(let kv):
            var keys = kv.keys
            var values = kv.values
            if keys.dtype == .float16 {
                keys = keys.asType(.bfloat16)
                values = values.asType(.bfloat16)
            }
            // Defensive shape guard — vmlx #68 / SmallVector out of range.
            // A 2D disk-stored layer (shape.count < 3) would crash on
            // `.dim(2)` below. Skip the whole restore as a clean miss so
            // the caller re-prefills instead of fatal-trapping. Belt &
            // suspenders on top of v2 layer-kind tagging.
            guard keys.shape.count >= 3, values.shape.count >= 3 else {
                FileHandle.standardError.write(Data(
                    "[disk cache] restore SKIPPED: layer \(i) has incompatible shape (k=\(keys.shape) v=\(values.shape)). Need >= 3D. Falling back to fresh prefill.\n".utf8))
                return 0
            }
            if totalTokens == 0 {
                totalTokens = keys.dim(2)
            }
            restoreKVLayer(keys: keys, values: values, into: cache[i])

        case .mamba(let comp):
            // Mamba state arrays are cumulative — no sequence dim to
            // measure, so they don't contribute to `totalTokens`. The
            // attention side already provides that number.
            restoreMambaLayer(comp, into: cache[i])

        case .tq(let comp):
            // Restore the compressed prefix into the existing
            // TurboQuantKVCache instance (or a TQ inside a CacheList).
            // If the runtime created a different cache class for this
            // layer we silently no-op and let the caller re-prefill —
            // matches the behavior for Mamba kind mismatches.
            restoreTQLayer(comp, into: cache[i])
            // TQ layers don't expose a sequence-dim tensor in the same way
            // as KV layers, so they can't drive `totalTokens`. The
            // attention sequence length is sourced from the .standard
            // entries; if a model is purely TQ-compressed we fall back to
            // the offset stored in the components.
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .qkv(let comp):
            // Restore quantized KV state. The runtime's QuantizedKVCache
            // must agree on group size and bit width (otherwise the qweight
            // shapes won't line up); the helper enforces that and silently
            // no-ops on mismatch so the caller re-prefills cleanly.
            restoreQKVLayer(comp, into: cache[i])
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .rotating(let comp):
            // Restore RotatingKVCache (sliding-window attention). Reseats
            // the ring buffer + 5-tuple metaState (keep, maxSize, step,
            // offset, idx) so the wrap position survives the restart.
            // SLIDING-1 (2026-04-15): closes the central skip in
            // CacheCoordinator.swift that previously dropped Gemma4 SWA,
            // Mistral4-with-maxKVSize, MiMoV2Flash, BaichuanM1.
            restoreRotatingLayer(comp, into: cache[i])
            if totalTokens == 0 {
                totalTokens = comp.offset
            }

        case .skip:
            // Cache type we don't know how to persist. No-op.
            continue
        }
    }

    return totalTokens
}

/// Legacy v1 block-format restore. KV-only. Hybrid-unsafe; if any Mamba
/// layer is in `cache`, the counts won't line up and this function returns
/// 0, forcing a re-prefill (which is correct, because old v1 entries never
/// captured Mamba state properly anyway).
@discardableResult
private func restoreFromLegacyArrays(
    _ arrays: [String: MLXArray],
    into cache: [any KVCache]
) -> Int {
    var kvByLayer: [Int: (keys: MLXArray, values: MLXArray)] = [:]

    if TQDiskSerializer.isTQNative(arrays) {
        // Legacy v1 "TQ-native" dict — flat deserialize, ignore tq entries.
        let layers = TQDiskSerializer.deserialize(arrays)
        for (i, layerData) in layers.enumerated() {
            if case .standard(let kv) = layerData {
                kvByLayer[i] = (keys: kv.keys, values: kv.values)
            }
        }
    } else {
        // Old block-layer format: b{block}_l{layer}_keys / b{block}_l{layer}_values.
        var layerBlocks: [Int: [(blockIdx: Int, keys: MLXArray, values: MLXArray)]] = [:]

        for (key, array) in arrays {
            guard key.hasSuffix("_keys") else { continue }
            let base = String(key.dropLast(5))
            let parts = base.split(separator: "_")
            guard parts.count == 2,
                  parts[0].hasPrefix("b"), parts[1].hasPrefix("l"),
                  let blockIdx = Int(parts[0].dropFirst()),
                  let layerIdx = Int(parts[1].dropFirst())
            else { continue }

            let valuesKey = "b\(blockIdx)_l\(layerIdx)_values"
            guard let valuesArray = arrays[valuesKey] else { continue }

            layerBlocks[layerIdx, default: []].append(
                (blockIdx: blockIdx, keys: array, values: valuesArray))
        }

        for (layerIdx, blocks) in layerBlocks {
            let sorted = blocks.sorted { $0.blockIdx < $1.blockIdx }
            let keySlices = sorted.map(\.keys)
            let valueSlices = sorted.map(\.values)

            let concatKeys = keySlices.count == 1
                ? keySlices[0] : concatenated(keySlices, axis: 2)
            let concatValues = valueSlices.count == 1
                ? valueSlices[0] : concatenated(valueSlices, axis: 2)

            kvByLayer[layerIdx] = (keys: concatKeys, values: concatValues)
        }
    }

    guard !kvByLayer.isEmpty else { return 0 }

    // Build mapping of KV-bearing cache layer indices (same logic as restoreLayerData).
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Legacy v1 entries have no layer-kind metadata, so hybrid models can't
    // be restored safely. Counts mismatch → abort and force re-prefill.
    let sortedLayers = kvByLayer.keys.sorted()
    guard sortedLayers.count == kvCacheIndices.count else { return 0 }

    var totalTokens = 0

    for (diskLayerIdx, cacheLayerIdx) in zip(sortedLayers, kvCacheIndices) {
        guard var (restoredKeys, restoredValues) = kvByLayer[diskLayerIdx] else { continue }

        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if totalTokens == 0 {
            totalTokens = restoredKeys.dim(2)
        }

        restoreKVLayer(keys: restoredKeys, values: restoredValues, into: cache[cacheLayerIdx])
    }

    return totalTokens
}

/// Helper: restore a pair of KV tensors into whatever KV-bearing cache
/// class `layer` happens to be. Mirrors the behavior of the original
/// legacy path but factored out so both v1 and v2 restores share it.
private func restoreKVLayer(
    keys restoredKeys: MLXArray,
    values restoredValues: MLXArray,
    into layer: any KVCache
) {
    if let simple = layer as? KVCacheSimple {
        simple.state = [restoredKeys, restoredValues]
    } else if let quantizedCache = layer as? QuantizedKVCache {
        let qKeys = quantized(
            restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
        let qValues = quantized(
            restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
        var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
        if let biases = qKeys.biases { stateArrays.append(biases) }
        stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
        if let biases = qValues.biases { stateArrays.append(biases) }
        quantizedCache.state = stateArrays
        quantizedCache.offset = restoredKeys.dim(2)
    } else if let tq = layer as? TurboQuantKVCache {
        tq.state = [restoredKeys, restoredValues]
    } else if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let simple = cacheList[i] as? KVCacheSimple {
                simple.state = [restoredKeys, restoredValues]
                return
            }
            if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                let qKeys = quantized(
                    restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                let qValues = quantized(
                    restoredValues,
                    groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                if let biases = qKeys.biases { stateArrays.append(biases) }
                stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                if let biases = qValues.biases { stateArrays.append(biases) }
                quantizedCache.state = stateArrays
                quantizedCache.offset = restoredKeys.dim(2)
                return
            }
            if let tq = cacheList[i] as? TurboQuantKVCache {
                tq.state = [restoredKeys, restoredValues]
                return
            }
        }
    }
}

/// Helper: restore quantized-KV state into a `QuantizedKVCache` layer.
/// Verifies group size + bit width match the runtime cache before
/// touching state — a mismatch means the model was reconfigured since the
/// disk entry was written and the qweight shapes will not align, so the
/// only safe behavior is to no-op and let the caller re-prefill.
private func restoreQKVLayer(
    _ comp: TQDiskSerializer.QKVLayerComponents,
    into layer: any KVCache
) {
    if let qkv = layer as? QuantizedKVCache {
        guard qkv.groupSize == comp.groupSize, qkv.bits == comp.bits else { return }
        qkv.state = comp.stateArrays
        qkv.offset = comp.offset
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let qkv = cacheList[i] as? QuantizedKVCache {
                guard qkv.groupSize == comp.groupSize, qkv.bits == comp.bits else { return }
                qkv.state = comp.stateArrays
                qkv.offset = comp.offset
                return
            }
        }
    }
}

/// Helper: restore TQ-compressed state into a `TurboQuantKVCache` layer
/// (or a `CacheList` containing one). Silently no-ops for other cache
/// classes — caller will re-prefill that layer via the standard path.
private func restoreTQLayer(
    _ comp: TQDiskSerializer.TQLayerComponents,
    into layer: any KVCache
) {
    if let tq = layer as? TurboQuantKVCache {
        tq.restoreCompressed(
            encodedKeys: comp.encodedKeys,
            encodedValues: comp.encodedValues,
            sourceOffset: comp.offset
        )
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let tq = cacheList[i] as? TurboQuantKVCache {
                tq.restoreCompressed(
                    encodedKeys: comp.encodedKeys,
                    encodedValues: comp.encodedValues,
                    sourceOffset: comp.offset
                )
                return
            }
        }
    }
}

/// Helper: restore RotatingKVCache state (ring buffer + metaState) into a
/// `RotatingKVCache` layer (or a `CacheList` containing one). The ring
/// buffer keys/values are reseated via `state =` and the wrap position
/// via `metaState = [keep, maxSize, step, offset, idx]` so generation
/// continues at exactly the same idx pointer. Silently no-ops on type
/// mismatch — caller falls back to re-prefill on this layer.
private func restoreRotatingLayer(
    _ comp: TQDiskSerializer.RotatingLayerComponents,
    into layer: any KVCache
) {
    func apply(_ rot: RotatingKVCache) {
        rot.state = [comp.keys, comp.values]
        rot.metaState = [
            String(comp.keep),
            String(comp.maxSize),
            String(comp.step),
            String(comp.offset),
            String(comp.idx),
        ]
    }
    if let rot = layer as? RotatingKVCache {
        apply(rot)
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let rot = cacheList[i] as? RotatingKVCache {
                apply(rot)
                return
            }
        }
    }
}

/// Helper: restore Mamba SSM state into a `MambaCache` layer (or a
/// `CacheList` containing one). Silently no-ops for other cache classes,
/// which can happen if the serialized model layout has drifted from the
/// current runtime.
private func restoreMambaLayer(
    _ comp: TQDiskSerializer.MambaLayerComponents,
    into layer: any KVCache
) {
    if let mamba = layer as? MambaCache {
        mamba.state = [comp.state0, comp.state1]
        mamba.offset = comp.offset
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let mamba = cacheList[i] as? MambaCache {
                mamba.state = [comp.state0, comp.state1]
                mamba.offset = comp.offset
                return
            }
        }
    }
}
