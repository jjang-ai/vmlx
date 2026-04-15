// Copyright © 2025 Osaurus & JANG. All rights reserved.
// Unified L2 disk serialization for JANG cache system.
//
// Format version 2 (2026-04-13): adds explicit per-layer kind tags so
// hybrid models (attention + MambaCache) round-trip correctly, and folds
// SSM companion state into the same dictionary so it persists across
// process restart. Format version 1 is still readable as a KV-only fallback.

import Foundation
import MLX

/// Serializes cache layers into a flat `[String: MLXArray]` dictionary for
/// disk persistence via `DiskCache`. Handles three layer kinds (TurboQuant,
/// standard KV, Mamba SSM) plus an optional companion list of SSM states
/// for hybrid models.
///
/// ## Format versioning
///
/// The serialized dict always contains `__jang_cache_format_version__` (a
/// scalar int32). Version 2 is the current format. Version 1 lacks the
/// explicit kind tag and is readable but write-only-legacy.
///
/// ## Key layout (version 2)
///
/// Per-layer kind tag:
///
///     __layer_kind_{i}__        — int32 scalar, one of `LayerKind.rawValue`
///
/// Per-layer payload by kind:
///
///     TQ-compressed:
///       tq_{i}_ck_indices       — EncodedKeys.indicesPacked (uint32)
///       tq_{i}_ck_qjl           — EncodedKeys.qjlPacked (uint32)
///       tq_{i}_ck_res_norms     — EncodedKeys.residualNorms (float16)
///       tq_{i}_ck_vec_norms     — EncodedKeys.vectorNorms (float16)
///       tq_{i}_ck_sink          — EncodedKeys.sinkData (float16, optional)
///       __tq_{i}_ck_shape__     — original compressed key shape (int32 array)
///       __tq_{i}_ck_index_bits__— key index bits (int32 scalar)
///       __tq_{i}_ck_seed__      — key encoding seed (int32 scalar)
///       cv_*                    — same for values (no `qjl`)
///
///     KVCacheSimple / TQ in fill phase:
///       kv_{i}_keys             — keys tensor
///       kv_{i}_values           — values tensor
///
///     MambaCache:
///       mamba_{i}_state0        — first state array (conv state)
///       mamba_{i}_state1        — second state array (ssm state)
///       __mamba_{i}_offset__    — int32 scalar for Mamba offset (restore hint)
///
///     Unknown / unsupported (e.g. QuantizedKVCache):
///       (no payload keys; kind tag alone records that the layer was skipped)
///
/// Companion SSM state list (hybrid models):
///
///     __ssm_count__             — int32 scalar; number of SSM entries that follow
///     ssm_{k}                   — one tensor per SSM entry, 0..<count
///
/// Legacy markers kept for old readers:
///
///     __tq_native_marker__      — presence indicates "dict came from this module"
public enum TQDiskSerializer {

    // MARK: - Format version

    /// Current on-disk format version. Bumped whenever the key schema
    /// changes in a way that old readers can't parse.
    public static let currentFormatVersion: Int32 = 2

    /// Key that holds the format version scalar.
    public static let formatVersionKey = "__jang_cache_format_version__"

    /// Legacy marker. Pre-v2 code checked this to decide whether to parse at
    /// all. Still written for back-compat with any external consumer.
    public static let legacyMarkerKey = "__tq_native_marker__"

    // MARK: - Layer kind

    /// Identifies what a given layer in the serialized dict represents.
    public enum LayerKind: Int32, Sendable {
        /// Never written. Sentinel for "metadata was missing".
        case unknown = 0
        /// `TurboQuantKVCache` in compressed phase.
        case tq = 1
        /// `KVCacheSimple`, `TurboQuantKVCache` in fill phase, or any other
        /// BaseKVCache subclass whose state is a pair of tensors treated
        /// as keys/values.
        case kv = 2
        /// `MambaCache` (conv state + ssm state pair).
        case mamba = 3
        /// `QuantizedKVCache`: 4 or 6 state arrays (qweight + scales [+biases]
        /// for both keys and values), plus group size / bit width metadata.
        case qkv = 5
        /// `RotatingKVCache` (sliding-window attention — Gemma4 SWA, Mistral4
        /// with maxKVSize, MiMoV2Flash, BaichuanM1 CacheList wrapper). Stores
        /// the full ring buffer (`state[0]=keys`, `state[1]=values`) plus the
        /// 5-tuple metaState `(keep, maxSize, step, offset, idx)` so the
        /// restored cache picks up exactly where it left off, preserving
        /// wrap-around context. Added 2026-04-15 — closes the central skip
        /// in CacheCoordinator.swift:424.
        case rotating = 6
        /// `CacheList` wrapper (BaichuanM1 hybrid, FalconH1 sibling-mix).
        /// Outer layer has sub-caches that each carry their own LayerKind
        /// tag at the compound key `__layer_kind_{i}_{sub}__`. Deserialize
        /// walks `__cachelist_{i}_count__` to discover sub-indices and
        /// routes each to its kind-specific handler. Added 2026-04-15
        /// (F-G2) — closes the silent-drop trap where the fallthrough
        /// at the bottom of the if-else ladder tagged CacheList as
        /// `.skip` and lost all inner state.
        case cachelist = 7
        /// Cache type we don't know how to persist. On restore, treated as
        /// a forced miss for the affected layer only.
        case skip = 4
    }

    private static func kindKey(for layer: Int) -> String {
        "__layer_kind_\(layer)__"
    }

    private static func kindArray(_ kind: LayerKind) -> MLXArray {
        // 1-element 1D — see metaInt32 below for why scalars don't round-trip.
        MLXArray([kind.rawValue])
    }

    /// Build a 1-element 1D Int32 metadata array. All scalar metadata
    /// values (offsets, counts, bit widths, format version) MUST go
    /// through this — never `metaInt32(Int32(x))` which produces a
    /// 0-dim scalar that doesn't survive the safetensors round-trip
    /// in a multi-key file. Observed live on Qwen3.5-VL-4B JANG: 32
    /// serialized `__layer_kind_*__` tags + 24 `__mamba_*_offset__`
    /// scalars came back as ~16 + garbage offsets after load+save.
    /// 1D arrays of length 1 round-trip cleanly.
    private static func metaInt32(_ value: Int32) -> MLXArray {
        MLXArray([value])
    }

    /// Read a 1-element Int32 metadata array. Tolerates both 0-dim
    /// (legacy entries written before the round-trip fix) and 1D
    /// shape-[1] (current).
    private static func readMetaInt32(_ arr: MLXArray) -> Int32 {
        if arr.shape.isEmpty {
            // Legacy 0-dim — Int32-typed scalar.
            return arr.item(Int32.self)
        }
        return arr[0].item(Int32.self)
    }


    // MARK: - Detection

    /// Check if a cache layer is a `TurboQuantKVCache` in compressed phase.
    public static func isTQCompressed(_ cache: any KVCache) -> Bool {
        guard let tq = cache as? TurboQuantKVCache else { return false }
        return tq.phase == .compressed
    }

    /// Check if a loaded dictionary was written by this module.
    ///
    /// True for both format version 1 (legacy TQ-only) and version 2 (the
    /// current unified format). Consumers should additionally inspect
    /// `formatVersion(of:)` if they need to distinguish the two.
    public static func isTQNative(_ arrays: [String: MLXArray]) -> Bool {
        arrays.keys.contains(legacyMarkerKey) || arrays.keys.contains(formatVersionKey)
    }

    /// Read the `__jang_cache_format_version__` scalar from a loaded dict.
    ///
    /// Returns `1` if the key is missing but the legacy TQ marker is present
    /// (pre-v2 entries). Returns `0` for dicts that don't come from this
    /// module at all.
    public static func formatVersion(of arrays: [String: MLXArray]) -> Int32 {
        if let v = arrays[formatVersionKey] {
            return readMetaInt32(v)
        }
        if arrays.keys.contains(legacyMarkerKey) {
            return 1
        }
        return 0
    }

    // MARK: - Serialize

    /// Serialize cache layers (and optional SSM companion state) into a flat
    /// dictionary suitable for safetensors persistence.
    ///
    /// - Parameters:
    ///   - cache: Array of per-layer KV caches from the model.
    ///   - ssmStates: Optional SSM companion state for hybrid models. These
    ///     are a sequential list with no layer index of their own — they're
    ///     stored in order and returned in the same order on deserialize.
    /// - Returns: Flat dictionary ready for `DiskCache.store()`.
    public static func serialize(
        cache: [any KVCache],
        ssmStates: [MLXArray]? = nil
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        result[formatVersionKey] = metaInt32(currentFormatVersion)
        result[legacyMarkerKey] = MLXArray([Int32(1)])

        for (i, layer) in cache.enumerated() {
            if let tq = layer as? TurboQuantKVCache, tq.phase == .compressed {
                serializeTQLayer(tq, index: i, into: &result)
                result[kindKey(for: i)] = kindArray(.tq)
            } else if let mamba = layer as? MambaCache {
                serializeMambaLayer(mamba, index: i, into: &result)
                result[kindKey(for: i)] = kindArray(.mamba)
            } else if let rot = layer as? RotatingKVCache {
                serializeRotatingLayer(rot, index: i, into: &result)
                // serializeRotatingLayer sets the kind tag itself so it can
                // mark empty (pre-prefill) caches as `.skip`.
            } else if let qkv = layer as? QuantizedKVCache {
                serializeQKVLayer(qkv, index: i, into: &result)
                // serializeQKVLayer sets the kind tag itself so it can mark
                // empty caches as `.skip` instead of `.qkv`.
            } else if layer is KVCacheSimple || layer is TurboQuantKVCache {
                // KVCacheSimple always, plus TurboQuantKVCache in fill phase.
                // Both expose `state` as a 2-array [keys, values] pair.
                let state = layer.state
                if state.count >= 2 {
                    result["kv_\(i)_keys"] = state[0]
                    result["kv_\(i)_values"] = state[1]
                    result[kindKey(for: i)] = kindArray(.kv)
                } else {
                    // Fill-phase layer with empty state — record as skip so
                    // restore knows not to fall through to KV decode.
                    result[kindKey(for: i)] = kindArray(.skip)
                }
            } else if let cacheList = layer as? CacheList {
                // Audit F-G2 (P0): BaichuanM1 wraps (MambaCache, KVCacheSimple)
                // pairs inside a CacheList per layer. The old fallthrough to
                // `.skip` (line below) lost ALL inner state on disk persist
                // — Mamba + KV silently dropped, validator rejected the
                // entry on restore, forced full re-prefill every turn.
                //
                // Fix: walk the sub-caches and emit a separate per-sub
                // LayerKind tag using the compound index `{i}_{sub}`. The
                // outer layer is tagged `.cachelist` so the deserializer
                // knows to walk sub-indices rather than expecting a
                // single KV/Mamba/rotating payload at index `i`.
                result[kindKey(for: i)] = kindArray(.cachelist)
                result["__cachelist_\(i)_count__"] = metaInt32(Int32(cacheList.count))
                for sub in 0 ..< cacheList.count {
                    let inner = cacheList[sub]
                    let compoundKey = "\(i)_\(sub)"
                    let subKindKey = "__layer_kind_\(compoundKey)__"
                    if let mamba = inner as? MambaCache {
                        serializeMambaLayer(mamba, compoundIndex: compoundKey, into: &result)
                        result[subKindKey] = kindArray(.mamba)
                    } else if inner is KVCacheSimple || inner is TurboQuantKVCache {
                        let state = inner.state
                        if state.count >= 2 {
                            result["kv_\(compoundKey)_keys"] = state[0]
                            result["kv_\(compoundKey)_values"] = state[1]
                            result[subKindKey] = kindArray(.kv)
                        } else {
                            result[subKindKey] = kindArray(.skip)
                        }
                    } else {
                        // Rotating / QKV / TQ inside CacheList: rare, no
                        // known production model ships this pattern.
                        // Tag skip — inner state is lost but the outer
                        // `.cachelist` tag keeps the validator happy.
                        result[subKindKey] = kindArray(.skip)
                    }
                }
            } else {
                // Unknown cache class. Record an explicit skip so restore
                // doesn't silently fall through to KV.
                result[kindKey(for: i)] = kindArray(.skip)
            }
        }

        // Companion SSM states — ordered list. Folded into the same dict
        // so one safetensors write captures everything needed to resume.
        if let ssmStates, !ssmStates.isEmpty {
            result["__ssm_count__"] = metaInt32(Int32(ssmStates.count))
            for (k, s) in ssmStates.enumerated() {
                result["ssm_\(k)"] = s
            }
        }

        return result
    }

    /// Serialize a single TQ-compressed layer's encoded data.
    private static func serializeTQLayer(
        _ tq: TurboQuantKVCache,
        index i: Int,
        into result: inout [String: MLXArray]
    ) {
        // --- Compressed keys ---
        if let ck = tq.compressedKeys {
            result["tq_\(i)_ck_indices"] = ck.indicesPacked
            result["tq_\(i)_ck_qjl"] = ck.qjlPacked
            result["tq_\(i)_ck_res_norms"] = ck.residualNorms
            result["tq_\(i)_ck_vec_norms"] = ck.vectorNorms

            if let sink = ck.sinkData {
                result["tq_\(i)_ck_sink"] = sink
            }

            result["__tq_\(i)_ck_shape__"] = MLXArray(ck.shape.map { Int32($0) })
            result["__tq_\(i)_ck_index_bits__"] = metaInt32(Int32(ck.indexBits))
            result["__tq_\(i)_ck_seed__"] = metaInt32(Int32(ck.seed))
        }

        // --- Compressed values ---
        if let cv = tq.compressedValues {
            result["tq_\(i)_cv_indices"] = cv.indicesPacked
            result["tq_\(i)_cv_norms"] = cv.vectorNorms

            if let sink = cv.sinkData {
                result["tq_\(i)_cv_sink"] = sink
            }

            result["__tq_\(i)_cv_shape__"] = MLXArray(cv.shape.map { Int32($0) })
            result["__tq_\(i)_cv_index_bits__"] = metaInt32(Int32(cv.indexBits))
            result["__tq_\(i)_cv_seed__"] = metaInt32(Int32(cv.seed))
        }

        // --- TQ-cache-level metadata (needed for in-place restore) ---
        // The token offset that the compressed prefix represents. Without
        // this the restored cache would report offset 0 and the next
        // generate() call would clobber the prefix on its first append.
        result["__tq_\(i)_offset__"] = metaInt32(Int32(tq.offset))
    }

    /// Serialize a single MambaCache layer's state (conv state + ssm state).
    private static func serializeMambaLayer(
        _ mamba: MambaCache,
        index i: Int,
        into result: inout [String: MLXArray]
    ) {
        serializeMambaLayer(mamba, compoundIndex: "\(i)", into: &result)
    }

    /// Compound-index variant used by the CacheList walker (F-G2). The
    /// `compoundIndex` is `"{outer}_{sub}"` so nested Mamba layers inside
    /// a CacheList wrapper can be disambiguated.
    private static func serializeMambaLayer(
        _ mamba: MambaCache,
        compoundIndex key: String,
        into result: inout [String: MLXArray]
    ) {
        let state = mamba.state
        guard state.count >= 2 else { return }
        result["mamba_\(key)_state0"] = state[0]
        result["mamba_\(key)_state1"] = state[1]
        result["__mamba_\(key)_offset__"] = metaInt32(Int32(mamba.offset))
    }

    /// Serialize a single RotatingKVCache layer (sliding-window attention).
    /// Captures both the ring buffer (`state[0]=keys`, `state[1]=values`)
    /// AND the metaState 5-tuple `(keep, maxSize, step, offset, idx)` so
    /// the restored cache picks up at exactly the same wrap position.
    ///
    /// Sets `__layer_kind_{i}__` to `.rotating` on success or `.skip` if
    /// the cache hasn't been prefilled yet (state is empty).
    private static func serializeRotatingLayer(
        _ rot: RotatingKVCache,
        index i: Int,
        into result: inout [String: MLXArray]
    ) {
        let state = rot.state
        guard state.count == 2 else {
            // Pre-prefill — nothing useful to persist.
            result[kindKey(for: i)] = kindArray(.skip)
            return
        }
        result["rot_\(i)_keys"] = state[0]
        result["rot_\(i)_values"] = state[1]
        // metaState is `[keep, maxSize, step, offset, idx]` per
        // RotatingKVCache.swift:614. Pack into a single Int32 array so we
        // round-trip cleanly through safetensors (1D arrays survive,
        // 0-dim scalars don't — see metaInt32 comment).
        let meta = rot.metaState
        if meta.count == 5,
           let keep = Int32(meta[0]),
           let maxSize = Int32(meta[1]),
           let step = Int32(meta[2]),
           let offset = Int32(meta[3]),
           let idx = Int32(meta[4])
        {
            result["__rot_\(i)_meta__"] = MLXArray([keep, maxSize, step, offset, idx])
            result[kindKey(for: i)] = kindArray(.rotating)
        } else {
            // metaState shape changed unexpectedly — refuse to persist.
            result[kindKey(for: i)] = kindArray(.skip)
        }
    }

    /// Serialize a single QuantizedKVCache layer's state (4 or 6 arrays
    /// covering qweight/scales/[biases] for both keys and values), plus
    /// group size, bit width and offset metadata so the restore path can
    /// rebuild the cache without prefilling.
    ///
    /// Sets `__layer_kind_{i}__` to `.qkv` on success or `.skip` if the
    /// cache hasn't been initialised yet (state has fewer than 4 arrays).
    private static func serializeQKVLayer(
        _ qkv: QuantizedKVCache,
        index i: Int,
        into result: inout [String: MLXArray]
    ) {
        let state = qkv.state
        guard state.count == 4 || state.count == 6 else {
            // Cache is uninitialised (no prefill yet) — nothing useful to
            // persist. Mark as skip so restore doesn't treat the absence
            // of qkv keys as a structural error.
            result[kindKey(for: i)] = kindArray(.skip)
            return
        }
        for (k, arr) in state.enumerated() {
            result["qkv_\(i)_\(k)"] = arr
        }
        result["__qkv_\(i)_count__"] = metaInt32(Int32(state.count))
        result["__qkv_\(i)_offset__"] = metaInt32(Int32(qkv.offset))
        result["__qkv_\(i)_group_size__"] = metaInt32(Int32(qkv.groupSize))
        result["__qkv_\(i)_bits__"] = metaInt32(Int32(qkv.bits))
        result[kindKey(for: i)] = kindArray(.qkv)
    }

    // MARK: - Deserialize types

    /// Parsed TQ components for a single attention layer.
    public struct TQLayerComponents {
        public let encodedKeys: EncodedKeys
        public let encodedValues: EncodedValues
        /// Token offset the compressed prefix represents. Falls back to 0
        /// for legacy entries that pre-date the offset metadata key.
        public let offset: Int
    }

    /// Standard (non-TQ) KV data for a single attention layer.
    public struct KVLayerComponents {
        public let keys: MLXArray
        public let values: MLXArray
    }

    /// Mamba SSM state for a single hybrid layer.
    public struct MambaLayerComponents {
        public let state0: MLXArray
        public let state1: MLXArray
        public let offset: Int
    }

    /// RotatingKVCache state for a single sliding-window attention layer.
    /// `keys` and `values` are the full ring buffer; the 5-tuple meta
    /// fields restore the wrap position so generation continues from
    /// exactly where the dump happened.
    public struct RotatingLayerComponents {
        public let keys: MLXArray
        public let values: MLXArray
        public let keep: Int
        public let maxSize: Int
        public let step: Int
        public let offset: Int
        public let idx: Int
    }

    /// QuantizedKVCache state for a single attention layer.
    public struct QKVLayerComponents {
        /// 4 or 6 arrays in the same order as `QuantizedKVCache.state`:
        /// `[keys.qweight, keys.scales, keys.biases?, values.qweight,
        /// values.scales, values.biases?]`.
        public let stateArrays: [MLXArray]
        public let offset: Int
        public let groupSize: Int
        public let bits: Int
    }

    /// One sub-cache of a CacheList (F-G2). Carried inside
    /// `CacheListComponents.subs` so the restore path can reseat Mamba
    /// state + KV tensors into the right nested slot.
    public enum CacheListSub {
        case mamba(MambaLayerComponents)
        case standard(KVLayerComponents)
        case skip
    }

    /// CacheList wrapper state (F-G2). BaichuanM1 and similar hybrid
    /// models wrap `(MambaCache, KVCacheSimple)` pairs inside a CacheList
    /// per layer; this struct captures each sub-cache's kind + payload.
    public struct CacheListComponents {
        public let subs: [CacheListSub]
    }

    /// Result of deserializing one cache layer from a dict.
    public enum LayerData {
        case tq(TQLayerComponents)
        case standard(KVLayerComponents)
        case mamba(MambaLayerComponents)
        case qkv(QKVLayerComponents)
        case rotating(RotatingLayerComponents)
        case cachelist(CacheListComponents)
        /// Layer was serialized as `.skip` (cache type we don't persist).
        case skip
    }

    /// One layer's position + kind + payload. Layers are indexed by their
    /// real cache position so restore can route them into `cache[index]`
    /// without any filtering.
    public struct IndexedLayerData {
        public let index: Int
        public let data: LayerData
    }

    // MARK: - Deserialize

    /// Deserialize a dict into per-layer components.
    ///
    /// Returns layers indexed by their original cache position. For format
    /// version 2 dicts the `__layer_kind_{i}__` tags are authoritative; for
    /// version 1 dicts the function falls back to the tq/kv prefix scan used
    /// by pre-v2 code, which is fine for homogeneous attention-only models.
    ///
    /// - Parameter arrays: Dictionary loaded from safetensors via `DiskCache.fetch()`.
    /// - Returns: Array of `IndexedLayerData` ordered by ascending layer index.
    public static func deserializeIndexed(_ arrays: [String: MLXArray]) -> [IndexedLayerData] {
        let version = formatVersion(of: arrays)
        if version >= 2 {
            return deserializeV2(arrays)
        } else {
            return deserializeV1(arrays)
        }
    }

    /// Legacy flat deserialize that discards layer indices.
    ///
    /// Kept for callers that previously consumed `[LayerData]` without
    /// position info. New code should use `deserializeIndexed` instead.
    public static func deserialize(_ arrays: [String: MLXArray]) -> [LayerData] {
        return deserializeIndexed(arrays).map(\.data)
    }

    /// Parse companion SSM state from a deserialized dict.
    ///
    /// - Returns: The ordered `[MLXArray]` captured at serialize time, or
    ///   `nil` if the dict has no SSM entries.
    public static func ssmStates(from arrays: [String: MLXArray]) -> [MLXArray]? {
        guard let countArr = arrays["__ssm_count__"] else { return nil }
        let count = Int(readMetaInt32(countArr))
        guard count > 0 else { return nil }
        var out: [MLXArray] = []
        out.reserveCapacity(count)
        for k in 0..<count {
            guard let s = arrays["ssm_\(k)"] else { return nil }
            out.append(s)
        }
        return out
    }

    // MARK: - Deserialize V2

    private static func deserializeV2(_ arrays: [String: MLXArray]) -> [IndexedLayerData] {
        // Discover every layer index that has a kind tag.
        var kindsByIndex: [Int: LayerKind] = [:]
        for key in arrays.keys {
            guard key.hasPrefix("__layer_kind_") && key.hasSuffix("__") else { continue }
            let body = key.dropFirst("__layer_kind_".count).dropLast(2)
            guard let idx = Int(body) else { continue }
            guard let kindArr = arrays[key] else { continue }
            let raw = readMetaInt32(kindArr)
            if let kind = LayerKind(rawValue: raw) {
                kindsByIndex[idx] = kind
            }
        }

        var out: [IndexedLayerData] = []
        for i in kindsByIndex.keys.sorted() {
            guard let kind = kindsByIndex[i] else { continue }
            switch kind {
            case .tq:
                if let components = deserializeTQLayer(index: i, from: arrays) {
                    out.append(IndexedLayerData(index: i, data: .tq(components)))
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .kv:
                if let keys = arrays["kv_\(i)_keys"],
                   let values = arrays["kv_\(i)_values"]
                {
                    out.append(
                        IndexedLayerData(
                            index: i,
                            data: .standard(KVLayerComponents(keys: keys, values: values))
                        )
                    )
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .mamba:
                if let s0 = arrays["mamba_\(i)_state0"],
                   let s1 = arrays["mamba_\(i)_state1"]
                {
                    let offset: Int
                    if let offArr = arrays["__mamba_\(i)_offset__"] {
                        offset = Int(readMetaInt32(offArr))
                    } else {
                        offset = 0
                    }
                    out.append(
                        IndexedLayerData(
                            index: i,
                            data: .mamba(
                                MambaLayerComponents(
                                    state0: s0,
                                    state1: s1,
                                    offset: offset
                                )
                            )
                        )
                    )
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .qkv:
                if let comp = deserializeQKVLayer(index: i, from: arrays) {
                    out.append(IndexedLayerData(index: i, data: .qkv(comp)))
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .rotating:
                if let comp = deserializeRotatingLayer(index: i, from: arrays) {
                    out.append(IndexedLayerData(index: i, data: .rotating(comp)))
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .cachelist:
                if let comp = deserializeCacheListLayer(index: i, from: arrays) {
                    out.append(IndexedLayerData(index: i, data: .cachelist(comp)))
                } else {
                    out.append(IndexedLayerData(index: i, data: .skip))
                }
            case .skip, .unknown:
                out.append(IndexedLayerData(index: i, data: .skip))
            }
        }
        return out
    }

    // MARK: - Deserialize V1 (legacy)

    private static func deserializeV1(_ arrays: [String: MLXArray]) -> [IndexedLayerData] {
        var tqIndices = Set<Int>()
        var kvIndices = Set<Int>()

        for key in arrays.keys {
            if key.hasPrefix("tq_"), let idx = parseLayerIndex(from: key, prefix: "tq_") {
                tqIndices.insert(idx)
            } else if key.hasPrefix("kv_"), let idx = parseLayerIndex(from: key, prefix: "kv_") {
                kvIndices.insert(idx)
            }
        }

        let allIndices = tqIndices.union(kvIndices).sorted()
        var out: [IndexedLayerData] = []

        for i in allIndices {
            if tqIndices.contains(i) {
                if let components = deserializeTQLayer(index: i, from: arrays) {
                    out.append(IndexedLayerData(index: i, data: .tq(components)))
                }
            } else if kvIndices.contains(i) {
                if let keys = arrays["kv_\(i)_keys"],
                   let values = arrays["kv_\(i)_values"]
                {
                    out.append(
                        IndexedLayerData(
                            index: i,
                            data: .standard(KVLayerComponents(keys: keys, values: values))
                        )
                    )
                }
            }
        }

        return out
    }

    /// Deserialize a single TQ layer's encoded data from the flat dict.
    private static func deserializeTQLayer(
        index i: Int,
        from arrays: [String: MLXArray]
    ) -> TQLayerComponents? {
        // --- Keys ---
        guard let ckIndices = arrays["tq_\(i)_ck_indices"],
              let ckQjl = arrays["tq_\(i)_ck_qjl"],
              let ckResNorms = arrays["tq_\(i)_ck_res_norms"],
              let ckVecNorms = arrays["tq_\(i)_ck_vec_norms"],
              let ckShapeArr = arrays["__tq_\(i)_ck_shape__"],
              let ckIndexBitsArr = arrays["__tq_\(i)_ck_index_bits__"],
              let ckSeedArr = arrays["__tq_\(i)_ck_seed__"]
        else {
            return nil
        }

        // --- Values ---
        guard let cvIndices = arrays["tq_\(i)_cv_indices"],
              let cvNorms = arrays["tq_\(i)_cv_norms"],
              let cvShapeArr = arrays["__tq_\(i)_cv_shape__"],
              let cvIndexBitsArr = arrays["__tq_\(i)_cv_index_bits__"],
              let cvSeedArr = arrays["__tq_\(i)_cv_seed__"]
        else {
            return nil
        }

        let ckShape = ckShapeArr.asArray(Int32.self).map { Int($0) }
        let ckIndexBits = Int(readMetaInt32(ckIndexBitsArr))
        let ckSeed = Int(readMetaInt32(ckSeedArr))

        let cvShape = cvShapeArr.asArray(Int32.self).map { Int($0) }
        let cvIndexBits = Int(readMetaInt32(cvIndexBitsArr))
        let cvSeed = Int(readMetaInt32(cvSeedArr))

        let ckSink = arrays["tq_\(i)_ck_sink"]
        let cvSink = arrays["tq_\(i)_cv_sink"]

        let encodedKeys = EncodedKeys(
            indicesPacked: ckIndices,
            qjlPacked: ckQjl,
            residualNorms: ckResNorms,
            vectorNorms: ckVecNorms,
            shape: ckShape,
            indexBits: ckIndexBits,
            seed: ckSeed,
            sinkData: ckSink
        )

        let encodedValues = EncodedValues(
            indicesPacked: cvIndices,
            vectorNorms: cvNorms,
            shape: cvShape,
            indexBits: cvIndexBits,
            seed: cvSeed,
            sinkData: cvSink
        )

        let offset: Int
        if let offArr = arrays["__tq_\(i)_offset__"] {
            offset = Int(readMetaInt32(offArr))
        } else {
            // Legacy entries that pre-date __tq_*_offset__ default to the
            // sequence length implied by the compressed key shape.
            offset = ckShape.count >= 3 ? ckShape[2] : 0
        }

        return TQLayerComponents(
            encodedKeys: encodedKeys,
            encodedValues: encodedValues,
            offset: offset
        )
    }

    /// Deserialize a single QuantizedKVCache layer from the flat dict.
    ///
    /// The state arrays are stored under `qkv_{i}_0` … `qkv_{i}_{count-1}`
    /// in the same order as `QuantizedKVCache.state`. Returns nil if any
    /// of the metadata or payload keys are missing — caller turns that
    /// into a `.skip` LayerData.
    private static func deserializeQKVLayer(
        index i: Int,
        from arrays: [String: MLXArray]
    ) -> QKVLayerComponents? {
        guard let countArr = arrays["__qkv_\(i)_count__"],
              let offArr = arrays["__qkv_\(i)_offset__"],
              let gsArr = arrays["__qkv_\(i)_group_size__"],
              let bitsArr = arrays["__qkv_\(i)_bits__"]
        else {
            return nil
        }
        let count = Int(readMetaInt32(countArr))
        guard count == 4 || count == 6 else { return nil }

        var stateArrays: [MLXArray] = []
        stateArrays.reserveCapacity(count)
        for k in 0..<count {
            guard let arr = arrays["qkv_\(i)_\(k)"] else { return nil }
            stateArrays.append(arr)
        }

        return QKVLayerComponents(
            stateArrays: stateArrays,
            offset: Int(readMetaInt32(offArr)),
            groupSize: Int(readMetaInt32(gsArr)),
            bits: Int(readMetaInt32(bitsArr))
        )
    }

    /// Deserialize a single RotatingKVCache layer from the v2 dict.
    /// Reads `rot_{i}_keys`, `rot_{i}_values`, and the 5-element
    /// `__rot_{i}_meta__` tuple. Returns nil when any field is missing
    /// (caller emits `.skip`).
    private static func deserializeRotatingLayer(
        index i: Int,
        from arrays: [String: MLXArray]
    ) -> RotatingLayerComponents? {
        guard let keys = arrays["rot_\(i)_keys"],
              let values = arrays["rot_\(i)_values"],
              let metaArr = arrays["__rot_\(i)_meta__"]
        else {
            return nil
        }
        // metaArr is a 1D Int32 array of length 5.
        guard !metaArr.shape.isEmpty, metaArr.shape[0] == 5 else { return nil }
        let m = metaArr.asArray(Int32.self)
        guard m.count == 5 else { return nil }
        return RotatingLayerComponents(
            keys: keys,
            values: values,
            keep: Int(m[0]),
            maxSize: Int(m[1]),
            step: Int(m[2]),
            offset: Int(m[3]),
            idx: Int(m[4])
        )
    }

    /// Deserialize a CacheList wrapper layer (F-G2). Reads
    /// `__cachelist_{i}_count__` to discover how many sub-caches the outer
    /// layer holds, then per-sub reads the compound-keyed kind tag
    /// (`__layer_kind_{i}_{sub}__`) and routes each sub to its kind-specific
    /// payload readers. Returns nil only when the count metadata is missing
    /// (treat as `.skip`); individual sub-cache misses degrade to
    /// `CacheListSub.skip` so one bad sub doesn't void the rest.
    private static func deserializeCacheListLayer(
        index i: Int,
        from arrays: [String: MLXArray]
    ) -> CacheListComponents? {
        guard let countArr = arrays["__cachelist_\(i)_count__"] else {
            return nil
        }
        let count = Int(readMetaInt32(countArr))
        guard count > 0 else { return nil }

        var subs: [CacheListSub] = []
        subs.reserveCapacity(count)
        for sub in 0 ..< count {
            let compoundKey = "\(i)_\(sub)"
            let subKindKey = "__layer_kind_\(compoundKey)__"
            guard let kindArr = arrays[subKindKey] else {
                subs.append(.skip)
                continue
            }
            let raw = readMetaInt32(kindArr)
            guard let kind = LayerKind(rawValue: raw) else {
                subs.append(.skip)
                continue
            }
            switch kind {
            case .mamba:
                if let s0 = arrays["mamba_\(compoundKey)_state0"],
                   let s1 = arrays["mamba_\(compoundKey)_state1"]
                {
                    let offset: Int
                    if let offArr = arrays["__mamba_\(compoundKey)_offset__"] {
                        offset = Int(readMetaInt32(offArr))
                    } else {
                        offset = 0
                    }
                    subs.append(.mamba(MambaLayerComponents(
                        state0: s0, state1: s1, offset: offset
                    )))
                } else {
                    subs.append(.skip)
                }
            case .kv:
                if let keys = arrays["kv_\(compoundKey)_keys"],
                   let values = arrays["kv_\(compoundKey)_values"]
                {
                    subs.append(.standard(KVLayerComponents(
                        keys: keys, values: values
                    )))
                } else {
                    subs.append(.skip)
                }
            default:
                // Rotating / TQ / QKV inside CacheList not emitted by
                // serialize — keep parallel with the serialize skip path.
                subs.append(.skip)
            }
        }
        return CacheListComponents(subs: subs)
    }

    // MARK: - Helpers

    /// Extract the layer index from a key like "tq_42_ck_indices" or "kv_7_keys".
    private static func parseLayerIndex(from key: String, prefix: String) -> Int? {
        let remainder = key.dropFirst(prefix.count)
        guard let underscoreIdx = remainder.firstIndex(of: "_") else { return nil }
        let indexStr = String(remainder[remainder.startIndex..<underscoreIdx])
        return Int(indexStr)
    }
}
