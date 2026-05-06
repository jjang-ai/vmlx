// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import os

// MARK: - BatchQuantize

/// Centralised KV-quantisation hook for ``BatchEngine``.
///
/// Stage 0 of the batch-engine blockers effort. Closes
/// `BatchEnginePlan.Blocker.kvQuantization` for `.turboQuant` requests by
/// wiring the existing single-sequence ``maybeQuantizeKVCache`` helper into
/// the engine's per-slot lifecycle.
///
/// ## Supported modes under batched decode
///
/// - **`.turboQuant(keyBits, valueBits)`** — fully supported. After prefill,
///   `KVCacheSimple` layers are swapped for ``TurboQuantKVCache`` via
///   ``TurboQuantKVCache/fromSimpleCache(_:keyBits:valueBits:sinkTokens:)``.
///   `TurboQuantKVCache.update()` returns plain `(MLXArray, MLXArray)` matching
///   `KVCacheSimple`'s shape contract, so the existing ``BatchKVCache`` split/
///   pad/stack logic wraps TQ slot caches natively — no dedicated subclass
///   needed at Stage 0 (see `stepBatchDecode` in `BatchEngine.swift`).
///
/// - **`.affine(bits, groupSize)` or legacy `kvBits`** — *not* supported.
///   ``QuantizedKVCache.update(keys:values:)`` traps at runtime; models must
///   route attention through ``quantizedScaledDotProductAttention`` which
///   requires per-layer quantised tuples. Threading those through the batched
///   path is deferred to a future spec. Affine / `kvBits` requests continue to
///   produce correct output but run with float KV — the knob is a silent
///   no-op under batch today. ``wrapNewCacheIfNeeded(slotID:parameters:)``
///   logs a warning so the gap is observable.
///
/// - **`.none`** — no-op.
///
/// ## Call sites
///
/// - ``wrapNewCacheIfNeeded(slotID:parameters:)`` — called from
///   ``BatchEngine.admitPendingRequests``. Warns-only. No mutation today;
///   reserved as the landing site for future affine + AWQ support so callers
///   don't need to change.
/// - ``maybeCompress(cache:parameters:)`` — called at the end of prefill and
///   after each batched decode step. Delegates to ``maybeQuantizeKVCache``
///   which already handles threshold checking, already-quantised skip, and
///   type gating (`RotatingKVCache` / `MambaCache` / `CacheList` / already-TQ
///   layers are skipped automatically).
public enum BatchQuantize {

    private static let logger = Logger(subsystem: "vmlx", category: "BatchQuantize")

    /// Called from the admission path. Emits a warning for request
    /// configurations whose KV-quant mode is not yet supported under batch.
    /// No mutation — the slot cache is untouched.
    ///
    /// Exposed `public` for direct unit testing. Callers outside the engine
    /// should not invoke this directly — it's wired into
    /// ``BatchEngine.admitPendingRequests``.
    public static func wrapNewCacheIfNeeded(
        slotID: BatchRequestID,
        parameters: GenerateParameters
    ) {
        switch parameters.kvMode {
        case .turboQuant:
            // Supported. Actual compression happens in `maybeCompress` after
            // prefill once offset > quantizedKVStart + 8 (TQ minimum threshold).
            break

        case .affine:
            logger.warning(
                "Slot \(slotID.description, privacy: .public): affine KV quantization (kvMode: .affine) is not supported under batched decode. Request will run with float KV. Use .turboQuant for memory-efficient batched decode."
            )

        case .none:
            if parameters.kvBits != nil {
                logger.warning(
                    "Slot \(slotID.description, privacy: .public): legacy kvBits is not supported under batched decode. Request will run with float KV. Use kvMode: .turboQuant(...) for memory-efficient batched decode."
                )
            }
        }
    }

    /// Called after prefill completes and after each batched decode step.
    ///
    /// Stage 0 routes **only** `.turboQuant` through the shared
    /// ``maybeQuantizeKVCache`` helper. Affine (`.affine` / legacy `kvBits`)
    /// intentionally does **not** run here: `QuantizedKVCache` traps on
    /// `update(keys:values:)` and requires every attention site to route
    /// through `quantizedScaledDotProductAttention` on quantized tuples —
    /// threading those through the batched path is out of scope for Stage 0
    /// (see §2 of the batch-engine-blockers spec). Affine requests run with
    /// float KV and a one-time admission warning (see
    /// ``wrapNewCacheIfNeeded(slotID:parameters:)``).
    ///
    /// For TQ mode, delegates to ``maybeQuantizeKVCache`` which:
    /// - Checks the first `KVCacheSimple` layer's offset against
    ///   `max(quantizedKVStart, 8)` (TQ minimum threshold).
    /// - Skips if any layer is already `TurboQuantKVCache`.
    /// - Preserves `RotatingKVCache`, `MambaCache`, `CacheList` layers
    ///   (hybrid SSM models have mixed cache arrays — only the KV layers
    ///   compress).
    ///
    /// Calling this multiple times is safe: the helper's internal guard
    /// prevents re-quantisation once a layer has been converted.
    ///
    /// Exposed `public` for direct unit testing.
    public static func maybeCompress(
        cache: inout [KVCache],
        parameters: GenerateParameters
    ) {
        // Gate on mode: only TQ compression runs under batch today.
        guard case .turboQuant = parameters.kvMode else { return }

        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: nil,  // force TQ-only path; never legacy affine
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            kvMode: parameters.kvMode
        )
    }
}
