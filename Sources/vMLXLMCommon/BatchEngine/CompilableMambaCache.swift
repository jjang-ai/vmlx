// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 4 — `CompilableMambaCache`.
//
// The Stage 4 probe (iter 8, `MambaCacheCompileProbeTests.swift`) showed
// that `MambaCache` / `ArraysCache` can't be captured by `MLX.compile(...)`
// as-is. Attempting to do so trips a fatal:
//
//   MLX/ErrorHandler.swift:343: Fatal error: [compile]
//     Attempting to compile a function with uncaptured inputs is not
//     allowed. at mlx-c/mlx/c/closure.cpp:104
//
// Root cause: `ArraysCache` stores state as `[MLXArray?]` — an array of
// optionals. `innerState()` returns `cache.compactMap { $0 }`, creating
// a fresh Swift array each call. The compile tracer can't flatten that
// indirection into stable state inputs, so reads/writes inside the
// traced closure reference "uncaptured" state.
//
// This subclass fixes the trace-compatibility issue by:
//  1. Storing conv state and hidden state as two DIRECT `MLXArray?`
//     properties instead of an `[MLXArray?]` array. The tracer sees
//     each as a first-class stateful tensor.
//  2. Overriding `subscript(index:)` to route slot-0/slot-1 reads to
//     these direct properties. Existing model code that does
//     `cache[0] = newValue` keeps working.
//  3. `innerState()` returns `[convState, hiddenState]` directly (both
//     must be non-nil at that point — promoted caches are always
//     populated).
//  4. All mutations go through `_updateInternal` so the tracer
//     preserves object identity across calls.
//
// Initialise via ``init(from:)`` on a populated `MambaCache` produced
// by the model's prefill path.

import Foundation
import MLX

/// Compile-traceable specialisation of ``MambaCache``.
///
/// `MambaCache` uses 2 state slots (size: 2) — by convention,
/// `cache[0]` is the conv state and `cache[1]` is the hidden/SSM state.
/// This subclass exposes both as direct MLXArray properties so the
/// compile tracer can capture them.
///
/// ## Usage
///
/// ```swift
/// // After prefill has populated the MambaCache:
/// let compilable = CompilableMambaCache(from: mambaCache)
/// // compile-safe: innerState() returns [convState, hiddenState]
/// ```
///
/// ## Caveat
///
/// `MambaCache` is used for SSM-only layers. Hybrid models (Qwen3.5,
/// Qwen3Next/GDN, LFM2, Jamba, GraniteMoeHybrid, NemotronH) interleave
/// Mamba layers with attention layers — their `[KVCache]` is
/// heterogeneous, which `CacheFamily.classify` returns `.heterogeneous`
/// for. The compile path specialises on shape; heterogeneous caches
/// can't share one trace. Full hybrid-model compile would require
/// either grouping layers by type in separate traces OR a
/// Stage-4-level refactor that's out of this subclass's scope.
///
/// Today: `CompilableMambaCache` unblocks pure-SSM testing and stands
/// ready as the state-storage fix for whichever higher-level approach
/// ships first.
public final class CompilableMambaCache: MambaCache, @unchecked Sendable {

    /// Convolution state for the SSM layer (`cache[0]`).
    public var convStateArray: MLXArray?

    /// Hidden / SSM state for the SSM layer (`cache[1]`).
    public var hiddenStateArray: MLXArray?

    // MARK: - Init

    /// Public direct initialiser — primarily for testing. The two state
    /// slots start nil; first call to `subscript(index:)=` populates them.
    public override init(leftPadding: [Int]? = nil) {
        self.convStateArray = nil
        self.hiddenStateArray = nil
        super.init(leftPadding: leftPadding)
    }

    /// Promote an existing populated ``MambaCache`` to a compile-traceable
    /// variant.
    ///
    /// - Parameter mamba: Source cache, typically produced by a model's
    ///   prefill of Mamba / GDN layers.
    public convenience init(from mamba: MambaCache) {
        self.init(leftPadding: nil)

        // Copy offset + leftPadding from source.
        self.offset = mamba.offset
        self.leftPadding = mamba.leftPadding

        // Copy state slots via the public subscript. `MambaCache`'s
        // state setter invokes the parent `ArraysCache.subscript` which
        // stores to its `[MLXArray?]` array — we override subscript here
        // so the writes land in our direct properties instead.
        if let conv = mamba[0] { self[0] = conv }
        if let hidden = mamba[1] { self[1] = hidden }
    }

    // MARK: - Subscript override

    /// Route slot reads/writes to the direct `convStateArray` /
    /// `hiddenStateArray` properties. Existing model code that does
    /// `cache[0] = ...` or `cache[1] = ...` keeps working — the
    /// difference is that our storage is stable across compile
    /// invocations because the two MLXArray properties persist as
    /// direct members rather than living inside a Swift array the
    /// tracer can't flatten.
    ///
    /// Writes use `_updateInternal` when possible to preserve object
    /// identity — matching the discipline that keeps compile traces
    /// coherent. First-time writes (when the slot is nil) bind the
    /// property directly; subsequent writes reuse the same underlying
    /// MLXArray via `_updateInternal`.
    public override subscript(index: Int) -> MLXArray? {
        get {
            switch index {
            case 0: return convStateArray
            case 1: return hiddenStateArray
            default:
                fatalError("CompilableMambaCache: index out of range \(index) (valid: 0 or 1)")
            }
        }
        set {
            switch index {
            case 0:
                if let existing = convStateArray, let newValue {
                    existing._updateInternal(newValue)
                } else {
                    convStateArray = newValue
                }
            case 1:
                if let existing = hiddenStateArray, let newValue {
                    existing._updateInternal(newValue)
                } else {
                    hiddenStateArray = newValue
                }
            default:
                fatalError("CompilableMambaCache: index out of range \(index) (valid: 0 or 1)")
            }
        }
    }

    // MARK: - innerState override

    /// Expose ONLY the direct MLXArray properties, in a predictable
    /// order `[convStateArray, hiddenStateArray]`. Compile's state-input
    /// capture needs stable identity and order — the parent's
    /// `compactMap` over `[MLXArray?]` doesn't give either.
    ///
    /// If a promoted cache has a nil slot (unusual — prefill should
    /// always populate both), fall back to the populated ones in order.
    public override func innerState() -> [MLXArray] {
        var out: [MLXArray] = []
        if let conv = convStateArray { out.append(conv) }
        if let hidden = hiddenStateArray { out.append(hidden) }
        return out
    }

    // MARK: - State override

    /// Match the parent's state contract but route through our direct
    /// properties. Called by the cache coordinator on restore. The
    /// setter validates exactly-2 entries because `MambaCache` always
    /// has 2 state slots.
    public override var state: [MLXArray] {
        get {
            var out: [MLXArray] = []
            if let conv = convStateArray { out.append(conv) }
            if let hidden = hiddenStateArray { out.append(hidden) }
            return out
        }
        set {
            precondition(
                newValue.count == 2 || newValue.isEmpty,
                "CompilableMambaCache.state must have exactly 2 entries (conv, hidden) or 0"
            )
            if newValue.isEmpty {
                self.convStateArray = nil
                self.hiddenStateArray = nil
            } else {
                self.convStateArray = newValue[0]
                self.hiddenStateArray = newValue[1]
            }
        }
    }

    // MARK: - Copy

    public override func copy() -> any KVCache {
        let new = CompilableMambaCache()
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        if let conv = convStateArray { new.convStateArray = conv[.ellipsis] }
        if let hidden = hiddenStateArray { new.hiddenStateArray = hidden[.ellipsis] }
        return new
    }
}
