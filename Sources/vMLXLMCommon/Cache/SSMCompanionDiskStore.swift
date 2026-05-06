// Copyright © 2024 Apple Inc.
//
// §441 — SSMCompanionDiskStore (native port of Python vmlx_engine #110).
//
// In-memory `SSMStateCache` (companion cache for hybrid Mamba+attention
// models — NemotronH / Cascade-2 / Nemotron-Omni / Qwen3.5-A3B / Jamba)
// is fast but volatile: a process restart re-prefills the prompt from
// scratch even if the user's system prompt + first turn haven't
// changed. For stable-system-prompt workloads (Terminal mode with a
// fixed scope-flagged agent prompt, server-side chat with one canonical
// system message) this re-prefill costs O(prompt_len) on every cold
// start.
//
// This store mirrors `DiskCache.swift`'s pattern: hash-keyed
// safetensors files under a flat directory, JSON sidecar for
// `is_complete` flag (parity with Python's `(states, is_complete)`
// tuple semantics from `vmlx_engine/utils/ssm_companion_cache.py`).
//
// Storage format per entry:
//   <cacheDir>/ssm-<sha>.safetensors    — N MLX arrays keyed `state_0`…`state_N-1`
//   <cacheDir>/ssm-<sha>.json           — metadata { is_complete, num_states, model_key }
//
// Cache key derivation matches the in-memory `SSMStateCache`:
//   key = SHA-256( modelKey + ":" + tokens[..<boundary].joined(",") )
//
// Concurrency: `OSAllocatedUnfairLock` for index mutation, IO outside
// the lock (parity with `DiskCache.swift:138`).
//
// NOT WIRED INTO `SSMStateCache.fetch/store` BY DEFAULT this iter —
// the primitive lands first so the wiring is a 5-LOC change in
// `SSMStateCache.swift` (write-through on store; fall-through on miss).
// Default OFF behind `enableSSMCompanionDiskCache` setting once wired.

import CryptoKit
import Foundation
import MLX
import os

/// Disk-backed extension to the in-memory `SSMStateCache`. See header
/// comment for storage format + concurrency model.
public final class SSMCompanionDiskStore: @unchecked Sendable {

    // MARK: - Properties

    private let lock = OSAllocatedUnfairLock()
    private let cacheDir: URL
    private let modelKey: String?
    /// Maximum total disk bytes before LRU eviction. 0 = unlimited
    /// (parity with `DiskCache`'s default — eviction is a follow-up).
    private let maxBytes: Int

    // MARK: - Initialization

    public init(cacheDir: URL, modelKey: String? = nil, maxBytes: Int = 0) throws {
        self.cacheDir = cacheDir
        self.modelKey = modelKey
        self.maxBytes = maxBytes
        try FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
    }

    // MARK: - Public API

    /// Persist SSM layer states for a given token prefix. Mirrors
    /// `SSMStateCache.store(ssmStates:tokens:boundary:)` with the
    /// addition of an `isComplete` flag (parity with Python tuple).
    ///
    /// Iter 143: `mediaSalt` is now threaded through to the disk key
    /// so VL/Omni paths don't collide with text-only prefixes that
    /// happen to share a token prefix. Previously hardcoded to `nil`
    /// here, which silently aliased text-only and audio/image variants
    /// of the same prefix on disk → wrong SSM state restored on cold
    /// start for hybrid-VL or Nemotron-Omni audio sessions.
    public func store(
        ssmStates: [MLXArray],
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil,
        isComplete: Bool = true
    ) throws {
        guard !ssmStates.isEmpty, boundary > 0, boundary <= tokens.count else { return }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)

        // Pre-realize on calling thread — same rationale as
        // DiskCache.swift:114-116. GPU work must complete before the
        // safetensors writer can read the storage. MLX's tensor
        // realization (NOT script eval — this is `mlx.core.eval`).
        MLX.eval(ssmStates)

        // Materialize key→array dict expected by `save(arrays:metadata:url:)`.
        // Ordering preserved by `state_<idx>` keys; `extractSSMStates`
        // returns layers in cache order, so the round-trip is positional.
        var arrays: [String: MLXArray] = [:]
        for (i, arr) in ssmStates.enumerated() {
            arrays["state_\(i)"] = arr
        }

        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        // Sync write — same rationale as DiskCache.swift:122-130.
        // Async dispatch races with SIGTERM on short-lived sessions,
        // leaving zero-byte files. Costs ~ms on already-realized arrays.
        try save(arrays: arrays, metadata: ["format": "mlx"], url: safetensorsURL)

        // JSON sidecar for is_complete flag + num_states.
        let sidecar: [String: Any] = [
            "is_complete": isComplete,
            "num_states": ssmStates.count,
            "model_key": modelKey ?? "",
            "boundary": boundary,
        ]
        let sidecarData = try JSONSerialization.data(
            withJSONObject: sidecar, options: [.sortedKeys])
        try sidecarData.write(to: sidecarURL, options: [.atomic])
    }

    /// Look up SSM layer states for a given token prefix + boundary.
    /// Returns nil on miss / corruption / decode failure.
    ///
    /// Iter 143: `mediaSalt` mirror of the store-side change. Pass the
    /// same salt the L1 store consumed (typically derived from
    /// `computeMediaSalt(images:videos:audios:)`).
    public func fetch(
        tokens: [Int],
        boundary: Int,
        mediaSalt: String? = nil
    ) -> SSMStateCache.FetchResult? {
        guard boundary > 0, boundary <= tokens.count else { return nil }
        let key = Self.keyFor(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey)
        let safetensorsURL = self.safetensorsURL(for: key)
        let sidecarURL = self.sidecarURL(for: key)

        guard FileManager.default.fileExists(atPath: safetensorsURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path)
        else { return nil }

        // Decode sidecar first — cheap, validates the entry shape.
        guard let sidecarData = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONSerialization.jsonObject(with: sidecarData)
                as? [String: Any],
              let isComplete = sidecar["is_complete"] as? Bool,
              let numStates = sidecar["num_states"] as? Int,
              numStates > 0
        else { return nil }

        // Decode safetensors. A failed deserialize is most often a
        // truncated file (process killed mid-write, rare on sync IO
        // but possible). Treat as miss.
        guard let arraysAndMeta = try? loadArraysAndMetadata(url: safetensorsURL)
        else { return nil }
        let arrays = arraysAndMeta.0

        // Reassemble in positional order. Bail if any `state_<idx>` is
        // missing — partial entries are unsafe to extend per the Python
        // `(states, is_complete)` contract.
        var states: [MLXArray] = []
        states.reserveCapacity(numStates)
        for i in 0 ..< numStates {
            guard let arr = arrays["state_\(i)"] else { return nil }
            states.append(arr)
        }

        return SSMStateCache.FetchResult(states: states, isComplete: isComplete)
    }

    /// Remove all entries for a given model key. Called on model
    /// unload so subsequent loads don't see stale state. No-op if the
    /// directory is empty.
    public func clear() {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: nil) else { return }
        for url in entries {
            let name = url.lastPathComponent
            if name.hasPrefix("ssm-") {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    // MARK: - Helpers

    private func safetensorsURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).safetensors")
    }

    private func sidecarURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("ssm-\(hash).json")
    }

    /// SHA-256 hash. P0-2 (2026-04-30): converged with `SSMStateCache.makeKey`
    /// AND with Python's `ssm_companion_cache._key`. Previous formula used
    /// `:` separator + Int32 LE bytes, which collided with NEITHER. Result
    /// was a write-only L2: every disk fetch missed L1's hash, so backfill
    /// silently failed (`AUDIT-SSM-WARMPASS-PARITY.md` §1). New formula
    /// delegates to `SSMStateCache.makeKey` so the two sites cannot drift.
    ///
    /// Iter 143: `mediaSalt` is now a real parameter (was hardcoded to
    /// nil — flagged "P1 follow-up" in the prior comment). Threading it
    /// through closes the L2 disk collision class for VL/Omni hybrid
    /// sessions: text-only and audio/image variants of the same token
    /// prefix used to share a key on disk → wrong SSM state restored.
    /// The 3-arg form (no mediaSalt) is preserved as a thin wrapper so
    /// existing tests + text-only callers don't need updating.
    public static func keyFor(
        tokens: [Int], boundary: Int,
        mediaSalt: String? = nil, modelKey: String?
    ) -> String {
        SSMStateCache.makeKey(
            tokens: tokens, boundary: boundary,
            mediaSalt: mediaSalt, modelKey: modelKey
        )
    }
}
