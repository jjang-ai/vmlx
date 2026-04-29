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
    public func store(
        ssmStates: [MLXArray],
        tokens: [Int],
        boundary: Int,
        isComplete: Bool = true
    ) throws {
        guard !ssmStates.isEmpty, boundary > 0, boundary <= tokens.count else { return }
        let key = Self.keyFor(tokens: tokens, boundary: boundary, modelKey: modelKey)

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
    public func fetch(
        tokens: [Int],
        boundary: Int
    ) -> SSMStateCache.FetchResult? {
        guard boundary > 0, boundary <= tokens.count else { return nil }
        let key = Self.keyFor(tokens: tokens, boundary: boundary, modelKey: modelKey)
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

    /// SHA-256 hash of `modelKey + ":" + tokens[..<boundary]`.
    /// Identity-aligned with `SSMStateCache`'s in-memory key so the
    /// disk store can be a plain extension of the LRU tier (write-
    /// through on store, fall-through on miss).
    public static func keyFor(tokens: [Int], boundary: Int, modelKey: String?) -> String {
        var hasher = SHA256()
        if let mk = modelKey, !mk.isEmpty {
            hasher.update(data: Data(mk.utf8))
            hasher.update(data: Data([0x3a]))  // ":"
        }
        // Encode tokens as little-endian Int32s. Same encoding the
        // in-memory cache uses (`SSMStateCache.keyFor`).
        var buf = [UInt8](repeating: 0, count: boundary * 4)
        for i in 0 ..< boundary {
            let t = Int32(truncatingIfNeeded: tokens[i])
            buf[i * 4 + 0] = UInt8(truncatingIfNeeded: t & 0xff)
            buf[i * 4 + 1] = UInt8(truncatingIfNeeded: (t >> 8) & 0xff)
            buf[i * 4 + 2] = UInt8(truncatingIfNeeded: (t >> 16) & 0xff)
            buf[i * 4 + 3] = UInt8(truncatingIfNeeded: (t >> 24) & 0xff)
        }
        hasher.update(data: Data(buf))
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
