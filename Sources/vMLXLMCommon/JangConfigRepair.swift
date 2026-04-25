// SPDX-License-Identifier: Apache-2.0
//
// §411 — One-time bundle config repair for JANG/JANGTQ bundles whose
// `jang_config.json` or `config.json::quantization` block disagrees
// with on-disk safetensors shapes.
//
// Background
// ----------
// Older JANGTQ converter runs emitted bundles with a uniform `bits=2`
// (or similar) declaration in `jang_config.json::quantization`, even
// though the actual safetensors stored 8-bit attention / embed /
// lm_head (per CRITICAL-tier classify rules) and 2-bit routed experts
// only. Loaders that trusted the declaration silently picked the wrong
// dequant (bits=4/gs=64 by default-fallback) → embed output dim
// doubled → attention shape collapsed → garbage or crash.
//
// §410 fixes this at runtime by making shape inference authoritative
// (preference order: (8,32) → (8,64) → … → (2,128) → 3/6-bit fallback).
// That covers EVERY load, including bundles already on disk.
//
// This file complements §410 with a *one-time on-disk repair*: when a
// bundle is loaded and the runtime inferer finds layer(s) where the
// declared (bits, gs) disagrees with shape-truth by more than a small
// tolerance, we write a backup of the original config and replace it
// with the inferred values. This is opt-in via env (off by default to
// keep load-path read-only by default) and idempotent.
//
// Why both runtime + on-disk?
//   - Runtime inferer keeps every load correct without filesystem writes.
//   - On-disk repair makes the bundle self-describing again so
//     downstream tooling (Python loaders, model-card scrapers, third-
//     party analyzers) sees the right values too.
//
// Behavior
// --------
// `JangConfigRepair.repairIfBad(at:inferredPerLayer:)` is called from
// `loadWeights` after `inferPerLayerQuantization`. It:
//
//   1. Reads the bundle's `config.json`. If absent, return unchanged.
//   2. Parses `quantization` block (if present). Builds a map of
//      declared (bits, gs) per leaf path.
//   3. For each path in `inferredPerLayer`, compares declared vs
//      inferred. Counts disagreements.
//   4. If disagreements exceed `mismatchThreshold` (default 1 — any
//      mismatch is a bug worth fixing), writes:
//        - `config.json.bak` — original, only on first repair.
//        - `config.json` — new, with corrected per-layer entries.
//   5. Sets a marker file `.jang-config-repaired-v1` to avoid re-
//      processing on subsequent loads.
//
// Opt-in: `VMLX_REPAIR_BAD_JANG_CONFIG=1` env or `repairBadJangConfig`
// flag in `LoadOptions` (TODO when settings wire is needed).
//
// Safety: writes are atomic (write to `config.json.tmp`, then rename).
// Backup is kept indefinitely. Bundle is unchanged if no mismatch.

import Foundation

public enum JangConfigRepair {

    /// Top-level entrypoint. No-op when the env flag is unset OR the
    /// bundle has no `config.json` OR no mismatches were detected OR
    /// the marker file is present.
    public static func repairIfBad(
        at modelDirectory: URL,
        inferredPerLayer: [String: BaseConfiguration.QuantizationOption]
    ) {
        let env = ProcessInfo.processInfo.environment
        // Default OFF — modifying user bundles is invasive enough that
        // we want explicit opt-in. The runtime inferer (§410) already
        // makes loads correct without on-disk changes.
        guard env["VMLX_REPAIR_BAD_JANG_CONFIG"] == "1" else { return }

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let markerURL = modelDirectory.appendingPathComponent(".jang-config-repaired-v1")
        let backupURL = modelDirectory.appendingPathComponent("config.json.bak")

        // Idempotent — skip if already repaired.
        if FileManager.default.fileExists(atPath: markerURL.path) { return }

        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return
        }

        guard let mismatches = collectMismatches(
            configJSON: json, inferred: inferredPerLayer
        ), !mismatches.isEmpty else {
            // No mismatches detected — nothing to repair.
            return
        }

        // Write backup (skip if it already exists).
        if !FileManager.default.fileExists(atPath: backupURL.path) {
            try? FileManager.default.copyItem(at: configURL, to: backupURL)
        }

        // Build patched config and write atomically.
        var patched = json
        var quantBlock = (json["quantization"] as? [String: Any]) ?? [:]
        for (path, (bits, gs)) in mismatches {
            quantBlock[path] = [
                "bits": bits,
                "group_size": gs,
                "mode": "affine",
            ]
        }
        patched["quantization"] = quantBlock

        let tmpURL = modelDirectory.appendingPathComponent("config.json.tmp")
        if let out = try? JSONSerialization.data(
            withJSONObject: patched, options: [.prettyPrinted, .sortedKeys]
        ) {
            try? out.write(to: tmpURL, options: .atomic)
            _ = try? FileManager.default.replaceItemAt(configURL, withItemAt: tmpURL)
        }

        // Stamp marker. Body is informational; presence is what the
        // idempotency check reads.
        let markerBody =
            "vmlx-swift JangConfigRepair v1 — repaired \(mismatches.count) per-layer "
          + "(bits, group_size) entries against on-disk safetensors shapes. "
          + "Original config saved at config.json.bak. "
          + "Timestamp: \(ISO8601DateFormatter().string(from: Date()))\n"
        try? markerBody.write(to: markerURL, atomically: true, encoding: .utf8)
    }

    /// Compare declared per-layer quantization (from config.json's
    /// `quantization` block, top-level + per-key overrides) against the
    /// shape-inferred per-layer map. Returns a `[path: (bits, gs)]`
    /// dict of layers where the on-disk shape disagrees with the
    /// declaration.
    ///
    /// Returns `nil` if `config.json` has no `quantization` block at
    /// all (we don't synthesize a quant block from scratch — that's
    /// outside this repair's scope; the runtime inferer handles it).
    static func collectMismatches(
        configJSON: [String: Any],
        inferred: [String: BaseConfiguration.QuantizationOption]
    ) -> [String: (bits: Int, gs: Int)]? {
        guard let quantBlock = configJSON["quantization"] as? [String: Any] else {
            return nil
        }
        let topLevelBits = quantBlock["bits"] as? Int
        let topLevelGs = quantBlock["group_size"] as? Int

        var out: [String: (bits: Int, gs: Int)] = [:]
        for (path, option) in inferred {
            // Only `.quantize(...)` entries can mismatch (`.skip` means
            // the layer wasn't quantized on disk, which is consistent
            // with config absence).
            guard case .quantize(let q) = option else { continue }
            let inferredBits = q.bits
            let inferredGs = q.groupSize

            // Resolve declared values for this path. Two sources:
            //   - Per-key override `quantization.<path>.bits` etc.
            //   - Top-level `quantization.bits` / `quantization.group_size`.
            // Per-key wins where present.
            var declaredBits: Int? = topLevelBits
            var declaredGs: Int? = topLevelGs
            if let perKey = quantBlock[path] as? [String: Any] {
                if let b = perKey["bits"] as? Int { declaredBits = b }
                if let g = perKey["group_size"] as? Int { declaredGs = g }
            }
            // Path key variants — try the bare key + `model.`-stripped
            // form, since converters write either convention.
            if let stripped = path.split(separator: ".").dropFirst().joined(separator: ".") as String?,
               !stripped.isEmpty,
               let perKey = quantBlock[stripped] as? [String: Any]
            {
                if let b = perKey["bits"] as? Int { declaredBits = b }
                if let g = perKey["group_size"] as? Int { declaredGs = g }
            }

            // No declaration at all = no mismatch (this layer just
            // inherits top-level which we already handled above).
            // If neither top-level nor per-key resolved, skip.
            guard let db = declaredBits, let dg = declaredGs else { continue }

            if db != inferredBits || dg != inferredGs {
                out[path] = (inferredBits, inferredGs)
            }
        }
        return out
    }
}
