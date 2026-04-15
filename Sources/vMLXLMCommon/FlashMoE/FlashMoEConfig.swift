// SPDX-License-Identifier: Apache-2.0
//
// Flash MoE — SSD-streamed expert loading for massive MoE models.
//
// Enables 35B–397B MoE models to run on Macs with limited RAM by keeping
// only a hot slot-bank of expert weight sets in memory and streaming the
// rest from disk on demand. Mirrors the Python
// `vmlx_engine/flash_moe_config.py` surface exactly so the CLI, settings
// store, and remote API can round-trip config dictionaries.
//
// Phase 1 (this file + ExpertIndex + SlotBankCache + ExpertWeightSet):
// config, cache, and scanner foundation. Phase 2 wires the actual
// layer-swap shims into LLMModel and VLMModel load paths.

import Foundation

/// Runtime configuration for Flash MoE SSD expert streaming.
public struct FlashMoEConfig: Sendable, Equatable {

    /// Prefetch strategy for warming the slot-bank cache.
    public enum Prefetch: String, Sendable, Equatable {
        /// Load on demand only. Lowest memory, highest first-token latency.
        case none
        /// Warm the cache with recently-used experts across layers.
        /// Higher memory but smoother tail latency on repeated prompts.
        case temporal
    }

    /// Whether Flash MoE is active. Default false (opt-in).
    public var enabled: Bool

    /// Maximum number of `ExpertWeightSet`s cached in RAM.
    /// Higher = more cache hits but more RAM. Typical range 64–256.
    /// Must be ≥ 1.
    public var slotBankSize: Int

    /// Prefetching strategy. See `Prefetch`.
    public var prefetch: Prefetch

    /// Number of parallel I/O operations for expert loading.
    /// Higher = faster loading but more I/O pressure. Must be ≥ 1.
    public var cacheIOSplit: Int

    /// Optional absolute path to a pre-built expert index JSON file.
    /// When `nil`, the index is built lazily by scanning safetensors
    /// headers at load time — which is fast (milliseconds) because
    /// only the header bytes are read, not weight data.
    public var expertIndexPath: String?

    public init(
        enabled: Bool = false,
        slotBankSize: Int = 64,
        prefetch: Prefetch = .none,
        cacheIOSplit: Int = 4,
        expertIndexPath: String? = nil
    ) {
        precondition(slotBankSize >= 1, "slotBankSize must be >= 1")
        precondition(cacheIOSplit >= 1, "cacheIOSplit must be >= 1")
        self.enabled = enabled
        self.slotBankSize = slotBankSize
        self.prefetch = prefetch
        self.cacheIOSplit = cacheIOSplit
        self.expertIndexPath = expertIndexPath
    }

    /// Decode from a `[String: Any]` shaped like the Python
    /// `FlashMoEConfig.from_dict` input. Unknown keys are ignored.
    /// Accepts both `slot_bank_size` and legacy `slot_bank`.
    public static func from(dictionary d: [String: Any]) -> FlashMoEConfig {
        let slot = (d["slot_bank_size"] as? Int)
            ?? (d["slot_bank"] as? Int)
            ?? 64
        let prefetchRaw = (d["prefetch"] as? String) ?? "none"
        let prefetch = Prefetch(rawValue: prefetchRaw) ?? .none
        return FlashMoEConfig(
            enabled: (d["enabled"] as? Bool) ?? false,
            slotBankSize: max(1, slot),
            prefetch: prefetch,
            cacheIOSplit: max(1, (d["cache_io_split"] as? Int) ?? 4),
            expertIndexPath: d["expert_index_path"] as? String
        )
    }

    /// Encode to a `[String: Any]` matching `FlashMoEConfig.to_dict`.
    public func toDictionary() -> [String: Any] {
        var out: [String: Any] = [
            "enabled": enabled,
            "slot_bank_size": slotBankSize,
            "prefetch": prefetch.rawValue,
            "cache_io_split": cacheIOSplit,
        ]
        if let p = expertIndexPath { out["expert_index_path"] = p }
        return out
    }
}
