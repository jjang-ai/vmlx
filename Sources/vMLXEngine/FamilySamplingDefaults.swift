// SPDX-License-Identifier: Apache-2.0
//
// §443 — Per-family sampling defaults map.
//
// Rationale (audit finding #10): the existing 3-tier sampling
// resolution in Stream.swift (request → loadedModelDefaults from
// generation_config.json → resolved.settings) doesn't include
// per-family corrections. For example, JANGTQ2-quantized hybrid
// models (Nemotron-H 2-bit) produce garbage tokens (emojis,
// fullwidth chars) at the global default `temp=1.0` even though
// `generation_config.json` doesn't ship a per-quant override.
// Per `~/jang/research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md` §9:
// recommended T=0.6 top_p=0.95 for nemotron_h reasoning models.
//
// This file ships ONLY the family→defaults map + a pure resolution
// function. Integration into Stream.swift's 3-tier resolution is a
// 5-LOC wire-up: insert this layer as a fallback BETWEEN the
// generation_config.json layer and the global default — so users
// who explicitly set temperature in their request still win.
//
// Resolution order with §443 wired:
//   1. request (caller's explicit value)
//   2. loadedModelDefaults from generation_config.json
//   3. **§443 family-specific defaults**  ← new layer
//   4. global default (resolved.settings)
//
// Adding a new family: append to `Self.familyDefaults` below. No
// other file changes needed.

import Foundation

/// Recommended sampling parameters for a model family. nil fields
/// fall through to the next resolution layer.
public struct FamilySamplingDefaults: Sendable, Equatable {
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let repetitionPenalty: Float?

    /// Source citation for the recommendation. Always include the
    /// authoritative reference so future maintainers can verify.
    public let source: String

    public init(
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        source: String
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.source = source
    }
}

public enum FamilySamplingDefaultsRegistry {

    /// Family name → recommended defaults. Keys must match the family
    /// strings produced by `ModelCapabilities.family` (per
    /// `jang_tools/capabilities.py:49` mapping).
    private static let familyDefaults: [String: FamilySamplingDefaults] = [

        // Nemotron-H family (NemotronH text-only, Cascade-2, Nemotron-
        // Omni text bundles). DeepSeek-style sampler for reasoning
        // models. Avoids JANGTQ2 garbage tokens at T=1.0.
        // Reference: ~/jang/research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md §9
        "nemotron_h": FamilySamplingDefaults(
            temperature: 0.6,
            topP: 0.95,
            source: "NEMOTRON-OMNI-RUNTIME-2026-04-28 §9"),

        // Mirror for the v2 stamp (per capabilities.py).
        "nemotron_h_v2": FamilySamplingDefaults(
            temperature: 0.6,
            topP: 0.95,
            source: "NEMOTRON-OMNI-RUNTIME-2026-04-28 §9"),

        // DeepSeek-V4 reasoning. Same DeepSeek-style sampler. Per
        // memory `project_dsv4_swift_loop_diagnosis.md`: greedy +
        // no rep penalty produces collapse on long reasoning, but
        // T=0.6 + top_p=0.95 stays coherent.
        "deepseek_v4": FamilySamplingDefaults(
            temperature: 0.6,
            topP: 0.95,
            source: "project_dsv4_swift_loop_diagnosis.md"),

        // DeepSeek-V3 likewise.
        "deepseek_v3": FamilySamplingDefaults(
            temperature: 0.6,
            topP: 0.95,
            source: "project_dsv4_swift_loop_diagnosis.md"),

        // Qwen3.5 / Qwen3.6 family ships its own generation_config
        // (T=0.7 top_p=0.8) which is fine. No override here so
        // generation_config.json wins via tier-2.

        // GPT-OSS uses Harmony-style channel headers; sampling per
        // OpenAI's published recommendations matches generation_config.
        // No override here.

        // Mistral4 ships its own recommended sampler in
        // generation_config; no override.

        // Gemma4 likewise.
    ]

    /// Look up the recommended defaults for a family. Returns nil if
    /// no override is registered (caller should fall through to
    /// global default).
    public static func defaults(forFamily family: String) -> FamilySamplingDefaults? {
        guard !family.isEmpty else { return nil }
        return familyDefaults[family]
    }

    /// Resolve a single sampling parameter through the §443 family
    /// layer ONLY. Caller is responsible for the full 3-tier chain
    /// (request → generation_config → §443 → global). This helper
    /// just answers: "if request and generation_config are both nil,
    /// what does §443 say?".
    public static func resolveTemperature(forFamily family: String) -> Float? {
        defaults(forFamily: family)?.temperature
    }

    public static func resolveTopP(forFamily family: String) -> Float? {
        defaults(forFamily: family)?.topP
    }

    public static func resolveTopK(forFamily family: String) -> Int? {
        defaults(forFamily: family)?.topK
    }

    /// Names of all families with registered overrides. Used by
    /// regression tests to detect accidental drift.
    public static var registeredFamilies: [String] {
        Array(familyDefaults.keys).sorted()
    }
}
