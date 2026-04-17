// SPDX-License-Identifier: Apache-2.0
// Automatic ModelCapabilities detection. Four tiers, synchronous, never
// throws — always returns SOMETHING (falling through to a bronze
// `.fallback` entry if nothing matches).
//
//   Tier 1 — JANG stamped (gold)
//     Read jang_config.json. If it has a top-level `capabilities` object,
//     take its values verbatim. This is the canonical path going forward
//     — the JANG repacker team stamps this at pack time.
//
//   Tier 2 — model_type lookup (silver)
//     Read config.json. Walk text_config.model_type → model_type. Look up
//     in the curated `ModelTypeTable`. Honours priority-based tiebreaks.
//
//   Tier 3 — vanilla HF heuristic (bronze)
//     Unknown model_type — guess a parser from substring matches on the
//     model_type string. Log a `.warn` hinting that the model_type should
//     be added to the silver allowlist.
//
//   Tier 4 — modality refinement
//     Piggy-back on ModelDetector for text/vision, then polish embedding/
//     rerank/image via path-name + model_type hints.

import Foundation

public enum CapabilityDetector {

    /// Detect capabilities for a model directory. Never throws — always
    /// returns a valid `ModelCapabilities`, possibly at the `.fallback`
    /// tier.
    ///
    /// - Parameters:
    ///   - directory: path to the model dir containing `config.json`.
    ///   - warnLog: optional callback invoked with a human-readable warning
    ///     when we had to fall through to the bronze tier. Engine wires
    ///     this into LogStore.
    public static func detect(
        at directory: URL,
        warnLog: ((String) -> Void)? = nil
    ) -> ModelCapabilities {
        let fm = FileManager.default

        // Load config.json (always required for HF models)
        let cfgURL = directory.appendingPathComponent("config.json")
        let config = loadJSON(cfgURL) ?? [:]

        // Load jang_config.json (may not exist)
        let jangURL = directory.appendingPathComponent("jang_config.json")
        let jangJSON = loadJSON(jangURL) ?? [:]
        let isJANG = fm.fileExists(atPath: jangURL.path)
            || config["jang_config"] != nil
            || config["jang"] != nil

        let isMXTQ = detectMXTQ(config: config, jang: jangJSON)
        let quantBits = detectQuantBits(config: config, jang: jangJSON)

        // Resolve the HF model_type up front (text_config wins for VLM
        // wrappers, matching ModelDetector tier 1).
        let (rawModelType, modalityFromConfig) = resolveModelType(
            config: config,
            jang: jangJSON
        )

        let modality = refineModality(
            base: modalityFromConfig,
            modelType: rawModelType,
            directory: directory
        )

        // ── Tier 1: JANG stamped ──
        if let cap = detectFromJANGStamp(
            jang: jangJSON,
            modelType: rawModelType,
            modality: modality,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            quantBits: quantBits
        ) {
            // Observability: surface the Tier-1 hit so ops can confirm
            // jang_config.capabilities actually won. Prior to the 94-bundle
            // stamp rollout the silver table was the primary source; this
            // log line makes it easy to spot a stamped model falling
            // through to silver (would indicate a parse bug).
            FileHandle.standardError.write(Data((
                "[vmlx][caps] detection_source=jang_stamped "
                + "model_type=\(rawModelType) family=\(cap.family) "
                + "reasoning=\(cap.reasoningParser ?? "nil") "
                + "tool=\(cap.toolParser ?? "nil") "
                + "modality=\(cap.modality.rawValue) cache=\(cap.cacheType)\n"
            ).utf8))
            return cap
        }

        // Dynamic hybrid override: matches Python
        // `ssm_companion_cache.is_hybrid_ssm_config` — config.json with
        // `hybrid_override_pattern` or a model_type in the hybrid set always
        // gets cacheType="hybrid" regardless of the allowlist row.
        let forceHybrid = isHybridSSMConfig(config: config, modelType: rawModelType)

        // ── Tier 2: silver allowlist ──
        if let entry = ModelTypeTable.lookup(modelType: rawModelType) {
            // If the table says isMLLM and we resolved .text, upgrade; if
            // the table says text-only but config says vision (wrapper),
            // keep the config answer.
            let finalModality: ModelCapabilities.Modality = {
                if modality == .vision { return .vision }
                if entry.isMLLM { return .vision }
                return modality
            }()
            return ModelCapabilities(
                family: entry.family,
                modality: finalModality,
                modelType: rawModelType,
                reasoningParser: entry.reasoningParser,
                toolParser: entry.toolParser,
                thinkInTemplate: entry.thinkInTemplate,
                cacheType: resolveCacheType(
                    silverDefault: entry.cacheType,
                    forceHybrid: forceHybrid,
                    config: config),
                supportsTools: entry.toolParser != nil,
                supportsThinking: entry.reasoningParser != nil,
                quantBits: quantBits,
                isJANG: isJANG,
                isMXTQ: isMXTQ,
                chatTemplate: silverChatTemplate(directory: directory),
                detectionSource: .modelTypeTable
            )
        }

        // ── Tier 2.5: chat-template sniff ──
        //
        // When the model_type is unknown to the silver table (Qwen 3.6
        // MXFP4 from HF, boutique fine-tunes with custom model_type
        // strings, community MLX quants of tomorrow's models), we can
        // still nail the parser family by scanning the chat_template
        // for tokens unique to each tool-calling convention:
        //
        //   `[TOOL_CALLS]`                  → mistral family
        //   `<|tool_calls_section_begin|>`  → deepseek family
        //   `<|channel|>analysis`           → gpt_oss (Harmony)
        //   `<|tool_call>`                  → gemma4
        //   `<minimax:tool_call>`           → minimax_m2
        //   `<|python_tag|>`                → llama3/4 Hermes-style
        //   `<tool_call>`                   → qwen/hermes (fallback)
        //
        // Plus the existing unconditional-`<think>` probe for reasoning.
        // Beats bronze because the evidence is the model's own template
        // output, not name matching. Saves users from manually passing
        // `--reasoning-parser`/`--tool-call-parser` on fresh HF pulls.
        let tokCfgEarly = loadJSON(directory.appendingPathComponent("tokenizer_config.json"))
            ?? [:]
        let templateEarly = (tokCfgEarly["chat_template"] as? String) ?? ""
        if let sniff = detectFromTemplateSniff(
            chatTemplate: templateEarly,
            modelType: rawModelType,
            modality: modality,
            forceHybrid: forceHybrid,
            quantBits: quantBits,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            config: config
        ) {
            FileHandle.standardError.write(Data((
                "[vmlx][caps] detection_source=template_sniff "
                + "model_type=\(rawModelType) family=\(sniff.family) "
                + "reasoning=\(sniff.reasoningParser ?? "nil") "
                + "tool=\(sniff.toolParser ?? "nil") "
                + "modality=\(sniff.modality.rawValue) cache=\(sniff.cacheType)\n"
            ).utf8))
            return sniff
        }

        // ── Tier 3: bronze heuristic ──
        var heuristic = bronzeHeuristic(modelType: rawModelType)

        // v1.3.53 parity: chat-template `<think>` probe.
        //
        // Bronze substring matching catches most known families by
        // name (qwen/deepseek/mistral/gemma etc.), but a brand-new
        // thinking model with an unrecognized model_type would fall
        // through with `reasoning = nil` — producing unparsed
        // `<think>...</think>` blobs in the visible content instead
        // of routing to `reasoning_content`. The fix is to scan the
        // model's chat template for `<think>` markers and force the
        // deepseek_r1 parser when the template actually stamps them.
        //
        // The parsed string is read from `tokenizer_config.json`
        // (standard HF location). Gemma 4 / Gemma 3n have
        // `{% if enable_thinking %}<think>...{% endif %}` blocks
        // where the think tags are GATED on the `enable_thinking`
        // flag — not unconditionally stamped. `hasUnconditionalThinkMarker`
        // narrows the detection to exclude those cases so Gemma
        // models don't get false-flagged.
        let hadNoReasoningBefore = (heuristic.reasoning == nil)
        let tokCfg = loadJSON(directory.appendingPathComponent("tokenizer_config.json"))
            ?? [:]
        let chatTemplate = (tokCfg["chat_template"] as? String) ?? ""
        let templateStampsThink = hasUnconditionalThinkMarker(chatTemplate)
        if templateStampsThink {
            // Force deepseek_r1 as the safe default for any template
            // that unconditionally stamps `<think>`. If the bronze
            // heuristic already picked a parser, leave it alone — an
            // explicit family match beats the generic fallback.
            if heuristic.reasoning == nil {
                heuristic = Bronze(
                    family: heuristic.family,
                    reasoning: "deepseek_r1",
                    tool: heuristic.tool,
                    thinkInTemplate: true)
            } else {
                heuristic = Bronze(
                    family: heuristic.family,
                    reasoning: heuristic.reasoning,
                    tool: heuristic.tool,
                    thinkInTemplate: true)
            }
        }

        // Startup warning. Two shapes:
        //   1. Unknown model_type entirely — bronze had to guess.
        //   2. Unknown model_type AND the template stamps `<think>` —
        //      loud warning with the actionable remediation.
        if templateStampsThink && hadNoReasoningBefore {
            warnLog?(
                "⚠️ THINKING MODEL WARNING: model_type '\(rawModelType)' is not in " +
                "the allowlist and its chat template unconditionally stamps " +
                "`<think>` tags. Forcing reasoning_parser=deepseek_r1 so " +
                "think blocks route to reasoning_content. Long-term fix: add " +
                "this model_type to ModelTypeTable in ModelCapabilities.swift " +
                "with the correct reasoning + tool parsers."
            )
        } else {
            warnLog?(
                "CapabilityDetector: model_type '\(rawModelType)' not in silver " +
                "allowlist — using bronze heuristic " +
                "(reasoning=\(heuristic.reasoning ?? "nil"), " +
                "tool=\(heuristic.tool ?? "nil"), " +
                "thinkInTemplate=\(heuristic.thinkInTemplate)). " +
                "Consider adding this model_type to ModelTypeTable."
            )
        }
        return ModelCapabilities(
            family: heuristic.family,
            modality: modality,
            modelType: rawModelType,
            reasoningParser: heuristic.reasoning,
            toolParser: heuristic.tool,
            thinkInTemplate: heuristic.thinkInTemplate,
            cacheType: forceHybrid ? "hybrid" : "kv",
            supportsTools: heuristic.tool != nil,
            supportsThinking: heuristic.reasoning != nil,
            quantBits: quantBits,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            chatTemplate: chatTemplate.isEmpty ? nil : chatTemplate,
            detectionSource: .fallback
        )
    }

    /// Scan a chat_template string for `<think>` markers that are
    /// NOT gated behind a Jinja `{% if enable_thinking %}` or
    /// equivalent conditional.
    ///
    /// Returns `true` when the template unconditionally stamps
    /// `<think>` or `<|think|>` tokens — in that case the model
    /// will always produce reasoning blocks and we need a parser
    /// to extract them.
    ///
    /// Returns `false` for enable_thinking-gated templates
    /// (Gemma 4, Gemma 3n, Mistral 4, Qwen3.5 when `enable_thinking`
    /// is optional) — those have opt-in think blocks driven by the
    /// request's `enable_thinking` flag, and forcing a reasoning
    /// parser would incorrectly route their NON-thinking output.
    ///
    /// Narrowing logic:
    ///   1. If the template has no `<think>` or `<|think|>`
    ///      substring at all → `false`.
    ///   2. If every occurrence of a think marker is inside an
    ///      `{% if enable_thinking %}` / `{%- if ... %}` block,
    ///      it's gated → `false`.
    ///   3. Otherwise → `true`.
    internal static func hasUnconditionalThinkMarker(_ template: String) -> Bool {
        if template.isEmpty { return false }
        let lower = template.lowercased()
        // Substring probe. Qwen3/DeepSeek/etc stamp `<think>`; some
        // templates use `<|think|>` / `<|start_of_thought|>`.
        let thinkMarkers = ["<think>", "<|think|>", "<|start_of_thought|>"]
        let hasAny = thinkMarkers.contains { lower.contains($0) }
        if !hasAny { return false }

        // Narrowing: if EVERY `<think>` substring is preceded
        // somewhere earlier in the same line (or recent line) by an
        // `enable_thinking` if-block, treat as gated and return false.
        //
        // The Jinja subset Gemma/Mistral use looks like:
        //   {%- if enable_thinking %}<think>
        //   {%- endif %}
        // or:
        //   {% if messages[-1].enable_thinking %}<think>\n{% endif %}
        //
        // A simple but effective check: split on lines, find the
        // lines containing a think marker, and verify that at least
        // one of them is NOT within 3 lines after an
        // `enable_thinking` / `reasoning_effort` / `thinking_budget`
        // conditional open.
        let lines = lower.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let gateTokens = [
            "enable_thinking", "reasoning_effort", "thinking_budget",
            "include_reasoning",
        ]
        // Scan for lines that contain the marker AND aren't recently
        // preceded by a gate open. If even one is unconditional,
        // return true.
        var recentGateOpen = false
        var linesSinceGate = 0
        for line in lines {
            // Track gate block opens: any `{% if ... enable_thinking ... %}`
            // or `{%- if ... enable_thinking ... %}`.
            if line.contains("{%") && line.contains("if")
                && gateTokens.contains(where: { line.contains($0) })
            {
                recentGateOpen = true
                linesSinceGate = 0
                continue
            }
            // Close the gate on `{% endif %}` or `{%- endif %}`.
            if line.contains("endif") {
                recentGateOpen = false
                linesSinceGate = 0
                continue
            }
            if recentGateOpen {
                linesSinceGate += 1
                // The gated block should be short (1–5 lines typically).
                // If we've walked too far without an endif, assume the
                // template is malformed and stop tracking.
                if linesSinceGate > 20 {
                    recentGateOpen = false
                }
            }
            // Check the line for a think marker.
            let hasMarker = thinkMarkers.contains { line.contains($0) }
            if hasMarker && !recentGateOpen {
                return true
            }
        }
        return false
    }

    // MARK: - Tier 1: JANG stamp

    private static func detectFromJANGStamp(
        jang: [String: Any],
        modelType: String,
        modality: ModelCapabilities.Modality,
        isJANG: Bool,
        isMXTQ: Bool,
        quantBits: Int?
    ) -> ModelCapabilities? {
        guard let caps = jang["capabilities"] as? [String: Any] else { return nil }
        let reasoning = caps["reasoning_parser"] as? String
        let tool = caps["tool_parser"] as? String
        let thinkInTemplate = caps["think_in_template"] as? Bool ?? false
        let cacheType = caps["cache_type"] as? String ?? "kv"
        let supportsTools = caps["supports_tools"] as? Bool ?? (tool != nil)
        let supportsThinking = caps["supports_thinking"] as? Bool ?? (reasoning != nil)
        let family = (caps["family"] as? String) ?? modelType
        let chatTemplate = caps["chat_template"] as? String

        // Modality override from the stamp if present.
        var stampedModality = modality
        if let m = caps["modality"] as? String,
           let mv = ModelCapabilities.Modality(rawValue: m) {
            stampedModality = mv
        }

        return ModelCapabilities(
            family: family,
            modality: stampedModality,
            modelType: modelType,
            reasoningParser: reasoning,
            toolParser: tool,
            thinkInTemplate: thinkInTemplate,
            cacheType: cacheType,
            supportsTools: supportsTools,
            supportsThinking: supportsThinking,
            quantBits: quantBits,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            chatTemplate: chatTemplate,
            detectionSource: .jangStamped
        )
    }

    // MARK: - Tier 2.5: chat-template sniff

    /// Family → `(reasoning, tool, thinkInTemplate, cache)` resolved from
    /// a chat_template scan alone. Returns nil when we can't find a
    /// high-confidence family signature — caller falls through to the
    /// bronze heuristic.
    ///
    /// Why this beats bronze: bronze guesses from the `model_type` STRING,
    /// which is what's in config.json. Brand-new architectures often
    /// have unrecognized model_type strings but ship well-known chat
    /// templates (Qwen 3.6 MXFP4 = HF Qwen chat template; third-party
    /// fine-tune of Mistral 4 = Mistral chat template). Template sniff
    /// catches those cleanly.
    ///
    /// Ordering: the check list is priority-ordered so the MORE SPECIFIC
    /// token always wins. `[TOOL_CALLS]` is tested before `<tool_call>`
    /// because Mistral's `[TOOL_CALLS]` is a unique marker, while
    /// `<tool_call>` is the most generic fallback.
    private static func detectFromTemplateSniff(
        chatTemplate: String,
        modelType: String,
        modality: ModelCapabilities.Modality,
        forceHybrid: Bool,
        quantBits: Int?,
        isJANG: Bool,
        isMXTQ: Bool,
        config: [String: Any]
    ) -> ModelCapabilities? {
        guard !chatTemplate.isEmpty else { return nil }

        // Tool parser dispatch — first hit wins.
        let (toolParser, toolFamily): (String?, String) = {
            if chatTemplate.contains("[TOOL_CALLS]") {
                return ("mistral", "mistral")
            }
            if chatTemplate.contains("<|tool_calls_section_begin|>")
                || chatTemplate.contains("<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>")
            {
                return ("deepseek", "deepseek")
            }
            if chatTemplate.contains("<|channel|>analysis")
                || chatTemplate.contains("<|channel|>final")
            {
                return ("glm47", "gpt_oss")
            }
            if chatTemplate.contains("<|tool_call>") {
                return ("gemma4", "gemma")
            }
            if chatTemplate.contains("<minimax:tool_call>") {
                return ("minimax", "minimax_m2")
            }
            if chatTemplate.contains("<|python_tag|>") {
                return ("llama", "llama")
            }
            if chatTemplate.contains("<tool_call>") {
                // Generic hermes-style. Qwen-family templates also use
                // this form; we pick "qwen" because it's the larger
                // HF ecosystem and the parser is equivalent.
                return ("qwen", "qwen")
            }
            return (nil, "")
        }()

        // Reasoning parser — use unconditional-`<think>` probe.
        let thinkUnconditional = hasUnconditionalThinkMarker(chatTemplate)
        let reasoningParser: String? = {
            if chatTemplate.contains("<|channel|>analysis") { return "openai_gptoss" }
            if chatTemplate.contains("<\u{FF5C}begin\u{2581}of\u{2581}reasoning\u{FF5C}>") {
                return "deepseek_r1"
            }
            if thinkUnconditional {
                // Template ALREADY stamps `<think>` — almost certainly
                // qwen3 or deepseek_r1. When the tool parser is
                // mistral/minimax the reasoning is still `<think>`-based
                // (qwen3 is the safest shared parser). When gpt_oss
                // handled it above we've already returned.
                if toolFamily == "qwen" || toolFamily == "minimax_m2" {
                    return "qwen3"
                }
                return "deepseek_r1"
            }
            return nil
        }()

        // Need at least one positive signal to claim this tier. If we
        // can't pick a parser AND there's no think marker, bail out and
        // let bronze handle it.
        guard toolParser != nil || reasoningParser != nil else { return nil }

        // Family name: if neither tool nor reasoning contributed, use
        // the model_type as the best label.
        let family = toolFamily.isEmpty ? modelType : toolFamily

        let finalCache = resolveCacheType(
            silverDefault: "kv",
            forceHybrid: forceHybrid,
            config: config)

        return ModelCapabilities(
            family: family,
            modality: modality,
            modelType: modelType,
            reasoningParser: reasoningParser,
            toolParser: toolParser,
            thinkInTemplate: thinkUnconditional,
            cacheType: finalCache,
            supportsTools: toolParser != nil,
            supportsThinking: reasoningParser != nil,
            quantBits: quantBits,
            isJANG: isJANG,
            isMXTQ: isMXTQ,
            chatTemplate: chatTemplate,
            detectionSource: .templateSniff
        )
    }

    // MARK: - Tier 3: bronze heuristic

    private struct Bronze {
        let family: String
        let reasoning: String?
        let tool: String?
        let thinkInTemplate: Bool
    }

    /// Read the model's `tokenizer_config.json` chat_template. Used to
    /// surface the loaded template on `ModelCapabilities` so the engine
    /// can apply the `--chat-template` CLI override against the real
    /// model template instead of silently no-op'ing. Returns nil when
    /// the file is missing or has no chat_template field.
    private static func silverChatTemplate(directory: URL) -> String? {
        let tokCfg = loadJSON(directory.appendingPathComponent("tokenizer_config.json"))
            ?? [:]
        if let s = tokCfg["chat_template"] as? String, !s.isEmpty { return s }
        return nil
    }

    /// Reconcile the silver-table cacheType default against config.json
    /// reality. Audit P1-CAP-1: pre-stamping `cacheType="mla"` in the
    /// silver table is fine for *typical* checkpoints, but a plain
    /// (non-MLA) fine-tune of the same model_type would silently get
    /// MLA cache routing and the new MLA ⊥ TurboQuant guard would skip
    /// TQ for no reason. Demote to "kv" when config.json doesn't
    /// actually declare MLA via `kv_lora_rank > 0`. Hybrid override
    /// always wins.
    private static func resolveCacheType(
        silverDefault: String,
        forceHybrid: Bool,
        config: [String: Any]
    ) -> String {
        if forceHybrid { return "hybrid" }
        if silverDefault == "mla" {
            // Look for MLA marker in either top-level or text_config.
            if isMLAConfig(config) { return "mla" }
            return "kv"
        }
        return silverDefault
    }

    /// Detect MLA from config.json. Matches Python
    /// `vmlx_engine/utils/model_inspector.py::is_mla_model` — looks for
    /// `kv_lora_rank > 0` in the top-level config OR inside text_config
    /// (VLM wrapper). Returns false when the field is absent or zero.
    private static func isMLAConfig(_ config: [String: Any]) -> Bool {
        if let rank = config["kv_lora_rank"] as? Int, rank > 0 { return true }
        if let textCfg = config["text_config"] as? [String: Any],
           let rank = textCfg["kv_lora_rank"] as? Int, rank > 0
        {
            return true
        }
        return false
    }

    private static func bronzeHeuristic(modelType: String) -> Bronze {
        let t = modelType.lowercased()
        if t.contains("qwen") {
            return Bronze(family: "qwen", reasoning: "qwen3", tool: "qwen", thinkInTemplate: true)
        }
        if t.contains("deepseek") {
            return Bronze(family: "deepseek", reasoning: "deepseek_r1", tool: "deepseek", thinkInTemplate: false)
        }
        if t.contains("mistral") || t.contains("mixtral") {
            return Bronze(family: "mistral", reasoning: "mistral", tool: "mistral", thinkInTemplate: false)
        }
        if t.contains("gemma3n") {
            // Gemma 3n is a Gemma family member with the same hermes-style
            // tool format as Gemma 4. Audit P1-CAP-4: previously left
            // `tool: nil` so tool calls silently dropped on bronze-tier
            // Gemma3n model_type variants. Promote to gemma4 parser to
            // match the silver entry.
            return Bronze(family: "gemma", reasoning: "gemma4", tool: "gemma4", thinkInTemplate: false)
        }
        if t.contains("gemma") {
            return Bronze(family: "gemma", reasoning: "gemma4", tool: "gemma4", thinkInTemplate: false)
        }
        if t.contains("gpt_oss") || t.contains("harmony") {
            return Bronze(family: "gpt_oss", reasoning: "openai_gptoss", tool: "glm47", thinkInTemplate: false)
        }
        if t.contains("llama") {
            return Bronze(family: "llama", reasoning: nil, tool: "llama", thinkInTemplate: false)
        }
        if t.contains("nemotron") {
            return Bronze(family: "nemotron", reasoning: "deepseek_r1", tool: "nemotron", thinkInTemplate: true)
        }
        if t.contains("phi") {
            return Bronze(family: "phi", reasoning: nil, tool: "hermes", thinkInTemplate: false)
        }
        if t.contains("granite") {
            return Bronze(family: "granite", reasoning: nil, tool: "granite", thinkInTemplate: false)
        }
        if t.contains("minimax") {
            return Bronze(family: "minimax", reasoning: "qwen3", tool: "minimax", thinkInTemplate: true)
        }
        // Absolute fallback — Hermes JSON parser catches most untrained
        // native tool-call formats. No reasoning parser.
        return Bronze(family: "unknown", reasoning: nil, tool: "native", thinkInTemplate: false)
    }

    // MARK: - model_type resolution

    private static func resolveModelType(
        config: [String: Any],
        jang: [String: Any]
    ) -> (String, ModelCapabilities.Modality) {
        // jang_config.has_vision is AUTHORITATIVE and MUST be checked BEFORE
        // the HF text_config tier — a text-only Mistral 4 / Qwen 3.5 JANG
        // still has `text_config` in its config.json because the source
        // wrapper arch (mistral3 / qwen3_vl) is a VLM, but jang_config
        // explicitly says `has_vision: false`. This mirrors
        // api/utils.py::is_mllm_model which resolves jang before config.json.
        // Audit 2026-04-14.
        let jangHasVision: Bool? = {
            if let hv = jang["has_vision"] as? Bool { return hv }
            if let arch = jang["architecture"] as? [String: Any],
               let hv = arch["has_vision"] as? Bool { return hv }
            return nil
        }()

        // Helper: resolve the textual model_type (prefer text_config.model_type
        // because the outer wrapper type is usually just "mistral3" / "llava"
        // / "qwen2_vl" which isn't useful for parser/family selection).
        //
        // Audit P1-CAP-3: Mistral 4 VLM checkpoints wrap a `text_config` of
        // model_type="mistral4" inside an outer `model_type="mistral3"`,
        // BUT older Mistral 4 packs flip it: outer is "mistral4" and
        // text_config still says "mistral3" because the upstream HF arch
        // was authored before the split. Detect either shape and prefer
        // mistral4 so the silver lookup hits the mistral4 reasoning row
        // (priority 30) instead of the dense mistral3 row (priority 10).
        let modelType: String = {
            let outer = (config["model_type"] as? String) ?? ""
            let inner = (config["text_config"] as? [String: Any])?["model_type"] as? String
            // Mistral4 promotion: if either outer or inner says mistral4,
            // use mistral4. Catches both wrapping orderings.
            if outer == "mistral4" || inner == "mistral4" {
                return "mistral4"
            }
            if let inner { return inner }
            return outer.isEmpty ? "unknown" : outer
        }()

        if let hv = jangHasVision {
            return (modelType, hv ? .vision : .text)
        }

        // No jang override — fall back to HF config.
        if config["text_config"] is [String: Any] {
            return (modelType, .vision)
        }
        if config["vision_config"] != nil {
            return (modelType, .vision)
        }
        return (modelType, .text)
    }

    // MARK: - Modality refinement

    private static func refineModality(
        base: ModelCapabilities.Modality,
        modelType: String,
        directory: URL
    ) -> ModelCapabilities.Modality {
        let mt = modelType.lowercased()
        if mt.contains("embed") || mt == "bert" || mt == "xlm-roberta" {
            return .embedding
        }
        if mt.contains("rerank") {
            return .rerank
        }
        let last = directory.lastPathComponent.lowercased()
        if last.contains("flux") || last.contains("z-image")
            || last.contains("sdxl") || last.contains("schnell") {
            return .image
        }
        return base
    }

    // MARK: - MXTQ / quant bits

    private static func detectMXTQ(config: [String: Any], jang: [String: Any]) -> Bool {
        if let q = config["quantization"] as? [String: Any],
           let m = q["method"] as? String, m.lowercased().contains("mxtq") {
            return true
        }
        if let q = jang["quantization"] as? [String: Any],
           let m = q["method"] as? String, m.lowercased().contains("mxtq") {
            return true
        }
        if config["mxtq_seed"] != nil || config["mxtq_bits"] != nil { return true }
        if jang["mxtq_seed"] != nil || jang["mxtq_bits"] != nil { return true }
        return false
    }

    private static func detectQuantBits(config: [String: Any], jang: [String: Any]) -> Int? {
        if let q = config["quantization"] as? [String: Any],
           let b = q["bits"] as? Int { return b }
        if let q = jang["quantization"] as? [String: Any] {
            // Audit P2-CAP-7: previously took `widths.first` which lost
            // every layer beyond the first. For mixed-precision JANGTQ
            // with `bit_widths_used=[2,4,6]`, the average best-represents
            // the model's effective bit budget for size estimates and UI
            // labels. Fall back to bits or first-element on degenerate
            // configs.
            if let widths = q["bit_widths_used"] as? [Int], !widths.isEmpty {
                let avg = Double(widths.reduce(0, +)) / Double(widths.count)
                return Int(avg.rounded())
            }
            if let b = q["bits"] as? Int { return b }
        }
        return nil
    }

    // MARK: - Hybrid SSM config detection

    /// Mirrors Python `vmlx_engine/utils/ssm_companion_cache.py::is_hybrid_ssm_config`.
    ///
    /// A model is hybrid (attention + SSM layers interleaved) if:
    ///   - config.json contains `hybrid_override_pattern` (explicit), OR
    ///   - model_type is in the known hybrid set (`nemotron_h`, `qwen3_next`,
    ///     `jamba`, `falcon_h1`, `granite_moe_hybrid`, `lfm2`, `lfm2_moe`,
    ///     `mimo_v2_flash`, `baichuan_m1`), OR
    ///   - `text_config` is a dict with the same markers (VLM wrappers).
    ///
    /// When this returns true the detector forces `cacheType = "hybrid"` and
    /// `Engine.setupCacheCoordinator` calls `coordinator.setHybrid(true)` so
    /// the SSMStateCache companion path activates.
    private static let hybridModelTypes: Set<String> = [
        "nemotron_h", "qwen3_next", "jamba",
        "falcon_h1", "granite_moe_hybrid",
        "lfm2", "lfm2_moe", "mimo_v2_flash", "baichuan_m1",
    ]

    private static func isHybridSSMConfig(
        config: [String: Any],
        modelType: String
    ) -> Bool {
        if config["hybrid_override_pattern"] != nil { return true }
        if hybridModelTypes.contains(modelType.lowercased()) { return true }
        // `layer_types` interleave marker — present in newer hybrid model
        // configs (Qwen3.5-VL JANG, Gemma SWA hybrids, etc) where the
        // top-level model_type is NOT in the legacy hybrid allowlist
        // but the layer-by-layer config explicitly lists `linear_attention`
        // (= Mamba/SSM layer) interleaved with `full_attention`.
        //
        // Production crash this catches: Qwen3.5-VL-4B-JANG_4S-CRACK has
        // model_type="qwen3_5" (NOT in allowlist) but
        // text_config.layer_types[0..7] = ["linear_attention" × 7,
        // "full_attention"]. Without this branch the model is treated
        // as plain attention, the disk cache stores 8 KV layers, the
        // restore tries to fit them into 32 cache slots, and the
        // mismatch fatal-aborts in mlx_array_dim. Detecting hybrid
        // here triggers `Engine.setupCacheCoordinator`'s hybrid guard
        // which gates disk cache off (until SSM-aware disk path lands)
        // and turns on the SSM companion store/fetch.
        if hasLinearAttentionLayer(in: config) { return true }
        if let text = config["text_config"] as? [String: Any] {
            if text["hybrid_override_pattern"] != nil { return true }
            if let mt = text["model_type"] as? String,
               hybridModelTypes.contains(mt.lowercased()) {
                return true
            }
            if hasLinearAttentionLayer(in: text) { return true }
        }
        return false
    }

    /// Scan a config dict for the `layer_types` marker that indicates
    /// interleaved linear-attention (SSM) layers. Used by the hybrid
    /// override above so newer hybrid VLMs are caught even when their
    /// model_type isn't in the legacy allowlist.
    ///
    /// Deliberately does NOT match `sliding_attention` — SWA-mixed
    /// models (Gemma 3) have standard attention with a rotating ring
    /// buffer, not SSM state. Conflating the two would incorrectly
    /// activate the SSMStateCache companion path for a model that
    /// has no SSM state to capture. See `hasMixedSlidingFullAttention`
    /// (F-G5) for the SWA-hybrid case.
    private static func hasLinearAttentionLayer(in dict: [String: Any]) -> Bool {
        guard let types = dict["layer_types"] as? [String] else { return false }
        for t in types {
            let lower = t.lowercased()
            if lower.contains("linear_attention") || lower.contains("mamba") ||
               lower.contains("ssm") {
                return true
            }
        }
        return false
    }

    /// F-G5 (2026-04-15): detect Gemma 3-style mixed `sliding_attention`
    /// + `full_attention` layer interleave. Gemma 3 model classes already
    /// return the correct per-layer cache mix (RotatingKVCache for SWA
    /// layers + KVCacheSimple for full) from `newCache()`, and SLIDING-1
    /// made disk L2 round-trip RotatingKVCache via the `.rotating`
    /// LayerKind tag — so there is no active bug, but surfacing this
    /// flag lets callers treat mixed-SWA models distinctly from both
    /// pure-dense and SSM-hybrid in logs, diagnostics, and any future
    /// cache-policy branching (e.g. disabling TQ on SWA layers where
    /// the ring buffer compresses poorly).
    public static func hasMixedSlidingFullAttention(
        in config: [String: Any]
    ) -> Bool {
        if hasMixedSlidingFullLayerTypes(in: config) { return true }
        if let text = config["text_config"] as? [String: Any],
           hasMixedSlidingFullLayerTypes(in: text) {
            return true
        }
        return false
    }

    private static func hasMixedSlidingFullLayerTypes(
        in dict: [String: Any]
    ) -> Bool {
        guard let types = dict["layer_types"] as? [String] else { return false }
        var sawSliding = false
        var sawFull = false
        for t in types {
            let lower = t.lowercased()
            if lower.contains("sliding_attention") { sawSliding = true }
            if lower.contains("full_attention") { sawFull = true }
            if sawSliding && sawFull { return true }
        }
        return false
    }

    // MARK: - JSON helper

    private static func loadJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}
