// SPDX-License-Identifier: Apache-2.0
// Ported from vmlx_engine/reasoning/*.py to pure Swift.
//
// Streaming-safe reasoning parsers: each parser receives (previous, current, delta)
// triples and emits DeltaMessage chunks with `reasoning` and/or `content` set.
//
// Parsers ported:
//   - Qwen3 / BaseThinking (<think>...</think>)
//   - DeepSeek-R1         (lenient <think>...</think>, may omit start)
//   - Mistral 4           ([THINK]...[/THINK])
//   - Gemma 4             (<|channel>thought ... <channel|>)
//   - GPT-OSS / Harmony   (<|start|>assistant<|channel|>analysis<|message|>...)

import Foundation

/// Streaming reasoning delta. Mirrors Python `DeltaMessage` in
/// vmlx_engine/reasoning/base.py.
public struct ReasoningDelta: Sendable, Equatable {
    public var content: String?
    public var reasoning: String?

    public init(reasoning: String? = nil, content: String? = nil) {
        self.reasoning = reasoning
        self.content = content
    }

    public var isEmpty: Bool { content == nil && reasoning == nil }
}

/// Abstract reasoning parser. Subclasses implement both the complete
/// `extractReasoning` and the streaming `extractReasoningStreaming` paths.
public protocol ReasoningParser: AnyObject {
    /// Complete-text extraction. Returns (reasoning, content) either of which may be nil.
    func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?)

    /// Streaming extraction. The caller is responsible for passing the full
    /// previous / current text along with this chunk's delta.
    func extractReasoningStreaming(previous: String, current: String, delta: String) -> ReasoningDelta?

    /// End-of-stream flush. Called once at stream termination so the parser
    /// can drain any content it was holding back for lookahead disambiguation
    /// (e.g. the Gemma 4 parser's 18-char `<|channel>thought` marker window).
    /// Returns nil if there is no residual content. Default impl does nothing.
    func finishStreaming(fullText: String) -> ReasoningDelta?

    /// Reset per-request state before starting a new stream.
    /// `thinkInPrompt`: true when `<think>`/`[THINK]` was injected by the chat template.
    /// `harmonyActive`: GPT-OSS only — true when Harmony analysis prefix was injected.
    func resetState(thinkInPrompt: Bool, harmonyActive: Bool)
}

public extension ReasoningParser {
    func resetState(thinkInPrompt: Bool = false, harmonyActive: Bool = false) {
        resetState(thinkInPrompt: thinkInPrompt, harmonyActive: harmonyActive)
    }
    /// Default: no residual. Parsers that buffer for lookahead override this.
    func finishStreaming(fullText: String) -> ReasoningDelta? { nil }
}

// MARK: - BaseThinkingReasoningParser

/// Base parser for models wrapping reasoning in `start_token ... end_token`
/// pairs like `<think>...</think>` or `[THINK]...[/THINK]`.
open class BaseThinkingReasoningParser: ReasoningParser {
    public let startToken: String
    public let endToken: String
    fileprivate var thinkInPrompt: Bool = false

    public init(startToken: String, endToken: String) {
        self.startToken = startToken
        self.endToken = endToken
    }

    public func resetState(thinkInPrompt: Bool, harmonyActive: Bool) {
        self.thinkInPrompt = thinkInPrompt
    }

    open func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        let text = modelOutput
        let hasStart = text.contains(startToken)
        let hasEnd = text.contains(endToken)

        // Case 1: both tags
        if hasStart && hasEnd {
            let afterStart = text.range(of: startToken).map { String(text[$0.upperBound...]) } ?? text
            if let endRange = afterStart.range(of: endToken) {
                let reasoning = String(afterStart[..<endRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                let content = String(afterStart[endRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            }
        }

        // Case 2: only closing tag (think was injected in prompt)
        if hasEnd {
            if let endRange = text.range(of: endToken) {
                let reasoning = String(text[..<endRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                let content = String(text[endRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            }
        }

        // Case 3: only start tag (truncated reasoning, max_tokens hit)
        if hasStart {
            if let startRange = text.range(of: startToken) {
                let reasoning = String(text[startRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, nil)
            }
        }

        // Case 4: no tags at all
        return (nil, modelOutput)
    }

    public func extractReasoningStreaming(previous: String, current: String, delta: String) -> ReasoningDelta? {
        // Skip if delta is just the special tokens themselves
        let stripped = delta.trimmingCharacters(in: .whitespacesAndNewlines)
        if stripped == startToken { return nil }
        if stripped == endToken { return nil }

        let startInPrev = previous.contains(startToken)
        let startInCurrent = current.contains(startToken)
        let endInPrev = previous.contains(endToken)
        let endInDelta = delta.contains(endToken)

        // Case 1: explicit <think> in output
        if startInCurrent {
            return handleExplicitThink(
                delta: delta,
                startInPrev: startInPrev,
                endInPrev: endInPrev,
                endInDelta: endInDelta
            )
        }

        // Case 2: implicit think (only end token)
        if current.contains(endToken) {
            return handleImplicitThink(delta: delta, endInPrev: endInPrev, endInDelta: endInDelta)
        }

        // Case 3: no think tags yet
        if thinkInPrompt {
            return ReasoningDelta(reasoning: delta)
        }
        return ReasoningDelta(content: delta)
    }

    /// §420 — strip stray `<think>` / `</think>` markers from a
    /// delta that's known to be in a single mode. Interleaved-
    /// thinking models occasionally emit duplicate or unmatched
    /// markers as artifacts (verified 2026-04-25 on MiniMax-Small
    /// JANGTQ in osaurus chat: model emitted `</think>` THREE times
    /// across one assistant turn — first legitimately closed
    /// reasoning, second + third were strays that leaked verbatim
    /// into the user-visible content stream).
    ///
    /// Mirrors `vmlx-swift-lm` c98ae5c stray-tag policy. Family-
    /// gated: only `BaseThinkingReasoningParser` subclasses (Qwen3,
    /// DeepSeek-R1, Mistral, MiniMax/GLM/Nemotron/Kimi via Qwen3
    /// alias) opt in. `Gemma4ReasoningParser` (channel format) is
    /// a separate class and keeps stray-tag passthrough by design.
    private func stripStrayMarkers(_ s: String) -> String {
        if s.isEmpty { return s }
        var out = s
        if out.contains(endToken) {
            out = out.replacingOccurrences(of: endToken, with: "")
        }
        if out.contains(startToken) {
            out = out.replacingOccurrences(of: startToken, with: "")
        }
        return out
    }

    private func handleExplicitThink(delta: String, startInPrev: Bool, endInPrev: Bool, endInDelta: Bool) -> ReasoningDelta? {
        let startInDelta = delta.contains(startToken)

        if startInPrev {
            // §420 — order matters: if `endInPrev`, we're ALREADY in
            // content mode. Any `</think>` in this delta is a stray, NOT
            // a boundary. Strip it and emit the whole delta as content.
            // If we don't check endInPrev first, the split-delta path
            // below would mis-treat a stray as a boundary and route the
            // pre-stray segment to reasoning.
            if endInPrev {
                let stripped = stripStrayMarkers(delta)
                return stripped.isEmpty ? nil : ReasoningDelta(content: stripped)
            }
            if endInDelta, let endRange = delta.range(of: endToken) {
                // Genuine boundary — first time we see </think>. The
                // pre-end segment is the tail of reasoning; the post-end
                // segment begins content. Strip stray duplicates from
                // either half.
                let r = stripStrayMarkers(String(delta[..<endRange.lowerBound]))
                let c = stripStrayMarkers(String(delta[endRange.upperBound...]))
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
            } else {
                // §420 — still in reasoning. Strip stray `<think>`
                // (model emitted a duplicate open marker).
                let stripped = stripStrayMarkers(delta)
                return stripped.isEmpty ? nil : ReasoningDelta(reasoning: stripped)
            }
        }

        if startInDelta, let startRange = delta.range(of: startToken) {
            if endInDelta, let endRange = delta.range(of: endToken), endRange.lowerBound >= startRange.upperBound {
                let r = stripStrayMarkers(String(delta[startRange.upperBound..<endRange.lowerBound]))
                let c = stripStrayMarkers(String(delta[endRange.upperBound...]))
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
            } else {
                let r = stripStrayMarkers(String(delta[startRange.upperBound...]))
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r)
            }
        }

        // No tags involved — pure content mode. Still strip stray
        // markers in case the delta contains an isolated `</think>`
        // emitted by the model after content already started.
        let stripped = stripStrayMarkers(delta)
        return stripped.isEmpty ? nil : ReasoningDelta(content: stripped)
    }

    private func handleImplicitThink(delta: String, endInPrev: Bool, endInDelta: Bool) -> ReasoningDelta? {
        // §420 — order matters here too (matches handleExplicitThink
        // fix). If we're already past the close from a prior chunk,
        // ANY `</think>` in this delta is a stray, NOT a boundary.
        // Without this `endInPrev` check first, the split-delta path
        // below would mis-treat a duplicate `</think>` as a fresh
        // boundary and route the pre-stray segment to reasoning when
        // it should all be content.
        if endInPrev {
            let stripped = stripStrayMarkers(delta)
            return stripped.isEmpty ? nil : ReasoningDelta(content: stripped)
        }
        if endInDelta, let endRange = delta.range(of: endToken) {
            // Genuine first close: pre-end is reasoning tail, post-end
            // begins content. Strip stray duplicates from either half.
            let r = stripStrayMarkers(String(delta[..<endRange.lowerBound]))
            let c = stripStrayMarkers(String(delta[endRange.upperBound...]))
            return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
        }
        // Still implicit reasoning. Strip stray `<think>` markers
        // (model emitted a duplicate open).
        let stripped = stripStrayMarkers(delta)
        return stripped.isEmpty ? nil : ReasoningDelta(reasoning: stripped)
    }
}

// MARK: - Qwen3 / DeepSeek / Mistral

public final class Qwen3ReasoningParser: BaseThinkingReasoningParser {
    public init() { super.init(startToken: "<think>", endToken: "</think>") }

    public override func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        if !modelOutput.contains(endToken) {
            if modelOutput.contains(startToken) {
                return super.extractReasoning(modelOutput)
            }
            // Iter 144 — Laguna leak fix. When the chat template
            // pre-stamps `<think>` (think_in_template=true family —
            // Laguna, Qwen3.5, MiniMax, GLM-5.1, etc.), the model's
            // generation starts INSIDE the think block. The first
            // assistant token is reasoning content, but the model
            // hasn't (and won't) emit a fresh `<think>` opener
            // because the template already did. Pre-iter-144 this
            // branch returned `(nil, modelOutput)` — leaking the
            // entire reasoning prose as visible content. Honor the
            // base class's `thinkInPrompt` flag (set by
            // `resetState(thinkInPrompt: true, ...)` when the
            // request's chat template stamped the opener).
            if thinkInPrompt {
                return (modelOutput, nil)
            }
            return (nil, modelOutput)
        }
        return super.extractReasoning(modelOutput)
    }
}

public final class DeepSeekR1ReasoningParser: BaseThinkingReasoningParser {
    public init() { super.init(startToken: "<think>", endToken: "</think>") }

    public override func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        let hasStart = modelOutput.contains(startToken)
        let hasEnd = modelOutput.contains(endToken)

        // Lenient: end but no start → everything before end is reasoning
        if hasEnd && !hasStart {
            if let r = modelOutput.range(of: endToken) {
                let reasoning = String(modelOutput[..<r.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                let content = String(modelOutput[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            }
        }
        if !hasStart && !hasEnd {
            // Iter 144 — same think_in_template fix as Qwen3 above.
            // GLM-5.1 / Kimi-K2.6 thinking bundles + DSV4 reasoning
            // alias all route here and may have stamped `<think>` in
            // the chat template, leaving generation to start inside.
            if thinkInPrompt {
                return (modelOutput, nil)
            }
            return (nil, modelOutput)
        }
        return super.extractReasoning(modelOutput)
    }
}

public final class MistralReasoningParser: BaseThinkingReasoningParser {
    public init() { super.init(startToken: "[THINK]", endToken: "[/THINK]") }

    public override func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        if !modelOutput.contains(endToken) {
            if modelOutput.contains(startToken) {
                return super.extractReasoning(modelOutput)
            }
            // Iter 144 — Mistral 4 / ministral3 / Mistral 3 same
            // think_in_template family. Honor `thinkInPrompt` so
            // template-stamped `[THINK]` doesn't leak reasoning
            // into delta.content.
            if thinkInPrompt {
                return (modelOutput, nil)
            }
            return (nil, modelOutput)
        }
        return super.extractReasoning(modelOutput)
    }
}

// MARK: - Gemma 4 (channel markers)

public final class Gemma4ReasoningParser: ReasoningParser {
    private static let soc = "<|channel>"
    private static let socAlt = "<|channel|>"
    private static let eoc = "<channel|>"
    private static let eot = "<turn|>"
    private static let thoughtStart = "<|channel>thought"
    private static let thoughtStartAlt = "<|channel|>thought"
    private static let degradedThoughtStart = "thought\n"
    private static let degradedThoughtStartAlt = "|thought\n"
    private static let degradedThoughtBare = "thought"
    private static let degradedThoughtBareAlt = "|thought"
    private static let xmlThoughtStart = "<thought>"
    private static let xmlThoughtEnd = "</thought>"

    private var emittedReasoning: Int = 0
    private var emittedContent: Int = 0
    private var sawThought: Bool = false
    private var sawEoc: Bool = false

    public init() {}

    public func resetState(thinkInPrompt: Bool, harmonyActive: Bool) {
        emittedReasoning = 0
        emittedContent = 0
        // 2026-05-01 FIX — when the chat template already stamped
        // `<|channel>thought\n` as part of the assistant prefix, the
        // model's output stream STARTS inside the thought block. The
        // `<|channel>thought` marker never appears in the OUTPUT, only
        // in the prompt. Without this seed the parser falls into the
        // else branch (no thoughtStart found) and emits the entire
        // reasoning buffer as plain content, leaking the literal word
        // "thought" + whatever the model wrote before `<channel|>`.
        // Live-caught on Gemma-4-26B-A4B JANG_4M-CRACK 2026-05-01:
        // a "What is 12+15?" prompt produced `content='thought27'`
        // because the model emitted `thought\n…calc…<channel|>27` and
        // the parser only stripped `<channel|>` not the leading
        // "thought" body.
        sawThought = thinkInPrompt
        sawEoc = false
        // Also: when sawThought is seeded true, the streaming
        // accumulator must treat the buffer-prefix as reasoning
        // BEFORE the parser sees `<|channel>thought` literally. We
        // achieve this in `extractReasoningStreaming` by checking
        // `sawThought` early.
        thoughtInPromptSeed = thinkInPrompt
    }

    /// True when `resetState` was called with `thinkInPrompt=true`
    /// — i.e., the chat template already stamped the open `<|channel>thought`
    /// marker. Used by the streaming extractor to seed reasoning mode
    /// before any output tokens arrive.
    private var thoughtInPromptSeed: Bool = false

    /// End-of-stream flush for Gemma 4's lookahead buffering.
    ///
    /// `extractReasoningStreaming` holds back the first 18 chars of the
    /// accumulated text waiting to disambiguate a `<|channel>thought` marker
    /// from plain content. When the full response is shorter than 18 chars
    /// (e.g. "Two plus two equals four." at 26 chars works fine, but a
    /// very short reply would be stuck in the buffer) we'd lose the entire
    /// response. Even for replies that cross 18 chars, the parser can still
    /// hold back the final trailing segment if the last delta didn't cross
    /// an emission boundary.
    ///
    /// On finish, run the non-streaming extractor on `fullText` to get the
    /// authoritative (reasoning, content) split, then compute and emit the
    /// residual beyond what was already streamed through the delta path.
    public func finishStreaming(fullText: String) -> ReasoningDelta? {
        let (r, c) = extractReasoning(fullText)
        var residualReasoning: String? = nil
        var residualContent: String? = nil
        if let r, r.count > emittedReasoning {
            let idx = r.index(r.startIndex, offsetBy: emittedReasoning)
            residualReasoning = String(r[idx...])
            emittedReasoning = r.count
        }
        if let c, c.count > emittedContent {
            let idx = c.index(c.startIndex, offsetBy: emittedContent)
            residualContent = String(c[idx...])
            emittedContent = c.count
        }
        if residualReasoning == nil && residualContent == nil { return nil }
        return ReasoningDelta(reasoning: residualReasoning, content: residualContent)
    }

    public func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        var text = modelOutput
        while text.hasSuffix(Self.eot) { text.removeLast(Self.eot.count) }

        if let thoughtRange = Self.firstThoughtStartRange(in: text) {
            var afterSoc = String(text[thoughtRange.upperBound...])
            if afterSoc.hasPrefix("\n") { afterSoc.removeFirst() }

            if let eocRange = afterSoc.range(of: Self.eoc) {
                var reasoning = String(afterSoc[..<eocRange.lowerBound])
                var content = String(afterSoc[eocRange.upperBound...])
                while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
                reasoning = stripResidualMarkers(reasoning).trimmingCharacters(in: .whitespacesAndNewlines)
                content = stripResidualMarkers(content).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            } else {
                let reasoning = stripResidualMarkers(afterSoc).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, nil)
            }
        }
        if let thoughtRange = Self.degradedThoughtStartRange(in: text) {
            let afterSoc = String(text[thoughtRange.upperBound...])
            if let eocRange = afterSoc.range(of: Self.eoc) {
                var reasoning = String(afterSoc[..<eocRange.lowerBound])
                var content = String(afterSoc[eocRange.upperBound...])
                while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
                reasoning = stripResidualMarkers(reasoning).trimmingCharacters(in: .whitespacesAndNewlines)
                content = stripResidualMarkers(content).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            } else {
                let reasoning = stripResidualMarkers(afterSoc).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, nil)
            }
        }
        if let thoughtRange = text.range(of: Self.xmlThoughtStart) {
            var afterSoc = String(text[thoughtRange.upperBound...])
            if afterSoc.hasPrefix("\n") { afterSoc.removeFirst() }

            if let endRange = afterSoc.range(of: Self.xmlThoughtEnd) {
                var reasoning = String(afterSoc[..<endRange.lowerBound])
                var content = String(afterSoc[endRange.upperBound...])
                while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
                reasoning = stripResidualMarkers(reasoning).trimmingCharacters(in: .whitespacesAndNewlines)
                content = stripResidualMarkers(content).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            } else {
                let reasoning = stripResidualMarkers(afterSoc).trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, nil)
            }
        }

        // 2026-05-01 FIX — when the chat template already stamped the
        // `<|channel>thought\n` open marker in the PROMPT, the output
        // stream starts AFTER that marker. The model's text begins
        // inside reasoning mode and we should split on the FIRST
        // `<channel|>` close: everything before is reasoning, after
        // is content.
        if thoughtInPromptSeed, let eocRange = text.range(of: Self.eoc) {
            var reasoning = String(text[..<eocRange.lowerBound])
            var content = String(text[eocRange.upperBound...])
            while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
            reasoning = stripResidualMarkers(reasoning).trimmingCharacters(in: .whitespacesAndNewlines)
            content = stripResidualMarkers(content).trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
        }
        if thoughtInPromptSeed {
            // No close yet — entire output is reasoning.
            let reasoning = stripResidualMarkers(text).trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, nil)
        }

        // No thoughtStart found AND not seeded — could still contain
        // stray `<|channel>` / `<channel|>` fragments from the model
        // (we observe this on Gemma 4 31B JANG_4M). Strip before
        // returning as plain content so the §15 reasoning-off reroute
        // doesn't surface them.
        let cleaned = stripResidualMarkers(text).trimmingCharacters(in: .whitespacesAndNewlines)
        return (nil, cleaned.isEmpty ? nil : cleaned)
    }

    public func extractReasoningStreaming(previous: String, current: String, delta: String) -> ReasoningDelta? {
        if delta.isEmpty { return nil }

        let (reasoningText, contentText) = parseAccumulated(current)
        let thoughtInCurrent =
            Self.firstThoughtStartRange(in: current) != nil
            || Self.degradedThoughtStartRange(in: current) != nil
            || current.contains(Self.xmlThoughtStart)
        let eocInCurrent = current.contains(Self.eoc)
            || current.contains(Self.xmlThoughtEnd)

        if thoughtInCurrent && !sawThought { sawThought = true }
        if eocInCurrent && sawThought && !sawEoc { sawEoc = true }

        // §241f: No `<|channel>thought` seen yet. Earlier we flushed the
        // accumulated buffer as content once it crossed 18 chars, but that
        // leaked partial marker fragments on models whose chat template
        // emits the marker late (Gemma-4-31B-JANG_4M is the canonical
        // offender — C2 content ended up `}hellonel>thought\n<|channel>...`
        // because deltas 1-N contained `<`, `|`, `channel`, `>`, `thought`
        // as separate tokens and the pre-sawThought branch emitted them
        // raw). Hold back the LAST 18 chars (length of `<|channel>thought`)
        // from emission: that suffix is always safe to park until either
        // the marker completes (sawThought flips) or 18+ more bytes
        // arrive that can't be a marker prefix.
        if !sawThought {
            if current.count < 18 { return nil }
            var cleaned = current.replacingOccurrences(of: Self.eot, with: "")
            // Strip any FULLY-FORMED markers that already appeared
            // (shouldn't happen in !sawThought, but defensive).
            cleaned = stripResidualMarkers(cleaned)
            cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
            // Hold back the last 18 chars in case they're the start of a
            // `<|channel>thought` marker still being streamed.
            guard cleaned.count > 18 else { return nil }
            let safeEnd = cleaned.index(cleaned.endIndex, offsetBy: -18)
            let safe = String(cleaned[..<safeEnd])
            if !safe.isEmpty && safe.count > emittedContent {
                let startIdx = safe.index(safe.startIndex, offsetBy: emittedContent)
                let new = String(safe[startIdx...])
                emittedContent = safe.count
                return new.isEmpty ? nil : ReasoningDelta(content: new)
            }
            return nil
        }

        // Skip pure special-token deltas
        let stripped = delta.trimmingCharacters(in: .whitespacesAndNewlines)
        if stripped == Self.soc || stripped == Self.eoc || stripped == Self.eot ||
           stripped == Self.socAlt ||
           stripped == Self.thoughtStart || stripped == Self.thoughtStart + "\n" ||
           stripped == Self.thoughtStartAlt || stripped == Self.thoughtStartAlt + "\n" ||
           stripped == Self.xmlThoughtStart || stripped == Self.xmlThoughtEnd {
            return nil
        }

        var newReasoning: String?
        var newContent: String?

        if let r = reasoningText {
            if r.count > emittedReasoning {
                let idx = r.index(r.startIndex, offsetBy: emittedReasoning)
                newReasoning = String(r[idx...])
                emittedReasoning = r.count
            } else if r.count < emittedReasoning {
                emittedReasoning = r.count
            }
        }
        if let c = contentText, c.count > emittedContent {
            let idx = c.index(c.startIndex, offsetBy: emittedContent)
            newContent = String(c[idx...])
            emittedContent = c.count
        }

        if newReasoning != nil || newContent != nil {
            return ReasoningDelta(reasoning: newReasoning, content: newContent)
        }
        return nil
    }

    private func parseAccumulated(_ input: String) -> (String?, String?) {
        var text = input
        while text.hasSuffix(Self.eot) { text.removeLast(Self.eot.count) }

        // 2026-05-01 FIX — when chat template stamped `<|channel>thought\n`
        // in the prompt, output starts inside reasoning. Look for the
        // close `<channel|>` once. If absent, return all-reasoning
        // (cheap — no extra full-string scans). Live-caught on
        // Gemma-4-26B-A4B JANG_4M-CRACK: content="thought27".
        if thoughtInPromptSeed {
            if let eocRange = text.range(of: Self.eoc) {
                let reasoning = String(text[..<eocRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                var content = String(text[eocRange.upperBound...])
                while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
                content = content.trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning,
                        content.isEmpty ? nil : content)
            }
            // No close yet — entire text is reasoning.
            // Skip stripResidualMarkers/stripPartialEoc on hot path
            // (per-token call); finishStreaming flush handles trailing
            // partial markers.
            let r = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return (r.isEmpty ? nil : r, nil)
        }

        let afterSoc: String
        let closeToken: String
        if let thoughtRange = Self.firstThoughtStartRange(in: text) {
            var full = String(text[thoughtRange.upperBound...])
            if full.hasPrefix("\n") { full.removeFirst() }
            afterSoc = full
            closeToken = Self.eoc
        } else if let thoughtRange = Self.degradedThoughtStartRange(in: text) {
            afterSoc = String(text[thoughtRange.upperBound...])
            closeToken = Self.eoc
        } else if let thoughtRange = text.range(of: Self.xmlThoughtStart) {
            var full = String(text[thoughtRange.upperBound...])
            if full.hasPrefix("\n") { full.removeFirst() }
            afterSoc = full
            closeToken = Self.xmlThoughtEnd
        } else {
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return (nil, t.isEmpty ? nil : t)
        }

        if let closeRange = afterSoc.range(of: closeToken) {
            var reasoning = String(afterSoc[..<closeRange.lowerBound])
            var content = String(afterSoc[closeRange.upperBound...])
            while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
            // §241b: models sometimes re-emit `<|channel>thought` inside
            // the reasoning block (observed live on Gemma-4-31B-JANG_4M).
            // Also strip stray `<|channel>` / `<channel|>` leftovers so
            // nothing structural leaks into either side when the Stream
            // reroutes reasoning → content under `!effectiveThinking`.
            reasoning = stripResidualMarkers(reasoning)
            content = stripResidualMarkers(content)
            reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
            content = content.trimmingCharacters(in: .whitespacesAndNewlines)
            return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
        } else {
            var partial = stripPartialEoc(afterSoc)
            partial = stripResidualMarkers(partial)
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            return (partial.isEmpty ? nil : partial, nil)
        }
    }

    /// Remove stray `<|channel>thought`, `<|channel>`, and `<channel|>`
    /// occurrences left behind when the model emits multiple reasoning
    /// blocks or partial markers inside the reasoning text. Without this
    /// the §15 reasoning-off re-route surfaces raw markers as visible
    /// content (observed on M11 Gemma-4-31B-JANG_4M, C2/C6/C8 fail).
    private func stripResidualMarkers(_ s: String) -> String {
        var out = s
        out = out.replacingOccurrences(of: Self.thoughtStart, with: "")
        out = out.replacingOccurrences(of: Self.thoughtStartAlt, with: "")
        out = out.replacingOccurrences(of: Self.soc, with: "")
        out = out.replacingOccurrences(of: Self.socAlt, with: "")
        out = out.replacingOccurrences(of: Self.eoc, with: "")
        out = out.replacingOccurrences(of: Self.xmlThoughtStart, with: "")
        out = out.replacingOccurrences(of: Self.xmlThoughtEnd, with: "")
        if out.hasPrefix(Self.degradedThoughtStartAlt) {
            out.removeFirst(Self.degradedThoughtStartAlt.count)
        }
        if out.hasPrefix(Self.degradedThoughtStart) {
            out.removeFirst(Self.degradedThoughtStart.count)
        }
        if let range = Self.degradedThoughtStartRange(in: out),
           range.lowerBound == out.startIndex {
            out.removeSubrange(range)
        }
        return out
    }

    private func stripPartialEoc(_ s: String) -> String {
        // §241e: strip partial trailing occurrences of ANY of the Gemma 4
        // markers, not just `<channel|>` (eoc). During token-by-token
        // streaming the reassembly of `<|channel>thought` or `<|channel>`
        // passes through every intermediate prefix (`<`, `<|`, `<|c`,
        // `<|ch`...`<|channel`). The previous implementation only checked
        // eoc, so prefixes like `<|channel` got emitted as reasoning →
        // leaked as content under the §15 reasoning-off reroute.
        let markers = [
            Self.thoughtStart,
            Self.thoughtStartAlt,
            Self.degradedThoughtStart,
            Self.degradedThoughtStartAlt,
            Self.soc,
            Self.socAlt,
            Self.eoc,
            Self.xmlThoughtStart,
            Self.xmlThoughtEnd,
        ]
        for marker in markers {
            var len = marker.count - 1
            while len > 0 {
                let prefix = String(marker.prefix(len))
                if s.hasSuffix(prefix) {
                    return String(s.dropLast(len))
                }
                len -= 1
            }
        }
        return s
    }

    private static func firstThoughtStartRange(in text: String) -> Range<String.Index>? {
        let ranges = [text.range(of: thoughtStart), text.range(of: thoughtStartAlt)]
            .compactMap { $0 }
        return ranges.min { lhs, rhs in lhs.lowerBound < rhs.lowerBound }
    }

    private static func degradedThoughtStartRange(in text: String) -> Range<String.Index>? {
        guard let first = text.firstIndex(where: { !$0.isWhitespace }) else { return nil }
        let rest = text[first...]
        if rest.hasPrefix(degradedThoughtStart) {
            return first..<text.index(first, offsetBy: degradedThoughtStart.count)
        }
        if rest.hasPrefix(degradedThoughtStartAlt) {
            return first..<text.index(first, offsetBy: degradedThoughtStartAlt.count)
        }
        if hasBareDegradedThoughtPrefix(rest, marker: degradedThoughtBare) {
            return first..<text.index(first, offsetBy: degradedThoughtBare.count)
        }
        if hasBareDegradedThoughtPrefix(rest, marker: degradedThoughtBareAlt) {
            return first..<text.index(first, offsetBy: degradedThoughtBareAlt.count)
        }
        return nil
    }

    private static func hasBareDegradedThoughtPrefix(
        _ rest: Substring,
        marker: String
    ) -> Bool {
        guard rest.hasPrefix(marker) else { return false }
        let after = rest.dropFirst(marker.count)
        guard let next = after.unicodeScalars.first else { return true }
        // The tokenizer can drop the special channel delimiters and newline,
        // leaving `thoughtThe answer...`. Do not strip ordinary words like
        // `thoughtful`, where the next scalar is lowercase.
        return !CharacterSet.lowercaseLetters.contains(next)
    }
}

// MARK: - GPT-OSS (Harmony protocol)

public final class GptOssReasoningParser: ReasoningParser {
    private static let channelTag = "<|channel|>"
    private static let messageTag = "<|message|>"
    private static let startTag = "<|start|>"
    private static let analysisMarker = "<|channel|>analysis<|message|>"
    private static let finalMarker = "<|channel|>final<|message|>"

    private var emittedReasoning: Int = 0
    private var emittedContent: Int = 0
    private var sawMarker: Bool = false
    private var fallbackEmitted: Int = 0
    private var harmonyActive: Bool = false
    private let fallbackThreshold = 3

    public init() {}

    public func resetState(thinkInPrompt: Bool, harmonyActive: Bool) {
        emittedReasoning = 0
        emittedContent = 0
        sawMarker = false
        fallbackEmitted = 0
        self.harmonyActive = harmonyActive
    }

    public func extractReasoning(_ modelOutput: String) -> (reasoning: String?, content: String?) {
        let (reasoningParts, contentParts) = parseChannels(modelOutput)
        if reasoningParts.isEmpty && contentParts.isEmpty {
            return (nil, modelOutput)
        }
        let reasoning = reasoningParts.isEmpty ? nil : reasoningParts.joined(separator: "\n")
        let content = contentParts.isEmpty ? nil : contentParts.joined(separator: "\n")
        return (reasoning, content)
    }

    public func extractReasoningStreaming(previous: String, current: String, delta: String) -> ReasoningDelta? {
        if delta.isEmpty { return nil }
        let (reasoningParts, contentParts) = parseChannels(current)
        if !sawMarker && (!reasoningParts.isEmpty || !contentParts.isEmpty) {
            sawMarker = true
        }
        if harmonyActive { sawMarker = true }

        if !sawMarker {
            if current.count < fallbackThreshold { return nil }
            if current.count > fallbackEmitted {
                let idx = current.index(current.startIndex, offsetBy: fallbackEmitted)
                let new = String(current[idx...])
                fallbackEmitted = current.count
                return ReasoningDelta(content: new)
            }
            return nil
        }

        let reasoningSoFar = reasoningParts.isEmpty ? nil : reasoningParts.joined(separator: "\n")
        let contentSoFar = contentParts.isEmpty ? nil : contentParts.joined(separator: "\n")

        var newReasoning: String?
        var newContent: String?
        if let r = reasoningSoFar {
            if r.count > emittedReasoning {
                let idx = r.index(r.startIndex, offsetBy: emittedReasoning)
                newReasoning = String(r[idx...])
                emittedReasoning = r.count
            } else if r.count < emittedReasoning {
                emittedReasoning = r.count
            }
        }
        if let c = contentSoFar, c.count > emittedContent {
            let idx = c.index(c.startIndex, offsetBy: emittedContent)
            newContent = String(c[idx...])
            emittedContent = c.count
        }
        if newReasoning != nil || newContent != nil {
            return ReasoningDelta(reasoning: newReasoning, content: newContent)
        }
        return nil
    }

    private func parseChannels(_ input: String) -> (reasoning: [String], content: [String]) {
        var reasoning: [String] = []
        var content: [String] = []
        var text = input

        // Strip leading <|start|>assistant
        let startAssistant = Self.startTag + "assistant"
        if text.hasPrefix(startAssistant) {
            text = String(text.dropFirst(startAssistant.count))
        }

        if !text.contains(Self.channelTag) {
            if harmonyActive {
                let safe = stripPartialMarker(text)
                let trimmed = safe.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { reasoning.append(trimmed) }
            }
            return (reasoning, content)
        }

        // Text before the first marker is implicit reasoning
        let firstChannel = text.range(of: Self.channelTag)?.lowerBound
        let firstStart = text.range(of: Self.startTag)?.lowerBound
        var firstMarker: String.Index? = firstChannel
        if let fs = firstStart {
            if firstMarker == nil || fs < firstMarker! { firstMarker = fs }
        }
        if let fm = firstMarker, fm > text.startIndex {
            let pre = String(text[..<fm]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !pre.isEmpty { reasoning.append(pre) }
            text = String(text[fm...])
        }

        var foundFinal = false
        while let chanRange = text.range(of: Self.channelTag) {
            text = String(text[chanRange.upperBound...])
            if text.hasPrefix("analysis" + Self.messageTag) {
                if foundFinal { break }
                text = String(text.dropFirst(("analysis" + Self.messageTag).count))
                if let next = findNextChannel(text) {
                    let part = String(text[..<next]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !part.isEmpty { reasoning.append(part) }
                    text = String(text[next...])
                } else {
                    let safe = stripPartialMarker(text).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !safe.isEmpty { reasoning.append(safe) }
                    text = ""
                }
            } else if text.hasPrefix("final" + Self.messageTag) {
                text = String(text.dropFirst(("final" + Self.messageTag).count))
                foundFinal = true
                if let next = findNextChannel(text) {
                    let part = String(text[..<next]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !part.isEmpty { content.append(part) }
                    break
                } else {
                    let safe = stripPartialMarker(text).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !safe.isEmpty { content.append(safe) }
                    text = ""
                }
            } else {
                break
            }
        }
        return (reasoning, content)
    }

    private func findNextChannel(_ s: String) -> String.Index? {
        let positions: [String.Index?] = [
            s.range(of: Self.channelTag)?.lowerBound,
            s.range(of: Self.startTag)?.lowerBound,
        ]
        return positions.compactMap { $0 }.min()
    }

    private func stripPartialMarker(_ s: String) -> String {
        var text = s
        for tag in [Self.startTag, Self.channelTag, Self.messageTag] {
            var len = tag.count - 1
            while len > 0 {
                let prefix = String(tag.prefix(len))
                if text.hasSuffix(prefix) {
                    text = String(text.dropLast(len))
                    break
                }
                len -= 1
            }
        }
        if text.trimmingCharacters(in: .whitespaces).hasSuffix("assistant") {
            text = text.trimmingCharacters(in: .whitespaces)
            text = String(text.dropLast("assistant".count))
        }
        return text
    }
}
