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

    private func handleExplicitThink(delta: String, startInPrev: Bool, endInPrev: Bool, endInDelta: Bool) -> ReasoningDelta? {
        let startInDelta = delta.contains(startToken)

        if startInPrev {
            if endInDelta, let endRange = delta.range(of: endToken) {
                let r = String(delta[..<endRange.lowerBound])
                let c = String(delta[endRange.upperBound...])
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
            } else if endInPrev {
                return ReasoningDelta(content: delta)
            } else {
                return ReasoningDelta(reasoning: delta)
            }
        }

        if startInDelta, let startRange = delta.range(of: startToken) {
            if endInDelta, let endRange = delta.range(of: endToken), endRange.lowerBound >= startRange.upperBound {
                let r = String(delta[startRange.upperBound..<endRange.lowerBound])
                let c = String(delta[endRange.upperBound...])
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
            } else {
                let r = String(delta[startRange.upperBound...])
                return ReasoningDelta(reasoning: r.isEmpty ? nil : r)
            }
        }

        return ReasoningDelta(content: delta)
    }

    private func handleImplicitThink(delta: String, endInPrev: Bool, endInDelta: Bool) -> ReasoningDelta? {
        if endInDelta, let endRange = delta.range(of: endToken) {
            let r = String(delta[..<endRange.lowerBound])
            let c = String(delta[endRange.upperBound...])
            return ReasoningDelta(reasoning: r.isEmpty ? nil : r, content: c.isEmpty ? nil : c)
        } else if endInPrev {
            return ReasoningDelta(content: delta)
        } else {
            return ReasoningDelta(reasoning: delta)
        }
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
            return (nil, modelOutput)
        }
        return super.extractReasoning(modelOutput)
    }
}

// MARK: - Gemma 4 (channel markers)

public final class Gemma4ReasoningParser: ReasoningParser {
    private static let soc = "<|channel>"
    private static let eoc = "<channel|>"
    private static let eot = "<turn|>"
    private static let thoughtStart = "<|channel>thought"

    private var emittedReasoning: Int = 0
    private var emittedContent: Int = 0
    private var sawThought: Bool = false
    private var sawEoc: Bool = false

    public init() {}

    public func resetState(thinkInPrompt: Bool, harmonyActive: Bool) {
        emittedReasoning = 0
        emittedContent = 0
        sawThought = false
        sawEoc = false
    }

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

        if let thoughtRange = text.range(of: Self.thoughtStart) {
            var afterSoc = String(text[thoughtRange.upperBound...])
            if afterSoc.hasPrefix("\n") { afterSoc.removeFirst() }

            if let eocRange = afterSoc.range(of: Self.eoc) {
                let reasoning = String(afterSoc[..<eocRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                var content = String(afterSoc[eocRange.upperBound...])
                while content.hasSuffix(Self.eot) { content.removeLast(Self.eot.count) }
                content = content.trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, content.isEmpty ? nil : content)
            } else {
                let reasoning = afterSoc.trimmingCharacters(in: .whitespacesAndNewlines)
                return (reasoning.isEmpty ? nil : reasoning, nil)
            }
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return (nil, trimmed.isEmpty ? nil : trimmed)
    }

    public func extractReasoningStreaming(previous: String, current: String, delta: String) -> ReasoningDelta? {
        if delta.isEmpty { return nil }

        let (reasoningText, contentText) = parseAccumulated(current)
        let thoughtInCurrent = current.contains(Self.thoughtStart)
        let eocInCurrent = current.contains(Self.eoc)

        if thoughtInCurrent && !sawThought { sawThought = true }
        if eocInCurrent && sawThought && !sawEoc { sawEoc = true }

        // No thought seen: buffer until we know it isn't a marker, then flush as content
        if !sawThought {
            if current.count < 18 { return nil }
            let cleaned = current.replacingOccurrences(of: Self.eot, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty && cleaned.count > emittedContent {
                let startIdx = cleaned.index(cleaned.startIndex, offsetBy: emittedContent)
                let new = String(cleaned[startIdx...])
                emittedContent = cleaned.count
                return new.isEmpty ? nil : ReasoningDelta(content: new)
            }
            return nil
        }

        // Skip pure special-token deltas
        let stripped = delta.trimmingCharacters(in: .whitespacesAndNewlines)
        if stripped == Self.soc || stripped == Self.eoc || stripped == Self.eot ||
           stripped == Self.thoughtStart || stripped == Self.thoughtStart + "\n" {
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

        guard let thoughtRange = text.range(of: Self.thoughtStart) else {
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return (nil, t.isEmpty ? nil : t)
        }
        var afterSoc = String(text[thoughtRange.upperBound...])
        if afterSoc.hasPrefix("\n") { afterSoc.removeFirst() }

        if let eocRange = afterSoc.range(of: Self.eoc) {
            var reasoning = String(afterSoc[..<eocRange.lowerBound])
            var content = String(afterSoc[eocRange.upperBound...])
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
        out = out.replacingOccurrences(of: Self.soc, with: "")
        out = out.replacingOccurrences(of: Self.eoc, with: "")
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
        let markers = [Self.thoughtStart, Self.soc, Self.eoc]
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
