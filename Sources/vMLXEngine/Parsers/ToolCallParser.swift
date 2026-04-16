// SPDX-License-Identifier: Apache-2.0
// Ported from vmlx_engine/tool_parsers/*.py.
//
// Each parser handles extraction of tool calls from a complete model output
// (the server uses a buffer-then-parse strategy, so streaming parsers just
// detect marker presence and delegate to extractToolCalls once the full call
// is buffered).
//
// Parsers ported:
//   hermes / nous, qwen / qwen3, llama / llama3 / llama4, mistral,
//   deepseek / deepseek_v3 / deepseek_r1, kimi / kimi_k2 / moonshot,
//   granite / granite3, nemotron / nemotron3, xlam, functionary / meetkai,
//   glm47 / glm4, step3p5 / stepfun, minimax / minimax_m2, gemma4,
//   native (raw JSON passthrough).

import Foundation

/// Result of extracting tool calls from a complete model output.
public struct ExtractedToolCallInformation: Sendable, Equatable {
    public var toolsCalled: Bool
    public var toolCalls: [ParsedToolCall]
    public var content: String?

    public init(toolsCalled: Bool, toolCalls: [ParsedToolCall], content: String?) {
        self.toolsCalled = toolsCalled
        self.toolCalls = toolCalls
        self.content = content
    }
}

/// A single parsed tool call. The `id` follows OpenAI format `call_<8hex>`.
public struct ParsedToolCall: Sendable, Equatable {
    public var id: String
    public var name: String
    public var arguments: String  // JSON string

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    public func toWire() -> ChatRequest.ToolCall {
        ChatRequest.ToolCall(
            id: id,
            type: "function",
            function: .init(name: name, arguments: arguments)
        )
    }
}

/// Generate an OpenAI-style `call_<8hex>` id.
public func generateToolId() -> String {
    let hex = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    return "call_" + String(hex.prefix(8))
}

public protocol ToolCallParser: AnyObject {
    func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation
}

public extension ToolCallParser {
    func extractToolCalls(_ modelOutput: String) -> ExtractedToolCallInformation {
        extractToolCalls(modelOutput, request: nil)
    }
}

// MARK: - Shared utilities

enum ParserUtils {
    /// Mirrors `ToolParser.strip_think_tags` — removes both `<think>...</think>`
    /// and `[THINK]...[/THINK]` blocks, including the implicit-closing case and
    /// the unclosed-open case.
    static func stripThinkTags(_ text: String) -> String {
        var result = text

        // Full: <think>...</think> | [THINK]...[/THINK]
        let fullPattern = #"(?s)(?:<think>.*?</think>|\[THINK\].*?\[/THINK\])"#
        result = regexReplace(result, pattern: fullPattern, with: "")

        // Implicit closing only
        if result == text && (text.contains("</think>") || text.contains("[/THINK]")) {
            result = regexReplace(text, pattern: #"(?s)^.*?(?:</think>|\[/THINK\])"#, with: "")
        }

        // Unclosed open
        if result.contains("<think>") || result.contains("[THINK]") {
            result = regexReplace(result, pattern: #"(?s)(?:<think>|\[THINK\]).*$"#, with: "")
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func regexReplace(_ input: String, pattern: String, with: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return input }
        let range = NSRange(input.startIndex..., in: input)
        return re.stringByReplacingMatches(in: input, range: range, withTemplate: with)
    }

    static func regexMatches(_ input: String, pattern: String) -> [[String]] {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(input.startIndex..., in: input)
        return re.matches(in: input, range: range).map { m in
            (0..<m.numberOfRanges).compactMap { i -> String? in
                let r = m.range(at: i)
                guard r.location != NSNotFound, let sr = Range(r, in: input) else { return nil }
                return String(input[sr])
            }
        }
    }

    /// JSON re-serialise to guarantee valid JSON output (or pass through the
    /// original string if it's already valid JSON). Used to normalise
    /// `arguments` strings.
    static func normaliseArgsJSON(_ args: Any) -> String {
        if let s = args as? String {
            if let _ = try? JSONSerialization.jsonObject(with: Data(s.utf8)) { return s }
            if let data = try? JSONSerialization.data(withJSONObject: s) {
                return String(data: data, encoding: .utf8) ?? "{}"
            }
            return "{}"
        }
        if JSONSerialization.isValidJSONObject(args),
           let data = try? JSONSerialization.data(withJSONObject: args) {
            return String(data: data, encoding: .utf8) ?? "{}"
        }
        return "{}"
    }

    /// Parse a JSON string into a dictionary (for tool call bodies).
    static func parseJSONObject(_ s: String) -> [String: Any]? {
        guard let data = s.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj
    }
}

// MARK: - Hermes / Nous / Raw JSON fallback (also used as "native" JSON)

public final class HermesToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var cleaned = ParserUtils.stripThinkTags(modelOutput)
        var calls: [ParsedToolCall] = []

        // <tool_call>{...}</tool_call>
        let tagged = ParserUtils.regexMatches(cleaned, pattern: #"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>"#)
        for m in tagged where m.count >= 2 {
            if let tc = parseHermesCall(m[1]) { calls.append(tc) }
        }
        if !tagged.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: #"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>"#, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Raw JSON fallback: {"name":"...","arguments":{...}}
        if calls.isEmpty, let tc = extractRawJSONCall(cleaned) {
            calls.append(tc.0)
            cleaned = tc.1
        }

        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: cleaned)
    }

    private func parseHermesCall(_ jsonStr: String) -> ParsedToolCall? {
        guard let obj = ParserUtils.parseJSONObject(jsonStr),
              let name = obj["name"] as? String, !name.isEmpty
        else { return nil }
        let args = obj["arguments"] ?? [:]
        return ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(args))
    }

    private func extractRawJSONCall(_ text: String) -> (ParsedToolCall, String)? {
        let pattern = #"\{"name":\s*"([^"]+)",\s*"arguments":\s*\{"#
        let matches = ParserUtils.regexMatches(text, pattern: pattern)
        guard let m = matches.first, m.count >= 2 else { return nil }
        let name = m[1]
        // Find balanced JSON object starting at the match location.
        guard let nameRange = text.range(of: m[0]) else { return nil }
        let startIdx = nameRange.lowerBound
        let bytes = Array(text[startIdx...])
        var depth = 0
        var end: Int? = nil
        var inString = false
        var escape = false
        for (i, ch) in bytes.enumerated() {
            if escape { escape = false; continue }
            if ch == "\\" { escape = true; continue }
            if ch == "\"" { inString.toggle(); continue }
            if inString { continue }
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 { end = i; break }
            }
        }
        guard let e = end else { return nil }
        let jsonStr = String(bytes[0...e])
        guard let obj = ParserUtils.parseJSONObject(jsonStr) else { return nil }
        let args = obj["arguments"] ?? [:]
        let tc = ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(args))
        var remaining = String(text[..<startIdx])
        let afterIdx = bytes.index(bytes.startIndex, offsetBy: e + 1)
        remaining += String(bytes[afterIdx...])
        return (tc, remaining.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

// MARK: - Native (OpenAI JSON array) parser — passthrough used by ToolCallExtractor

public final class NativeToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        // Delegate to Hermes for the raw JSON fallback path.
        return HermesToolCallParser().extractToolCalls(modelOutput, request: request)
    }
}

// MARK: - Qwen3 / Qwen

public final class QwenToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var cleaned = ParserUtils.stripThinkTags(modelOutput)
        var calls: [ParsedToolCall] = []

        // [Calling tool: name({...})]
        let brackets = ParserUtils.regexMatches(cleaned, pattern: #"(?s)\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"#)
        for m in brackets where m.count >= 3 {
            if let obj = ParserUtils.parseJSONObject(m[2]) {
                calls.append(ParsedToolCall(
                    id: generateToolId(),
                    name: m[1],
                    arguments: ParserUtils.normaliseArgsJSON(obj)
                ))
            }
        }
        if !brackets.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: #"(?s)\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"#, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // <tool_call>{json}</tool_call>
        let xml = ParserUtils.regexMatches(cleaned, pattern: #"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>"#)
        for m in xml where m.count >= 2 {
            if let obj = ParserUtils.parseJSONObject(m[1]),
               let name = obj["name"] as? String {
                calls.append(ParsedToolCall(
                    id: generateToolId(),
                    name: name,
                    arguments: ParserUtils.normaliseArgsJSON(obj["arguments"] ?? [:])
                ))
            }
        }
        if !xml.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: #"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>"#, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
    }
}

// MARK: - Llama3 / Llama4: <function=name>{...}</function>

public final class LlamaToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        let matches = ParserUtils.regexMatches(modelOutput, pattern: #"(?s)<function=([^>]+)>(\{.*?\})</function>"#)
        if matches.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        var calls: [ParsedToolCall] = []
        for m in matches where m.count >= 3 {
            let name = m[1]
            let args = ParserUtils.parseJSONObject(m[2]) ?? [:]
            calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(args)))
        }
        let cleaned = ParserUtils.regexReplace(modelOutput, pattern: #"(?s)<function=([^>]+)>(\{.*?\})</function>"#, with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ExtractedToolCallInformation(toolsCalled: !calls.isEmpty, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
    }
}

// MARK: - Mistral: [TOOL_CALLS] prefix

public final class MistralToolCallParser: ToolCallParser {
    public static let botToken = "[TOOL_CALLS]"
    public init() {}

    private func generateId() -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0..<9).map { _ in chars.randomElement()! })
    }

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        guard modelOutput.contains(Self.botToken) else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        let parts = modelOutput.components(separatedBy: Self.botToken)
        let content = parts.first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var calls: [ParsedToolCall] = []

        for raw in parts.dropFirst() {
            let rawTC = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if rawTC.isEmpty { continue }

            // New format: func_name{"arg": "value"} (may include [ARGS])
            if !rawTC.hasPrefix("["), let brace = rawTC.firstIndex(of: "{") {
                var name = String(rawTC[..<brace]).trimmingCharacters(in: .whitespacesAndNewlines)
                name = name.replacingOccurrences(of: "[ARGS]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                let argsStr = String(rawTC[brace...])
                if !name.isEmpty, let _ = try? JSONSerialization.jsonObject(with: Data(argsStr.utf8)) {
                    calls.append(ParsedToolCall(id: generateId(), name: name, arguments: argsStr))
                    continue
                }
            }

            // Old format: [{"name":"...","arguments":{...}}]
            if let data = rawTC.data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                for item in arr {
                    if let name = item["name"] as? String {
                        calls.append(ParsedToolCall(
                            id: generateId(),
                            name: name,
                            arguments: ParserUtils.normaliseArgsJSON(item["arguments"] ?? [:])
                        ))
                    }
                }
                continue
            }

            // Regex fallback [{...}]
            let match = ParserUtils.regexMatches(rawTC, pattern: #"(?s)\[\{.*\}\]"#)
            if let m = match.first, let first = m.first,
               let data = first.data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                for item in arr {
                    if let name = item["name"] as? String {
                        calls.append(ParsedToolCall(
                            id: generateId(),
                            name: name,
                            arguments: ParserUtils.normaliseArgsJSON(item["arguments"] ?? [:])
                        ))
                    }
                }
            }
        }

        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: content.isEmpty ? nil : content)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
    }
}

// MARK: - DeepSeek (<｜tool▁calls▁begin｜> ...)

public final class DeepSeekToolCallParser: ToolCallParser {
    public static let callsBegin = "<｜tool▁calls▁begin｜>"
    public static let callsEnd = "<｜tool▁calls▁end｜>"
    public static let callBegin = "<｜tool▁call▁begin｜>"
    public static let callEnd = "<｜tool▁call▁end｜>"
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        // Round 16 / Rank 8: DeepSeek R1 emits `<think>...</think>` blocks
        // before tool-call markers. Strip reasoning blocks first so a mention
        // of the marker inside the think block never false-matches, and the
        // cleaned prefix content we return to callers doesn't contain dangling
        // think tags. Parity with Python v1.3.54 deepseek_v3 parser.
        let source = ParserUtils.stripThinkTags(modelOutput)
        guard source.contains(Self.callsBegin) else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        let cleanedContent: String
        if let r = source.range(of: Self.callsBegin) {
            cleanedContent = String(source[..<r.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            cleanedContent = ""
        }
        let pattern = #"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>"#
        var calls: [ParsedToolCall] = []
        for m in ParserUtils.regexMatches(source, pattern: pattern) where m.count >= 4 {
            let name = m[2].trimmingCharacters(in: .whitespacesAndNewlines)
            let args = m[3]
            calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: args))
        }
        // Simple fallback: <｜tool▁call▁begin｜>name\n{...}<｜tool▁call▁end｜>
        if calls.isEmpty {
            let simple = #"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>"#
            for m in ParserUtils.regexMatches(source, pattern: simple) where m.count >= 2 {
                let body = m[1]
                // Extract name (first line) and args (rest if JSON-like)
                let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
                if let brace = trimmed.firstIndex(of: "{") {
                    let name = trimmed[..<brace].trimmingCharacters(in: .whitespacesAndNewlines)
                    let args = String(trimmed[brace...])
                    if !name.isEmpty {
                        calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: args))
                    }
                }
            }
        }
        if calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        return ExtractedToolCallInformation(
            toolsCalled: true,
            toolCalls: calls,
            content: cleanedContent.isEmpty ? nil : cleanedContent
        )
    }
}

// MARK: - Kimi (Moonshot)

public final class KimiToolCallParser: ToolCallParser {
    public static let sectionBegin = "<|tool_calls_section_begin|>"
    public static let sectionBeginAlt = "<|tool_call_section_begin|>"
    public static let sectionEnd = "<|tool_calls_section_end|>"
    public static let callBegin = "<|tool_call_begin|>"
    public static let callEnd = "<|tool_call_end|>"
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        let hasMarker = modelOutput.contains(Self.sectionBegin) || modelOutput.contains(Self.sectionBeginAlt) || modelOutput.contains(Self.callBegin)
        guard hasMarker else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        var content = modelOutput
        for marker in [Self.sectionBegin, Self.sectionBeginAlt] {
            if let r = content.range(of: marker) {
                content = String(modelOutput[..<r.lowerBound])
                break
            }
        }
        let cleaned = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let pattern = #"(?s)<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>"#
        var calls: [ParsedToolCall] = []
        for m in ParserUtils.regexMatches(modelOutput, pattern: pattern) where m.count >= 3 {
            let name = m[1].trimmingCharacters(in: .whitespacesAndNewlines)
            let args = m[2].trimmingCharacters(in: .whitespacesAndNewlines)
            calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: args))
        }
        if calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
    }
}

// MARK: - Granite

public final class GraniteToolCallParser: ToolCallParser {
    public static let botToken = "<|tool_call|>"
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var stripped = modelOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasMarker = stripped.hasPrefix(Self.botToken) || stripped.hasPrefix("[{")
        guard hasMarker else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        if stripped.hasPrefix(Self.botToken) {
            stripped = String(stripped.dropFirst(Self.botToken.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let data = stripped.data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        var calls: [ParsedToolCall] = []
        for item in arr {
            if let name = item["name"] as? String {
                let args = item["arguments"] ?? [:]
                calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(args)))
            }
        }
        return ExtractedToolCallInformation(toolsCalled: !calls.isEmpty, toolCalls: calls, content: nil)
    }
}

// MARK: - Nemotron / Step3.5 (<tool_call><function=name>…</function></tool_call>)

public final class NemotronToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        let pattern = #"(?s)<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"#
        let matches = ParserUtils.regexMatches(modelOutput, pattern: pattern)
        if matches.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        var calls: [ParsedToolCall] = []
        for m in matches where m.count >= 3 {
            let name = m[1]
            let body = m[2]
            // Body may be JSON or a series of <parameter=k>v</parameter>
            if let obj = ParserUtils.parseJSONObject(body.trimmingCharacters(in: .whitespacesAndNewlines)) {
                calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(obj)))
            } else {
                let params = ParserUtils.regexMatches(body, pattern: #"(?s)<parameter=([^>]+)>(.*?)</parameter>"#)
                var dict: [String: Any] = [:]
                for p in params where p.count >= 3 {
                    dict[p[1]] = p[2].trimmingCharacters(in: .whitespacesAndNewlines)
                }
                calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(dict)))
            }
        }
        let cleaned = ParserUtils.regexReplace(modelOutput, pattern: pattern, with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ExtractedToolCallInformation(toolsCalled: !calls.isEmpty, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
    }
}

public final class Step3p5ToolCallParser: ToolCallParser {
    private let inner = NemotronToolCallParser()
    public init() {}
    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        inner.extractToolCalls(modelOutput, request: request)
    }
}

// MARK: - xLAM (```json fence / [TOOL_CALLS] / </think>)

public final class XlamToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var text = modelOutput
        // After think
        if let r = text.range(of: "</think>") {
            text = String(text[r.upperBound...])
        }
        // [TOOL_CALLS]
        if let r = text.range(of: "[TOOL_CALLS]") {
            text = String(text[r.upperBound...])
        }
        // ```json fence
        let fenced = ParserUtils.regexMatches(text, pattern: #"(?s)```(?:json)?\s*([\s\S]*?)```"#)
        if let m = fenced.first, m.count >= 2 {
            text = m[1]
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else {
            return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
        }
        if let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            var calls: [ParsedToolCall] = []
            for item in arr {
                if let name = item["name"] as? String {
                    calls.append(ParsedToolCall(
                        id: generateToolId(),
                        name: name,
                        arguments: ParserUtils.normaliseArgsJSON(item["arguments"] ?? item["parameters"] ?? [:])
                    ))
                }
            }
            if !calls.isEmpty {
                return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: nil)
            }
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
    }
}

// MARK: - Functionary (<|recipient|>name\n<|content|>{...})

public final class FunctionaryToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        let pattern = #"(?s)<\|recipient\|>\s*(\w+)\s*\n?<\|content\|>\s*(\{.*?\})"#
        let matches = ParserUtils.regexMatches(modelOutput, pattern: pattern)
        if !matches.isEmpty {
            var calls: [ParsedToolCall] = []
            for m in matches where m.count >= 3 {
                calls.append(ParsedToolCall(
                    id: generateToolId(),
                    name: m[1],
                    arguments: m[2]
                ))
            }
            let cleaned = ParserUtils.regexReplace(modelOutput, pattern: pattern, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ExtractedToolCallInformation(toolsCalled: !calls.isEmpty, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        // JSON array fallback
        let trimmed = modelOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("[") && trimmed.hasSuffix("]"),
           let data = trimmed.data(using: .utf8),
           let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            var calls: [ParsedToolCall] = []
            for item in arr {
                if let name = item["name"] as? String {
                    calls.append(ParsedToolCall(
                        id: generateToolId(),
                        name: name,
                        arguments: ParserUtils.normaliseArgsJSON(item["arguments"] ?? [:])
                    ))
                }
            }
            if !calls.isEmpty {
                return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: nil)
            }
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: modelOutput)
    }
}

// MARK: - GLM-4.7

public final class Glm47ToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var cleaned = ParserUtils.stripThinkTags(modelOutput)
        var calls: [ParsedToolCall] = []

        // <tool_call>{json or XML}</tool_call>
        let blocks = ParserUtils.regexMatches(cleaned, pattern: #"(?s)<tool_call>(.*?)</tool_call>"#)
        for m in blocks where m.count >= 2 {
            let body = m[1].trimmingCharacters(in: .whitespacesAndNewlines)
            // Try JSON
            if let obj = ParserUtils.parseJSONObject(body),
               let name = obj["name"] as? String {
                calls.append(ParsedToolCall(
                    id: generateToolId(),
                    name: name,
                    arguments: ParserUtils.normaliseArgsJSON(obj["arguments"] ?? [:])
                ))
                continue
            }
            // Try XML-ish: function_name\n<arg_key>val</arg_key>
            let firstNewline = body.firstIndex(of: "\n") ?? body.endIndex
            let name = String(body[..<firstNewline]).trimmingCharacters(in: .whitespacesAndNewlines)
            let rest = firstNewline == body.endIndex ? "" : String(body[body.index(after: firstNewline)...])
            let args = ParserUtils.regexMatches(rest, pattern: #"(?s)<(\w+)>(.*?)</\1>"#)
            var dict: [String: Any] = [:]
            for a in args where a.count >= 3 {
                dict[a[1]] = a[2].trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if !name.isEmpty {
                calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(dict)))
            }
        }
        if !blocks.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: #"(?s)<tool_call>(.*?)</tool_call>"#, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: cleaned)
    }
}

// MARK: - MiniMax (<minimax:tool_call><invoke name=...><parameter name=...>)

public final class MiniMaxToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var cleaned = ParserUtils.stripThinkTags(modelOutput)
        var calls: [ParsedToolCall] = []

        let blocks = ParserUtils.regexMatches(cleaned, pattern: #"(?s)<minimax:tool_call>(.*?)</minimax:tool_call>"#)
        for m in blocks where m.count >= 2 {
            let block = m[1]
            let invokes = ParserUtils.regexMatches(block, pattern: #"(?s)<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>"#)
            for inv in invokes where inv.count >= 3 {
                let name = inv[1]
                let body = inv[2]
                let params = ParserUtils.regexMatches(body, pattern: #"(?s)<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>"#)
                var dict: [String: Any] = [:]
                for p in params where p.count >= 3 {
                    dict[p[1]] = p[2].trimmingCharacters(in: .whitespacesAndNewlines)
                }
                calls.append(ParsedToolCall(id: generateToolId(), name: name, arguments: ParserUtils.normaliseArgsJSON(dict)))
            }
        }
        if !blocks.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: #"(?s)<minimax:tool_call>(.*?)</minimax:tool_call>"#, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: cleaned)
    }
}

// MARK: - Gemma 4 (<|tool_call>{json}<|/tool_call> and hermes fallback)

public final class Gemma4ToolCallParser: ToolCallParser {
    public init() {}

    public func extractToolCalls(_ modelOutput: String, request: ChatRequest?) -> ExtractedToolCallInformation {
        var cleaned = ParserUtils.stripThinkTags(modelOutput)
        var calls: [ParsedToolCall] = []

        // Native: <|tool_call>{json}<|/tool_call>
        let nativePattern = #"(?s)<\|tool_call\|?>\s*(\{.*?\})\s*<\|?/tool_call\|?>"#
        for m in ParserUtils.regexMatches(cleaned, pattern: nativePattern) where m.count >= 2 {
            if let obj = ParserUtils.parseJSONObject(m[1]),
               let name = obj["name"] as? String {
                calls.append(ParsedToolCall(
                    id: generateToolId(),
                    name: name,
                    arguments: ParserUtils.normaliseArgsJSON(obj["arguments"] ?? [:])
                ))
            }
        }
        if !calls.isEmpty {
            cleaned = ParserUtils.regexReplace(cleaned, pattern: nativePattern, with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Fallback: Hermes tags
        if calls.isEmpty {
            let hermes = HermesToolCallParser().extractToolCalls(cleaned, request: request)
            if hermes.toolsCalled {
                return hermes
            }
        }

        if !calls.isEmpty {
            return ExtractedToolCallInformation(toolsCalled: true, toolCalls: calls, content: cleaned.isEmpty ? nil : cleaned)
        }
        return ExtractedToolCallInformation(toolsCalled: false, toolCalls: [], content: cleaned)
    }
}
