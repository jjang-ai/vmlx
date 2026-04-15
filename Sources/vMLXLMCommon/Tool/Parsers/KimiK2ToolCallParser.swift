// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Kimi K2 format: functions.name:0<|tool_call_argument_begin|>{JSON}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/kimi_k2.py
public struct KimiK2ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_calls_section_begin|>"
    public let endTag: String? = "<|tool_calls_section_end|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip outer tags if present
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        // Strip inner tags
        text = text.replacingOccurrences(of: "<|tool_call_begin|>", with: "")
        text = text.replacingOccurrences(of: "<|tool_call_end|>", with: "")
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Pattern: (.+):\d+\s*<|tool_call_argument_begin|>
        // Find the function name (before :N<|tool_call_argument_begin|>)
        guard let argBeginRange = text.range(of: "<|tool_call_argument_begin|>") else { return nil }

        let beforeArgBegin = String(text[..<argBeginRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Find the last colon followed by a number
        guard let lastColonIdx = beforeArgBegin.lastIndex(of: ":") else { return nil }

        var funcName = String(beforeArgBegin[..<lastColonIdx])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Strip "functions." prefix if present
        if funcName.hasPrefix("functions.") {
            funcName = String(funcName.dropFirst("functions.".count))
        } else if let dotIdx = funcName.firstIndex(of: ".") {
            // Also handle other prefixes like "tools."
            funcName = String(funcName[funcName.index(after: dotIdx)...])
        }

        guard !funcName.isEmpty else { return nil }

        // Extract arguments JSON (everything after <|tool_call_argument_begin|>)
        let argsStr = String(text[argBeginRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Deserialize the JSON arguments
        guard let arguments = tryParseJSON(argsStr) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
