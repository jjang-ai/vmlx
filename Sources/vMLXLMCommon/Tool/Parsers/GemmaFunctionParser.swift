// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    private let escapeMarker: String

    /// Initialize with Gemma 3 tags (default)
    public init() {
        self.startTag = "<start_function_call>"
        self.endTag = "<end_function_call>"
        self.escapeMarker = "<escape>"
    }

    /// Initialize with custom tags (for Gemma 4 which uses `<|tool_call>` / `<tool_call|>`)
    public init(startTag: String, endTag: String, escapeMarker: String = "<|\"\\|>") {
        self.startTag = startTag
        self.endTag = endTag
        self.escapeMarker = escapeMarker
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name (word characters until {)
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let funcName = String(remaining[..<braceStart])

        guard !funcName.isEmpty else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        var argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])

        var arguments: [String: any Sendable] = [:]

        // Parse key:value pairs
        while !argsStr.isEmpty {
            // Find the key (everything before :)
            guard let colonIdx = argsStr.firstIndex(of: ":") else { break }
            let key = String(argsStr[..<colonIdx])
            argsStr = String(argsStr[argsStr.index(after: colonIdx)...])

            // Handle escaped strings
            if argsStr.hasPrefix(escapeMarker) {
                argsStr = String(argsStr.dropFirst(escapeMarker.count))
                guard let endEscape = argsStr.range(of: escapeMarker) else { break }
                let value = String(argsStr[..<endEscape.lowerBound])
                arguments[key] = value
                argsStr = String(argsStr[endEscape.upperBound...])
                // Skip comma if present
                if argsStr.hasPrefix(",") {
                    argsStr = String(argsStr.dropFirst())
                }
                continue
            }

            // Handle regular values (until comma or end)
            let commaIdx = argsStr.firstIndex(of: ",") ?? argsStr.endIndex
            let value = String(argsStr[..<commaIdx])
            argsStr =
                commaIdx < argsStr.endIndex
                ? String(argsStr[argsStr.index(after: commaIdx)...]) : ""

            // Try JSON decode, fallback to string
            if let data = value.data(using: .utf8),
                let json = deserializeJSON(data)
            {
                arguments[key] = json
            } else {
                arguments[key] = value
            }
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
