// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for GLM4 format: func<arg_key>k</arg_key><arg_value>v</arg_value>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public struct GLM4ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Extract function name (everything before first <arg_key>)
        guard let argKeyStart = text.range(of: "<arg_key>") else { return nil }
        let funcName = String(text[..<argKeyStart.lowerBound]).trimmingCharacters(
            in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        var arguments: [String: any Sendable] = [:]

        // Find all arg_key/arg_value pairs
        var searchRange = text.startIndex ..< text.endIndex
        while let keyStart = text.range(of: "<arg_key>", range: searchRange) {
            // Find </arg_key>
            guard
                let keyEnd = text.range(
                    of: "</arg_key>", range: keyStart.upperBound ..< text.endIndex)
            else { break }

            let key = String(text[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Find <arg_value> after </arg_key>
            guard
                let valueStart = text.range(
                    of: "<arg_value>", range: keyEnd.upperBound ..< text.endIndex)
            else { break }

            // Find </arg_value>
            guard
                let valueEnd = text.range(
                    of: "</arg_value>", range: valueStart.upperBound ..< text.endIndex)
            else { break }

            let value = String(text[valueStart.upperBound ..< valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // GLM4: deserialize if NOT a string type in schema
            if !isStringType(funcName: funcName, argName: key, tools: tools) {
                arguments[key] = tryParseJSON(value) ?? value
            } else {
                arguments[key] = value
            }

            searchRange = valueEnd.upperBound ..< text.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
