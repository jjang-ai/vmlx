// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Pythonic tool call format: [function_name(arg1='value1', arg2='value2')]
/// Used by LFM2.5 and similar models that output tool calls in Python function call syntax.
/// Reference: LiquidAI LFM2.5 chat template format
public struct PythonicToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String? = nil, endTag: String? = nil) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip tags if present
        if let start = startTag, let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let end = endTag, let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let funcName: String
        let argsString: String

        // Required brackets pattern (matches Python reference: r"\[(\w+)\((.*?)\)\]")
        // The required \] forces .*? to backtrack past nested ) inside argument values.
        let bracketPattern = #"\[(\w+)\((.*?)\)\]"#
        if let regex = try? NSRegularExpression(
            pattern: bracketPattern, options: [.dotMatchesLineSeparators]),
            let match = regex.firstMatch(
                in: text, options: [], range: NSRange(text.startIndex..., in: text)),
            let nameRange = Range(match.range(at: 1), in: text),
            let argsRange = Range(match.range(at: 2), in: text)
        {
            funcName = String(text[nameRange])
            argsString = String(text[argsRange])
        } else {
            // Fallback for without-brackets case: use string indices to find the
            // outermost parentheses, avoiding the greedy/non-greedy regex pitfall.
            guard let openParen = text.firstIndex(of: "("),
                let closeParen = text.lastIndex(of: ")")
            else { return nil }

            let name = text[text.startIndex ..< openParen]
            guard !name.isEmpty, name.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" })
            else { return nil }

            funcName = String(name)
            argsString = String(text[text.index(after: openParen) ..< closeParen])
        }

        let arguments = parseArguments(argsString, funcName: funcName, tools: tools)
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Parse Pythonic keyword arguments: arg1='value1', arg2="value2", arg3=123
    private func parseArguments(
        _ argsString: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]

        // Pattern for key=value pairs, handling quoted strings with possible commas inside
        // This handles: key='value', key="value", key=123, key=True, key=None
        let argPattern = #"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,\)]+)"#

        guard let regex = try? NSRegularExpression(pattern: argPattern, options: []) else {
            return arguments
        }

        let matches = regex.matches(
            in: argsString, options: [], range: NSRange(argsString.startIndex..., in: argsString))

        for match in matches {
            guard let keyRange = Range(match.range(at: 1), in: argsString),
                let valueRange = Range(match.range(at: 2), in: argsString)
            else { continue }

            let key = String(argsString[keyRange])
            var value = String(argsString[valueRange]).trimmingCharacters(in: .whitespaces)

            // Remove surrounding quotes if present
            if (value.hasPrefix("'") && value.hasSuffix("'"))
                || (value.hasPrefix("\"") && value.hasSuffix("\""))
            {
                value = String(value.dropFirst().dropLast())
                // Unescape escaped quotes
                value = value.replacingOccurrences(of: "\\'", with: "'")
                value = value.replacingOccurrences(of: "\\\"", with: "\"")
                value = value.replacingOccurrences(of: "\\\\", with: "\\")
            }

            // Convert value based on schema type if available
            arguments[key] = convertParameterValue(
                value, paramName: key, funcName: funcName, tools: tools)
        }

        return arguments
    }
}
