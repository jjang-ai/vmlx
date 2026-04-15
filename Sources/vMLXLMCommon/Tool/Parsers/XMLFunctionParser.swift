// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Pattern: <function=(content)</function> — [\s\S] matches newlines
        guard
            let funcMatch = content.range(
                of: #"<function=([\s\S]*?)</function>"#, options: .regularExpression)
        else { return nil }

        let funcContent = String(content[funcMatch])

        // Extract function name (everything between <function= and first >)
        guard let nameStart = funcContent.range(of: "<function="),
            let nameEnd = funcContent.range(
                of: ">", range: nameStart.upperBound ..< funcContent.endIndex)
        else { return nil }

        let funcName = String(funcContent[nameStart.upperBound ..< nameEnd.lowerBound])
        let paramSection = String(funcContent[nameEnd.upperBound...])

        var arguments: [String: any Sendable] = [:]

        // Find all parameter tags
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<parameter=", range: searchRange) {
            // Find the parameter name (between = and >)
            guard
                let nameEnd = paramSection.range(
                    of: ">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { break }

            let paramName = String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound])

            // Find the closing </parameter> tag
            guard
                let paramEnd = paramSection.range(
                    of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex)
            else { break }

            var paramValue = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])

            // Trim leading/trailing newlines (matching Python behavior)
            if paramValue.hasPrefix("\n") {
                paramValue = String(paramValue.dropFirst())
            }
            if paramValue.hasSuffix("\n") {
                paramValue = String(paramValue.dropLast())
            }

            // Convert value based on schema type
            arguments[paramName] = convertParameterValue(
                paramValue, paramName: paramName, funcName: funcName, tools: tools)

            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
