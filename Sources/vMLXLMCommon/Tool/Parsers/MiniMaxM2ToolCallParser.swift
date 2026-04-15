// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for MiniMax M2 format: <invoke name="f"><parameter name="k">v</parameter></invoke>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
public struct MiniMaxM2ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<minimax:tool_call>"
    public let endTag: String? = "</minimax:tool_call>"

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
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Find <invoke name=...>...</invoke>
        guard let invokeStart = text.range(of: "<invoke name=") else { return nil }
        guard let invokeEnd = text.range(of: "</invoke>") else { return nil }

        let invokeContent = String(text[invokeStart.upperBound ..< invokeEnd.lowerBound])

        // Extract function name (between name= and first >)
        guard let nameEnd = invokeContent.firstIndex(of: ">") else { return nil }
        let funcName = extractName(String(invokeContent[..<nameEnd]))

        guard !funcName.isEmpty else { return nil }

        // Get parameter config from tools schema
        let paramConfig = getParameterConfig(funcName: funcName, tools: tools)

        var arguments: [String: any Sendable] = [:]
        let paramSection = String(invokeContent[invokeContent.index(after: nameEnd)...])

        // Find all <parameter name=...>...</parameter> tags
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<parameter name=", range: searchRange) {
            // Find the parameter name
            guard
                let nameEnd = paramSection.range(
                    of: ">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { break }

            let paramName = extractName(
                String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound]))

            // Find </parameter>
            guard
                let paramEnd = paramSection.range(
                    of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex)
            else { break }

            var paramValue = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])

            // Trim leading/trailing whitespace and newlines (matching Python behavior)
            paramValue = paramValue.trimmingCharacters(in: .whitespaces)
            if paramValue.hasPrefix("\n") {
                paramValue = String(paramValue.dropFirst())
            }
            if paramValue.hasSuffix("\n") {
                paramValue = String(paramValue.dropLast())
            }

            // Get types from schema for this parameter
            let paramSchema = paramConfig[paramName] as? [String: any Sendable]
            let paramTypes = extractTypesFromSchema(paramSchema)
            arguments[paramName] = convertValueWithTypes(paramValue, types: paramTypes)

            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
