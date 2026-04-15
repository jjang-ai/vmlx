// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Mistral V13 tool call format: `[TOOL_CALLS]name[ARGS]{"json_args"}`
///
/// This format is used by Mistral3/Ministral-3 2512 models and Devstral 2.
/// The special tokens `[TOOL_CALLS]` (token ID 9) and `[ARGS]` (ID 32) are used
/// as delimiters. Multiple tool calls use repeated `[TOOL_CALLS]` tokens.
///
/// Also handles the older V11 format which includes an optional `[CALL_ID]`
/// between the function name and `[ARGS]` (V13 does not use `[CALL_ID]`).
///
/// Examples:
/// - `[TOOL_CALLS]get_weather[ARGS]{"location": "Tokyo"}`
/// - `[TOOL_CALLS]fn1[ARGS]{...}[TOOL_CALLS]fn2[ARGS]{...}` (multiple calls)
///
/// Mistral does not use an end tag — tool calls end at EOS. Since stop tokens
/// are intercepted at the token ID level before detokenization, tool calls are
/// extracted via `ToolCallProcessor.processEOS()` at generation end.
public struct MistralToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "[TOOL_CALLS]"
    public let endTag: String? = nil

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // Strip wrapper tags only when they appear at boundaries.
        // This keeps literal tag strings inside argument values intact.
        if let start = startTag, text.hasPrefix(start) {
            text = String(text.dropFirst(start.count))
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // Split on [ARGS] to get function name and arguments
        guard let argsRange = text.range(of: "[ARGS]") else {
            return nil
        }

        var namePart = String(text[..<argsRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let argsPart = String(text[argsRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Handle optional [CALL_ID] between name and [ARGS]
        if let callIdRange = namePart.range(of: "[CALL_ID]") {
            namePart = String(namePart[..<callIdRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        guard !namePart.isEmpty else { return nil }

        // Parse arguments as JSON using tryParseJSON from ParserUtilities
        guard let argsDict = tryParseJSON(argsPart) as? [String: any Sendable] else {
            return nil
        }

        return ToolCall(
            function: ToolCall.Function(name: namePart, arguments: argsDict)
        )
    }
}
