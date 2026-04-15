// SPDX-License-Identifier: Apache-2.0
// Port of the last-chance tool-call extraction logic from
// vmlx_engine/server.py lines ~6234-6306.
//
// When reasoning is suppressed (enable_thinking=False) and the model produces
// ONLY reasoning (no visible content), this helper scans the suppressed
// reasoning buffer for any of the known tool-call markers and, if found,
// runs the configured parser against the raw reasoning text. Used to rescue
// tool calls emitted inside <think> blocks by models like MiniMax M2.7.

import Foundation

public enum ToolCallExtractor {
    public struct Rescued: Sendable {
        public var toolCalls: [ParsedToolCall]
        public var content: String?
    }

    /// Check the accumulated reasoning buffer for tool-call markers and, if
    /// present, run the configured parser to extract any calls.
    ///
    /// - Parameters:
    ///   - suppressedReasoning: buffer accumulated while reasoning was suppressed.
    ///   - parser: the tool-call parser to use (e.g., registered for this model).
    ///   - request: the originating chat request (for tool-name validation).
    /// - Returns: `nil` if no markers were found or the parser produced no
    ///            calls. Otherwise, the rescued tool calls plus any leftover
    ///            content after stripping.
    public static func rescueFromSuppressedReasoning(
        suppressedReasoning: String,
        parser: ToolCallParser,
        request: ChatRequest? = nil
    ) -> Rescued? {
        guard !suppressedReasoning.isEmpty else { return nil }
        let hasMarker = toolCallMarkers.contains { suppressedReasoning.contains($0) }
        guard hasMarker else { return nil }

        let result = parser.extractToolCalls(
            suppressedReasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            request: request
        )
        guard result.toolsCalled && !result.toolCalls.isEmpty else { return nil }
        return Rescued(toolCalls: result.toolCalls, content: result.content)
    }
}
