// SPDX-License-Identifier: Apache-2.0
// Streaming accumulator for model-output SSE chunks.
//
// The Python engine's tool-call parsers are text-based and work on the full
// buffered output, but the server still needs to decide _when_ to route a
// streamed delta to reasoning, content, or tool-call. This helper keeps
// `previous` / `current` buffers in sync and exposes a small convenience
// wrapper that feeds a `ReasoningParser` and detects any of the
// `_TOOL_CALL_MARKERS` that indicate we should start buffering rather than
// streaming.

import Foundation

/// Tool-call markers from vmlx_engine/server.py `_TOOL_CALL_MARKERS`.
/// Presence of any of these in an accumulated buffer means we should stop
/// emitting incremental content to the client and instead wait for the
/// complete buffer so a `ToolCallParser` can parse the whole call atomically.
public let toolCallMarkers: [String] = [
    "<tool_call>",
    "<|tool_call>",            // Gemma 4 native tool call
    "<|tool_call|>",
    "[TOOL_CALLS]",
    "<function=",
    "<minimax:tool_call>",
    "[Calling tool:",
    "<|recipient|>",
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>",  // DeepSeek unicode
    "<|python_tag|>",          // Llama 3.1+
]

/// Incremental accumulator. Safe to reuse across deltas in a single stream.
public final class StreamingAccumulator {
    public private(set) var previous: String = ""
    public private(set) var current: String = ""
    private let reasoningParser: ReasoningParser?

    public init(reasoningParser: ReasoningParser? = nil, thinkInPrompt: Bool = false, harmonyActive: Bool = false) {
        self.reasoningParser = reasoningParser
        reasoningParser?.resetState(thinkInPrompt: thinkInPrompt, harmonyActive: harmonyActive)
    }

    /// Consume one delta. Returns:
    ///   - a `ReasoningDelta` if the current parser produced incremental output,
    ///   - `nil` if the accumulator is buffering (e.g., we saw a tool-call
    ///     marker and are holding bytes until the call completes).
    public func feed(_ delta: String) -> ReasoningDelta? {
        previous = current
        current += delta
        if hasToolCallMarker(current) {
            return nil
        }
        guard let parser = reasoningParser else {
            return ReasoningDelta(content: delta)
        }
        return parser.extractReasoningStreaming(previous: previous, current: current, delta: delta)
    }

    /// Does the current buffer contain any tool-call marker?
    public var buffered: Bool { hasToolCallMarker(current) }

    public func hasToolCallMarker(_ text: String) -> Bool {
        for m in toolCallMarkers where text.contains(m) { return true }
        return false
    }

    public func reset() {
        previous = ""
        current = ""
    }
}
