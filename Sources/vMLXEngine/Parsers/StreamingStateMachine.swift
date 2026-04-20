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
        if hasToolCallMarkerAfterAppend(delta: delta) {
            _bufferedCache = true
            return nil
        }
        _bufferedCache = false
        guard let parser = reasoningParser else {
            return ReasoningDelta(content: delta)
        }
        return parser.extractReasoningStreaming(previous: previous, current: current, delta: delta)
    }

    /// Iter-31: per-token O(N) → O(1) amortized. Pre-fix this scanned
    /// the full `current` buffer for 8 markers on every token. For a
    /// 200-token response the last feed scanned ~8000 char positions;
    /// quadratic in response length. Optimization: `previous` was
    /// already marker-free (else we'd be buffered and `buffered`
    /// below would short-circuit). So a newly-introduced marker must
    /// straddle the `previous`/`delta` boundary or live entirely
    /// inside `delta`. We only need to scan
    /// `previous.suffix(maxMarkerLen - 1) + delta`.
    private let maxMarkerLen: Int = toolCallMarkers.map { $0.count }.max() ?? 0

    private func hasToolCallMarkerAfterAppend(delta: String) -> Bool {
        // Short-circuit cheap case: delta is small, marker is small.
        // Build just the needed suffix + delta.
        let tailLen = max(0, maxMarkerLen - 1)
        let prevTail = previous.suffix(tailLen)
        let scan = String(prevTail) + delta
        for m in toolCallMarkers where scan.contains(m) { return true }
        return false
    }

    /// Legacy full-buffer scan — kept for `buffered` initial state
    /// probes and tests that seed a non-empty `current` without
    /// calling `feed()` to build it up. Hot path is
    /// `hasToolCallMarkerAfterAppend`.
    public func hasToolCallMarker(_ text: String) -> Bool {
        for m in toolCallMarkers where text.contains(m) { return true }
        return false
    }

    /// Does the current buffer contain any tool-call marker? Cached
    /// from the last `feed()` call to avoid re-scanning. Initial
    /// state (before any feed) returns false via default.
    public var buffered: Bool { _bufferedCache }
    private var _bufferedCache: Bool = false

    public func reset() {
        previous = ""
        current = ""
    }
}
