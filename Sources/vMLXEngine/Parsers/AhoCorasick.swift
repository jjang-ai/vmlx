// SPDX-License-Identifier: Apache-2.0
//
// Aho-Corasick multi-pattern string matcher for stop-sequence handling.
//
// The naive approach (scan every visible-content delta with `String.contains`
// for each pattern) is O(n × Σ|p|) per chunk. With many patterns or long
// patterns it becomes the dominant cost on the hot streaming path. Aho-
// Corasick builds a deterministic finite automaton from the pattern set so
// every additional input character is O(1) regardless of how many patterns
// you've registered. Match detection is online: feed bytes one at a time
// and ask "are we at the end of any pattern right now."
//
// Used by `Engine.streamReal` to detect when a stop sequence has been
// emitted in the running visible-content stream so generation can be
// halted cleanly at the boundary (truncating the delta back to *just
// before* the stop string).

import Foundation

/// Multi-pattern matcher. Build once, feed many.
///
/// The implementation operates on the UTF-8 byte stream of the input,
/// not Swift `Character` units, because chat-template stop strings are
/// always plain ASCII (`\n`, `<|im_end|>`, `</tool_call>`, etc.). Going
/// through `Character` adds zero benefit and a substantial perf cost
/// (grapheme-cluster boundary detection per char).
public final class AhoCorasickMatcher {

    /// Empty matcher — `feed(_:)` always returns `nil`. Used when the
    /// caller passes no stop sequences so the streaming hot path doesn't
    /// pay any per-byte cost.
    public static let empty = AhoCorasickMatcher(patterns: [])

    private struct Node {
        var next: [UInt8: Int] = [:]   // direct (goto) transitions
        var fail: Int = 0              // failure link (suffix back-edge)
        var output: [Int] = []         // pattern indices matched at this node
        var depth: Int = 0             // distance from root (= match length)
    }

    /// Original patterns in registration order. Match index returned by
    /// `feed(_:)` indexes into this array.
    public let patterns: [String]
    private var nodes: [Node]

    /// True when the pattern set is empty. Lets callers skip a function
    /// call entirely on the hot path.
    public var isEmpty: Bool { patterns.isEmpty }

    /// Maximum pattern length in bytes. Used by callers to size the
    /// streaming-tail buffer they hold back for cross-chunk match safety.
    public let maxPatternByteLength: Int

    public init(patterns: [String]) {
        self.patterns = patterns
        var nodes: [Node] = [Node()]   // root at index 0
        var maxLen = 0
        for (idx, p) in patterns.enumerated() where !p.isEmpty {
            let bytes = Array(p.utf8)
            maxLen = max(maxLen, bytes.count)
            var cur = 0
            for b in bytes {
                if let nxt = nodes[cur].next[b] {
                    cur = nxt
                } else {
                    nodes.append(Node())
                    let newIdx = nodes.count - 1
                    nodes[cur].next[b] = newIdx
                    nodes[newIdx].depth = nodes[cur].depth + 1
                    cur = newIdx
                }
            }
            nodes[cur].output.append(idx)
        }
        // BFS to populate failure links.
        var queue: [Int] = []
        for child in nodes[0].next.values {
            nodes[child].fail = 0
            queue.append(child)
        }
        while !queue.isEmpty {
            let u = queue.removeFirst()
            for (b, v) in nodes[u].next {
                queue.append(v)
                var f = nodes[u].fail
                while f != 0 && nodes[f].next[b] == nil { f = nodes[f].fail }
                nodes[v].fail = nodes[f].next[b] ?? 0
                if nodes[v].fail == v { nodes[v].fail = 0 }
                nodes[v].output.append(contentsOf: nodes[nodes[v].fail].output)
            }
        }
        self.nodes = nodes
        self.maxPatternByteLength = maxLen
    }

    /// One match event surfaced by `feedFromZero(_:)`. `patternIndex`
    /// indexes into `patterns`. `endByteOffset` is the byte position of
    /// the LAST byte of the match in the input scanned by this call
    /// (zero-indexed).
    public struct Match: Equatable, Sendable {
        public let patternIndex: Int
        public let endByteOffset: Int
        public init(patternIndex: Int, endByteOffset: Int) {
            self.patternIndex = patternIndex
            self.endByteOffset = endByteOffset
        }
    }

    /// Scan `text` from a fresh state and return the FIRST match found
    /// (left-most, shortest-match-wins among ties at the same end-offset).
    /// Returns nil if no pattern is contained.
    public func firstMatch(in text: String) -> Match? {
        if isEmpty { return nil }
        var state = 0
        var firstHit: Match? = nil
        for (offset, b) in text.utf8.enumerated() {
            while state != 0 && nodes[state].next[b] == nil {
                state = nodes[state].fail
            }
            state = nodes[state].next[b] ?? 0
            if !nodes[state].output.isEmpty {
                let idx = nodes[state].output.min()!
                let candidate = Match(patternIndex: idx, endByteOffset: offset)
                if firstHit == nil || candidate.endByteOffset < firstHit!.endByteOffset {
                    firstHit = candidate
                }
                // Earliest end wins; we can't shorten further by scanning
                // more so break out immediately.
                break
            }
        }
        return firstHit
    }
}
