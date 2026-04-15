//
//  DDTreeBuilder.swift
//  vMLXLMCommon / DFlash
//
//  Dynamic-Draft-Tree primitives for JANG-DFlash speculative decoding.
//
//  Algorithm references:
//    - EAGLE-2 "Dynamic Draft Trees" — arXiv 2406.16858, §4.1–4.2
//    - DFlash block diffusion drafter — arXiv 2602.06036
//
//  A JANG-DFlash decode cycle looks like:
//      1. drafter runs ONE forward pass, emits B parallel softmaxes over V
//      2. DDTreeBuilder.beamTopMLattice extracts the top-m joint-prob paths
//      3. DDTreeBuilder.flatten deduplicates shared prefixes into a prefix
//         trie and returns (flatTokens, ancestryMask) suitable for feeding
//         to the target model with a tree-attention mask
//      4. target verifies all nodes in one forward pass; caller walks the
//         trie top-down applying rejection sampling to pick the accepted
//         prefix
//
//  Because DFlash's drafter produces all B positions in a single
//  non-autoregressive pass, the per-slot distributions are conditionally
//  independent given the injected target context (h_ctx). That makes the
//  joint probability of a path the plain product of its per-slot
//  marginals — no depth-wise re-expansion like EAGLE-2 requires for its
//  autoregressive drafter. The tree we build is therefore a *lattice* of
//  width-k columns at each of B-1 positions, pruned by joint log-prob
//  to m survivors, then rebuilt as a prefix trie for dedup.
//
//  This file is pure Swift (no MLX dependency); the tree mask is emitted
//  as `[[Bool]]` and converted to an MLXArray additive bias by the
//  caller (see JangDFlashSpecDec.swift).
//

import Foundation

public enum DDTreeBuilder {

    /// One candidate path through the per-slot lattice.
    public struct Path: Hashable, Sendable, CustomStringConvertible {
        public var tokens: [Int]
        /// Joint log-probability along the path (sum of per-slot log-probs).
        public var logProb: Float

        public init(tokens: [Int], logProb: Float) {
            self.tokens = tokens
            self.logProb = logProb
        }

        public var description: String {
            "Path(tokens=\(tokens), logProb=\(logProb))"
        }
    }

    /// Result of flattening a set of paths into a prefix trie.
    public struct FlatTree: Sendable {
        /// DFS-order token IDs, one per trie node (excluding the implicit root).
        public var flatTokens: [Int]
        /// `mask[i][j] == true` iff node `j` is an ancestor of node `i`,
        /// or `i == j`. Both `i` and `j` index into `flatTokens`.
        public var ancestryMask: [[Bool]]
        /// For each input path, the flat index of its leaf node. Used by the
        /// caller to resolve which path ended at which position in the final
        /// trie after duplicate prefixes collapse.
        public var leafIndices: [Int]

        public init(flatTokens: [Int], ancestryMask: [[Bool]], leafIndices: [Int]) {
            self.flatTokens = flatTokens
            self.ancestryMask = ancestryMask
            self.leafIndices = leafIndices
        }
    }

    // MARK: - Beam search over the per-slot lattice

    /// Runs a log-space beam search over the per-slot top-k candidates and
    /// returns up to `m` paths sorted by joint log-prob descending.
    ///
    /// - Parameters:
    ///   - vals: per-slot top-k *probabilities* (not log-probs). Shape
    ///           `[numSlots][k]`. Must be strictly positive; zero-floored
    ///           to `1e-20` before taking logs so `log(0)` never occurs.
    ///   - ids:  per-slot top-k token ids. Shape `[numSlots][k]`, same
    ///           layout as `vals`.
    ///   - m:    maximum number of surviving paths (width of the beam).
    ///
    /// Complexity is `O(numSlots · m · k · log(m · k))` — with numSlots ≤ 16,
    /// k = 4, m = 60 that's ~3800 comparisons per decode cycle, sub-millisecond
    /// in pure Swift.
    public static func beamTopMLattice(
        vals: [[Float]], ids: [[Int]], m: Int
    ) -> [Path] {
        precondition(vals.count == ids.count, "vals/ids slot count mismatch")
        precondition(m > 0, "m must be positive")

        var beams: [Path] = [Path(tokens: [], logProb: 0)]
        for slot in 0 ..< vals.count {
            precondition(
                vals[slot].count == ids[slot].count,
                "slot \(slot) vals/ids length mismatch")
            let slotIds = ids[slot]
            let slotLogs: [Float] = vals[slot].map { Float(log(Double(max($0, 1e-20)))) }

            var next: [Path] = []
            next.reserveCapacity(beams.count * slotIds.count)
            for b in beams {
                for i in 0 ..< slotIds.count {
                    next.append(Path(
                        tokens: b.tokens + [slotIds[i]],
                        logProb: b.logProb + slotLogs[i]
                    ))
                }
            }
            // Keep top-m by joint log-prob. Partial sort would be faster but
            // m and beam size are tiny in practice; full sort keeps the code
            // obvious.
            next.sort { $0.logProb > $1.logProb }
            if next.count > m { next.removeSubrange(m...) }
            beams = next
        }
        return beams
    }

    // MARK: - Prefix trie flattening

    /// Flattens a set of paths into a prefix trie, assigns each node a DFS
    /// flat index, and builds the ancestry mask used by tree-attention
    /// verification on the target model.
    ///
    /// Shared prefixes are deduplicated so the target model only evaluates
    /// each distinct ancestor once per decode cycle.
    public static func flatten(paths: [Path]) -> FlatTree {
        // Use a class-based trie so children can be mutated in place.
        final class Node {
            let token: Int
            var children: [Int: Node] = [:]
            var flatIdx: Int = -1
            init(token: Int) { self.token = token }
        }

        let root = Node(token: -1)
        // Build the trie.
        for p in paths {
            var node = root
            for t in p.tokens {
                if let existing = node.children[t] {
                    node = existing
                } else {
                    let fresh = Node(token: t)
                    node.children[t] = fresh
                    node = fresh
                }
            }
        }

        // DFS assigning flat indices. Track ancestor-index stack as we recurse
        // so we can materialise each row of the ancestry mask inline.
        var flatTokens: [Int] = []
        var ancestorRows: [[Int]] = []  // for node i, the list of j ≤ i reachable by ancestry (inclusive)

        func dfs(_ node: Node, parents: [Int]) {
            var selfParents = parents
            if node !== root {
                let myIdx = flatTokens.count
                node.flatIdx = myIdx
                flatTokens.append(node.token)
                selfParents.append(myIdx)
                ancestorRows.append(selfParents)
            }
            // Sort children for deterministic output — stabilises tests and
            // makes the flat indices reproducible across runs.
            for key in node.children.keys.sorted() {
                dfs(node.children[key]!, parents: selfParents)
            }
        }
        dfs(root, parents: [])

        // Materialise the NxN boolean ancestry mask from the collected rows.
        let n = flatTokens.count
        var mask = Array(repeating: Array(repeating: false, count: n), count: n)
        for i in 0 ..< n {
            for j in ancestorRows[i] {
                mask[i][j] = true
            }
        }

        // Resolve per-input-path leaf indices after dedup.
        var leaves: [Int] = []
        leaves.reserveCapacity(paths.count)
        for p in paths {
            var node = root
            for t in p.tokens {
                guard let next = node.children[t] else {
                    fatalError("DDTreeBuilder.flatten: path missing from trie — internal error")
                }
                node = next
            }
            leaves.append(node.flatIdx)
        }

        return FlatTree(flatTokens: flatTokens, ancestryMask: mask, leafIndices: leaves)
    }
}
