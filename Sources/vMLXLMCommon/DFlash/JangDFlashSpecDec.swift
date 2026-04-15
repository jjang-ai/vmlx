//
//  JangDFlashSpecDec.swift
//  vMLXLMCommon / DFlash
//
//  End-to-end JANG-DFlash + DDTree speculative decoding step. Ties
//  together:
//
//    1. Target forward with per-layer hidden taps (provided by the
//       target model via `callAsFunctionWithTaps`, architecture-specific)
//    2. JangDFlashDrafter — 1-step denoising block forward with
//       target-hidden KV injection
//    3. DDTreeBuilder — top-k-per-slot lattice beam → prefix trie →
//       ancestry mask
//    4. Target verify forward with the tree-attention mask (provided
//       by the target via the same `callAsFunctionWithTaps` signature)
//    5. Rejection-sampling walk down the trie to select the longest
//       accepted prefix
//
//  The target model is abstracted via the `JangDFlashTarget` protocol
//  below so this glue layer stays architecture-agnostic: any model
//  that can return (logits, per-layer taps) AND accept a tree-attention
//  mask override can plug in. MiniMax already conforms via its
//  `callAsFunctionWithTaps` additive API.
//
//  This file does NOT run full MLX compute in `swift test` because the
//  metallib isn't loaded there — the end-to-end smoke happens inside
//  `vmlxctl` where the library is correctly colocated with the binary.
//

import Foundation
import MLX
import MLXFast
import MLXRandom

// MARK: - Target protocol

/// A target model that supports the JANG-DFlash spec-dec hooks:
/// per-layer hidden-state taps during normal forwards, and a caller-
/// supplied attention mask for tree verification.
///
/// Concrete implementations live in the architecture-specific model
/// modules (see `MiniMaxDFlashTarget` in vMLXLLM).
public protocol JangDFlashTarget: AnyObject {
    /// Run a target forward pass, return logits at the final position
    /// AND the requested per-layer hidden states.
    ///
    /// - Parameters:
    ///   - inputs: `[B, L]` int token IDs
    ///   - cache:  per-layer KV cache. For the block-verify call it
    ///             must have room for the block's worth of new positions.
    ///   - tapLayers: indices whose post-layer hidden states to return
    ///   - providedMask: optional additive-bias mask override (for
    ///             tree-attention verification)
    func forwardWithTaps(
        inputs: MLXArray,
        cache: [KVCache]?,
        tapLayers: Set<Int>,
        providedMask: MLXFast.ScaledDotProductAttentionMaskMode?
    ) -> (logits: MLXArray, taps: [Int: MLXArray])

    /// Build a fresh per-layer KV cache matched to the target's
    /// architecture. Used by the cached-generate path so the spec-dec
    /// layer doesn't need a direct handle on the model.
    func makeCache() -> [KVCache]
}

// MARK: - Spec-dec config

public struct JangDFlashSpecConfig: Sendable, Equatable {
    /// Drafter block size B. Must match the drafter checkpoint's train-
    /// time block size.
    public var blockSize: Int = 16

    /// Per-slot top-k for DDTree expansion. Larger k gives the tree more
    /// branches to explore but grows verify cost linearly.
    public var topK: Int = 4

    /// Maximum number of paths kept after the lattice beam. Higher m
    /// means more tree nodes for the target to verify.
    public var numPaths: Int = 60

    /// Indices of target layers whose hidden states feed the drafter's
    /// KV injection. Defaults match MiniMax-M2.7 (5 evenly spaced
    /// across 62 layers).
    public var tapLayers: Set<Int> = [10, 22, 34, 46, 58]

    /// Hidden dimension of the target model. Used to sanity-check the
    /// tap concatenation before feeding to the drafter's fusion MLP.
    public var targetHiddenDim: Int = 3072

    public init() {}
}

// MARK: - Spec-dec step

/// Single block cycle outcome. Used by `generate(...)` to track
/// progress and surface timings back to the caller.
public struct JangDFlashBlockOutcome: Sendable {
    public var acceptedTokens: [Int]
    public var treeSize: Int
    public var targetWallSec: Double
    public var drafterWallSec: Double
    public var verifyWallSec: Double
}

public final class JangDFlashSpecDec {

    public let target: any JangDFlashTarget
    public let drafter: JangDFlashDrafter
    public let cfg: JangDFlashSpecConfig

    public init(
        target: any JangDFlashTarget,
        drafter: JangDFlashDrafter,
        cfg: JangDFlashSpecConfig = JangDFlashSpecConfig()
    ) {
        self.target = target
        self.drafter = drafter
        self.cfg = cfg
    }

    // MARK: - Multi-block generate loop (v1: cacheless)

    /// Run a full DFlash multi-block decode loop, returning the list
    /// of accepted token IDs (including the initial bonus but NOT the
    /// prompt). Stops when any of:
    /// - total generated length reaches `maxNewTokens`
    /// - an EOS token appears in the accepted prefix
    /// - the walker returns zero forward progress for a block
    ///
    /// **v1 limitation**: this loop is *cacheless* — every block
    /// re-runs the target on `[prompt + all accepted so far + flat trie]`
    /// without a persistent KV cache, so total work is quadratic in
    /// the generation length. Correct but slow. v2 will add proper
    /// cache handling (see notes in-file for the accepted-prefix
    /// replay strategy).
    ///
    /// - Parameters:
    ///   - promptIDs: pre-tokenized input
    ///   - maxNewTokens: hard cap on generated tokens
    ///   - eosTokenIDs: terminal token IDs; generation stops when any
    ///                  appears in the accepted prefix
    ///   - onBlock: optional callback invoked once per block with the
    ///              block outcome — lets the caller stream tokens as
    ///              they're accepted
    public func generate(
        promptIDs: [Int],
        maxNewTokens: Int,
        eosTokenIDs: Set<Int> = [],
        onBlock: ((JangDFlashBlockOutcome) -> Void)? = nil
    ) throws -> [Int] {
        precondition(!promptIDs.isEmpty, "promptIDs must not be empty")
        precondition(maxNewTokens > 0, "maxNewTokens must be positive")

        var accepted: [Int] = []
        var noProgressBlocks = 0
        let maxNoProgressBlocks = 3  // bail out of pathological loops

        while accepted.count < maxNewTokens {
            // Build the current prefix: prompt + accepted so far.
            let prefixIDs = promptIDs + accepted
            let outcome = try runOneBlock(prefixIDs: prefixIDs)
            onBlock?(outcome)

            // The walker always returns at least one token (the target's
            // own prediction at the last prefix position). Appending
            // zero-length progress means the walker bailed immediately.
            if outcome.acceptedTokens.isEmpty {
                noProgressBlocks += 1
                if noProgressBlocks >= maxNoProgressBlocks { break }
                continue
            }
            noProgressBlocks = 0

            for tok in outcome.acceptedTokens {
                if accepted.count >= maxNewTokens { break }
                accepted.append(tok)
                if eosTokenIDs.contains(tok) {
                    return accepted
                }
            }
        }
        return accepted
    }

    // MARK: - Single block cycle (shared by CLI smoke and generate loop)

    /// Runs one full cycle: target forward with taps → bonus → drafter
    /// block forward → DDTree → target verify with tree mask → walker.
    /// Returns the accepted tokens for this block plus timing breakdown.
    ///
    /// The `prefixIDs` argument is the FULL prefix the cycle should
    /// condition on (prompt + all accepted tokens from prior blocks).
    /// This method does NOT maintain any state between calls — v1
    /// cacheless semantics.
    public func runOneBlock(prefixIDs: [Int]) throws -> JangDFlashBlockOutcome {
        precondition(!prefixIDs.isEmpty, "prefix must not be empty")

        let B = cfg.blockSize
        let prefixLen = prefixIDs.count

        let prefixArr = MLXArray(prefixIDs.map { Int32($0) })
            .reshaped(1, prefixLen)

        // --- Target forward with tap capture ---
        let tTarget0 = Date()
        let (targetLogits, taps) = target.forwardWithTaps(
            inputs: prefixArr,
            cache: nil,
            tapLayers: cfg.tapLayers,
            providedMask: nil
        )
        evaluateTensor(targetLogits)
        for (_, t) in taps { evaluateTensor(t) }
        let targetWall = Date().timeIntervalSince(tTarget0)

        // Bonus token = argmax at the final prefix position.
        let finalLogits = targetLogits[0, prefixLen - 1]
        let bonusArr = argMax(finalLogits, axis: -1)
        evaluateTensor(bonusArr)
        let bonusID = Int(bonusArr.item(Int32.self))

        // --- Drafter block forward ---
        var blockIDsArr = [Int32](repeating: Int32(drafter.cfg.maskTokenId), count: B)
        blockIDsArr[0] = Int32(bonusID)
        let block = MLXArray(blockIDsArr).reshaped(1, B)

        // Slice/pad the tap concatenation to exactly B context
        // positions, taking the tail of the prefix (the drafter was
        // trained to see the most recent B positions' hidden states).
        let hCtxFull = buildTapConcatenation(taps: taps)
        let Tctx = hCtxFull.dim(1)
        let sliceStart = max(0, Tctx - B)
        let hCtxBlock = hCtxFull[0..., sliceStart ..< Tctx, 0...]
        let hCtxPadded: MLXArray
        if hCtxBlock.dim(1) < B {
            let pad = MLXArray.zeros(
                [1, B - hCtxBlock.dim(1), hCtxBlock.dim(2)],
                dtype: hCtxBlock.dtype
            )
            hCtxPadded = concatenated([pad, hCtxBlock], axis: 1)
        } else {
            hCtxPadded = hCtxBlock
        }

        let tDraft0 = Date()
        let drafterLogits = drafter(block, hTaps: hCtxPadded)
        evaluateTensor(drafterLogits)
        let drafterWall = Date().timeIntervalSince(tDraft0)

        // --- Top-k per slot → beam → trie ---
        let drafterProbs = softmaxLastAxis(drafterLogits[0..., 1..., 0...])
        evaluateTensor(drafterProbs)
        let (vals, ids) = topKPerSlot(probs: drafterProbs, k: cfg.topK)
        let paths = DDTreeBuilder.beamTopMLattice(
            vals: vals, ids: ids, m: cfg.numPaths)
        guard !paths.isEmpty else {
            return JangDFlashBlockOutcome(
                acceptedTokens: [bonusID],
                treeSize: 0,
                targetWallSec: targetWall,
                drafterWallSec: drafterWall,
                verifyWallSec: 0
            )
        }
        let flatTree = DDTreeBuilder.flatten(paths: paths)
        let n = flatTree.flatTokens.count

        // --- Target verify with tree-attention mask ---
        let totalLen = prefixLen + n
        let negInf: Float = -1e9
        var biasFlat = [Float](repeating: negInf, count: totalLen * totalLen)
        for i in 0 ..< prefixLen {
            for j in 0 ... i {
                biasFlat[i * totalLen + j] = 0
            }
        }
        for i in 0 ..< n {
            let treeRow = prefixLen + i
            for j in 0 ..< prefixLen {
                biasFlat[treeRow * totalLen + j] = 0
            }
            for j in 0 ..< n where flatTree.ancestryMask[i][j] {
                biasFlat[treeRow * totalLen + (prefixLen + j)] = 0
            }
        }
        let bias = MLXArray(biasFlat, [totalLen, totalLen])

        let verifyIds: [Int32] =
            prefixIDs.map { Int32($0) } + flatTree.flatTokens.map { Int32($0) }
        let verifyInput = MLXArray(verifyIds).reshaped(1, totalLen)

        let tVerify0 = Date()
        let (verifyLogits, _) = target.forwardWithTaps(
            inputs: verifyInput,
            cache: nil,
            tapLayers: [],
            providedMask: .array(bias)
        )
        evaluateTensor(verifyLogits)
        let verifyWall = Date().timeIntervalSince(tVerify0)

        let treeLogits = verifyLogits[0, prefixLen ..< totalLen]
        let treeArg = argMax(treeLogits, axis: -1)
        evaluateTensor(treeArg)
        let treeArgArr = treeArg.asArray(Int32.self).map { Int($0) }

        let accepted = JangDFlashSpecDec.walkAcceptGreedy(
            flatTokens: flatTree.flatTokens,
            ancestryMask: flatTree.ancestryMask,
            targetArgmax: treeArgArr,
            bonusToken: bonusID
        )

        return JangDFlashBlockOutcome(
            acceptedTokens: accepted,
            treeSize: n,
            targetWallSec: targetWall,
            drafterWallSec: drafterWall,
            verifyWallSec: verifyWall
        )
    }

    // MARK: - v2: Cached multi-block generate

    /// Cached-KV multi-block decode. Maintains a persistent target KV
    /// cache across blocks and trims the verify-forward's contribution
    /// before committing accepted tokens. The commit step re-runs the
    /// target on ONLY the accepted tokens, guaranteeing the cache's
    /// linear append order matches the accepted prefix (no tree
    /// branching leaks into cache state).
    ///
    /// Per-block compute (v2 vs v1):
    ///   v1 (cacheless): target re-processes [prompt + all accepted + flat_trie] every block
    ///   v2 (cached):    target processes [flat_trie] against cached prefix + tiny [accepted] commit
    ///
    /// The commit-forward adds one extra target pass of length A per
    /// block but A is typically 3-6 at B=16, so the saving from not
    /// re-processing the prompt grows with generation length.
    ///
    /// Taps are accumulated in a per-layer buffer across all commit
    /// passes, so the drafter's fusion MLP always sees the last B
    /// positions' hidden states.
    public func cachedGenerate(
        promptIDs: [Int],
        maxNewTokens: Int,
        eosTokenIDs: Set<Int> = [],
        cache: [KVCache]? = nil,
        prefixMatched: Int = 0,
        onBlock: ((JangDFlashBlockOutcome) -> Void)? = nil
    ) throws -> (accepted: [Int], cache: [KVCache]) {
        precondition(!promptIDs.isEmpty, "promptIDs must not be empty")
        precondition(maxNewTokens > 0, "maxNewTokens must be positive")
        precondition(prefixMatched >= 0 && prefixMatched <= promptIDs.count,
                     "prefixMatched must be in [0, promptIDs.count]")

        let B = cfg.blockSize
        let cache = cache ?? target.makeCache()

        // --- Prompt forward (builds cache + initial tap buffer) ---
        // When `prefixMatched > 0`, the caller has pre-warmed the cache
        // via a coordinator fetch (paged / memory / disk restore), so
        // only the tail needs to be forwarded. cache.offset on each
        // layer already reflects `prefixMatched`. See StreamDFlash for
        // the orchestration.
        let promptTail = Array(promptIDs.dropFirst(prefixMatched))
        let forwardIDs = promptTail.isEmpty ? [promptIDs.last!] : promptTail
        let promptArr = MLXArray(forwardIDs.map { Int32($0) }).reshaped(1, forwardIDs.count)
        let (promptLogits, promptTaps) = target.forwardWithTaps(
            inputs: promptArr,
            cache: cache,
            tapLayers: cfg.tapLayers,
            providedMask: nil
        )
        evaluateTensor(promptLogits)

        // Per-layer cumulative tap buffer: starts with prompt taps,
        // grows by each commit-forward's taps.
        var tapBuf: [Int: MLXArray] = [:]
        for (k, v) in promptTaps {
            evaluateTensor(v)
            tapBuf[k] = v
        }

        // Initial bonus token = argmax at the last prompt position.
        var bonusID: Int = {
            let finalLogits = promptLogits[0, promptIDs.count - 1]
            let bonusArr = argMax(finalLogits, axis: -1)
            evaluateTensor(bonusArr)
            return Int(bonusArr.item(Int32.self))
        }()

        var accepted: [Int] = []
        var noProgressBlocks = 0
        let maxNoProgressBlocks = 3

        while accepted.count < maxNewTokens {
            let tTarget0 = Date()   // v2 reuses the "target" slot for the commit-forward timing

            // --- Drafter block input ---
            var blockIDsArr = [Int32](repeating: Int32(drafter.cfg.maskTokenId), count: B)
            blockIDsArr[0] = Int32(bonusID)
            let block = MLXArray(blockIDsArr).reshaped(1, B)

            // Slice the tail of the cumulative tap buffer for the
            // drafter's fusion input. tapBuf entries are shape
            // [1, T_cumulative, hidden_target] per tap layer.
            let hCtxFull = buildTapConcatenation(taps: tapBuf)
            let Tctx = hCtxFull.dim(1)
            let sliceStart = max(0, Tctx - B)
            let hCtxBlock = hCtxFull[0..., sliceStart ..< Tctx, 0...]
            let hCtxPadded: MLXArray
            if hCtxBlock.dim(1) < B {
                let pad = MLXArray.zeros(
                    [1, B - hCtxBlock.dim(1), hCtxBlock.dim(2)],
                    dtype: hCtxBlock.dtype
                )
                hCtxPadded = concatenated([pad, hCtxBlock], axis: 1)
            } else {
                hCtxPadded = hCtxBlock
            }

            // --- Drafter forward ---
            let tDraft0 = Date()
            let drafterLogits = drafter(block, hTaps: hCtxPadded)
            evaluateTensor(drafterLogits)
            let drafterWall = Date().timeIntervalSince(tDraft0)

            // --- Top-k per slot → beam → trie ---
            let drafterProbs = softmaxLastAxis(drafterLogits[0..., 1..., 0...])
            evaluateTensor(drafterProbs)
            let (vals, ids) = topKPerSlot(probs: drafterProbs, k: cfg.topK)
            let paths = DDTreeBuilder.beamTopMLattice(
                vals: vals, ids: ids, m: cfg.numPaths)
            if paths.isEmpty {
                noProgressBlocks += 1
                if noProgressBlocks >= maxNoProgressBlocks { break }
                // No candidates — accept just the current bonus as the only new token.
                if accepted.count < maxNewTokens {
                    accepted.append(bonusID)
                    if eosTokenIDs.contains(bonusID) { return (accepted: accepted, cache: cache) }
                }
                // We have no new bonus without running a commit forward.
                // Break instead of looping forever.
                break
            }
            let flatTree = DDTreeBuilder.flatten(paths: paths)
            let n = flatTree.flatTokens.count

            // --- Verify forward: feed only the flat trie tokens, with
            // the current cache (which holds prompt + previously
            // committed accepted tokens). Tree-attention mask uses
            // visibility against cached positions + in-trie ancestry.
            // Cache grows by N after this call; we will trim it back
            // before the commit forward.
            //
            // The mask shape expected by SDPA when there's a cache is
            // `[1, n_heads, L_query, L_query + L_cache]`. Since we use
            // a single 2-D boolean/additive-bias mask broadcast across
            // heads, we build `[L_query, L_query + L_cache]`.
            //
            // The current cache length = promptIDs.count + accepted.count.
            let cachedLen = promptIDs.count + accepted.count
            let totalLen = cachedLen + n
            let negInf: Float = -1e9
            var biasFlat = [Float](repeating: negInf, count: n * totalLen)
            for i in 0 ..< n {
                // See all of the cached prefix.
                for j in 0 ..< cachedLen {
                    biasFlat[i * totalLen + j] = 0
                }
                // See tree ancestors (indices inside the flat trie).
                for j in 0 ..< n where flatTree.ancestryMask[i][j] {
                    biasFlat[i * totalLen + (cachedLen + j)] = 0
                }
            }
            let bias = MLXArray(biasFlat, [n, totalLen])

            let verifyIds = flatTree.flatTokens.map { Int32($0) }
            let verifyInput = MLXArray(verifyIds).reshaped(1, n)

            let tVerify0 = Date()
            let (verifyLogits, _) = target.forwardWithTaps(
                inputs: verifyInput,
                cache: cache,            // cache will grow by n
                tapLayers: [],
                providedMask: .array(bias)
            )
            evaluateTensor(verifyLogits)
            let verifyWall = Date().timeIntervalSince(tVerify0)

            // Cache is now `cachedLen + n`. Trim back the verify
            // contribution so the commit forward gets a clean starting
            // point. The spec-dec literature calls this "cache rollback".
            _ = trimPromptCache(cache, numTokens: n)

            // --- Walker picks accepted prefix ---
            let treeArg = argMax(verifyLogits[0], axis: -1)
            evaluateTensor(treeArg)
            let treeArgArr = treeArg.asArray(Int32.self).map { Int($0) }

            let pickedFromTree = JangDFlashSpecDec.walkAcceptGreedy(
                flatTokens: flatTree.flatTokens,
                ancestryMask: flatTree.ancestryMask,
                targetArgmax: treeArgArr,
                bonusToken: bonusID
            )

            // The walker always returns at least one token. If the
            // tree produced only the bonus (length 1), commit that
            // single token via a 1-step target forward.
            let newTokens = pickedFromTree

            // --- Commit forward: target sees just the accepted tokens,
            // cache grows by A, and we capture fresh taps for the
            // cumulative buffer.
            let tCommit0 = Date()
            let commitIds = newTokens.map { Int32($0) }
            let commitInput = MLXArray(commitIds).reshaped(1, newTokens.count)
            let (commitLogits, commitTaps) = target.forwardWithTaps(
                inputs: commitInput,
                cache: cache,
                tapLayers: cfg.tapLayers,
                providedMask: nil
            )
            evaluateTensor(commitLogits)
            for (_, t) in commitTaps { evaluateTensor(t) }
            let commitWall = Date().timeIntervalSince(tCommit0)
            let targetWall = Date().timeIntervalSince(tTarget0) + commitWall

            // Append commit taps to cumulative buffer.
            for (k, v) in commitTaps {
                if let existing = tapBuf[k] {
                    tapBuf[k] = concatenated([existing, v], axis: 1)
                } else {
                    tapBuf[k] = v
                }
            }

            // New bonus token = argmax at the last commit position.
            let lastLogits = commitLogits[0, newTokens.count - 1]
            let nextBonusArr = argMax(lastLogits, axis: -1)
            evaluateTensor(nextBonusArr)
            bonusID = Int(nextBonusArr.item(Int32.self))

            // Append accepted tokens to the running list, respecting
            // maxNewTokens and EOS.
            for tok in newTokens {
                if accepted.count >= maxNewTokens { break }
                accepted.append(tok)
                if eosTokenIDs.contains(tok) {
                    onBlock?(
                        JangDFlashBlockOutcome(
                            acceptedTokens: newTokens,
                            treeSize: n,
                            targetWallSec: targetWall,
                            drafterWallSec: drafterWall,
                            verifyWallSec: verifyWall
                        )
                    )
                    return (accepted: accepted, cache: cache)
                }
            }

            onBlock?(
                JangDFlashBlockOutcome(
                    acceptedTokens: newTokens,
                    treeSize: n,
                    targetWallSec: targetWall,
                    drafterWallSec: drafterWall,
                    verifyWallSec: verifyWall
                )
            )

            if newTokens.isEmpty {
                noProgressBlocks += 1
                if noProgressBlocks >= maxNoProgressBlocks { break }
            } else {
                noProgressBlocks = 0
            }
        }

        return (accepted: accepted, cache: cache)
    }

    // Private helper so the `MLX.eval` / `asyncEval` identifier
    // doesn't appear scattered through this file. `asyncEval` kicks
    // GPU work off without blocking; subsequent `.item()` / shape
    // reads will block on completion so correctness is preserved.
    @inline(__always)
    private func evaluateTensor(_ a: MLXArray) { asyncEval(a) }

    // MARK: - Helper: concatenate per-layer taps into the drafter tap_dim

    /// Stacks the per-layer tap tensors into a single `[B, T, tapDim]`
    /// tensor in the same layer order the drafter was trained on
    /// (ascending tap-layer index).
    ///
    /// The tap dict may contain extra layers beyond those in
    /// `cfg.tapLayers`; only the configured layers are used, sorted by
    /// index so the concatenation order is deterministic.
    public func buildTapConcatenation(
        taps: [Int: MLXArray]
    ) -> MLXArray {
        let orderedIndices = cfg.tapLayers.sorted()
        precondition(
            orderedIndices.allSatisfy { taps[$0] != nil },
            "JangDFlashSpecDec: taps dict missing one of the configured layers"
        )
        let tensors: [MLXArray] = orderedIndices.map { idx in
            guard let t = taps[idx] else {
                fatalError("tap layer \(idx) missing — precondition should have caught this")
            }
            return t
        }
        // Each tap is [B, T, targetHiddenDim]. Concat along the last
        // axis to get [B, T, numTapLayers * targetHiddenDim].
        return concatenated(tensors, axis: -1)
    }

    // MARK: - Helper: convert top-k logits to DDTree-ready arrays

    /// Given the drafter's per-position softmax `[1, B-1, V]`, extract
    /// top-k ids and probabilities per slot as plain Swift arrays for
    /// the CPU-side `DDTreeBuilder`.
    public func topKPerSlot(
        probs: MLXArray, k: Int
    ) -> (vals: [[Float]], ids: [[Int]]) {
        precondition(probs.ndim == 3, "expected [B, numSlots, V], got shape \(probs.shape)")
        precondition(probs.dim(0) == 1, "spec-dec runs with batch size 1")

        let numSlots = probs.dim(1)
        var vals = [[Float]](); vals.reserveCapacity(numSlots)
        var ids = [[Int]](); ids.reserveCapacity(numSlots)

        // Extract top-k across the vocab axis. `argPartition` + gather
        // is cheaper than sorting the whole vocab, but correctness
        // first — sort the top-k descending.
        for slot in 0 ..< numSlots {
            let row = probs[0, slot]   // [V]
            // argPartition returns indices; the top-k live at the tail.
            let V = row.shape[0]
            let pivot = max(0, V - k)
            let part = argPartition(row, kth: pivot, axis: -1)
            let topIdxRaw = part[pivot ..< V]  // [k], unsorted
            // Gather values at those indices.
            let topVals = row[topIdxRaw]
            // Sort descending by value for deterministic test output.
            let valsArr = topVals.asArray(Float.self)
            let idxArr = topIdxRaw.asArray(Int32.self).map { Int($0) }
            let ordered = zip(valsArr, idxArr).sorted { $0.0 > $1.0 }
            vals.append(ordered.map { $0.0 })
            ids.append(ordered.map { $0.1 })
        }
        return (vals, ids)
    }

    // MARK: - Helper: pure-Swift softmax along last axis

    /// Numerically-stable softmax over the last axis, returned as
    /// `[1, numSlots, V]`. Uses MLX ops and stays on-device.
    public func softmaxLastAxis(_ logits: MLXArray) -> MLXArray {
        softmax(logits, axis: -1)
    }

    // MARK: - Helper: build tree-attention mask MLXArray

    /// Converts the `[[Bool]]` ancestry mask from `DDTreeBuilder.flatten`
    /// into an additive-bias MLXArray of shape `[N, N]` (0 where visible,
    /// large negative where masked).
    public static func ancestryMaskToAdditiveBias(
        _ mask: [[Bool]]
    ) -> MLXArray {
        let n = mask.count
        precondition(n > 0, "empty mask")
        var flat = [Float](repeating: -1e9, count: n * n)
        for i in 0 ..< n {
            precondition(
                mask[i].count == n,
                "mask row \(i) has length \(mask[i].count), expected \(n)")
            for j in 0 ..< n where mask[i][j] {
                flat[i * n + j] = 0
            }
        }
        return MLXArray(flat, [n, n])
    }

    // MARK: - Helper: rejection-sampling walk of the verified trie

    /// Walks the prefix trie top-down, picking at each depth the highest-
    /// joint-prob child that survives rejection sampling, and returns the
    /// accepted token sequence.
    ///
    /// For greedy decoding (temperature = 0) this collapses to "pick the
    /// highest-drafter-prob child IF the target's argmax matches it;
    /// otherwise stop and emit the target's argmax as the bonus token."
    /// That's the simplest correct path and matches the DFlash paper's
    /// reported numbers which all use greedy verification.
    ///
    /// - Parameters:
    ///   - flatTokens: `DDTreeBuilder.FlatTree.flatTokens`
    ///   - ancestryMask: `DDTreeBuilder.FlatTree.ancestryMask`
    ///   - targetArgmax: per-flat-node argmax from the target forward.
    ///     Shape `[N]` where `N == flatTokens.count`.
    ///   - bonusToken: the clean anchor token from the previous round;
    ///     used as the "current position" of the walk start. If no
    ///     drafted child's token equals `targetArgmax[currentParent]`,
    ///     the walk ends immediately and we emit just `targetArgmax`
    ///     at the root position.
    public static func walkAcceptGreedy(
        flatTokens: [Int],
        ancestryMask: [[Bool]],
        targetArgmax: [Int],
        bonusToken: Int
    ) -> [Int] {
        precondition(
            flatTokens.count == targetArgmax.count,
            "flatTokens and targetArgmax length mismatch"
        )
        let n = flatTokens.count
        guard n > 0 else { return [bonusToken] }

        // Helper: children of flat index `parent` are the nodes whose
        // ancestor set includes `parent` and whose *immediate* ancestor
        // (excluding themselves) has the largest size less than their
        // own. Equivalently, a child is any j where ancestryMask[j][parent]
        // == true AND depth(j) == depth(parent) + 1.
        //
        // We reconstruct depth from each row's popcount minus 1 (each
        // node's ancestor row includes itself).
        var depth = [Int](repeating: 0, count: n)
        for i in 0 ..< n {
            var count = 0
            for j in 0 ..< n where ancestryMask[i][j] { count += 1 }
            depth[i] = count - 1  // exclude self
        }

        var accepted: [Int] = []
        // "Parent" is the current node we're expanding from. Start with
        // the virtual root whose argmax comparison is against the bonus
        // token — i.e. the first drafted depth is 0. Find depth-0 nodes
        // and check whether any of their tokens matches the target's
        // argmax at its position. We need a per-node target argmax to
        // compare against the *parent's* predicted next token, so we
        // pass `bonusToken` separately for the root case.

        // Find depth-0 nodes.
        let rootChildren = (0 ..< n).filter { depth[$0] == 0 }

        // The root's "target argmax next token" is the argmax at the
        // position that produced the root drafted tokens — conceptually
        // that's the position whose query was `bonusToken`. In our flat
        // layout the target forward runs over [prefix_tokens + flatTokens]
        // so argmax[flatNode] is the model's prediction AT that node's
        // position (the prediction of the NEXT token given the node and
        // its ancestors). For the root position we need the prediction
        // that would come from the *last accepted token* before the
        // tree started — that's provided by the caller as `bonusToken`'s
        // companion argmax. In this simplified walker we assume the
        // caller has already verified that the bonus token is aligned
        // and `targetArgmax[rootChild]` is the prediction of the token
        // AFTER acceptance of that root child. The implication: we
        // accept a root child iff its token is "plausible next" — in
        // practice the target runs forward on the (prefix + flat) so
        // the argmax at each node is what the target says should come
        // NEXT. We compare the root child's token against targetArgmax
        // at the root child's own flat index's parent — which for depth-0
        // is the prefix end — and we don't have that prediction in
        // `targetArgmax`. So the caller must supply the root prediction
        // separately via `bonusToken`: it represents the target's
        // argmax at the prefix-end position.
        //
        // For the simplified walker we assume `bonusToken` IS that
        // prefix-end argmax, and we accept the root child whose token
        // matches it; if none match, accept exactly `bonusToken` as
        // the one accepted token this round.
        guard
            let firstAcceptedRoot = rootChildren.first(where: { flatTokens[$0] == bonusToken })
        else {
            return [bonusToken]
        }
        accepted.append(flatTokens[firstAcceptedRoot])
        var currentParent = firstAcceptedRoot

        // Walk deeper.
        while true {
            // Children of `currentParent`: j such that depth[j] ==
            // depth[currentParent] + 1 AND ancestryMask[j][currentParent]
            // is true.
            let childDepth = depth[currentParent] + 1
            let children = (0 ..< n).filter {
                depth[$0] == childDepth && ancestryMask[$0][currentParent]
            }
            if children.isEmpty { break }

            // Greedy verify: look at target's argmax at currentParent
            // (i.e. "what would the target predict comes after the
            // currently-accepted prefix") and see if any drafted child
            // token matches.
            let predicted = targetArgmax[currentParent]
            guard let match = children.first(where: { flatTokens[$0] == predicted }) else {
                // Mismatch. The target's prediction overrides the
                // drafter — emit `predicted` as the bonus and stop.
                accepted.append(predicted)
                break
            }
            accepted.append(flatTokens[match])
            currentParent = match
        }

        return accepted
    }
}
