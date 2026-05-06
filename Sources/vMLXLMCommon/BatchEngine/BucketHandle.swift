// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1B.4 scaffolding — `BucketHandle` and row allocation logic.
//
// When the compile path goes multi-row (maxBatchSize > 1), each active
// bucket owns per-layer `[B, H, maxLen, D]` CompilableKVCache objects.
// Slots claim individual rows (indices into the 0..<B dimension) and
// release them when finished. The bucket cache objects have stable
// identity across every decode step so the compiled forward trace holds.
//
// This file ships the *data types and pure-logic helpers* — not the
// live per-bucket `CompilableKVCache` layer allocation (that's Stage
// 1B.4 proper, which requires model introspection and is tighter to
// the engine loop than a pure-logic file should be).
//
// Pattern matches the Stage 1A `BatchCompile.swift` split: types in
// one file, testable in isolation, wired into the engine in a follow-up.

import Foundation
import MLX

// MARK: - Row Allocator

/// Per-bucket row allocator. Tracks which rows in `0..<bucketSize` are
/// assigned to active slots and which are free for the next admission.
///
/// ## Invariants
///
/// - `bucketSize` is fixed after init. A bucket of size 4 always has
///   exactly rows 0, 1, 2, 3 — never grows or shrinks.
/// - `freeRows` and the key set of `rowOf` are disjoint unions
///   covering `0..<bucketSize`.
/// - Row indices don't move when slots are released. Slot 0 might be
///   at row 2; if slot 0 finishes, row 2 becomes free, but the other
///   slots' row assignments don't shift.
///
/// ## Thread safety
///
/// `BucketRowAllocator` is NOT thread-safe on its own. It's designed
/// to be held inside an actor (`BatchCompile` at the engine level) so
/// the actor's isolation handles the concurrency.
public struct BucketRowAllocator {

    public let bucketSize: Int

    /// Rows currently free for admission. Kept sorted ascending so
    /// `claim()` deterministically picks the smallest free row. This
    /// aids compile-trace debugging (dead rows cluster at the tail
    /// of the bucket, making mask inspection predictable).
    private var freeRows: [Int]

    /// Slot-ID → row-index map for currently active slots.
    private var rowOf: [BatchRequestID: Int]

    public init(bucketSize: Int) {
        precondition(bucketSize >= 1, "bucketSize must be >= 1")
        self.bucketSize = bucketSize
        self.freeRows = Array(0 ..< bucketSize)
        self.rowOf = [:]
    }

    /// Number of rows currently assigned to active slots.
    public var liveCount: Int { rowOf.count }

    /// Number of rows currently available for admission.
    public var freeCount: Int { freeRows.count }

    /// Whether the bucket is full (no free rows).
    public var isFull: Bool { freeRows.isEmpty }

    /// Whether the bucket is completely empty (all rows free).
    public var isEmpty: Bool { freeRows.count == bucketSize }

    /// Claim the smallest-indexed free row for `slotID`. Returns the
    /// row index. Traps if the bucket is full (callers must check
    /// `isFull` first or route through a different bucket).
    @discardableResult
    public mutating func claim(slotID: BatchRequestID) -> Int {
        precondition(!freeRows.isEmpty,
            "BucketRowAllocator.claim on full bucket — caller must check isFull first")
        precondition(rowOf[slotID] == nil,
            "BucketRowAllocator.claim called twice for slot \(slotID)")
        let row = freeRows.removeFirst()
        rowOf[slotID] = row
        return row
    }

    /// Release the row assigned to `slotID`. Returns the released row
    /// index for logging / test assertions. Returns `nil` if the slot
    /// wasn't holding a row (double-release is a no-op by design — the
    /// engine loop calls this on slot teardown, which can race with
    /// cancellation).
    @discardableResult
    public mutating func release(slotID: BatchRequestID) -> Int? {
        guard let row = rowOf.removeValue(forKey: slotID) else {
            return nil
        }
        // Re-insert sorted so next claim picks smallest-index row
        // deterministically.
        let insertAt = freeRows.firstIndex(where: { $0 > row }) ?? freeRows.count
        freeRows.insert(row, at: insertAt)
        return row
    }

    /// Get the row assigned to `slotID`. Returns `nil` if the slot
    /// isn't in this bucket.
    public func row(for slotID: BatchRequestID) -> Int? {
        rowOf[slotID]
    }

    /// Rows currently live, sorted ascending. Used by `makeLiveMask`
    /// callers at the top of each decode step.
    public var liveRows: [Int] {
        rowOf.values.sorted()
    }
}

// MARK: - Dead-row input builder

/// Pure-logic helpers for dead-row masking and decode input construction.
///
/// When a bucket has fewer live rows than its `bucketSize`, the compile
/// trace still runs at full `B = bucketSize`. Dead rows need:
///  1. A placeholder token (any valid token ID works; we use `0`).
///  2. A liveness mask entry of `false` so attention ignores them.
///  3. Their `offsetArray[row]` clamped to `0` so the dead row's mask
///     doesn't spuriously extend valid regions of OTHER rows.
///
/// These three helpers produce the three required MLXArrays.
public enum BucketDeadRow {

    /// Placeholder token ID used for dead rows. Any valid token ID
    /// works — the attention mask zeroes out the contribution anyway.
    /// Using `0` keeps buffer writes simple.
    public static let placeholderTokenID: Int32 = 0

    /// Build the `[B, 1]` decode input tensor by placing each live
    /// slot's next token at its row index, and the placeholder at dead
    /// rows. Stable ordering: `liveTokens[i]` lands at `liveRows[i]`.
    ///
    /// - Parameters:
    ///   - bucketSize: The full bucket batch size (dimension 0 of the
    ///     compile-traced forward input).
    ///   - liveRows: Row indices of live slots, sorted ascending.
    ///   - liveTokens: Swift array of token IDs; `liveTokens.count ==
    ///     liveRows.count`. Order must match `liveRows`.
    /// - Returns: An `MLXArray` of shape `[bucketSize, 1]` dtype int32.
    public static func decodeInput(
        bucketSize: Int, liveRows: [Int], liveTokens: [Int32]
    ) -> MLXArray {
        precondition(liveRows.count == liveTokens.count,
            "liveRows.count (\(liveRows.count)) must equal liveTokens.count (\(liveTokens.count))")
        precondition(liveRows.allSatisfy { $0 >= 0 && $0 < bucketSize },
            "All liveRows must be in 0..<bucketSize")

        var flat = [Int32](repeating: placeholderTokenID, count: bucketSize)
        for (i, row) in liveRows.enumerated() {
            flat[row] = liveTokens[i]
        }
        return MLXArray(flat).reshaped(bucketSize, 1)
    }

    /// Build a per-row boolean liveness flag of shape `[bucketSize]`.
    /// Live rows are `true`, dead rows are `false`. Used by the
    /// engine-level attention mask computation (not inside the
    /// compile trace directly — live-set changes every step, so the
    /// mask is passed in as a state input).
    ///
    /// Same semantics as `BatchCompile.makeLiveMask` but canonicalised
    /// here under the Stage 1B.4 namespace.
    public static func liveFlags(
        bucketSize: Int, liveRows: [Int]
    ) -> MLXArray {
        var flags = [Int32](repeating: 0, count: bucketSize)
        for row in liveRows {
            precondition(row >= 0 && row < bucketSize,
                "liveRows entries must be in 0..<bucketSize")
            flags[row] = 1
        }
        return MLXArray(flags)
    }
}
