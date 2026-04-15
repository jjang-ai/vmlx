// SPDX-License-Identifier: Apache-2.0
//
// SlotBankCache — thread-safe LRU cache for MoE `ExpertWeightSet`s.
//
// Port of `vmlx_engine/utils/flash_moe_loader.py:SlotBankCache`.
// Each slot holds one expert's full tensor bundle (gate/up/down
// projection weights, scales, biases). When the cache is full, the
// least-recently-used slot is evicted. Lookups promote the hit to
// most-recently-used.
//
// Concurrency: guarded by `OSAllocatedUnfairLock` on Darwin, plain
// NSLock elsewhere. The underlying `OrderedDictionary` analogue is
// implemented with a `[String: Int]` head-pointer + linked list so
// get/put/evict are all O(1).

import Foundation
import os

/// Statistics for slot-bank cache utilisation.
public struct SlotBankStats: Sendable, Equatable {
    public var slotsUsed: Int = 0
    public var maxSlots: Int = 0
    public var hits: Int = 0
    public var misses: Int = 0

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0.0
    }
}

/// Thread-safe LRU cache for `ExpertWeightSet`. Keys are
/// `"<layerIdx>:<expertIdx>"` strings for parity with Python.
public final class SlotBankCache: @unchecked Sendable {

    // MARK: - Node (doubly-linked list)

    private final class Node {
        let key: String
        var value: ExpertWeightSet
        var prev: Node?
        var next: Node?
        init(key: String, value: ExpertWeightSet) {
            self.key = key
            self.value = value
        }
    }

    // MARK: - Properties

    public let maxSlots: Int
    private let lock = OSAllocatedUnfairLock()

    /// Key → node map for O(1) lookup.
    private var map: [String: Node] = [:]
    /// LRU doubly-linked list head (least recently used).
    private var head: Node?
    /// LRU doubly-linked list tail (most recently used).
    private var tail: Node?

    private var hits: Int = 0
    private var misses: Int = 0

    // MARK: - Init

    public init(maxSlots: Int) {
        precondition(maxSlots >= 1, "maxSlots must be >= 1")
        self.maxSlots = maxSlots
    }

    // MARK: - Key helper

    @inline(__always)
    public static func key(layerIdx: Int, expertIdx: Int) -> String {
        "\(layerIdx):\(expertIdx)"
    }

    // MARK: - Lookup

    /// Retrieve an expert from the cache and promote it to
    /// most-recently-used. Returns `nil` on miss.
    public func get(layerIdx: Int, expertIdx: Int) -> ExpertWeightSet? {
        let k = Self.key(layerIdx: layerIdx, expertIdx: expertIdx)
        return lock.withLock {
            guard let node = map[k] else {
                misses += 1
                return nil
            }
            hits += 1
            moveToTail(node)
            return node.value
        }
    }

    // MARK: - Insert

    /// Insert an expert weight set, evicting the LRU slot if full.
    /// If the key already exists, the existing entry is updated and
    /// promoted to most-recently-used.
    public func put(_ expert: ExpertWeightSet) {
        let k = Self.key(layerIdx: expert.layerIdx, expertIdx: expert.expertIdx)
        lock.withLock {
            if let existing = map[k] {
                existing.value = expert
                moveToTail(existing)
                return
            }
            if map.count >= maxSlots, let lru = head {
                // Evict LRU.
                unlink(lru)
                map.removeValue(forKey: lru.key)
            }
            let node = Node(key: k, value: expert)
            appendToTail(node)
            map[k] = node
        }
    }

    // MARK: - Clear

    public func clear() {
        lock.withLock {
            map.removeAll()
            head = nil
            tail = nil
            hits = 0
            misses = 0
        }
    }

    // MARK: - Stats

    public func stats() -> SlotBankStats {
        lock.withLock {
            SlotBankStats(
                slotsUsed: map.count,
                maxSlots: maxSlots,
                hits: hits,
                misses: misses
            )
        }
    }

    // MARK: - Private: linked-list plumbing

    private func appendToTail(_ node: Node) {
        node.prev = tail
        node.next = nil
        if let t = tail { t.next = node } else { head = node }
        tail = node
    }

    private func unlink(_ node: Node) {
        if let p = node.prev { p.next = node.next } else { head = node.next }
        if let n = node.next { n.prev = node.prev } else { tail = node.prev }
        node.prev = nil
        node.next = nil
    }

    private func moveToTail(_ node: Node) {
        guard node !== tail else { return }
        unlink(node)
        appendToTail(node)
    }
}
