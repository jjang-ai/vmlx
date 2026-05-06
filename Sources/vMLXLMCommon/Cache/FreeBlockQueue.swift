// Copyright © 2024 Apple Inc.

import Foundation

/// A doubly-linked list implementing O(1) LRU eviction for ``CacheBlock`` instances.
///
/// Blocks at the **front** (head) are the *least* recently used and will be
/// evicted first by ``popFirst()``. Blocks at the **back** (tail) are the
/// *most* recently used. Call ``touch(_:)`` to move a block to the back,
/// marking it as recently used.
///
/// All mutating operations run in O(1) time thanks to `nodeMap`, a dictionary
/// mapping `blockId` to the corresponding linked-list node.
public final class FreeBlockQueue: @unchecked Sendable {

    // MARK: - Node

    private final class Node {
        let block: CacheBlock
        var prev: Node?
        var next: Node?
        init(_ block: CacheBlock) { self.block = block }
    }

    // MARK: - Properties

    private var head: Node?
    private var tail: Node?

    /// O(1) lookup from blockId to node.
    private var nodeMap: [Int: Node] = [:]

    /// The number of blocks currently in the queue.
    public private(set) var count: Int = 0

    // MARK: - Initialization

    public init() {}

    // MARK: - Public API

    /// Append a block to the back of the queue (most recently used position).
    ///
    /// - Parameter block: The cache block to enqueue.
    /// - Complexity: O(1).
    public func append(_ block: CacheBlock) {
        // Guard against duplicate insertion.
        guard nodeMap[block.blockId] == nil else { return }

        let node = Node(block)
        nodeMap[block.blockId] = node
        count += 1

        if let oldTail = tail {
            oldTail.next = node
            node.prev = oldTail
            tail = node
        } else {
            // Queue was empty.
            head = node
            tail = node
        }
    }

    /// Remove and return the block at the front of the queue (least recently used).
    ///
    /// - Returns: The least-recently-used ``CacheBlock``, or `nil` if the queue is empty.
    /// - Complexity: O(1).
    public func popFirst() -> CacheBlock? {
        guard let frontNode = head else { return nil }
        unlinkNode(frontNode)
        return frontNode.block
    }

    /// Remove a specific block from the queue by its ``CacheBlock/blockId``.
    ///
    /// This is a no-op if the block is not in the queue.
    ///
    /// - Parameter block: The cache block to remove.
    /// - Complexity: O(1).
    @discardableResult
    public func remove(_ block: CacheBlock) -> Bool {
        guard let node = nodeMap[block.blockId] else { return false }
        unlinkNode(node)
        return true
    }

    /// Move a block to the back of the queue, marking it as most recently used.
    ///
    /// Equivalent to ``remove(_:)`` followed by ``append(_:)``, but avoids
    /// an extra dictionary lookup.
    ///
    /// This is a no-op if the block is not in the queue.
    ///
    /// - Parameter block: The cache block to touch.
    /// - Complexity: O(1).
    public func touch(_ block: CacheBlock) {
        guard let node = nodeMap[block.blockId] else { return }

        // Already at the tail — nothing to do.
        if node === tail { return }

        // Detach from current position (without removing from nodeMap).
        if let prev = node.prev {
            prev.next = node.next
        } else {
            // node is the head.
            head = node.next
        }
        node.next?.prev = node.prev

        // Re-attach at the tail.
        node.prev = tail
        node.next = nil
        tail?.next = node
        tail = node
    }

    // MARK: - Private Helpers

    /// Unlink a node from the list and remove it from `nodeMap`.
    private func unlinkNode(_ node: Node) {
        nodeMap.removeValue(forKey: node.block.blockId)
        count -= 1

        if let prev = node.prev {
            prev.next = node.next
        } else {
            head = node.next
        }

        if let next = node.next {
            next.prev = node.prev
        } else {
            tail = node.prev
        }

        node.prev = nil
        node.next = nil
    }
}
