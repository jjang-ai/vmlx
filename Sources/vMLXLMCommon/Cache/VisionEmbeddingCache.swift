// SPDX-License-Identifier: Apache-2.0
//
// Vision Embedding Cache — per-image pixel-values cache for VLM
// continuous batching. Port of `vmlx_engine/vision_embedding_cache.py`.
//
// Without this cache, every multi-turn VLM request re-runs the vision
// tower from scratch (image load → resize → normalize → patch-embed →
// ViT forward) even when the user is still chatting about the same
// image. Hit rate gains on a single-image multi-turn session are
// effectively 100% after the first turn.
//
// Key construction:
//   hash = SHA-256(concatenated SHA-256(image_bytes) || SHA-256(prompt))
//
// For file inputs we hash the bytes; for URL / base64 inputs we hash
// the raw string. Both are truncated to 16 hex chars for the image
// part and 12 for the prompt part (same as Python).
//
// Cache payload: `PixelCacheEntry` bundles the post-processing outputs
// — `pixelValues`, `inputIds`, optional attention mask, optional
// `image_grid_thw`, plus a free-form `extraKwargs` dict for per-model
// bonus tensors (Qwen's `rope_deltas`, Gemma's `image_soft_token_mask`,
// etc.). LRU eviction via an ordered-dict pattern (doubly-linked list
// + hash map for O(1) get/put/evict).

import Foundation
import CryptoKit
import MLX

// MARK: - Stats

public struct VisionCacheStats: Sendable, Equatable {
    public var pixelCacheHits: Int = 0
    public var pixelCacheMisses: Int = 0
    public var totalTimeSavedSec: Double = 0.0
    public var totalImagesProcessed: Int = 0

    public var pixelHitRate: Double {
        let total = pixelCacheHits + pixelCacheMisses
        return total > 0 ? Double(pixelCacheHits) / Double(total) : 0.0
    }

    public var entryCount: Int = 0

    public func toDictionary() -> [String: Any] {
        [
            "pixel_cache_hits": pixelCacheHits,
            "pixel_cache_misses": pixelCacheMisses,
            "pixel_hit_rate": pixelHitRate,
            "total_time_saved": totalTimeSavedSec,
            "total_images_processed": totalImagesProcessed,
            "pixel_cache_size": entryCount,
        ]
    }
}

// MARK: - Entry

/// One cached output of `processor.prepare(input:)`. We store the
/// post-fusion `pixelValues` + the text tokens + whatever ancillary
/// tensors the specific VLM needs (image_grid_thw for Qwen3-VL, soft
/// token mask for Gemma 4, etc). `extraTensors` is a Dictionary so
/// each model family can stuff whatever it needs without bloating
/// the base entry.
///
/// All tensor fields are optional so unit tests can exercise the
/// cache logic (keys, LRU, stats) without a loaded Metal library.
/// In real use, `pixelValues` and `inputIds` will always be populated.
public final class PixelCacheEntry: @unchecked Sendable {
    public let pixelValues: MLXArray?
    public let inputIds: MLXArray?
    public let attentionMask: MLXArray?
    public let imageGridTHW: MLXArray?
    public let extraTensors: [String: MLXArray]
    public let processingTimeSec: Double

    public init(
        pixelValues: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        imageGridTHW: MLXArray? = nil,
        extraTensors: [String: MLXArray] = [:],
        processingTimeSec: Double = 0.0
    ) {
        self.pixelValues = pixelValues
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.imageGridTHW = imageGridTHW
        self.extraTensors = extraTensors
        self.processingTimeSec = processingTimeSec
    }
}

// MARK: - Hashing helpers

/// Compute a stable 16-char hex key for one image.
///
/// If `image` resolves to a readable file, we hash the file contents
/// (same as Python — this catches the case where the path is the same
/// but the file got overwritten). Otherwise — URL / data: URL /
/// base64 string — we hash the raw string.
public func vmlxComputeImageHash(_ image: String) -> String {
    let fm = FileManager.default
    let url = URL(fileURLWithPath: image)
    if fm.fileExists(atPath: url.path) {
        if let data = try? Data(contentsOf: url) {
            let digest = SHA256.hash(data: data)
            return String(digest.compactMap { String(format: "%02x", $0) }
                .joined().prefix(16))
        }
    }
    let digest = SHA256.hash(data: Data(image.utf8))
    return String(digest.compactMap { String(format: "%02x", $0) }
        .joined().prefix(16))
}

/// Combine hashes for multiple images. `no_images` sentinel matches
/// the Python behavior so text-only requests always skip the cache.
public func vmlxComputeImagesHash(_ images: [String]) -> String {
    guard !images.isEmpty else { return "no_images" }
    let joined = images.map(vmlxComputeImageHash).joined(separator: "_")
    let digest = SHA256.hash(data: Data(joined.utf8))
    return String(digest.compactMap { String(format: "%02x", $0) }
        .joined().prefix(16))
}

// MARK: - VisionEmbeddingCache

/// Thread-safe LRU cache for pre-processed VLM pixel tensors.
///
/// Keyed on `hash(images) || hash(prompt)` so the same image with a
/// different prompt still hits for the image-processing portion of
/// the work (though we cache whole entries, not image-only, because
/// the prompt also affects `inputIds` + `imageGridTHW` layout).
///
/// Concurrency: guarded by `OSAllocatedUnfairLock` on Darwin, plain
/// `NSLock` elsewhere. Lookups promote the hit to most-recently-used
/// via a doubly-linked list; inserts evict the LRU head at capacity.
public final class VisionEmbeddingCache: @unchecked Sendable {

    // MARK: - Node (doubly-linked list)

    private final class Node {
        let key: String
        var value: PixelCacheEntry
        var prev: Node?
        var next: Node?
        init(key: String, value: PixelCacheEntry) {
            self.key = key
            self.value = value
        }
    }

    // MARK: - Config

    public let maxPixelEntries: Int
    public var enabled: Bool

    // MARK: - State

    private let lock = NSLock()
    private var map: [String: Node] = [:]
    private var head: Node?   // least recently used
    private var tail: Node?   // most recently used
    private var stats = VisionCacheStats()

    // MARK: - Init

    public init(maxPixelEntries: Int = 16, enabled: Bool = true) {
        precondition(maxPixelEntries >= 1, "maxPixelEntries must be >= 1")
        self.maxPixelEntries = maxPixelEntries
        self.enabled = enabled
    }

    // MARK: - Key

    /// Compose the cache key for a given (images, prompt) pair.
    /// Prompt is hashed separately at 12-char precision because the
    /// 16-char image hash is already doing most of the uniqueness
    /// work — matches Python exactly.
    public func makeKey(images: [String], prompt: String) -> String {
        let imgHash = vmlxComputeImagesHash(images)
        let promptHash = SHA256.hash(data: Data(prompt.utf8))
            .compactMap { String(format: "%02x", $0) }
            .joined()
            .prefix(12)
        return "\(imgHash)_\(promptHash)"
    }

    // MARK: - Get

    /// Retrieve a cached entry for the given (images, prompt) pair.
    /// Returns `nil` when disabled, empty images, or a miss.
    /// Promotes the hit to most-recently-used.
    public func get(images: [String], prompt: String) -> PixelCacheEntry? {
        guard enabled, !images.isEmpty else { return nil }
        let key = makeKey(images: images, prompt: prompt)
        return lock.withLock {
            guard let node = map[key] else {
                stats.pixelCacheMisses += 1
                return nil
            }
            stats.pixelCacheHits += 1
            stats.totalTimeSavedSec += node.value.processingTimeSec
            moveToTail(node)
            return node.value
        }
    }

    // MARK: - Put

    /// Store a processed entry. Evicts the LRU head if at capacity.
    public func put(
        images: [String],
        prompt: String,
        entry: PixelCacheEntry
    ) {
        guard enabled, !images.isEmpty else { return }
        let key = makeKey(images: images, prompt: prompt)
        lock.withLock {
            if let existing = map[key] {
                existing.value = entry
                moveToTail(existing)
                stats.totalImagesProcessed += images.count
                return
            }
            while map.count >= maxPixelEntries, let lru = head {
                unlink(lru)
                map.removeValue(forKey: lru.key)
            }
            let node = Node(key: key, value: entry)
            appendToTail(node)
            map[key] = node
            stats.totalImagesProcessed += images.count
        }
    }

    // MARK: - Clear / stats

    public func clear() {
        lock.withLock {
            map.removeAll()
            head = nil
            tail = nil
            stats = VisionCacheStats()
        }
    }

    public func snapshotStats() -> VisionCacheStats {
        lock.withLock {
            var s = stats
            s.entryCount = map.count
            return s
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
