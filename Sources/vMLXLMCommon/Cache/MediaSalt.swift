// Copyright © 2026 Osaurus AI
//
// Media salt: a stable fingerprint of VLM image/video inputs for cache keying.
//
// Problem
// -------
// The multi-tier cache coordinator keys its lookups by the flat text token list.
// For VLMs, the model replaces placeholder `<image>` tokens with the vision
// tower's output embeddings during forward pass — so the same token prefix
// with a different image produces different KV state. Naive text-only keying
// would yield false-positive cache hits with the wrong KV for image positions,
// which is why `TokenIterator.init` historically bypassed the cache entirely
// whenever `input.image != nil || input.video != nil`.
//
// Bypassing the cache is extremely expensive: every turn of a multi-turn VLM
// chat re-runs the full vision tower AND re-prefills every prior text token,
// even though neither has changed. On Gemma 4 VLM at 2K context that's
// hundreds of milliseconds to seconds of wasted work per turn.
//
// Fix
// ---
// Compute a SHA256 fingerprint of the raw pixel bytes + shape + dtype of any
// image/video input, and pass it alongside the token list to the cache tiers.
// Each tier mixes the salt into its internal hash (same way modelKey is
// mixed), so "same text prefix + same image" cache-hits while "same text +
// different image" misses, as required.
//
// The salt is computed once per TokenIterator init, stored on the iterator,
// and reused at store time. Cost: one SHA256 pass over the pixel tensor
// (~a few milliseconds for a typical 448x448 image), negligible vs vision
// tower forward (~100 ms).

import CryptoKit
import Foundation
import MLX

/// Computes a stable fingerprint for the media portion of an ``LMInput``.
///
/// Returns `nil` when the input has no image and no video — callers can then
/// fall through the cache coordinator exactly as they did for text-only
/// inputs, preserving byte-for-byte behavior on text-only paths.
///
/// The fingerprint is a lowercase hex SHA256 of:
/// - The literal UTF-8 tag `"image:"` (if an image is present) followed by
///   shape, dtype, and raw contiguous pixel bytes
/// - The literal UTF-8 tag `"video:"` (if a video is present) followed by
///   shape, dtype, and raw contiguous pixel bytes
///
/// Shape + dtype are hashed before bytes so different-shaped tensors with
/// identical bytes never collide. Ordering is deterministic: image before
/// video. The `frames: [THW]?` field is ignored because it is recoverable
/// from the pixel tensor shape.
///
/// Thread-safety: safe to call concurrently; CryptoKit's SHA256 is
/// value-typed and the underlying MLXArray reads are via `asData` which
/// takes a read-only snapshot.
public func computeMediaSalt(for input: LMInput) -> String? {
    let hasImage = input.image != nil
    let hasVideo = input.video != nil
    guard hasImage || hasVideo else { return nil }

    var hasher = SHA256()

    if let image = input.image {
        hasher.update(data: Data("image:".utf8))
        hashMLXArray(image.pixels, into: &hasher)
    }
    if let video = input.video {
        hasher.update(data: Data("video:".utf8))
        hashMLXArray(video.pixels, into: &hasher)
    }

    let digest = hasher.finalize()
    return digest.map { String(format: "%02x", $0) }.joined()
}

/// Feeds an MLXArray's shape, dtype, and raw contiguous bytes into an
/// in-progress SHA256 hasher.
///
/// Uses `asData(access: .noCopyIfContiguous)` so the common case of a
/// contiguous pixel tensor avoids any allocation. The data accessor
/// materializes any pending lazy computation before reading bytes — this is
/// safe for already-materialized pixel tensors and a necessary cost for
/// lazy ones.
private func hashMLXArray(_ array: MLXArray, into hasher: inout SHA256) {
    // Shape: fixed-count Int header followed by each dim as an Int64. Hashing
    // the dim count first makes rank-2 [3, 448*448] distinguishable from
    // rank-4 [1, 3, 448, 448] even if the byte blob would otherwise match.
    var dimCount = Int64(array.shape.count)
    withUnsafeBytes(of: &dimCount) { hasher.update(bufferPointer: $0) }
    for dim in array.shape {
        var d = Int64(dim)
        withUnsafeBytes(of: &d) { hasher.update(bufferPointer: $0) }
    }

    // Dtype: hash its string representation (stable across runs).
    hasher.update(data: Data(String(describing: array.dtype).utf8))

    // Pixel bytes: noCopy when contiguous (zero allocation on the hot path).
    let data = array.asData(access: .noCopyIfContiguous)
    hasher.update(data: data.data)
}
