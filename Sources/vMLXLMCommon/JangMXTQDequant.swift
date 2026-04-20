// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
// JANG MXTQ (TurboQuant-packed) dequantization for mlx-swift-lm.
//
// Ported from:
//   vmlx_engine/utils/jang_loader.py
//   Lines 619-707 (`_load_jang_v2` MXTQ branch)
//
// And the Python reference primitives:
//   panel/bundled-python/.../jang_tools/turboquant/codebook.py    (compute_codebook)
//   panel/bundled-python/.../jang_tools/turboquant/rotation.py    (hadamard_inverse, generate_random_signs)
//   panel/bundled-python/.../jang_tools/turboquant/pipeline.py    (unpack_bits)
//
// ## What MXTQ is
//
// MXTQ (JANG TurboQuant) is an alternate JANG storage format where per-linear
// weights are stored not as the standard affine (uint32 packed + scales + biases)
// triplet, but as:
//
//     <base>.tq_packed   (uint32, shape [out_features, packed_cols])
//     <base>.tq_norms    (float, shape [out_features])   — per-row L2 norm
//     <base>.tq_bits     (optional)                      — bit width hint
//
// The packed tensor holds Lloyd-Max codebook *indices* (one per input feature,
// packed 32//bits values per uint32 along the last axis). Decoding a row:
//
//     1. unpack uint32 → uint8 indices of length in_features
//     2. look up indices in a Lloyd-Max codebook computed for (in_features, bits)
//     3. scale the resulting row by the stored norm
//     4. apply the inverse randomized Hadamard transform
//        (H is self-inverse up to sign; see TQHadamard.swift)
//
// The result is the original fp weight row. We then stack all out_features
// rows back into [out_features, in_features], cast to float16, and re-quantize
// into the standard affine triplet (.weight / .scales / .biases) so that every
// downstream MLX QuantizedLinear path sees a perfectly ordinary quantized layer.
// The model itself never learns MXTQ exists.
//
// ## Bit width selection
//
// The Python loader reads `mxtq_bits` from jang_config.json:
//   - any key containing `shared_expert` → mxtq_bits["shared_expert"] (default 3)
//   - any key containing `expert`        → mxtq_bits["routed_expert"] (default 2)
//   - otherwise                           → 2
//
// We mirror that exactly. `mxtq_seed` (default 42) seeds the +/-1 sign vector
// used by the randomized Hadamard — same seed must be used at decode time.
//
// ## Re-quantization target
//
// After dequant we re-quantize back to affine at the body bits/blockSize
// from JangConfig.quantization (typically 2-bit, groupSize 64). This matches
// `_q_bits = config.get("quantization", {}).get("bits", 2)` in Python.

import Foundation
import MLX

// MARK: - Public API

/// MXTQ (JANG TurboQuant-packed) dequantization + re-quantization.
///
/// Call this on a weight dictionary AFTER `loadArraysAndMetadata` and
/// BEFORE `model.sanitize(...)`. If no `.tq_packed` keys are present this
/// is a cheap no-op and returns `false`.
///
/// On success the dictionary is mutated in place: every `<base>.tq_packed` /
/// `<base>.tq_norms` pair is removed and replaced by affine-quantized
/// `<base>.weight` / `<base>.scales` / `<base>.biases` tensors sized according
/// to `jangConfig.quantization.blockSize` / its effective body bit width.
///
/// - Parameters:
///   - weights: The merged weight dictionary (mutated in place).
///   - jangConfig: Parsed JANG config (provides body block_size, seed, bit map).
///   - mxtqSeed: Seed used by `generate_random_signs` on the encode side.
///               Must match the writer. Defaults to 42 (the Python default).
///   - mxtqBits: Map with optional keys `shared_expert` / `routed_expert`.
///               If nil, defaults `shared_expert=3`, `routed_expert=2`, other=2.
///   - bodyBits: Re-quantization target bits. If nil, uses the smallest
///               entry of `jangConfig.quantization.bitWidthsUsed`, falling
///               back to 2.
/// - Returns: `true` if MXTQ tensors were found and processed.
@discardableResult
public func dequantizeJangMXTQ(
    weights: inout [String: MLXArray],
    jangConfig: JangConfig,
    mxtqSeed: Int = 42,
    mxtqBits: [String: Int]? = nil,
    bodyBits: Int? = nil
) throws -> Bool {
    // MARK: Group tq_packed + tq_norms by base path
    var groups: [String: (packed: MLXArray?, norms: MLXArray?)] = [:]
    var toRemove: [String] = []

    for (key, value) in weights {
        if key.hasSuffix(".tq_packed") {
            let base = String(key.dropLast(".tq_packed".count))
            var e = groups[base] ?? (nil, nil)
            e.packed = value
            groups[base] = e
            toRemove.append(key)
        } else if key.hasSuffix(".tq_norms") {
            let base = String(key.dropLast(".tq_norms".count))
            var e = groups[base] ?? (nil, nil)
            e.norms = value
            groups[base] = e
            toRemove.append(key)
        } else if key.hasSuffix(".tq_bits") {
            // Python ignores tq_bits (the bit width comes from mxtq_bits map).
            toRemove.append(key)
        }
    }

    guard !groups.isEmpty else { return false }

    // MARK: Derive re-quantization target
    let reQuantGroupSize = jangConfig.quantization.blockSize
    let reQuantBits: Int = bodyBits
        ?? jangConfig.quantization.bitWidthsUsed.min()
        ?? 2

    // MARK: Per-group decode
    var decoded: [(String, MLXArray, MLXArray, MLXArray?)] = []
    decoded.reserveCapacity(groups.count)

    for (base, parts) in groups {
        guard let packed = parts.packed, let norms = parts.norms else {
            // Orphan tq_packed without tq_norms (or vice versa) — mirror
            // Python `continue` rather than throw. A partial MXTQ pair is
            // unusable; leaving the base path missing will surface as a
            // downstream strict=False silent drop, which is the same
            // behavior as Python.
            continue
        }

        // Shape: packed is [out_features, packed_cols], dtype uint32.
        // in_features = packed_cols * (32 / bits).
        guard packed.shape.count == 2 else {
            throw JangLoaderError.loadFailed(
                "MXTQ tq_packed expected 2D, got shape \(packed.shape) for \(base)")
        }
        let outFeat = packed.shape[0]
        let packedCols = packed.shape[1]

        let bits = mxtqBitsFor(basePath: base, override: mxtqBits)
        // Python pack_bits/unpack_bits uses `vals_per_u32 = 32 // bits`.
        // Valid bit widths are those where valsPerU32 > 0 (i.e. bits <= 32).
        // Python supports 1,2,3,4,8 explicitly; 3/5/6 truncate the packing
        // but still round-trip as long as the same formula is used. We
        // accept 1...8 here.
        guard bits >= 1, bits <= 8 else {
            throw JangLoaderError.loadFailed(
                "MXTQ unsupported bits=\(bits) for \(base)")
        }
        let valsPerU32 = 32 / bits
        let inFeatures = packedCols * valsPerU32

        // MARK: 1. Codebook for this (dim, bits) pair.
        // Python: from jang_tools.turboquant.codebook import compute_codebook
        //         cb = mx.array(compute_codebook(in_features, bits))
        let codebook = TQCodebook.computeCodebook(dim: inFeatures, bits: bits)

        // MARK: 2. Random signs for inverse Hadamard (seeded).
        // Python: from jang_tools.turboquant.rotation import generate_random_signs
        //         signs = mx.array(generate_random_signs(in_features, _mxtq_seed))
        //
        // Python's generate_random_signs uses numpy's PCG64 via
        // `default_rng(seed).choice([-1, 1], dim)`. The Swift port of that
        // PRNG lives in `NumPyPCG64.generateRandomSigns` and produces a
        // bit-identical sign stream. The older `TQHadamard.generateRandomSigns`
        // used srand48/drand48 which decodes garbage against Python-written
        // MXTQ weights — do NOT revert to it. Fixes the MXTQ blocker
        // flagged in session 2026-04-13.
        let signs = NumPyPCG64.generateRandomSigns(dim: inFeatures, seed: mxtqSeed)

        // MARK: 3. Unpack bits → indices of shape [out_features, in_features].
        //
        // Python pipeline.unpack_bits operates on a 1D packed array and
        // returns uint8 values, then the caller reshapes per-row. We do the
        // whole 2D tensor in one shot: shift+mask then stack on a new axis
        // then reshape (out, packed_cols, valsPerU32) → (out, in_features).
        let indices = unpackBits2D(packed: packed, bits: bits, inFeatures: inFeatures)

        // MARK: 4. Codebook lookup — rotated unit-sphere row values.
        // Shape: [out_features, in_features], dtype float32.
        let rotatedUnit = TQCodebook.dequantizeScalar(indices, codebook: codebook)

        // MARK: 5. Scale rows by per-row norms.
        //   rotated = rotatedUnit * norms[:, None]
        let scaled = rotatedUnit * norms.reshaped([outFeat, 1]).asType(rotatedUnit.dtype)

        // MARK: 6. Inverse randomized Hadamard → original weight row.
        // Shape stays [out_features, in_features]; cast to float16 to match
        // Python `.astype(mx.float16)`.
        let dq = TQHadamard.hadamardInverse(scaled, signs: signs).asType(.float16)
        MLX.eval(dq)

        // MARK: 7. Re-quantize affine → (weight, scales, biases).
        let (qW, qS, qB) = MLX.quantized(
            dq, groupSize: reQuantGroupSize, bits: reQuantBits, mode: .affine)
        MLX.eval(qW)
        MLX.eval(qS)
        if let qB { MLX.eval(qB) }

        decoded.append((base, qW, qS, qB))
    }

    // MARK: Commit — strip MXTQ keys, insert affine triplets.
    for key in toRemove { weights.removeValue(forKey: key) }
    for (base, qW, qS, qB) in decoded {
        weights["\(base).weight"] = qW
        weights["\(base).scales"] = qS
        if let qB {
            weights["\(base).biases"] = qB
        }
    }
    return true
}

// MARK: - Helpers

/// Resolve the MXTQ bit width for a given weight base path.
///
/// Mirrors jang_loader.py lines 667-672:
///   if "shared_expert" in base.lower():
///       bits = _mxtq_bits_map.get("shared_expert", 3)
///   elif "expert" in base.lower():
///       bits = _mxtq_bits_map.get("routed_expert", 2)
///   else:
///       bits = 2
private func mxtqBitsFor(basePath: String, override: [String: Int]?) -> Int {
    let lower = basePath.lowercased()
    if lower.contains("shared_expert") {
        return override?["shared_expert"] ?? 3
    }
    if lower.contains("expert") {
        return override?["routed_expert"] ?? 2
    }
    return 2
}

/// Unpack a 2D uint32 tensor `[out_features, packed_cols]` into
/// `[out_features, in_features]` uint8 codebook indices.
///
/// Matches `jang_tools.turboquant.pipeline.unpack_bits` but vectorized
/// across all rows at once:
///
///     for i in 0..<valsPerU32:
///         slot_i = (packed >> (i * bits)) & mask   // shape [out, packed_cols]
///     indices = stack(slots, axis=-1)              // [out, packed_cols, valsPerU32]
///                 .reshape([out, in_features])
///
/// Python iterates `result.append(((packed >> (i*bits)) & mask).astype(uint8))`
/// then `stack(...).reshape(-1)[:n_elements]`. Because our packed tensor
/// already has exactly packed_cols * valsPerU32 == in_features elements per
/// row, no truncation is needed.
private func unpackBits2D(packed: MLXArray, bits: Int, inFeatures: Int) -> MLXArray {
    let valsPerU32 = 32 / bits
    let outFeat = packed.shape[0]
    _ = packed.shape[1]  // packedCols — not needed after stacked()
    let mask = UInt32((1 << bits) - 1)
    let maskArr = MLXArray(mask).asType(.uint32)

    // Build slot-i tensors and stack on a new last axis.
    var slots = [MLXArray]()
    slots.reserveCapacity(valsPerU32)
    let packedU32 = packed.asType(.uint32)
    for i in 0..<valsPerU32 {
        let shift = MLXArray(UInt32(i * bits)).asType(.uint32)
        let shifted = packedU32 >> shift
        let masked = shifted & maskArr
        slots.append(masked.asType(.uint8))
    }
    // stacked: [out_features, packed_cols, valsPerU32]
    let stackedSlots = stacked(slots, axis: -1)
    // flatten the last two axes: [out_features, packed_cols * valsPerU32]
    //                          = [out_features, in_features]
    return stackedSlots.reshaped([outFeat, inFeatures])
}
