// Copyright © 2026 Jinho Jang. All rights reserved.
//
// NumPy-compatible PCG64 PRNG. Required so Swift MXTQ decode produces the
// same ±1 sign sequence as the Python encoder, which uses
//
//     numpy.random.default_rng(seed).choice([-1, 1], dim)
//
// Without a bit-identical port, MXTQ-packed weights decode to garbage.
//
// This is a full, verified re-implementation of:
//
//   1. NumPy `SeedSequence` (`numpy/random/_bit_generator.pyx`)
//      — mix_entropy + generate_state (uint32 pool expansion).
//
//   2. NumPy `PCG64` bit generator (`numpy/random/src/pcg64/pcg64.h`)
//      — 128-bit LCG + XSL-RR output function, including the
//      `set_seed` two-step initialization that NumPy performs
//      after pulling state+inc from the SeedSequence pool.
//
//   3. NumPy `Generator.choice([-1, 1], dim)` which reduces to
//      `integers(0, 2, dtype=int64)`. For range=2 that uses the
//      buffered uint32 Lemire path: each PCG64 uint64 output is
//      split into two uint32 (low half first), and each uint32 `u`
//      produces a bit equal to `u >> 31` (equivalently `(u * 2) >> 32`).
//      Bit 0 → `-1`, bit 1 → `+1`.
//
// This file was bit-verified against CPython+NumPy 2.x for seeds
// `{0, 1, 42, 2048, 2^32-1}` × 8192 elements: zero mismatches.
// An earlier version of this file used a simplified hash_mix and
// XSL-RR of already-advanced state with a fixed increment — it was
// wrong in at least three ways (SeedSequence algorithm, PCG init
// two-step dance, and using doubles instead of uint32 top-bits for
// `choice`). All fixed here.
//
// If you need JUMP/ADVANCE for multi-stream RNGs, port PCG's
// `advance_lcg_128` separately. MXTQ only consumes the single
// primary stream via `generateRandomSigns`.

import Foundation
import MLX

/// NumPy-compatible PCG64 pseudo-random number generator.
///
/// Matches `numpy.random.default_rng(seed)` (which internally is
/// `Generator(PCG64(SeedSequence(seed)))`) for the single-stream
/// integer-draw path used by `choice([-1, 1], N)`.
public struct NumPyPCG64: Sendable {
    /// 128-bit state, represented as two UInt64 halves (high, low).
    public var stateHigh: UInt64
    public var stateLow: UInt64
    /// 128-bit increment (odd), two UInt64 halves (high, low).
    public let incHigh: UInt64
    public let incLow: UInt64

    /// PCG64 multiplier 0x2360ED051FC65DA44385DF649FCCF645 split (high, low).
    @usableFromInline static let multHigh: UInt64 = 0x2360ED051FC65DA4
    @usableFromInline static let multLow: UInt64 = 0x4385DF649FCCF645

    /// Initialize exactly as `numpy.random.PCG64(SeedSequence(seed))`.
    public init(seed: UInt64) {
        // Step 1. SeedSequence entropy pool.
        let entropy = Self.seedToUInt32Array(seed: seed)
        let pool = Self.mixEntropy(entropy: entropy)

        // Step 2. generate_state(8 uint32) → interpret as 4 uint64
        //         (little-endian pair per uint64). NumPy's PCG64 C
        //         code then splits into: state = (s[0]<<64)|s[1],
        //         inc = (s[2]<<64)|s[3].
        let w = Self.generateStateU32(pool: pool, n: 8)
        let s0 = (UInt64(w[1]) << 32) | UInt64(w[0])
        let s1 = (UInt64(w[3]) << 32) | UInt64(w[2])
        let s2 = (UInt64(w[5]) << 32) | UInt64(w[4])
        let s3 = (UInt64(w[7]) << 32) | UInt64(w[6])

        // seed_state = (s0, s1) as (high, low) of a 128-bit int
        // seed_inc   = (s2, s3) as (high, low)
        let seedStateHigh = s0
        let seedStateLow = s1
        let seedIncHigh = s2
        let seedIncLow = s3

        // Step 3. `pcg_setseq_128_srandom_r`:
        //   rng.inc   = (seed_inc << 1) | 1
        //   rng.state = 0
        //   rng.state = rng.state * MULT + rng.inc          (step)
        //   rng.state = rng.state + seed_state
        //   rng.state = rng.state * MULT + rng.inc          (step)
        //
        // The shift-by-1 is a 128-bit shift:
        //   newHigh = (oldHigh << 1) | (oldLow >> 63)
        //   newLow  = (oldLow  << 1) | 1        (force odd)
        let ih = (seedIncHigh << 1) | (seedIncLow >> 63)
        let il = (seedIncLow << 1) | 1
        self.incHigh = ih
        self.incLow = il

        var sh: UInt64 = 0
        var sl: UInt64 = 0
        Self.lcgStep(stateHigh: &sh, stateLow: &sl, incHigh: ih, incLow: il)
        // Add seed_state (128-bit add)
        let (addLow, carry1) = sl.addingReportingOverflow(seedStateLow)
        sl = addLow
        sh = sh &+ seedStateHigh &+ (carry1 ? 1 : 0)
        Self.lcgStep(stateHigh: &sh, stateLow: &sl, incHigh: ih, incLow: il)

        self.stateHigh = sh
        self.stateLow = sl
    }

    /// One PCG64 LCG step: state = state * MULT + inc  (mod 2^128).
    @inlinable
    static func lcgStep(
        stateHigh: inout UInt64, stateLow: inout UInt64,
        incHigh: UInt64, incLow: UInt64
    ) {
        // 128-bit multiply: (sh, sl) * (mh, ml) → low 128 bits.
        //
        //   new_low  = sl*ml        (full 128-bit; we keep low 64 + carry)
        //   new_high = sl*mh + sh*ml + (carry from sl*ml)
        //
        // We need the full 128-bit product of sl*ml for the carry.
        let sl = stateLow
        let sh = stateHigh
        let ml = multLow
        let mh = multHigh

        // Full 128-bit product of sl * ml using split 32-bit halves.
        let slLo = sl & 0xFFFFFFFF
        let slHi = sl >> 32
        let mlLo = ml & 0xFFFFFFFF
        let mlHi = ml >> 32
        let p00 = slLo &* mlLo
        let p01 = slLo &* mlHi
        let p10 = slHi &* mlLo
        let p11 = slHi &* mlHi
        let mid = (p00 >> 32) &+ (p01 & 0xFFFFFFFF) &+ (p10 & 0xFFFFFFFF)
        let lowLow = (p00 & 0xFFFFFFFF) | (mid << 32)
        let lowHigh = p11 &+ (p01 >> 32) &+ (p10 >> 32) &+ (mid >> 32)

        // Cross terms (truncated to 64 bits for the high half of state).
        let crossHigh = sl &* mh &+ sh &* ml

        // Low 64 of new state + inc low → with carry propagation.
        let (newLowA, c1) = lowLow.addingReportingOverflow(incLow)
        let newLow = newLowA

        // High 64 = lowHigh + crossHigh + incHigh + carry
        var newHigh = lowHigh &+ crossHigh &+ incHigh
        if c1 { newHigh = newHigh &+ 1 }

        stateLow = newLow
        stateHigh = newHigh
    }

    /// XSL-RR 64-bit output function from the *current* (post-step)
    /// 128-bit state.
    @inlinable
    static func xslRR(stateHigh: UInt64, stateLow: UInt64) -> UInt64 {
        let xored = stateHigh ^ stateLow
        let rot = UInt32(stateHigh >> 58)
        return (xored >> rot) | (xored << ((64 &- rot) & 63))
    }

    /// Draw the next 64-bit output.
    public mutating func nextUInt64() -> UInt64 {
        Self.lcgStep(
            stateHigh: &stateHigh, stateLow: &stateLow,
            incHigh: incHigh, incLow: incLow)
        return Self.xslRR(stateHigh: stateHigh, stateLow: stateLow)
    }

    /// Produce a ±1 MLXArray of the given length, matching
    /// `numpy.random.default_rng(seed).choice([-1, 1], dim)`.
    ///
    /// NumPy's `choice` with a length-2 equal-probability array reduces
    /// to `integers(0, 2, dtype=int64)`. For range 2, that uses the
    /// buffered uint32 path: each uint64 provides two uint32s (low half
    /// first, then high half), and each uint32 `u` yields a bit equal
    /// to `(u * 2) >> 32`, i.e. the top bit. 0 → `-1`, 1 → `+1`.
    public static func generateRandomSigns(dim: Int, seed: Int) -> MLXArray {
        MLXArray(generateRandomSignsFloat(dim: dim, seed: seed))
    }

    /// Core sign generator returning a host `[Float]`. Factored out so
    /// unit tests can run without touching Metal / MLXArray.
    public static func generateRandomSignsFloat(dim: Int, seed: Int) -> [Float] {
        var rng = NumPyPCG64(seed: UInt64(bitPattern: Int64(seed)))
        var signs = [Float](repeating: 0, count: dim)
        var haveLow = false
        var lowHalf: UInt32 = 0
        for i in 0..<dim {
            let u32: UInt32
            if haveLow {
                u32 = lowHalf
                haveLow = false
            } else {
                let u64 = rng.nextUInt64()
                u32 = UInt32(u64 & 0xFFFFFFFF)
                lowHalf = UInt32(u64 >> 32)
                haveLow = true
            }
            // Lemire range=2: bit = (u32 * 2) >> 32 = u32 >> 31.
            let bit = u32 >> 31
            signs[i] = (bit == 0) ? -1.0 : 1.0
        }
        return signs
    }

    // MARK: - NumPy SeedSequence (verified port)

    private static let initA: UInt32 = 0x43b0d7e5
    private static let multA: UInt32 = 0x931e8875
    private static let initB: UInt32 = 0x8b51f9dd
    private static let multB: UInt32 = 0x58f38ded
    private static let mixMultL: UInt32 = 0xca01f9dd
    private static let mixMultR: UInt32 = 0x4973f715
    private static let xshift: UInt32 = 16

    /// `int_to_uint32_array(n)` — little-endian base-2^32 digits,
    /// minimum one element.
    static func seedToUInt32Array(seed: UInt64) -> [UInt32] {
        if seed == 0 { return [0] }
        var out: [UInt32] = []
        var n = seed
        while n > 0 {
            out.append(UInt32(n & 0xFFFFFFFF))
            n >>= 32
        }
        return out
    }

    /// Stateful `hashmix` used inside `mix_entropy` / `generate_state`.
    /// The `hashConst` parameter is `inout` because each call advances it.
    @inline(__always)
    private static func hashmix(_ value: UInt32, _ hashConst: inout UInt32) -> UInt32 {
        var v = value ^ hashConst
        hashConst = hashConst &* multA
        v = v &* hashConst
        v ^= v >> xshift
        return v
    }

    /// Symmetric `mix(x, y)` used to fold pool entries together.
    @inline(__always)
    private static func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
        var r = (mixMultL &* x) &- (mixMultR &* y)
        r ^= r >> xshift
        return r
    }

    /// Full NumPy `SeedSequence.mix_entropy(pool_size=4, entropy)`.
    static func mixEntropy(entropy: [UInt32]) -> [UInt32] {
        var hashConst: UInt32 = initA
        var pool = [UInt32](repeating: 0, count: 4)
        for i in 0..<4 {
            let v: UInt32 = i < entropy.count ? entropy[i] : 0
            pool[i] = hashmix(v, &hashConst)
        }
        // Mix any remaining entropy words into every pool slot.
        if entropy.count > 4 {
            for iSrc in 4..<entropy.count {
                let mixedSrc = hashmix(entropy[iSrc], &hashConst)
                for iDst in 0..<4 {
                    pool[iDst] = mix(pool[iDst], mixedSrc)
                }
            }
        }
        // Cross-mix the pool so every slot depends on every other.
        for iSrc in 0..<4 {
            for iDst in 0..<4 where iDst != iSrc {
                let mixedSrc = hashmix(pool[iSrc], &hashConst)
                pool[iDst] = mix(pool[iDst], mixedSrc)
            }
        }
        return pool
    }

    /// NumPy `SeedSequence.generate_state(n, dtype=uint32)`.
    static func generateStateU32(pool: [UInt32], n: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: n)
        var hashConst: UInt32 = initB
        for i in 0..<n {
            var dataVal = pool[i % pool.count]
            dataVal ^= hashConst
            hashConst = hashConst &* multB
            dataVal = dataVal &* hashConst
            dataVal ^= dataVal >> xshift
            out[i] = dataVal
        }
        return out
    }
}
