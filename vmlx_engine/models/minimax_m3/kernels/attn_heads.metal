// SPDX-License-Identifier: Apache-2.0
// K2: per-head GemmaRMSNorm (q, k, index q, index k; 1 + w in fp32) + partial RoPE (MLX non-traditional, first ROT
// dims) + casts + the appends of K, V and index keys into the M3 cache lanes, one launch for R rows (1 .. 16).
//
// Host-prepended constants:
//   using kt / vt / it = <K / V / index-key lane dtypes>;   using qnt / knt / iqnt / iknt = <norm weight dtypes>
//   constexpr int H, KVH, GI = q heads, KV heads, index q heads (GI = 0: no indexer);  D = head dim (multiple of 32)
//   constexpr int ROT = rotary dims (ROT / 2 a multiple of 32, or ROT <= 32);  INPLACE = 1: write the cache lanes
//   constexpr int CNT = length of the zeroed counter output (>= 8);  constexpr float EPS, LOG2BASE (= log2(theta))
// Inputs (names): y [R, (H + 2 KVH + GI + (GI ? 1 : 0)) * D] float32 (K1 output), qn, kn, iqn, ikn [D] norm
//   weights, meta int32 [>= 8] = {offset, ...}, kc / vc / ic = the cache lanes [1, KVH | 1, capacity, D] (strided;
//   INPLACE only, else any >= 8-element arrays).
// Outputs: q [R, H, D] float32, iq [R, GI, D] float32 (normed, roped), cnt [CNT] uint32 = 0 (the counters of K4),
//   kn_out / vn_out [R, KVH, D] kt / vt and ikn_out [R, 1, D] it: the new rows (INPLACE = 0; else 1-element dummies).
// Grid: (32 * R * HT, 1, 1) rounded up to the threadgroup, threadgroup (128, 1, 1); HT = H + 2 KVH + GI + (GI ? 1 : 0):
//   one simdgroup per (row, head); lane l owns dims l, l + 32, ... of the head.
//
// Row r sits at position offset + r: RoPE angle float(offset + r) * 2^(-(i / (ROT / 2)) * LOG2BASE) with fast::cos /
// fast::sin (MLX's rope kernel); K / V / index key rows land at lane position offset + r. In-place writes are only
// requested by the host while it owns the lane buffers exclusively (cache.py: MiniMaxM3SparseCache.fused_reserve);
// every later reader of the lanes also reads an output of this launch, so MLX orders it after this one.

  constexpr int DPL = D / 32;
  constexpr int HALF = ROT / 2;
  constexpr int HT = H + 2 * KVH + GI + (GI > 0 ? 1 : 0);
  const uint lane = thread_index_in_simdgroup;
  const uint gid = thread_position_in_grid.x;
  if (gid < uint(CNT)) cnt[gid] = 0u;
  const int task = int(gid / 32u);
  const int R = int(y_shape[0]);
  if (task >= R * HT) return;
  const int r = task / HT;
  const int hi = task - r * HT;
  const int pos = meta[0] + r;

  // head kind: 0 q, 1 k, 2 v, 3 index q, 4 index k
  int kind, hh;
  if (hi < H) { kind = 0; hh = hi; }
  else if (hi < H + KVH) { kind = 1; hh = hi - H; }
  else if (hi < H + 2 * KVH) { kind = 2; hh = hi - H - KVH; }
  else if (hi < H + 2 * KVH + GI) { kind = 3; hh = hi - H - 2 * KVH; }
  else { kind = 4; hh = 0; }

  const device float* src = y + (size_t)r * (HT * D) + (size_t)hi * D;
  float v[DPL];
  for (int e = 0; e < DPL; ++e) v[e] = src[lane + 32 * e];

  if (kind != 2) {
    float ss = 0.0f;
    for (int e = 0; e < DPL; ++e) ss = fma(v[e], v[e], ss);
    ss = simd_sum(ss);
    const float inv = metal::precise::rsqrt(ss / float(D) + EPS);
    for (int e = 0; e < DPL; ++e) {
      const int d = int(lane) + 32 * e;
      float w;
      if (kind == 0) w = float(qn[d]);
      else if (kind == 1) w = float(kn[d]);
      else if (kind == 3) w = float(iqn[d]);
      else w = float(ikn[d]);
      v[e] = v[e] * inv * (1.0f + w);
    }
    // partial RoPE on dims [0, ROT): pairs (i, i + HALF), angle float(pos) * inv_freq(i)
    float rv[DPL];
    for (int e = 0; e < DPL; ++e) rv[e] = v[e];
    if (HALF % 32 == 0) {
      for (int e = 0; e < DPL; ++e) {
        const int d = int(lane) + 32 * e;
        if (d < ROT) {
          const int i = d < HALF ? d : d - HALF;
          const float inv_freq = metal::exp2(-(float(i) / float(HALF)) * LOG2BASE);
          const float theta = float(pos) * inv_freq;
          const float c = metal::fast::cos(theta);
          const float s = metal::fast::sin(theta);
          if (d < HALF) {
            const float x1 = v[e], x2 = v[e + HALF / 32];
            rv[e] = x1 * c - x2 * s;
          } else {
            const float x1 = v[e - HALF / 32], x2 = v[e];
            rv[e] = x1 * s + x2 * c;
          }
        }
      }
    } else {
      // ROT <= 32: every rotated dim is in e = 0, its partner HALF lanes away
      const uint pl = lane < uint(HALF) ? lane + uint(HALF) : lane - uint(HALF);
      const float other = simd_shuffle(v[0], ushort(pl));
      const int d = int(lane);
      if (d < ROT) {
        const int i = d < HALF ? d : d - HALF;
        const float inv_freq = metal::exp2(-(float(i) / float(HALF)) * LOG2BASE);
        const float theta = float(pos) * inv_freq;
        const float c = metal::fast::cos(theta);
        const float s = metal::fast::sin(theta);
        rv[0] = d < HALF ? v[0] * c - other * s : other * s + v[0] * c;
      }
    }
    for (int e = 0; e < DPL; ++e) v[e] = rv[e];
  }

  for (int e = 0; e < DPL; ++e) {
    const int d = int(lane) + 32 * e;
    if (kind == 0) {
      q[((size_t)r * H + hh) * D + d] = v[e];
    } else if (kind == 3) {
      iq[((size_t)r * GI + hh) * D + d] = v[e];
    } else if (kind == 1) {
      if (INPLACE) {
        device kt* dst = (device kt*)kc;
        dst[(size_t)hh * kc_strides[1] + (size_t)pos * kc_strides[2] + (size_t)d * kc_strides[3]] = kt(v[e]);
      } else {
        kn_out[((size_t)r * KVH + hh) * D + d] = kt(v[e]);
      }
    } else if (kind == 2) {
      if (INPLACE) {
        device vt* dst = (device vt*)vc;
        dst[(size_t)hh * vc_strides[1] + (size_t)pos * vc_strides[2] + (size_t)d * vc_strides[3]] = vt(v[e]);
      } else {
        vn_out[((size_t)r * KVH + hh) * D + d] = vt(v[e]);
      }
    } else {
      if (INPLACE) {
        device it* dst = (device it*)ic;
        dst[(size_t)pos * ic_strides[2] + (size_t)d * ic_strides[3]] = it(v[e]);
      } else {
        ikn_out[(size_t)r * D + d] = it(v[e]);
      }
    }
  }
