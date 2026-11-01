// SPDX-License-Identifier: Apache-2.0
// K3: Lightning-indexer block scores for every (row, index head) pair of R rows, no [rows, keys] score sheet.
//
// Host-prepended constants:
//   using it = <index-key lane dtype>;  constexpr int GI = index heads (= KV groups), D = index head dim,
//   BLOCK = keys per block (multiple of 32, <= 1024), LOCAL = forced local blocks, PT = pairs per pass (4 / 8 / 16: the host's choice)
// Inputs (names): iq [R, GI, D] float32 (K2 output), ic = index-key lane [1, 1, capacity, D] (strided),
//   meta int32 [>= 8] = {offset, R, NB, ...}.
// Output: sc [R, GI, NB] float32, NB = ceil((offset + R) / BLOCK) (meta[2]).
// Grid: (BLOCK * NB, 1, 1), threadgroup (BLOCK, 1, 1): one threadgroup per key block, thread t = key block * BLOCK + t.
//
// Pair p = r * GI + g sits at position q = offset + r and sees keys 0 .. q. Block score = max over the block's keys
// <= q of iq[p] . k (fp32, one fma per dim in ascending order), -inf when the block has no such key, +inf for the
// pair's LOCAL block(s) ending at block(q). A pair's scores depend on its own query and keys only.

  const int tid = int(thread_position_in_threadgroup.x);
  const int n = int(threadgroup_position_in_grid.x);
  const int R = int(iq_shape[0]);
  const int P = R * GI;
  const int offset = meta[0];
  const int NB = meta[2];
  constexpr int NSG = BLOCK / 32;
  constexpr int DV = D / 8;
  threadgroup float qs[PT * D];
  threadgroup float red[PT * NSG];

  const int kp = n * BLOCK + tid;
  const int last = offset + R - 1;                       // last position any pair sees
  const bool live = kp <= last;
  device const it* krow = (device const it*)ic + (size_t)kp * ic_strides[2];
  const size_t kds = ic_strides[3];
  const bool vec_ok = kds == 1 && ((ulong)krow % (8 * sizeof(it))) == 0;

  for (int p0 = 0; p0 < P; p0 += PT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < PT * D; i += BLOCK) {
      const int p = p0 + i / D;
      qs[i] = p < P ? iq[(size_t)p * D + (i % D)] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[PT];
    for (int pp = 0; pp < PT; ++pp) s[pp] = 0.0f;
    if (live) {
      for (int c = 0; c < DV; ++c) {
        float kv[8];
        if (vec_ok) {
          const vec<it, 8> k8 = *((device const vec<it, 8>*)(krow + c * 8));
          for (int e = 0; e < 8; ++e) kv[e] = float(k8[e]);
        } else {
          for (int e = 0; e < 8; ++e) kv[e] = float(krow[(size_t)(c * 8 + e) * kds]);
        }
        for (int pp = 0; pp < PT; ++pp) {
          for (int e = 0; e < 8; ++e) s[pp] = fma(qs[pp * D + c * 8 + e], kv[e], s[pp]);
        }
      }
    }
    const int sg = tid / 32;
    for (int pp = 0; pp < PT; ++pp) {
      const int p = p0 + pp;
      const int qpos = offset + (p < P ? p / GI : 0);
      const float x = (p < P && kp <= qpos) ? s[pp] : -INFINITY;
      const float m = simd_max(x);
      if ((tid & 31) == 0) red[pp * NSG + sg] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < PT && p0 + tid < P) {
      const int p = p0 + tid;
      float m = -INFINITY;
      for (int i = 0; i < NSG; ++i) m = max(m, red[tid * NSG + i]);
      const int qb = (offset + p / GI) / BLOCK;
      const bool forced = n <= qb && n >= max(0, qb - LOCAL + 1);
      sc[(size_t)p * NB + n] = forced ? INFINITY : m;
    }
  }
