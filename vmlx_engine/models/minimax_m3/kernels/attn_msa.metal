// SPDX-License-Identifier: Apache-2.0
// K4: per-(row, KV group) top-KEEP block selection + MSA attention over the selected blocks + the merge, ONE launch.
//
// Host-prepended constants:
//   using kt / vt = <K / V lane dtypes>
//   constexpr int G = KV groups (= index heads), HG = query heads per group, D = head dim (multiple of 32),
//   BLOCK = keys per block (multiple of 32), KEEP = selected blocks per row and group (top-k),
//   HS = query heads per simdgroup (HG % HS == 0), NSG = HG / HS head slices; NT = 32 * NSG * BLOCK / 32 threads
//   KQF = K / V lane format (kv_quant.metal; 0 = plain lanes of kt / vt, else packed uint8 lanes decoded in registers
//   to kt / vt values, m3q_load: only the rows of the selected blocks are read)
// Inputs (names): q [R, G * HG, D] float32 (K2 output), kc / vc = the K / V lanes [1, G, capacity, D] (strided),
//   sc [R, G, NB] float32 block scores (K3 output; any >= 8-element array when no row is sparse),
//   cnt [>= R * G] uint32 zeros (K2 output), meta int32 [>= 8] = {offset, ...}, scale float32 [1] (softmax scale).
// Outputs: out [R, G * HG, D] float32; po [R, G, KEEP, HG, D], pm / pl [R, G, KEEP, HG] float32 (scratch partials).
// Grid: (NT * KEEP, R * G, 1), threadgroup (NT, 1, 1): threadgroup (j, r * G + g) handles the j-th block of row r's
//   selection for group g. Threadgroup memory ~ 4 (HG D + (BLOCK / 64) HG D) + 1 KB (M3.1: 25 KB).
//
// Row r sits at q = offset + r and sees keys 0 .. q (n = q / BLOCK + 1 blocks). When n <= KEEP every block is
// visible: threadgroup j takes block j (none when j >= n). Otherwise the row's selection is the KEEP highest scores of
// sc[r, g, 0 .. n) (the local block is +inf), ties to the lower block id, listed in ascending block id; threadgroup j
// takes the j-th. Every threadgroup of (r, g) computes that same selection: a rank count when the row has at most NT
// blocks, else a radix select on order-preserving keys (the same set either way).
// Attention per threadgroup: one simdgroup per (32-key chunk of the block, slice of HS heads); lane = key for Q K
// (softmax in base 2, keys > q masked and never read), lane = D / 32 dims for P V; the chunks merge in a fixed tree
// in threadgroup memory. Block partials go to po / pm / pl; the last threadgroup of (r, g) to finish (device-scope
// fence + atomic counter) merges the KEEP partials in ascending j (log-sum-exp) and writes out. Nothing a row computes
// depends on the other rows of the launch.

  constexpr int CPB = BLOCK / 32;                          // 32-key chunks per block (a power of two)
  constexpr int NSGT = NSG * CPB;                          // simdgroups per threadgroup
  constexpr int NT = 32 * NSGT;
  constexpr int DPL = D / 32;
  constexpr float NEG = -1.0e30f;
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.x);
  const int rg = int(threadgroup_position_in_grid.y);
  const int r = rg / G;
  const int g = rg - r * G;
  const int H = G * HG;
  const int q_pos = meta[0] + r;
  const int nbk = q_pos / BLOCK + 1;

  threadgroup float qs[HG * D];
  threadgroup uint hist[256];
  threadgroup int sel[KEEP];
  threadgroup int shared_i[4];
  threadgroup uint scan[NSGT];
  threadgroup bool last_tg;
  threadgroup float cmo[(CPB > 1 ? CPB / 2 : 1) * NSG * HS * D];   // chunk-merge partial o, m / l
  threadgroup float cml[(CPB > 1 ? CPB / 2 : 1) * NSG * HS * 2];

  // ---------------------------------------------------------------- selection
  int blk;
  if (nbk <= KEEP) {
    blk = j < nbk ? j : -1;
  } else if (nbk <= NT) {
    // one block per thread: rank = blocks ahead of it (higher score, or equal score and lower id); the KEEP lowest
    // ranks are the selection (the radix path's set), listed in ascending block id by one prefix scan
    device const float* srow = sc + (size_t)rg * sc_shape[2];
    threadgroup uint* keys = (threadgroup uint*)qs;        // qs is staged after the selection
    const uint mine = tid < nbk ? m3a_fkey(srow[tid]) : 0u;
    if (tid < nbk) keys[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int rank = 0;
    if (tid < nbk) {
      for (int i = 0; i < nbk; ++i) {
        const uint k = keys[i];
        rank += (k > mine || (k == mine && i < tid)) ? 1 : 0;
      }
    }
    const uint flag = (tid < nbk && rank < KEEP) ? 1u : 0u;
    uint total;
    const uint at = m3a_tg_exclusive_scan<NSGT>(flag, lane, sg, scan, total);
    if (flag) sel[at] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    blk = sel[j];
  } else {
    device const float* srow = sc + (size_t)rg * sc_shape[2];
    uint prefix = 0u, pmask = 0u;
    int remaining = KEEP;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      for (int i = tid; i < 256; i += NT) hist[i] = 0u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < nbk; i += NT) {
        const uint key = m3a_fkey(srow[i]);
        if ((key & pmask) == prefix) {
          atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        // lane l covers digits 255 - 8 l .. 248 - 8 l (descending); find the digit holding the remaining-th key
        uint c[8];
        uint tot = 0u;
        for (int e = 0; e < 8; ++e) {
          c[e] = hist[255 - (8 * int(lane) + e)];
          tot += c[e];
        }
        uint before = simd_prefix_exclusive_sum(tot);
        for (int e = 0; e < 8; ++e) {
          if (before < uint(remaining) && uint(remaining) <= before + c[e]) {
            shared_i[0] = 255 - (8 * int(lane) + e);
            shared_i[1] = remaining - int(before);
          }
          before += c[e];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      prefix |= uint(shared_i[0]) << shift;
      pmask |= 255u << shift;
      remaining = shared_i[1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // selected: key > prefix, plus the `remaining` lowest-id keys == prefix; list them in ascending block id
    constexpr int MAXCH = 64;                             // ids per thread per sweep
    int base_out = 0, base_tie = 0;
    for (int i0 = 0; i0 < nbk; i0 += NT * MAXCH) {
      const int lo = i0 + tid * MAXCH;
      const int hi = min(lo + MAXCH, nbk);
      uint ngt = 0u, neq = 0u;
      for (int i = lo; i < hi; ++i) {
        const uint key = m3a_fkey(srow[i]);
        ngt += key > prefix ? 1u : 0u;
        neq += key == prefix ? 1u : 0u;
      }
      // exclusive scans over threads (in thread order) of the tie counts, then of the selected counts
      uint eq_total;
      const uint eq_before = m3a_tg_exclusive_scan<NSGT>(neq, lane, sg, scan, eq_total);
      const int tie_lo = base_tie + int(eq_before);    // global tie rank of this thread's first tie
      const int take_eq = max(0, min(int(neq), remaining - tie_lo));
      const uint nsel = ngt + uint(take_eq);
      uint sel_total;
      const uint sel_before = m3a_tg_exclusive_scan<NSGT>(nsel, lane, sg, scan, sel_total);
      int at = base_out + int(sel_before);
      int tie = tie_lo;
      for (int i = lo; i < hi; ++i) {
        const uint key = m3a_fkey(srow[i]);
        bool take = key > prefix;
        if (key == prefix) {
          take = tie < remaining;
          ++tie;
        }
        if (take && at < KEEP) sel[at++] = i;
      }
      base_out += int(sel_total);
      base_tie += int(eq_total);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    blk = sel[j];
  }

  // ---------------------------------------------------------------- attention over block blk
  // simdgroup sg = (chunk c, head slice hs): the 32 keys blk * BLOCK + 32 c + lane against heads hs * HS .. + HS
  const int hs = sg % NSG;
  const int c = sg / NSG;
  const float qscale = scale[0] * 1.4426950408889634f;
  device const float* qg = q + ((size_t)r * H + (size_t)g * HG) * D;
  for (int i = tid; i < HG * D; i += NT) qs[i] = qg[i] * qscale;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_run[HS], l_run[HS], o[HS][DPL];
  for (int h = 0; h < HS; ++h) {
    m_run[h] = NEG;
    l_run[h] = 0.0f;
    for (int e = 0; e < DPL; ++e) o[h][e] = 0.0f;
  }
  if (blk >= 0) {
    const int start = blk * BLOCK + c * 32;
    if (start <= q_pos) {                                  // a chunk past the row's position stays empty
      device const kt* kb = (device const kt*)kc + (size_t)g * kc_strides[1];
      device const vt* vb = (device const vt*)vc + (size_t)g * vc_strides[1];
      device const uint8_t* kbq = (device const uint8_t*)kc + (size_t)g * kc_strides[1];   // packed lanes (KQF != 0)
      device const uint8_t* vbq = (device const uint8_t*)vc + (size_t)g * vc_strides[1];
      const size_t kss = kc_strides[2], kds = kc_strides[3], vss = vc_strides[2], vds = vc_strides[3];
      const bool kvec = kds == 1 && (kss % 8) == 0 && ((ulong)kb % (8 * sizeof(kt))) == 0;
      const bool vvec = vds == 1 && (vss % 4) == 0 && ((ulong)vb % (4 * sizeof(vt))) == 0 && DPL % 4 == 0;
      threadgroup const float* qh = qs + hs * HS * D;
      const int kp = start + int(lane);
      const bool valid = kp <= q_pos;
      float s[HS];
      for (int h = 0; h < HS; ++h) s[h] = 0.0f;
      if (valid) {
        device const kt* krow = kb + (size_t)kp * kss;
        if constexpr (KQF != 0) {
          device const uint8_t* kr = kbq + (size_t)kp * kss;
          for (int d = 0; d < D; d += 8) {
            float k8[8];
            m3q_load<KQF, D, kt, 8>(kr, d, k8);
            for (int e = 0; e < 8; ++e) {
              for (int h = 0; h < HS; ++h) s[h] = fma(qh[h * D + d + e], k8[e], s[h]);
            }
          }
        } else if (kvec) {
          for (int d = 0; d < D; d += 8) {
            const vec<kt, 8> k8 = *((device const vec<kt, 8>*)(krow + d));
            for (int e = 0; e < 8; ++e) {
              const float kd = float(k8[e]);
              for (int h = 0; h < HS; ++h) s[h] = fma(qh[h * D + d + e], kd, s[h]);
            }
          }
        } else {
          for (int d = 0; d < D; ++d) {
            const float kd = float(krow[(size_t)d * kds]);
            for (int h = 0; h < HS; ++h) s[h] = fma(qh[h * D + d], kd, s[h]);
          }
        }
      }
      float p[HS];
      for (int h = 0; h < HS; ++h) {
        const float x = valid ? s[h] : NEG;
        m_run[h] = simd_max(x);
        p[h] = valid ? fast::exp2(x - m_run[h]) : 0.0f;
        l_run[h] = simd_sum(p[h]);
      }
      // all 32 keys, unrolled so the V loads overlap: a key past q_pos has p = 0 and reads the row at q_pos instead
      // (finite, contributes nothing), so nothing past the row's extent is read
      #pragma unroll(8)
      for (int jj = 0; jj < 32; ++jj) {
        device const vt* vrow = vb + (size_t)min(start + jj, q_pos) * vss + (size_t)(lane * DPL) * vds;
        float vv[DPL];
        if constexpr (KQF != 0) {
          m3q_load<KQF, D, vt, DPL>(vbq + (size_t)min(start + jj, q_pos) * vss, int(lane) * DPL, vv);
        } else if (vvec) {
          for (int e = 0; e < DPL; e += 4) {
            const vec<vt, 4> v4 = *((device const vec<vt, 4>*)(vrow + e));
            vv[e] = float(v4[0]); vv[e + 1] = float(v4[1]); vv[e + 2] = float(v4[2]); vv[e + 3] = float(v4[3]);
          }
        } else {
          for (int e = 0; e < DPL; ++e) vv[e] = float(vrow[(size_t)e * vds]);
        }
        for (int h = 0; h < HS; ++h) {
          const float pj = simd_shuffle(p[h], ushort(jj));
          for (int e = 0; e < DPL; ++e) o[h][e] = fma(pj, vv[e], o[h][e]);
        }
      }
    }
    // the block's chunks, merged in a fixed tree: at step t, chunk c + t folds into chunk c (c < t)
    for (int step = CPB / 2; step >= 1; step /= 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c >= step && c < 2 * step) {
        for (int h = 0; h < HS; ++h) {
          const int slot = ((c - step) * NSG + hs) * HS + h;
          for (int e = 0; e < DPL; ++e) cmo[slot * D + lane * DPL + e] = o[h][e];
          if (lane == 0) {
            cml[2 * slot] = m_run[h];
            cml[2 * slot + 1] = l_run[h];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c < step) {
        for (int h = 0; h < HS; ++h) {
          const int slot = (c * NSG + hs) * HS + h;
          const float mb = cml[2 * slot], lb = cml[2 * slot + 1];
          const float m = max(m_run[h], mb);
          const float fa = fast::exp2(m_run[h] - m), fb = fast::exp2(mb - m);
          l_run[h] = l_run[h] * fa + lb * fb;
          m_run[h] = m;
          for (int e = 0; e < DPL; ++e) o[h][e] = o[h][e] * fa + cmo[slot * D + lane * DPL + e] * fb;
        }
      }
    }
  }
  // partial (unnormalized) of this (r, g, j): chunk 0's simdgroups hold the merged block
  if (c == 0) {
    for (int h = 0; h < HS; ++h) {
      const size_t pidx = ((size_t)rg * KEEP + j) * HG + hs * HS + h;
      for (int e = 0; e < DPL; ++e) po[pidx * D + lane * DPL + e] = o[h][e];
      if (lane == 0) {
        pm[pidx] = m_run[h];
        pl[pidx] = l_run[h];
      }
    }
  }

  // ---------------------------------------------------------------- merge: the last threadgroup of (r, g)
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    const uint prev = atomic_fetch_add_explicit((device atomic_uint*)cnt + rg, 1u, memory_order_relaxed);
    last_tg = prev == uint(KEEP - 1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last_tg) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  // weights w[jj][h] = 2^(pm - max) and 1 / l per head, in threadgroup memory (reuses qs)
  threadgroup float* wts = qs;                             // KEEP * HG weights, then HG inverse sums
  for (int h = tid; h < HG; h += NT) {
    const size_t b0 = (size_t)rg * KEEP * HG + h;
    float mx = NEG;
    for (int jj = 0; jj < KEEP; ++jj) mx = max(mx, pm[b0 + (size_t)jj * HG]);
    float ls = 0.0f;
    for (int jj = 0; jj < KEEP; ++jj) {
      const size_t pidx = b0 + (size_t)jj * HG;
      const float w = fast::exp2(pm[pidx] - mx);
      wts[jj * HG + h] = w;
      ls += w * pl[pidx];
    }
    wts[KEEP * HG + h] = 1.0f / ls;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // four consecutive dims per thread and step (float4), the KEEP partials of each in ascending j
  constexpr int NV4 = HG * D / 4;
  #pragma unroll
  for (int i0 = 0; i0 < NV4; i0 += NT) {
    const int i = i0 + tid;
    if (i < NV4) {
      const int h = (4 * i) / D;
      const int d = 4 * i - h * D;
      float4 acc = float4(0.0f);
      #pragma unroll
      for (int jj = 0; jj < KEEP; ++jj) {
        const size_t pidx = ((size_t)rg * KEEP + jj) * HG + h;
        acc += wts[jj * HG + h] * *((device const float4*)(po + pidx * D + d));
      }
      *((device float4*)(out + ((size_t)r * H + (size_t)g * HG + h) * D + d)) = acc * wts[KEEP * HG + h];
    }
  }
