// SPDX-License-Identifier: Apache-2.0
// K4: per-(row, KV group) top-KEEP block selection + MSA attention over the selected blocks + the merge, ONE launch.
//
// Host-prepended constants:
//   using kt / vt = <K / V lane dtypes>
//   constexpr int G = KV groups (= index heads), HG = query heads per group, D = head dim (multiple of 32),
//   BLOCK = keys per block (multiple of 32), KEEP = selected blocks per row and group (top-k),
//   HS = query heads per simdgroup (HG % HS == 0), NSG = HG / HS simdgroups (NT = 32 * NSG threads)
// Inputs (names): q [R, G * HG, D] float32 (K2 output), kc / vc = the K / V lanes [1, G, capacity, D] (strided),
//   sc [R, G, NB] float32 block scores (K3 output; any >= 8-element array when no row is sparse),
//   cnt [>= R * G] uint32 zeros (K2 output), meta int32 [>= 8] = {offset, ...}, scale float32 [1] (softmax scale).
// Outputs: out [R, G * HG, D] float32; po [R, G, KEEP, HG, D], pm / pl [R, G, KEEP, HG] float32 (scratch partials).
// Grid: (NT * KEEP, R * G, 1), threadgroup (NT, 1, 1): threadgroup (j, r * G + g) handles the j-th block of row r's
//   selection for group g.
//
// Row r sits at q = offset + r and sees keys 0 .. q (n = q / BLOCK + 1 blocks). When n <= KEEP every block is
// visible: threadgroup j takes block j (none when j >= n). Otherwise the row's selection is the KEEP highest scores of
// sc[r, g, 0 .. n) (the local block is +inf), ties to the lower block id, listed in ascending block id; threadgroup j
// takes the j-th. Every threadgroup of (r, g) computes that same selection (radix select on order-preserving keys).
// Attention per threadgroup: simdgroup s owns heads s * HS .. + HS; lane = key of a 32-key chunk; online softmax in
// base 2 over the block's chunks (keys > q masked, never read), P V with lane = D / 32 dims. Partials go to po / pm /
// pl; the last threadgroup of (r, g) to finish (device-scope fence + atomic counter) merges the KEEP partials in
// ascending j (log-sum-exp) and writes out. Nothing a row computes depends on the other rows of the launch.

  constexpr int NT = 32 * NSG;
  constexpr int DPL = D / 32;
  constexpr int CPB = BLOCK / 32;
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
  threadgroup uint scan[NT];
  threadgroup bool last_tg;

  // ---------------------------------------------------------------- selection
  int blk;
  if (nbk <= KEEP) {
    blk = j < nbk ? j : -1;
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
      // exclusive scans over threads of eq counts, then of selected counts
      scan[tid] = neq;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint eq_before = 0u, eq_total = 0u;
      for (int t = 0; t < NT; ++t) {
        const uint vv = scan[t];
        eq_before += t < tid ? vv : 0u;
        eq_total += vv;
      }
      const int tie_lo = base_tie + int(eq_before);    // global tie rank of this thread's first tie
      const int take_eq = max(0, min(int(neq), remaining - tie_lo));
      const uint nsel = ngt + uint(take_eq);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scan[tid] = nsel;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint sel_before = 0u, sel_total = 0u;
      for (int t = 0; t < NT; ++t) {
        const uint vv = scan[t];
        sel_before += t < tid ? vv : 0u;
        sel_total += vv;
      }
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
    device const kt* kb = (device const kt*)kc + (size_t)g * kc_strides[1];
    device const vt* vb = (device const vt*)vc + (size_t)g * vc_strides[1];
    const size_t kss = kc_strides[2], kds = kc_strides[3], vss = vc_strides[2], vds = vc_strides[3];
    const bool kvec = kds == 1 && (kss % 4) == 0 && ((ulong)kb % (4 * sizeof(kt))) == 0;
    const bool vvec = vds == 1 && (vss % 4) == 0 && ((ulong)vb % (4 * sizeof(vt))) == 0 && DPL % 4 == 0;
    threadgroup const float* qh = qs + sg * HS * D;
    for (int c = 0; c < CPB; ++c) {
      const int start = blk * BLOCK + c * 32;
      if (start > q_pos) break;                            // uniform: later chunks are in the future too
      const int kp = start + int(lane);
      const bool valid = kp <= q_pos;
      float s[HS];
      for (int h = 0; h < HS; ++h) s[h] = 0.0f;
      if (valid) {
        device const kt* krow = kb + (size_t)kp * kss;
        if (kvec) {
          for (int d = 0; d < D; d += 4) {
            const vec<kt, 4> k4 = *((device const vec<kt, 4>*)(krow + d));
            for (int e = 0; e < 4; ++e) {
              const float kd = float(k4[e]);
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
        const float m_new = max(m_run[h], simd_max(x));
        const float factor = fast::exp2(m_run[h] - m_new);
        p[h] = valid ? fast::exp2(x - m_new) : 0.0f;
        l_run[h] = l_run[h] * factor + simd_sum(p[h]);
        m_run[h] = m_new;
        for (int e = 0; e < DPL; ++e) o[h][e] *= factor;
      }
      const int nvalid = min(32, q_pos - start + 1);
      for (int jj = 0; jj < nvalid; ++jj) {
        device const vt* vrow = vb + (size_t)(start + jj) * vss + (size_t)(lane * DPL) * vds;
        float vv[DPL];
        if (vvec) {
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
  }
  // partial (unnormalized) for heads sg * HS .. + HS of this (r, g, j)
  for (int h = 0; h < HS; ++h) {
    const size_t pidx = ((size_t)rg * KEEP + j) * HG + sg * HS + h;
    for (int e = 0; e < DPL; ++e) po[pidx * D + lane * DPL + e] = o[h][e];
    if (lane == 0) {
      pm[pidx] = m_run[h];
      pl[pidx] = l_run[h];
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
    for (int jj = 0; jj < KEEP; ++jj) mx = max(mx, ((volatile device const float*)pm)[b0 + (size_t)jj * HG]);
    float ls = 0.0f;
    for (int jj = 0; jj < KEEP; ++jj) {
      const size_t pidx = b0 + (size_t)jj * HG;
      const float w = fast::exp2(((volatile device const float*)pm)[pidx] - mx);
      wts[jj * HG + h] = w;
      ls += w * ((volatile device const float*)pl)[pidx];
    }
    wts[KEEP * HG + h] = 1.0f / ls;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tid; i < HG * D; i += NT) {
    const int h = i / D;
    const int d = i - h * D;
    float acc = 0.0f;
    for (int jj = 0; jj < KEEP; ++jj) {
      const size_t pidx = ((size_t)rg * KEEP + jj) * HG + h;
      acc += wts[jj * HG + h] * ((volatile device const float*)po)[pidx * D + d];
    }
    out[((size_t)r * H + (size_t)g * HG + h) * D + d] = acc * wts[KEEP * HG + h];
  }
