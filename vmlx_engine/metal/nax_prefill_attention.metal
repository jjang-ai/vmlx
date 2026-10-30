// SPDX-License-Identifier: Apache-2.0
// Prefill attention on the M5 tensor units for heads whose value width differs from the query/key width
// (MiMo-V2: q/k 192, v 128). Built on MLX's own NAX fragment and tile types (mlx::steel::NAXTile / BaseNAXFrag),
// whose header text nax_attention_bdv.py reads from the installed MLX package; the per-block arithmetic follows
// MLX's attention_nax kernel (steel_attention_nax.h). Our additions: a separate value head width, key-range
// passes that carry the online-softmax row state between dispatches, an optional head-dim split over simdgroup
// pairs, and a causal sliding-window band.
namespace vmlx_attn {

using namespace mlx::steel;

// Geometry block (int32), filled by nax_attention_bdv.py.
enum GeomSlot : int {
  kBatch = 0,
  kHeads,
  kQueryLen,
  kKeyLen,
  kGqa,
  kQueryTiles,     // ceil(qL / BQ)
  kKeyBlocks,      // ceil(kL / 32)
  kFullQueryTiles, // qL / BQ
  kFullKeyBlocks,  // kL / 32
  kQueryTail,      // qL - kFullQueryTiles * BQ
  kKeyTail,        // kL - kFullKeyBlocks * 32
  kQueryOffset,    // kL - qL: bottom-right alignment of query positions
  kBlockBegin,     // first key block of this dispatch
  kBlockEnd,       // one past its last key block
  kGeomSlots
};

struct RowAdd {
  template <typename U>
  METAL_FUNC static constexpr U apply(U a, U b) {
    return a + b;
  }
};
struct RowScale {
  template <typename U>
  METAL_FUNC static constexpr U apply(U a, U b) {
    return a * b;
  }
};
struct RowExp2Shift {
  template <typename U>
  METAL_FUNC static constexpr U apply(U a, U b) {
    return fast::exp2(a - b);
  }
};

// One threadgroup = BQ query rows of one (batch, head); WM row groups of 16 rows, each served by WN simdgroups
// that split the head dims (WN = 2: half of q/k for the scores, half of v for the output; partial scores are
// summed through `exchange`). Key blocks of 32.
template <
    typename T,
    int BQ,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool ALIGN_Q,
    bool ALIGN_K,
    bool HAS_MASK,
    bool CAUSAL,
    int WINDOW,
    bool HAS_SINKS,
    bool FIRST,
    bool LAST,
    typename Strides,
    typename GeomPtr,
    typename MaskPtr,
    typename SinkPtr,
    typename CarryPtr>
METAL_FUNC void split_value_attention(
    const device T* q,
    const device T* k,
    const device T* v,
    device T* out,
    GeomPtr geom,
    const float scale,
    Strides q_str,
    Strides k_str,
    Strides v_str,
    Strides m_str,
    MaskPtr mask,
    SinkPtr sinks,
    CarryPtr carry_in,
    device float* carry_out,
    threadgroup float* exchange,
    const uint sg,
    const uint lane,
    const uint3 tg) {
  constexpr short F = 16;            // fragment edge
  constexpr int BK = 32;             // keys per block
  constexpr short KF = BK / F;       // key fragments per block
  constexpr short QKF = BD / F / WN; // q/k head-dim fragments held by this simdgroup
  constexpr short VF = BDV / F / WN; // value fragments held by this simdgroup
  constexpr int CARRY = BDV + 2;     // carried floats per query row: O, running max, running sum
  static_assert(BQ == WM * F, "one 16-row fragment per row group");
  static_assert(WN == 1 || WN == 2, "head dims split over one or two simdgroups");
  static_assert(QKF * F * WN == BD && VF * F * WN == BDV, "head dims split evenly");
  static_assert(VF % 2 == 0 && KF % 2 == 0, "fragments are multiplied in pairs");

  using acc_t = NAXTile<float, 1, VF>;
  using score_t = NAXTile<float, 1, KF>;
  constexpr short kRows = acc_t::kRowsPerThread;

  const int heads = geom[kHeads];
  const int q_len = geom[kQueryLen];
  const int q_tile = int(tg.x);
  const int head = int(tg.y);
  const int batch = int(tg.z);
  const int group = int(sg) / WN;
  const int part = int(sg) % WN;
  const int group_row = group * F; // first row of this row group within the tile
  const int kv_head = head / geom[kGqa];

  const int q_ld = int(q_str[2]);
  const int k_ld = int(k_str[2]);
  const int v_ld = int(v_str[2]);
  const int out_ld = heads * BDV; // output rows are [B, qL, H, BDV]

  q += int64_t(batch) * q_str[0] + int64_t(head) * q_str[1] + int64_t(q_tile * BQ + group_row) * q_str[2] +
      part * (BD / WN);
  k += int64_t(batch) * k_str[0] + int64_t(kv_head) * k_str[1] + part * (BD / WN);
  v += int64_t(batch) * v_str[0] + int64_t(kv_head) * v_str[1] + part * (BDV / WN);
  out += (int64_t(batch) * q_len + q_tile * BQ + group_row) * out_ld + head * BDV + part * (BDV / WN);
  if (HAS_MASK) {
    mask += int64_t(batch) * m_str[0] + int64_t(head) * m_str[1];
  }

  const float scale_log2 = scale * 1.44269504089f; // scores in the exp2 domain
  const short2 coord = acc_t::NAXFrag_t::get_coord();
  const short lane_row = coord.y;
  const short lane_col = coord.x;

  // Online-softmax state of this thread's rows.
  acc_t acc;
  metal::vec<float, kRows> run_max;
  metal::vec<float, kRows> run_sum{0};
  const int64_t carry_row0 =
      (int64_t(batch) * heads + head) * (int64_t(geom[kQueryTiles]) * BQ) + q_tile * BQ + group_row;
  if constexpr (FIRST) {
    acc.clear();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; ++i) {
      run_max[i] = Limits<float>::finite_min;
    }
    if (HAS_SINKS) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRows; ++i) {
        run_max[i] = M_LOG2E_F * static_cast<float>(sinks[head]);
        run_sum[i] = 1;
      }
    }
  } else {
    acc.load(carry_in + carry_row0 * CARRY + part * (BDV / WN), CARRY);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; ++i) {
      const int64_t r = carry_row0 + lane_row + i * acc_t::kFragRowsJump;
      run_max[i] = carry_in[r * CARRY + BDV];
      run_sum[i] = carry_in[r * CARRY + BDV + 1];
    }
  }

  // Key blocks this tile visits: causal end, window start, and this dispatch's slice.
  const int q_offset = geom[kQueryOffset];
  const int tile_first_pos = q_tile * BQ + q_offset;
  int block_stop = geom[kKeyBlocks];
  int first_causal_block = geom[kKeyBlocks];
  if (CAUSAL) {
    block_stop = min(geom[kKeyBlocks], (tile_first_pos + BQ + BK - 1) / BK);
    first_causal_block = max(0, tile_first_pos) / BK;
  }
  const int first_window_block = WINDOW > 0 ? max(0, tile_first_pos - WINDOW + 1) / BK : 0;
  const int kb_lo = max(geom[kBlockBegin], first_window_block);
  const int kb_hi = min(block_stop, geom[kBlockEnd]);

  const bool tail_tile = !ALIGN_Q && q_tile == geom[kFullQueryTiles];
  const short tail_rows = short(geom[kQueryTail] - group_row);
  const short key_tail = short(geom[kKeyTail]);

  NAXTile<T, 1, QKF> q_frags; // loaded once, reused for every key block
  if (tail_tile) {
    q_frags.load_rows(q, q_ld, tail_rows);
  } else {
    q_frags.load(q, q_ld);
  }

  for (int kb = kb_lo; kb < kb_hi; kb++) {
    const bool tail_block = !ALIGN_K && kb == geom[kFullKeyBlocks];
    const device T* kblk = k + int64_t(kb) * BK * k_str[2];
    const device T* vblk = v + int64_t(kb) * BK * v_str[2];

    // Scores of this block: two 16-key fragments per product, head-dim fragments in order.
    score_t scores;
    scores.clear();
    STEEL_PRAGMA_UNROLL
    for (short kf = 0; kf < KF; kf += 2) {
      auto qk_step = [&](short d) {
        NAXTile<T, 2, 1> kt;
        const device T* src = kblk + kf * F * k_ld + d * F;
        if (tail_block) {
          kt.load_rows(src, k_ld, short(key_tail - kf * F));
        } else {
          kt.load(src, k_ld);
        }
        score_t::NAXFrag_t::mma(
            scores.frag_at(0, kf),
            scores.frag_at(0, kf + 1),
            q_frags.frag_at(0, d),
            metal::false_type{},
            kt.frag_at(0, 0),
            kt.frag_at(1, 0),
            metal::true_type{});
      };
      if constexpr (WN == 2) {
        // Split path: fully unrolled, so every fragment index is static.
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < QKF; d++) {
          qk_step(d);
        }
      } else {
        // Partial unroll lets the next K loads overlap the running product chain (as MLX's attention_nax does).
#pragma clang loop unroll_count(4)
        for (short d = 0; d < QKF; d++) {
          qk_step(d);
        }
      }
      if constexpr (WN == 2) {
        // Each simdgroup of the pair holds a partial dot product over its half of the head dims: add the peer's.
        constexpr short E = score_t::kElemsPerFrag;
        threadgroup float* mine = exchange + (group * 2 + part) * (2 * E * 32) + lane * (2 * E);
        const threadgroup float* theirs = exchange + (group * 2 + 1 - part) * (2 * E * 32) + lane * (2 * E);
        thread auto& lo = scores.frag_at(0, kf);
        thread auto& hi = scores.frag_at(0, kf + 1);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < E; e++) {
          mine[e] = lo[e];
          mine[E + e] = hi[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < E; e++) {
          lo[e] += theirs[e];
          hi[e] += theirs[E + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < score_t::kElemsPerTile; e++) {
      scores.elems()[e] *= scale_log2;
    }

    // Masks: keys past the end, causal, window band, explicit bool mask. Each only ever drops a score.
    constexpr float dropped = Limits<float>::finite_min;
    const int row_pos0 = tile_first_pos + group_row + lane_row; // bottom-right aligned position of row 0
    const int col_pos0 = kb * BK + lane_col;
    const bool check_causal = CAUSAL && kb >= first_causal_block;
    const bool check_window = WINDOW > 0 && kb * BK <= tile_first_pos + group_row + F - 1 - WINDOW;
    // Drop every score of this block for which pred(row position, key position, key index in block) holds.
    auto drop_where = [&](auto pred) {
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < KF; kf++) {
        thread auto& frag = scores.frag_at(0, kf);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < score_t::kFragThrRows; i++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < score_t::kFragThrCols; j++) {
            const short e = i * score_t::kFragThrCols + j;
            const int in_block = kf * F + lane_col + j;
            frag[e] = pred(row_pos0 + i * score_t::kFragRowsJump, col_pos0 + kf * F + j, in_block) ? dropped : frag[e];
          }
        }
      }
    };
    if (tail_block) {
      drop_where([&](int, int, int in_block) { return in_block >= key_tail; });
    }
    if (check_causal) {
      drop_where([&](int row, int col, int) { return row < col; });
    }
    if (check_window) {
      drop_where([&](int row, int col, int) { return col <= row - WINDOW; });
    }
    if (HAS_MASK) {
      using mtile_t = NAXTile<bool, 1, KF>;
      const int mrow = q_tile * BQ + group_row;
      const bool inside = mrow + BQ <= q_len && kb * BK + BK <= geom[kKeyLen];
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < KF; kf++) {
        typename mtile_t::frag_type keep;
        if (inside) {
          mtile_t::NAXFrag_t::load(keep, mask, int64_t(m_str[2]), int64_t(m_str[3]), mrow, kb * BK + kf * F);
        } else {
          mtile_t::NAXFrag_t::load_safe(
              keep, mask, int64_t(m_str[2]), int64_t(m_str[3]), q_len, geom[kKeyLen], mrow, kb * BK + kf * F);
        }
        thread auto& frag = scores.frag_at(0, kf);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < mtile_t::kElemsPerFrag; e++) {
          frag[e] = keep[e] ? frag[e] : dropped;
        }
      }
    }

    // Online softmax (base 2).
    // Row max: every score of the row this lane holds (all fragments), then across the row's four lanes. Max is
    // exact, so the grouping does not change the result; it halves the shuffles of a per-fragment reduction.
    metal::vec<float, kRows> new_max = run_max;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < score_t::kFragThrRows; i++) {
      float m = scores.frag_at(0, 0)[i * score_t::kFragThrCols];
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < KF; kf++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < score_t::kFragThrCols; j++) {
          m = metal::max(m, scores.frag_at(0, kf)[i * score_t::kFragThrCols + j]);
        }
      }
      m = metal::max(m, simd_shuffle_xor(m, ushort(1)));
      m = metal::max(m, simd_shuffle_xor(m, ushort(8)));
      new_max[i] = metal::max(new_max[i], m);
    }
    scores.template row_bin_op<RowExp2Shift>(new_max);
    metal::vec<float, kRows> rescale;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; ++i) {
      rescale[i] = fast::exp2(run_max[i] - new_max[i]);
      run_max[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRows; ++i) {
      run_sum[i] = run_sum[i] * rescale[i];
    }
    scores.template row_reduce<RowAdd>(run_sum);
    acc.template row_bin_op<RowScale>(rescale);

    simdgroup_barrier(mem_flags::mem_none);

    // acc += P @ V, value fragments in pairs.
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < VF; d += 2) {
      if constexpr (BDV == 128 && WN == 1) {
        if (d == 4) {
          threadgroup_barrier(mem_flags::mem_none);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short kf = 0; kf < KF; kf++) {
        NAXTile<T, 1, 2> vt;
        const device T* src = vblk + kf * F * v_ld + d * F;
        if (tail_block) {
          vt.load_rows(src, v_ld, short(key_tail - kf * F));
        } else {
          vt.load(src, v_ld);
        }
        acc_t::NAXFrag_t::mma(
            acc.frag_at(0, d),
            acc.frag_at(0, d + 1),
            scores.frag_at(0, kf),
            metal::false_type{},
            vt.frag_at(0, 0),
            vt.frag_at(0, 1),
            metal::false_type{});
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_none);

  if constexpr (!LAST) {
    acc.store(carry_out + carry_row0 * CARRY + part * (BDV / WN), CARRY);
    if (lane_col == 0 && part == 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRows; ++i) {
        const int64_t r = carry_row0 + lane_row + i * acc_t::kFragRowsJump;
        carry_out[r * CARRY + BDV] = run_max[i];
        carry_out[r * CARRY + BDV + 1] = run_sum[i];
      }
    }
    return;
  }

  metal::vec<float, kRows> inv_sum;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRows; ++i) {
    inv_sum[i] = 1.f / run_sum[i];
  }
  acc.template row_bin_op<RowScale>(inv_sum);
  if (tail_tile) {
    if (tail_rows <= 0) {
      return;
    }
    acc.store_rows(out, out_ld, tail_rows);
  } else {
    acc.store(out, out_ld);
  }
}

} // namespace vmlx_attn
