// SPDX-License-Identifier: Apache-2.0
// flat_mm: up to five quantized weight segments (concatenated along the output) x R rows (1 .. 16) on the matrix
// units, one 16-row matmul2d tile per weight group whatever R is (row-invariant), fp32 accumulate.
//
// Host-prepended constants:
//   using st0 .. st4 = <segment scale (and bias) dtypes>  (uint8_t for MXFP8);  using ot = <output dtype (EPI 1 / 3)>;
//   using rt = <residual dtype (EPI 1)>;
//   constexpr int K = input width (multiple of 64), N0 .. N4 = segment output rows (0 = unused, multiples of NT),
//   F0 .. F4 = segment formats (0 affine 4-bit g64, 1 MXFP8 g32, 2 affine 8-bit g64, 3 affine 6-bit g64),
//   NT = 32 output columns per threadgroup, SK = simdgroups splitting the weight groups (fixed per shape),
//   EPI = 0: Y [R, NTOT] float; 1: Y [R, N0] ot = ot(y + RS) (one rounding; one segment); 2: SwiGLU-OAI of segment
//   0 (gate) and segment 1 (up, N1 == N0): Y [R, N0] ot; 3: Y [R, NTOT] ot = ot(y)
//   constexpr float ALPHA, LIMIT (EPI 2)
// Inputs (names): A [R, K] bf16, A16 [R, K] half (MXFP8 segments; else any >= 8-element array), XS [>= R K / 64] f32
//   (flat_prep), w0 s0 b0 .. w4 s4 b4 (MLX-layout packed weights, scales, biases; unused / MXFP8 biases: any >= 8
//   element arrays), RS [R, N0] rt (EPI 1; else any >= 8-element array).
// Grid (32 SK x NTILES, 1, 1), threadgroup (32 SK, 1, 1); NTILES = NTOT / NT (EPI 2: N0 / NT); tile t covers
//   output columns t NT .. t NT + NT - 1 of the concatenated segments.
// Every output row's arithmetic: per weight group in order, the 16-row tile's fp32 dot of that row, then
// fma(scale, dot, fma(bias, XS, acc)); simdgroup k's partial covers groups [k KG / SK, (k + 1) KG / SK) and the SK
// partials are added in simdgroup order. None of it depends on R or on the other rows.

  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int R = int(A_shape[0]);
  constexpr int NTOT = N0 + N1 + N2 + N3 + N4;
  constexpr int CAP = 16 * NT / 32;
  constexpr int T0 = N0 / NT, T1 = N1 / NT, T2 = N2 / NT, T3 = N3 / NT;
  const int tile = int(threadgroup_position_in_grid.x);
  int seg = 0, lt = tile, base = 0;
  if (EPI != 2) {
    if (tile >= T0) { seg = 1; lt -= T0; base += N0; }
    if (seg == 1 && lt >= T1) { seg = 2; lt -= T1; base += N1; }
    if (seg == 2 && lt >= T2) { seg = 3; lt -= T2; base += N2; }
    if (seg == 3 && lt >= T3) { seg = 4; lt -= T3; base += N3; }
  }
  const int n0 = lt * NT;
  constexpr int HAS6 = (F0 == 3 && N0) || (F1 == 3 && N1) || (F2 == 3 && N2) || (F3 == 3 && N3) || (F4 == 3 && N4);
  threadgroup uint8_t stage_all[HAS6 ? SK * NT * 64 : 4];
  threadgroup uint8_t* stage = stage_all + (HAS6 ? int(sg) * NT * 64 : 0);

  float C[CAP], Cu[CAP];
  int mi[CAP], ni[CAP];
  bool live[CAP];
  for (int i = 0; i < CAP; ++i) { C[i] = 0.0f; Cu[i] = 0.0f; }
  const device bfloat16_t* Ap = (const device bfloat16_t*)A;
  const device half* A16p = (const device half*)A16;
  // groups of this simdgroup (by the segment's group size)
#define FLAT_RANGE(F) const int kg_ = (F) == 1 ? K / 32 : K / 64; const int g0 = (int(sg) * kg_) / SK; \
    const int g1 = ((int(sg) + 1) * kg_) / SK;
  if (seg == 0) {
    FLAT_RANGE(F0)
    flat_acc<F0, K, NT, st0>(Ap, A16p, XS, (const device uint8_t*)w0, s0, (const device st0*)b0, n0, R, g0, g1, C,
                             mi, ni, live, stage, lane);
    if (EPI == 2) {
      flat_acc<F1, K, NT, st1>(Ap, A16p, XS, (const device uint8_t*)w1, s1, (const device st1*)b1, n0, R, g0, g1,
                               Cu, mi, ni, live, stage, lane);
    }
  } else if (seg == 1) {
    FLAT_RANGE(F1)
    flat_acc<F1, K, NT, st1>(Ap, A16p, XS, (const device uint8_t*)w1, s1, (const device st1*)b1, n0, R, g0, g1, C,
                             mi, ni, live, stage, lane);
  } else if (seg == 2) {
    FLAT_RANGE(F2)
    flat_acc<F2, K, NT, st2>(Ap, A16p, XS, (const device uint8_t*)w2, s2, (const device st2*)b2, n0, R, g0, g1, C,
                             mi, ni, live, stage, lane);
  } else if (seg == 3) {
    FLAT_RANGE(F3)
    flat_acc<F3, K, NT, st3>(Ap, A16p, XS, (const device uint8_t*)w3, s3, (const device st3*)b3, n0, R, g0, g1, C,
                             mi, ni, live, stage, lane);
  } else {
    FLAT_RANGE(F4)
    flat_acc<F4, K, NT, st4>(Ap, A16p, XS, (const device uint8_t*)w4, s4, (const device st4*)b4, n0, R, g0, g1, C,
                             mi, ni, live, stage, lane);
  }
#undef FLAT_RANGE

  // the SK partials, added in simdgroup order
  constexpr int NACC = EPI == 2 ? 2 : 1;
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NACC * CAP * 32];
  if (SK > 1) {
    if (sg > 0) {
      for (int i = 0; i < CAP; ++i) {
        part[((int(sg) - 1) * NACC * CAP + i) * 32 + lane] = C[i];
        if (EPI == 2) part[((int(sg) - 1) * NACC * CAP + CAP + i) * 32 + lane] = Cu[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    for (int s2 = 1; s2 < SK; ++s2) {
      for (int i = 0; i < CAP; ++i) {
        C[i] += part[((s2 - 1) * NACC * CAP + i) * 32 + lane];
        if (EPI == 2) Cu[i] += part[((s2 - 1) * NACC * CAP + CAP + i) * 32 + lane];
      }
    }
  }
  for (int i = 0; i < CAP; ++i) {
    if (!live[i]) continue;
    const int m = mi[i];
    const int n = n0 + ni[i];
    if (EPI == 0) {
      Y[(size_t)m * NTOT + base + n] = static_cast<ot>(C[i]);     // ot = float for EPI 0
    } else if (EPI == 1) {
      Y[(size_t)m * N0 + n] = static_cast<ot>(C[i] + float(RS[(size_t)m * N0 + n]));
    } else if (EPI == 2) {
      // the stock shared expert's roundings: gate / up projections in the activation dtype, SwiGLU-OAI, result
      float gt = float(static_cast<ot>(C[i]));
      float up = float(static_cast<ot>(Cu[i]));
      gt = metal::min(gt, LIMIT);
      up = metal::clamp(up, -LIMIT, LIMIT);
      const float sgm = 1.0f / (1.0f + metal::precise::exp(-ALPHA * gt));
      Y[(size_t)m * N0 + n] = static_cast<ot>((gt * sgm) * (up + 1.0f));
    } else {
      Y[(size_t)m * NTOT + base + n] = static_cast<ot>(C[i]);
    }
  }
