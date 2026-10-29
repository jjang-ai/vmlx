// SPDX-License-Identifier: Apache-2.0
// K6 m3_router: the MoE router of ROWS rows in one launch -- fp32 gate logits + sigmoid -- and the Hadamard-128
// rotation of the routed experts' input (x * su_gu), which the JANGT gate/up kernel (K7) reads.
//
// Template: T (activation dtype of x), D (hidden, multiple of 128), E (experts), ROWS (1..16).
// Inputs:  x [ROWS, D] T (post-attention-norm hidden), gw [E, D] float32 (router), su [D] float32 (gate/up signs,
//          0 on the layer's outlier columns).
// Outputs: sig [ROWS, E] float32 = sigmoid(x @ gw.T) (group_expert_select's orig_scores; K7 adds the correction bias
//          and picks the top-k), xr [ROWS, D] float32 = rotate_rows(x, su).
// Grid: threadgroups of 256 threads (8 simdgroups) on x: E router threadgroups, then ROWS * ceil(D / 1024) rotation
// threadgroups. grid = (256 * (E + ROWS * ceil(D / 1024)), 1, 1), threadgroup = (256, 1, 1).
//
// Router threadgroup e: MLX's one-row gemv for the logit of expert e (GEMVKernel<float, BM=1, BN=8, SM=1, SN=32,
// TM=4, TN=4> per output row: lane l of simdgroup s reads columns (s * 32 + l) * 4 + i * 1024 .. + 3, fma in column
// order, shuffle-down tree, simdgroup partials added in order), once per row; then MLX's Sigmoid functor with the
// precise exp of MLX's precompiled sigmoid (group_expert_select's sigmoid is not fused into a JIT kernel, whose
// metal::exp is the fast one: 4 % of the selected scores differed with it).
// Rotation threadgroup: moe_kernels.rotate_rows, one Hadamard-128 block per simdgroup.

  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tg = threadgroup_position_in_grid.x;
  threadgroup float part[8 * ROWS];
  if (tg < uint(E)) {
    const uint e = tg;
    const device float* wrow = gw + (size_t)e * D;
    float result[ROWS];
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; r++) result[r] = 0.0f;
    int bn = int(sg * 32u + lane) * 4;
    constexpr int blockN = 1024;
    for (int i = 0; i < D / blockN; ++i) {
      float inter[4];
#pragma clang loop unroll(full)
      for (int tn = 0; tn < 4; tn++) inter[tn] = wrow[bn + tn];
#pragma clang loop unroll(full)
      for (int r = 0; r < ROWS; r++) {
        float v_coeff[4];
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) v_coeff[tn] = static_cast<float>(x[(size_t)r * D + bn + tn]);
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) result[r] += inter[tn] * v_coeff[tn];
      }
      bn += blockN;
    }
    constexpr int leftover = D % blockN;
    if (leftover > 0) {
      float inter[4];
#pragma clang loop unroll(full)
      for (int tn = 0; tn < 4; tn++) inter[tn] = bn + tn < D ? wrow[bn + tn] : 0.0f;
#pragma clang loop unroll(full)
      for (int r = 0; r < ROWS; r++) {
        float v_coeff[4];
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) v_coeff[tn] = bn + tn < D ? static_cast<float>(x[(size_t)r * D + bn + tn]) : 0.0f;
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; tn++) result[r] += inter[tn] * v_coeff[tn];
      }
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; r++) {
#pragma clang loop unroll(full)
      for (ushort sn = 16; sn >= 1; sn >>= 1) result[r] += simd_shuffle_down(result[r], sn);
      if (lane == 0u) part[sg * ROWS + r] = result[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane < uint(ROWS)) {
      float acc = part[lane];
      for (int s = 1; s < 8; s++) acc += part[s * ROWS + lane];
      float y = 1 / (1 + metal::precise::exp(metal::abs(acc)));
      sig[(size_t)lane * E + e] = (acc < 0) ? y : 1 - y;
    }
    return;
  }
  const uint r = tg - uint(E);
  constexpr uint chunks = (uint(D) + 1023u) / 1024u;
  const uint t = r / chunks;
  const uint blk = (r % chunks) * 8u + sg;
  if (t >= uint(ROWS) || blk * 128u >= uint(D)) return;
  const size_t base = (size_t)t * D + blk * 128u + lane * 4u;
  float v[4];
  for (uint i = 0; i < 4u; i++) v[i] = float(x[base + i]) * su[blk * 128u + lane * 4u + i];
  { float a = v[0], b = v[1]; v[0] = a + b; v[1] = a - b; a = v[2]; b = v[3]; v[2] = a + b; v[3] = a - b; }
  { float a = v[0], b = v[2]; v[0] = a + b; v[2] = a - b; a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b; }
  for (uint d = 1; d < 32u; d <<= 1) {
    bool upper = (lane & d) != 0u;
    for (uint i = 0; i < 4u; i++) { float o = simd_shuffle_xor(v[i], d); v[i] = upper ? (o - v[i]) : (v[i] + o); }
  }
  for (uint i = 0; i < 4u; i++) xr[base + i] = v[i] * 0.08838834764831845f;
