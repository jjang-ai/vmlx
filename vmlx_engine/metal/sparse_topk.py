# SPDX-License-Identifier: Apache-2.0
"""Exact 2-pass / 3-level Metal radix histogram Top-K for long-context sparse attention.

Replaces MLX's O(S log S) ``mx.argpartition`` on 16K..1M key sequences with linear-time
exact 32-bit IEEE-754 order-preserving radix selection (11 + 11 + 10 = 32 bits).
Zero full-sequence sort, zero [B, 1, L, S] boolean mask allocation, exact tie handling.

Shared across ERNIE 5.5 Flash, GLM-5.3, openPangu V2, Naive N0.5, Dots3, and Qwen4.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx


_EQ_CAP = 16384

_RADIX_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline uint float_to_ord(float v) {
    if (isnan(v)) return 0u;
    uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
"""

# ---------------------------------------------------------------------------
# Path A: Multi-row kernel (1 threadgroup of 256 threads per query row)
# All 3 radix levels (11 + 11 + 10 = 32 bits) execute inside ONE dispatch using
# 8 KB of threadgroup memory (2048 atomic_uint bins) — ZERO global atomics!
# Uses float4 vectorized loads for the aligned bulk of each row.
# ---------------------------------------------------------------------------
_SINGLE_TG_TOPK_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint N = uint(scores_shape[1]);
    const uint K = uint(params[0]);
    const uint recent_win = uint(params[1]);
    const uint out_cols = K + recent_win;
    const uint n_valid = HAS_VALID_END ? min(uint(valid_end[row]), N) : N;
    const uint K_eff = min(K, n_valid);
    const uint n4 = n_valid & ~3u;

    const size_t s_base = size_t(row) * size_t(N);
    device uint* o_row = out_ids + size_t(row) * size_t(out_cols);

    threadgroup atomic_uint hist[2048];
    threadgroup uint meta_thresh;
    threadgroup uint meta_cum;
    threadgroup atomic_uint write_pos;

    if (tid == 0u) {
        atomic_store_explicit(&write_pos, 0u, memory_order_relaxed);
    }

    // --- Level 0: top 11 bits (bits 31..21, 2048 bins) ---
    for (uint b = tid; b < 2048u; b += 256u) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid * 4u; i < n4; i += 1024u) {
        size_t si = s_base + i;
        atomic_fetch_add_explicit(&hist[float_to_ord(scores[si]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[float_to_ord(scores[si + 1u]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[float_to_ord(scores[si + 2u]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&hist[float_to_ord(scores[si + 3u]) >> 21], 1u, memory_order_relaxed);
    }
    for (uint i = n4 + tid; i < n_valid; i += 256u) {
        atomic_fetch_add_explicit(&hist[float_to_ord(scores[s_base + i]) >> 21], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint cum = 0u;
        uint b0 = 0u;
        for (int b = 2047; b >= 0; --b) {
            uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (cum + c >= K_eff) {
                b0 = uint(b);
                break;
            }
            cum += c;
        }
        meta_thresh = b0;
        meta_cum = cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b0 = meta_thresh;
    const uint cum0 = meta_cum;

    // --- Level 1: emit (> b0) and build 11-bit histogram (bits 20..10) for (== b0) ---
    for (uint b = tid; b < 2048u; b += 256u) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid * 4u; i < n4; i += 1024u) {
        size_t si = s_base + i;
        uint us[4] = {
            float_to_ord(scores[si]),
            float_to_ord(scores[si + 1u]),
            float_to_ord(scores[si + 2u]),
            float_to_ord(scores[si + 3u]),
        };
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint top11 = us[j] >> 21;
            if (top11 > b0) {
                uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                if (pos < K_eff) o_row[pos] = i + j;
            } else if (top11 == b0) {
                atomic_fetch_add_explicit(&hist[(us[j] >> 10) & 0x7FFu], 1u, memory_order_relaxed);
            }
        }
    }
    for (uint i = n4 + tid; i < n_valid; i += 256u) {
        uint u = float_to_ord(scores[s_base + i]);
        uint top11 = u >> 21;
        if (top11 > b0) {
            uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
            if (pos < K_eff) o_row[pos] = i;
        } else if (top11 == b0) {
            atomic_fetch_add_explicit(&hist[(u >> 10) & 0x7FFu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (cum0 < K_eff) {
        if (tid == 0u) {
            uint cum = cum0;
            uint b1 = 0u;
            for (int b = 2047; b >= 0; --b) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + c >= K_eff) {
                    b1 = uint(b);
                    break;
                }
                cum += c;
            }
            meta_thresh = b1;
            meta_cum = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b1 = meta_thresh;
        const uint cum1 = meta_cum;
        const uint p22 = (b0 << 11) | b1;

        // --- Level 2: emit (top11==b0 && mid11>b1) and build 10-bit hist (bits 9..0) ---
        for (uint b = tid; b < 1024u; b += 256u) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid * 4u; i < n4; i += 1024u) {
            size_t si = s_base + i;
            uint us[4] = {
                float_to_ord(scores[si]),
                float_to_ord(scores[si + 1u]),
                float_to_ord(scores[si + 2u]),
                float_to_ord(scores[si + 3u]),
            };
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                if ((us[j] >> 21) == b0) {
                    uint top22 = us[j] >> 10;
                    if (top22 > p22) {
                        uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                        if (pos < K_eff) o_row[pos] = i + j;
                    } else if (top22 == p22) {
                        atomic_fetch_add_explicit(&hist[us[j] & 0x3FFu], 1u, memory_order_relaxed);
                    }
                }
            }
        }
        for (uint i = n4 + tid; i < n_valid; i += 256u) {
            uint u = float_to_ord(scores[s_base + i]);
            if ((u >> 21) == b0) {
                uint top22 = u >> 10;
                if (top22 > p22) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                } else if (top22 == p22) {
                    atomic_fetch_add_explicit(&hist[u & 0x3FFu], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (cum1 < K_eff) {
            if (tid == 0u) {
                uint cum = cum1;
                uint b2 = 0u;
                for (int b = 1023; b >= 0; --b) {
                    uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                    if (cum + c >= K_eff) {
                        b2 = uint(b);
                        break;
                    }
                    cum += c;
                }
                meta_thresh = b2;
                meta_cum = cum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint exact_u = (p22 << 10) | meta_thresh;
            for (uint i = tid * 4u; i < n4; i += 1024u) {
                size_t si = s_base + i;
                uint us[4] = {
                    float_to_ord(scores[si]),
                    float_to_ord(scores[si + 1u]),
                    float_to_ord(scores[si + 2u]),
                    float_to_ord(scores[si + 3u]),
                };
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    if ((us[j] >> 10) == p22 && us[j] > exact_u) {
                        uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                        if (pos < K_eff) o_row[pos] = i + j;
                    }
                }
            }
            for (uint i = n4 + tid; i < n_valid; i += 256u) {
                uint u = float_to_ord(scores[s_base + i]);
                if ((u >> 10) == p22 && u > exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < n_valid; i += 256u) {
                if (atomic_load_explicit(&write_pos, memory_order_relaxed) >= K_eff) break;
                uint u = float_to_ord(scores[s_base + i]);
                if (u == exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                }
            }
        }
    }

    // Pad invalid older slots (if n_valid < K) with sentinel 0xFFFFFFFFu
    for (uint p = K_eff + tid; p < K; p += 256u) {
        o_row[p] = 0xFFFFFFFFu;
    }
    // Optional fused recent sliding window append: [q_end - recent_win .. q_end)
    if (recent_win > 0u) {
        const uint q_end = HAS_VALID_END ? uint(recent_end[row]) : N;
        const uint w_start = (q_end > recent_win) ? (q_end - recent_win) : 0u;
        const uint w_len = q_end - w_start;
        for (uint j = tid; j < recent_win; j += 256u) {
            o_row[K + j] = (j < w_len) ? (w_start + j) : 0xFFFFFFFFu;
        }
    }
"""

# ---------------------------------------------------------------------------
# Path B: Multi-block parallel kernel with compact boundary gather
# Splits N across G=32 threadgroups per row; Pass 2 gathers ONLY the ~200-2000
# boundary elements (top11 == b0) into compact eq_ids/eq_ords so Pass 3 finishes
# in < 5 us out of L1 cache instead of rescanning N!
# ---------------------------------------------------------------------------
_MB_HIST11_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint G = uint(params[0]);
    const uint N = uint(scores_shape[1]);
    const uint n_valid = HAS_VALID_END ? min(uint(valid_end[row]), N) : N;
    const uint chunk = ((n_valid + G - 1u) / G + 3u) & ~3u;
    const uint i0 = min(blk * chunk, n_valid);
    const uint i1 = min(i0 + chunk, n_valid);
    const uint i1_4 = i0 + ((i1 - i0) & ~3u);

    const size_t s_base = size_t(row) * size_t(N);
    threadgroup atomic_uint local_hist[2048];
    for (uint b = tid; b < 2048u; b += 256u) {
        atomic_store_explicit(&local_hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = i0 + tid * 4u; i < i1_4; i += 1024u) {
        size_t si = s_base + i;
        atomic_fetch_add_explicit(&local_hist[float_to_ord(scores[si]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[float_to_ord(scores[si + 1u]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[float_to_ord(scores[si + 2u]) >> 21], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[float_to_ord(scores[si + 3u]) >> 21], 1u, memory_order_relaxed);
    }
    for (uint i = i1_4 + tid; i < i1; i += 256u) {
        atomic_fetch_add_explicit(&local_hist[float_to_ord(scores[s_base + i]) >> 21], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint* b_out = block_hists + (size_t(row) * size_t(G) + size_t(blk)) * 2048u;
    for (uint b = tid; b < 2048u; b += 256u) {
        b_out[b] = atomic_load_explicit(&local_hist[b], memory_order_relaxed);
    }
"""

_MB_REDUCE_THRESH_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint G = uint(params[0]);
    const uint K = uint(params[1]);
    const uint N = uint(params[2]);
    const uint n_valid = HAS_VALID_END ? min(uint(valid_end[row]), N) : N;
    const uint K_eff = min(K, n_valid);

    const device uint* bh = block_hists + size_t(row) * size_t(G) * 2048u;
    threadgroup uint tot[2048];
    for (uint b = tid; b < 2048u; b += 256u) {
        uint sum = 0u;
        for (uint g = 0u; g < G; ++g) sum += bh[size_t(g) * 2048u + b];
        tot[b] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint sh_b0;
    if (tid == 0u) {
        uint cum = 0u;
        uint b0 = 0u;
        for (int b = 2047; b >= 0; --b) {
            uint c = tot[b];
            if (cum + c >= K_eff) {
                b0 = uint(b);
                break;
            }
            cum += c;
        }
        sh_b0 = b0;
        meta[row * 4u + 0u] = b0;
        meta[row * 4u + 1u] = cum;
        meta[row * 4u + 2u] = K_eff;
        meta[row * 4u + 3u] = tot[b0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b0 = sh_b0;
    threadgroup uint blk_above[128];
    threadgroup uint blk_eq[128];
    if (tid < G) {
        uint a = 0u;
        const device uint* g_ptr = bh + size_t(tid) * 2048u;
        for (uint b = b0 + 1u; b < 2048u; ++b) a += g_ptr[b];
        blk_above[tid] = a;
        blk_eq[tid] = g_ptr[b0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint off_a = 0u;
        uint off_e = 0u;
        for (uint g = 0u; g < G; ++g) {
            blk_offsets[(size_t(row) * size_t(G) + g) * 2u + 0u] = off_a;
            blk_offsets[(size_t(row) * size_t(G) + g) * 2u + 1u] = off_e;
            off_a += blk_above[g];
            off_e += blk_eq[g];
        }
    }
"""

_MB_EMIT_COMPACT_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint G = uint(params[0]);
    const uint K = uint(params[1]);
    const uint recent_win = uint(params[2]);
    const uint eq_cap = uint(params[3]);
    const uint out_cols = K + recent_win;
    const uint N = uint(scores_shape[1]);
    const uint n_valid = HAS_VALID_END ? min(uint(valid_end[row]), N) : N;
    const uint b0 = meta[row * 4u + 0u];
    const uint K_eff = meta[row * 4u + 2u];
    const uint n_eq = meta[row * 4u + 3u];
    const uint base_above = blk_offsets[(size_t(row) * size_t(G) + blk) * 2u + 0u];
    const uint base_eq = blk_offsets[(size_t(row) * size_t(G) + blk) * 2u + 1u];
    const bool gather_eq = (n_eq <= eq_cap);

    const uint chunk = ((n_valid + G - 1u) / G + 3u) & ~3u;
    const uint i0 = min(blk * chunk, n_valid);
    const uint i1 = min(i0 + chunk, n_valid);
    const uint i1_4 = i0 + ((i1 - i0) & ~3u);

    const size_t s_base = size_t(row) * size_t(N);
    device uint* o_row = out_ids + size_t(row) * size_t(out_cols);
    device uint* eq_i_row = eq_ids + size_t(row) * size_t(eq_cap);
    device uint* eq_u_row = eq_ords + size_t(row) * size_t(eq_cap);

    threadgroup atomic_uint local_above;
    threadgroup atomic_uint local_eq;
    if (tid == 0u) {
        atomic_store_explicit(&local_above, 0u, memory_order_relaxed);
        atomic_store_explicit(&local_eq, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = i0 + tid * 4u; i < i1_4; i += 1024u) {
        size_t si = s_base + i;
        uint us[4] = {
            float_to_ord(scores[si]),
            float_to_ord(scores[si + 1u]),
            float_to_ord(scores[si + 2u]),
            float_to_ord(scores[si + 3u]),
        };
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint top11 = us[j] >> 21;
            if (top11 > b0) {
                uint p = base_above + atomic_fetch_add_explicit(&local_above, 1u, memory_order_relaxed);
                if (p < K_eff) o_row[p] = i + j;
            } else if (gather_eq && top11 == b0) {
                uint q = base_eq + atomic_fetch_add_explicit(&local_eq, 1u, memory_order_relaxed);
                if (q < eq_cap) {
                    eq_i_row[q] = i + j;
                    eq_u_row[q] = us[j];
                }
            }
        }
    }
    for (uint i = i1_4 + tid; i < i1; i += 256u) {
        uint u = float_to_ord(scores[s_base + i]);
        uint top11 = u >> 21;
        if (top11 > b0) {
            uint p = base_above + atomic_fetch_add_explicit(&local_above, 1u, memory_order_relaxed);
            if (p < K_eff) o_row[p] = i;
        } else if (gather_eq && top11 == b0) {
            uint q = base_eq + atomic_fetch_add_explicit(&local_eq, 1u, memory_order_relaxed);
            if (q < eq_cap) {
                eq_i_row[q] = i;
                eq_u_row[q] = u;
            }
        }
    }

    if (blk == 0u) {
        for (uint p = K_eff + tid; p < K; p += 256u) {
            o_row[p] = 0xFFFFFFFFu;
        }
        if (recent_win > 0u) {
            const uint q_end = HAS_VALID_END ? uint(recent_end[row]) : N;
            const uint w_start = (q_end > recent_win) ? (q_end - recent_win) : 0u;
            const uint w_len = q_end - w_start;
            for (uint j = tid; j < recent_win; j += 256u) {
                o_row[K + j] = (j < w_len) ? (w_start + j) : 0xFFFFFFFFu;
            }
        }
    }
"""

_MB_FINISH_COMPACT_SRC = r"""
    // 1 TG per row: finishes boundary [cum0 .. K_eff) from compact eq_ids/eq_ords (fast path)
    // or full s_row fallback if n_eq > eq_cap.
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint K = uint(params[1]);
    const uint recent_win = uint(params[2]);
    const uint eq_cap = uint(params[3]);
    const uint out_cols = K + recent_win;
    const uint N = uint(scores_shape[1]);
    const uint n_valid = HAS_VALID_END ? min(uint(valid_end[row]), N) : N;
    const uint b0 = meta[row * 4u + 0u];
    const uint cum0 = meta[row * 4u + 1u];
    const uint K_eff = meta[row * 4u + 2u];
    const uint n_eq = meta[row * 4u + 3u];

    const device uint* in_row = out_ids_in + size_t(row) * size_t(out_cols);
    device uint* o_row = out_ids + size_t(row) * size_t(out_cols);
    for (uint j = tid; j < out_cols; j += 256u) {
        o_row[j] = in_row[j];
    }
    if (cum0 >= K_eff) return;

    threadgroup atomic_uint hist[2048];
    threadgroup uint sh_b1;
    threadgroup uint sh_cum1;
    threadgroup uint sh_b2;
    threadgroup atomic_uint write_pos;
    if (tid == 0u) atomic_store_explicit(&write_pos, cum0, memory_order_relaxed);
    for (uint b = tid; b < 2048u; b += 256u) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n_eq <= eq_cap) {
        // Fast path: scan ONLY the ~200-2000 gathered boundary candidates!
        const device uint* eq_i = eq_ids + size_t(row) * size_t(eq_cap);
        const device uint* eq_u = eq_ords + size_t(row) * size_t(eq_cap);
        for (uint q = tid; q < n_eq; q += 256u) {
            atomic_fetch_add_explicit(&hist[(eq_u[q] >> 10) & 0x7FFu], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint cum = cum0;
            uint b1 = 0u;
            for (int b = 2047; b >= 0; --b) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + c >= K_eff) { b1 = uint(b); break; }
                cum += c;
            }
            sh_b1 = b1;
            sh_cum1 = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint p22 = (b0 << 11) | sh_b1;
        const uint cum1 = sh_cum1;
        for (uint b = tid; b < 1024u; b += 256u) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint q = tid; q < n_eq; q += 256u) {
            uint u = eq_u[q];
            uint top22 = u >> 10;
            if (top22 > p22) {
                uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                if (pos < K_eff) o_row[pos] = eq_i[q];
            } else if (top22 == p22) {
                atomic_fetch_add_explicit(&hist[u & 0x3FFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (cum1 < K_eff) {
            if (tid == 0u) {
                uint cum = cum1;
                uint b2 = 0u;
                for (int b = 1023; b >= 0; --b) {
                    uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                    if (cum + c >= K_eff) { b2 = uint(b); break; }
                    cum += c;
                }
                sh_b2 = b2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint exact_u = (p22 << 10) | sh_b2;
            for (uint q = tid; q < n_eq; q += 256u) {
                uint u = eq_u[q];
                if ((u >> 10) == p22 && u > exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = eq_i[q];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = tid; q < n_eq; q += 256u) {
                if (atomic_load_explicit(&write_pos, memory_order_relaxed) >= K_eff) break;
                if (eq_u[q] == exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = eq_i[q];
                }
            }
        }
    } else {
        // Fallback when > eq_cap elements share the exact same top-11 bits (e.g. all-zero scores)
        const size_t s_base = size_t(row) * size_t(N);
        for (uint i = tid; i < n_valid; i += 256u) {
            uint u = float_to_ord(scores[s_base + i]);
            if ((u >> 21) == b0) {
                atomic_fetch_add_explicit(&hist[(u >> 10) & 0x7FFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cum = cum0;
            uint b1 = 0u;
            for (int b = 2047; b >= 0; --b) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + c >= K_eff) { b1 = uint(b); break; }
                cum += c;
            }
            sh_b1 = b1;
            sh_cum1 = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint p22 = (b0 << 11) | sh_b1;
        const uint cum1 = sh_cum1;
        for (uint b = tid; b < 1024u; b += 256u) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n_valid; i += 256u) {
            uint u = float_to_ord(scores[s_base + i]);
            if ((u >> 21) == b0) {
                uint top22 = u >> 10;
                if (top22 > p22) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                } else if (top22 == p22) {
                    atomic_fetch_add_explicit(&hist[u & 0x3FFu], 1u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cum1 < K_eff) {
            if (tid == 0u) {
                uint cum = cum1;
                uint b2 = 0u;
                for (int b = 1023; b >= 0; --b) {
                    uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                    if (cum + c >= K_eff) { b2 = uint(b); break; }
                    cum += c;
                }
                sh_b2 = b2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint exact_u = (p22 << 10) | sh_b2;
            for (uint i = tid; i < n_valid; i += 256u) {
                uint u = float_to_ord(scores[s_base + i]);
                if ((u >> 10) == p22 && u > exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < n_valid; i += 256u) {
                if (atomic_load_explicit(&write_pos, memory_order_relaxed) >= K_eff) break;
                uint u = float_to_ord(scores[s_base + i]);
                if (u == exact_u) {
                    uint pos = atomic_fetch_add_explicit(&write_pos, 1u, memory_order_relaxed);
                    if (pos < K_eff) o_row[pos] = i;
                }
            }
        }
    }
"""


@lru_cache(maxsize=4)
def _single_tg_kernel(has_valid_end: bool):
    return mx.fast.metal_kernel(
        name=f"vmlx_radix_topk_1tg_v2_{int(has_valid_end)}",
        input_names=["scores", "params", "valid_end", "recent_end"],
        output_names=["out_ids"],
        header=_RADIX_HEADER,
        source=_SINGLE_TG_TOPK_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mb_hist_kernel(has_valid_end: bool):
    return mx.fast.metal_kernel(
        name=f"vmlx_radix_topk_mb_hist_v2_{int(has_valid_end)}",
        input_names=["scores", "params", "valid_end"],
        output_names=["block_hists"],
        header=_RADIX_HEADER,
        source=_MB_HIST11_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mb_reduce_kernel(has_valid_end: bool):
    return mx.fast.metal_kernel(
        name=f"vmlx_radix_topk_mb_reduce_v2_{int(has_valid_end)}",
        input_names=["block_hists", "params", "valid_end"],
        output_names=["meta", "blk_offsets"],
        header=_RADIX_HEADER,
        source=_MB_REDUCE_THRESH_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mb_emit_kernel(has_valid_end: bool):
    return mx.fast.metal_kernel(
        name=f"vmlx_radix_topk_mb_emit_v2_{int(has_valid_end)}",
        input_names=["scores", "meta", "blk_offsets", "params", "valid_end", "recent_end"],
        output_names=["out_ids", "eq_ids", "eq_ords"],
        header=_RADIX_HEADER,
        source=_MB_EMIT_COMPACT_SRC,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def _mb_finish_kernel(has_valid_end: bool):
    return mx.fast.metal_kernel(
        name=f"vmlx_radix_topk_mb_finish_v2_{int(has_valid_end)}",
        input_names=["scores", "out_ids_in", "meta", "eq_ids", "eq_ords", "params", "valid_end"],
        output_names=["out_ids"],
        header=_RADIX_HEADER,
        source=_MB_FINISH_COMPACT_SRC,
        ensure_row_contiguous=True,
    )


def metal_radix_topk(
    scores: mx.array,
    k: int,
    *,
    valid_end: Optional[mx.array] = None,
    recent_window: int = 0,
    recent_end: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Exact 32-bit IEEE-754 Metal radix Top-K over the last dimension of ``scores``.

    Args:
        scores: FP32 array of shape ``(*lead, N)``.
        k: Number of highest-scoring indices to select per row (``1 <= k <= N``).
        valid_end: Optional ``(*lead,)`` uint32/int32 per-row exclusive cutoff into ``[0, N)``
            (only indices ``< valid_end[r]`` are considered for Top-K; slots ``[n_valid, k)``
            are padded with ``0xFFFFFFFF``).
        recent_window: If > 0, appends the trailing ``recent_window`` indices ending at
            ``recent_end[r]`` (or ``N`` if None) directly into the output array of width
            ``k + recent_window``.

    Returns:
        ``uint32`` array of shape ``(*lead, k + recent_window)``, or ``None`` if not admitted.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None
    if scores.ndim < 1 or scores.dtype != mx.float32:
        return None
    lead = scores.shape[:-1]
    N = int(scores.shape[-1])
    R = int(scores.size // N) if N > 0 else 0
    k = int(k)
    recent_window = int(recent_window)
    if R < 1 or N <= k or k < 1 or k > 8192 or N > 2_097_152 or R > 65535:
        return None
    if scores.size >= 2**31:
        return None

    s2 = scores.reshape(R, N)
    has_ve = valid_end is not None
    ve = valid_end.astype(mx.uint32).reshape(R) if has_ve else mx.zeros((1,), dtype=mx.uint32)
    re = (
        recent_end.astype(mx.uint32).reshape(R)
        if (has_ve and recent_end is not None)
        else (ve if has_ve else mx.zeros((1,), dtype=mx.uint32))
    )
    out_cols = k + recent_window

    # Multi-block compact-boundary path when R is small (decode/verify) and N >= 8K
    # (single-TG leaves 39-79 GPU cores idle at 16K-48K decode!).
    if R <= 16 and N >= 8192:
        G = min(32, max(8, (N + 1023) // 1024))
        p_hist = mx.array([G], dtype=mx.uint32)
        bh = _mb_hist_kernel(has_ve)(
            inputs=[s2, p_hist, ve],
            template=[("HAS_VALID_END", int(has_ve))],
            grid=(256 * G, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R, G, 2048)],
            output_dtypes=[mx.uint32],
        )[0]
        p_red = mx.array([G, k, N], dtype=mx.uint32)
        meta, blk_off = _mb_reduce_kernel(has_ve)(
            inputs=[bh, p_red, ve],
            template=[("HAS_VALID_END", int(has_ve))],
            grid=(256, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R, 4), (R, G, 2)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        p_emit = mx.array([G, k, recent_window, _EQ_CAP], dtype=mx.uint32)
        out_partial, eq_ids, eq_ords = _mb_emit_kernel(has_ve)(
            inputs=[s2, meta, blk_off, p_emit, ve, re],
            template=[("HAS_VALID_END", int(has_ve))],
            grid=(256 * G, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R, out_cols), (R, _EQ_CAP), (R, _EQ_CAP)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
        )
        out_ids = _mb_finish_kernel(has_ve)(
            inputs=[s2, out_partial, meta, eq_ids, eq_ords, p_emit, ve],
            template=[("HAS_VALID_END", int(has_ve))],
            grid=(256, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R, out_cols)],
            output_dtypes=[mx.uint32],
        )[0]
    else:
        params = mx.array([k, recent_window], dtype=mx.uint32)
        out_ids = _single_tg_kernel(has_ve)(
            inputs=[s2, params, ve, re],
            template=[("HAS_VALID_END", int(has_ve))],
            grid=(256, R, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(R, out_cols)],
            output_dtypes=[mx.uint32],
        )[0]

    return out_ids.reshape(*lead, out_cols)
