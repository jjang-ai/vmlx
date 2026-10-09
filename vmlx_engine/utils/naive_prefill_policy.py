# SPDX-License-Identifier: Apache-2.0
"""Naive full-attention prefill arithmetic policy, shared with the cache identity.

Naive's full-attention layers use Q/K width 192 and V width 128. MLX's fused SDPA needs equal widths, so the stock path
MATERIALIZES the score tensor (heads x chunk x history x fp32): ~11 GB for a 2,048-token chunk at 22.5k of history. With
96.8 GB of weights resident the prefill admission guard then refused the chunk ("projected transient 10.94GB exceeds the
device working-set limit"), so every Naive prompt above ~20k tokens failed (audit 2026-10-07). The padded path zero-pads
V to 192 and runs the fused kernel: the same attention, a different reduction order (different greedy text than stock on
long prompts, so it cannot silently replace stock everywhere).

Modes (env VMLX_NAIVE_PADDED_PREFILL):
* unset / "auto" (default): stock while the score tensor stays <= VMLX_NAIVE_PADDED_PREFILL_AUTO_GIB (default 4 GiB =
  8k of history at 2,048-token chunks), padded above it. Prompts the stock path can serve keep byte-identical output;
  longer ones are served instead of refused. Deterministic by geometry, never by free memory.
* "1" / "on": padded for every chunk that qualifies (the original opt-in).
* "0" / "off": stock always (the pre-2026-10-07 behaviour).
"""
import os

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


def naive_padded_prefill_mode() -> str:
    raw = os.environ.get("VMLX_NAIVE_PADDED_PREFILL", "auto").strip().lower()
    if raw in _ON:
        return "on"
    if raw in _OFF:
        return "off"
    return "auto"


def naive_padded_prefill_requested() -> bool:
    """True when the padded path is forced for every qualifying chunk (mode "on")."""
    return naive_padded_prefill_mode() == "on"


def naive_padded_prefill_auto_bytes() -> int:
    try:
        gib = float(os.environ.get("VMLX_NAIVE_PADDED_PREFILL_AUTO_GIB", "4"))
    except ValueError:
        gib = 4.0
    return max(0, int(gib * (1 << 30)))


def naive_use_padded_prefill(n_heads: int, q_len: int, k_len: int) -> bool:
    """Whether a chunk takes the padded fused path under the current mode."""
    mode = naive_padded_prefill_mode()
    if mode == "on":
        return True
    if mode == "off":
        return False
    return int(n_heads) * int(q_len) * int(k_len) * 4 > naive_padded_prefill_auto_bytes()


NAIVE_PADDED_PREFILL_IDENTITY = "naive_padded_prefill_v1"


def naive_padded_prefill_identity() -> str | None:
    """Cache-identity fragment: states made under different prefill arithmetic are never mixed."""
    mode = naive_padded_prefill_mode()
    sub = ("+dsa_stock_subchunk_v1" if naive_dsa_stock_subchunk() else "") + \
        ("+dsa_sparse_kernel_v1" if naive_dsa_sparse_kernel() else "")
    if mode == "on":
        return NAIVE_PADDED_PREFILL_IDENTITY + sub
    if mode == "auto":
        return f"naive_padded_prefill_auto_v1:{naive_padded_prefill_auto_bytes()}" + sub
    return ("naive_stock" + sub) if sub else None


def naive_dsa_stock_subchunk() -> bool:
    """OPT-IN (VMLX_NAIVE_DSA_STOCK_SUBCHUNK=1): DSA top-k (array-mask) prefill chunks use STOCK SDPA, query-sub-chunked
    to <= 2 GiB of scores, instead of the padded fused kernel. Standalone the padded kernel is 2.0-2.7x slower with an
    array mask, but end to end the gain measured only +2-4% (16k/24k, 2026-10-09, gap unexplained) and stock is LESS
    accurate (6.9e-4 vs the padded path's < 4.9e-4 bf16 unit roundoff against the fp32 oracle in
    tests/test_naive_padded_prefill.py). Default OFF. Part of the cache identity."""
    return os.environ.get("VMLX_NAIVE_DSA_STOCK_SUBCHUNK", "0").strip().lower() in _ON


def naive_dsa_sparse_kernel() -> bool:
    """DSA top-k prefill chunks with >= 4096 keys use the per-query sparse kernel (vmlx_engine/jangt/dsa_sparse.py):
    O(L x top_k), ~30 ms per 2048-query chunk at any history vs 63/125/250 ms dense at 8k/16k/32k, with the same error
    against an fp32 oracle as the padded dense path (1.7e-3, bf16 output rounding; 2026-10-09). Default ON;
    VMLX_NAIVE_DSA_SPARSE_KERNEL=0 restores dense. Part of the cache identity."""
    return os.environ.get("VMLX_NAIVE_DSA_SPARSE_KERNEL", "1").strip().lower() not in _OFF
