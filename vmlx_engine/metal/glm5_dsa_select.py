# SPDX-License-Identifier: Apache-2.0
"""Opt-in FP32 GLM DSA prefill selection with bounded Metal specializations.

The packaged MLX 0.32.2 BlockMergeSort helper preserves its comparator and
stable index order. Each adjacent block emits only its top 512 candidates;
subsequent stages merge those candidates without full-index gather passes.
Scoring, causal masks, pool expansion and cache ownership remain in the model.
This is not a decode optimization or a model-throughput claim.
"""

from functools import lru_cache
import importlib.metadata
import logging
from pathlib import Path
import os

import mlx.core as mx

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_VERIFIED_VARIANTS: set[tuple[int, bool]] = set()

_SOURCE = r"""
const uint lane = thread_index_in_threadgroup;
const uint chunk = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint n = values_shape[values_ndim - 1];
const uint out_n = (n / BLOCK) * K + min(uint(K), n % BLOCK);
const uint start = chunk * BLOCK;
const uint length = min(uint(BLOCK), n - start);
threadgroup float shared_values[BLOCK];
threadgroup uint shared_ids[BLOCK];
for (uint i = lane; i < BLOCK; i += THREADS) {
    shared_values[i] = i < length ? values[row * n + start + i] : LessThan<float>::init;
    if (FIRST) shared_ids[i] = start + i;
    else shared_ids[i] = i < length ? ids[row * n + start + i] : 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
BlockMergeSort<float,uint,true,THREADS,4,LessThan<float>>::sort(
    shared_values, shared_ids, int(length), uint3(lane,0,0));
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = lane; i < min(uint(K), length); i += THREADS) {
    kept_values[row * out_n + chunk * K + i] = shared_values[i];
    kept_ids[row * out_n + chunk * K + i] = shared_ids[i];
}
"""


def glm5_dsa_select_requested() -> bool:
    return os.environ.get("VMLX_GLM5_DSA_BLOCK_SELECT", "0") == "1"


@lru_cache(maxsize=1)
def _compatible_sort_version() -> bool:
    # The public argpartition API does not promise this backend's order.
    # Retain stock behavior until a different MLX version is qualified.
    try:
        return importlib.metadata.version("mlx") == "0.32.2"
    except importlib.metadata.PackageNotFoundError:
        return False


@lru_cache(maxsize=1)
def _kernel():
    header = Path(__file__).with_name("glm5_dsa_sort.metal").read_text()
    return mx.fast.metal_kernel(
        name="vmlx_glm5_dsa_block_candidates",
        input_names=["values", "ids"],
        output_names=["kept_values", "kept_ids"],
        header=header, source=_SOURCE, compile_options={"math_mode": "safe"},
    )


def glm5_dsa_select(negative_scores, *, k: int, enabled: bool):
    """Return stock-order pool indices, or None for the unchanged stock path.

Only multi-query, single-batch FP32 selection with more than one 2048-pool
block is eligible. The threshold identifies actual hierarchical work, not a
universal speed crossover. Decode, unsupported layouts/k/dtypes, unqualified
MLX versions, and first-launch errors keep the existing selection path.
No input/model/cache arrays are retained in Python or compiled graph caches.
"""
    global _FAILED, _OBSERVED
    if not enabled or _FAILED:
        return None
    if (
        negative_scores.ndim != 3
        or negative_scores.shape[0] != 1
        or negative_scores.shape[1] <= 1
        or negative_scores.shape[-1] <= 2048
        or negative_scores.dtype != mx.float32
        or type(k) is not int or k != 512
        or negative_scores.size >= 2**32
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or not _compatible_sort_version()
    ):
        return None
    try:
        lead = negative_scores.shape[:-1]
        rows = negative_scores.shape[1]
        values = negative_scores
        ids = mx.array([0], dtype=mx.uint32)
        first = True
        while True:
            n = values.shape[-1]
            tile = min(2048, max(128, 1 << (n-1).bit_length()))
            full, tail = divmod(n, tile)
            out_n = full*k + min(k, tail)
            threads = tile//4
            values, ids = _kernel()(
                inputs=[values, ids],
                template=[("BLOCK", tile), ("K", k),
                          ("THREADS", threads), ("FIRST", first)],
                grid=(threads*(full+bool(tail)), rows, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[(*lead, out_n), (*lead, out_n)],
                output_dtypes=[mx.float32, mx.uint32],
            )
            variant = (tile, first)
            if variant not in _VERIFIED_VARIANTS:
                # Surface a new shader's compilation/launch error before
                # returning lazy indices to attention/cache consumers.
                mx.eval(values, ids)
                _VERIFIED_VARIANTS.add(variant)
            if out_n == k:
                break
            first = False
        if not _OBSERVED:
            mx.eval(ids)
            _OBSERVED = True
            logger.info(
                "GLM DSA block selection active: rows=%d pools=%d k=%d "
                "scores=fp32 scope=base_prefill context_specialization=false "
                "cache_state=unchanged",
                rows, negative_scores.shape[-1], k,
            )
        return ids
    except (OSError, RuntimeError, ValueError) as exc:
        _FAILED = True
        logger.warning("GLM DSA block selection disabled after launch failure: %s", exc)
        return None
