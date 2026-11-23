"""Copy-only kernels for committing short speculative prefixes."""

import math
from functools import lru_cache

import mlx.core as mx


def _metal_decode(length):
    return (
        1 <= length <= 8 and mx.metal.is_available() and mx.default_device() == mx.gpu
    )


@lru_cache(maxsize=1)
def _select_states_kernel():
    return mx.fast.metal_kernel(
        name="select_speculative_temporal_states",
        input_names=["middle", "final", "initial", "keep"],
        output_names=["out"],
        ensure_row_contiguous=False,
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = final_shape[0], D = final_shape[1];
        if (i >= B * D) return;
        int b = i / D, j = i % D, n = keep[b];
        if (n == 0)
            out[i] = initial[b * initial_strides[0] + j * initial_strides[1]];
        else if (n == TOTAL)
            out[i] = final[b * final_strides[0] + j * final_strides[1]];
        else
            out[i] = middle[b * middle_strides[0] + (n - 1) * middle_strides[1]
                            + j * middle_strides[2]];
        """,
    )


def select_temporal_states(middle, final, initial, lengths, total):
    if (
        not _metal_decode(total)
        or middle is None
        or final.ndim < 2
        or not final.size
        or final.size >= 2**31
        or middle.dtype != final.dtype
        or middle.shape[0] != final.shape[0]
        or middle.shape[1] < total - 1
        or middle.shape[2:] != final.shape[1:]
        or (
            initial is not None
            and (initial.dtype != final.dtype or initial.shape != final.shape)
        )
    ):
        return None
    batch = final.shape[0]
    if initial is None:
        initial = mx.zeros_like(final) if 0 in lengths else final
    return _select_states_kernel()(
        inputs=[
            middle.reshape(batch, middle.shape[1], -1),
            final.reshape(batch, -1),
            initial.reshape(batch, -1),
            mx.array(lengths, mx.int32),
        ],
        template=[("TOTAL", total)],
        grid=(final.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[final.shape],
        output_dtypes=[final.dtype],
    )[0]


@lru_cache(maxsize=1)
def _rotating_prefix_kernel():
    # The same physical copy applies to K and V, including differing head widths.
    source = "uint i = thread_position_in_grid.x;\n"
    for name in ("keys", "values"):
        source += (
            r"""
        {
            int B = OLD_shape[0], H = OLD_shape[1], D = OLD_shape[3];
            if (i < B * H * OUTLEN * D) {
                int d = i % D, t = (i / D) % OUTLEN;
                int h = (i / (D * OUTLEN)) % H, b = i / (D * OUTLEN * H);
                int pos = (t - (KEEP - lengths[b]) + OUTLEN) % OUTLEN;
                int old_length = layout[0], trim = layout[1], idx = layout[2];
                if (pos < old_length) {
                    int src = pos + trim;
                    if (ROTATED) src = (src + idx) % OLD_shape[2];
                    OUT[i] = OLD[b * OLD_strides[0] + h * OLD_strides[1]
                                 + src * OLD_strides[2] + d * OLD_strides[3]];
                } else {
                    int src = pos - old_length;
                    OUT[i] = NEW[b * NEW_strides[0] + h * NEW_strides[1]
                                 + src * NEW_strides[2] + d * NEW_strides[3]];
                }
            }
        }
        """.replace("OLD", "old_" + name)
            .replace("NEW", "new_" + name)
            .replace("OUT[", "out_" + name + "[")
        )
    return mx.fast.metal_kernel(
        name="commit_speculative_rotating_prefix",
        input_names=[
            "old_keys",
            "old_values",
            "new_keys",
            "new_values",
            "lengths",
            "layout",
        ],
        output_names=["out_keys", "out_values"],
        ensure_row_contiguous=False,
        source=source,
    )


def commit_rotating_prefix(cache, original, updates, lengths, length):
    """Fuse chronological ordering, concat and per-row roll for a full window.

    Preserve the native cache's physical layout, including invalid padding.
    Return False for empty/growing caches, replayed updates, and in-place writes.
    The transaction adapter handles those through its existing replay path.
    """
    from ..models.cache import BatchRotatingKVCache

    keep = max(lengths)
    ragged = len(set(lengths)) > 1
    if (
        not _metal_decode(length)
        or type(cache) is not BatchRotatingKVCache
        or "update_and_fetch" in original
        or len(updates) != 1
        or original.get("keys") is None
        or original.get("_lengths") is not None
        or not keep
        or (keep == 1 and not ragged)
        or updates[0][0].shape[2] < keep
    ):
        return False
    keys, values = updates[0]
    old_keys, old_values = original["keys"], original["values"]
    if any(
        old.ndim != 4
        or new.ndim != 4
        or old.dtype != new.dtype
        or old.shape[:2] != new.shape[:2]
        or old.shape[3:] != new.shape[3:]
        or new.shape[2] < keep
        for old, new in ((old_keys, keys), (old_values, values))
    ):
        return False
    rotated, idx = original["rotated"], original["_idx"]
    count = old_keys.shape[2] if rotated else idx
    if count < cache.max_size:
        return False
    trim = max(0, count - cache.max_size + 1)
    old_length = count - trim
    out_length = old_length + keep
    shapes = [(*v.shape[:2], out_length, v.shape[-1]) for v in (old_keys, old_values)]
    if max(map(math.prod, shapes)) >= 2**31:
        return False
    retained = mx.array(lengths, mx.int32)
    outputs = _rotating_prefix_kernel()(
        inputs=[
            old_keys,
            old_values,
            keys,
            values,
            retained,
            mx.array([old_length, trim, idx], mx.int32),
        ],
        template=[("KEEP", keep), ("OUTLEN", out_length), ("ROTATED", rotated)],
        grid=(max(map(math.prod, shapes)), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=[old_keys.dtype, old_values.dtype],
    )
    cache.__dict__.clear()
    cache.__dict__.update(original)
    cache.keys, cache.values = outputs
    cache.left_padding = original["left_padding"] - trim + (keep - retained)
    cache.offset = original["offset"] + retained
    cache._offset = original["_offset"] + keep
    cache._idx = out_length
    cache.rotated = False
    cache._lengths = None
    cache.keys = mx.depends(cache.keys, (cache.left_padding, cache.offset))
    return True
