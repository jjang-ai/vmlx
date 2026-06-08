"""Runtime compatibility patches for upstream mlx_vlm bugs.

These are monkey-patches applied once at import time. They wrap known-broken
methods so callers don't need to fork vendored files.

Patches applied
---------------

* Qwen3-VL ``VisionModel.rot_pos_emb`` — upstream types ``grid_thw`` as
  ``mx.array`` but the caller sometimes passes a numpy ``ndarray`` (seen on
  Qwen3.5-35B-A3B bf16, issue #69). ``mx.max(ndarray)`` raises
  ``TypeError: max(): incompatible function arguments``. Coerce on entry.
* Qwen3-VL ``VisionModel.__call__`` — same ``grid_thw`` typing issue when
  ``fast_pos_embed_interpolate`` iterates over a numpy array.
* Qwen3.5/3.6 VL ``Model.sanitize`` — HF-native 3D patch-embed weights can
  arrive as ``(out, channels, temporal, height, width)`` while MLX Conv3D
  expects channels-last ``(out, temporal, height, width, channels)``.
* mlx-vlm ``PromptCacheState`` restore trims cached KV as rank-4 tensors only.
  Qwen3.5/N2 VLM cache tensors can be rank-3 ``(batch, seq, hidden)``; wrap
  them so upstream's rank-4 trim syntax maps back to rank-3 slicing.
* mlx-vlm full-image prefill creates a prompt cache before Qwen/N2 language
  forward, but only primes mRoPE deltas on cached-prefix reuse. Prime the same
  Qwen ``get_rope_index`` state for first full multimodal prefill.
* Qwen3.5/N2 language forward can still reach its nonzero-cache delta branch
  with ``self._rope_deltas`` unset. Use explicit request deltas or recompute
  from Qwen's own ``get_rope_index`` instead of adding a cache offset to None.
  Single-element MLX cache offsets are also normalized to Python ints so the
  precomputed ``_position_ids`` slice path is not skipped by array truthiness.
"""

from __future__ import annotations

import logging
import textwrap

_logger = logging.getLogger(__name__)
_applied = False


def apply() -> None:
    """Apply all mlx_vlm compat patches (idempotent)."""
    global _applied
    _applied = True
    _patch_qwen3_vl_grid_thw()
    _patch_qwen35_patch_embed_layout()
    _patch_prompt_cache_rank3_trim()
    _patch_qwen35_language_mrope_none_delta()


class _Rank3KVTrimView:
    """Rank-4 trim facade for rank-3 mlx-vlm KV tensors.

    Upstream mlx-vlm checks ``keys.shape[2]`` and slices
    ``keys[:, :, :prefix_len, :]``. For rank-3 caches the sequence axis is 1,
    so this view exposes a rank-4-looking shape and remaps that slice to
    ``array[:, :prefix_len, :]``. The reported sequence length is one larger
    than the underlying tensor so upstream always assigns the real trimmed
    MLX array back to ``cache.keys``/``cache.values`` before model forward.
    """

    def __init__(self, array):
        self._array = array

    @property
    def shape(self):
        bsz, seq, hidden = self._array.shape
        return (bsz, 1, seq + 1, hidden)

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 4:
            batch_index, _head_index, seq_index, hidden_index = index
            return self._array[batch_index, seq_index, hidden_index]
        return self._array[index]


def _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm(cache):
    """Wrap rank-3 prompt-cache tensors for upstream mlx-vlm trim code."""
    if cache is None:
        return None
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if (
            keys is not None
            and values is not None
            and getattr(keys, "ndim", len(getattr(keys, "shape", ()))) == 3
            and getattr(values, "ndim", len(getattr(values, "shape", ()))) == 3
        ):
            wrapped_keys = _Rank3KVTrimView(keys)
            wrapped_values = _Rank3KVTrimView(values)
            layer_cache.keys = wrapped_keys
            layer_cache.values = wrapped_values
            state = getattr(layer_cache, "state", None)
            if state is not None and len(state) >= 2:
                offset = state[2] if len(state) >= 3 else getattr(layer_cache, "offset", 0)
                layer_cache.state = (wrapped_keys, wrapped_values, offset)
    return cache


def _vmlx_trim_prompt_cache(cache, prefix_len: int):
    """Rank-aware prompt-cache trim used by tests and local callers."""
    if cache is None:
        return None
    prefix_len = max(int(prefix_len or 0), 0)
    for layer_cache in cache:
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            continue
        ndim = getattr(keys, "ndim", len(getattr(keys, "shape", ())))
        if ndim >= 4:
            cached_len = keys.shape[2]
            if cached_len > prefix_len:
                layer_cache.keys = keys[:, :, :prefix_len, :]
                layer_cache.values = values[:, :, :prefix_len, :]
                if hasattr(layer_cache, "offset"):
                    layer_cache.offset = prefix_len
        elif ndim == 3:
            cached_len = keys.shape[1]
            if cached_len > prefix_len:
                layer_cache.keys = keys[:, :prefix_len, :]
                layer_cache.values = values[:, :prefix_len, :]
                if hasattr(layer_cache, "offset"):
                    layer_cache.offset = prefix_len
    return cache


def _vmlx_prime_qwen_mrope_for_full_prompt(model, input_ids, mask, kwargs) -> bool:
    """Prime Qwen/N2 mRoPE state before full multimodal cached prefill.

    Upstream already computes this state before cached-prefix suffix reuse. The
    same state is required when a fresh full image/video prompt is forwarded
    with a newly created prompt cache; otherwise Qwen language code can attempt
    ``cache_offset + None`` during first prefill.
    """
    lm = getattr(model, "language_model", None)
    get_rope_index = getattr(lm, "get_rope_index", None)
    if not callable(get_rope_index):
        return True
    if not (hasattr(lm, "_rope_deltas") or hasattr(lm, "_position_ids")):
        return True
    if kwargs.get("rope_deltas", None) is not None:
        if hasattr(lm, "_rope_deltas") and getattr(lm, "_rope_deltas", None) is None:
            lm._rope_deltas = kwargs["rope_deltas"]
        return True
    if kwargs.get("image_grid_thw", None) is None and kwargs.get("video_grid_thw", None) is None:
        return True
    try:
        position_ids, rope_deltas = get_rope_index(
            input_ids,
            kwargs.get("image_grid_thw", None),
            kwargs.get("video_grid_thw", None),
            mask,
        )
    except Exception as exc:
        _logger.warning("mlx_vlm_compat: could not prime Qwen mRoPE state: %s", exc)
        return False
    if hasattr(lm, "_position_ids"):
        lm._position_ids = position_ids
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = rope_deltas
    kwargs["rope_deltas"] = rope_deltas
    return True


def _patch_prompt_cache_rank3_trim() -> None:
    try:
        import importlib
        import inspect

        generate_mod = importlib.import_module("mlx_vlm.generate")
    except Exception:
        return
    if not hasattr(generate_mod, "_vmlx_trim_prompt_cache"):
        generate_mod._vmlx_trim_prompt_cache = _vmlx_trim_prompt_cache
    if not hasattr(generate_mod, "_vmlx_wrap_rank3_prompt_cache_for_mlx_vlm"):
        generate_mod._vmlx_wrap_rank3_prompt_cache_for_mlx_vlm = (
            _vmlx_wrap_rank3_prompt_cache_for_mlx_vlm
        )
    if not hasattr(generate_mod, "_vmlx_prime_qwen_mrope_for_full_prompt"):
        generate_mod._vmlx_prime_qwen_mrope_for_full_prompt = (
            _vmlx_prime_qwen_mrope_for_full_prompt
        )
    stream_generate = getattr(generate_mod, "stream_generate", None)
    if stream_generate is None or getattr(
        stream_generate, "_vmlx_rank3_cache_trim_patched", False
    ) and getattr(
        stream_generate, "_vmlx_mrope_full_prefill_patched", False
    ):
        return
    try:
        source = inspect.getsource(stream_generate)
    except Exception as exc:
        _logger.debug("mlx_vlm_compat: stream_generate source unavailable: %s", exc)
        return
    source = textwrap.dedent(source)
    start = source.find("# Reuse the saved KV cache (trimmed to prefix length)")
    end = source.find('kwargs["prompt_cache"] = kv_cache', start)
    if start < 0 or end < 0:
        _logger.debug("mlx_vlm_compat: stream_generate trim snippet not found")
        return
    end = source.find("\n", end)
    if end < 0:
        return
    old = source[start : end + 1]
    new = """\
# Reuse the saved KV cache (trimmed to prefix length)
            kv_cache = prompt_cache_state.cache
            _vmlx_trim_prompt_cache(kv_cache, prefix_len)
            kwargs["prompt_cache"] = kv_cache
"""
    patched = source[:start] + new + source[end + 1 :]
    needle = "    total_prompt_tokens = reused_prefix_len + input_ids.size\n"
    if needle not in patched:
        _logger.debug("mlx_vlm_compat: stream_generate full-prefill mRoPE insertion point not found")
        return
    patched = patched.replace(
        needle,
        "    _vmlx_prime_qwen_mrope_for_full_prompt(model, input_ids, mask, kwargs)\n"
        + needle,
        1,
    )
    namespace = dict(generate_mod.__dict__)
    namespace["_vmlx_trim_prompt_cache"] = _vmlx_trim_prompt_cache
    namespace["_vmlx_prime_qwen_mrope_for_full_prompt"] = (
        _vmlx_prime_qwen_mrope_for_full_prompt
    )
    exec(compile(patched, getattr(generate_mod, "__file__", "<mlx_vlm.generate>"), "exec"), namespace)
    patched_fn = namespace.get("stream_generate")
    if patched_fn is not None:
        patched_fn._vmlx_rank3_cache_trim_patched = True
        patched_fn._vmlx_mrope_full_prefill_patched = True
        generate_mod.stream_generate = patched_fn
        _logger.debug("mlx_vlm_compat: patched stream_generate VLM cache compatibility")


def _patch_qwen35_language_mrope_none_delta() -> None:
    try:
        import importlib
        import inspect
    except Exception:
        return

    module_names = (
        "mlx_vlm.models.qwen3_5.language",
        "mlx_vlm.models.qwen3_5_moe.language",
    )
    for module_name in module_names:
        try:
            language_mod = importlib.import_module(module_name)
        except Exception:
            continue
        LanguageModel = getattr(language_mod, "LanguageModel", None)
        if LanguageModel is None:
            continue
        original = getattr(LanguageModel, "__call__", None)
        if original is None or getattr(original, "_vmlx_mrope_none_delta_patched", False):
            continue
        try:
            source = textwrap.dedent(inspect.getsource(original))
        except Exception as exc:
            _logger.debug("mlx_vlm_compat: Qwen language source unavailable: %s", exc)
            continue
        old = """\
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
"""
        new = """\
                rope_deltas = (
                    rope_deltas_kw
                    if rope_deltas_kw is not None
                    else self._rope_deltas
                )
                if cache_offset is None:
                    cache_offset = 0
                if rope_deltas is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
                delta = mx.array(
                    cache_offset + rope_deltas if cache is not None else 0
                )
"""
        if old not in source:
            _logger.debug("mlx_vlm_compat: Qwen language delta snippet not found in %s", module_name)
            continue
        patched = source.replace(old, new, 1)
        offset_old = """\
                cache_offsets = mx.maximum(c0.offset, 0)

        # Check if mask shape matches input shape (for chunked prefill compatibility)
"""
        offset_new = """\
                cache_offsets = mx.maximum(c0.offset, 0)
            if cache_offset is None:
                cache_offset = 0
            if isinstance(cache_offset, mx.array) and cache_offset.size == 1:
                cache_offset = int(cache_offset.item())

        # Check if mask shape matches input shape (for chunked prefill compatibility)
"""
        if offset_old in patched:
            patched = patched.replace(offset_old, offset_new, 1)
        namespace = dict(language_mod.__dict__)
        try:
            exec(
                compile(
                    patched,
                    getattr(language_mod, "__file__", "<qwen_language>"),
                    "exec",
                ),
                namespace,
            )
        except Exception as exc:
            _logger.debug("mlx_vlm_compat: Qwen language patch compile failed: %s", exc)
            continue
        patched_fn = namespace.get("__call__")
        if patched_fn is None:
            continue
        patched_fn._vmlx_mrope_none_delta_patched = True
        LanguageModel.__call__ = patched_fn
    _logger.debug("mlx_vlm_compat: patched Qwen3.5/N2 language mRoPE delta fallback")


def _patch_qwen3_vl_grid_thw() -> None:
    try:
        import mlx.core as mx
        from mlx_vlm.models.qwen3_vl import vision as _qv
    except ImportError:
        return

    VisionModel = getattr(_qv, "VisionModel", None)
    if VisionModel is None:
        return

    def _as_mx(x):
        if isinstance(x, mx.array):
            return x
        try:
            return mx.array(x)
        except Exception:
            return x

    orig_rot = VisionModel.rot_pos_emb
    if not getattr(orig_rot, "_vmlx_patched", False):
        def rot_pos_emb(self, grid_thw):
            return orig_rot(self, _as_mx(grid_thw))
        rot_pos_emb._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.rot_pos_emb = rot_pos_emb  # type: ignore[assignment]

    orig_call = VisionModel.__call__
    if not getattr(orig_call, "_vmlx_patched", False):
        def __call__(self, hidden_states, grid_thw, **kwargs):
            return orig_call(self, hidden_states, _as_mx(grid_thw), **kwargs)
        __call__._vmlx_patched = True  # type: ignore[attr-defined]
        VisionModel.__call__ = __call__  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3-VL VisionModel grid_thw coercion")


def _qwen35_patch_embed_to_mlx_layout(key, value):
    if (
        str(key).endswith("patch_embed.proj.weight")
        and getattr(value, "ndim", None) == 5
        and int(value.shape[1]) in (1, 3)
        and int(value.shape[-1]) not in (1, 3)
    ):
        return value.transpose(0, 2, 3, 4, 1)
    return value


def _patch_qwen35_patch_embed_layout() -> None:
    try:
        from mlx_vlm.models.qwen3_5 import qwen3_5 as _qwen_vl
    except ImportError:
        _qwen_vl = None
    try:
        from mlx_vlm.models.qwen3_5_moe import qwen3_5_moe as _qwen_moe_vl
    except ImportError:
        _qwen_moe_vl = None

    for module in (_qwen_vl, _qwen_moe_vl):
        Model = getattr(module, "Model", None)
        if Model is None:
            continue
        original = getattr(Model, "sanitize", None)
        if original is None or getattr(original, "_vmlx_patch_embed_layout", False):
            continue

        def sanitize(self, weights, _original=original):
            fixed = {}
            for key, value in weights.items():
                fixed[key] = _qwen35_patch_embed_to_mlx_layout(key, value)
            return _original(self, fixed)

        sanitize._vmlx_patch_embed_layout = True  # type: ignore[attr-defined]
        Model.sanitize = sanitize  # type: ignore[assignment]

    _logger.debug("mlx_vlm_compat: patched Qwen3.5/3.6 patch_embed layout")
