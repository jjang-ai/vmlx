# SPDX-License-Identifier: Apache-2.0
"""Hybrid-aware TurboQuant live KV cache helpers."""

from __future__ import annotations

from typing import Any, Callable


QWEN36_HYBRID_MODEL_TYPES = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)

TURBOQUANT_MAKE_CACHE_NAMES = frozenset(
    {
        "_tq_make_cache",
        "_turboquant_make_cache",
        "_hybrid_turboquant_make_cache",
    }
)


def _model_types(config: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    stack = [config]
    while stack:
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        model_type = str(item.get("model_type") or "").lower()
        if model_type:
            values.add(model_type)
        for key in ("text_config", "language_config", "llm_config"):
            child = item.get(key)
            if isinstance(child, dict):
                stack.append(child)
    return values


def is_qwen36_hybrid_tq_supported(
    model_config: dict[str, Any],
    layer_types: list[str] | tuple[str, ...],
) -> bool:
    """Return True for Qwen3.6-style hybrid configs with attention + SSM slots."""
    if "ssm" not in layer_types:
        return False
    return bool(_model_types(model_config) & QWEN36_HYBRID_MODEL_TYPES)


def is_turboquant_make_cache(make_cache: Any) -> bool:
    return getattr(make_cache, "__name__", "") in TURBOQUANT_MAKE_CACHE_NAMES


def _is_kv_cache_slot(slot: Any) -> bool:
    return type(slot).__name__ in (
        "KVCache",
        "RotatingKVCache",
        "QuantizedKVCache",
        "TurboQuantKVCache",
    )


def build_hybrid_turboquant_make_cache(
    native_make_cache: Callable[[], list[Any]],
    tq_config: Any,
    key_dim: int,
    val_dim: int,
    layer_types: list[str],
):
    """Build a make_cache patch that TQs attention KV slots only.

    The native cache factory is still called on every invocation so all
    path-dependent hybrid companion state keeps its original class and
    initialization contract.
    """
    from jang_tools.turboquant.cache import TurboQuantKVCache

    n_layers = len(layer_types)
    attention_layers = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type == "attention"
    )
    companion_layers = tuple(
        i for i, layer_type in enumerate(layer_types) if layer_type != "attention"
    )

    def _hybrid_turboquant_make_cache(
        _native_make_cache=native_make_cache,
        _cfg=tq_config,
        _n=n_layers,
        _kd=key_dim,
        _vd=val_dim,
        _lt=tuple(layer_types),
    ):
        native_cache = list(_native_make_cache() or [])
        caches = []
        for i, slot in enumerate(native_cache):
            layer_type = _lt[i] if i < _n else "attention"
            if layer_type == "attention" and _is_kv_cache_slot(slot):
                caches.append(
                    TurboQuantKVCache(
                        key_dim=_kd,
                        value_dim=_vd,
                        key_bits=_cfg.key_bits_for_layer(i),
                        value_bits=_cfg.value_bits_for_layer(i),
                        seed=_cfg.seed + i,
                        sink_tokens=_cfg.sink_tokens,
                    )
                )
            else:
                caches.append(slot)
        return caches

    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_policy = "attention_kv_only"
    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_attention_layers = attention_layers
    _hybrid_turboquant_make_cache._vmlx_hybrid_tq_companion_layers = companion_layers
    return _hybrid_turboquant_make_cache
