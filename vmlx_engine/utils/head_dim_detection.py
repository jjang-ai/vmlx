# SPDX-License-Identifier: Apache-2.0
"""Model config traversal helpers for KV cache shape validation."""

from __future__ import annotations

from typing import Any, Optional, Tuple


MLX_QUANTIZED_KV_GROUP_SIZES = (128, 64, 32)
_MLA_EXPANDED_KV_MODEL_TYPES = {"bailing_hybrid", "bailing_moe_v2_5"}


def read_config_field(obj: Any, field: str) -> Any:
    """Read a field from object-style or dict-style model configs."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def positive_int_or_none(value: Any) -> Optional[int]:
    """Return a positive int for config values that are safe to use as dims."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def iter_config_candidates(model: Any, *, max_items: int = 32) -> list[Any]:
    """Return model args/config/text_config candidates across common wrappers."""
    objects: list[Any] = []
    configs: list[Any] = []
    seen_objects: set[int] = set()
    seen_configs: set[int] = set()

    def add_object(obj: Any) -> None:
        if obj is None:
            return
        marker = id(obj)
        if marker in seen_objects:
            return
        seen_objects.add(marker)
        objects.append(obj)

    def add_config(cfg: Any) -> None:
        if cfg is None:
            return
        marker = id(cfg)
        if marker in seen_configs:
            return
        seen_configs.add(marker)
        configs.append(cfg)

    add_object(model)
    index = 0
    while index < len(objects) and index < max_items:
        obj = objects[index]
        index += 1
        for attr in ("args", "config", "text_config"):
            cfg = read_config_field(obj, attr)
            add_config(cfg)
            raw = read_config_field(cfg, "_raw_config")
            if isinstance(raw, dict):
                add_config(raw)
            add_config(read_config_field(cfg, "text_config"))
        add_object(read_config_field(obj, "language_model"))
        add_object(read_config_field(obj, "model"))

    config_index = 0
    while config_index < len(configs) and config_index < max_items:
        cfg = configs[config_index]
        config_index += 1
        raw = read_config_field(cfg, "_raw_config")
        if isinstance(raw, dict):
            add_config(raw)
        add_config(read_config_field(cfg, "text_config"))

    return configs


def _config_model_type(cfg: Any) -> str:
    model_type = read_config_field(cfg, "model_type")
    return str(model_type or "").lower()


def detect_cache_head_dims(model: Any) -> Tuple[int, ...]:
    """Detect cache tensor trailing dims for KV quant group-size validation.

    Standard attention caches use a single head_dim. MLA-family caches in
    mlx-lm store two different tensors: latent KV with width ``kv_lora_rank``
    and RoPE keys with width ``qk_rope_head_dim``. Validating only
    hidden_size / num_attention_heads is wrong for Kimi/DeepSeek/Mistral MLA.
    """
    configs = iter_config_candidates(model)

    for cfg in configs:
        model_type = _config_model_type(cfg)
        kv_lora_rank = positive_int_or_none(read_config_field(cfg, "kv_lora_rank"))
        if kv_lora_rank and model_type not in _MLA_EXPANDED_KV_MODEL_TYPES:
            dims = [kv_lora_rank]
            qk_rope_head_dim = positive_int_or_none(
                read_config_field(cfg, "qk_rope_head_dim")
            )
            if qk_rope_head_dim and qk_rope_head_dim not in dims:
                dims.append(qk_rope_head_dim)
            return tuple(dims)

    for cfg in configs:
        dims: list[int] = []
        for field in ("head_dim", "global_head_dim"):
            dim = positive_int_or_none(read_config_field(cfg, field))
            if dim and dim not in dims:
                dims.append(dim)
        if dims:
            return tuple(dims)

    for cfg in configs:
        hidden = positive_int_or_none(read_config_field(cfg, "hidden_size"))
        n_heads = positive_int_or_none(read_config_field(cfg, "num_attention_heads"))
        if hidden and n_heads:
            return (hidden // n_heads,)

    return ()


def choose_supported_kv_group_size(
    cache_head_dims: Tuple[int, ...],
    requested_group_size: int,
) -> Optional[int]:
    """Return a supported group size that divides every detected cache dim."""
    if not cache_head_dims:
        return requested_group_size
    if (
        requested_group_size in MLX_QUANTIZED_KV_GROUP_SIZES
        and all(dim % requested_group_size == 0 for dim in cache_head_dims)
    ):
        return requested_group_size
    for candidate in (64, 32, 128):
        if all(dim % candidate == 0 for dim in cache_head_dims):
            return candidate
    return None
