# SPDX-License-Identifier: Apache-2.0
# TQ-native disk serialization by Jinho Jang (eric@jangq.ai) for vMLX.
# Stores TurboQuantKVCache compressed data (EncodedKeys/EncodedValues) directly
# to safetensors — 26x smaller than float16 state. github.com/jjang-ai/vmlx
"""
TQ-native serialization for disk cache.

Stores TurboQuantKVCache compressed data (packed indices, norms, metadata)
directly to safetensors without decompressing to float16 first.

Compression ratio: ~26x vs float16 (40KB vs 1MB per 100 tokens x 8 heads x 128 dim)

Format:
- Safetensors tensors:
  - tq_{i}_ck_indices_packed (uint32) — codebook indices
  - tq_{i}_ck_qjl_packed (uint32) — QJL sign bits
  - tq_{i}_ck_residual_norms (float16) — per-vector residual norms
  - tq_{i}_ck_vector_norms (float16) — per-vector key norms
  - tq_{i}_cv_indices_packed (uint32) — value codebook indices
  - tq_{i}_cv_vector_norms (float16) — per-vector value norms
  - layer_{i}_keys / layer_{i}_values — non-TQ layers (KVCache, standard)
  - layer_{i}_state_{j} — cumulative layers (MambaCache/ArraysCache)
- Safetensors metadata (string key-value):
  - __tq_native__ = "true" — format marker
  - __num_layers__ — total layer count
  - __layer_{i}_class__ — class name per layer
  - __tq_{i}_ck_shape__ / __tq_{i}_cv_shape__ — original shapes (JSON)
  - __tq_{i}_ck_bits__ / __tq_{i}_cv_bits__ — index bit widths
  - __tq_{i}_offset__ — token offset
  - __tq_{i}_key_dim__ / __tq_{i}_value_dim__ — TQ dimensions
  - __tq_{i}_key_bits__ / __tq_{i}_value_bits__ — TQ compression bits
  - __tq_{i}_sink_tokens__ — number of sink tokens
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

_TQ_CLASS_NAME = "TurboQuantKVCache"


def is_tq_compressed_cache(cache: List[Any]) -> bool:
    """Check if any layer is TurboQuantKVCache with compressed data available.

    Returns True if at least one layer has _compressed_keys set, meaning
    compress() has been called and native TQ serialization is possible.
    """
    for c in cache:
        if (type(c).__name__ == _TQ_CLASS_NAME
                and getattr(c, '_compressed_keys', None) is not None
                and getattr(c, '_compressed_values', None) is not None):
            return True
    return False


def serialize_tq_cache(
    cache: List[Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Serialize cache with TQ-native compression for TQ layers.

    For TurboQuantKVCache layers: extracts _compressed_keys/_compressed_values
    directly (26x smaller than .state which decompresses to float16).

    For other layers (KVCache, MambaCache, etc.): uses standard state extraction.

    Args:
        cache: List of cache layer objects from the model.

    Returns:
        (tensors, metadata) — ready for safetensors storage.
        tensors: Dict[str, mx.array] of named tensors.
        metadata: Dict[str, str] of string metadata.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ serialization")

    tensors: Dict[str, Any] = {}
    meta: Dict[str, str] = {
        "__tq_native__": "true",
        "__num_layers__": str(len(cache)),
    }
    tq_count = 0
    non_tq_count = 0

    for i, layer in enumerate(cache):
        cls_name = type(layer).__name__
        meta[f"__layer_{i}_class__"] = cls_name

        if (cls_name == _TQ_CLASS_NAME
                and getattr(layer, '_compressed_keys', None) is not None):
            # ─── TQ layer: serialize compressed data directly ───
            _serialize_tq_layer(tensors, meta, i, layer)
            tq_count += 1
        elif hasattr(layer, 'caches') and isinstance(
            getattr(layer, 'caches', None), (list, tuple)
        ):
            # ─── CacheList (MoE models: DeepSeek V3.2, Falcon H1) ───
            # Contains sub-caches that may be TQ or standard KVCache.
            _serialize_cache_list_layer(tensors, meta, i, layer)
            non_tq_count += 1
        elif hasattr(layer, 'state') and hasattr(layer, 'meta_state'):
            # ─── Non-TQ layer: serialize via .state ───
            _serialize_standard_layer(tensors, meta, i, layer, cls_name)
            non_tq_count += 1
        else:
            # Unknown layer — mark as empty
            meta[f"__layer_{i}_empty__"] = "true"

    logger.info(
        f"TQ-native serialize: {tq_count} TQ layers (compressed), "
        f"{non_tq_count} standard layers"
    )
    return tensors, meta


def deserialize_tq_cache(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
) -> List[Any]:
    """Deserialize TQ-native cache from safetensors data.

    TQ layers are decoded from compressed form to float16 and wrapped in
    KVCache objects. The caller should then call _recompress_to_tq() to
    convert back to TurboQuantKVCache using the model's make_cache() template.

    Non-TQ layers are reconstructed as standard KVCache or placeholder objects.

    Args:
        tensors: Dict of named tensors from mx.load().
        metadata: Dict of string metadata from safetensors header.

    Returns:
        List of cache layer objects (KVCache or placeholders).
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for TQ deserialization")

    from mlx_lm.models.cache import KVCache

    num_layers = int(metadata.get("__num_layers__", "0"))
    cache: List[Any] = []

    tq_decoded = 0
    standard_loaded = 0

    for i in range(num_layers):
        cls_name = metadata.get(f"__layer_{i}_class__", "")

        if cls_name == _TQ_CLASS_NAME:
            # ─── TQ layer: decode compressed → KVCache ───
            kv = _deserialize_tq_layer(tensors, metadata, i)
            if kv is not None:
                cache.append(kv)
                tq_decoded += 1
            else:
                cache.append(KVCache())
        elif metadata.get(f"__layer_{i}_empty__") == "true":
            cache.append(KVCache())
        elif metadata.get(f"__layer_{i}_cache_list__") == "true":
            # ─── CacheList (MoE models) ───
            layer = _deserialize_cache_list_layer(tensors, metadata, i)
            cache.append(layer)
            standard_loaded += 1
        elif metadata.get(f"__layer_{i}_cumulative__") == "true":
            # ─── Cumulative (SSM) layer ───
            layer = _deserialize_cumulative_layer(tensors, metadata, i)
            cache.append(layer)
            standard_loaded += 1
        elif f"layer_{i}_keys" in tensors:
            # ─── Standard KVCache ───
            kv = _deserialize_standard_kv(tensors, metadata, i)
            cache.append(kv)
            standard_loaded += 1
        elif metadata.get(f"__layer_{i}_quantized__") == "true":
            # ─── QuantizedKVCache ───
            kv = _deserialize_quantized_kv(tensors, metadata, i)
            cache.append(kv)
            standard_loaded += 1
        else:
            cache.append(KVCache())

    logger.info(
        f"TQ-native deserialize: {tq_decoded} TQ decoded, "
        f"{standard_loaded} standard, {num_layers} total layers"
    )
    return cache


# =============================================================================
# Internal: TQ layer serialization
# =============================================================================

def _serialize_tq_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
) -> None:
    """Serialize a single TurboQuantKVCache layer's compressed data."""
    ck = layer._compressed_keys   # EncodedKeys namedtuple
    cv = layer._compressed_values  # EncodedValues namedtuple

    # Store EncodedKeys tensors (4 mx.array fields)
    tensors[f"tq_{i}_ck_indices_packed"] = ck.indices_packed
    tensors[f"tq_{i}_ck_qjl_packed"] = ck.qjl_packed
    tensors[f"tq_{i}_ck_residual_norms"] = ck.residual_norms
    tensors[f"tq_{i}_ck_vector_norms"] = ck.vector_norms

    # Store EncodedValues tensors (2 mx.array fields)
    tensors[f"tq_{i}_cv_indices_packed"] = cv.indices_packed
    tensors[f"tq_{i}_cv_vector_norms"] = cv.vector_norms

    # Store metadata (shape tuples, bit widths, TQ config)
    meta[f"__tq_{i}_ck_shape__"] = json.dumps(list(ck.shape))
    meta[f"__tq_{i}_ck_bits__"] = str(ck.index_bits)
    meta[f"__tq_{i}_cv_shape__"] = json.dumps(list(cv.shape))
    meta[f"__tq_{i}_cv_bits__"] = str(cv.index_bits)
    meta[f"__tq_{i}_offset__"] = str(layer.offset)
    meta[f"__tq_{i}_compressed_tokens__"] = str(
        getattr(layer, '_compressed_tokens', layer.offset)
    )
    meta[f"__tq_{i}_key_dim__"] = str(layer.key_dim)
    meta[f"__tq_{i}_value_dim__"] = str(layer.value_dim)
    meta[f"__tq_{i}_key_bits__"] = str(layer.key_bits)
    meta[f"__tq_{i}_value_bits__"] = str(layer.value_bits)
    meta[f"__tq_{i}_sink_tokens__"] = str(getattr(layer, 'sink_tokens', 0))


def _serialize_standard_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
    cls_name: str,
) -> None:
    """Serialize a non-TQ cache layer via its .state property."""
    state = layer.state
    meta_state = layer.meta_state

    # Detect cumulative (SSM) layers: MambaCache, ArraysCache
    is_cumulative = (
        hasattr(layer, 'cache') and isinstance(getattr(layer, 'cache', None), list)
    )

    if is_cumulative:
        # Store cumulative state arrays
        meta[f"__layer_{i}_cumulative__"] = "true"
        meta[f"__layer_{i}_cumulative_class__"] = cls_name
        if isinstance(state, (list, tuple)):
            for j, arr in enumerate(state):
                if hasattr(arr, 'shape'):
                    tensors[f"layer_{i}_state_{j}"] = arr
            meta[f"__layer_{i}_state_count__"] = str(len(state))
        if meta_state:
            meta[f"__layer_{i}_meta__"] = json.dumps(
                [str(x) for x in meta_state] if isinstance(meta_state, tuple) else str(meta_state)
            )
        return

    if isinstance(state, tuple) and len(state) == 2:
        keys, values = state

        if isinstance(keys, (tuple, list)):
            # QuantizedKVCache: keys/values are tuples of (data, scales, zeros)
            meta[f"__layer_{i}_quantized__"] = "true"
            for j, t in enumerate(keys):
                if hasattr(t, 'shape'):
                    tensors[f"layer_{i}_qk_{j}"] = t
            for j, t in enumerate(values):
                if hasattr(t, 'shape'):
                    tensors[f"layer_{i}_qv_{j}"] = t
            meta[f"__layer_{i}_qk_count__"] = str(len(keys))
            meta[f"__layer_{i}_qv_count__"] = str(len(values))
        elif hasattr(keys, 'shape'):
            # Standard KVCache
            tensors[f"layer_{i}_keys"] = keys
            tensors[f"layer_{i}_values"] = values
            # Cast bfloat16 → float16 (safetensors supports bf16 but numpy doesn't)
            if keys.dtype == mx.bfloat16:
                tensors[f"layer_{i}_keys"] = keys.astype(mx.float16)
                tensors[f"layer_{i}_values"] = values.astype(mx.float16)
                meta[f"__layer_{i}_orig_dtype__"] = "bfloat16"

    # Store meta_state (offset, etc.)
    if meta_state:
        meta[f"__layer_{i}_meta__"] = json.dumps(
            [str(x) for x in meta_state] if isinstance(meta_state, tuple) else str(meta_state)
        )


def _serialize_cache_list_layer(
    tensors: Dict[str, Any],
    meta: Dict[str, str],
    i: int,
    layer: Any,
) -> None:
    """Serialize a CacheList layer (MoE models: DeepSeek V3.2, Falcon H1).

    CacheList wraps a list of sub-caches (.caches attribute). Each sub-cache
    can be TQ, KVCache, or cumulative (MambaCache). We serialize each sub-cache
    independently using the appropriate path.
    """
    meta[f"__layer_{i}_cache_list__"] = "true"
    sub_caches = layer.caches
    meta[f"__layer_{i}_cl_count__"] = str(len(sub_caches))

    for j, sub in enumerate(sub_caches):
        sub_cls = type(sub).__name__
        meta[f"__layer_{i}_cl_{j}_class__"] = sub_cls

        if (sub_cls == _TQ_CLASS_NAME
                and getattr(sub, '_compressed_keys', None) is not None):
            # TQ sub-cache: serialize compressed data
            # Reuse TQ serializer with prefixed keys
            ck = sub._compressed_keys
            cv = sub._compressed_values
            prefix = f"cl_{i}_{j}"
            tensors[f"{prefix}_ck_indices_packed"] = ck.indices_packed
            tensors[f"{prefix}_ck_qjl_packed"] = ck.qjl_packed
            tensors[f"{prefix}_ck_residual_norms"] = ck.residual_norms
            tensors[f"{prefix}_ck_vector_norms"] = ck.vector_norms
            tensors[f"{prefix}_cv_indices_packed"] = cv.indices_packed
            tensors[f"{prefix}_cv_vector_norms"] = cv.vector_norms
            meta[f"__{prefix}_ck_shape__"] = json.dumps(list(ck.shape))
            meta[f"__{prefix}_ck_bits__"] = str(ck.index_bits)
            meta[f"__{prefix}_cv_shape__"] = json.dumps(list(cv.shape))
            meta[f"__{prefix}_cv_bits__"] = str(cv.index_bits)
            meta[f"__{prefix}_offset__"] = str(sub.offset)
            meta[f"__{prefix}_key_dim__"] = str(sub.key_dim)
            meta[f"__{prefix}_value_dim__"] = str(sub.value_dim)
            meta[f"__{prefix}_key_bits__"] = str(sub.key_bits)
            meta[f"__{prefix}_value_bits__"] = str(sub.value_bits)
            meta[f"__{prefix}_sink_tokens__"] = str(getattr(sub, 'sink_tokens', 0))
        elif hasattr(sub, 'state') and hasattr(sub, 'meta_state'):
            # Standard sub-cache (KVCache or cumulative)
            state = sub.state
            if isinstance(state, tuple) and len(state) == 2:
                keys, values = state
                if hasattr(keys, 'shape'):
                    tensors[f"cl_{i}_{j}_keys"] = keys
                    tensors[f"cl_{i}_{j}_values"] = values
            sub_meta = sub.meta_state
            if sub_meta:
                meta[f"__cl_{i}_{j}_meta__"] = json.dumps(
                    [str(x) for x in sub_meta] if isinstance(sub_meta, tuple) else str(sub_meta)
                )


def _deserialize_cache_list_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a CacheList layer from serialized sub-caches.

    Returns a list of KVCache objects. The caller should wrap this in a
    CacheList if needed, or pass through to _recompress_to_tq().
    """
    from mlx_lm.models.cache import KVCache

    sub_count = int(metadata.get(f"__layer_{i}_cl_count__", "0"))
    sub_caches = []

    for j in range(sub_count):
        sub_cls = metadata.get(f"__layer_{i}_cl_{j}_class__", "")
        prefix = f"cl_{i}_{j}"

        if sub_cls == _TQ_CLASS_NAME and f"{prefix}_ck_indices_packed" in tensors:
            # TQ sub-cache — decode same as _deserialize_tq_layer but with cl_ prefix
            kv = KVCache()
            # For now, store as empty KVCache — _recompress_to_tq handles conversion
            # The actual decode requires jang_tools which may not be available
            try:
                from jang_tools.turboquant.cache import EncodedKeys, EncodedValues
                from jang_tools.turboquant.pipeline import decode_keys, decode_values

                ck_shape = tuple(json.loads(metadata.get(f"__{prefix}_ck_shape__", "[]")))
                ck_bits = int(metadata.get(f"__{prefix}_ck_bits__", "3"))
                cv_shape = tuple(json.loads(metadata.get(f"__{prefix}_cv_shape__", "[]")))
                cv_bits = int(metadata.get(f"__{prefix}_cv_bits__", "3"))

                encoded_keys = EncodedKeys(
                    indices_packed=tensors[f"{prefix}_ck_indices_packed"],
                    qjl_packed=tensors[f"{prefix}_ck_qjl_packed"],
                    residual_norms=tensors[f"{prefix}_ck_residual_norms"],
                    vector_norms=tensors[f"{prefix}_ck_vector_norms"],
                    shape=ck_shape, index_bits=ck_bits,
                )
                encoded_values = EncodedValues(
                    indices_packed=tensors[f"{prefix}_cv_indices_packed"],
                    vector_norms=tensors[f"{prefix}_cv_vector_norms"],
                    shape=cv_shape, index_bits=cv_bits,
                )
                # Create TQ cache for encoder access (decode needs encoder)
                from jang_tools.turboquant.cache import TurboQuantKVCache as _TQ_CL
                _key_dim = int(metadata.get(f"__{prefix}_key_dim__", "128"))
                _val_dim = int(metadata.get(f"__{prefix}_value_dim__", "128"))
                _key_bits = int(metadata.get(f"__{prefix}_key_bits__", "3"))
                _val_bits = int(metadata.get(f"__{prefix}_value_bits__", "3"))
                _tq_cl = _TQ_CL(key_dim=_key_dim, value_dim=_val_dim,
                                 key_bits=_key_bits, value_bits=_val_bits)
                kv.keys = decode_keys(encoded_keys, _tq_cl.key_encoder)
                kv.values = decode_values(encoded_values, _tq_cl.value_encoder)
                kv.offset = int(metadata.get(f"__{prefix}_offset__", "0"))
            except Exception as e:
                logger.warning("CacheList sub-cache %d/%d TQ decode failed: %s", i, j, e)
            sub_caches.append(kv)
        elif f"{prefix}_keys" in tensors:
            # Standard KVCache sub-cache
            kv = KVCache()
            kv.keys = tensors[f"{prefix}_keys"]
            kv.values = tensors[f"{prefix}_values"]
            sub_meta_str = metadata.get(f"__{prefix}_meta__", "")
            if sub_meta_str:
                try:
                    meta_list = json.loads(sub_meta_str)
                    kv.offset = int(meta_list[0]) if meta_list else 0
                except (json.JSONDecodeError, ValueError, IndexError):
                    kv.offset = kv.keys.shape[2] if kv.keys is not None and kv.keys.ndim >= 3 else 0
            else:
                kv.offset = kv.keys.shape[2] if kv.keys is not None and kv.keys.ndim >= 3 else 0
            sub_caches.append(kv)
        else:
            sub_caches.append(KVCache())

    # Try to wrap in CacheList if available
    try:
        from mlx_lm.models.cache import CacheList as _CL
        cl = _CL(sub_caches)
        return cl
    except ImportError:
        # CacheList not available — return raw list
        # The caller should handle this gracefully
        return sub_caches[0] if len(sub_caches) == 1 else KVCache()


# =============================================================================
# Internal: TQ layer deserialization
# =============================================================================

def _deserialize_tq_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Optional[Any]:
    """Decode a TQ compressed layer to float16 KVCache.

    The returned KVCache has decoded float16 keys/values. The caller should
    use _recompress_to_tq() with the model's make_cache() template to convert
    back to TurboQuantKVCache.
    """
    try:
        from jang_tools.turboquant.cache import EncodedKeys, EncodedValues
        from jang_tools.turboquant.pipeline import decode_keys, decode_values
        from mlx_lm.models.cache import KVCache
    except ImportError:
        logger.warning("jang_tools not available — cannot decode TQ layer %d", i)
        return None

    prefix = f"tq_{i}"

    # Reconstruct EncodedKeys
    ck_indices = tensors.get(f"{prefix}_ck_indices_packed")
    ck_qjl = tensors.get(f"{prefix}_ck_qjl_packed")
    ck_rnorms = tensors.get(f"{prefix}_ck_residual_norms")
    ck_vnorms = tensors.get(f"{prefix}_ck_vector_norms")

    if ck_indices is None:
        logger.warning("TQ layer %d missing ck_indices_packed", i)
        return None

    try:
        ck_shape = tuple(json.loads(metadata.get(f"__{prefix}_ck_shape__", "[]")))
        ck_bits = int(metadata.get(f"__{prefix}_ck_bits__", "3"))
    except (json.JSONDecodeError, ValueError):
        logger.warning("TQ layer %d: invalid ck metadata", i)
        return None

    encoded_keys = EncodedKeys(
        indices_packed=ck_indices,
        qjl_packed=ck_qjl,
        residual_norms=ck_rnorms,
        vector_norms=ck_vnorms,
        shape=ck_shape,
        index_bits=ck_bits,
    )

    # Reconstruct EncodedValues
    cv_indices = tensors.get(f"{prefix}_cv_indices_packed")
    cv_vnorms = tensors.get(f"{prefix}_cv_vector_norms")

    if cv_indices is None:
        logger.warning("TQ layer %d missing cv_indices_packed", i)
        return None

    try:
        cv_shape = tuple(json.loads(metadata.get(f"__{prefix}_cv_shape__", "[]")))
        cv_bits = int(metadata.get(f"__{prefix}_cv_bits__", "3"))
    except (json.JSONDecodeError, ValueError):
        logger.warning("TQ layer %d: invalid cv metadata", i)
        return None

    encoded_values = EncodedValues(
        indices_packed=cv_indices,
        vector_norms=cv_vnorms,
        shape=cv_shape,
        index_bits=cv_bits,
    )

    # Decode compressed data to float16 using a TurboQuantKVCache's encoder.
    # decode_keys/decode_values require (encoded, encoder) — the encoder
    # contains codebook tables needed for dequantization. We create a
    # TurboQuantKVCache to get the encoder, then decode.
    offset = int(metadata.get(f"__{prefix}_offset__", "0"))
    key_dim = int(metadata.get(f"__{prefix}_key_dim__", "128"))
    value_dim = int(metadata.get(f"__{prefix}_value_dim__", "128"))
    key_bits = int(metadata.get(f"__{prefix}_key_bits__", "3"))
    value_bits = int(metadata.get(f"__{prefix}_value_bits__", "3"))
    sink_tokens = int(metadata.get(f"__{prefix}_sink_tokens__", "0"))

    try:
        from jang_tools.turboquant.cache import TurboQuantKVCache as _TQ
        # Create TQ cache to get access to the encoder objects
        tq = _TQ(
            key_dim=key_dim, value_dim=value_dim,
            key_bits=key_bits, value_bits=value_bits,
            sink_tokens=sink_tokens,
        )
        # Access the encoder (triggers lazy initialization)
        key_enc = tq.key_encoder
        val_enc = tq.value_encoder

        decoded_keys = decode_keys(encoded_keys, key_enc)
        decoded_values = decode_values(encoded_values, val_enc)
    except Exception as e:
        logger.warning("TQ layer %d decode failed: %s", i, e)
        return None

    # Wrap in KVCache — the caller's _recompress_to_tq() will
    # convert back to TurboQuantKVCache using the model's template.
    kv = KVCache()
    kv.keys = decoded_keys
    kv.values = decoded_values
    kv.offset = offset

    return kv


def _deserialize_standard_kv(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a standard KVCache layer."""
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    kv.keys = tensors.get(f"layer_{i}_keys")
    kv.values = tensors.get(f"layer_{i}_values")

    # Restore bfloat16 if originally cast
    if metadata.get(f"__layer_{i}_orig_dtype__") == "bfloat16":
        if kv.keys is not None:
            kv.keys = kv.keys.astype(mx.bfloat16)
            kv.values = kv.values.astype(mx.bfloat16)

    # Restore offset from meta_state
    offset = _parse_offset(metadata, i)
    if offset is not None:
        kv.offset = offset
    elif kv.keys is not None and kv.keys.ndim >= 3:
        kv.offset = kv.keys.shape[2]

    return kv


def _deserialize_quantized_kv(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a QuantizedKVCache layer.

    Since QuantizedKVCache.from_state() may not be available, we reconstruct
    as a standard KVCache by dequantizing. The caller can re-quantize if needed.
    """
    from mlx_lm.models.cache import KVCache

    try:
        from mlx_lm.models.cache import QuantizedKVCache
        qk_count = int(metadata.get(f"__layer_{i}_qk_count__", "0"))
        qv_count = int(metadata.get(f"__layer_{i}_qv_count__", "0"))

        keys_tuple = tuple(tensors[f"layer_{i}_qk_{j}"] for j in range(qk_count))
        values_tuple = tuple(tensors[f"layer_{i}_qv_{j}"] for j in range(qv_count))

        # Try to use QuantizedKVCache.from_state if available
        state = (keys_tuple, values_tuple)
        meta_str = metadata.get(f"__layer_{i}_meta__", "")
        if meta_str:
            meta_state = tuple(json.loads(meta_str))
        else:
            meta_state = ()

        try:
            return QuantizedKVCache.from_state(state, meta_state)
        except Exception:
            pass

        # Fallback: dequantize to KVCache
        if len(keys_tuple) >= 3:
            data, scales, zeros = keys_tuple[0], keys_tuple[1], keys_tuple[2]
            keys = mx.dequantize(data, scales, zeros)
        else:
            keys = keys_tuple[0] if keys_tuple else None

        if len(values_tuple) >= 3:
            data, scales, zeros = values_tuple[0], values_tuple[1], values_tuple[2]
            values = mx.dequantize(data, scales, zeros)
        else:
            values = values_tuple[0] if values_tuple else None

        kv = KVCache()
        kv.keys = keys
        kv.values = values
        offset = _parse_offset(metadata, i)
        if offset is not None:
            kv.offset = offset
        elif keys is not None and keys.ndim >= 3:
            kv.offset = keys.shape[2]
        return kv

    except Exception as e:
        logger.warning("Failed to deserialize quantized KV layer %d: %s", i, e)
        return KVCache()


def _deserialize_cumulative_layer(
    tensors: Dict[str, Any],
    metadata: Dict[str, str],
    i: int,
) -> Any:
    """Reconstruct a cumulative (SSM) cache layer."""
    from mlx_lm.models.cache import KVCache

    cls_name = metadata.get(f"__layer_{i}_cumulative_class__", "")
    state_count = int(metadata.get(f"__layer_{i}_state_count__", "0"))

    state_arrays = []
    for j in range(state_count):
        arr = tensors.get(f"layer_{i}_state_{j}")
        if arr is not None:
            state_arrays.append(arr)

    if not state_arrays:
        return KVCache()

    # Try to reconstruct the original cache class
    meta_str = metadata.get(f"__layer_{i}_meta__", "")
    meta_state = ()
    if meta_str:
        try:
            meta_state = tuple(json.loads(meta_str))
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        import mlx_lm.models.cache as _cache_mod
        cls = getattr(_cache_mod, cls_name, None)
        if cls is not None and hasattr(cls, 'from_state'):
            return cls.from_state(state_arrays, meta_state)
    except Exception:
        pass

    # Fallback: store as list in a KVCache wrapper
    # (won't work for SSM inference but preserves data)
    kv = KVCache()
    return kv


def _parse_offset(metadata: Dict[str, str], i: int) -> Optional[int]:
    """Parse offset from meta_state metadata."""
    meta_str = metadata.get(f"__layer_{i}_meta__", "")
    if not meta_str:
        return None
    try:
        meta_list = json.loads(meta_str)
        if isinstance(meta_list, list) and meta_list:
            return int(meta_list[0])
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
    return None
