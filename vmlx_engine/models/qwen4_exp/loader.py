"""Memory-bounded qwen4_exp VLM loader with SSD-backed quantized PLE."""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vmlx_engine.utils.jang_affine_storage import (
    expand_affine1_shard_mlx,
    prepare_affine1_runtime_config,
)

from .register import register_qwen4_exp_runtime
from .table_reader import (
    FileBackedQuantizedNGramTable,
    resolve_jang_bit_map_spec,
)

logger = logging.getLogger("vmlx_engine")

_PLE_TABLE_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"ngram_embedding\.(?P<layout>shard_|shards\.)(?P<shard>\d+)\."
    r"(?P<suffix>weight|scales|biases)$"
)
_PLE_BUFFER_RE = re.compile(
    r"(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?"
    r"(?P<suffix>layer_multipliers|ngram_heads_vocab_sizes|"
    r"ngram_heads_offsets)$"
)
_PLE_REQUIRED_SHARD_SUFFIXES = ("weight", "scales", "biases")
_HYPER_UP_SUFFIX = ".input_mix_weight_up.weight"
_LOW_PRECISION_FLOAT_DTYPES = {mx.float16, mx.bfloat16}
_MLX_AFFINE_BITS = {2, 3, 4, 5, 6, 8}


def is_qwen4_exp_bundle(model_path: str | Path) -> bool:
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return False
    model_types = {
        str(cfg.get("model_type") or "").lower(),
        str((cfg.get("text_config") or {}).get("model_type") or "").lower(),
    }
    return bool(model_types.intersection({"qwen4_exp", "qwen4_exp_text"}))


def _load_runtime_config(
    model_path: Path,
) -> tuple[dict, frozenset[str], dict[str, object] | None, bool]:
    config = json.loads((model_path / "config.json").read_text())
    embedded = config.get("jang_config")
    if embedded is None:
        embedded = config.get("jang")
    if embedded is not None and not isinstance(embedded, dict):
        raise ValueError("qwen4_exp embedded JANG metadata must be an object")
    jang_path = model_path / "jang_config.json"
    sidecar = json.loads(jang_path.read_text()) if jang_path.is_file() else None
    if sidecar is not None and not isinstance(sidecar, dict):
        raise ValueError("qwen4_exp jang_config.json must contain an object")
    if sidecar is not None and embedded is not None and sidecar != embedded:
        raise ValueError(
            "qwen4_exp sidecar and embedded JANG metadata disagree"
        )
    jang_config = sidecar if sidecar is not None else embedded
    if jang_config is None:
        return config, frozenset(), None, False

    norm_convention = str(jang_config.get("norm_convention") or "").strip()
    if norm_convention != "runtime_plus1_applied":
        raise ValueError(
            "qwen4_exp JANG weights require norm_convention="
            "runtime_plus1_applied"
        )

    runtime_config, affine1_modules = prepare_affine1_runtime_config(
        config, jang_config
    )
    mtp = jang_config.get("mtp") if isinstance(jang_config.get("mtp"), dict) else {}
    mtp_mode = str(mtp.get("mtp_mode") or "").strip().lower()
    if mtp_mode in {"none", "absent", "disabled", "off"}:
        # Upstream Qwen4Exp configs describe the architecture that was trained,
        # so ``text_config.mtp_num_hidden_layers`` may remain non-zero after a
        # converter intentionally omits the optional head.  The emitted bundle
        # contract is authoritative for construction.  Keep strict=True below:
        # if a bundle stamps no MTP but still contains mtp.* tensors, loading
        # must fail on those unexpected weights instead of silently discarding
        # them.
        runtime_config = copy.deepcopy(runtime_config)
        for container in (
            runtime_config,
            runtime_config.get("text_config"),
        ):
            if not isinstance(container, dict):
                continue
            for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
                if key in container:
                    container[key] = 0
    raw_bit_map = jang_config.get("bit_map")
    if raw_bit_map is None:
        return runtime_config, affine1_modules, None, True
    if not isinstance(raw_bit_map, Mapping):
        raise ValueError("qwen4_exp JANG bit_map must be an object")
    bit_map = copy.deepcopy(dict(raw_bit_map))
    default_spec = resolve_jang_bit_map_spec("__vmlx_default__", bit_map)
    quantization = runtime_config.get("quantization")
    if quantization is None:
        quantization = {}
    elif not isinstance(quantization, dict):
        raise ValueError("qwen4_exp config quantization must be an object")
    else:
        quantization = copy.deepcopy(quantization)
    for key in ("bits", "group_size", "mode"):
        quantization.setdefault(key, default_spec[key])
    runtime_config["quantization"] = quantization
    return runtime_config, affine1_modules, bit_map, True


def _normalize_jang_hyper_fold_dtypes(
    weights: dict[str, mx.array],
    *,
    target_dtype=None,
) -> tuple[int, int]:
    """Undo the Qwen4Exp AWQ-fold dtype leak without another value transform.

    The converter's stream fold historically multiplied retained BF16
    mix-down, injection, and grouped-norm tensors by F32 scales without casting
    back. When the runtime has resolved one uniform packed-affine compute dtype,
    it is authoritative; otherwise the unaffected mix-up sibling retains the
    original floating dtype. Require the complete F32 norm+down signature before
    changing anything so intentional F32 modules and packed projections remain
    untouched.
    """

    changed_matrices = 0
    changed_norms = 0
    for up_name, up_weight in tuple(weights.items()):
        if not up_name.endswith(_HYPER_UP_SUFFIX):
            continue
        if up_weight.dtype not in _LOW_PRECISION_FLOAT_DTYPES:
            continue
        resolved_dtype = target_dtype or up_weight.dtype
        if resolved_dtype not in _LOW_PRECISION_FLOAT_DTYPES:
            continue
        base = up_name[: -len(_HYPER_UP_SUFFIX)]
        norm_name = base + ".hc_norm.weight"
        down_name = base + ".input_mix_weight_down.weight"
        norm = weights.get(norm_name)
        down = weights.get(down_name)
        if (
            norm is None
            or down is None
            or norm.dtype != mx.float32
            or down.dtype != mx.float32
        ):
            continue
        weights[norm_name] = norm.astype(resolved_dtype)
        weights[down_name] = down.astype(resolved_dtype)
        changed_norms += 1
        changed_matrices += 1
        injection_name = base + ".block_inject_weight.weight"
        injection = weights.get(injection_name)
        if injection is not None and injection.dtype == mx.float32:
            weights[injection_name] = injection.astype(resolved_dtype)
            changed_matrices += 1
    return changed_matrices, changed_norms


def _resolve_jang_runtime_compute_dtype(weights: dict[str, mx.array]):
    """Resolve compute dtype from packed affine metadata, not config intent.

    A converted JANG bundle may retain the upstream BF16 declaration while its
    packed affine scales/biases and SSD-backed PLE rows are F16. Rewriting those
    tensors to the declared dtype after load changes MLX kernel dispatch. A
    single metadata dtype is therefore the concrete runtime contract. Mixed
    metadata dtypes fail closed instead of silently choosing one.
    """

    packed_dtypes = set()
    for name, value in weights.items():
        stem, separator, leaf = name.rpartition(".")
        if not separator or leaf not in {"scales", "biases"}:
            continue
        packed_weight = weights.get(f"{stem}.weight")
        if packed_weight is None or packed_weight.dtype != mx.uint32:
            continue
        if value.dtype in _LOW_PRECISION_FLOAT_DTYPES:
            packed_dtypes.add(value.dtype)
    if not packed_dtypes:
        raise ValueError("qwen4_exp JANG bundle has no packed affine metadata dtype")
    if len(packed_dtypes) != 1:
        rendered = ", ".join(sorted(str(dtype) for dtype in packed_dtypes))
        raise ValueError(
            "qwen4_exp JANG packed affine metadata has mixed compute dtypes: "
            f"{rendered}"
        )
    return next(iter(packed_dtypes))


def _is_jangt_format_tensor(name: str) -> bool:
    """JANGT trellis tensors (jt_scale/jt_su/jt_cs fp32, jt_wout fp16, jt_ocol int32, jt_packed u32) are format, not
    compute weights: casting the fp32 scales to the compute dtype loses precision on decode and breaks the NAX
    prefill kernel, which takes float scales (GLM JANGHT, 2026-10-10)."""
    return name.rsplit(".", 1)[-1].startswith("jt_")


def _normalize_jang_runtime_compute_dtypes(
    weights: dict[str, mx.array],
    model: nn.Module,
    target_dtype,
) -> dict[str, int]:
    """Normalize cast-eligible Qwen4Exp tensors to one compute dtype.

    The model's cast predicate preserves deliberately F32 recurrent state
    coefficients. All other floating compute tensors, including routers,
    norms, dense projections, and packed affine metadata, are aligned with the
    dtype carried by the bundle's packed scales/biases.
    """

    if target_dtype not in _LOW_PRECISION_FLOAT_DTYPES:
        raise ValueError(
            "qwen4_exp JANG runtime compute dtype must be float16 or "
            f"bfloat16, got {target_dtype}"
        )
    cast_predicate = getattr(model, "cast_predicate", lambda _path: True)
    summary = {"f16": 0, "bf16": 0, "f32": 0, "cast": 0, "preserved": 0}
    source_keys = {
        mx.float16: "f16",
        mx.bfloat16: "bf16",
        mx.float32: "f32",
    }
    for name, value in tuple(weights.items()):
        source_key = source_keys.get(value.dtype)
        if source_key is None:
            continue
        summary[source_key] += 1
        if not cast_predicate(name) or name.endswith(".tq2_scales") or _is_jangt_format_tensor(name):
            summary["preserved"] += 1
            continue
        if value.dtype != target_dtype:
            weights[name] = value.astype(target_dtype)
            summary["cast"] += 1
    return summary


def _warn_runtime_dtype_mismatches(model: nn.Module, target_dtype) -> int:
    """Warn about cast-eligible parameters outside the runtime compute dtype."""

    try:
        from mlx.utils import tree_flatten

        cast_predicate = getattr(model, "cast_predicate", lambda _path: True)
        mismatches = [
            (name, str(value.dtype))
            for name, value in tree_flatten(model.parameters())
            if value.dtype in {mx.float16, mx.bfloat16, mx.float32}
            and cast_predicate(name)
            and not _is_jangt_format_tensor(name)
            and value.dtype != target_dtype
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not audit qwen4_exp runtime dtypes: %s", exc)
        return -1
    if mismatches:
        preview = ", ".join(
            f"{name}={dtype}" for name, dtype in mismatches[:16]
        )
        logger.warning(
            "qwen4_exp loader-owned compute dtype %s has %d floating "
            "parameter mismatch(es): %s%s",
            target_dtype,
            len(mismatches),
            preview,
            " ..." if len(mismatches) > 16 else "",
        )
    return len(mismatches)


def _classify_ple_tensor(key: str) -> tuple[str, str] | None:
    table_match = _PLE_TABLE_RE.search(key)
    if table_match is not None:
        return "table", table_match.group("suffix")
    buffer_match = _PLE_BUFFER_RE.search(key)
    if buffer_match is not None:
        return "buffer", buffer_match.group("suffix")
    return None


def _is_ple_table_tensor(key: str) -> bool:
    return _classify_ple_tensor(key) is not None


def _resolve_ple_module_key_format(weight_map: dict[str, str], n_shards: int) -> str:
    """Resolve the raw checkpoint's PLE prefix without assuming sanitization.

    Official and converted checkpoints have used both ``ple.ple_embedding``
    and the runtime's shorter ``ple`` nesting, with either ``model`` or
    ``language_model.model`` roots.  MLX converters also use both
    ``shard_N`` and ``shards.N`` spellings.  The index is authoritative:
    accept one complete layout and reject missing or ambiguous tables before
    loading.
    """
    if n_shards <= 0:
        raise ValueError("qwen4_exp split_ngram_parts must be positive")
    candidates: set[str] = set()
    for key in weight_map:
        match = _PLE_TABLE_RE.search(key)
        if match is None or match.group("suffix") != "weight":
            continue
        module_path = key.rsplit(".", 1)[0]
        marker = f"{match.group('layout')}{match.group('shard')}"
        if not module_path.endswith(marker):
            continue
        candidates.add(module_path[: -len(marker)] + match.group("layout") + "{}")

    complete = []
    for key_format in sorted(candidates):
        if all(
            f"{key_format.format(shard)}.{suffix}" in weight_map
            for shard in range(n_shards)
            for suffix in _PLE_REQUIRED_SHARD_SUFFIXES
        ):
            complete.append(key_format)
    if not complete:
        raise ValueError(
            "qwen4_exp bundle has no complete indexed PLE n-gram table for "
            f"{n_shards} shards"
        )
    if len(complete) != 1:
        raise ValueError(
            "qwen4_exp bundle has ambiguous PLE n-gram table layouts: "
            + ", ".join(complete)
        )
    return complete[0]


def _load_bfloat16_tensor(path: Path, key: str) -> mx.array:
    """Read one BF16 tensor when safetensors' MLX adapter cannot expose it."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
        info = header[key]
        if str(info["dtype"]).upper() != "BF16":
            raise TypeError(
                f"selective fallback only supports BF16, got {info['dtype']}"
            )
        shape = tuple(int(dim) for dim in info["shape"])
        start, end = (int(offset) for offset in info["data_offsets"])
        handle.seek(8 + header_len + start)
        raw = handle.read(end - start)
    bits = np.frombuffer(raw, dtype="<u2").copy().reshape(shape)
    fp32 = (bits.astype(np.uint32) << 16).view(np.float32)
    return mx.array(fp32).astype(mx.bfloat16)


def _load_non_table_weight_files(
    weight_files: list[Path],
    affine1_modules: frozenset[str],
    *,
    safe_open_fn,
) -> tuple[dict[str, mx.array], bool, dict[str, mx.array]]:
    """Selectively read non-PLE tensors; never request a 51B table tensor."""
    weights: dict[str, mx.array] = {}
    ple_buffers: dict[str, mx.array] = {}
    is_mlx_format = False

    def _report_shards(completed: int) -> None:
        # Lifecycle-progress contract: completed counts only shards whose
        # tensors have actually been read (never raises, never a guess).
        try:
            from ...load_progress import report_shard

            report_shard(completed, len(weight_files))
        except Exception:
            pass

    for file_index, weight_file in enumerate(weight_files):
        _report_shards(file_index)
        with safe_open_fn(str(weight_file), framework="mlx") as handle:
            if file_index == 0:
                is_mlx_format = (handle.metadata() or {}).get("format") == "mlx"
            shard: dict[str, mx.array] = {}
            for key in handle.keys():
                classification = _classify_ple_tensor(str(key))
                if classification is not None and classification[0] == "table":
                    continue
                try:
                    value = handle.get_tensor(key)
                except TypeError as exc:
                    if "bfloat16" not in str(exc).lower():
                        raise
                    value = _load_bfloat16_tensor(weight_file, str(key))
                if classification is not None:
                    suffix = classification[1]
                    if suffix in ple_buffers:
                        raise ValueError(f"duplicate qwen4_exp PLE buffer {suffix}")
                    ple_buffers[suffix] = value
                    continue
                if key in weights or key in shard:
                    raise ValueError(f"duplicate qwen4_exp tensor {key}")
                shard[str(key)] = value
        if affine1_modules:
            affine_aliases = set(affine1_modules)
            for module_path in tuple(affine1_modules):
                affine_aliases.add(
                    module_path.replace(
                        "model.language_model", "language_model.model", 1
                    )
                )
                affine_aliases.add(
                    module_path.replace(
                        "language_model.model", "model.language_model", 1
                    )
                )
                affine_aliases.add(module_path.replace(".ple.ple_embedding.", ".ple."))
            shard, _expanded = expand_affine1_shard_mlx(
                shard, frozenset(affine_aliases)
            )
        weights.update(shard)
    if weight_files:
        _report_shards(len(weight_files))
    return weights, is_mlx_format, ple_buffers


def _enumerate_weight_files(model_path: Path) -> list[Path]:
    """Weight files come from the index, never from a filename glob.

    The Flash-Next bundle contract: ``model.safetensors.index.json`` is the
    sole authority for which files hold model weights (mtp.* tensors live in
    the regular model-* shards it names). An extra file the index does not
    mention — the calibrated proposal-head sidecar, a stray export, an
    AppleDouble copy — must never be swept into the weight load or fail it
    (a root-level sidecar carrying lm_head.* copies aborted every load with
    "duplicate qwen4_exp tensor lm_head.biases", found live 2026-09-04).
    Bundles without an index (single-file layouts) keep the original glob.
    """

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map")
            if isinstance(weight_map, dict) and weight_map:
                files = sorted(
                    {str(name) for name in weight_map.values() if name}
                )
                resolved = [model_path / name for name in files]
                missing = [p.name for p in resolved if not p.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "model.safetensors.index.json references missing "
                        f"files: {', '.join(missing)}"
                    )
                return resolved
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - unreadable index -> glob
            logger.warning(
                "qwen4_exp index unreadable (%s); falling back to glob", exc
            )

    return sorted(
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    )


def _load_non_table_weights(
    model_path: Path,
    affine1_modules: frozenset[str],
) -> tuple[dict[str, mx.array], bool, dict[str, mx.array]]:
    from safetensors import safe_open

    weight_files = _enumerate_weight_files(model_path)
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    return _load_non_table_weight_files(
        weight_files,
        affine1_modules,
        safe_open_fn=safe_open,
    )


def _validate_ple_hash_buffers(model, ple_buffers: dict[str, mx.array]) -> None:
    """Fail closed if bundle hash constants differ from the runtime hasher."""
    import numpy as np

    ple = model.language_model.model.layers[1].ple
    expected = {
        "layer_multipliers": ple.hasher.layer_multipliers,
        "ngram_heads_vocab_sizes": ple.hasher._head_sizes_np,
        "ngram_heads_offsets": ple.hasher._head_offsets_np,
    }
    for suffix, wanted in expected.items():
        value = ple_buffers.get(suffix)
        if value is None:
            raise ValueError(
                f"qwen4_exp bundle is missing required PLE buffer {suffix}"
            )
        mx.eval(value)
        actual = np.asarray(value).astype(np.int64, copy=False)
        if not np.array_equal(actual, wanted):
            raise ValueError(f"qwen4_exp PLE hash buffer mismatch: {suffix}")


def _install_jangt_routed_experts(model, config: dict) -> int:
    """JANGHT bundles: replace the backbone SwitchGLU modules with JTMixedSwitchGLU when config.json carries a
    top-level "jangt" block (same contract as GLM-5.3 / Naive-N0.5 JANGHT, vmlx_engine.jangt.switch):
      config["jangt"] = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                         "outlier_columns": {"model.layers.L.mlp.switch_mlp.<proj>": n_out}}
      config["quantization"]["model.layers.L.mlp.switch_mlp.<proj>"] = {"mode": "jangt", "kbits": K}  (K in 1, 1.5,
        2, 2.5, 3; gate_proj == up_proj) or a JANGH {"mode": "jangtq2", ...} group (mixed stacks).
    Shared experts, routers and the SSD-backed PLE path are untouched; MTP experts stay affine (install_jangt skips
    paths containing "mtp"). Qwen experts use plain SwiGLU (no clamp). A JANGHT bundle may also carry a "jangtq" block
    for its JANGH groups; the routed stacks follow the JANGT contract and are never validated as all-JANGH (the
    GLM-5.3 VLM rule). Fail closed: a declared JANGT bundle with an incomplete contract raises."""
    jangt = config.get("jangt")
    if not jangt:
        return 0
    from vmlx_engine.jangt.switch import install_jangt

    installed = install_jangt(
        model,
        {"jangt": jangt, "quantization": config.get("quantization") or {}, "swiglu_limit": 0.0},
    )
    if installed <= 0:
        raise ValueError("qwen4_exp declares JANGT routed experts but none were installed")
    logger.info("qwen4_exp: %d JANGT/JANGH mixed routed expert banks installed", installed)
    return installed


def _install_routed_expert_formats(model, config: dict) -> tuple[int, int]:
    """(JANGT stacks, JANGH stacks) installed before weights load. JANGT owns the routed stacks when declared."""
    jangt_modules = _install_jangt_routed_experts(model, config)
    if jangt_modules:
        return jangt_modules, 0
    return 0, _install_jangh_routed_experts(model, config)


def _install_jangh_routed_experts(model, config: dict) -> int:
    """Replace the 48 backbone SwitchGLU modules with JANGH TQSwitchGLU when the bundle declares
    config["jangtq"] (format v2, mode jangtq2 per routed projection). MTP experts stay affine
    (install_jangh skips paths containing "mtp"). Fail closed: a declared JANGH bundle whose
    contract does not match the model raises instead of loading affine garbage."""
    from vmlx_engine.jangh.contract import validate_format

    if not validate_format(config):
        return 0
    from vmlx_engine.jangh.install import install_jangh

    installed = install_jangh(model, config)
    if installed <= 0:
        raise ValueError("qwen4_exp declares JANGH routed experts but none were installed")
    # qwen4_exp prefill speed options (measured one MoE layer, D=2560 I=640 E=512 k=10, f16: JANGH/affine 1.235 ->
    # 0.995 at 4k tokens): single-launch Hadamard-32 row rotation + fused fp32 weighted unsort. Scoped to this family
    # so GLM/Naive keep their qualified numerics. Env JANGH_QWEN4_PREFILL_FUSED=0 restores the generic path.
    from vmlx_engine.jangh.runtime_identity import QWEN4_PREFILL_FUSED, qwen4_decode_max_tokens

    # Decode-vs-sorted routed path by TOKENS, not routed rows (switch._use_sorted). Measured
    # crossover for this geometry (D=2560, I=640, E=512, k=10, JANGH 4/6-bit, M5 Max, MLX
    # 0.32.3): gather wins through 96 tokens, sorted from 128. The legacy 64-row rule sent a
    # continuous-batching decode step of >= 7 sequences to the prefill kernels (8 concurrent
    # measured 35 tok/s aggregate, below single-stream). JANGH_DECODE_MAX_TOKENS pins it for A/B.
    decode_max_tokens = qwen4_decode_max_tokens()
    for _path, mod in model.named_modules():
        if getattr(mod, "is_jangtq2", False):
            mod.decode_max_tokens = decode_max_tokens
            if QWEN4_PREFILL_FUSED == "1":
                mod.use_weighted_unsort = True
                for lin in (mod.gate_proj, mod.up_proj, mod.down_proj):
                    lin.use_h32_rows = True
    logger.info("qwen4_exp: %d JANGH routed expert banks installed", installed)
    return installed


def _normalize_runtime_weight_names(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Map checkpoint roots to the owning VLM wrapper without hiding extras.

    The official checkpoint is sanitized into wrapper-owned names, while the
    JANG converter writes already-sanitized MLX names without those wrapper
    segments: ``language_model.*``, ``visual.*``, ``lm_head.*``, and
    ``mtp.*``.  Normalize both forms to the instantiated VLM tree without
    hiding extras.  This is deliberately separate from value sanitization:
    JANG's ``runtime_plus1_applied`` stamp means shifted norms and convolution
    layouts must not be transformed a second time.
    """
    normalized: dict[str, mx.array] = {}

    def store(source_key: str, runtime_key: str, value: mx.array) -> None:
        if runtime_key in normalized:
            raise ValueError(
                "qwen4_exp weight-name collision after runtime normalization: "
                f"{source_key} -> {runtime_key}"
            )
        normalized[runtime_key] = value

    for key, value in weights.items():
        runtime_key = key
        if runtime_key.startswith("mtp."):
            runtime_key = f"language_model.{runtime_key}"
        elif runtime_key.startswith("visual."):
            runtime_key = runtime_key.replace("visual.", "vision_tower.", 1)
        elif runtime_key.startswith("lm_head."):
            runtime_key = runtime_key.replace(
                "lm_head.", "language_model.lm_head.", 1
            )
        elif runtime_key.startswith("model.layers.") and ".mlp.switch_mlp." in runtime_key:
            # JANGH routed experts: on-disk model.layers.L.mlp.switch_mlp.*_proj.tq2_* (the namespace the
            # vmlx-swift JANGH partition and the MLX named-mmap regex require) -> the VLM tree.
            runtime_key = f"language_model.{runtime_key}"
        elif runtime_key.startswith("language_model.") and not runtime_key.startswith(
            ("language_model.model.", "language_model.mtp.", "language_model.lm_head.")
        ):
            runtime_key = runtime_key.replace(
                "language_model.", "language_model.model.", 1
            )
        if (
            runtime_key == "vision_tower.patch_embed.proj.weight"
            and value.ndim == 5
            and value.shape[1] in (1, 3)
            and value.shape[-1] not in (1, 3)
        ):
            # HF Conv3d layout is [out, in, temporal, height, width]; MLX's
            # convolution kernel keeps input channels last.  The JANG converter
            # preserves the HF layout for this passthrough tensor, so perform
            # the same one-time transform used by the established VL converters.
            value = value.transpose(0, 2, 3, 4, 1)
        if runtime_key.endswith(".ple.conv1d.weight"):
            runtime_key = runtime_key.removesuffix(".conv1d.weight") + ".conv1d_weight"
            if value.ndim == 3:
                if value.shape[1] != 1:
                    raise ValueError(
                        "qwen4_exp PLE convolution must have a singleton input "
                        f"channel, got {value.shape}"
                    )
                value = value.squeeze(1)
            if value.ndim != 2:
                raise ValueError(
                    "qwen4_exp PLE convolution must resolve to [channels, taps], "
                    f"got {value.shape}"
                )
        store(key, runtime_key, value)

    # The converter already unfuses every backbone MoE, but preserves the
    # trained top-level MTP head before reaching that transform.  Normalize its
    # fused expert tensors here, including packed affine weight/scales/biases;
    # splitting the expert-intermediate axis preserves each quantization group
    # byte-for-byte.
    fused_suffix = ".experts.gate_up_proj.weight"
    fused_prefixes = [
        key.removesuffix(fused_suffix)
        for key in tuple(normalized)
        if key.startswith("language_model.mtp.") and key.endswith(fused_suffix)
    ]
    for prefix in fused_prefixes:
        gate_up_prefix = f"{prefix}.experts.gate_up_proj"
        down_prefix = f"{prefix}.experts.down_proj"
        for suffix in ("weight", "scales", "biases"):
            gate_up_key = f"{gate_up_prefix}.{suffix}"
            down_key = f"{down_prefix}.{suffix}"
            if gate_up_key not in normalized or down_key not in normalized:
                raise ValueError(
                    "qwen4_exp fused MTP expert tensors are incomplete for "
                    f"{prefix}.{suffix}"
                )
            gate_up = normalized.pop(gate_up_key)
            down = normalized.pop(down_key)
            if gate_up.ndim < 2 or gate_up.shape[-2] % 2:
                raise ValueError(
                    "qwen4_exp fused MTP gate/up tensor must have an even "
                    f"expert-intermediate axis, got {gate_up.shape}"
                )
            midpoint = gate_up.shape[-2] // 2
            store(
                gate_up_key,
                f"{prefix}.switch_mlp.gate_proj.{suffix}",
                gate_up[..., :midpoint, :],
            )
            store(
                gate_up_key,
                f"{prefix}.switch_mlp.up_proj.{suffix}",
                gate_up[..., midpoint:, :],
            )
            store(
                down_key,
                f"{prefix}.switch_mlp.down_proj.{suffix}",
                down,
            )
    return normalized


def _quantize_model(
    model,
    config: dict,
    weights: dict[str, mx.array],
    bit_map: Mapping[str, object] | None = None,
) -> None:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return
    model_predicate = getattr(model, "quant_predicate", None)

    def checkpoint_spec(path, module, aliases, declared):
        stems = [alias for alias in aliases if f"{alias}.scales" in weights]
        stems = list(dict.fromkeys(stems))
        if not stems:
            return None
        complete = [
            stem
            for stem in stems
            if f"{stem}.weight" in weights and f"{stem}.scales" in weights
        ]
        if len(complete) != 1:
            raise ValueError(
                "qwen4_exp quantized checkpoint aliases are incomplete or "
                f"ambiguous for {path}: {', '.join(stems)}"
            )
        stem = complete[0]
        packed = weights[f"{stem}.weight"]
        scales = weights[f"{stem}.scales"]
        logical = getattr(module, "weight", None)
        if logical is None or not hasattr(logical, "shape"):
            raise ValueError(
                f"qwen4_exp quantized checkpoint module {path} has no logical weight"
            )
        logical_shape = tuple(int(dim) for dim in logical.shape)
        packed_shape = tuple(int(dim) for dim in packed.shape)
        scales_shape = tuple(int(dim) for dim in scales.shape)
        if packed.dtype != mx.uint32:
            raise ValueError(
                f"qwen4_exp quantized checkpoint weight {stem}.weight must be U32"
            )
        if (
            len(logical_shape) < 2
            or len(packed_shape) != len(logical_shape)
            or len(scales_shape) != len(logical_shape)
            or packed_shape[:-1] != logical_shape[:-1]
            or scales_shape[:-1] != logical_shape[:-1]
        ):
            raise ValueError(
                "qwen4_exp quantized checkpoint leading shapes disagree for "
                f"{path}: logical={logical_shape}, packed={packed_shape}, "
                f"scales={scales_shape}"
            )
        input_dims = logical_shape[-1]
        if input_dims <= 0 or scales_shape[-1] <= 0:
            raise ValueError(
                f"qwen4_exp quantized checkpoint dimensions must be positive for {path}"
            )
        packed_bits = 32 * packed_shape[-1]
        if packed_bits % input_dims:
            raise ValueError(
                "qwen4_exp packed width does not encode an integral bit width "
                f"for {path}: packed_cols={packed_shape[-1]}, "
                f"input_dims={input_dims}"
            )
        bits = packed_bits // input_dims
        if bits not in _MLX_AFFINE_BITS:
            raise ValueError(
                "qwen4_exp checkpoint-derived bit width is unsupported for "
                f"{path}: {bits}"
            )
        if input_dims % scales_shape[-1]:
            raise ValueError(
                "qwen4_exp scale width does not encode an integral group size "
                f"for {path}: scale_cols={scales_shape[-1]}, "
                f"input_dims={input_dims}"
            )
        group_size = input_dims // scales_shape[-1]
        if packed_shape[-1] != input_dims * bits // 32:
            raise ValueError(
                f"qwen4_exp checkpoint packed shape mismatch for {path}"
            )
        mode = str(
            (declared or {}).get("mode", quantization.get("mode", "affine"))
        )
        biases = weights.get(f"{stem}.biases")
        if mode == "affine" and biases is None:
            raise ValueError(
                f"qwen4_exp affine checkpoint module {path} is missing biases"
            )
        if (
            biases is not None
            and tuple(int(dim) for dim in biases.shape) != scales_shape
        ):
            raise ValueError(
                "qwen4_exp checkpoint scales/biases shapes disagree for "
                f"{path}: {scales_shape} != "
                f"{tuple(int(dim) for dim in biases.shape)}"
            )
        return {"bits": bits, "group_size": group_size, "mode": mode}

    def predicate(path, module):
        aliases = (
            path,
            path.replace("language_model.model", "language_model", 1),
            path.replace("language_model.mtp", "mtp", 1),
            path.replace("language_model.lm_head", "lm_head", 1),
        )
        checkpoint_quantized = any(
            f"{alias}.scales" in weights for alias in aliases
        )
        blocked_by_model = callable(model_predicate) and not model_predicate(
            path, module
        )
        if blocked_by_model:
            # JANG_4M explicitly stores 8-bit router gates.  They cannot be
            # treated as float weights or discarded; instantiate the matching
            # affine module only when the complete prequantized checkpoint
            # tensor is present. Recurrent coefficients remain forbidden.
            if not (
                bit_map is not None
                and checkpoint_quantized
                and path.endswith("mlp.gate")
            ):
                return False
        override = (
            resolve_jang_bit_map_spec(path, bit_map)
            if bit_map is not None
            else quantization.get(path)
        )
        derived = checkpoint_spec(path, module, aliases, override)
        if derived is not None:
            return derived
        if isinstance(override, dict):
            if checkpoint_quantized:
                return override
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=int(quantization.get("group_size", 64)),
        bits=int(quantization.get("bits", 4)),
        mode=str(quantization.get("mode", "affine")),
        class_predicate=predicate,
    )


def _text_only_processor(model_path: Path, eos_token_ids):
    """Processor for a text-only qwen4_exp bundle (no preprocessor_config.json): the tokenizer, with the chat template
    and eos ids the MLLM path reads from a processor. Image/video inputs are refused by the model itself."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    template = model_path / "chat_template.jinja"
    if template.is_file() and not getattr(tok, "chat_template", None):
        tok.chat_template = template.read_text()
    if eos_token_ids is not None:
        tok.eos_token_ids = eos_token_ids if isinstance(eos_token_ids, (list, tuple)) else [eos_token_ids]
    tok.tokenizer = tok          # MLLM call sites read processor.tokenizer
    return tok


def load_qwen4_exp_vlm_model(model_path: str | Path, *, lazy: bool = False):
    """Load qwen4_exp without ever adding its packed 51B PLE table to weights."""
    from mlx_vlm.utils import (
        get_model_and_args,
        load_image_processor,
        load_processor,
        update_module_configs,
    )

    model_path = Path(model_path)
    if not register_qwen4_exp_runtime():
        raise RuntimeError("qwen4_exp MLX-VLM runtime registration failed")
    config, affine1_modules, bit_map, jang_runtime_sanitized = _load_runtime_config(
        model_path
    )
    model_class, _ = get_model_and_args(config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    # Text-only bundle (the 48 GB JANGHT build ships no vision tower): no `vision_config` in config.json. Same rule as
    # vmlx-swift (`visionConfiguration` optional -> text-only). The wrapper still builds a default tower so the module
    # tree matches upstream; it is removed below once the checkpoint confirms there are no vision weights.
    declares_vision = "vision_config" in config
    config.setdefault("vision_config", {})
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config, ["text", "vision"]
    )
    model = model_class.Model(model_config)
    jangt_modules, jangh_modules = _install_routed_expert_formats(model, config)

    weights, is_mlx_format, ple_buffers = _load_non_table_weights(
        model_path, affine1_modules
    )
    if not (is_mlx_format or jang_runtime_sanitized):
        weights = model.sanitize(weights)
    weights = _normalize_runtime_weight_names(weights)
    text_only = not declares_vision and not any(k.startswith("vision_tower.") for k in weights)
    if text_only:
        # strict=True below still covers every remaining parameter; only the declared-absent tower is dropped
        model.pop("vision_tower")
        logger.info("qwen4_exp: text-only bundle (no vision_config, no vision_tower.* weights): vision tower not built")
    elif not declares_vision:
        raise ValueError("qwen4_exp: checkpoint has vision_tower.* weights but config.json has no vision_config")
    if jangt_modules:
        from vmlx_engine.jangt.switch import normalize_jangt_payload

        # bundle (E, N, W) or encoder stage (ragged) layout -> module layout; fp32 format tensors; fail closed on a
        # width/stride that disagrees with the per-module contract
        normalize_jangt_payload(model, weights)
    normalized_hyper = (0, 0)
    runtime_dtype = None
    runtime_dtype_summary = None
    if jang_runtime_sanitized:
        runtime_dtype = _resolve_jang_runtime_compute_dtype(weights)
        normalized_hyper = _normalize_jang_hyper_fold_dtypes(
            weights,
            target_dtype=runtime_dtype,
        )
        runtime_dtype_summary = _normalize_jang_runtime_compute_dtypes(
            weights,
            model,
            runtime_dtype,
        )
    _quantize_model(model, config, weights, bit_map)
    if jangh_modules:
        from vmlx_engine.jangh.payload import validate_complete_payload, validate_payload

        # strict load checks shapes, not dtypes: tq2_packed must stay U32 and tq2_scales F16
        validate_complete_payload(model, validate_payload(model, weights.items()))
    # The PLE table and its three hash buffers were intentionally removed from
    # the ordinary parameter tree above.  Everything else, including MTP and
    # vision, must match exactly; a permissive load previously allowed an
    # entire top-level MTP head to be silently ignored.
    model.load_weights(list(weights.items()), strict=True)

    from mlx_vlm.models.qwen4_exp.language import (
        compile_hyper_connections,
        fold_zero_centered_norm_offsets,
        fuse_hyper_connection_projections,
        prepare_quantized_projection_groups,
    )

    fused_hyper = fuse_hyper_connection_projections(model)
    compiled_hyper = compile_hyper_connections(model)
    folded_zero_norms = fold_zero_centered_norm_offsets(model)
    projection_groups = prepare_quantized_projection_groups(model)
    draft_head_status = model.language_model.prepare_mtp_draft_head()

    from vmlx_engine.metal.qwen4_affine_moe_decode import (
        install_qwen4_affine_moe,
        qwen4_affine_moe_status,
    )
    from vmlx_engine.metal.affine_moe_pair_decode import (
        affine_moe_pair_status,
        install_affine_moe_pair_decode,
    )
    from vmlx_engine.acceleration_contract import record_acceleration_attestation

    install_qwen4_affine_moe(model)
    from vmlx_engine.native_mtp_ar_safety import env_flag as _env_flag

    native_mtp_active = bool(
        _env_flag(True, "VMLINUX_NATIVE_MTP", "VMLX_NATIVE_MTP")
        and getattr(model.language_model, "mtp", None) is not None
    )
    install_affine_moe_pair_decode(
        model, family="qwen4_exp", native_mtp_active=native_mtp_active
    )
    record_acceleration_attestation(
        model,
        "qwen4_exp",
        {
            "projection_groups": {
                "installed": sum(int(value) for value in projection_groups.values()),
                "groups": dict(projection_groups),
            },
            "hyper_connection_fusion": {
                "installed": bool(fused_hyper or compiled_hyper),
                "fused": fused_hyper,
                "compiled": compiled_hyper,
                "zero_norms_folded": folded_zero_norms,
            },
            "affine_moe": qwen4_affine_moe_status(),
            "affine_moe_pair": affine_moe_pair_status("qwen4_exp"),
            "mtp_draft_head": draft_head_status,
        },
    )

    _validate_ple_hash_buffers(model, ple_buffers)
    weight_map = json.loads(
        (model_path / "model.safetensors.index.json").read_text()
    ).get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("qwen4_exp safetensors index has no weight_map")
    ngram_shards = int(model_config.text_config.split_ngram_parts)
    ple_key_format = _resolve_ple_module_key_format(weight_map, ngram_shards)
    ngram_heads = (int(model_config.text_config.ngram_size) - 1) * int(
        model_config.text_config.heads_per_ngram
    )
    if ngram_heads <= 0 or int(model_config.text_config.ple_embed_dim) % ngram_heads:
        raise ValueError(
            "qwen4_exp ple_embed_dim must divide evenly across n-gram heads"
        )
    ple_head_dim = int(model_config.text_config.ple_embed_dim) // ngram_heads
    table = FileBackedQuantizedNGramTable(
        model_path,
        ple_key_format,
        ngram_shards,
        expected_head_dim=ple_head_dim,
        bit_map=bit_map,
    )
    model.language_model.model.layers[1].ple.ngram_embedding.set_file_backed(
        table,
        output_dtype=runtime_dtype,
    )
    model.language_model.model.layers[1].ple.prepare_runtime()
    runtime_dtype_mismatches = (
        _warn_runtime_dtype_mismatches(model, runtime_dtype)
        if runtime_dtype is not None
        else 0
    )

    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    # After the weights are read: fill the page cache with the file-mapped
    # n-gram tables so cold prompts do not page-fault every new row from the
    # model drive (table_reader.FileBackedQuantizedNGramTable.start_page_cache_warm).
    table.start_page_cache_warm()
    if text_only:
        processor = _text_only_processor(model_path, getattr(model.config, "eos_token_id", None))
    else:
        image_processor = load_image_processor(model_path)
        processor = load_processor(
            model_path,
            True,
            eos_token_ids=getattr(model.config, "eos_token_id", None),
        )
        if image_processor is not None:
            processor.image_processor = image_processor
    logger.info(
        "Loaded qwen4_exp with SSD-backed PLE (%s shards, MTP=%s, "
        "hyper_fused=%s, hyper_compiled=%s, hyper_dtype_normalized=%s/%s, "
        "zero_norms_folded=%s, projection_groups=%s, mtp_draft_head=%s, runtime_compute_dtype=%s, runtime_dtype_casts=%s, "
        "runtime_dtype_mismatches=%s, ple_storage_dtype=%s, "
        "ple_output_dtype=%s, ple_random_access_advised=%s/%s)",
        model_config.text_config.split_ngram_parts,
        model_config.text_config.mtp_num_hidden_layers,
        fused_hyper,
        compiled_hyper,
        normalized_hyper[0],
        normalized_hyper[1],
        folded_zero_norms,
        projection_groups,
        draft_head_status,
        str(runtime_dtype) if runtime_dtype is not None else "checkpoint",
        (runtime_dtype_summary or {}).get("cast", 0),
        runtime_dtype_mismatches,
        table.output_dtype,
        runtime_dtype if runtime_dtype is not None else table.output_dtype,
        table.random_access_advised_readers,
        table.random_access_reader_count,
    )
    return model, processor
