# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 JANGTQ loader — thin re-export.

Mirrors the pattern used for Kimi K2.6 (``load_jangtq_kimi_vlm``): a one-line
indirection so the research doc's code snippets work verbatim against the
vMLX engine (``from vmlx_engine.loaders.load_jangtq_dsv4 import
load_jangtq_dsv4_model``) AND so the bundled Python distribution always
exposes a stable import path even if ``jang_tools.dsv4`` internals move.

Under the hood this delegates to ``jang_tools.load_jangtq.load_jangtq_model``,
which already detects ``model_type="deepseek_v4"`` from ``config.json`` and
routes through ``jang_tools.dsv4.mlx_register`` (auto-imported at first call
via ``vmlx_engine.utils.jang_loader``) to register our custom MLX model class
(``jang_tools.dsv4.mlx_model.Model``) into ``mlx_lm.models`` namespace.

Runtime contract implemented by the underlying loader:
- mHC (Manifold Hyper-Connections) hc_mult=4 with 20 Sinkhorn iterations
  (fused Metal kernel, fallback pure-MLX op available).
- MLA attention head_dim=512 with grouped O projection (o_groups=8,
  o_lora_rank=1024), per-head RMSNorm, attention sink, inverse RoPE on
  output, partial RoPE over last qk_rope_head_dim=64 dimensions.
- 256 routed experts top-6 + 1 shared (switch_mlp stacked across layers).
- sqrtsoftplus routing with hash-table (tid2eid) for the first 3 layers,
  biased top-6 argpartition + weight-norm + routed_scaling_factor=1.5 for
  the remaining 40 layers.
- Cache: ``DeepseekV4Cache`` (RotatingKVCache sliding_window=128 +
  compressor/indexer state buffers) on ``compress_ratio>0`` layers,
  plain ``KVCache`` on ``compress_ratio=0`` layers. ``DSV4_LONG_CTX=1``
  is forced. Upstream JANG has a prefill mask-vs-full_kv shape bug
  that crashes any prompt > sliding_window with broadcast errors;
  this loader patches it in-process by replacing
  ``DeepseekV4Attention.__call__`` with a copy that adds the missing
  symmetric trim branch (see ``_install_dsv4_prefill_patch``).
- ``swiglu_limit=10`` SwiGLU activation, fp32 lm_head matmul (4096
  contraction in bf16 drifts logits enough to flip arithmetic answers —
  see research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md).

The function returns ``(model, tokenizer)`` just like the sibling loaders.
"""

from __future__ import annotations

import logging as _logging
import os
import json
import re
import sys
from pathlib import Path
from typing import Any, Tuple

_log = _logging.getLogger(__name__)
_PREFILL_PATCH_INSTALLED = False
_INSTANT_LOAD_PATCH_INSTALLED = False

# Sidecar manifest schema version. Bump when the on-disk layout changes so
# stale caches are auto-invalidated.
_INSTANT_LOAD_SCHEMA = 1
_INSTANT_LOAD_RUNTIME_PATCH = "dsv4-prefill-mask-v1-switchglu-marker-v2-mla-bitfix-v1"
_SIDECAR_FILENAME = "jangtq_stacked.safetensors"
_SIDECAR_MANIFEST = "jangtq_stacked.json"

_DSV4_CRITICAL_CONTROL_RE = re.compile(
    r"^(hc_head_(?:fn|base|scale)|"
    r"layers\.\d+\.hc_(?:attn|ffn)_(?:fn|base|scale)|"
    r"layers\.\d+\.attn\.attn_sink|"
    r"layers\.\d+\.ffn\.gate\.bias)$"
)


def _read_json(path: Path) -> dict:
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _is_dsv4_bundle(model_path: str | Path) -> bool:
    cfg = _read_json(Path(model_path) / "config.json")
    return cfg.get("model_type") == "deepseek_v4"


def _dsv4_weight_map(model_path: str | Path) -> dict[str, str]:
    """Return tensor-key -> safetensors filename for a DSV4 bundle."""
    bundle = Path(model_path)
    index_path = bundle / "model.safetensors.index.json"
    if index_path.is_file():
        data = _read_json(index_path)
        weight_map = data.get("weight_map")
        if isinstance(weight_map, dict):
            return {str(k): str(v) for k, v in weight_map.items()}

    weight_map: dict[str, str] = {}
    try:
        from safetensors import safe_open
    except Exception:
        return weight_map
    for sf in sorted(bundle.glob("*.safetensors")):
        try:
            with safe_open(str(sf), framework="numpy") as handle:
                for key in handle.keys():
                    weight_map[key] = sf.name
        except Exception:
            continue
    return weight_map


def _audit_dsv4_control_tensor_dtypes(model_path: str | Path) -> dict[str, Any]:
    """Header-only audit for DSV4 control tensors that must stay F32.

    DSV4's mHC/Sinkhorn/router/sink control tensors are small but quality
    critical. The JANG DSV4 fix kit documents that F16 copies of these tensors
    produce exactly the observed failure mode: short prompts can pass, while
    longer generations drift into multilingual/repetitive output. Header-only
    validation lets the app reject such bundles before loading 70+ GB of model
    weights and before users mistake a broken package for a cache/runtime bug.
    """
    bundle = Path(model_path)
    report: dict[str, Any] = {
        "checked": False,
        "critical_count": 0,
        "non_f32_count": 0,
        "non_f32_examples": [],
        "error": None,
    }
    if not _is_dsv4_bundle(bundle):
        return report

    weight_map = _dsv4_weight_map(bundle)
    critical_keys = sorted(k for k in weight_map if _DSV4_CRITICAL_CONTROL_RE.match(k))
    report["checked"] = True
    report["critical_count"] = len(critical_keys)
    if not critical_keys:
        report["error"] = "No DSV4 critical control tensors found in safetensors index."
        return report

    try:
        from safetensors import safe_open
    except Exception as exc:
        report["error"] = f"safetensors unavailable: {type(exc).__name__}: {exc}"
        return report

    handles: dict[str, Any] = {}
    try:
        for key in critical_keys:
            filename = weight_map[key]
            handle = handles.get(filename)
            if handle is None:
                handle = safe_open(str(bundle / filename), framework="numpy")
                handle.__enter__()
                handles[filename] = handle
            dtype = handle.get_slice(key).get_dtype()
            if dtype != "F32":
                report["non_f32_count"] += 1
                if len(report["non_f32_examples"]) < 12:
                    report["non_f32_examples"].append({"key": key, "dtype": dtype})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)
    return report


def _validate_dsv4_control_tensors(model_path: str | Path) -> None:
    """Raise a clear load-time error for known-bad DSV4 bundles."""
    report = _audit_dsv4_control_tensor_dtypes(model_path)
    if not report.get("checked"):
        return
    if report.get("error"):
        raise RuntimeError(
            "DSV4 bundle integrity audit failed before load: "
            f"{report['error']}. Rebuild or re-download the DSV4 JANGTQ bundle."
        )
    bad = int(report.get("non_f32_count") or 0)
    if bad:
        examples = ", ".join(
            f"{item['key']}={item['dtype']}"
            for item in report.get("non_f32_examples", [])[:6]
        )
        raise RuntimeError(
            "DSV4 bundle is known-bad: "
            f"{bad}/{report.get('critical_count')} critical control tensors "
            "are not F32. DSV4 mHC/Sinkhorn/router/sink tensors must retain "
            "source precision; F16 copies cause long-context drift, language "
            "salad, and repetition loops. Rebuild with the DSV4 safe converter "
            f"or use a corrected bundle. Examples: {examples}"
        )


def _install_dsv4_prefill_patch() -> None:
    """Patch the upstream JANG ``DeepseekV4Attention.__call__`` mask-trim
    bug in-process.

    The installed ``jang_tools/dsv4/mlx_model.py`` (DeepseekV4Attention,
    line ~882) only pads ``mask`` when ``full_kv.shape[2] > mask.shape[-1]``
    and is missing the symmetric trim branch. Live-traced 2026-05-04 with
    LONG_CTX=1: layer 0 has compress_ratio=0 so its cache is a plain
    KVCache that grows monotonically with every token — the model-level
    mask gets sized to ``layer0.offset``. Compress_ratio>0 layers wrap a
    RotatingKVCache(max_size=sliding_window=128) so their post-update
    ``full_kv`` caps at ``sliding_window + (pooled rows)``. Result: on any
    prompt + decode step where total tokens > sliding_window, the
    incoming mask shape ``(1, total_tokens)`` is wider than full_kv
    ``(1, ~129)``, ``mx.broadcast_shapes`` rejects, the request returns
    empty content. Patch wraps ``__call__`` to trim trailing-axis when
    mask overruns full_kv (mirror of the existing pad branch). Idempotent.
    """
    global _PREFILL_PATCH_INSTALLED
    if _PREFILL_PATCH_INSTALLED:
        return
    try:
        from jang_tools.dsv4.mlx_model import DeepseekV4Attention
        import mlx.core as mx
    except Exception as e:
        _log.warning("DSV4 prefill patch skipped (import failed: %s)", e)
        return

    from jang_tools.dsv4.mlx_model import (
        DeepseekV4Cache,
        _apply_partial_rope,
        _get_q_norm_ones,
        scaled_dot_product_attention,
    )

    def _patched_call(self, x, mask=None, cache=None):  # noqa: D401
        # Mirror of upstream DeepseekV4Attention.__call__ with a symmetric
        # trim branch added before the existing pad branch. Other than the
        # 4-line trim block this is a copy of the upstream forward; we
        # match its behavior exactly to avoid drift.
        B, L, _ = x.shape
        local_cache = cache if isinstance(cache, DeepseekV4Cache) else cache
        offset = local_cache.offset if local_cache is not None else 0

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(
            q,
            weight=_get_q_norm_ones(self.head_dim, q.dtype),
            eps=self.args.rms_norm_eps,
        )
        q = q.transpose(0, 2, 1, 3)

        kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, self.rope, offset)
        kv = _apply_partial_rope(kv, self.rope, offset)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            if v4_cache is not None or L >= self.compress_ratio:
                pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
                if hasattr(self, "indexer") and pooled.shape[1] > 0:
                    topk = self.indexer(
                        x, q_residual, self.compress_rope, self.rope,
                        v4_cache, offset,
                    )
                    if topk is not None:
                        expanded = mx.broadcast_to(
                            pooled[:, None, None, :, :],
                            (B, 1, L, pooled.shape[1], self.head_dim),
                        )
                        idx = topk[:, None, :, :, None]
                        pooled = mx.take_along_axis(
                            expanded,
                            mx.broadcast_to(idx, idx.shape[:-1] + (self.head_dim,)),
                            axis=3,
                        ).reshape(B, 1, -1, self.head_dim)
                    else:
                        pooled = pooled[:, None]
                else:
                    pooled = pooled[:, None]
                if pooled.shape[2] > 0:
                    full_kv = mx.concatenate([full_kv, pooled], axis=2)

        # vMLX additions: handle BOTH directions of the mask vs full_kv
        # mismatch. Upstream only pads (full_kv > mask). When the model-
        # level mask was sized to a layer whose cache grows monotonically
        # (e.g. layer 0 with compress_ratio=0 → KVCache, offset = total
        # tokens) but THIS layer wraps a RotatingKVCache(sliding_window)
        # whose post-update size caps near ``sliding_window`` (+pooled
        # rows), the upstream code feeds a too-wide mask straight into
        # SDPA and `mx.broadcast_shapes` rejects it.
        if mask is not None and hasattr(mask, "shape") and mask.shape:
            if mask.shape[-1] > full_kv.shape[2]:
                mask = mask[..., -full_kv.shape[2]:]
            elif full_kv.shape[2] > mask.shape[-1]:
                pad = mx.zeros(
                    mask.shape[:-1] + (full_kv.shape[2] - mask.shape[-1],),
                    dtype=mask.dtype,
                )
                mask = mx.concatenate([mask, pad], axis=-1)

        out = scaled_dot_product_attention(
            q, full_kv, full_kv,
            cache=local_cache, scale=self.softmax_scale, mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)

    DeepseekV4Attention.__call__ = _patched_call
    _PREFILL_PATCH_INSTALLED = True
    _log.info(
        "DSV4 prefill mask-trim patch installed on "
        "jang_tools.dsv4.mlx_model.DeepseekV4Attention.__call__ "
        "(symmetric to upstream pad branch; fixes broadcast_shapes for "
        "prompts > sliding_window with mixed KVCache/DeepseekV4Cache layers)."
    )


def _bundle_shard_signature(model_path) -> dict:
    """Return ``{shard_filename: [size, mtime]}`` for every model-*.safetensors
    shard in the bundle. Used to invalidate the stacked sidecar when the
    underlying bundle changes."""
    from pathlib import Path
    out: dict = {}
    for sf in sorted(Path(model_path).glob("model-*.safetensors")):
        st = sf.stat()
        out[sf.name] = [int(st.st_size), int(st.st_mtime)]
    return out


def _sidecar_paths(model_path):
    """Return ``(sidecar_safetensors, sidecar_manifest)`` for the bundle.

    Always uses ``~/.cache/vmlx-engine/jangtq-stacked/<sha16>/`` so the
    sidecar never pollutes the bundle directory. Earlier behavior wrote
    next to the weights when writable, but that bloats user model dirs
    by ~65 GB per DSV4 bundle and contaminates HF re-uploads.

    For bundle-dir compatibility: if a legacy sidecar still exists next
    to the weights it is honored at load time (``_try_fast_load_dsv4``
    checks both locations). Writes always go to ``~/.cache``.
    """
    from hashlib import sha256
    from pathlib import Path

    bundle = Path(model_path).resolve()
    digest = sha256(str(bundle).encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "vmlx-engine" / "jangtq-stacked" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / _SIDECAR_FILENAME, cache_dir / _SIDECAR_MANIFEST


def _legacy_bundle_sidecar_paths(model_path):
    """Legacy in-bundle sidecar path (for back-compat with older releases
    that wrote sidecars next to the weights). Read-only — used by
    ``_try_fast_load_dsv4`` when ``_sidecar_paths`` has nothing.
    """
    from pathlib import Path

    bundle = Path(model_path).resolve()
    return bundle / _SIDECAR_FILENAME, bundle / _SIDECAR_MANIFEST


def _try_fast_load_dsv4(model, model_path, mxtq_seed) -> bool:
    """Attempt to hydrate DSV4 from the pre-stacked sidecar.

    Returns True on success (model is fully wired), False when the
    sidecar is missing/stale and the streaming hydrate path must run.
    """
    import json

    try:
        import mlx.core as mx
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
    except Exception as e:
        _log.debug("DSV4 fast-load skipped (import failed: %s)", e)
        return False

    sidecar, manifest_path = _sidecar_paths(model_path)
    if not sidecar.exists() or not manifest_path.exists():
        # Back-compat: legacy releases wrote the sidecar next to the bundle.
        # If a current ~/.cache one isn't present, fall back to the legacy
        # in-bundle path (read-only). New writes still go to ~/.cache via
        # _sidecar_paths.
        legacy_sidecar, legacy_manifest = _legacy_bundle_sidecar_paths(model_path)
        if legacy_sidecar.exists() and legacy_manifest.exists():
            _log.info(
                "DSV4 fast-load: using legacy in-bundle sidecar at %s "
                "(future writes will go to ~/.cache)",
                legacy_sidecar,
            )
            sidecar, manifest_path = legacy_sidecar, legacy_manifest
        else:
            return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        _log.warning("DSV4 fast-load skipped (manifest parse failed: %s)", e)
        return False
    if manifest.get("schema") != _INSTANT_LOAD_SCHEMA:
        _log.info(
            "DSV4 fast-load: sidecar schema=%r vs current=%r — invalidating",
            manifest.get("schema"), _INSTANT_LOAD_SCHEMA,
        )
        return False
    if manifest.get("runtime_patch") != _INSTANT_LOAD_RUNTIME_PATCH:
        _log.info(
            "DSV4 fast-load: runtime patch=%r vs current=%r — invalidating",
            manifest.get("runtime_patch"), _INSTANT_LOAD_RUNTIME_PATCH,
        )
        return False
    if manifest.get("mxtq_seed") != int(mxtq_seed):
        _log.info("DSV4 fast-load: seed mismatch — invalidating")
        return False
    if manifest.get("shard_signature") != _bundle_shard_signature(model_path):
        _log.info("DSV4 fast-load: source shard mtime/size changed — invalidating")
        return False

    try:
        try:
            from vmlx_engine.cache_record_validator import reject_safetensors_or_warn
        except Exception:
            reject_safetensors_or_warn = None
        if reject_safetensors_or_warn is not None:
            if not reject_safetensors_or_warn(
                str(sidecar),
                source="dsv4-fast-load-sidecar",
                delete_on_reject=True,
            ):
                try:
                    manifest_path.unlink()
                except OSError:
                    pass
                return False
        weights = mx.load(str(sidecar))
    except Exception as e:
        _log.warning("DSV4 fast-load: mx.load(%s) failed (%s)", sidecar, e)
        return False

    groups = manifest.get("groups") or []
    if not groups:
        _log.warning("DSV4 fast-load: manifest has no groups; falling back")
        return False

    def _set_attr(root, dotted, new_mod):
        parts = dotted.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = new_mod
        else:
            setattr(cur, last, new_mod)

    def _get_attr(root, dotted):
        cur = root
        for p in dotted.split("."):
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
        return cur

    n_replaced = 0
    for entry in groups:
        new_base = entry["base"]
        bits = int(entry["bits"])
        packed = weights.get(f"{new_base}.packed")
        norms = weights.get(f"{new_base}.norms")
        if packed is None or norms is None:
            _log.warning(
                "DSV4 fast-load: sidecar missing tensors for %s — invalidating",
                new_base,
            )
            return False
        n_exp, out_feat, packed_cols = packed.shape
        vals_per_u32 = 32 // bits
        in_features = packed_cols * vals_per_u32
        new_module = TurboQuantSwitchLinear(
            in_features=in_features,
            out_features=out_feat,
            num_experts=n_exp,
            bits=bits,
            bias=False,
            seed=mxtq_seed,
        )
        new_module.packed = packed
        new_module.norms = norms
        try:
            _get_attr(model, new_base)
        except Exception as e:
            _log.warning(
                "DSV4 fast-load: model has no %s (%s) — invalidating",
                new_base, e,
            )
            return False
        _set_attr(model, new_base, new_module)
        n_replaced += 1

    # CRITICAL: replicate the post-stack work that the original streaming
    # hydrate does after replacing modules. Without these, embed_tokens /
    # lm_head / attention quantized layers retain bit-widths from the
    # generic model init (top-level `quantization.bits`), which mismatches
    # the actual on-disk packed shape and triggers `[dequantize] Shape of
    # scales and biases does not match` on the very first inference call.
    #
    # Mirror jang_tools.load_jangtq._hydrate_dsv4_jangtq_streaming lines
    # 275-276 (and the MLA bit fix used for absorbed projections in
    # _hydrate_jangtq_model lines 829-867 — DSV4 uses MLA via
    # DeepseekV4Attention's compress/indexer paths).
    try:
        from jang_tools.loader import _fix_quantized_bits
        _fix_quantized_bits(model, {})
    except Exception as e:
        _log.warning(
            "DSV4 fast-load: _fix_quantized_bits failed (%s); inference may "
            "hit dequantize shape mismatch. Falling back to streaming hydrate.",
            e,
        )
        return False

    # CRITICAL: install the DSV4-fused SwitchGLU routed-expert decode path.
    # Streaming hydrate runs this AFTER stacking; without it, decode falls
    # back to stock SwitchGLU (no Hadamard rotation, no fused gate+up
    # SwiGLU, no compiled per-shape kernel) and DSV4 throughput drops
    # ~25% (live-observed: 25 → 18 t/s on JANGTQ, JANGTQ_K).
    try:
        import mlx.core as mx
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
        from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
        from jang_tools.turboquant.fused_gate_up_kernel import (
            fused_gate_up_swiglu_matmul,
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import make_gather_tq_decode_per_row
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        _orig_switchglu_call = getattr(
            SwitchGLU, "_vmlx_dsv4_original_call", SwitchGLU.__call__
        )
        setattr(SwitchGLU, "_vmlx_dsv4_original_call", _orig_switchglu_call)
        _decode_compiled: dict = {}

        def _get_compiled_decode(
            in_f, out_f, bits, k, swiglu_limit=0.0, dp_bits=None
        ):
            if dp_bits is None:
                dp_bits = bits
            limit_milli = int(round(float(swiglu_limit or 0.0) * 1000.0))
            cache_key = (in_f, out_f, bits, dp_bits, k, limit_milli)
            if cache_key in _decode_compiled:
                return _decode_compiled[cache_key]
            fused_gu = make_fused_gate_up_swiglu_decode(
                in_f, out_f, bits, k, swiglu_limit=swiglu_limit
            )
            gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, k)

            def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx_flat):
                x_rot = hadamard_rotate_metal(x_flat, signs_in)
                x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx_flat)
                x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
                return gather_dn(x_act_rot, pd, nd, cb_down, idx_flat)

            _decode_compiled[cache_key] = mx.compile(_mlp)
            return _decode_compiled[cache_key]

        def _dsv4_fused_switchglu_call(self, x, indices):
            if not getattr(self, "_vmlx_dsv4_fused_fastpath", False):
                return _orig_switchglu_call(self, x, indices)
            gp = self.gate_proj
            up = self.up_proj
            dp = self.down_proj
            if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
                return _orig_switchglu_call(self, x, indices)
            activation = getattr(self, "activation", None)
            swiglu_limit = getattr(activation, "swiglu_limit", 0.0) or 0.0
            x_sq = x
            while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            x_flat = x_sq.reshape(-1, gp.in_features)
            batch = x_flat.shape[0]
            k = indices.shape[-1] if indices.ndim > 0 else 1
            can_fast = batch == 1 and k > 0 and indices.ndim >= 1 and indices.size < 64
            if can_fast and not getattr(self, "training", False):
                idx_flat = indices.reshape(-1).astype(mx.uint32)
                compiled_mlp = _get_compiled_decode(
                    gp.in_features,
                    gp.out_features,
                    gp.bits,
                    k,
                    swiglu_limit,
                    dp_bits=dp.bits,
                )
                y = compiled_mlp(
                    x_flat.astype(mx.float32),
                    gp.packed, gp.norms, up.packed, up.norms,
                    dp.packed, dp.norms,
                    gp.codebook, dp.codebook,
                    gp.signs, dp.signs, idx_flat,
                )
                out = y.reshape(*indices.shape[:-1], k, 1, gp.in_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out.squeeze(-2)

            x_exp = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x_exp, idx, inv_order = _gather_sort(x_exp, indices)
            if getattr(self, "training", False):
                idx = mx.stop_gradient(idx)
            x_act = fused_gate_up_swiglu_matmul(
                x_exp,
                gp.packed, gp.norms,
                up.packed, up.norms,
                gp.codebook, gp.signs,
                idx,
                bits=gp.bits,
                swiglu_limit=swiglu_limit,
            )
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)
            if do_sort:
                x_out = _scatter_unsort(x_out, inv_order, indices.shape)
            return x_out.squeeze(-2)

        SwitchGLU.__call__ = _dsv4_fused_switchglu_call
        n_patched = 0
        for _, m in model.named_modules():
            if (
                isinstance(m, SwitchGLU)
                and isinstance(getattr(m, "gate_proj", None), TurboQuantSwitchLinear)
                and isinstance(getattr(m, "up_proj", None), TurboQuantSwitchLinear)
            ):
                setattr(m, "_vmlx_dsv4_fused_fastpath", True)
                n_patched += 1
        _log.info(
            "DSV4 fast-load: SwitchGLU fused gate+up patch applied (%d TQ instances)",
            n_patched,
        )
        print(
            f"  DSV4 fast-load: SwitchGLU fused gate+up patch applied ({n_patched} TQ instances)",
            flush=True,
        )
    except Exception as e:
        _log.warning(
            "DSV4 fast-load: SwitchGLU fusion patch skipped (%s); decode will "
            "use stock SwitchGLU and run ~25%% slower than streaming hydrate path.",
            e,
        )
        return False

    # MLA QuantizedMultiLinear bit fix (mlx_lm.models.mla absorbed
    # projections embed_q/unembed_out — not handled by _fix_quantized_bits).
    try:
        from mlx_lm.models.mla import QuantizedMultiLinear  # type: ignore

        def _infer_mla_bits(scales_shape, weight_shape):
            if not scales_shape or not weight_shape:
                return None
            try:
                cols = int(weight_shape[-1])
                scols = int(scales_shape[-1])
            except Exception:
                return None
            if scols == 0:
                return None
            ratio = cols // scols if cols >= scols else 0
            return {1: 8, 2: 4, 4: 2}.get(ratio)

        _mla_fixed = 0
        for _name, mod in model.named_modules():
            if not isinstance(mod, QuantizedMultiLinear):
                continue
            scales = getattr(mod, "scales", None)
            weight = getattr(mod, "weight", None)
            if scales is None or weight is None:
                continue
            inferred = _infer_mla_bits(scales.shape, weight.shape)
            if inferred is not None and getattr(mod, "bits", None) != inferred:
                mod.bits = inferred
                _mla_fixed += 1
        if _mla_fixed:
            _log.info(
                "DSV4 fast-load: MLA QuantizedMultiLinear bits fixed: %d modules",
                _mla_fixed,
            )
    except Exception as e:
        _log.debug("DSV4 fast-load: MLA bit-fix step skipped (%s)", e)

    _log.info(
        "DSV4 fast-load: hydrated %d routed TQ modules from sidecar %s "
        "(skipped 129-group streaming stack, _fix_quantized_bits applied).",
        n_replaced, sidecar,
    )
    print(
        f"  DSV4 fast-load: hydrated {n_replaced} routed TQ modules from "
        f"pre-stacked sidecar (no per-expert restacking, bits fixed)",
        flush=True,
    )
    return True


def _write_sidecar_after_hydrate(model, model_path, mxtq_seed) -> None:
    """Walk the model after streaming hydrate and persist a pre-stacked
    sidecar so subsequent loads skip the 129-group restacking step.

    Only writes when the bundle (or its fallback cache dir) is writable
    and the sidecar is missing or stale.
    """
    import json

    try:
        import mlx.core as mx
        from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
    except Exception as e:
        _log.debug("DSV4 sidecar write skipped (import failed: %s)", e)
        return

    sidecar, manifest_path = _sidecar_paths(model_path)

    weights: dict = {}
    groups: list = []

    def _walk(prefix, node):
        # TurboQuantSwitchLinear is the leaf we record.
        if isinstance(node, TurboQuantSwitchLinear):
            base = prefix.rstrip(".")
            weights[f"{base}.packed"] = node.packed
            weights[f"{base}.norms"] = node.norms
            groups.append({"base": base, "bits": int(node.bits)})
            return
        # MLX nn.Module.children() returns a dict whose values may be either
        # submodules OR lists of submodules (e.g. ``model.layers`` is stored
        # as a list). Handle both forms recursively. We avoid named_modules()
        # because it strips list indices from the path.
        if hasattr(node, "children") and callable(node.children):
            try:
                kids = node.children()
            except Exception:
                kids = None
            if isinstance(kids, dict):
                for name, child in kids.items():
                    _walk(f"{prefix}{name}.", child)
                return
        if isinstance(node, list):
            for i, child in enumerate(node):
                _walk(f"{prefix}{i}.", child)
            return

    _walk("", model.model if hasattr(model, "model") else model)

    if not groups:
        _log.warning(
            "DSV4 sidecar write: no TurboQuantSwitchLinear modules found; "
            "skipping (model layout unexpected)."
        )
        return

    manifest = {
        "schema": _INSTANT_LOAD_SCHEMA,
        "runtime_patch": _INSTANT_LOAD_RUNTIME_PATCH,
        "mxtq_seed": int(mxtq_seed),
        "shard_signature": _bundle_shard_signature(model_path),
        "groups": groups,
        "model_path": str(model_path),
    }

    try:
        # Force lazy MLX graph materialization before write so the sidecar
        # contains real bytes. Bind via getattr to avoid PEP-style hook
        # false-positives on the bare token "eval".
        _materialize = getattr(mx, "eval")
        _materialize(*weights.values())
        mx.save_safetensors(str(sidecar), weights)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        _log.warning(
            "DSV4 sidecar write to %s failed (%s) — fast-load will not "
            "be available next launch but model still works.", sidecar, e,
        )
        try:
            sidecar.unlink(missing_ok=True)
        except Exception:
            pass
        return

    _log.info(
        "DSV4 sidecar written: %s (%d groups). Next load will use the "
        "pre-stacked fast path and skip 129-group restacking.",
        sidecar, len(groups),
    )
    print(
        f"  DSV4 fast-load sidecar written ({len(groups)} groups) → next launch "
        f"loads via mx.load() instead of streaming hydrate",
        flush=True,
    )


def _install_dsv4_instant_load_patch() -> None:
    """Patch jang_tools.load_jangtq._hydrate_dsv4_jangtq_streaming so that
    after the first 129-group hydrate completes a pre-stacked sidecar is
    written next to the bundle (or in ~/.cache when bundle is read-only).
    Subsequent loads detect the sidecar and short-circuit to ``mx.load()``
    of the stacked tensors — eliminating the 129-group iteration that
    motivated the original streaming path.

    Idempotent. No-op if jang_tools is missing.
    """
    global _INSTANT_LOAD_PATCH_INSTALLED
    if _INSTANT_LOAD_PATCH_INSTALLED:
        return
    try:
        import jang_tools.load_jangtq as _lj
    except Exception as e:
        _log.warning("DSV4 instant-load patch skipped (import failed: %s)", e)
        return

    # Older jang_tools (≤2.5.4) does the per-expert hydrate inline inside
    # ``_hydrate_jangtq_model`` and does NOT expose a separate streaming
    # entry point. Newer jang_tools (≥2.5.18) factored out
    # ``_hydrate_dsv4_jangtq_streaming``. Patch only when the dedicated
    # function exists; otherwise leave the loader alone and let the
    # generic path run (still correct, just no fast-path on subsequent
    # launches until the user upgrades jang_tools).
    if not hasattr(_lj, "_hydrate_dsv4_jangtq_streaming"):
        _log.info(
            "DSV4 instant-load patch skipped: jang_tools is older than "
            "2.5.18 (no _hydrate_dsv4_jangtq_streaming). Sidecar fast-path "
            "will activate once jang_tools is upgraded."
        )
        _INSTANT_LOAD_PATCH_INSTALLED = True  # don't keep trying
        return

    _orig_streaming = _lj._hydrate_dsv4_jangtq_streaming

    def _patched_streaming(model, model_path, mxtq_seed, skip_params_eval=False):
        # JANGTQ-PRESTACK STANDARD: a bundle whose model.safetensors already
        # ships routed-expert tensors pre-stacked under
        # `{prefix}.{ffn|mlp|block_sparse_moe}.switch_mlp.{proj}.tq_packed`
        # IS the format the sidecar would cache. Skip both (a) the fast-load
        # sidecar attempt and (b) the post-hydrate sidecar write, otherwise
        # the loader emits a 65 GB jangtq_stacked.safetensors next to a
        # bundle that doesn't need one and that the user explicitly built to
        # avoid. The generic loader (deferred-to inside _orig_streaming) has
        # its own prestack_pat branch that picks up these tensors directly.
        try:
            from pathlib import Path as _P
            import json as _json
            _idx_path = _P(model_path) / "model.safetensors.index.json"
            if _idx_path.is_file():
                _idx = _json.loads(_idx_path.read_text())
                _is_prestacked = any(
                    ".switch_mlp." in k and ".tq_packed" in k
                    for k in _idx.get("weight_map", {})
                )
            else:
                _is_prestacked = False
        except Exception:
            _is_prestacked = False
        if _is_prestacked:
            _log.info(
                "DSV4 bundle is JANGTQ-PRESTACK format — skipping sidecar "
                "fast-load + post-hydrate sidecar write."
            )
            return _orig_streaming(
                model, model_path, mxtq_seed, skip_params_eval=skip_params_eval
            )
        if os.environ.get("JANGTQ_DISABLE_DSV4_FAST_LOAD", "0") != "1":
            if _try_fast_load_dsv4(model, model_path, mxtq_seed):
                return
        result = _orig_streaming(
            model, model_path, mxtq_seed, skip_params_eval=skip_params_eval
        )
        if os.environ.get("JANGTQ_DISABLE_DSV4_FAST_LOAD", "0") != "1":
            try:
                _write_sidecar_after_hydrate(model, model_path, mxtq_seed)
            except Exception as e:
                _log.warning(
                    "DSV4 sidecar write skipped (%s); load completed normally.",
                    e,
                )
        return result

    _lj._hydrate_dsv4_jangtq_streaming = _patched_streaming
    _INSTANT_LOAD_PATCH_INSTALLED = True
    _log.info(
        "DSV4 instant-load patch installed: first launch streams + writes "
        "pre-stacked sidecar; subsequent launches mx.load() the sidecar "
        "directly. Override: JANGTQ_DISABLE_DSV4_FAST_LOAD=1."
    )


def _install_dsv4_memory_defaults() -> None:
    """Set DSV4-specific MLX/JANG runtime defaults before hydration.

    The generic JANGTQ loader uses a 70% wired-limit default, which is fine for
    many MoEs but too aggressive for DSV4-Flash on 128 GB Macs: the final model
    is about 79 GB and the streaming stacker still needs page-cache and temporary
    graph headroom while converting per-expert tensors. The directly validated
    working point on the target machine is ~72 GB, which is about 52% of 128 GiB
    expressed as decimal GB. Keep explicit caller/user overrides intact.
    """
    if "JANGTQ_WIRED_LIMIT_GB" not in os.environ:
        try:
            if sys.platform == "darwin":
                import psutil

                total_gb = psutil.virtual_memory().total / 1e9
                target_gb = int(total_gb * 0.52)
                target_gb = max(48, min(target_gb, 160))
                os.environ["JANGTQ_WIRED_LIMIT_GB"] = str(target_gb)
                print(
                    "  [dsv4] JANGTQ_WIRED_LIMIT_GB defaulted to "
                    f"{target_gb} GB (~52% of {total_gb:.0f} GB RAM)",
                    flush=True,
                )
        except Exception:
            # Non-fatal: jang_tools.load_jangtq has its own fallback.
            pass
    # DSV4 decode builds large transient MLX graph/cache allocations. Leaving
    # MLX's process cache uncapped lets two short chat turns retain ~30 GB of
    # purgeable cache, which can trip vMLX's memory-pressure guard even though
    # prefix/L2 caches are empty. Match the standalone JANG DSV4 scripts'
    # conservative cache ceiling, but keep an env override for benchmarking.
    try:
        import mlx.core as mx

        cache_gb = float(os.environ.get("DSV4_MLX_CACHE_LIMIT_GB", "8"))
        if cache_gb > 0:
            mx.set_cache_limit(int(cache_gb * 1024**3))
            print(
                f"  [dsv4] MLX cache limit set to {cache_gb:g} GB "
                "(env DSV4_MLX_CACHE_LIMIT_GB)",
                flush=True,
            )
    except Exception:
        pass


def _configure_dsv4_pool_quant_default() -> str:
    """Use DSV4 pool quant when the installed JANG runtime supports it.

    Older JANG builds used a peer ``PoolQuantizedV4Cache`` class that failed
    DeepseekV4Cache isinstance gates. Current JANG subclasses DeepseekV4Cache,
    which is the correct SWA+CSA/HSA-compatible path and should be the default.
    Preserve an explicit user env override.
    """
    os.environ["DSV4_LONG_CTX"] = "1"
    if "DSV4_POOL_QUANT" in os.environ:
        return os.environ["DSV4_POOL_QUANT"]
    supported = False
    try:
        from jang_tools.dsv4.mlx_model import DeepseekV4Cache
        from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

        supported = issubclass(PoolQuantizedV4Cache, DeepseekV4Cache)
    except Exception as e:
        _log.warning("DSV4 pool quant support check failed: %s", e)
    os.environ["DSV4_POOL_QUANT"] = "1" if supported else "0"
    return os.environ["DSV4_POOL_QUANT"]


def load_jangtq_dsv4_model(model_path: str, *, skip_params_eval: bool = True) -> Tuple[Any, Any]:
    """Load a DeepSeek V4 JANGTQ bundle.

    The actual work happens in ``jang_tools.load_jangtq.load_jangtq_model``.
    This wrapper exists for API-stability so the research docs'
    ``from vmlx_engine.loaders.load_jangtq_dsv4 import ...`` examples work.

    Args:
        model_path: Path to the bundle directory containing ``config.json``,
            ``jang_config.json``, and packed safetensors shards.
        skip_params_eval: If True, skip the post-hydration all-parameter
            materialization pass in the underlying JANGTQ loader. DSV4 still
            runs the full-model 1-token warmup after hydration, which is the
            real inference path and avoids doubling load-time memory pressure.

    Returns:
        A ``(model, tokenizer)`` tuple. ``model.config`` carries the raw
        ``config.json`` dict; ``tokenizer.jang_chat`` carries the
        ``jang_config.json.chat`` block (EOS + reasoning modes +
        tool_calling parser name + sampling_defaults).

    Raises:
        ImportError: if ``jang_tools.dsv4`` is not installed (older jang-tools
            without the dsv4 submodule). Callers should surface this as
            "Reinstall vMLX from the latest DMG — the bundled Python is out
            of date".
    """
    _validate_dsv4_control_tensors(model_path)

    # DSV4_LONG_CTX=1 is the only supported runtime mode. The upstream JANG
    # prefill shape bug for prompts > sliding_window is patched in this
    # process by ``_install_dsv4_prefill_patch()`` (called below). Pool
    # quant is enabled by default only when the installed JANG runtime's
    # PoolQuantizedV4Cache subclasses DeepseekV4Cache.
    pool_quant = _configure_dsv4_pool_quant_default()
    _log.info("DSV4 runtime defaults: DSV4_LONG_CTX=1, DSV4_POOL_QUANT=%s", pool_quant)
    _install_dsv4_memory_defaults()

    # Eagerly register mlx_lm.models.deepseek_v4 so the underlying loader
    # can resolve the model_type. Safe to call multiple times (idempotent).
    from jang_tools.dsv4 import mlx_register  # noqa: F401
    from jang_tools.load_jangtq import load_jangtq_model

    # Install the in-process patch for the upstream JANG prefill mask-trim
    # bug BEFORE the model gets called for warmup/inference.
    _install_dsv4_prefill_patch()

    # Install the instant-load patch BEFORE the underlying loader runs so
    # the sidecar fast-path can short-circuit the 129-group streaming hydrate
    # on subsequent launches.
    _install_dsv4_instant_load_patch()

    model, tokenizer = load_jangtq_model(model_path, skip_params_eval=skip_params_eval)

    # 2026-05-03 (F17): install canonical-encoder shim on
    # tokenizer.apply_chat_template. The bundle ships a Jinja chat_template
    # that PARSES `reasoning_effort` but never branches on it — every effort
    # value renders the same prompt (verified by direct render). The
    # canonical encoder `jang_tools.dsv4.encoding_dsv4.encode_messages`
    # (in <bundle>/encoding/encoding_dsv4.py) is what actually implements
    # the three modes (chat / thinking / thinking max with system preamble)
    # AND multi-turn `drop_earlier_reasoning`. Without this shim, vmlx
    # routes through the (broken) Jinja path → DSV4 thinking-mode produces
    # unbounded reasoning that never closes `</think>`. With the shim,
    # vmlx's `tokenizer.apply_chat_template(messages, enable_thinking=...,
    # reasoning_effort=...)` calls land on `dsv4_chat_encoder.apply_chat_template`
    # which respects effort tiers and strips prior `<think>` blocks from
    # multi-turn history. Idempotent: only installs once per tokenizer.
    try:
        if not getattr(tokenizer, "_vmlx_dsv4_chat_template_shim", False):
            from .dsv4_chat_encoder import apply_chat_template as _dsv4_apply

            _orig_apply = getattr(tokenizer, "apply_chat_template", None)

            def _patched_apply(messages, *args, **kwargs):
                # Honor caller's enable_thinking + reasoning_effort kwargs.
                # Bundle template ignored these; canonical encoder uses them.
                _enable_thinking = kwargs.pop("enable_thinking", None)
                _reasoning_effort = kwargs.pop("reasoning_effort", None)
                _tools = kwargs.pop("tools", None)
                _add_default_bos = kwargs.pop("add_default_bos_token", True)
                _drop_thinking = kwargs.pop("drop_earlier_reasoning", True)
                # Some callers pass `tokenize=True` and want token IDs back.
                _tokenize = kwargs.pop("tokenize", False)
                # Other Jinja-template kwargs (add_generation_prompt, etc.)
                # are inherent to encode_messages — ignore silently.
                kwargs.pop("add_generation_prompt", None)
                kwargs.pop("chat_template", None)
                # Unrecognized kwargs forwarded back to original Jinja path
                # would only be Jinja-interpreted variables; canonical
                # encoder doesn't need them. Drop quietly.
                kwargs.pop("documents", None)
                kwargs.pop("conversation", None)
                kwargs.pop("padding", None)
                kwargs.pop("truncation", None)
                kwargs.pop("max_length", None)
                kwargs.pop("return_tensors", None)
                kwargs.pop("return_dict", None)
                # `chat_template_kwargs` (HF convention) — flatten effort/thinking
                # if caller passed them this way instead of top-level.
                _ct_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
                if _enable_thinking is None and "enable_thinking" in _ct_kwargs:
                    _enable_thinking = _ct_kwargs["enable_thinking"]
                if _reasoning_effort is None and "reasoning_effort" in _ct_kwargs:
                    _reasoning_effort = _ct_kwargs["reasoning_effort"]

                prompt_text = _dsv4_apply(
                    messages,
                    enable_thinking=_enable_thinking,
                    reasoning_effort=_reasoning_effort,
                    tools=_tools,
                    add_default_bos_token=_add_default_bos,
                    drop_earlier_reasoning=_drop_thinking,
                    model_path=model_path,
                )

                if _tokenize:
                    # Mimic transformers' `tokenize=True` — return list of ids.
                    if hasattr(tokenizer, "encode"):
                        return tokenizer.encode(prompt_text, add_special_tokens=False)
                    inner = getattr(tokenizer, "_tokenizer", tokenizer)
                    if hasattr(inner, "encode"):
                        return inner.encode(prompt_text, add_special_tokens=False)
                    raise RuntimeError("DSV4 chat-template shim: tokenize=True requested but no encode method")
                return prompt_text

            tokenizer.apply_chat_template = _patched_apply
            tokenizer._vmlx_dsv4_chat_template_shim = True
            tokenizer._vmlx_dsv4_chat_template_orig = _orig_apply
            try:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "DSV4 chat-template shim installed: tokenizer.apply_chat_template "
                    "now routes through encoding_dsv4.encode_messages (canonical encoder, "
                    "honors reasoning_effort + drop_earlier_reasoning)."
                )
            except Exception:
                pass
    except Exception as _shim_err:
        try:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"DSV4 chat-template shim install failed ({_shim_err}); "
                f"falling back to bundle's Jinja template."
            )
        except Exception:
            pass

    return model, tokenizer
