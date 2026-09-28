"""Runtime bridge for JANG affine bundles stored in a Hadamard-rotated basis.

Prism's Bonsai 2 packs (and JANG bundles repacked from them) keep every
language-model matrix in a blockwise Hadamard-rotated basis: the rotation was
folded into the ternary weights offline, so the runtime must apply the
matching transform to the *activations*:

    forward module (Linear):   y = qmm(H(signs * x), W_rot)
    inverse module (Embedding): e = signs * H(dequant(W_rot)[ids])

``H`` is the normalized Sylvester-Walsh-Hadamard transform over blocks of
``block_size`` along the last axis, ``signs`` is a per-input-column +-1
vector stored as ``<module>.signs`` in the bundle. A plain affine loader that
skips the transform returns wrong output silently, so bundles declare the
contract in ``config.json["hadamard"]`` and this module fails closed when any
declared module or sign vector is missing.

The compute mirrors Prism's reference runtime exactly (float32 transform,
cast back to the activation dtype).
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn

SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
CONTRACT = "prism.hadamard.v1"
TRANSFORM = "normalized-sylvester-walsh-hadamard"
_PREFILL_DQ_CALLS = itertools.count(1)


class HadamardSpec:
    __slots__ = ("block_size", "forward", "inverse", "compute_dtype")

    def __init__(self, block_size: int, forward: list[str], inverse: list[str], compute_dtype):
        self.block_size = block_size
        self.forward = forward
        self.inverse = inverse
        self.compute_dtype = compute_dtype

    @property
    def modules(self) -> list[str]:
        return list(self.forward) + list(self.inverse)


def hadamard_spec_from_config(
    config: Mapping[str, Any] | None,
    jang_config: Mapping[str, Any] | None = None,
) -> HadamardSpec | None:
    """Return the declared Hadamard contract, or None for ordinary bundles."""
    raw = None
    for owner in (config, jang_config):
        if isinstance(owner, Mapping) and "hadamard" in owner:
            declared = owner["hadamard"]
            if not isinstance(declared, Mapping):
                raise ValueError("Hadamard contract must be an object")
            if raw is not None and declared != raw:
                raise ValueError("config and JANG Hadamard contracts disagree")
            raw = declared
    if raw is None:
        runtime = (jang_config or {}).get("runtime", {})
        if isinstance(runtime, Mapping) and runtime.get("requires_hadamard_activation_transform"):
            raise ValueError("Hadamard runtime is required but its contract is missing")
        return None
    if raw.get("contract") != CONTRACT:
        raise ValueError(f"unsupported Hadamard contract {raw.get('contract')!r}")
    if raw.get("transform", TRANSFORM) != TRANSFORM:
        raise ValueError(f"unsupported Hadamard transform {raw.get('transform')!r}")
    if raw.get("axis", "input-last-dimension") != "input-last-dimension":
        raise ValueError("unsupported Hadamard axis")
    if raw.get("sign_mode", "explicit") != "explicit":
        raise ValueError("Hadamard bundles must carry explicit sign vectors")
    block = raw.get("block_size")
    if type(block) is not int or block not in SUPPORTED_BLOCKS:
        raise ValueError(f"unsupported Hadamard block size {block}")
    def paths(key):
        values = raw.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(p, str) or not p or any(not part for part in p.split("."))
            for p in values
        ):
            raise ValueError(f"invalid Hadamard {key}")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate Hadamard {key}")
        return list(values)
    forward, inverse = paths("forward_modules"), paths("inverse_modules")
    if not forward and not inverse:
        raise ValueError("Hadamard contract declares no modules")
    if set(forward) & set(inverse):
        raise ValueError("Hadamard forward/inverse manifests overlap")
    dtype_name = str(raw.get("compute_dtype", "float32"))
    if dtype_name != "float32":
        raise ValueError(f"unsupported Hadamard compute dtype {dtype_name!r}")
    if raw.get("signs_dtype", "float32") != "float32" or raw.get("signs_tensor_suffix", "signs") != "signs":
        raise ValueError("Hadamard signs must be float32 .signs tensors")
    return HadamardSpec(block, forward, inverse, mx.float32)


def hadamard_activation(x: mx.array, block: int, signs: mx.array, *, inverse: bool = False, compute_dtype=mx.float32) -> mx.array:
    """Apply ``H(signs*x)`` (forward) or ``signs*H(x)`` (inverse) blockwise."""
    shape, dtype = x.shape, x.dtype
    if tuple(signs.shape) != (shape[-1],):
        raise ValueError("Hadamard signs must match the activation width exactly")
    if shape[-1] % block:
        raise ValueError(f"Hadamard block {block} does not divide activation width {shape[-1]}")
    if compute_dtype == mx.float32 and os.environ.get("VMLX_BONSAI_FUSED_HADAMARD") == "1":
        from vmlx_engine.metal.bonsai_hadamard_rotation import rotate

        fused = rotate(x, signs, block, inverse=inverse)
        if fused is not None:
            return fused
    x = x.astype(compute_dtype)
    s = signs.astype(compute_dtype)
    if not inverse:
        x = x * s
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * s
    return x.astype(dtype)


class HadamardQuantizedLinear(nn.QuantizedLinear):
    """QuantizedLinear whose input is rotated into the stored weight basis."""

    def __call__(self, x):
        activation_dtype = getattr(self, "hadamard_activation_dtype", None)
        if getattr(self, "hadamard_prefill_only", False):
            activation_dtype = mx.float16 if math.prod(x.shape[:-1]) >= 2048 else None
        if activation_dtype is not None:
            x = x.astype(activation_dtype)
        x = hadamard_activation(x, self.hadamard_block, self.signs, compute_dtype=self.hadamard_compute_dtype)
        if os.environ.get("VMLX_BONSAI_FP16_QMM") == "1" and self.scales.dtype == mx.float16:
            lowered = x.astype(mx.float16)
            rows = math.prod(lowered.shape[:-1])
            if getattr(self, "_bonsai_ane_hybrid", False):
                from vmlx_engine.metal.bonsai_ane_prefill import maybe_apply

                accelerated = maybe_apply(self, lowered)
                if accelerated is not None:
                    return accelerated.astype(activation_dtype or mx.float32)
            if (
                os.environ.get("VMLX_BONSAI_PREFILL_DQ_GEMM") == "1"
                and rows >= 2048
                and self.bits == 2
                and self.group_size == 128
            ):
                dense = mx.dequantize(
                    self.weight, self.scales, self.biases,
                    group_size=128, bits=2, mode="affine", dtype=mx.float16,
                )
                out = mx.matmul(lowered, dense.T)
                if "bias" in self:
                    out = out + self.bias
                cadence = int(os.environ.get("VMLX_BONSAI_PREFILL_DQ_EVAL_EVERY", "8"))
                if not 1 <= cadence <= 32:
                    raise ValueError("Bonsai prefill dequant eval cadence must be 1..32")
                if next(_PREFILL_DQ_CALLS) % cadence == 0:
                    mx.eval(out)
            else:
                out = None
                if (
                    os.environ.get("VMLX_BONSAI_TERNARY_VERIFY") == "1"
                    and getattr(self, "hadamard_exact_ternary", False)
                ):
                    from vmlx_engine.metal.bonsai_ternary_verify import (
                        maybe_ternary_verify,
                    )

                    out = maybe_ternary_verify(lowered, self)
                if out is None:
                    out = super().__call__(lowered)
            return out.astype(activation_dtype or mx.float32)
        out = super().__call__(x)
        return out.astype(activation_dtype) if activation_dtype is not None else out


class HadamardQuantizedEmbedding(nn.QuantizedEmbedding):
    """QuantizedEmbedding whose rows are rotated back into the model basis."""

    def __call__(self, x):
        out = super().__call__(x)
        activation_dtype = getattr(self, "hadamard_activation_dtype", None)
        if getattr(self, "hadamard_prefill_only", False):
            activation_dtype = mx.float16 if x.size >= 2048 else None
        if activation_dtype is not None:
            out = out.astype(activation_dtype)
        return hadamard_activation(out, self.hadamard_block, self.signs, inverse=True, compute_dtype=self.hadamard_compute_dtype)

    def as_linear(self, x):
        raise NotImplementedError("Hadamard-rotated embeddings cannot serve as a tied output head")


class _HadamardActivationRMSNorm(nn.RMSNorm):
    def __call__(self, x):
        # Preserve checkpoint weights and FP32 norm arithmetic, not its
        # accidental promotion of the residual stream and attention KV.
        return super().__call__(x).astype(mx.float16)


class _HadamardActivationConv1d(nn.Conv1d):
    def __call__(self, x):
        return super().__call__(x).astype(mx.float16)


class _HadamardPrefillOnlyRMSNorm(nn.RMSNorm):
    def __call__(self, x):
        out = super().__call__(x)
        return out.astype(mx.float16) if math.prod(x.shape[:-1]) >= 2048 else out


class _HadamardPrefillOnlyConv1d(nn.Conv1d):
    def __call__(self, x):
        out = super().__call__(x)
        return out.astype(mx.float16) if math.prod(x.shape[:-1]) >= 2048 else out


def configure_hadamard_activation_precision(model, config, *, enabled=True, prefill_only=False):
    """Scope FP16 activations to the declared Qwen Hadamard language graph.

    This is a numerical execution policy, NOT an SSD serialization cast.
    Hadamard accumulation/signs, norm/conv weights, A_log/dt_bias and recurrent
    GDN state remain in their native precision. Vision modules are untouched.
    The execution marker separates old/native and FP16 prefix namespaces.
    """
    if config.get("model_type") != "qwen3_5" or hadamard_spec_from_config(config) is None:
        return {}
    if prefill_only and not enabled:
        raise ValueError("Hadamard FP16 prefill-only policy requires enabled activation precision")
    if (
        os.environ.get("VMLX_BONSAI_FP16_QMM") != "1"
        and any(os.environ.get(name) == "1" for name in (
            "VMLX_BONSAI_PREFILL_DQ_GEMM", "VMLX_BONSAI_TERNARY_VERIFY",
        ))
    ):
        raise ValueError("Bonsai prefill DQ GEMM and ternary verification require FP16 QMM")
    language = getattr(model, "language_model", None)
    if language is None:
        raise ValueError("Qwen Hadamard activation policy requires its language graph")
    modules = list(language.named_modules())
    packed = [module for _, module in modules
              if isinstance(module, (HadamardQuantizedLinear, HadamardQuantizedEmbedding))]
    if not packed:
        raise ValueError("Qwen Hadamard activation policy requires installed wrappers")
    if enabled and any(module.scales.dtype != mx.float16 for module in packed):
        raise ValueError("FP16 Hadamard activation policy requires FP16 affine scales")
    signature = (
        "qwen-hadamard-fp16-prefill-only-v1" if prefill_only
        else "qwen-hadamard-fp16-v1" if enabled else "qwen-hadamard-native-v1"
    )
    numerical_flags = (
        ("fused-hadamard", "VMLX_BONSAI_FUSED_HADAMARD"),
        ("qmm-fp16", "VMLX_BONSAI_FP16_QMM"),
        ("dq-gemm", "VMLX_BONSAI_PREFILL_DQ_GEMM"),
        ("ternary-verify", "VMLX_BONSAI_TERNARY_VERIFY"),
        ("gdn-fp32", "VMLX_BONSAI_GDN_FP32"),
        ("ane-prefill", "VMLX_BONSAI_ANE_PREFILL"),
    )
    for label, variable in numerical_flags:
        if os.environ.get(variable) == "1":
            signature += f"+{label}"
    if os.environ.get("VMLX_BONSAI_ANE_PREFILL") == "1":
        signature += (
            f"-cpu{os.environ.get('VMLX_BONSAI_ANE_CPU_FRACTION', '0.1')}"
            f"-down{os.environ.get('VMLX_BONSAI_ANE_DOWN_CPU_FRACTION', '0.2')}"
            f"-gdn{os.environ.get('VMLX_BONSAI_ANE_GDN', '1')}"
        )
    previous = getattr(model, "_vmlx_hadamard_activation_precision", None)
    if previous is not None and previous != signature:
        raise ValueError("Hadamard activation policy cannot change on a loaded model")
    counts = {"projections": 0, "norms": 0, "convolutions": 0}
    if enabled:
        for _, module in modules:
            if isinstance(module, (HadamardQuantizedLinear, HadamardQuantizedEmbedding)):
                module.hadamard_activation_dtype = mx.float16
                module.hadamard_prefill_only = prefill_only
                if isinstance(module, HadamardQuantizedLinear) and os.environ.get(
                    "VMLX_BONSAI_TERNARY_VERIFY"
                ) == "1":
                    module.hadamard_exact_ternary = bool(
                        module.bits == 2
                        and module.group_size == 128
                        and module.scales.dtype == mx.float16
                        and module.biases.dtype == mx.float16
                        and mx.all(module.biases == -module.scales).item()
                        and mx.all(
                            (module.weight & (module.weight >> 1) & 0x55555555) == 0
                        ).item()
                    )
                counts["projections"] += 1
            elif type(module) is nn.RMSNorm:
                module.__class__ = (
                    _HadamardPrefillOnlyRMSNorm if prefill_only else _HadamardActivationRMSNorm
                )
                counts["norms"] += 1
            elif type(module) is nn.Conv1d:
                module.__class__ = (
                    _HadamardPrefillOnlyConv1d if prefill_only else _HadamardActivationConv1d
                )
                counts["convolutions"] += 1
    for owner in (model, language, getattr(language, "model", None)):
        if owner is not None:
            owner._vmlx_hadamard_activation_precision = signature
    return {"signature": signature, **counts}


def _resolve(model, path: str):
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    child = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
    return parent, leaf, child


def _wrap(module: nn.Module, cls, block: int, width: int, compute_dtype) -> nn.Module:
    """Rebuild ``module`` as ``cls`` without re-materializing its weights."""
    new = cls.__new__(cls)
    nn.Module.__init__(new)
    for attr in ("group_size", "bits", "mode", "num_embeddings", "dims"):
        if hasattr(module, attr):
            setattr(new, attr, getattr(module, attr))
    for key, value in module.items():
        new[key] = value
    new.hadamard_block = block
    new.hadamard_width = width
    new.hadamard_compute_dtype = compute_dtype
    # Placeholder that can never pass verification: loaded signs are +-1.
    new.signs = mx.zeros((width,), dtype=mx.float32)
    new.freeze()
    return new


def install_hadamard_modules(model, spec: HadamardSpec) -> int:
    """Swap declared quantized modules for their Hadamard-aware subclasses."""
    installed = 0
    for path in spec.forward:
        parent, leaf, child = _resolve(model, path)
        if not isinstance(child, nn.QuantizedLinear):
            raise RuntimeError(f"Hadamard forward module {path!r} is {type(child).__name__}, not QuantizedLinear")
        width = child.scales.shape[-1] * child.group_size
        if width % spec.block_size:
            raise RuntimeError(f"Hadamard module {path!r} width {width} not divisible by block {spec.block_size}")
        wrapped = _wrap(child, HadamardQuantizedLinear, spec.block_size, width, spec.compute_dtype)
        if leaf.isdigit():
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
        installed += 1
    for path in spec.inverse:
        parent, leaf, child = _resolve(model, path)
        if not isinstance(child, nn.QuantizedEmbedding):
            raise RuntimeError(f"Hadamard inverse module {path!r} is {type(child).__name__}, not QuantizedEmbedding")
        width = child.scales.shape[-1] * child.group_size
        if width % spec.block_size:
            raise RuntimeError(f"Hadamard module {path!r} width {width} not divisible by block {spec.block_size}")
        wrapped = _wrap(child, HadamardQuantizedEmbedding, spec.block_size, width, spec.compute_dtype)
        if leaf.isdigit():
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
        installed += 1
    return installed


def verify_hadamard_signs_loaded(model, spec: HadamardSpec) -> int:
    """Fail closed unless every declared module received a +-1 sign vector."""
    checked = 0
    bad: list[str] = []
    for path in spec.modules:
        _, _, child = _resolve(model, path)
        signs = getattr(child, "signs", None)
        if signs is None or not isinstance(child, (HadamardQuantizedLinear, HadamardQuantizedEmbedding)):
            bad.append(f"{path}: not a Hadamard module after load")
            continue
        if signs.dtype != mx.float32 or tuple(signs.shape) != (child.hadamard_width,):
            bad.append(f"{path}: signs dtype/width does not match the contract")
            continue
        if child.scales.shape[-1] * child.group_size != child.hadamard_width:
            bad.append(f"{path}: loaded quantization width changed")
            continue
        ok = mx.all((signs == 1) | (signs == -1)).item()
        if not ok:
            bad.append(f"{path}: signs are not all +-1 (missing from shards?)")
            continue
        checked += 1
    if bad:
        raise RuntimeError(
            "JANG Hadamard bundle failed sign verification for "
            f"{len(bad)}/{len(spec.modules)} modules: {bad[:5]}"
        )
    return checked
