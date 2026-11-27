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

import math
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
CONTRACT = "prism.hadamard.v1"
TRANSFORM = "normalized-sylvester-walsh-hadamard"

_FWHT_1024_BODY = r"""
  // Fused FP32 Sylvester-Walsh-Hadamard (B=1024) + signs mul + dtype conversion.
  // 128 threads (4 simdgroups x 32 lanes) per 1024-element block, ONE barrier.
  const uint blk = threadgroup_position_in_grid.x;
  const uint tid = thread_index_in_threadgroup;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint blocks_per_row = params[0];
  const uint col0 = (blk % blocks_per_row) * 1024u;
  const size_t blk_base = size_t(blk) * 1024u;

  threadgroup float smem[32 * 33];

  // Phase 1: low 5 bits (stride 1..16) in registers via simd_shuffle_xor
  #pragma unroll
  for (uint j = 0; j < 8; j++) {
    uint hi = sg * 8u + j;
    uint idx = hi * 32u + lane;
    float v = float(x[blk_base + idx]);
    if (!INVERSE) {
      v *= signs[col0 + idx];
    }
    #pragma unroll
    for (uint s = 1; s < 32; s <<= 1) {
      float o = simd_shuffle_xor(v, s);
      v = (lane & s) ? (o - v) : (v + o);
    }
    smem[hi * 33u + lane] = v;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: high 5 bits (stride 32..512) in registers via simd_shuffle_xor
  constexpr float INV_SQRT = 0.03125f;
  #pragma unroll
  for (uint j = 0; j < 8; j++) {
    uint lo = sg * 8u + j;
    float v = smem[lane * 33u + lo];
    #pragma unroll
    for (uint s = 1; s < 32; s <<= 1) {
      float o = simd_shuffle_xor(v, s);
      v = (lane & s) ? (o - v) : (v + o);
    }
    v *= INV_SQRT;
    uint idx = lane * 32u + lo;
    if (INVERSE) {
      v *= signs[col0 + idx];
    }
    out[blk_base + idx] = T(v);
  }
"""

_FWHT_512_BODY = r"""
  // Fused FP32 Sylvester-Walsh-Hadamard (B=512 = 16 x 32) + signs + dtype conversion.
  // 64 threads (2 simdgroups x 32 lanes) per 512-element block, ONE barrier.
  const uint blk = threadgroup_position_in_grid.x;
  const uint tid = thread_index_in_threadgroup;
  const uint sg = tid >> 5;
  const uint lane = tid & 31u;
  const uint blocks_per_row = params[0];
  const uint col0 = (blk % blocks_per_row) * 512u;
  const size_t blk_base = size_t(blk) * 512u;

  threadgroup float smem[16 * 33];

  #pragma unroll
  for (uint j = 0; j < 8; j++) {
    uint hi = sg * 8u + j;
    uint idx = hi * 32u + lane;
    float v = float(x[blk_base + idx]);
    if (!INVERSE) {
      v *= signs[col0 + idx];
    }
    #pragma unroll
    for (uint s = 1; s < 32; s <<= 1) {
      float o = simd_shuffle_xor(v, s);
      v = (lane & s) ? (o - v) : (v + o);
    }
    smem[hi * 33u + lane] = v;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr float INV_SQRT = 0.04419417382415922f;  // 1 / sqrt(512)
  const uint hi = lane & 15u;
  const uint half_sg = lane >> 4;
  #pragma unroll
  for (uint j = 0; j < 8; j++) {
    uint lo = sg * 16u + j * 2u + half_sg;
    float v = smem[hi * 33u + lo];
    #pragma unroll
    for (uint s = 1; s < 16; s <<= 1) {
      float o = simd_shuffle_xor(v, s);
      v = (hi & s) ? (o - v) : (v + o);
    }
    v *= INV_SQRT;
    uint idx = hi * 32u + lo;
    if (INVERSE) {
      v *= signs[col0 + idx];
    }
    out[blk_base + idx] = T(v);
  }
"""


@lru_cache(maxsize=16)
def _fused_hadamard_kernel(block: int, inverse: bool, dtype_name: str):
    if block == 1024:
        src = _FWHT_1024_BODY
    elif block == 512:
        src = _FWHT_512_BODY
    else:
        return None
    return mx.fast.metal_kernel(
        name=f"vmlx_hadamard_b{block}_{'inv' if inverse else 'fwd'}_{dtype_name}",
        input_names=["x", "signs", "params"],
        output_names=["out"],
        source=src,
    )


def _fused_hadamard_enabled() -> bool:
    return os.environ.get("VMLX_FUSED_HADAMARD", "1").strip().lower() not in {"0", "false", "no", "off"}


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
    if (
        _fused_hadamard_enabled()
        and compute_dtype == mx.float32
        and block in (512, 1024)
        and dtype in (mx.float16, mx.bfloat16, mx.float32)
        and x.size > 0
    ):
        dtype_name = {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}[dtype]
        kernel = _fused_hadamard_kernel(block, bool(inverse), dtype_name)
        if kernel is not None:
            x_c = mx.contiguous(x)
            s_c = mx.contiguous(signs if signs.dtype == mx.float32 else signs.astype(mx.float32))
            blocks_per_row = shape[-1] // block
            num_blocks = x_c.size // block
            tpg = 128 if block == 1024 else 64
            (out,) = kernel(
                inputs=[x_c, s_c, mx.array([blocks_per_row], dtype=mx.uint32)],
                template=[("T", dtype), ("INVERSE", bool(inverse))],
                grid=(num_blocks * tpg, 1, 1),
                threadgroup=(tpg, 1, 1),
                output_shapes=[shape],
                output_dtypes=[dtype],
            )
            return out
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
        from vmlx_engine.metal.row_exact_qmv import row_exact_qmv, row_exact_scope_active

        activation_dtype = getattr(self, "hadamard_activation_dtype", None)
        if activation_dtype is not None:
            x = x.astype(activation_dtype)
        x = hadamard_activation(x, self.hadamard_block, self.signs, compute_dtype=self.hadamard_compute_dtype)
        if row_exact_scope_active():
            exact = row_exact_qmv(
                x, self.weight, self.scales, getattr(self, "biases", None),
                group_size=self.group_size, bits=self.bits, mode=self.mode,
            )
            if exact is not None:
                return exact.astype(activation_dtype) if activation_dtype is not None else exact
        out = super().__call__(x)
        return out.astype(activation_dtype) if activation_dtype is not None else out


class HadamardProjectionGroup(nn.Module):
    """Exact sibling group for HadamardQuantizedLinear projections with identical signs.

    Applies ``hadamard_activation`` ONCE to the shared input and runs a single
    grouped ``mx.quantized_matmul`` (with row-exact decode parity during
    speculative verify), then splits the packed output along the last axis.
    """

    def __init__(self, linears: Sequence[HadamardQuantizedLinear]):
        super().__init__()
        linears = tuple(linears)
        reason = hadamard_projection_group_reason(linears)
        if reason is not None:
            raise ValueError(reason)
        first = linears[0]
        self.group_size = int(first.group_size)
        self.bits = int(first.bits)
        self.mode = str(first.mode)
        self.hadamard_block = int(first.hadamard_block)
        self.hadamard_width = int(first.hadamard_width)
        self.hadamard_compute_dtype = first.hadamard_compute_dtype
        self.hadamard_activation_dtype = getattr(first, "hadamard_activation_dtype", None)
        self.signs = first.signs
        self.weight = mx.concatenate([linear.weight for linear in linears], axis=0)
        self.scales = mx.concatenate([linear.scales for linear in linears], axis=0)
        self.biases = (
            mx.concatenate([linear.biases for linear in linears], axis=0)
            if self.mode == "affine"
            else None
        )
        splits: list[int] = []
        offset = 0
        for linear in linears[:-1]:
            offset += int(linear.weight.shape[0])
            splits.append(offset)
        self.output_dims = offset + int(linears[-1].weight.shape[0])
        self.split_indices = tuple(splits)
        self.freeze()

    def _rotate(self, x: mx.array) -> mx.array:
        if self.hadamard_activation_dtype is not None:
            x = x.astype(self.hadamard_activation_dtype)
        return hadamard_activation(
            x, self.hadamard_block, self.signs, compute_dtype=self.hadamard_compute_dtype
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        from vmlx_engine.metal.row_exact_qmv import row_exact_qmv, row_exact_scope_active

        x_rot = self._rotate(x)
        if row_exact_scope_active():
            exact = row_exact_qmv(
                x_rot, self.weight, self.scales, self.biases,
                group_size=self.group_size, bits=self.bits, mode=self.mode,
            )
            if exact is not None:
                if self.hadamard_activation_dtype is not None:
                    exact = exact.astype(self.hadamard_activation_dtype)
                return tuple(mx.split(exact, self.split_indices, axis=-1))
        out = mx.quantized_matmul(
            x_rot,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if self.hadamard_activation_dtype is not None:
            out = out.astype(self.hadamard_activation_dtype)
        return tuple(mx.split(out, self.split_indices, axis=-1))


def hadamard_projection_group_reason(linears: Sequence[nn.Module]) -> str | None:
    """Return why Hadamard projections cannot share one exact rotation + QMM."""
    linears = tuple(linears)
    if len(linears) < 2:
        return "need at least two projections"
    if not all(isinstance(linear, HadamardQuantizedLinear) for linear in linears):
        return "projection is not HadamardQuantizedLinear"
    first = linears[0]
    first_act = getattr(first, "hadamard_activation_dtype", None)
    for linear in linears[1:]:
        if int(linear.hadamard_block) != int(first.hadamard_block):
            return "Hadamard block sizes differ"
        if int(linear.hadamard_width) != int(first.hadamard_width):
            return "Hadamard widths differ"
        if linear.hadamard_compute_dtype != first.hadamard_compute_dtype:
            return "Hadamard compute dtypes differ"
        if getattr(linear, "hadamard_activation_dtype", None) != first_act:
            return "Hadamard activation dtypes differ"
        if (
            int(linear.bits) != int(first.bits)
            or int(linear.group_size) != int(first.group_size)
            or str(linear.mode) != str(first.mode)
        ):
            return "quantization parameters differ"
        if "bias" in linear or "bias" in first:
            return "post-matmul bias is unsupported"
        if (
            linear.weight.dtype != first.weight.dtype
            or linear.scales.dtype != first.scales.dtype
            or (
                str(first.mode) == "affine"
                and (
                    getattr(linear, "biases", None) is None
                    or getattr(first, "biases", None) is None
                    or linear.biases.dtype != first.biases.dtype
                )
            )
        ):
            return "quantized tensor dtypes differ"
    # Verify exact sign identity across all siblings in one batched check.
    same_signs = [mx.array_equal(first.signs, linear.signs) for linear in linears[1:]]
    if not all(bool(x.item()) for x in same_signs):
        return "Hadamard sign vectors differ"
    return None


def prepare_hadamard_projection_groups(model: nn.Module) -> dict[str, int]:
    """Group sibling Hadamard projections that share identical signs, replacing originals.

    Also groups unquantized GDN (in_proj_b, in_proj_a) and casts them (plus
    conv1d.weight) to FP16 when the FP16 activation policy is active, keeping
    resident memory neutral or lower.
    """
    if os.environ.get("VMLX_HADAMARD_PROJECTION_GROUPS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {}
    fp16_policy = getattr(model, "_vmlx_hadamard_activation_precision", "") == "qwen-hadamard-fp16-v1"
    counts = {"attn_qkv": 0, "mlp_gate_up": 0, "gdn_qkv_z": 0, "gdn_ba": 0, "conv1d_fp16": 0}
    seen: set[int] = set()
    to_eval: list[mx.array] = []

    for _, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))

        # Full-attention (q_proj, k_proj, v_proj)
        if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj")):
            q, k, v = module.q_proj, module.k_proj, module.v_proj
            if hadamard_projection_group_reason((q, k, v)) is None:
                group = HadamardProjectionGroup((q, k, v))
                to_eval.extend(t for t in (group.weight, group.scales, group.biases) if t is not None)
                module._vmlx_hadamard_qkv_group = group
                module.q_proj = None
                module.k_proj = None
                module.v_proj = None
                counts["attn_qkv"] += 1

        # Dense MLP (gate_proj, up_proj)
        if (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and not hasattr(module, "switch_mlp")
        ):
            g, u = module.gate_proj, module.up_proj
            if hadamard_projection_group_reason((g, u)) is None:
                group = HadamardProjectionGroup((g, u))
                to_eval.extend(t for t in (group.weight, group.scales, group.biases) if t is not None)
                module._vmlx_hadamard_gate_up_group = group
                module.gate_proj = None
                module.up_proj = None
                counts["mlp_gate_up"] += 1

        # GatedDeltaNet (in_proj_qkv, in_proj_z) + unquantized (in_proj_b, in_proj_a)
        if all(hasattr(module, attr) for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")):
            qkv, z = module.in_proj_qkv, module.in_proj_z
            if hadamard_projection_group_reason((qkv, z)) is None:
                group = HadamardProjectionGroup((qkv, z))
                to_eval.extend(t for t in (group.weight, group.scales, group.biases) if t is not None)
                module._vmlx_hadamard_qkv_z_group = group
                module.in_proj_qkv = None
                module.in_proj_z = None
                counts["gdn_qkv_z"] += 1

                b_lin, a_lin = module.in_proj_b, module.in_proj_a
                if (
                    type(b_lin) is nn.Linear
                    and type(a_lin) is nn.Linear
                    and "bias" not in b_lin
                    and "bias" not in a_lin
                ):
                    target_dtype = mx.float16 if fp16_policy else b_lin.weight.dtype
                    bw = b_lin.weight.astype(target_dtype)
                    aw = a_lin.weight.astype(target_dtype)
                    ba_w = mx.concatenate([bw, aw], axis=0)
                    b_out = int(bw.shape[0])
                    ba_lin = nn.Linear(int(ba_w.shape[1]), int(ba_w.shape[0]), bias=False)
                    ba_lin.weight = ba_w
                    ba_lin.freeze()
                    to_eval.append(ba_w)
                    module._vmlx_ba_linear = ba_lin
                    module._vmlx_ba_split = b_out
                    module.in_proj_b = None
                    module.in_proj_a = None
                    counts["gdn_ba"] += 1

                    # Provide _project_inputs so both vendored and patched GDN paths use it automatically.
                    def _make_project_inputs(mod):
                        def _project_inputs(inputs):
                            qkv_out, z_out = mod._vmlx_hadamard_qkv_z_group(inputs)
                            ba_in = inputs.astype(mod._vmlx_ba_linear.weight.dtype)
                            ba_out = mod._vmlx_ba_linear(ba_in)
                            split = mod._vmlx_ba_split
                            return qkv_out, z_out, ba_out[..., :split], ba_out[..., split:]
                        return _project_inputs

                    module._project_inputs = _make_project_inputs(module)
                else:
                    def _make_project_inputs_qkv_z(mod):
                        def _project_inputs(inputs):
                            qkv_out, z_out = mod._vmlx_hadamard_qkv_z_group(inputs)
                            return qkv_out, z_out, mod.in_proj_b(inputs), mod.in_proj_a(inputs)
                        return _project_inputs

                    module._project_inputs = _make_project_inputs_qkv_z(module)

            if fp16_policy and hasattr(module, "conv1d") and hasattr(module.conv1d, "weight"):
                if module.conv1d.weight.dtype != mx.float16:
                    module.conv1d.weight = module.conv1d.weight.astype(mx.float16)
                    to_eval.append(module.conv1d.weight)
                    counts["conv1d_fp16"] += 1

    if to_eval:
        mx.eval(*to_eval)
        mx.clear_cache()
    return counts


class HadamardQuantizedEmbedding(nn.QuantizedEmbedding):
    """QuantizedEmbedding whose rows are rotated back into the model basis."""

    def __call__(self, x):
        out = super().__call__(x)
        activation_dtype = getattr(self, "hadamard_activation_dtype", None)
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


def configure_hadamard_activation_precision(model, config, *, enabled=True):
    """Scope FP16 activations to the declared Qwen Hadamard language graph.

    This is a numerical execution policy, NOT an SSD serialization cast.
    Hadamard accumulation/signs, norm/conv weights, A_log/dt_bias and recurrent
    GDN state remain in their native precision. Vision modules are untouched.
    The execution marker separates old/native and FP16 prefix namespaces.
    """
    if config.get("model_type") != "qwen3_5" or hadamard_spec_from_config(config) is None:
        return {}
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
    signature = "qwen-hadamard-fp16-v1" if enabled else "qwen-hadamard-native-v1"
    previous = getattr(model, "_vmlx_hadamard_activation_precision", None)
    if previous is not None and previous != signature:
        raise ValueError("Hadamard activation policy cannot change on a loaded model")
    counts = {"projections": 0, "norms": 0, "convolutions": 0}
    if enabled:
        for _, module in modules:
            if isinstance(module, (HadamardQuantizedLinear, HadamardQuantizedEmbedding)):
                module.hadamard_activation_dtype = mx.float16
                counts["projections"] += 1
            elif type(module) is nn.RMSNorm:
                module.__class__ = _HadamardActivationRMSNorm
                counts["norms"] += 1
            elif type(module) is nn.Conv1d:
                module.__class__ = _HadamardActivationConv1d
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
    pending: list[tuple[str, mx.array]] = []
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
        pending.append((path, mx.all((signs == 1) | (signs == -1))))
    if pending:
        stacked = mx.stack([flag for _, flag in pending])
        if bool(mx.all(stacked).item()):
            checked = len(pending)
        else:
            results = stacked.tolist()
            for (path, _), ok in zip(pending, results):
                if ok:
                    checked += 1
                else:
                    bad.append(f"{path}: signs are not all +-1 (missing from shards?)")
    if bad:
        raise RuntimeError(
            "JANG Hadamard bundle failed sign verification for "
            f"{len(bad)}/{len(spec.modules)} modules: {bad[:5]}"
        )
    return checked
