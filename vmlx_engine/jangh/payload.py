"""Validate incoming JANGTQ2 arrays before module parameter replacement."""


def expected_payload(model):
    expected = {}
    for path, module in model.named_modules():
        if getattr(module, "is_jangtq2_dense", False):          # dense one-expert projection (jangh/dense.py)
            expected[f"{path}.tq2_packed"] = ((1, module.output_dims, module.input_dims * module.bits // 32), "uint32")
            expected[f"{path}.tq2_scales"] = ((1, module.output_dims), "float16")
            continue
        mixed = getattr(module, "is_jangt", False)                  # JANGT mixed stack: only its JANGH projections
        if not getattr(module, "is_jangtq2", False) and not mixed:
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(module, projection)
            if mixed and not hasattr(linear, "tq2_packed"):
                continue
            # Generic text loaders may already have bound shard arrays. Their
            # shapes are not an independent oracle for validating that shard.
            shapes = {
                "tq2_packed": (linear.num_experts, linear.output_dims,
                               linear.input_dims * linear.bits // 32),
                "tq2_scales": (linear.num_experts, linear.output_dims),
            }
            for name, dtype in (("tq2_packed", "uint32"), ("tq2_scales", "float16")):
                expected[f"{path}.{projection}.{name}"] = (
                    shapes[name], dtype
                )
    return expected


def validate_payload(model, weights):
    expected = expected_payload(model)
    seen = set()
    for key, value in weights:
        if not key.endswith((".tq2_packed", ".tq2_scales")):
            continue
        if key not in expected:
            raise ValueError(f"jangtq2: unrecognized payload {key}")
        seen.add(key)
        shape, dtype = expected[key]
        actual_dtype = str(getattr(value, "dtype", "")).removeprefix("mlx.core.")
        if tuple(getattr(value, "shape", ())) != shape or actual_dtype != dtype:
            raise ValueError(f"jangtq2: invalid shape or dtype for {key}; expected {shape} {dtype}")
    return seen


def validate_complete_payload(model, seen):
    missing = set(expected_payload(model)) - set(seen)
    if missing:
        raise ValueError(f"jangtq2: missing {len(missing)} payload tensors: {sorted(missing)[:3]}")
