"""Validate incoming JANGTQ2 arrays before module parameter replacement."""


def expected_payload(model):
    expected = {}
    for path, module in model.named_modules():
        if not getattr(module, "is_jangtq2", False):
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(module, projection)
            for name, dtype in (("tq2_packed", "uint32"), ("tq2_scales", "float16")):
                expected[f"{path}.{projection}.{name}"] = (
                    tuple(getattr(linear, name).shape), dtype
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
