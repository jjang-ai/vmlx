# SPDX-License-Identifier: Apache-2.0
"""Shared GLM prefill lifetime policy for execution and cache identity."""

import os
from functools import lru_cache


GLM5_REGISTER_SUM_MATH_ABI = "mlx0322_row128_v1"


@lru_cache(maxsize=1)
def _register_sum_default_qualified() -> bool:
    """Automatic selection is limited to the repeatedly qualified runtime.

    Other devices/MLX versions retain stock behavior. The kernel still checks
    actual operand dtype/geometry; this is not a quantization-folder allowlist.
    Import lazily so policy/namespace inspection never requires a GPU build.
    """
    try:
        import importlib.metadata
        import mlx.core as mx

        return (
            importlib.metadata.version("mlx") in ("0.32.2", "0.32.3", "0.32.5.dev20261101+360078a8")
            and mx.metal.is_available()
            and mx.device_info().get("device_name") == "Apple M5 Max"
        )
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


def glm5_register_pairwise_sum_requested() -> bool:
    """Select the qualified prefill reduction, preserving explicit overrides.

    Automatic use is single-request only (enforced by the KDA caller). An
    explicit 1 retains the experimental wider-shape opt-in, still subject to
    kernel safety checks. Explicit 0 and existing non-1 values retain stock.
    """
    value = os.environ.get("VMLX_GLM5_REGISTER_PAIRWISE_SUM")
    return _register_sum_default_qualified() if value is None else value == "1"


def glm5_prefill_layer_fence_enabled() -> bool:
    """Realize completed-layer state by default; retain an explicit off switch.

    GLM's mixed recurrent/latent caches retain lazy side outputs. Evaluating
    each layer's streams and native state bounds those graph lifetimes without
    changing token partitioning, projection shapes, arithmetic, or state ABI.
    The model applies this only to cached prefill longer than 64 tokens, never
    ordinary decode or short speculative verification.
    """
    return os.environ.get("VMLX_GLM5_PREFILL_LAYER_FENCE", "1") == "1"


# Startup-frozen selection: math, cache identity and health use one value.
# Default ON since 2026-10-08 (bit-identical to the stock solver on MLX 0.32.2
# and 0.32.3; served A/B GLM-5.3 JANGH2: 11/11 texts byte-identical in 6 arms,
# cold prefill +7-13 % at matched clock). The kernel itself still runs only on
# its qualified runtime (metal/glm5_kda_substitution._compatible_runtime);
# elsewhere stock math runs. Unset or "1" selects it; any other value is stock.
_KDA_SUBSTITUTION_VALUE = os.environ.get("VMLX_GLM5_KDA_SUBSTITUTION")
_KDA_SUBSTITUTION_REQUESTED = _KDA_SUBSTITUTION_VALUE is None or _KDA_SUBSTITUTION_VALUE == "1"
GLM5_KDA_SUBSTITUTION_MATH_ABI = "mlx0322_col32x32_i1stock_v1"


def glm5_kda_substitution_requested() -> bool:
    return _KDA_SUBSTITUTION_REQUESTED


@lru_cache(maxsize=1)
def glm5_kda_substitution_identity() -> str:
    import hashlib
    from pathlib import Path

    source = Path(__file__).with_name("metal") / "glm5_kda_substitution.py"
    return GLM5_KDA_SUBSTITUTION_MATH_ABI + ":" + hashlib.sha256(source.read_bytes()).hexdigest()
