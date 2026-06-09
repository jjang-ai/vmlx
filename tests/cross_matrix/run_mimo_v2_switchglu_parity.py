#!/usr/bin/env python3
"""No-heavy MiMo V2 affine SwitchGLU fast-path parity proof.

This runner proves the selected-expert affine SwitchGLU decode fast path against
the stock mlx-lm SwitchGLU path on tiny quantized tensors. It does not load the
MiMo model bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmlx_engine.models import mllm


DEFAULT_OUT = Path(
    "build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json"
)


def _reset_switchglu(original_call) -> None:
    SwitchGLU.__call__ = original_call
    for attr in (
        "_vmlx_mimo_affine_decode_fast_path",
        "_vmlx_mimo_original_affine_decode_call",
        "_vmlx_mimo_affine_decode_fast_calls",
        "_vmlx_mimo_tq_decode_fast_calls",
        "_vmlx_mimo_affine_decode_fallback_reasons",
    ):
        if hasattr(SwitchGLU, attr):
            delattr(SwitchGLU, attr)


def build_proof() -> dict:
    switch = SwitchGLU(32, 32, 4, bias=False)
    nn.quantize(switch, group_size=32, bits=4, mode="affine")
    switch.eval()
    x = mx.arange(32, dtype=mx.float32).reshape(1, 32) / 32.0
    indices = mx.array([[0, 2]], dtype=mx.uint32)

    original_call = SwitchGLU.__call__
    expected = original_call(switch, x, indices)
    mx.eval(expected)
    installed = mllm._install_mimo_v2_affine_switchglu_decode_fast_path()
    try:
        actual = switch(x, indices)
        mx.eval(actual)
        fast_calls = int(getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fast_calls", 0))
        fallback_reasons = dict(
            getattr(SwitchGLU, "_vmlx_mimo_affine_decode_fallback_reasons", {})
        )
    finally:
        _reset_switchglu(original_call)

    diff = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    mx.eval(diff)
    max_abs_diff = float(mx.max(diff).item())
    mean_abs_diff = float(mx.mean(diff).item())
    got_shape = [int(item) for item in actual.shape]
    manual_shape = [int(item) for item in expected.shape]
    ok = bool(
        installed
        and got_shape == manual_shape
        and max_abs_diff <= 0.001
        and mean_abs_diff <= 0.0002
        and fast_calls >= 1
        and not fallback_reasons
    )
    return {
        "status": "pass" if ok else "open",
        "ok": ok,
        "generated_at": int(time.time()),
        "got_shape": got_shape,
        "manual_shape": manual_shape,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "fast_calls": fast_calls,
        "fallback_reasons": fallback_reasons,
        "installed_fast_path": bool(installed),
        "release_boundary": (
            "This proves only the affine selected-expert SwitchGLU decode "
            "fast-path parity on tiny tensors. It does not clear MiMo artifact "
            "literal exactness, cache-vs-no-cache live logits, media E2E, "
            "JANG_2L live rows, or installed-app UI proof."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = build_proof()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)
    print(f"status={result['status']}")
    print(f"max_abs_diff={result['max_abs_diff']}")
    print(f"mean_abs_diff={result['mean_abs_diff']}")
    print(f"fast_calls={result['fast_calls']}")
    if result["fallback_reasons"]:
        print(f"fallback_reasons={result['fallback_reasons']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
