"""DSV4 real-activation routed-expert diagnostic.

This is a private release-gate probe, not a runtime patch. It captures the
actual MoE routed-expert inputs/indices/outputs for a ds4 official-vector prompt
and compares the selected experts against source FP4-dequantized weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.cross_matrix.dsv4_vector_probe import (
    _prepare_vector_probe_environment,
    parse_official_vec,
)


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected = np.asarray(expected, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(expected))
    if denom == 0.0:
        return float(np.linalg.norm(actual))
    return float(np.linalg.norm(actual - expected) / denom)


def _cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected = np.asarray(expected, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    if denom == 0.0:
        return 0.0
    return float(np.dot(actual, expected) / denom)


def _token_rows(
    *,
    x: np.ndarray,
    indices: np.ndarray,
    runtime_y: np.ndarray,
    scores: np.ndarray | None,
    token_indices: Iterable[int] | None,
) -> list[dict[str, Any]]:
    """Extract one row per selected expert for requested prompt tokens.

    In the production DSV4 model, MoE input is normally non-laned:

    - x: `(B, T, D)`
    - indices: `(B, T, K)`
    - runtime_y: `(B, T, K, D)`

    The helper also accepts the laned form for synthetic tests and future
    variants:

    - x: `(B, T, H, D)`
    - indices: `(B, T, H, K)`
    - runtime_y: `(B, T, H, K, D)`

    `scores`, when present, follows the `indices` shape.
    """
    x = np.asarray(x)
    indices = np.asarray(indices)
    runtime_y = np.asarray(runtime_y)
    scores_arr = np.asarray(scores) if scores is not None else None
    if x.ndim == 3:
        x = x[:, :, None, :]
    if indices.ndim == 3:
        indices = indices[:, :, None, :]
    if runtime_y.ndim == 4:
        runtime_y = runtime_y[:, :, None, :, :]
    if scores_arr is not None and scores_arr.ndim == 3:
        scores_arr = scores_arr[:, :, None, :]
    if x.ndim != 4 or indices.ndim != 4 or runtime_y.ndim != 5:
        raise ValueError(
            "expected x=(B,T,H,D), indices=(B,T,H,K), runtime_y=(B,T,H,K,D); "
            f"got {x.shape}, {indices.shape}, {runtime_y.shape}"
        )
    if x.shape[:3] != indices.shape[:3] or indices.shape != runtime_y.shape[:4]:
        raise ValueError(
            "shape mismatch for x/indices/runtime_y: "
            f"{x.shape}, {indices.shape}, {runtime_y.shape}"
        )
    if scores_arr is not None and scores_arr.shape != indices.shape:
        raise ValueError(f"score shape mismatch: scores={scores_arr.shape}, indices={indices.shape}")

    rows: list[dict[str, Any]] = []
    tokens = list(range(x.shape[1])) if token_indices is None else [int(t) for t in token_indices]
    for b in range(x.shape[0]):
        for t in tokens:
            for h in range(x.shape[2]):
                for k in range(indices.shape[-1]):
                    rows.append(
                        {
                            "batch": b,
                            "token_index": t,
                            "hc_lane": h,
                            "expert_slot": k,
                            "expert": int(indices[b, t, h, k]),
                            "score": (
                                float(scores_arr[b, t, h, k])
                                if scores_arr is not None
                                else None
                            ),
                            "x": np.array(x[b, t, h], dtype=np.float32, copy=True),
                            "runtime_y": np.array(runtime_y[b, t, h, k], dtype=np.float32, copy=True),
                        }
                    )
    return rows


def _last_token_rows(
    *,
    x: np.ndarray,
    indices: np.ndarray,
    runtime_y: np.ndarray,
    scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    x_arr = np.asarray(x)
    return _token_rows(
        x=x,
        indices=indices,
        runtime_y=runtime_y,
        scores=scores,
        token_indices=[x_arr.shape[1] - 1],
    )


def _score_weighted_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("score") is None or "source_y" not in row:
            continue
        key = (int(row["batch"]), int(row["token_index"]), int(row["hc_lane"]))
        groups.setdefault(key, []).append(row)

    rel_values: list[float] = []
    cos_values: list[float] = []
    for group in groups.values():
        runtime_sum = None
        source_sum = None
        score_sum = 0.0
        for row in group:
            score = float(row["score"])
            score_sum += score
            runtime_part = score * np.asarray(row["runtime_y"], dtype=np.float32)
            source_part = score * np.asarray(row["source_y"], dtype=np.float32)
            runtime_sum = runtime_part if runtime_sum is None else runtime_sum + runtime_part
            source_sum = source_part if source_sum is None else source_sum + source_part
        if runtime_sum is None or source_sum is None:
            continue
        rel_values.append(_relative_l2(runtime_sum, source_sum))
        cos_values.append(_cosine(runtime_sum, source_sum))

    rel = np.array(rel_values, dtype=np.float32)
    cos = np.array(cos_values, dtype=np.float32)
    return {
        "group_count": int(rel.size),
        "rel_l2_mean": float(rel.mean()) if rel.size else None,
        "rel_l2_max": float(rel.max()) if rel.size else None,
        "rel_l2_min": float(rel.min()) if rel.size else None,
        "cosine_mean": float(cos.mean()) if cos.size else None,
        "cosine_min": float(cos.min()) if cos.size else None,
    }


def _read_routed_bits(model_path: Path, layer: int) -> int | None:
    from safetensors import safe_open

    from vmlx_engine.loaders.load_jangtq_dsv4 import _dsv4_weight_map

    key = f"layers.{layer}.mlp.switch_mlp.gate_proj.tq_bits"
    weight_map = _dsv4_weight_map(model_path)
    shard = weight_map.get(key)
    if not shard:
        return None
    with safe_open(str(model_path / shard), framework="np") as f:
        bits = np.asarray(f.get_tensor(key)).reshape(-1)
    uniq = sorted({int(v) for v in bits.tolist()})
    return uniq[0] if len(uniq) == 1 else None


class _SourceExpertReader:
    def __init__(self, source_path: Path, layer: int, *, swiglu_limit: float):
        import torch
        from jang_tools.dsv4.weight_loader import ShardIndex

        self.torch = torch
        self.idx = ShardIndex(source_path)
        self.layer = layer
        self.swiglu_limit = float(swiglu_limit)

    def _weight(self, expert: int, proj: str):
        src_proj = {"gate": "w1", "down": "w2", "up": "w3"}[proj]
        tensor_key = f"layers.{self.layer}.ffn.experts.{int(expert)}.{src_proj}.weight"
        return self.idx.read_tensor(
            tensor_key,
            out_dtype=self.torch.float32,
        )

    def output_batch(self, expert: int, xs: np.ndarray) -> np.ndarray:
        torch = self.torch
        x_t = torch.from_numpy(np.asarray(xs, dtype=np.float32))
        w1 = self._weight(expert, "gate")
        w2 = self._weight(expert, "down")
        w3 = self._weight(expert, "up")
        with torch.inference_mode():
            gate = x_t @ w1.T
            up = x_t @ w3.T
            if self.swiglu_limit > 0:
                gate = torch.clamp(gate, max=self.swiglu_limit)
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            out = hidden @ w2.T
        return out.detach().cpu().float().numpy()


def _attach_source_outputs(
    *,
    rows: list[dict[str, Any]],
    source_path: Path,
    layer: int,
    swiglu_limit: float,
) -> list[dict[str, Any]]:
    reader = _SourceExpertReader(source_path, layer, swiglu_limit=swiglu_limit)
    by_expert: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_expert.setdefault(int(row["expert"]), []).append(i)
    for expert, positions in sorted(by_expert.items()):
        xs = np.stack([rows[i]["x"] for i in positions]).astype(np.float32, copy=False)
        outputs = reader.output_batch(expert, xs)
        for offset, row_index in enumerate(positions):
            rows[row_index]["source_y"] = outputs[offset]
    return rows


@contextmanager
def _capture_moe_layers(layers: Iterable[int]):
    import mlx.core as mx
    from jang_tools.dsv4.mlx_model import MoE

    wanted = {int(layer) for layer in layers}
    captures: dict[int, dict[str, np.ndarray]] = {}
    original_call = MoE.__call__

    def patched_call(self, x, input_ids=None):
        if int(getattr(self, "layer_id", -1)) not in wanted:
            return original_call(self, x, input_ids=input_ids)
        inds, scores = self.gate(x, input_ids=input_ids)
        inds = inds.astype(mx.uint32)
        routed_y = self.switch_mlp(x, inds)
        mx.eval(x, inds, routed_y, scores)
        captures[int(self.layer_id)] = {
            "x": np.array(x, dtype=np.float32, copy=True),
            "indices": np.array(inds, dtype=np.uint32, copy=True),
            "runtime_y": np.array(routed_y, dtype=np.float32, copy=True),
            "scores": np.array(scores, dtype=np.float32, copy=True),
        }
        y = (routed_y * scores[..., None]).sum(axis=-2).astype(routed_y.dtype).reshape(x.shape)
        y = y + self.shared_experts(x)
        return y

    MoE.__call__ = patched_call
    try:
        yield captures
    finally:
        MoE.__call__ = original_call


def _case_prompt(vec_path: Path, base_dir: Path, case_id: str) -> str:
    cases = parse_official_vec(vec_path, base_dir=base_dir)
    for case in cases:
        if case.case_id == case_id:
            return case.prompt_path.read_text(encoding="utf-8")
    raise ValueError(f"case {case_id!r} not found in {vec_path}")


def run_probe(
    *,
    model_path: Path,
    source_path: Path,
    vec_path: Path,
    base_dir: Path,
    case_id: str,
    layers: Iterable[int],
    token_scope: str = "final",
) -> dict[str, Any]:
    import mlx.core as mx

    from vmlx_engine.loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model

    os.environ.setdefault("DSV4_POOL_QUANT", "1")
    diagnostics = _prepare_vector_probe_environment()
    diagnostics["dsv4_pool_quant"] = os.environ.get("DSV4_POOL_QUANT")
    if token_scope not in {"final", "all"}:
        raise ValueError("token_scope must be 'final' or 'all'")
    model, tokenizer = load_jangtq_dsv4_model(str(model_path))
    prompt_text = _case_prompt(vec_path, base_dir, case_id)
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        enable_thinking=False,
        reasoning_effort=None,
        tokenize=True,
        add_default_bos_token=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    token_ids = [int(t) for t in token_ids]

    cache = model.make_cache() if hasattr(model, "make_cache") else None
    ids = mx.array(token_ids, dtype=mx.int32)[None, :]
    with _capture_moe_layers(layers) as captures:
        logits = model(ids, cache=cache)
        mx.eval(logits)
        if hasattr(mx, "synchronize"):
            mx.synchronize()

    layer_results: list[dict[str, Any]] = []
    for layer in [int(l) for l in layers]:
        capture = captures.get(layer)
        if capture is None:
            layer_results.append({"layer": layer, "captured": False})
            continue
        token_indices = None if token_scope == "all" else [capture["x"].shape[1] - 1]
        rows = _token_rows(
            x=capture["x"],
            indices=capture["indices"],
            runtime_y=capture["runtime_y"],
            scores=capture["scores"],
            token_indices=token_indices,
        )
        activation = getattr(
            getattr(model.model.layers[layer].mlp.switch_mlp, "activation", None),
            "swiglu_limit",
            10.0,
        )
        swiglu_limit = float(activation)
        rows = _attach_source_outputs(
            rows=rows,
            source_path=source_path,
            layer=layer,
            swiglu_limit=swiglu_limit,
        )
        row_results: list[dict[str, Any]] = []
        for row in rows:
            src_y = row["source_y"]
            runtime_y = row["runtime_y"]
            row_results.append(
                {
                    "batch": row["batch"],
                    "token_index": row["token_index"],
                    "hc_lane": row["hc_lane"],
                    "expert_slot": row["expert_slot"],
                    "expert": row["expert"],
                    "score": row["score"],
                    "rel_l2": _relative_l2(runtime_y, src_y),
                    "cosine": _cosine(runtime_y, src_y),
                    "max_abs": float(np.max(np.abs(runtime_y - src_y))),
                    "mean_abs": float(np.mean(np.abs(runtime_y - src_y))),
                    "runtime_norm": float(np.linalg.norm(runtime_y.reshape(-1))),
                    "source_norm": float(np.linalg.norm(src_y.reshape(-1))),
                }
            )
            row["rel_l2"] = row_results[-1]["rel_l2"]
            row["cosine"] = row_results[-1]["cosine"]
        rel = np.array([r["rel_l2"] for r in row_results], dtype=np.float32)
        cos = np.array([r["cosine"] for r in row_results], dtype=np.float32)
        weighted_rows = []
        for row in rows:
            weighted_rows.append(
                {
                    "batch": row["batch"],
                    "token_index": row["token_index"],
                    "hc_lane": row["hc_lane"],
                    "score": row["score"],
                    "runtime_y": row["runtime_y"],
                    "source_y": row["source_y"],
                }
            )
        layer_results.append(
            {
                "layer": layer,
                "captured": True,
                "routed_bits": _read_routed_bits(model_path, layer),
                "swiglu_limit": swiglu_limit,
                "token_scope": token_scope,
                "capture_shapes": {k: list(v.shape) for k, v in capture.items()},
                "unique_experts": sorted({r["expert"] for r in row_results}),
                "rows": row_results,
                "summary": {
                    "row_count": len(row_results),
                    "rel_l2_mean": float(rel.mean()) if rel.size else None,
                    "rel_l2_max": float(rel.max()) if rel.size else None,
                    "rel_l2_min": float(rel.min()) if rel.size else None,
                    "cosine_mean": float(cos.mean()) if cos.size else None,
                    "cosine_min": float(cos.min()) if cos.size else None,
                    "score_weighted": _score_weighted_metrics(weighted_rows),
                },
            }
        )

    return {
        "schema": "vmlx-dsv4-real-activation-expert-probe-v1",
        "model_path": str(model_path),
        "source_path": str(source_path),
        "vec_path": str(vec_path),
        "case_id": case_id,
        "prompt_tokens": len(token_ids),
        "token_scope": token_scope,
        "diagnostics": diagnostics,
        "layers": layer_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vec", type=Path, default=Path("/tmp/ds4-read/tests/test-vectors/official.vec"))
    parser.add_argument("--base-dir", type=Path, default=Path("/tmp/ds4-read"))
    parser.add_argument("--case", default="short_code_completion")
    parser.add_argument("--layer", type=int, action="append", default=None)
    parser.add_argument("--token-scope", choices=("final", "all"), default="final")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_probe(
        model_path=args.model,
        source_path=args.source,
        vec_path=args.vec,
        base_dir=args.base_dir,
        case_id=args.case,
        layers=args.layer or [39, 42],
        token_scope=args.token_scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
