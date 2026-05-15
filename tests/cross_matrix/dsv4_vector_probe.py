"""DSV4 ds4-style official-vector diagnostics.

This module is intentionally split into a cheap parser and an optional runtime
probe. Unit tests cover the parser. The runtime path is opt-in because it
hydrates the full DSV4 bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VectorTop:
    token_hex: str
    logprob: float


@dataclass(frozen=True)
class VectorStep:
    index: int
    selected_hex: str
    tops: tuple[VectorTop, ...]


@dataclass(frozen=True)
class VectorCase:
    case_id: str
    ctx: int
    steps: tuple[VectorStep, ...]
    prompt_path: Path


def bytes_to_hex(data: bytes) -> str:
    return data.hex()


def hex_to_bytes(data: str) -> bytes:
    return bytes.fromhex(data)


def token_bytes(tokenizer: Any, token_id: int) -> bytes:
    """Return UTF-8 bytes for one decoded token.

    ds4's vectors compare token *bytes*, not token IDs, so this stays tokenizer
    implementation agnostic. DSV4's official selected tokens are normal text
    bytes in the checked-in vectors.
    """
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode([int(token_id)])
    else:
        inner = getattr(tokenizer, "_tokenizer", tokenizer)
        text = inner.decode([int(token_id)])
    if isinstance(text, bytes):
        return text
    return str(text).encode("utf-8")


def token_id_for_bytes(tokenizer: Any, data: bytes) -> int | None:
    """Resolve one official token byte string to a local tokenizer ID.

    Returns ``None`` if the bytes do not UTF-8 decode, tokenize into multiple
    IDs, or fail a decode round trip. The probe uses this for teacher-forced
    diagnostic runs only; a failed resolve is reported instead of guessed.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    encoder = tokenizer if hasattr(tokenizer, "encode") else getattr(tokenizer, "_tokenizer", tokenizer)
    if not hasattr(encoder, "encode"):
        return None
    try:
        ids = encoder.encode(text, add_special_tokens=False)
    except TypeError:
        ids = encoder.encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(t) for t in ids]
    if len(ids) != 1:
        return None
    tid = ids[0]
    return tid if token_bytes(tokenizer, tid) == data else None


def _resolve_prompt_path(raw: str, *, vec_path: Path, base_dir: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / p)
    candidates.append(vec_path.parent / p)
    candidates.append(Path.cwd() / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Keep deterministic path in errors/artifacts even if the prompt is absent.
    return candidates[0].resolve()


def parse_official_vec(path: Path, *, base_dir: Path | None = None) -> list[VectorCase]:
    """Parse ds4 ``official.vec`` with strict row-count validation."""
    vec_path = Path(path)
    cases: list[VectorCase] = []
    current_id: str | None = None
    current_ctx: int | None = None
    current_prompt: Path | None = None
    current_steps: list[VectorStep] = []
    pending_index: int | None = None
    pending_selected: str | None = None
    pending_expected_tops = 0
    pending_tops: list[VectorTop] = []

    def flush_step() -> None:
        nonlocal pending_index, pending_selected, pending_expected_tops, pending_tops
        if pending_index is None:
            return
        if len(pending_tops) != pending_expected_tops:
            raise ValueError(
                f"{vec_path}: step {pending_index} expected "
                f"{pending_expected_tops} top rows, got {len(pending_tops)}"
            )
        current_steps.append(
            VectorStep(
                index=pending_index,
                selected_hex=pending_selected or "",
                tops=tuple(pending_tops),
            )
        )
        pending_index = None
        pending_selected = None
        pending_expected_tops = 0
        pending_tops = []

    def flush_case() -> None:
        nonlocal current_id, current_ctx, current_prompt, current_steps
        flush_step()
        if current_id is None:
            return
        if current_ctx is None or current_prompt is None:
            raise ValueError(f"{vec_path}: incomplete case {current_id}")
        cases.append(
            VectorCase(
                case_id=current_id,
                ctx=current_ctx,
                steps=tuple(current_steps),
                prompt_path=current_prompt,
            )
        )
        current_id = None
        current_ctx = None
        current_prompt = None
        current_steps = []

    for line_no, raw_line in enumerate(vec_path.read_text(encoding="ascii").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "case":
            flush_case()
            if len(parts) != 5:
                raise ValueError(f"{vec_path}:{line_no}: malformed case line")
            current_id = parts[1]
            current_ctx = int(parts[2])
            expected_steps = int(parts[3])
            current_prompt = _resolve_prompt_path(
                parts[4],
                vec_path=vec_path,
                base_dir=base_dir,
            )
            current_steps = []
            if expected_steps < 0:
                raise ValueError(f"{vec_path}:{line_no}: negative step count")
            continue
        if tag == "step":
            if current_id is None:
                raise ValueError(f"{vec_path}:{line_no}: step before case")
            flush_step()
            if len(parts) != 4:
                raise ValueError(f"{vec_path}:{line_no}: malformed step line")
            pending_index = int(parts[1])
            pending_selected = parts[2].lower()
            pending_expected_tops = int(parts[3])
            pending_tops = []
            continue
        if tag == "top":
            if pending_index is None:
                raise ValueError(f"{vec_path}:{line_no}: top before step")
            if len(parts) != 3:
                raise ValueError(f"{vec_path}:{line_no}: malformed top line")
            pending_tops.append(VectorTop(parts[1].lower(), float(parts[2])))
            continue
        if tag == "end":
            flush_case()
            continue
        raise ValueError(f"{vec_path}:{line_no}: unknown row {tag!r}")

    if current_id is not None or pending_index is not None:
        raise ValueError(f"{vec_path}: unterminated vector case")
    return cases


def _case_to_json(case: VectorCase) -> dict[str, Any]:
    doc = asdict(case)
    doc["prompt_path"] = str(case.prompt_path)
    return doc


def _disable_attention_sinks(model: Any) -> int:
    """Disable MLX generic attention-sink contribution for vector A/B probes."""
    import mlx.core as mx

    count = 0
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return 0
    for _, module in named_modules():
        sink = getattr(module, "attn_sink", None)
        if sink is None or not hasattr(sink, "shape"):
            continue
        module.attn_sink = mx.full(sink.shape, -1e9, dtype=getattr(sink, "dtype", mx.float32))
        count += 1
    return count


def _disable_attention_sinks_requested() -> bool:
    """Return whether the vector harness should run the no-sinks ablation."""
    if os.environ.get("VMLX_DSV4_VECTOR_DISABLE_SINKS", "0") == "1":
        return True
    # Backward-compatible for artifacts produced while the harness used the
    # misspelled prefix. Keep this local to the private diagnostic harness.
    return os.environ.get("VMLINUX_DSV4_VECTOR_DISABLE_SINKS", "0") == "1"


def _cache_disabled_requested() -> bool:
    """Return whether the vector harness should bypass model.make_cache()."""
    if os.environ.get("VMLX_DSV4_VECTOR_NO_CACHE", "0") == "1":
        return True
    return os.environ.get("VMLINUX_DSV4_VECTOR_NO_CACHE", "0") == "1"


def _prepare_vector_probe_environment() -> dict[str, Any]:
    """Prepare and report parity-critical environment state for vector probes."""
    return {
        "runtime_path": "direct_model_call",
        "dsv4_batch_generator_used": False,
        "logit_warps": {
            "hard_repetition_block": "removed",
        },
    }


def _top_logprobs(logprobs: Any, k: int) -> list[tuple[int, float]]:
    import numpy as np

    arr = np.asarray(logprobs)
    if arr.ndim == 2:
        arr = arr[0]
    if k <= 0:
        return []
    k = min(k, arr.shape[-1])
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return [(int(i), float(arr[i])) for i in idx]


def run_probe(
    *,
    model_path: Path,
    vec_path: Path,
    base_dir: Path | None = None,
    case_filter: Iterable[str] | None = None,
    top_k: int = 20,
    continue_with: str = "local",
) -> dict[str, Any]:
    """Hydrate DSV4 and compare local greedy next-token bytes to ds4 vectors."""
    import mlx.core as mx

    from vmlx_engine.loaders.load_jangtq_dsv4 import load_jangtq_dsv4_model

    diagnostics = _prepare_vector_probe_environment()
    model, tokenizer = load_jangtq_dsv4_model(str(model_path))
    ablations: dict[str, Any] = {}
    if _disable_attention_sinks_requested():
        ablations["attention_sinks_disabled"] = _disable_attention_sinks(model)
    wanted = set(case_filter or [])
    cases = [
        c for c in parse_official_vec(vec_path, base_dir=base_dir)
        if not wanted or c.case_id in wanted
    ]
    out_cases: list[dict[str, Any]] = []

    if continue_with not in {"local", "official"}:
        raise ValueError("continue_with must be 'local' or 'official'")

    for case in cases:
        prompt_text = case.prompt_path.read_text(encoding="utf-8")
        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            reasoning_effort=None,
            tokenize=True,
            add_default_bos_token=True,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        token_ids = [int(t) for t in token_ids]
        if _cache_disabled_requested():
            cache = None
            ablations["cache_disabled"] = True
        else:
            cache = model.make_cache() if hasattr(model, "make_cache") else None
        ids = mx.array(token_ids, dtype=mx.int32)[None, :]
        logits = model(ids, cache=cache)
        last_logits = logits[:, -1, :]
        logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        if hasattr(mx, "synchronize"):
            mx.synchronize()

        case_rows: list[dict[str, Any]] = []
        for step in case.steps:
            local_id = int(mx.argmax(logprobs, axis=-1).tolist()[0])
            local_hex = bytes_to_hex(token_bytes(tokenizer, local_id))
            top_rows = [
                {
                    "token_id": tid,
                    "token_hex": bytes_to_hex(token_bytes(tokenizer, tid)),
                    "logprob": lp,
                }
                for tid, lp in _top_logprobs(logprobs, top_k)
            ]
            official_hits = []
            for top in step.tops:
                match = next((row for row in top_rows if row["token_hex"] == top.token_hex), None)
                official_hits.append(
                    {
                        "token_hex": top.token_hex,
                        "official_logprob": top.logprob,
                        "found": match is not None,
                        "local_logprob": match["logprob"] if match else None,
                        "delta": abs(match["logprob"] - top.logprob) if match else None,
                    }
                )
            selected_match = local_hex == step.selected_hex
            case_rows.append(
                {
                    "step": step.index,
                    "expected_selected_hex": step.selected_hex,
                    "local_selected_hex": local_hex,
                    "local_selected_token_id": local_id,
                    "selected_match": selected_match,
                    "official_top_hits": official_hits,
                    "local_top": top_rows,
                }
            )
            if continue_with == "official":
                next_id = token_id_for_bytes(tokenizer, hex_to_bytes(step.selected_hex))
                if next_id is None:
                    case_rows[-1]["official_next_token_resolved"] = False
                    break
                case_rows[-1]["official_next_token_resolved"] = True
            else:
                # Continue with the local token, matching ds4's test loop once
                # the selected-token check passes. If it fails, later rows are
                # a local divergence trace.
                next_id = local_id
            ids = mx.array([[next_id]], dtype=mx.int32)
            logits = model(ids, cache=cache)
            last_logits = logits[:, -1, :]
            logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
            mx.eval(logprobs)
            if hasattr(mx, "synchronize"):
                mx.synchronize()

        out_cases.append(
            {
                "case_id": case.case_id,
                "ctx": case.ctx,
                "prompt_path": str(case.prompt_path),
                "prompt_tokens": len(token_ids),
                "steps": case_rows,
                "all_selected_match": all(r["selected_match"] for r in case_rows),
            }
        )
    return {
        "schema": "vmlx-dsv4-ds4-vector-probe-v1",
        "model_path": str(model_path),
        "vec_path": str(vec_path),
        "continue_with": continue_with,
        "diagnostics": diagnostics,
        "ablations": ablations,
        "cases": out_cases,
        "all_selected_match": all(c["all_selected_match"] for c in out_cases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vec", type=Path, default=Path("/tmp/ds4-read/tests/test-vectors/official.vec"))
    parser.add_argument("--base-dir", type=Path, default=Path("/tmp/ds4-read"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--continue-with", choices=("local", "official"), default="local")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = run_probe(
        model_path=args.model,
        vec_path=args.vec,
        base_dir=args.base_dir,
        case_filter=args.case,
        top_k=args.top_k,
        continue_with=args.continue_with,
    )
    data = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    return 0 if result.get("all_selected_match") else 2


if __name__ == "__main__":
    raise SystemExit(main())
