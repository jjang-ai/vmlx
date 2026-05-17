from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


QWEN36_ENDOFTEXT_ID = 248044
QWEN36_IM_END_ID = 248046


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _token_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    try:
        return [int(value)]
    except Exception:
        return []


def _duplicate_ids(ids: Iterable[int]) -> List[int]:
    seen = set()
    duplicates = set()
    for token_id in ids:
        if token_id in seen:
            duplicates.add(token_id)
        seen.add(token_id)
    return sorted(duplicates)


def _added_token_id(tokenizer_config: Dict[str, Any], token: str) -> Optional[int]:
    added = tokenizer_config.get("added_tokens_decoder")
    if isinstance(added, dict):
        for key, value in added.items():
            if isinstance(value, dict) and value.get("content") == token:
                try:
                    return int(key)
                except Exception:
                    return None
    for value in tokenizer_config.get("added_tokens", []) or []:
        if isinstance(value, dict) and value.get("content") == token:
            try:
                return int(value.get("id"))
            except Exception:
                return None
    return None


def _nested(mapping: Dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def audit_artifact(model_dir: str | Path) -> Dict[str, Any]:
    path = Path(model_dir)
    config = _read_json(path / "config.json")
    generation = (
        _read_json(path / "generation_config.json")
        if (path / "generation_config.json").exists()
        else {}
    )
    tokenizer_config = (
        _read_json(path / "tokenizer_config.json")
        if (path / "tokenizer_config.json").exists()
        else {}
    )

    endoftext_id = (
        _added_token_id(tokenizer_config, "<|endoftext|>") or QWEN36_ENDOFTEXT_ID
    )
    im_end_id = _added_token_id(tokenizer_config, "<|im_end|>") or QWEN36_IM_END_ID
    gen_eos = _token_ids(generation.get("eos_token_id"))
    text_eos = _token_ids(_nested(config, "text_config", "eos_token_id"))
    top_eos = _token_ids(config.get("eos_token_id"))

    issues: List[str] = []
    if gen_eos:
        if endoftext_id not in gen_eos:
            issues.append(
                "generation_config.eos_token_id missing "
                f"<|endoftext|> id {endoftext_id}"
            )
        if im_end_id not in gen_eos:
            issues.append(
                f"generation_config.eos_token_id missing <|im_end|> id {im_end_id}"
            )
        duplicates = _duplicate_ids(gen_eos)
        if duplicates:
            issues.append(
                f"generation_config.eos_token_id contains duplicate ids: {duplicates}"
            )
    else:
        issues.append("generation_config.eos_token_id is empty or missing")

    if text_eos and text_eos != [endoftext_id]:
        issues.append(
            "text_config.eos_token_id should be "
            f"<|endoftext|> id {endoftext_id}, got {text_eos[0]}"
        )

    if top_eos and endoftext_id not in top_eos:
        issues.append(
            f"config.eos_token_id missing <|endoftext|> id {endoftext_id}"
        )

    trained_num_experts = _nested(config, "text_config", "num_experts_per_tok")
    if trained_num_experts is None:
        trained_num_experts = config.get("num_experts_per_tok")

    report = {
        "path": str(path),
        "ok": not issues,
        "issues": issues,
        "token_ids": {
            "endoftext": endoftext_id,
            "im_end": im_end_id,
        },
        "config": {
            "model_type": config.get("model_type"),
            "text_model_type": _nested(config, "text_config", "model_type"),
            "eos_token_id": config.get("eos_token_id"),
            "text_config_eos_token_id": _nested(
                config, "text_config", "eos_token_id"
            ),
        },
        "generation": {
            "eos_token_id": gen_eos,
            "top_k": generation.get("top_k"),
            "top_p": generation.get("top_p"),
            "temperature": generation.get("temperature"),
            "do_sample": generation.get("do_sample"),
        },
        "routing": {
            "trained_num_experts_per_tok": trained_num_experts,
            "generation_top_k_is_sampler_top_k": generation.get("top_k"),
        },
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit private Qwen3.6 MTP artifact stop/sampling metadata."
    )
    parser.add_argument("model_dir", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSONL reports.")
    args = parser.parse_args(argv)

    reports = [audit_artifact(path) for path in args.model_dir]
    if args.json:
        for report in reports:
            print(json.dumps(report, sort_keys=True))
    else:
        for report in reports:
            status = "OK" if report["ok"] else "FAIL"
            print(f"{status} {report['path']}")
            for issue in report["issues"]:
                print(f"  - {issue}")
            print(
                "  generation: "
                f"eos={report['generation']['eos_token_id']} "
                f"top_k={report['generation']['top_k']} "
                f"top_p={report['generation']['top_p']} "
                f"temp={report['generation']['temperature']}"
            )
            print(
                "  routing: "
                f"trained_num_experts_per_tok="
                f"{report['routing']['trained_num_experts_per_tok']}"
            )
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
