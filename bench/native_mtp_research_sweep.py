#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmlx_engine.native_mtp_research import (  # noqa: E402
    ResearchSamplerSettings,
    speculative_policy_metrics,
)


def _log(values: list[float]) -> np.ndarray:
    return np.log(np.asarray(values, dtype=np.float64))


SYNTHETIC_CASES: dict[str, dict[str, np.ndarray]] = {
    # Easy, low-entropy row: close to the current count-prompt acceptance shape.
    "low_entropy_close": {
        "target": _log([0.82, 0.08, 0.04, 0.03, 0.02, 0.01]),
        "draft": _log([0.78, 0.10, 0.05, 0.03, 0.025, 0.015]),
    },
    # Third-position drift row: useful for deciding when D3 is only marginal.
    "medium_entropy_drift": {
        "target": _log([0.45, 0.20, 0.14, 0.10, 0.07, 0.04]),
        "draft": _log([0.35, 0.26, 0.17, 0.11, 0.07, 0.04]),
    },
    # High-entropy row: rejects should be common, residual correction matters.
    "high_entropy_hard": {
        "target": _log([0.24, 0.20, 0.17, 0.15, 0.13, 0.11]),
        "draft": _log([0.18, 0.23, 0.19, 0.16, 0.14, 0.10]),
    },
    # Mismatched tail row: filters can improve acceptance but may collapse
    # diversity if the target distribution still carries tail mass.
    "tail_mismatch": {
        "target": _log([0.38, 0.18, 0.14, 0.11, 0.08, 0.06, 0.05]),
        "draft": _log([0.46, 0.22, 0.12, 0.08, 0.05, 0.04, 0.03]),
    },
}


def build_settings_grid() -> list[tuple[ResearchSamplerSettings, ResearchSamplerSettings]]:
    target_temps = [0.2, 0.6, 0.8]
    temp_deltas = [0.0, 0.1, 0.2]
    min_ps = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]
    top_ps = [0.90, 0.95, 1.0]
    top_ks = [0, 20, 40]
    pairs: list[tuple[ResearchSamplerSettings, ResearchSamplerSettings]] = []
    for target_temp in target_temps:
        for delta in temp_deltas:
            draft_temp = round(target_temp + delta, 3)
            for min_p in min_ps:
                for top_p in top_ps:
                    for top_k in top_ks:
                        target = ResearchSamplerSettings(
                            temperature=target_temp,
                            top_p=top_p,
                            min_p=min_p,
                            top_k=top_k,
                        )
                        draft = ResearchSamplerSettings(
                            temperature=draft_temp,
                            top_p=top_p,
                            min_p=min_p,
                            top_k=top_k,
                        )
                        pairs.append((target, draft))
    return pairs


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _score_row(row: dict[str, Any]) -> tuple[float, float, float]:
    acceptance = float(row["expected_acceptance_rate"])
    delta = row["delta"]
    max_abs = float(delta["max_abs"])
    l1 = float(delta["l1"])
    return (-acceptance, max_abs, l1)


def run_sweep(top_n: int) -> dict[str, Any]:
    settings_grid = build_settings_grid()
    case_results: dict[str, Any] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for case_name, case in SYNTHETIC_CASES.items():
        rows: list[dict[str, Any]] = []
        for target_settings, draft_settings in settings_grid:
            metrics = speculative_policy_metrics(
                case["target"],
                case["draft"],
                target_settings=target_settings,
                draft_settings=draft_settings,
            )
            settings_key = json.dumps(
                {
                    "target": metrics["target_settings"],
                    "draft": metrics["draft_settings"],
                },
                sort_keys=True,
            )
            metrics["case"] = case_name
            metrics["settings_key"] = settings_key
            rows.append(metrics)
            agg = aggregate.setdefault(
                settings_key,
                {
                    "target_settings": metrics["target_settings"],
                    "draft_settings": metrics["draft_settings"],
                    "cases": 0,
                    "mean_acceptance": 0.0,
                    "mean_max_abs_delta": 0.0,
                    "mean_l1_delta": 0.0,
                    "worst_acceptance": 1.0,
                    "worst_max_abs_delta": 0.0,
                },
            )
            agg["cases"] += 1
            agg["mean_acceptance"] += float(metrics["expected_acceptance_rate"])
            agg["mean_max_abs_delta"] += float(metrics["delta"]["max_abs"])
            agg["mean_l1_delta"] += float(metrics["delta"]["l1"])
            agg["worst_acceptance"] = min(
                float(agg["worst_acceptance"]),
                float(metrics["expected_acceptance_rate"]),
            )
            agg["worst_max_abs_delta"] = max(
                float(agg["worst_max_abs_delta"]),
                float(metrics["delta"]["max_abs"]),
            )
        rows.sort(key=_score_row)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        case_results[case_name] = {
            "top": rows[:top_n],
            "bottom": rows[-min(top_n, len(rows)) :],
        }

    aggregate_rows = []
    for row in aggregate.values():
        cases = max(1, int(row["cases"]))
        row["mean_acceptance"] = float(row["mean_acceptance"]) / cases
        row["mean_max_abs_delta"] = float(row["mean_max_abs_delta"]) / cases
        row["mean_l1_delta"] = float(row["mean_l1_delta"]) / cases
        aggregate_rows.append(row)
    aggregate_rows.sort(
        key=lambda row: (
            -float(row["mean_acceptance"]),
            -float(row["worst_acceptance"]),
            float(row["mean_max_abs_delta"]),
            float(row["mean_l1_delta"]),
        )
    )
    for rank, row in enumerate(aggregate_rows, start=1):
        row["rank"] = rank

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "Synthetic clean-room p/q sampler sweep for Qwen3.6 native-MTP "
            "research. This does not prove a real-model policy."
        ),
        "grid_size": len(settings_grid),
        "case_count": len(SYNTHETIC_CASES),
        "aggregate_top": aggregate_rows[:top_n],
        "cases": case_results,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Native MTP Research Sweep",
        "",
        f"Created: `{result['created_at']}`",
        "",
        "Synthetic p/q sweep only. This ranks sampler-policy shapes before "
        "real-model shadow testing; it is not product evidence.",
        "",
        "## Aggregate Top Policies",
        "",
        "| Rank | Target temp | Draft temp | min_p | top_p | top_k | Mean acceptance | Worst acceptance |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["aggregate_top"]:
        t = row["target_settings"]
        d = row["draft_settings"]
        lines.append(
            "| {rank} | {tt:.2f} | {dt:.2f} | {mp:.2f} | {tp:.2f} | {tk} | {ma:.4f} | {wa:.4f} |".format(
                rank=row["rank"],
                tt=t["temperature"],
                dt=d["temperature"],
                mp=t["min_p"],
                tp=t["top_p"],
                tk=t["top_k"],
                ma=row["mean_acceptance"],
                wa=row["worst_acceptance"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rows with higher acceptance are only useful if real-model stochastic "
            "shadow tests preserve target distribution.",
            "- Greedy D3 exactness remains the product control.",
            "- Real Qwen logits are required before stamping any sidecar or runtime default.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/internal/release-gates/qwen36_d3_mtp_research_synthetic",
    )
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_sweep(max(1, int(args.top_n)))
    json_path = out_dir / "synthetic_sampler_sweep.json"
    md_path = out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(_jsonable(result), indent=2))
    write_markdown(_jsonable(result), md_path)
    print(json.dumps({"json": str(json_path), "summary": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
