#!/usr/bin/env python3
"""Aggregate cached native-MTP real-logit policy sweeps across prompt classes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Native-MTP Policy Suite Summary",
        "",
        "Scope: private/internal shadow-logit research. This is not a runtime enablement decision.",
        "",
        "## Coverage",
        "",
        f"- Prompt artifacts: `{summary['aggregate']['prompt_count']}`",
        f"- Present classes: `{', '.join(summary['aggregate']['coverage']['present_classes'])}`",
        f"- Missing classes: `{', '.join(summary['aggregate']['coverage']['missing_classes'])}`",
        f"- Has VL image row: `{summary['aggregate']['coverage']['has_vl_image']}`",
        f"- Has video row: `{summary['aggregate']['coverage']['has_video']}`",
        f"- Has cache-repeat row: `{summary['aggregate']['coverage']['has_cache_repeat']}`",
        f"- D3 shadow ready: `{summary['aggregate']['readiness']['d3_shadow_ready']}`",
        f"- Readiness reasons: `{', '.join(summary['aggregate']['readiness']['reasons'])}`",
        "",
        "## Model Metadata",
        "",
    ]
    model = summary.get("model_metadata") or {}
    routing = model.get("trained_routing_top_k") or {}
    generation = (model.get("generation_config") or {}).get("sampling") or {}
    stop_audit = model.get("stop_token_audit") or {}
    metadata_readiness = model.get("metadata_readiness") or {}
    lines.extend(
        [
            f"- Model path: `{model.get('model_path')}`",
            f"- Model type: `{model.get('model_type')}`",
            f"- Text model type: `{model.get('text_model_type')}`",
            f"- Trained routing active experts: `{routing.get('active_experts')}`",
            f"- Trained routing source: `{routing.get('source')}`",
            f"- Routed experts: `{routing.get('n_routed_experts')}`",
            f"- generation_config temperature: `{generation.get('temperature')}`",
            f"- generation_config top_p: `{generation.get('top_p')}`",
            f"- generation_config sampler top_k: `{generation.get('top_k')}`",
            f"- generation_config min_p: `{generation.get('min_p')}`",
            f"- generation_config do_sample: `{generation.get('do_sample')}`",
            f"- Stop-token config clean: `{stop_audit.get('stop_config_clean')}`",
            f"- generation EOS token IDs: `{stop_audit.get('generation_eos_token_ids')}`",
            f"- deduplicated generation EOS token IDs: `{stop_audit.get('deduplicated_generation_eos_token_ids')}`",
            f"- required EOS token IDs: `{stop_audit.get('required_token_ids')}`",
            f"- missing required EOS token IDs: `{stop_audit.get('missing_required_token_ids')}`",
            f"- duplicate EOS token IDs: `{stop_audit.get('duplicate_token_ids')}`",
            f"- metadata readiness: `{metadata_readiness.get('ready')}`",
            f"- metadata readiness reasons: `{', '.join(metadata_readiness.get('reasons') or [])}`",
            "",
            "## Cost Model",
            "",
            f"- verify_ms: `{summary['aggregate']['cost_model']['verify_ms']}`",
            f"- draft_ms_per_depth: `{summary['aggregate']['cost_model']['draft_ms_per_depth']}`",
            f"- fixed_ms: `{summary['aggregate']['cost_model']['fixed_ms']}`",
            "",
            "## Ranked Policies",
            "",
            "| Rank | Target temp | Draft temp | top_p | min_p | top_k | prompts | mean D3 tok/ms | worst D3 tok/ms | mean D3-D2 tok/ms | worst D3 acc |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["aggregate"]["top"]:
        target = row["target_settings"]
        draft = row["draft_settings"]
        lines.append(
            "| {rank} | {tt:.3g} | {dt:.3g} | {top_p:.3g} | {min_p:.3g} | {top_k} | {prompts} | {mean:.6f} | {worst:.6f} | {delta:.6f} | {acc} |".format(
                rank=row["rank"],
                tt=float(target["temperature"]),
                dt=float(draft["temperature"]),
                top_p=float(target["top_p"]),
                min_p=float(target["min_p"]),
                top_k=int(target["top_k"]),
                prompts=row["prompt_count"],
                mean=float(row["mean_tokens_per_ms_d3"]),
                worst=float(row["worst_tokens_per_ms_d3"]),
                delta=float(row["mean_d3_minus_d2_tokens_per_ms"]),
                acc=row["worst_d3_acceptance"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Rows without full `real_logit_policy_sweep.rows` are rejected; rerun probes with `--policy-sweep-save-all`.",
            "- Missing VL/video/cache classes mean this summary is not a production gate.",
            "- Bad stop-token metadata means stop/finish parity is not production-cleared.",
            "- Trained routing top-k is not sampler top_k; both are recorded separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Prompt artifact in class=path form. Requires --policy-sweep-save-all rows.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--verify-ms", type=float, required=True)
    parser.add_argument("--draft-ms-per-depth", type=float, required=True)
    parser.add_argument("--fixed-ms", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    from vmlx_engine.native_mtp_policy_suite import (
        aggregate_policy_suite,
        load_prompt_artifacts,
        read_model_policy_metadata,
    )

    prompt_results = load_prompt_artifacts(args.artifact)
    summary = {
        "model_metadata": (
            read_model_policy_metadata(args.model_path)
            if args.model_path is not None
            else None
        ),
        "artifacts": args.artifact,
        "aggregate": aggregate_policy_suite(
            prompt_results,
            verify_ms=args.verify_ms,
            draft_ms_per_depth=args.draft_ms_per_depth,
            fixed_ms=args.fixed_ms,
            top_n=args.top_n,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "summary.json"
    markdown_path = args.out_dir / "SUMMARY.md"
    json_path.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary, markdown_path)
    print(json.dumps({"summary": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
