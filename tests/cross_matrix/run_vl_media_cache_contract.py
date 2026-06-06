#!/usr/bin/env python3
"""Run no-heavy VLM media/cache/tool-followup contracts.

This gate protects the VLM side of the current release hardening: image/video
request serialization, media-salted cache keys, hybrid SSM cache follow-up, and
tool-result media replay must stay covered without implying live Omni/video
quality clearance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("build/current-vl-media-cache-contract-after-gemma4-prefill-guard-proof-20260606.json")

PYTEST_PATTERN = (
    "video_url or video_fallback or media_salt or mediaSalt or tool_replay "
    "or mllm_chat_template_receives_effective_tool_schemas "
    "or forwards_effective_tools or hybrid_ssm or qwen36 or zaya "
    "or content_part or multi_turn_alternating_image_text "
    "or mllm_side_captures_at_prefill or deferred_rederive "
    "or turboquant_kv_cache_importable or mllm_scheduler_wires_paged "
    "or image_and_video_mixed or mllm_batch_generator_wires_resume_default_on "
    "or video_token_marker_distinct or multimodal_and_tools "
    "or tool_choice_auto_coexists or anthropic_tools_on_vl_messages "
    "or qwen36_affine_jang"
)

SOURCE_HASH_FILES = (
    "vmlx_engine/models/mllm.py",
    "vmlx_engine/mllm_batch_generator.py",
    "vmlx_engine/mllm_scheduler.py",
    "vmlx_engine/mllm_cache.py",
    "vmlx_engine/engine/batched.py",
    "vmlx_engine/api/models.py",
    "panel/src/main/ipc/chat.ts",
    "panel/src/main/tools/executor.ts",
    "panel/src/main/tools/registry.ts",
    "panel/src/main/sessions.ts",
    "panel/tests/tool-media-followup.test.ts",
    "panel/tests/image-display-consistency.test.ts",
    "panel/tests/model-config-registry.test.ts",
    "panel/tests/settings-flow.test.ts",
    "tests/test_vl_video_regression.py",
    "tests/test_mllm_scheduler_cache.py",
    "tests/test_mllm_tool_replay.py",
    "tests/test_model_config_registry.py",
    "tests/test_vl_media_cache_contract.py",
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_vl_media_cache_contracts": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_vl_video_regression.py",
            "tests/test_mllm_scheduler_cache.py",
            "tests/test_mllm_tool_replay.py",
            "tests/test_model_config_registry.py",
            "-k",
            PYTEST_PATTERN,
        ],
    ),
    "panel_vl_media_followup_contracts": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/tool-media-followup.test.ts",
            "tests/image-display-consistency.test.ts",
            "--reporter=verbose",
        ],
    ),
    "panel_vlm_settings_contracts": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/settings-flow.test.ts",
            "--reporter=verbose",
            "--testNamePattern",
            (
                "VLM|multimodal/VLM detection suppresses --enable-jit|"
                "continuous batching off is a real master switch for VLM cache flags"
            ),
        ],
    ),
    "panel_vlm_family_detection": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/model-config-registry.test.ts",
            "--reporter=verbose",
            "--testNamePattern",
            "ZAYA1-VL|Qwen.*VLM|Qwen.*video|indexed MTP|mxfp4 Qwen|mxfp8 Qwen|Nemotron-H|Step3\\.7",
        ],
    ),
}

REQUIRED_VL_MEDIA_CACHE_TEST_MARKERS = (
    "test_video_url_parsed",
    "test_image_and_video_mixed",
    "test_image_and_video_url_content_part_schemas",
    "test_video_url_content_part_extracted",
    "test_jangtq_video_fallback_no_crash",
    "test_video_fallback_installs_on_jang_vlm_load",
    "test_openai_content_part_allows_video",
    "test_qwen36_jangtq_is_detected_as_vl",
    "test_jang_config_qwen36_can_be_video",
    "test_qwen36_affine_jang_vlm_stays_text_loader_until_mrope_fixed",
    "test_hybrid_ssm_auto_mode_disables_live_tq_kv",
    "test_deferred_rederive_covers_post_output_hybrid_ssm_paths",
    "test_mllm_side_captures_at_prefill_not_finalize",
    "test_mllm_batch_generator_wires_resume_default_on",
    "test_mllm_scheduler_wires_paged_plus_ssm_companion",
    "test_turboquant_kv_cache_importable",
    "test_multi_turn_alternating_image_text_distinct_keys",
    "test_video_token_marker_distinct_from_image",
    "test_multimodal_and_tools_both_wire_through",
    "test_tool_choice_auto_coexists_with_enable_thinking_false",
    "test_anthropic_tools_on_vl_messages",
    "test_mllm_tool_replay_normalizes_tool_call_arguments_for_chat_template",
    "test_mllm_chat_template_receives_effective_tool_schemas",
    "test_simple_engine_forwards_effective_tools_to_mllm_chat",
)

REQUIRED_PANEL_VL_MEDIA_TEST_MARKERS = (
    "exposes read_video as a file tool next to read_image",
    "injects image and video tool results as multimodal follow-up content parts",
    "builds real multimodal follow-up content in text-image-video order",
    "does not create an empty media follow-up when no tool returned media bytes",
    "keeps media bytes out of plain tool text and returns data URLs separately",
    "shows directory basename for Mark's external FLUX.2-klein-9B (NOT stripped FLUX2)",
    "buildResult priority chain puts modelPath basename before cfg.servedModelName in the ASSIGNMENT (not just comments)",
    "VLM gets --continuous-batching for BatchedEngine with MLLMScheduler",
    "VLM continuous batching off emits explicit opt-out and suppresses cache stack",
    "auto-detected VLM wins over stale isMultimodal=false",
    "marks Qwen3.6 VL JANG bundles with indexed MTP tensors as native MTP capable",
    "keeps ZAYA1-VL multimodal when a stale stamp says text",
    "keeps MXTQ/JANGTQ Qwen hybrid VLM multimodal",
    "keeps mxfp4 Qwen hybrid VLM multimodal",
    "keeps mxfp8 Qwen hybrid VLM multimodal",
    "marks non-JANG Qwen 3.6 MoE bundles with vision/video metadata as multimodal",
    "does not route Nemotron-H text extracts through MLLM from stale sidecars",
    "routes Step3.7 JANG bridge through the source VLM runtime when available",
    "multimodal/VLM detection suppresses --enable-jit because mlx-vlm streaming is not compile-safe",
    "continuous batching off is a real master switch for VLM cache flags",
    "VLM with all caching features works together",
    "VLM suppresses external speculative decoding at launch",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_counts(output: str) -> dict[str, int | None]:
    passed = None
    skipped = None
    deselected = None
    match = re.search(r"Tests\s+(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    if match and passed is None:
        passed = int(match.group(1))
    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))
    match = re.search(r"(\d+) deselected", output)
    if match:
        deselected = int(match.group(1))
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def _run(root: Path, name: str, cwd_rel: Path, cmd: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=root / cwd_rel,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd_rel),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "counts": _parse_counts(proc.stdout),
        "stdout": proc.stdout,
        "stdout_tail": proc.stdout.splitlines()[-120:],
    }


def _combined_stdout(results: dict[str, dict[str, Any]], names: tuple[str, ...]) -> str:
    chunks: list[str] = []
    for name in names:
        result = results.get(name, {})
        chunks.append(str(result.get("stdout", "")))
        chunks.extend(str(line) for line in result.get("stdout_tail", []))
    return "\n".join(chunks)


def _missing_engine_markers(results: dict[str, dict[str, Any]]) -> list[str]:
    text = _combined_stdout(results, ("engine_vl_media_cache_contracts",))
    return [marker for marker in REQUIRED_VL_MEDIA_CACHE_TEST_MARKERS if marker not in text]


def _missing_panel_markers(results: dict[str, dict[str, Any]]) -> list[str]:
    text = _combined_stdout(
        results,
        (
            "panel_vl_media_followup_contracts",
            "panel_vlm_settings_contracts",
            "panel_vlm_family_detection",
        ),
    )
    return [marker for marker in REQUIRED_PANEL_VL_MEDIA_TEST_MARKERS if marker not in text]


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    missing_engine_markers = _missing_engine_markers(results)
    missing_panel_markers = _missing_panel_markers(results)
    settings_passed = results["panel_vlm_settings_contracts"]["counts"]["passed"] or 0
    checks = {
        "video_url_request_schema": (
            not failed
            and not missing_engine_markers
            and "test_video_url_parsed" not in missing_engine_markers
        ),
        "video_fallback_processing": (
            not failed
            and "test_jangtq_video_fallback_no_crash" not in missing_engine_markers
            and "test_video_fallback_installs_on_jang_vlm_load" not in missing_engine_markers
        ),
        "qwen36_vl_video_detection": (
            not failed
            and "test_qwen36_jangtq_is_detected_as_vl" not in missing_engine_markers
            and "test_jang_config_qwen36_can_be_video" not in missing_engine_markers
        ),
        "media_cache_salt_separates_modal_inputs": (
            not failed
            and "test_multi_turn_alternating_image_text_distinct_keys" not in missing_engine_markers
            and "test_video_token_marker_distinct_from_image" not in missing_engine_markers
        ),
        "hybrid_ssm_vlm_cache_contracts": (
            not failed
            and "test_hybrid_ssm_auto_mode_disables_live_tq_kv" not in missing_engine_markers
            and "test_deferred_rederive_covers_post_output_hybrid_ssm_paths" not in missing_engine_markers
            and "test_mllm_scheduler_wires_paged_plus_ssm_companion" not in missing_engine_markers
        ),
        "mllm_tool_replay_preserves_effective_tools": (
            not failed
            and "test_mllm_tool_replay_normalizes_tool_call_arguments_for_chat_template" not in missing_engine_markers
            and "test_mllm_chat_template_receives_effective_tool_schemas" not in missing_engine_markers
            and "test_simple_engine_forwards_effective_tools_to_mllm_chat" not in missing_engine_markers
        ),
        "panel_read_video_builtin_tool": (
            not failed
            and "exposes read_video as a file tool next to read_image" not in missing_panel_markers
        ),
        "panel_media_tool_followup_content_parts": (
            not failed
            and "injects image and video tool results as multimodal follow-up content parts" not in missing_panel_markers
            and "builds real multimodal follow-up content in text-image-video order" not in missing_panel_markers
        ),
        "panel_image_display_consistency": (
            not failed
            and "shows directory basename for Mark's external FLUX.2-klein-9B (NOT stripped FLUX2)" not in missing_panel_markers
        ),
        "panel_vlm_launch_settings": (
            not failed
            and "VLM gets --continuous-batching for BatchedEngine with MLLMScheduler" not in missing_panel_markers
            and "VLM continuous batching off emits explicit opt-out and suppresses cache stack" not in missing_panel_markers
            and "multimodal/VLM detection suppresses --enable-jit because mlx-vlm streaming is not compile-safe" not in missing_panel_markers
            and settings_passed >= 6
        ),
        "all_required_engine_markers_present": not failed and not missing_engine_markers,
        "all_required_panel_markers_present": not failed and not missing_panel_markers,
    }
    public_results = {
        name: {key: value for key, value in result.items() if key != "stdout"}
        for name, result in results.items()
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "missing_engine_markers": missing_engine_markers,
        "missing_panel_markers": missing_panel_markers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": public_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    artifact = build_artifact(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={artifact['status']}")
    print("failed=" + json.dumps(artifact["failed"]))
    print("missing_engine_markers=" + json.dumps(artifact["missing_engine_markers"]))
    print("missing_panel_markers=" + json.dumps(artifact["missing_panel_markers"]))
    for name, result in artifact["results"].items():
        counts = result["counts"]
        print(
            f"{name}: rc={result['returncode']} "
            f"passed={counts['passed']} skipped={counts['skipped']} "
            f"deselected={counts['deselected']}"
        )
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
