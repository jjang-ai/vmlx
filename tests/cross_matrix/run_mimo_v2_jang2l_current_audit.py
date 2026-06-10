#!/usr/bin/env python3
"""No-heavy MiMo V2.5 JANG_2L current artifact/runtime-boundary audit.

This script intentionally does not launch the 106G model. It answers a narrower
release question from current evidence:

* Is the local canonical bundle byte-size aligned with the TB5/HTTP source
  manifest?
* Are stale local MiMo cache/backup copies absent?
* Do the current live/probe artifacts still prove runtime/tool/long-prompt
  blockers?

It must not convert a narrow short-text/cache pass into release clearance.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANGTQ_2")
DEFAULT_MANIFEST = Path("build/current-mimo-jangtq2-local-manifest-20260607.tsv")
DEFAULT_OUT = Path(
    "build/current-mimo-v2-jang2l-current-audit-after-cache-vs-nocache-logprobs-20260609.json"
)

STRUCTURAL_ARTIFACT = Path("build/current-mimo-jang2l-local-structural-verify-20260606.json")
TEXT_CACHE_ARTIFACT = Path("build/current-mimo-jang2l-live-text-cache-smoke-20260606.json")
SWITCHGLU_ARTIFACT = Path(
    "build/current-mimo-v2-switchglu-selected-expert-parity-20260609.json"
)
LENGTH_SWEEP_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-direct-length-sweep-20260606.json"
)
TOOL_FAILURE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-tool-dialect-failure-20260606.json"
)
ALL_LOCAL_SMOKE_ARTIFACT = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-current-source-textonly-l2-after-capability-fix-20260609/JANGQ_MiMo-V2.5-JANGTQ_2/result.json"
)
JANG2L_ALL_LOCAL_SMOKE_ARTIFACT = Path(
    "build/current-all-local-model-smoke-mimo-v25-jang2l-live-refresh-20260608/summary.json"
)
KVNONE_NOPREFIX_SMOKE_ARTIFACT = Path(
    "build/current-all-local-model-smoke-mimo-v25-jangtq2-bundled-tools-nomedia-kvnone-noprefix-20260607/summary.json"
)
CACHE_VS_NOCACHE_ARTIFACT = Path(
    "build/current-mimo-v2-jangtq2-cache-vs-nocache-next-token-logprobs-after-unit-label-20260609.json"
)
SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-sink-mode-length-diagnostic-20260606.json"
)
DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-disable-sink-length-diagnostic-20260606.json"
)
PROMPT_SHAPE_SWEEP_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-prompt-shape-sweep-20260606.json"
)
RENDERED_PROMPT_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-rendered-prompt-compare-20260606.json"
)
FIRST_TOKEN_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-first-token-probe-registered-20260606.json"
)
SOURCE_VS_QUANT_ARTIFACT = Path(
    "build/current-mimo-v2-jangtq2-source-vs-quant-first-divergence-preflight-20260607.json"
)
NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT = Path(
    "build/current-mimo-v2-no-source-exactness-classifier-after-artifact-diagnosis-20260609.json"
)
SYNCED_LONG_TOOL_CACHE_ARTIFACT = Path(
    "build/current-mimo-v25-jang2l-synced-long-tool-cache-proof-20260606.json"
)
TEXT_ROUTE_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-text-route-live-proof-20260606.json"
)
CB_ONESHOT_PREFILL_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cb-cache-after-mimo-oneshot-prefill-20260606.json"
)
CB_NATIVE_THINKING_OFF_ARTIFACT = Path(
    "build/current-mimo-v2-jang2l-cb-cache-after-native-thinking-off-live-20260606.json"
)
CB_METAL_BASELINE_ARTIFACT = Path(
    "build/current-mimo-v2-jangtq2-cb-cache-lossless-auto-live-20260609.json"
)
MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT = Path(
    "build/current-mimo-v2-mllm-inputs-embeds-interface-proof-20260609.json"
)
LATEST_DECODE_SPEED_ARTIFACT = Path(
    "build/current-mimo-v25-jangtq2-speed-boundary-after-singlebatch-tokenbuffer-skip-20260609.json"
)
JANGTQ2_SOURCE_MEDIA_PROOF_ARTIFACT = Path(
    "build/current-mimo-v25-jangtq2-video-audio-source-proof-20260610.json"
)
JANGTQ2_DEV_APP_VIDEO_MEDIA_PROOF_ARTIFACT = Path(
    "build/current-real-ui-dev-app-mimo-v25-jangtq2-video-after-mllm-source-media-20260610.json"
)
JANGTQ2_VIDEO_CACHE_PROOF_ARTIFACT = Path(
    "build/current-mimo-v25-jangtq2-video-cache-proof-after-video-tensor-cache-fix-20260610.json"
)
NOHEAVY_API_CACHE_CONTRACT_ARTIFACT = Path(
    "build/current-noheavy-api-cache-contract-after-responses-reasoning-empty-final-args-gateway-20260609.json"
)
CLEANUP_LOG = Path("build/current-mimo-stale-local-cleanup-20260606.txt")

STALE_TARGETS = [
    Path("/Users/eric/vmlx-sync-backups/local-installed-mimo-v2-mlx-model-20260527-swa-mask"),
    Path(
        "/Users/eric/.cache/huggingface/modules/transformers_modules/"
        "MiMo_hyphen_V2_dot_5_hyphen_JANG_2L"
    ),
    Path("build/mimo-block-cache"),
    Path("build/current-mimo-jang2l-block-cache-20260606"),
    Path("build/current-mimo-jang2l-tool-block-cache-20260606"),
    Path("build/current-mimo-jang2l-block-cache-after-head-fix-20260606"),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _mimo_bundle_kind_from_text(text: str) -> str:
    lowered = text.lower()
    if "jangtq_2" in lowered or "jangtq2" in lowered:
        return "jangtq2"
    if "jang_2l" in lowered or "jang2l" in lowered:
        return "jang2l"
    return "unknown"


def _artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact": str(path), "exists": False, "status": "missing"}
    try:
        data = _load_json(path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted artifacts
        return {
            "artifact": str(path),
            "exists": True,
            "status": "invalid_json",
            "error": str(exc),
        }
    status = data.get("status") if isinstance(data, dict) else None
    return {"artifact": str(path), "exists": True, "status": status, "data": data}


def _ensure_smoke_bundle_identity(
    evidence: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    if not evidence.get("exists") or evidence.get("bundle_kind") != "unknown":
        return evidence
    inferred_bundle_kind = _mimo_bundle_kind_from_text(str(artifact))
    if inferred_bundle_kind == "unknown":
        return evidence
    inferred_bundle_name = (
        "MiMo-V2.5-JANGTQ_2"
        if inferred_bundle_kind == "jangtq2"
        else "MiMo-V2.5-JANG_2L"
    )
    evidence["bundle_kind"] = inferred_bundle_kind
    evidence["bundle_name"] = inferred_bundle_name
    coverage = evidence.get("bundle_media_l2_coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    for bundle_kind in ("jangtq2", "jang2l"):
        existing = coverage.get(bundle_kind)
        if not isinstance(existing, dict):
            existing = {}
        coverage[bundle_kind] = {
            "bundle_name": (
                inferred_bundle_name
                if bundle_kind == inferred_bundle_kind
                else existing.get("bundle_name")
            ),
            "live_media_e2e_pass": bool(
                bundle_kind == inferred_bundle_kind
                and evidence.get("live_media_e2e_pass")
                and evidence.get("audio_waveform_e2e_pass")
            ),
            "block_disk_l2_restart_restore_pass": bool(
                bundle_kind == inferred_bundle_kind
                and evidence.get("block_disk_l2_restart_restore_pass")
            ),
            "block_disk_l2_restart_cache_hit": bool(
                bundle_kind == inferred_bundle_kind
                and evidence.get("block_disk_l2_restart_cache_hit")
            ),
            "block_disk_l2_restart_visible_output_pass": bool(
                bundle_kind == inferred_bundle_kind
                and evidence.get("block_disk_l2_restart_visible_output_pass")
            ),
        }
    evidence["bundle_media_l2_coverage"] = coverage
    return evidence


def _merge_bundle_media_l2_coverage(
    *evidences: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        "jangtq2": {
            "bundle_name": None,
            "live_media_e2e_pass": False,
            "block_disk_l2_restart_cache_hit": False,
            "block_disk_l2_restart_restore_pass": False,
            "block_disk_l2_restart_visible_output_pass": False,
        },
        "jang2l": {
            "bundle_name": None,
            "live_media_e2e_pass": False,
            "block_disk_l2_restart_cache_hit": False,
            "block_disk_l2_restart_restore_pass": False,
            "block_disk_l2_restart_visible_output_pass": False,
        },
    }
    for evidence in evidences:
        coverage = evidence.get("bundle_media_l2_coverage")
        if not isinstance(coverage, dict):
            continue
        for bundle_kind in ("jangtq2", "jang2l"):
            item = coverage.get(bundle_kind)
            if not isinstance(item, dict):
                continue
            target = merged[bundle_kind]
            if item.get("bundle_name"):
                target["bundle_name"] = item.get("bundle_name")
            target["live_media_e2e_pass"] = bool(
                target["live_media_e2e_pass"] or item.get("live_media_e2e_pass")
            )
            target["block_disk_l2_restart_cache_hit"] = bool(
                target["block_disk_l2_restart_cache_hit"]
                or item.get("block_disk_l2_restart_cache_hit")
            )
            target["block_disk_l2_restart_restore_pass"] = bool(
                target["block_disk_l2_restart_restore_pass"]
                or item.get("block_disk_l2_restart_restore_pass")
            )
            target["block_disk_l2_restart_visible_output_pass"] = bool(
                target["block_disk_l2_restart_visible_output_pass"]
                or item.get("block_disk_l2_restart_visible_output_pass")
            )
    return merged


def _verify_manifest(model_parent: Path, manifest: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifest": str(manifest),
        "model_parent": str(model_parent),
        "status": "missing_manifest",
        "rows": 0,
        "total_bytes": 0,
        "ok_files": 0,
        "missing_files": [],
        "mismatched_files": [],
    }
    if not manifest.exists():
        return result

    rows: list[tuple[int, str]] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        size_text, rel = line.split("\t", 1)
        rows.append((int(size_text), rel))

    result["rows"] = len(rows)
    result["total_bytes"] = sum(size for size, _ in rows)
    for expected_size, rel in rows:
        path = model_parent / rel
        if not path.exists():
            result["missing_files"].append(rel)
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            result["mismatched_files"].append(
                {"path": rel, "expected_size": expected_size, "actual_size": actual_size}
            )
            continue
        result["ok_files"] += 1

    result["status"] = (
        "pass"
        if result["ok_files"] == len(rows)
        and not result["missing_files"]
        and not result["mismatched_files"]
        else "fail"
    )
    return result


def _bookend_quantization_audit(model_path: Path) -> dict[str, Any]:
    """Inspect MiMo q8 bookend sidecars without loading the model."""
    required = [
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
    ]
    result: dict[str, Any] = {
        "status": "missing_index",
        "required_keys": required,
        "missing_keys": [],
        "keys": {},
        "boundary": (
            "This no-load check only proves lm_head/embed_tokens affine sidecars "
            "exist with q8-compatible shapes. It does not clear MiMo exactness; "
            "literal mutation can still be artifact/logit or runtime decode quality."
        ),
    }
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return result
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            result["status"] = "invalid_index"
            return result
        missing = [key for key in required if key not in weight_map]
        result["missing_keys"] = missing
        if missing:
            result["status"] = "missing_bookend_sidecars"
            return result
        from safetensors import safe_open

        grouped: dict[str, list[str]] = {}
        for key in required:
            grouped.setdefault(str(weight_map[key]), []).append(key)
        for shard, keys in grouped.items():
            with safe_open(str(model_path / shard), framework="numpy") as sf:
                for key in keys:
                    tensor = sf.get_tensor(key)
                    result["keys"][key] = {
                        "file": shard,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
        lm_weight = result["keys"].get("lm_head.weight", {})
        lm_scales = result["keys"].get("lm_head.scales", {})
        embed_weight = result["keys"].get("model.embed_tokens.weight", {})
        embed_scales = result["keys"].get("model.embed_tokens.scales", {})
        q8_shapes = (
            lm_weight.get("dtype") == "uint32"
            and embed_weight.get("dtype") == "uint32"
            and lm_scales.get("dtype") == "float16"
            and embed_scales.get("dtype") == "float16"
            and isinstance(lm_weight.get("shape"), list)
            and isinstance(embed_weight.get("shape"), list)
            and lm_weight.get("shape")[-1] == 1024
            and embed_weight.get("shape")[-1] == 1024
        )
        result["q8_shape_compatible"] = bool(q8_shapes)
        result["status"] = "pass" if q8_shapes else "unexpected_bookend_shapes"
        return result
    except Exception as exc:  # pragma: no cover - diagnostic robustness
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _text_cache_narrow_pass(data: dict[str, Any]) -> bool:
    requests = data.get("requests")
    if not isinstance(requests, list):
        return False
    for entry in requests:
        if not isinstance(entry, dict):
            continue
        response = entry.get("response")
        if not isinstance(response, dict):
            continue
        payload = response.get("json")
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if response.get("http_status") == 200 and content == "cache ok":
            return True
    return False


def _cache_vs_nocache_match(data: dict[str, Any]) -> bool:
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return False
    top_lists = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        top10 = row.get("top10")
        if not isinstance(top10, list):
            return False
        top_lists.append([(item.get("id"), item.get("text")) for item in top10])
    return all(items == top_lists[0] for items in top_lists[1:])


def _length_sweep_blocked(data: dict[str, Any]) -> bool:
    cases = data.get("cases")
    if not isinstance(cases, list):
        return data.get("status") == "fail"
    return any(
        isinstance(case, dict) and str(case.get("status", "")).startswith("fail")
        for case in cases
    )


def _tool_protocol_blocked(data: dict[str, Any]) -> bool:
    if data.get("status") == "fail":
        return True
    observations = data.get("runtime_observations")
    if not isinstance(observations, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("result", "")).startswith("fail")
        for item in observations
    )


def _loads_json_or_single_markdown_fence(text: str) -> tuple[Any | None, str | None]:
    """Parse JSON, accepting one ordinary markdown fence wrapper only."""
    stripped = str(text or "").strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            first = lines[0].strip().lower()
            if first in {"```", "```json", "```javascript", "```js"}:
                candidates.append("\n".join(lines[1:-1]).strip())
    last_error: str | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error


def _all_local_smoke_evidence(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    result = results[0] if isinstance(results, list) and results else data
    if not isinstance(result, dict):
        return {
            "exists": True,
            "tool_protocol_pass": False,
            "exact_cache_blocked": True,
            "speed_blocked": True,
        }
    row = result.get("row")
    row = row if isinstance(row, dict) else {}
    bundle_name = str(
        result.get("name")
        or result.get("model_name")
        or result.get("relative_path")
        or result.get("model_path")
        or row.get("name")
        or row.get("model_name")
        or row.get("relative_path")
        or row.get("path")
        or ""
    )
    bundle_kind = _mimo_bundle_kind_from_text(bundle_name)

    requests = result.get("requests")
    failures = result.get("failures")
    if not isinstance(requests, list):
        requests = []
    if not isinstance(failures, list):
        failures = []

    media_labels = [
        "vl_blue_image",
        "vl_blue_image_repeat",
        "vl_red_image_changed",
        "text_no_media_after_image",
        "vl_blue_video",
        "text_no_media_after_video",
    ]
    requests_by_label = {
        str(request.get("label")): request
        for request in requests
        if isinstance(request, dict) and request.get("label") is not None
    }
    live_capabilities = (
        result.get("capabilities", {}).get("body", {})
        if isinstance(result.get("capabilities"), dict)
        else {}
    )
    capabilities = live_capabilities if isinstance(live_capabilities, dict) else {}
    if not capabilities:
        capabilities = row.get("capabilities") if isinstance(row, dict) else None
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    media_capabilities = (
        capabilities.get("media") if isinstance(capabilities.get("media"), dict) else {}
    )
    media_status_by_modality = (
        media_capabilities.get("status_by_modality")
        if isinstance(media_capabilities.get("status_by_modality"), dict)
        else {}
    )
    media_memory_gate = (
        media_capabilities.get("memory_gate")
        if isinstance(media_capabilities.get("memory_gate"), dict)
        else {}
    )
    modalities = capabilities.get("modalities")
    if not isinstance(modalities, list):
        modalities = []
    preserved_modalities = media_capabilities.get("preserved_modalities")
    if not isinstance(preserved_modalities, list):
        preserved_modalities = capabilities.get("preserved_modalities")
    if not isinstance(preserved_modalities, list):
        preserved_modalities = []
    media_capability_memory_gated = bool(
        bundle_kind == "jang2l"
        and (
            media_memory_gate.get("safe") is False
            or any(
                str(value) == "memory_gated"
                for value in media_status_by_modality.values()
            )
        )
    )
    media_capability_downscoped_to_text = bool(
        bundle_kind == "jang2l"
        and not media_capability_memory_gated
        and "text" in {str(item) for item in modalities}
        and not ({"image", "video", "audio"} & {str(item) for item in modalities})
        and ({"vision", "audio"} & {str(item) for item in preserved_modalities})
    )
    live_media_e2e_labels = []
    live_media_failures = []
    for label in media_labels:
        request = requests_by_label.get(label)
        if not isinstance(request, dict):
            live_media_failures.append({"label": label, "reason": "missing_request"})
            continue
        validation_failures = request.get("validation_failures")
        if not isinstance(validation_failures, list):
            validation_failures = []
        if request.get("code") != 200:
            live_media_failures.append(
                {"label": label, "reason": "http_status", "code": request.get("code")}
            )
            continue
        if validation_failures:
            live_media_failures.extend(
                failure
                for failure in validation_failures
                if isinstance(failure, dict)
            )
            continue
        live_media_e2e_labels.append(label)
    live_media_e2e_pass = len(live_media_e2e_labels) == len(media_labels)

    audio_labels = [
        "audio_blue",
        "text_no_media_after_audio",
    ]
    audio_waveform_e2e_labels = []
    audio_waveform_failures = []
    for label in audio_labels:
        request = requests_by_label.get(label)
        if not isinstance(request, dict):
            audio_waveform_failures.append({"label": label, "reason": "missing_request"})
            continue
        validation_failures = request.get("validation_failures")
        if not isinstance(validation_failures, list):
            validation_failures = []
        if request.get("code") != 200:
            audio_waveform_failures.append(
                {"label": label, "reason": "http_status", "code": request.get("code")}
            )
            continue
        if validation_failures:
            audio_waveform_failures.extend(
                failure
                for failure in validation_failures
                if isinstance(failure, dict)
            )
            continue
        audio_waveform_e2e_labels.append(label)
    audio_waveform_e2e_pass = len(audio_waveform_e2e_labels) == len(audio_labels)
    audio_failure_reasons = {
        str(failure.get("reason"))
        for failure in audio_waveform_failures
        if isinstance(failure, dict)
    }
    if audio_waveform_e2e_pass:
        audio_waveform_classification = "audio_waveform_e2e_clear"
    elif "audio_processor_payload_missing" in audio_failure_reasons:
        audio_waveform_classification = "mimo_waveform_to_audio_codes_bridge_missing"
    elif "expected_audio_transcript_missing" in audio_failure_reasons:
        audio_waveform_classification = "audio_reached_runtime_but_transcript_wrong"
    else:
        audio_waveform_classification = "audio_waveform_e2e_missing_or_failed"

    media_prefill_budget_labels = [
        label
        for label in [
            "vl_blue_image",
            "vl_blue_image_repeat",
            "vl_red_image_changed",
            "vl_blue_video",
            "audio_blue",
        ]
        if isinstance(requests_by_label.get(label), dict)
        and requests_by_label[label].get("code") == 413
    ]
    post_media_working_set_pressure_labels = [
        label
        for label in [
            "text_no_media_after_image",
            "text_no_media_after_video",
            "text_no_media_after_audio",
        ]
        if isinstance(requests_by_label.get(label), dict)
        and requests_by_label[label].get("code") == 503
    ]

    tool_protocol_pass = False
    for request in requests:
        if not isinstance(request, dict) or request.get("label") != "tool_required":
            continue
        tool_calls = request.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and function.get("name") == "record_fact":
                tool_protocol_pass = True
                break
        if tool_protocol_pass:
            break

    exact_cache_blocked = any(
        isinstance(failure, dict)
        and failure.get("label") in {"text_cache_repeat_1", "text_cache_repeat_2"}
        for failure in failures
    )
    cache_repeat_1 = requests_by_label.get("text_cache_repeat_1")
    cache_repeat_2 = requests_by_label.get("text_cache_repeat_2")

    def _request_text(request):
        if not isinstance(request, dict):
            return None
        return (
            request.get("content")
            or request.get("text")
            or request.get("output_text")
        )

    def _request_cached_tokens(request) -> int:
        if not isinstance(request, dict):
            return 0
        usage = request.get("usage") if isinstance(request.get("usage"), dict) else {}
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage.get("prompt_tokens_details"), dict)
            else {}
        )
        cache_summary = (
            request.get("cache_summary")
            if isinstance(request.get("cache_summary"), dict)
            else {}
        )
        cached_tokens = details.get("cached_tokens")
        if isinstance(cached_tokens, int):
            return cached_tokens
        cached_tokens = cache_summary.get("cache_hit_tokens")
        return cached_tokens if isinstance(cached_tokens, int) else 0

    text_cache_narrow_pass = bool(
        isinstance(cache_repeat_1, dict)
        and isinstance(cache_repeat_2, dict)
        and _request_text(cache_repeat_1) == "ACK"
        and _request_text(cache_repeat_2) == "ACK"
        and not exact_cache_blocked
        and _request_cached_tokens(cache_repeat_2) > 0
    )
    exactness_failures = [
        failure
        for failure in failures
        if isinstance(failure, dict)
        and failure.get("label")
        in {
            "tool_required",
            "mimo_tool_required_sentinel",
            "structured_json_exact",
            "mimo_structured_json_sentinel",
            "exact_code_whitespace",
            "tool_result_continuation",
        }
    ]
    generation_tps = (
        result.get("cache_after", {})
        .get("body", {})
        .get("scheduler_stats", {})
        .get("batch_generator", {})
        .get("generation_tps")
    )
    speed_blocked = not (
        isinstance(generation_tps, (int, float)) and generation_tps >= 40.0
    )
    l2_restart = result.get("l2_restart")
    l2_restart = l2_restart if isinstance(l2_restart, dict) else {}
    l2_restart_summary = (
        l2_restart.get("summary")
        if isinstance(l2_restart.get("summary"), dict)
        else {
            "status": "missing",
            "pass": False,
            "reason": "l2_restart_probe_not_run",
        }
    )
    block_disk_l2_restart_restore_pass = bool(l2_restart_summary.get("pass"))
    l2_usage = l2_restart.get("usage") if isinstance(l2_restart, dict) else None
    l2_details = (
        l2_usage.get("prompt_tokens_details") if isinstance(l2_usage, dict) else None
    )
    l2_cache_stats = l2_restart.get("cache_stats") if isinstance(l2_restart, dict) else None
    l2_scheduler_stats = (
        l2_cache_stats.get("scheduler_stats")
        if isinstance(l2_cache_stats, dict)
        else None
    )
    l2_block_disk = (
        l2_cache_stats.get("block_disk_cache")
        if isinstance(l2_cache_stats, dict)
        else None
    )
    block_disk_l2_restart_cache_hit = bool(
        (
            isinstance(l2_details, dict)
            and isinstance(l2_details.get("cached_tokens"), int)
            and l2_details.get("cached_tokens", 0) > 0
            and "disk" in str(l2_details.get("cache_detail") or "")
        )
        or (
            isinstance(l2_scheduler_stats, dict)
            and isinstance(l2_scheduler_stats.get("cache_hit_tokens"), int)
            and l2_scheduler_stats.get("cache_hit_tokens", 0) > 0
            and "disk" in str(l2_scheduler_stats.get("cache_hit_tokens_by_detail") or "")
        )
        or (
            isinstance(l2_block_disk, dict)
            and isinstance(l2_block_disk.get("disk_hits"), int)
            and l2_block_disk.get("disk_hits", 0) > 0
        )
        or block_disk_l2_restart_restore_pass
    )
    bundle_media_l2_coverage = {
        "jangtq2": {
            "bundle_name": bundle_name if bundle_kind == "jangtq2" else None,
            "live_media_e2e_pass": bool(
                bundle_kind == "jangtq2"
                and live_media_e2e_pass
                and audio_waveform_e2e_pass
            ),
            "block_disk_l2_restart_cache_hit": bool(
                bundle_kind == "jangtq2" and block_disk_l2_restart_cache_hit
            ),
            "block_disk_l2_restart_restore_pass": bool(
                bundle_kind == "jangtq2" and block_disk_l2_restart_restore_pass
            ),
            "block_disk_l2_restart_visible_output_pass": bool(
                bundle_kind == "jangtq2" and block_disk_l2_restart_restore_pass
            ),
        },
        "jang2l": {
            "bundle_name": bundle_name if bundle_kind == "jang2l" else None,
            "live_media_e2e_pass": bool(
                bundle_kind == "jang2l"
                and live_media_e2e_pass
                and audio_waveform_e2e_pass
            ),
            "block_disk_l2_restart_cache_hit": bool(
                bundle_kind == "jang2l" and block_disk_l2_restart_cache_hit
            ),
            "block_disk_l2_restart_restore_pass": bool(
                bundle_kind == "jang2l" and block_disk_l2_restart_restore_pass
            ),
            "block_disk_l2_restart_visible_output_pass": bool(
                bundle_kind == "jang2l" and block_disk_l2_restart_restore_pass
            ),
        },
    }
    exactness_boundary = _artifact_exactness_boundary(result, exactness_failures)
    server_log_tail = str(result.get("server_log_tail") or "")
    request_errors = " ".join(
        str(request.get("error") or "")
        for request in requests
        if isinstance(request, dict)
    )
    metal_oom_crash = (
        "OutOfMemory" in server_log_tail
        or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in server_log_tail
        or "Insufficient Memory" in server_log_tail
    )
    server_died_mid_probe = any(
        isinstance(request, dict)
        and request.get("code") == 0
        and (
            "RemoteDisconnected" in str(request.get("error") or "")
            or "Connection refused" in str(request.get("error") or "")
        )
        for request in requests
    )
    typed_tight_memory_prompt_rejections = [
        request
        for request in requests
        if isinstance(request, dict)
        and request.get("code") == 413
        and "mimo_v2_tight_memory_text_prefill"
        in str(request.get("error") or "")
    ]
    return {
        "exists": True,
        "bundle_name": bundle_name or None,
        "bundle_kind": bundle_kind,
        "bundle_media_l2_coverage": bundle_media_l2_coverage,
        "tool_protocol_pass": tool_protocol_pass,
        "text_cache_narrow_pass": text_cache_narrow_pass,
        "artifact_exactness_pass": not bool(exactness_failures),
        "artifact_exactness_failures": exactness_failures,
        "artifact_exactness_boundary": exactness_boundary,
        "live_media_e2e_pass": live_media_e2e_pass,
        "live_media_e2e_status": "pass" if live_media_e2e_pass else "missing_or_failed",
        "live_media_e2e_labels": live_media_e2e_labels,
        "live_media_e2e_failures": live_media_failures,
        "audio_waveform_e2e_pass": audio_waveform_e2e_pass,
        "audio_waveform_e2e_status": (
            "pass" if audio_waveform_e2e_pass else "missing_or_failed"
        ),
        "audio_waveform_e2e_classification": audio_waveform_classification,
        "audio_waveform_e2e_labels": audio_waveform_e2e_labels,
        "audio_waveform_e2e_failures": audio_waveform_failures,
        "media_capability_downscoped_to_text": media_capability_downscoped_to_text,
        "media_capability_memory_gated": media_capability_memory_gated,
        "media_capability_status_by_modality": media_status_by_modality,
        "media_capability_memory_gate": media_memory_gate,
        "media_capability_modalities": [str(item) for item in modalities],
        "media_capability_preserved_modalities": [
            str(item) for item in preserved_modalities
        ],
        "typed_media_prefill_budget_rejection": bool(media_prefill_budget_labels),
        "typed_media_prefill_budget_rejection_labels": media_prefill_budget_labels,
        "post_media_working_set_pressure_rejection": bool(
            post_media_working_set_pressure_labels
        ),
        "post_media_working_set_pressure_rejection_labels": (
            post_media_working_set_pressure_labels
        ),
        "exact_cache_blocked": exact_cache_blocked,
        "speed_blocked": speed_blocked,
        "generation_tps": generation_tps,
        "block_disk_l2_restart_restore_pass": bool(block_disk_l2_restart_restore_pass),
        "block_disk_l2_restart_cache_hit": bool(block_disk_l2_restart_cache_hit),
        "block_disk_l2_restart_visible_output_pass": bool(
            block_disk_l2_restart_restore_pass
        ),
        "block_disk_l2_restart_restore_status": str(
            l2_restart_summary.get("status") or "missing"
        ),
        "block_disk_l2_restart_restore_reason": str(
            l2_restart_summary.get("reason") or "unknown"
        ),
        "block_disk_l2_restart_restore_summary": l2_restart_summary,
        "server_died_mid_probe": bool(server_died_mid_probe),
        "metal_oom_crash": bool(metal_oom_crash),
        "typed_tight_memory_prompt_rejection": bool(
            typed_tight_memory_prompt_rejections
        ),
        "typed_tight_memory_prompt_rejection_labels": [
            str(request.get("label"))
            for request in typed_tight_memory_prompt_rejections
        ],
        "runtime_crash_error_excerpt": (
            " ".join((request_errors + " " + server_log_tail).split())[:500]
            if (server_died_mid_probe or metal_oom_crash)
            else None
        ),
        "failures": failures,
    }


def _kvnone_noprefix_exactness_evidence(data: dict[str, Any]) -> dict[str, Any]:
    evidence = _all_local_smoke_evidence(data)
    results = data.get("results")
    result = results[0] if isinstance(results, list) and results else data
    if not isinstance(result, dict):
        evidence.update(
            {
                "classification": "invalid_kvnone_noprefix_smoke_shape",
                "exactness_not_caused_by_prefix_or_kv_quant": False,
            }
        )
        return evidence

    health_before = result.get("health_before")
    health_body = health_before if isinstance(health_before, dict) else {}
    if isinstance(health_body.get("body"), dict):
        health_body = health_body["body"]
    kv_cache_quantization = health_body.get("kv_cache_quantization")
    kv_quant_disabled = (
        isinstance(kv_cache_quantization, dict)
        and kv_cache_quantization.get("enabled") is False
    )
    native_cache = health_body.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    storage_quantization = native_cache.get("storage_quantization")
    storage_quant_disabled = (
        isinstance(storage_quantization, dict)
        and storage_quantization.get("enabled") is False
    )
    prefix_disabled = native_cache.get("prefix") is False
    paged_disabled = native_cache.get("paged") is False
    block_l2_disabled = native_cache.get("block_disk_l2") is False
    cache_hits_zero = bool(
        result.get("cache_after", {})
        .get("body", {})
        .get("scheduler_stats", {})
        .get("cache", {})
        .get("cache_hits", 0)
        == 0
    )
    failed_labels = [
        failure.get("label")
        for failure in evidence.get("artifact_exactness_failures", [])
        if isinstance(failure, dict)
    ]
    required_exactness_failures = {
        "tool_required",
        "mimo_tool_required_sentinel",
        "structured_json_exact",
        "mimo_structured_json_sentinel",
    }
    exactness_reproduced = required_exactness_failures.issubset(set(failed_labels))
    exactness_not_cache = (
        kv_quant_disabled
        and storage_quant_disabled
        and prefix_disabled
        and paged_disabled
        and block_l2_disabled
        and cache_hits_zero
        and exactness_reproduced
    )
    evidence.update(
        {
            "kv_cache_quantization_disabled": kv_quant_disabled,
            "native_storage_quantization_disabled": storage_quant_disabled,
            "prefix_cache_disabled": prefix_disabled,
            "paged_cache_disabled": paged_disabled,
            "block_disk_l2_disabled": block_l2_disabled,
            "cache_hits_zero": cache_hits_zero,
            "failed_labels": failed_labels,
            "exactness_reproduced_without_prefix_or_kv_quant": exactness_reproduced,
            "exactness_not_caused_by_prefix_or_kv_quant": exactness_not_cache,
            "classification": (
                "mimo_literal_exactness_reproduces_with_prefix_paged_l2_and_kv_quant_disabled"
                if exactness_not_cache
                else "mimo_kvnone_noprefix_isolation_inconclusive"
            ),
        }
    )
    return evidence


def _artifact_exactness_boundary(
    result: dict[str, Any],
    exactness_failures: list[Any],
) -> dict[str, Any]:
    exact_labels = {
        "tool_required",
        "mimo_tool_required_sentinel",
        "structured_json_exact",
        "mimo_structured_json_sentinel",
        "exact_code_whitespace",
        "tool_result_continuation",
    }
    failures = [
        failure
        for failure in exactness_failures
        if isinstance(failure, dict) and failure.get("label") in exact_labels
    ]
    if not failures:
        return {
            "classification": "artifact_exactness_clear",
            "parser_structure_valid_for_failed_rows": True,
            "model_generated_literal_mutation": False,
            "failed_labels": [],
            "examples": [],
        }

    requests = result.get("requests")
    requests = requests if isinstance(requests, list) else []
    request_by_label = {
        str(request.get("label")): request
        for request in requests
        if isinstance(request, dict) and request.get("label") is not None
    }

    examples: list[dict[str, Any]] = []
    parser_structure_valid = True
    literal_mutation = False
    empty_visible_output = False
    server_died_mid_probe = False
    typed_tight_memory_prompt_rejection = False
    for failure in failures:
        label = str(failure.get("label"))
        request = request_by_label.get(label) or {}
        expected = failure.get("expected")
        actual = failure.get("actual")
        reason = str(failure.get("reason") or "")
        content = request.get("content")
        request_error = str(request.get("error") or failure.get("error") or "")
        if request.get("code") == 0 or reason == "http_status":
            if "RemoteDisconnected" in request_error or "Connection refused" in request_error:
                server_died_mid_probe = True
            if (
                request.get("code") == 413
                and "mimo_v2_tight_memory_text_prefill" in request_error
            ):
                typed_tight_memory_prompt_rejection = True
        row_empty_visible = reason == "empty_visible"
        empty_visible_output = empty_visible_output or row_empty_visible
        validation_failures = request.get("validation_failures")
        if isinstance(validation_failures, list):
            for request_failure in validation_failures:
                if not isinstance(request_failure, dict):
                    continue
                if request_failure.get("label") != label:
                    continue
                if request_failure.get("reason") == "empty_visible":
                    row_empty_visible = True
                    empty_visible_output = True
                if expected is None:
                    expected = request_failure.get("expected")
                if actual is None:
                    actual = request_failure.get("actual")
                break
        structure_valid = False
        structure_kind = "unknown"

        if label in {"tool_required", "mimo_tool_required_sentinel"}:
            structure_kind = "openai_tool_call_json_arguments"
            tool_calls = request.get("tool_calls")
            tool_calls = tool_calls if isinstance(tool_calls, list) else []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                if function.get("name") != "record_fact":
                    continue
                arguments = function.get("arguments")
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else {}
                except Exception:
                    parsed = {}
                if isinstance(parsed, dict):
                    structure_valid = True
                    if actual is None:
                        actual = parsed
                    break
        elif label in {"structured_json_exact", "mimo_structured_json_sentinel"}:
            structure_kind = "visible_json_object"
            if not row_empty_visible:
                parsed, _ = _loads_json_or_single_markdown_fence(
                    content if isinstance(content, str) else ""
                )
                if not isinstance(parsed, dict):
                    parsed = {}
                if isinstance(parsed, dict):
                    structure_valid = True
                    if actual is None:
                        actual = parsed
        elif label == "tool_result_continuation":
            structure_kind = "visible_tool_result_continuation_text"
            structure_valid = isinstance(request.get("content"), str)
        elif label == "exact_code_whitespace":
            structure_kind = "visible_exact_text"
            structure_valid = isinstance(request.get("content"), str)

        parser_structure_valid = parser_structure_valid and structure_valid
        if structure_valid and expected is not None and actual is not None and actual != expected:
            literal_mutation = True
        examples.append(
            {
                "label": label,
                "reason": reason,
                "structure_kind": structure_kind,
                "parser_structure_valid": structure_valid,
                "expected": expected,
                "actual": actual,
                "content_head": failure.get("content_head"),
            }
        )

    server_log_tail = str(result.get("server_log_tail") or "")
    metal_oom_crash = (
        "OutOfMemory" in server_log_tail
        or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in server_log_tail
        or "Insufficient Memory" in server_log_tail
    )

    if server_died_mid_probe and metal_oom_crash:
        classification = "server_died_mid_probe_metal_oom"
    elif server_died_mid_probe:
        classification = "server_died_mid_probe_before_exactness_rows_completed"
    elif typed_tight_memory_prompt_rejection:
        classification = "typed_tight_memory_prompt_budget_rejection"
    elif parser_structure_valid and literal_mutation:
        classification = "model_generated_literal_mutation_after_valid_parser_structure"
    elif empty_visible_output:
        classification = "empty_visible_output_after_generation_stop"
    elif parser_structure_valid:
        classification = "exactness_failure_after_valid_parser_structure"
    else:
        classification = "parser_or_output_structure_failure"

    if server_died_mid_probe and metal_oom_crash:
        release_boundary = (
            "Do not clear MiMo JANG_2L exactness by parser repair, JSON repair, "
            "or hiding failed rows. The current bundled-runtime proof shows "
            "the server dies mid-probe from Metal OOM before tool/JSON/code "
            "rows can complete; fix memory/runtime scheduling or use a smaller "
            "release artifact before claiming these rows."
        )
    elif server_died_mid_probe:
        release_boundary = (
            "Do not clear MiMo JANG_2L exactness by parser repair or prompt "
            "rewrites. The server disconnected before exactness rows completed; "
            "classify and fix the runtime crash first."
        )
    elif typed_tight_memory_prompt_rejection:
        release_boundary = (
            "Do not clear MiMo JANG_2L tool/long-prompt rows by hiding typed "
            "413 responses or lowering harness budgets. The current runtime "
            "keeps the server alive under extreme Metal pressure, but the "
            "JANG_2L bundle still cannot execute these rows with the requested "
            "prompt plus generation budget on this 107.5GB working-set host."
        )
    elif empty_visible_output:
        release_boundary = (
            "Do not clear MiMo exactness by JSON repair, parser rewrites, "
            "cache-hit accounting, or hiding failed rows. empty visible output "
            "after a 200 response/completion token means the remaining blocker "
            "is generation stop/decode-loop/template interaction or artifact "
            "quality, not JSON parser mutation."
        )
    elif literal_mutation:
        release_boundary = (
            "Do not clear MiMo exactness by rewriting parsed tool arguments, "
            "repairing generated JSON values, disabling cache, or hiding the "
            "failed rows. Valid parser structure plus wrong literal values "
            "means the remaining blocker is model/runtime decode quality or "
            "artifact quantization, not XML/JSON parser mutation."
        )
    else:
        release_boundary = (
            "Do not clear MiMo exactness by parser repair, prompt-only folding, "
            "cache disabling, or hiding failed rows. The remaining blocker "
            "requires runtime/model output proof for the failed exactness rows."
        )

    return {
        "classification": classification,
        "parser_structure_valid_for_failed_rows": parser_structure_valid,
        "model_generated_literal_mutation": literal_mutation,
        "empty_visible_output": empty_visible_output,
        "server_died_mid_probe": server_died_mid_probe,
        "metal_oom_crash": metal_oom_crash,
        "typed_tight_memory_prompt_rejection": typed_tight_memory_prompt_rejection,
        "failed_labels": [str(failure.get("label")) for failure in failures],
        "examples": examples,
        "release_boundary": release_boundary,
    }


def _all_sink_diagnostic_cases_fail(data: dict[str, Any]) -> bool:
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return False
    return all(
        isinstance(case, dict) and case.get("passes_length_ok") is False
        for case in cases
    )


def _mimo_prompt_shape_first_token_evidence(
    prompt_shape: dict[str, Any],
    rendered_prompt: dict[str, Any],
    first_token: dict[str, Any],
) -> dict[str, Any]:
    results = prompt_shape.get("results")
    if not isinstance(results, list):
        results = []
    by_name = {
        str(item.get("name")): item
        for item in results
        if isinstance(item, dict) and item.get("name") is not None
    }

    failing = by_name.get("long_cache_system_exact") or {}
    folded = by_name.get("long_cache_no_system_exact") or {}
    short_system = by_name.get("short_system_exact") or {}

    rendered_rows = rendered_prompt.get("rows")
    if not isinstance(rendered_rows, list):
        rendered_rows = []
    rendered_by_name = {
        str(item.get("name")): item
        for item in rendered_rows
        if isinstance(item, dict) and item.get("name") is not None
    }
    failing_rendered = rendered_by_name.get("long_cache_system_exact") or {}

    first_rows = first_token.get("rows")
    if not isinstance(first_rows, list):
        first_rows = []
    first_by_name = {
        str(item.get("name")): item
        for item in first_rows
        if isinstance(item, dict) and item.get("name") is not None
    }
    failing_top = (first_by_name.get("failing_system_long") or {}).get("top")
    folded_top = (first_by_name.get("working_folded_long") or {}).get("top")
    short_top = (first_by_name.get("working_short_system") or {}).get("top")
    if not isinstance(failing_top, list):
        failing_top = []
    if not isinstance(folded_top, list):
        folded_top = []
    if not isinstance(short_top, list):
        short_top = []

    failing_top0 = failing_top[0] if failing_top and isinstance(failing_top[0], dict) else {}
    folded_top0 = folded_top[0] if folded_top and isinstance(folded_top[0], dict) else {}
    short_top0 = short_top[0] if short_top and isinstance(short_top[0], dict) else {}
    ack_in_failing_rank = None
    for index, item in enumerate(failing_top, start=1):
        if isinstance(item, dict) and item.get("text") == "ACK":
            ack_in_failing_rank = index
            break

    failing_empty_stop = (
        failing.get("content") == ""
        and failing.get("finish_reason") == "stop"
        and (failing.get("completion_tokens") == 1)
    )
    folded_ack = folded.get("content") == "ACK"
    short_ack = short_system.get("content") == "ACK"
    rendered_valid_generation_prefix = (
        failing_rendered.get("contains_empty_think_generation_prefix") is True
    )
    first_token_stop = (
        failing_top0.get("text") == "<|im_end|>"
        and failing_top0.get("is_special") is True
    )
    working_first_tokens_ack = (
        folded_top0.get("text") == "ACK" and short_top0.get("text") == "ACK"
    )

    blocked = (
        failing_empty_stop
        and folded_ack
        and short_ack
        and rendered_valid_generation_prefix
        and first_token_stop
        and working_first_tokens_ack
    )
    return {
        "blocked": bool(blocked),
        "classification": (
            "decode_loop_or_model_artifact_first_token_stop"
            if blocked
            else "insufficient_prompt_shape_evidence"
        ),
        "failing_empty_stop": bool(failing_empty_stop),
        "folded_user_ack": bool(folded_ack),
        "short_system_ack": bool(short_ack),
        "rendered_valid_generation_prefix": bool(rendered_valid_generation_prefix),
        "failing_top_token": failing_top0,
        "working_folded_top_token": folded_top0,
        "working_short_system_top_token": short_top0,
        "ack_rank_in_failing_prompt": ack_in_failing_rank,
        "failing_prompt_tokens": failing.get("prompt_tokens"),
        "folded_prompt_tokens": folded.get("prompt_tokens"),
    }


def _synced_long_tool_cache_evidence(data: dict[str, Any]) -> dict[str, Any]:
    checks = data.get("checks")
    if not isinstance(checks, list):
        checks = []
    by_name = {
        str(item.get("name")): item
        for item in checks
        if isinstance(item, dict) and item.get("name") is not None
    }
    failed = data.get("failed_checks")
    if not isinstance(failed, list):
        failed = []
    long_row = by_name.get("long_prompt_recall") or {}
    tool_row = by_name.get("tool_required") or {}
    cache_rows = [by_name.get("cache_repeat_1") or {}, by_name.get("cache_repeat_2") or {}]
    long_failed = any(
        isinstance(item, dict) and item.get("label") == "long_prompt_recall"
        for item in failed
    )
    tool_failed = any(
        isinstance(item, dict) and item.get("label") == "tool_required"
        for item in failed
    ) or not bool(tool_row.get("tool_calls"))
    cache_exact_pass = all(
        isinstance(row, dict) and row.get("text") == "ACK-CACHE-742"
        for row in cache_rows
    )
    return {
        "exists": True,
        "verdict": data.get("verdict"),
        "long_prompt_blocked": bool(long_failed),
        "tool_protocol_blocked": bool(tool_failed),
        "short_cache_exact_pass": bool(cache_exact_pass),
        "long_prompt_elapsed_s": long_row.get("elapsed_s"),
        "tool_elapsed_s": tool_row.get("elapsed_s"),
        "long_prompt_text_head": str(long_row.get("text") or "")[:200],
        "tool_text_head": str(tool_row.get("text") or "")[:200],
        "failed_checks": failed,
    }


def _source_vs_quant_first_divergence_pass(data: dict[str, Any]) -> bool:
    if data.get("status") != "pass" or data.get("remote_evidence_only") is True:
        return False
    if not data.get("source_model_path") or not data.get("quant_model_path"):
        return False
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    allowed = {
        "source_and_quant_match",
        "quant_diverges_from_source",
        "source_also_fails",
    }
    for row in rows:
        if not isinstance(row, dict):
            return False
        if not row.get("prompt_name"):
            return False
        if row.get("classification") not in allowed:
            return False
        if "source_output" not in row or "quant_output" not in row:
            return False
    return True


def _source_vs_quant_policy_skip(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "policy_skipped": False,
            "reason": None,
            "classifier_status": None,
            "boundary": "no no-source classifier artifact was available",
        }
    excluded = data.get("excluded_surfaces")
    if not isinstance(excluded, dict):
        excluded = {}
    reason = data.get("source_vs_quant_load_skipped_reason")
    classifier_status = data.get("status")
    required_exclusions = {
        "parser_argument_rewrite": excluded.get("parser_argument_rewrite") is True,
        "prefix_paged_l2_or_kv_quant_primary_cause": (
            excluded.get("prefix_paged_l2_or_kv_quant_primary_cause") is True
        ),
        "hidden_stochastic_sampling_primary_cause": (
            excluded.get("hidden_stochastic_sampling_primary_cause") is True
        ),
    }
    policy_skipped = (
        data.get("source_vs_quant_load_performed") is False
        and reason
        in {
            "user_disallowed_source_vs_quant_due_ram",
            "user_disallowed_source_vs_quant",
            "source_vs_quant_skipped_by_policy",
        }
        and classifier_status in {"open", "pass"}
        and all(required_exclusions.values())
    )
    return {
        "policy_skipped": bool(policy_skipped),
        "reason": reason,
        "classifier_status": classifier_status,
        "required_exclusions": required_exclusions,
        "classification": data.get("classification"),
        "secondary_classification": data.get("secondary_classification"),
        "unresolved_surfaces": data.get("unresolved_surfaces")
        if isinstance(data.get("unresolved_surfaces"), dict)
        else {},
        "boundary": (
            "source-vs-quant was not run because policy/RAM disallows it; "
            "this removes only the source-comparison blocker and does not "
            "release-clear unresolved MiMo exactness, long-prompt, CB, or media rows"
            if policy_skipped
            else "source-vs-quant remains required unless a valid policy-skip classifier is present"
        ),
    }


def _api_cache_responses_contract_evidence(data: dict[str, Any]) -> dict[str, Any]:
    checks = data.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    required = {
        "responses_sampling_kwargs": checks.get("responses_sampling_kwargs") is True,
        "streaming_cache_detail_usage": checks.get("streaming_cache_detail_usage")
        is True,
        "responses_previous_response_history": checks.get(
            "responses_previous_response_history"
        )
        is True,
        "cache_stats_reuse_skip_telemetry": checks.get(
            "cache_stats_reuse_skip_telemetry"
        )
        is True,
        "cache_reuse_endpoints": checks.get("cache_reuse_endpoints") is True,
        "request_output_caps_override_server_default": checks.get(
            "request_output_caps_override_server_default"
        )
        is True,
        "prompt_context_caps_stay_separate_from_output_caps": checks.get(
            "prompt_context_caps_stay_separate_from_output_caps"
        )
        is True,
        "ollama_adapter_surface": checks.get("ollama_adapter_surface") is True,
        "anthropic_bundle_defaults": checks.get("anthropic_bundle_defaults") is True,
    }
    missing = [name for name, ok in required.items() if not ok]
    return {
        "exists": True,
        "status": data.get("status"),
        "pass": data.get("status") == "pass" and not missing,
        "required_checks": required,
        "missing_required_checks": missing,
        "missing_markers": data.get("missing_markers")
        if isinstance(data.get("missing_markers"), list)
        else [],
        "boundary": (
            "No-heavy source contract only. This proves API/cache/Responses "
            "route behavior, telemetry, and endpoint wiring; it does not replace "
            "live MiMo model output, tool, cache-hit, L2/restart, media, or UI proof."
        ),
    }


def _mimo_runtime_capabilities_from_smoke(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results")
    result = results[0] if isinstance(results, list) and results else data
    if not isinstance(result, dict):
        return {"exists": False}
    capabilities = (
        result.get("capabilities", {}).get("body", {})
        if isinstance(result.get("capabilities"), dict)
        else {}
    )
    media = capabilities.get("media") if isinstance(capabilities.get("media"), dict) else {}
    status_by_modality = (
        media.get("status_by_modality")
        if isinstance(media.get("status_by_modality"), dict)
        else {}
    )
    unwired = media.get("unwired_modalities")
    unwired = unwired if isinstance(unwired, list) else []
    preserved = media.get("preserved_modalities")
    preserved = preserved if isinstance(preserved, list) else []
    runtime_modalities = capabilities.get("modalities")
    runtime_modalities = runtime_modalities if isinstance(runtime_modalities, list) else []
    safe_text_only = (
        runtime_modalities == ["text"]
        and status_by_modality.get("text") == "runtime_supported"
        and all(
            status_by_modality.get(name) == "preserved_unwired"
            for name in ("vision", "image", "video", "audio")
        )
    )
    return {
        "exists": True,
        "source": "live_smoke",
        "runtime_modalities": runtime_modalities,
        "media": media,
        "preserved_modalities": preserved,
        "unwired_modalities": unwired,
        "status_by_modality": status_by_modality,
        "safe_text_only_runtime_capabilities": bool(safe_text_only),
        "runtime_capabilities_safe": bool(safe_text_only),
        "runtime_media_supported": False,
    }


def _mimo_runtime_capabilities_from_source_detector(model_path: Path) -> dict[str, Any]:
    """Return current server media-capability detection without loading weights."""
    try:
        from vmlx_engine import server
    except Exception as exc:
        return {"exists": False, "source": "server_detector", "error": str(exc)}

    try:
        modalities = server._mimo_v2_runtime_modalities(str(model_path))
    except Exception as exc:
        return {"exists": False, "source": "server_detector", "error": str(exc)}
    if not isinstance(modalities, list) or not modalities:
        return {"exists": False, "source": "server_detector", "runtime_modalities": []}

    old_model_path = getattr(server, "_model_path", None)
    old_model_name = getattr(server, "_model_name", None)
    try:
        server._model_path = str(model_path)
        server._model_name = None
        media = server._loaded_media_capability_status(modalities)
    finally:
        server._model_path = old_model_path
        server._model_name = old_model_name

    status_by_modality = (
        media.get("status_by_modality")
        if isinstance(media.get("status_by_modality"), dict)
        else {}
    )
    runtime_media_supported = all(
        status_by_modality.get(name) == "runtime_supported"
        for name in ("vision", "image", "video", "audio")
    )
    safe_text_only = (
        modalities == ["text"]
        and status_by_modality.get("text") == "runtime_supported"
        and all(
            status_by_modality.get(name) == "preserved_unwired"
            for name in ("vision", "image", "video", "audio")
        )
    )
    return {
        "exists": True,
        "source": "server_detector",
        "runtime_modalities": modalities,
        "media": media,
        "preserved_modalities": media.get("preserved_modalities", []),
        "unwired_modalities": media.get("unwired_modalities", []),
        "status_by_modality": status_by_modality,
        "safe_text_only_runtime_capabilities": bool(safe_text_only),
        "runtime_capabilities_safe": bool(safe_text_only or runtime_media_supported),
        "runtime_media_supported": bool(runtime_media_supported),
    }


def _count_index_names(model_path: Path, prefixes: tuple[str, ...]) -> int:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return 0
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        return 0
    return sum(
        1
        for name in weight_map
        if any(str(name).startswith(prefix) for prefix in prefixes)
    )


def _mimo_media_runtime_evidence(
    root: Path,
    model_path: Path,
    all_local_smoke_data: dict[str, Any] | None,
) -> dict[str, Any]:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        config = {}
        config_error = str(exc)
    else:
        config_error = None

    capabilities = (
        config.get("capabilities") if isinstance(config.get("capabilities"), dict) else {}
    )
    config_modalities = capabilities.get("modalities")
    config_preserved = capabilities.get("preserved_modalities")
    config_unwired = capabilities.get("unwired_modalities")
    config_multimodal_status = capabilities.get("multimodal_status")
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    runtime_multimodal_mode = runtime.get("multimodal_mode")

    config_text_only_contract = (
        config_modalities == ["text"]
        and config_preserved == ["vision", "audio"]
        and config_unwired == ["vision", "audio"]
        and config_multimodal_status == "weights_preserved_text_runtime"
        and runtime_multimodal_mode == "weights_preserved_text_runtime"
    )

    visual_count = _count_index_names(model_path, ("visual.",))
    audio_count = _count_index_names(
        model_path,
        ("audio_encoder.", "speech_embeddings."),
    )
    has_processor = (model_path / "preprocessor_config.json").exists()
    has_audio_tokenizer = (model_path / "audio_tokenizer").exists()
    has_audio_tokenizer_config = (model_path / "audio_tokenizer" / "config.json").exists()
    has_audio_tokenizer_weights = (
        model_path / "audio_tokenizer" / "model.safetensors"
    ).exists()
    smoke_caps = (
        _mimo_runtime_capabilities_from_smoke(all_local_smoke_data)
        if isinstance(all_local_smoke_data, dict)
        else {"exists": False}
    )
    source_caps = _mimo_runtime_capabilities_from_source_detector(model_path)
    runtime_caps = source_caps if source_caps.get("exists") else smoke_caps

    runtime_adapter = root / "vmlx_engine/models/mllm.py"
    adapter_text = (
        runtime_adapter.read_text(encoding="utf-8", errors="replace")
        if runtime_adapter.exists()
        else ""
    )
    batch_generator = root / "vmlx_engine/mllm_batch_generator.py"
    batch_generator_text = (
        batch_generator.read_text(encoding="utf-8", errors="replace")
        if batch_generator.exists()
        else ""
    )
    scheduler_path = root / "vmlx_engine/mllm_scheduler.py"
    scheduler_text = (
        scheduler_path.read_text(encoding="utf-8", errors="replace")
        if scheduler_path.exists()
        else ""
    )
    batched_engine_path = root / "vmlx_engine/engine/batched.py"
    batched_engine_text = (
        batched_engine_path.read_text(encoding="utf-8", errors="replace")
        if batched_engine_path.exists()
        else ""
    )
    scheduler_cache_tests = root / "tests/test_mllm_scheduler_cache.py"
    scheduler_cache_test_text = (
        scheduler_cache_tests.read_text(encoding="utf-8", errors="replace")
        if scheduler_cache_tests.exists()
        else ""
    )
    zaya_runtime_tests = root / "tests/test_zaya_runtime.py"
    zaya_runtime_test_text = (
        zaya_runtime_tests.read_text(encoding="utf-8", errors="replace")
        if zaya_runtime_tests.exists()
        else ""
    )
    runtime_explicitly_unwired = all(
        marker in adapter_text
        for marker in (
            "MiMo-V2.5 JANG_2L vision input is not wired",
            "key.startswith(\"model.mtp.\")",
            "class VisionModel",
            "class AudioModel",
        )
    )
    missing_multimodal_module = not any(
        path.exists()
        for path in (
            root
            / ".venv/lib/python3.13/site-packages/jang_tools/mimo_v2/mimo_v2_multimodal.py",
            Path("/Users/eric/jang/jang-tools/jang_tools/mimo_v2/mimo_v2_multimodal.py"),
        )
    )
    vision_config_parser_present = all(
        marker in adapter_text
        for marker in (
            "class VisionConfig(_MiMoV2MediaConfig)",
            "VisionConfig.from_dict(params.get(\"vision_config\") or {})",
        )
    )
    audio_config_parser_present = all(
        marker in adapter_text
        for marker in (
            "class AudioConfig(_MiMoV2MediaConfig)",
            "AudioConfig.from_dict(params.get(\"audio_config\") or {})",
        )
    )
    processor_config_parser_present = all(
        marker in adapter_text
        for marker in (
            "processor_config=params.get(\"processor_config\") or {}",
            "self.processor_config = dict(processor_config or {})",
            "self.media_token_ids = {",
        )
    )
    visual_load_weights_binding_present = all(
        marker in adapter_text
        for marker in (
            "self._mimo_v2_media_weights: dict[str, Any] = {}",
            "def _bind_mimo_v2_media_weight(self, key, value):",
            "key.startswith(\"visual.\")",
            "self._mimo_v2_media_weights[key] = value",
        )
    )
    audio_load_weights_binding_present = all(
        marker in adapter_text
        for marker in (
            "def _bind_mimo_v2_media_weight(self, key, value):",
            "key.startswith(\"audio_encoder.\")",
            "key.startswith(\"speech_embeddings.\")",
            "self._mimo_v2_media_weight_counts[group] += 1",
        )
    )
    mixed_inputs_embeds_present = all(
        marker in adapter_text
        for marker in (
            "def _mimo_v2_replace_modal_embeddings(",
            "image_embeds is not None",
            "video_embeds is not None",
            "audio_embeds is not None",
            "token count",
        )
    )
    vision_patch_embed_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoVisionPatchEmbed",
            "def embed_patches(self, pixel_values):",
            "self.patch_embed = MiMoVisionPatchEmbed(",
            "flat_patch_width",
        )
    )
    vision_merger_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoVisionPatchMerger",
            "def merge_patches(self, hidden_states):",
            "self.merger = MiMoVisionPatchMerger(",
            "self.mlp = [",
            "nn.Linear(self.hidden_size, dim)",
        )
    )
    vision_attention_blocks_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoVisionAttention",
            "class MiMoVisionBlock",
            "class MiMoVisionSwiGLUMLP",
            "def run_blocks(self, hidden_states, grid_thw=None):",
            "mx.fast.scaled_dot_product_attention",
            "self.blocks = [",
        )
    )
    vision_grid_forward_present = all(
        marker in adapter_text
        for marker in (
            "def rot_pos_emb(self, grid_thw):",
            "def get_window_index_1d(self, grid_thw, col: bool = True):",
            "reverse_window_index_1d_col = mx.argsort(window_index_1d_col)",
            "position_embeddings=position_embeddings",
            "x = self.run_blocks(x, grid_thw=grid_thw)",
            "return self.merge_patches(x)",
        )
    )
    model_vision_bridge_present = all(
        marker in adapter_text
        for marker in (
            "self.visual = (",
            "if self.visual is None:",
            "image_embeds = self.visual(",
            "video_embeds = self.visual(",
            "image_grid_thw=image_grid_thw",
            "video_grid_thw=video_grid_thw",
        )
    )
    audio_projection_bridge_present = all(
        marker in adapter_text
        for marker in (
            "class AudioProjection",
            "class AudioModel(nn.Module):",
            "self.audio_encoder = (",
            "def project_local_audio(self, audio_hidden):",
            "audio_embeds = self.audio_encoder(audio_embeds=audio_arr)",
        )
    )
    audio_code_local_transformer_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoAudioLocalTransformer",
            "class MiMoAudioLocalAttention",
            "self.input_local_transformer = MiMoAudioLocalTransformer",
            "def process_audio_codes(self, audio_codes, speech_embeddings):",
            "speech_embeddings=self.speech_embeddings",
            "_pad_and_group_mimo_v2_audio_codes",
        )
    )
    audio_tokenizer_runtime_execution_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoAudioTokenizer",
            "class MiMoAudioTokenizerEncoder",
            "def encode_audio_to_codes",
            "def load_mimo_audio_tokenizer_from_bundle",
            "audio_tokenizer/model.safetensors",
            "num_quantizers",
            "codebook_size",
            "n_mels",
            "return_codes_only",
        )
    ) and (
        "test_mimo_v2_audio_tokenizer_executes_mel_to_audio_codes"
        in (
            (root / "tests/test_mimo_v2_media_runtime.py").read_text(
                encoding="utf-8",
                errors="replace",
            )
            if (root / "tests/test_mimo_v2_media_runtime.py").exists()
            else ""
        )
    )
    audio_tokenizer_rvq_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoAudioEuclideanCodebook",
            "class MiMoAudioResidualVectorQuantizer",
            "def encode(self, hidden_states",
            "def decode(self, codes",
            "mx.argmax(dist, axis=-1)",
        )
    ) and (
        "test_mimo_v2_audio_residual_vector_quantizer_matches_nearest_codebook"
        in scheduler_cache_test_text
        or "test_mimo_v2_audio_residual_vector_quantizer_matches_nearest_codebook"
        in (
            (root / "tests/test_mimo_v2_media_runtime.py").read_text(
                encoding="utf-8",
                errors="replace",
            )
            if (root / "tests/test_mimo_v2_media_runtime.py").exists()
            else ""
        )
    )
    audio_tokenizer_rvq_weight_loader_present = all(
        marker in adapter_text
        for marker in (
            "def load_mimo_audio_rvq_from_bundle",
            "audio_tokenizer/model.safetensors",
            "encoder.quantizer.vq.layers",
            "MiMoAudioResidualVectorQuantizer(codebooks",
        )
    ) and (
        "test_mimo_v2_audio_rvq_loader_reads_bundle_codebooks"
        in (
            (root / "tests/test_mimo_v2_media_runtime.py").read_text(
                encoding="utf-8",
                errors="replace",
            )
            if (root / "tests/test_mimo_v2_media_runtime.py").exists()
            else ""
        )
    )
    audio_tokenizer_config_segmenter_present = all(
        marker in adapter_text
        for marker in (
            "class MiMoAudioTokenizerConfig",
            "def get_output_length(self, mel_len",
            "def get_code_length(self, mel_len",
            "def plan_mimo_audio_mel_segments",
            "segments_per_mel",
            "code_lengths",
        )
    ) and (
        "test_mimo_v2_audio_tokenizer_config_and_mel_segment_plan"
        in (
            (root / "tests/test_mimo_v2_media_runtime.py").read_text(
                encoding="utf-8",
                errors="replace",
            )
            if (root / "tests/test_mimo_v2_media_runtime.py").exists()
            else ""
        )
    )
    media_weight_assignment_present = all(
        marker in adapter_text
        for marker in (
            "def _apply_mimo_v2_media_weights(self):",
            "_mimo_v2_assign_weight(self, key, value)",
            "MiMo-V2 load assigned %d preserved media tensors",
        )
    )
    media_aware_prefix_l2_cache_present = all(
        marker in batch_generator_text
        for marker in (
            "def _mllm_media_cache_extra_keys(request: Any)",
            "audio_codes",
            "audio_embeds",
            "audio_features",
            "cache_extra_keys=_cache_extra_keys",
            "cache_extra_keys=_ssm_extra_keys",
        )
    ) and all(
        marker in zaya_runtime_test_text
        for marker in (
            "test_mllm_media_cache_key_includes_audio_codes",
            "test_mllm_audio_payloads_are_media_cache_context",
        )
    )
    audio_payload_forwarding_present = all(
        marker in batch_generator_text
        for marker in (
            'kwargs["audio_codes"] = request.audio_codes',
            'kwargs["audio_embeds"] = request.audio_embeds',
            'kwargs["audio_embeds"] = request.audio_features',
            "has_media_payload = has_images or has_audio_payload",
            "not has_media_payload",
        )
    ) and (
        "test_mllm_audio_payload_prefill_uses_model_wrapper_not_text_fast_path"
        in scheduler_cache_test_text
    )
    audio_processor_output_bridge_present = all(
        marker in batch_generator_text
        for marker in (
            'request.extra_kwargs.pop("audio_codes", None)',
            'request.extra_kwargs.pop("audio_embeds", None)',
            'request.extra_kwargs.pop("audio_features", None)',
            "request.audio_codes = _ensure_mx_array(",
            "request.audio_embeds = _ensure_mx_array(",
            "request.audio_features = _ensure_mx_array(",
        )
    ) and (
        "test_mllm_processor_audio_outputs_are_promoted_to_request_fields"
        in scheduler_cache_test_text
    )
    raw_audio_request_ingestion_present = (
        all(
            marker in batch_generator_text
            for marker in (
                "audio: Optional[List",
                "process_audio_input",
                "not request.images and not request.videos and not request.audio",
                'kwargs["audio"] = audio',
                "skip_audios_alias",
            )
        )
        and all(
            marker in scheduler_text
            for marker in (
                "audio: Optional[List[Any]] = None",
                "audio=request.audio",
                "audio=audio",
                "not images and not videos and not audio",
            )
        )
        and all(
            marker in batched_engine_text
            for marker in (
                "def _extract_audio_content",
                "extracted_audio = self._extract_audio_content(messages)",
                "all_audio = (audio or []) + extracted_audio",
                "audio=all_audio if all_audio else None",
            )
        )
        and "test_mllm_processor_direct_forwards_raw_audio_to_processor"
        in scheduler_cache_test_text
        and "test_mllm_processor_direct_omits_invalid_audios_alias_for_mimo_v2_processor"
        in scheduler_cache_test_text
        and "test_mllm_scheduler_and_batched_engine_route_raw_audio_requests"
        in scheduler_cache_test_text
    )
    media_weights_preserved = bool(visual_count > 0 and audio_count > 0)
    local_mimo_v2_multimodal_module_present = all(
        (
            vision_patch_embed_present,
            vision_merger_present,
            vision_attention_blocks_present,
            vision_grid_forward_present,
            model_vision_bridge_present,
            audio_projection_bridge_present,
            audio_code_local_transformer_present,
            audio_tokenizer_runtime_execution_present,
            media_weight_assignment_present,
            mixed_inputs_embeds_present,
        )
    )
    missing_multimodal_module = bool(
        missing_multimodal_module and not local_mimo_v2_multimodal_module_present
    )
    metadata_overadvertises = bool(
        config_modalities != ["text"]
        or config_preserved != ["vision", "audio"]
        or config_unwired != ["vision", "audio"]
        or config_multimodal_status != "weights_preserved_text_runtime"
        or runtime_multimodal_mode != "weights_preserved_text_runtime"
    )
    source_media_components_present = bool(
        local_mimo_v2_multimodal_module_present
        and media_weight_assignment_present
        and mixed_inputs_embeds_present
        and model_vision_bridge_present
        and audio_projection_bridge_present
        and audio_payload_forwarding_present
        and audio_processor_output_bridge_present
        and raw_audio_request_ingestion_present
    )
    runtime_gap = bool(
        runtime_explicitly_unwired
        or missing_multimodal_module
        or not audio_tokenizer_runtime_execution_present
        or not source_media_components_present
    )
    runtime_media_supported = bool(runtime_caps.get("runtime_media_supported"))
    runtime_media_wired = bool(
        source_media_components_present and not runtime_gap and runtime_media_supported
    )
    runtime_capabilities_text_only = bool(
        runtime_caps.get("safe_text_only_runtime_capabilities")
    )
    status_open = bool(
        runtime_gap
        or metadata_overadvertises
        or not runtime_media_wired
    )
    if runtime_gap and metadata_overadvertises:
        classification = "runtime_implementation_gap_with_model_metadata_overadvertising"
    elif runtime_gap:
        classification = "runtime_implementation_gap"
    elif metadata_overadvertises:
        classification = "model_metadata_overadvertising"
    elif runtime_capabilities_text_only:
        classification = "media_components_present_runtime_capabilities_text_only"
    elif not runtime_media_supported:
        classification = "media_components_present_runtime_capabilities_unwired"
    else:
        classification = "media_runtime_and_metadata_clear"
    return {
        "status": "open" if status_open else "pass",
        "classification": classification,
        "config_error": config_error,
        "config_modalities": config_modalities,
        "config_preserved_modalities": config_preserved,
        "config_unwired_modalities": config_unwired,
        "config_multimodal_status": config_multimodal_status,
        "runtime_multimodal_mode": runtime_multimodal_mode,
        "config_text_only_contract": bool(config_text_only_contract),
        "metadata_overadvertises_runtime_media": metadata_overadvertises,
        "visual_tensor_count": visual_count,
        "audio_tensor_count": audio_count,
        "has_preprocessor_config": has_processor,
        "has_audio_tokenizer_dir": has_audio_tokenizer,
        "has_audio_tokenizer_config": has_audio_tokenizer_config,
        "has_audio_tokenizer_weights": has_audio_tokenizer_weights,
        "media_weights_preserved": media_weights_preserved,
        "runtime_capabilities": runtime_caps,
        "smoke_runtime_capabilities": smoke_caps,
        "runtime_capabilities_text_only_safe": bool(
            runtime_caps.get("safe_text_only_runtime_capabilities")
        ),
        "runtime_capabilities_safe": bool(
            runtime_caps.get("runtime_capabilities_safe")
        ),
        "runtime_capabilities_media_supported": bool(
            runtime_media_supported
        ),
        "source_media_components_present": bool(source_media_components_present),
        "runtime_explicitly_unwired": bool(runtime_explicitly_unwired),
        "missing_mimo_v2_multimodal_module": bool(missing_multimodal_module),
        "local_mimo_v2_multimodal_module": bool(
            local_mimo_v2_multimodal_module_present
        ),
        "config_parser_components": {
            "vision_config": bool(vision_config_parser_present),
            "audio_config": bool(audio_config_parser_present),
            "processor_config": bool(processor_config_parser_present),
        },
        "load_weights_bindings": {
            "visual": bool(visual_load_weights_binding_present),
            "audio": bool(audio_load_weights_binding_present),
        },
        "mixed_inputs_embeds_splice": bool(mixed_inputs_embeds_present),
        "vision_patch_embed_module": bool(vision_patch_embed_present),
        "vision_merger_module": bool(vision_merger_present),
        "vision_attention_block_modules": bool(vision_attention_blocks_present),
        "vision_grid_forward": bool(vision_grid_forward_present),
        "model_vision_bridge": bool(model_vision_bridge_present),
        "audio_projection_bridge": bool(audio_projection_bridge_present),
        "audio_code_local_transformer": bool(audio_code_local_transformer_present),
        "audio_tokenizer_rvq_codebook": bool(audio_tokenizer_rvq_present),
        "audio_tokenizer_rvq_weight_loader": bool(
            audio_tokenizer_rvq_weight_loader_present
        ),
        "audio_tokenizer_config_segmenter": bool(
            audio_tokenizer_config_segmenter_present
        ),
        "audio_tokenizer_model_execution": bool(
            audio_tokenizer_runtime_execution_present
        ),
        "media_weight_assignment": bool(media_weight_assignment_present),
        "media_aware_prefix_l2_cache": bool(media_aware_prefix_l2_cache_present),
        "audio_payload_forwarding": bool(audio_payload_forwarding_present),
        "audio_processor_output_bridge": bool(audio_processor_output_bridge_present),
        "raw_audio_request_ingestion": bool(raw_audio_request_ingestion_present),
        "runtime_media_wired": bool(runtime_media_wired),
        "missing_runtime_components": (
            [
                component
                for component, present in (
                    ("VisionConfig parser", vision_config_parser_present),
                    ("AudioConfig parser", audio_config_parser_present),
                    ("Processor/media token config parser", processor_config_parser_present),
                    ("visual.* load_weights binding", visual_load_weights_binding_present),
                    (
                        "audio_encoder.* and speech_embeddings.* load_weights binding",
                        audio_load_weights_binding_present,
                    ),
                    (
                        "mixed media/text inputs_embeds construction",
                        mixed_inputs_embeds_present,
                    ),
                    (
                        "visual patch embedding module",
                        vision_patch_embed_present,
                    ),
                    ("vision merger to 4096 hidden size", vision_merger_present),
                    (
                        "vision qkv attention and MLP block modules",
                        vision_attention_blocks_present,
                    ),
                    (
                        "grid-aware rotary/window index reorder and end-to-end ViT forward",
                        vision_grid_forward_present,
                    ),
                    (
                        "model-level image/video request bridge to MiMo vision tower",
                        model_vision_bridge_present,
                    ),
                    (
                        "audio projection bridge to text hidden size",
                        audio_projection_bridge_present,
                    ),
                    (
                        "audio code local transformer forward",
                        audio_code_local_transformer_present,
                    ),
                    (
                        "audio tokenizer RVQ codebook weight loader",
                        audio_tokenizer_rvq_weight_loader_present,
                    ),
                    (
                        "audio tokenizer config and mel segment planner",
                        audio_tokenizer_config_segmenter_present,
                    ),
                    (
                        "audio tokenizer model execution bridge (waveform/mel -> 20-channel audio_codes)",
                        audio_tokenizer_runtime_execution_present,
                    ),
                    (
                        "media weight assignment to runtime modules",
                        media_weight_assignment_present,
                    ),
                    (
                        "media-aware prefix/L2 cache proof",
                        media_aware_prefix_l2_cache_present,
                    ),
                    (
                        "audio payload forwarding through MLLM wrapper",
                        audio_payload_forwarding_present,
                    ),
                    (
                        "processor-produced audio tensor field bridge",
                        audio_processor_output_bridge_present,
                    ),
                    (
                        "raw audio request bridge from API/UI to processor",
                        raw_audio_request_ingestion_present,
                    ),
                )
                if not present
            ]
            if runtime_gap
            else []
        ),
        "required_model_metadata_if_runtime_unwired": {
            "capabilities.modalities": ["text"],
            "capabilities.preserved_modalities": ["vision", "audio"],
            "capabilities.unwired_modalities": ["vision", "audio"],
            "capabilities.multimodal_status": "weights_preserved_text_runtime",
            "runtime.multimodal_mode": "weights_preserved_text_runtime",
        },
    }


def _mimo_jangtq2_media_route_evidence(*artifacts: dict[str, Any]) -> dict[str, Any]:
    source = artifacts[0] if len(artifacts) > 0 else {}
    dev_video = artifacts[1] if len(artifacts) > 1 else {}
    video_cache = artifacts[2] if len(artifacts) > 2 else {}
    source_data = source.get("data") if isinstance(source.get("data"), dict) else {}
    dev_video_data = (
        dev_video.get("data") if isinstance(dev_video.get("data"), dict) else {}
    )
    video_cache_data = (
        video_cache.get("data") if isinstance(video_cache.get("data"), dict) else {}
    )
    source_proven = (
        source_data.get("proven") if isinstance(source_data.get("proven"), dict) else {}
    )
    dev_checks = (
        dev_video_data.get("checks")
        if isinstance(dev_video_data.get("checks"), dict)
        else {}
    )
    cache_proven = (
        video_cache_data.get("proven")
        if isinstance(video_cache_data.get("proven"), dict)
        else {}
    )
    transport_proven = bool(
        source_proven.get("source_server_loads_as_mllm")
        and source_proven.get("media_weights_bound")
        and source_proven.get("video_request_reaches_runtime")
        and source_proven.get("audio_request_reaches_runtime")
        and source_proven.get("video_http_200")
        and source_proven.get("audio_http_200")
    )
    dev_video_transport_proven = bool(
        dev_video_data.get("transport_status") == "pass"
        and dev_checks.get("real_electron_dev_build")
        and dev_checks.get("real_model_loaded")
        and dev_checks.get("mllm_flag_used")
        and dev_checks.get("server_media_diag_video_url")
        and dev_checks.get("http_200")
        and dev_checks.get("block_l2_write")
    )
    video_cache_proven = bool(
        cache_proven.get("real_mimo_jangtq2_model_loaded")
        and cache_proven.get("video_request_reaches_runtime")
        and cache_proven.get("repeated_video_request_http_200")
        and cache_proven.get("video_pixel_cache_hit_after_first_request")
        and cache_proven.get("video_cache_contract_preserves_video_tensor_fields_in_source")
    )
    semantic_green = bool(
        dev_video_data.get("semantic_status") == "pass"
        and dev_checks.get("video_semantic_red_fixture")
    )
    return {
        "exists": bool(source.get("exists") or dev_video.get("exists") or video_cache.get("exists")),
        "source_artifact": source.get("artifact"),
        "dev_app_video_artifact": dev_video.get("artifact"),
        "video_cache_artifact": video_cache.get("artifact"),
        "status": (
            "open"
            if transport_proven or dev_video_transport_proven or video_cache_proven
            else "missing"
        ),
        "transport_proven": bool(
            transport_proven or dev_video_transport_proven or video_cache_proven
        ),
        "source_mllm_media_transport_proven": bool(transport_proven),
        "dev_app_video_transport_proven": bool(dev_video_transport_proven),
        "video_cache_proven": bool(video_cache_proven),
        "semantic_quality_proven": bool(semantic_green),
        "semantic_status": str(dev_video_data.get("semantic_status") or "missing"),
        "classification": (
            "forced_mllm_media_transport_and_cache_live_semantics_red"
            if (transport_proven or dev_video_transport_proven or video_cache_proven)
            and not semantic_green
            else "forced_mllm_media_transport_semantics_green"
            if semantic_green
            else "missing_mimo_jangtq2_media_route_proof"
        ),
        "boundary": (
            "MiMo JANGTQ2 has current source/dev-app evidence that media weights "
            "bind and video/audio requests reach the MLLM runtime when launched "
            "as MLLM. This clears the stale 'implementation missing' wording only. "
            "It does not clear default metadata-gated text-only capability, image "
            "color semantics, video semantic quality, audio transcript semantics, "
            "fresh-process L2 restore for media, installed-app parity, or release readiness."
        ),
        "must_not_claim": [
            "Do not advertise preserved-text-runtime MiMo bundles as media-capable by default.",
            "Do not claim MiMo visual or audio semantic quality is release-green.",
            "Do not use this route proof to clear MiMo literal/tool/JSON exactness.",
        ],
    }


def _text_route_evidence(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("live_rows")
    rows = rows if isinstance(rows, dict) else {}
    long_row = rows.get("long_prompt_first_request")
    long_row = long_row if isinstance(long_row, dict) else {}
    cache1 = rows.get("cache_repeat_1")
    cache1 = cache1 if isinstance(cache1, dict) else {}
    cache2 = rows.get("cache_repeat_2")
    cache2 = cache2 if isinstance(cache2, dict) else {}
    tool_row = rows.get("tool_required")
    tool_row = tool_row if isinstance(tool_row, dict) else {}
    blockers = data.get("remaining_blockers")
    blockers = blockers if isinstance(blockers, list) else []
    return {
        "exists": True,
        "status": data.get("status"),
        "fix_status": data.get("fix_status"),
        "long_prompt_recall_pass": bool(long_row.get("sentinel_recall_pass")),
        "long_prompt_oom_cleared": bool(long_row.get("oom_cleared")),
        "cache_visible_pass": bool(cache1.get("empty_visible_cleared"))
        and bool(cache2.get("empty_visible_cleared")),
        "cache_release_exact_pass": bool(cache1.get("exact_expected_ack_cache_742"))
        and bool(cache2.get("exact_expected_ack_cache_742")),
        "tool_call_pass": bool(tool_row.get("tool_call_pass")),
        "speed_blocked": "decode_speed_below_release_target" in blockers,
        "cache_l2_not_reproved": "prefix_paged_l2_cache_hit_not_reproved_on_continuous_batching_route"
        in blockers,
        "source_vs_quant_missing": "source_vs_quant_first_divergence_missing" in blockers,
        "media_unwired": "mimo_vl_audio_video_unwired" in blockers,
        "remaining_blockers": blockers,
        "release_boundary": data.get("release_boundary"),
    }


def _cb_oneshot_prefill_evidence(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and row.get("name") is not None
    }

    def _response(name: str) -> dict[str, Any]:
        row = by_name.get(name) or {}
        response = row.get("response")
        return response if isinstance(response, dict) else {}

    def _health(name: str) -> dict[str, Any]:
        row = by_name.get(name) or {}
        health = row.get("health")
        payload = health.get("json") if isinstance(health, dict) else {}
        return payload if isinstance(payload, dict) else {}

    cache1 = _response("cache_repeat_1")
    cache2 = _response("cache_repeat_2")
    tool = _response("tool_call")
    long_prompt = _response("long_prompt")
    cache2_health = _health("cache_repeat_2_health")
    tool_health = _health("tool_health")

    scheduler = cache2_health.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    cache = cache2_health.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    totals = cache.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    bg = scheduler.get("batch_generator")
    bg = bg if isinstance(bg, dict) else {}

    tool_scheduler = tool_health.get("scheduler")
    tool_scheduler = tool_scheduler if isinstance(tool_scheduler, dict) else {}
    tool_bg = tool_scheduler.get("batch_generator")
    tool_bg = tool_bg if isinstance(tool_bg, dict) else {}

    cache_exact_pass = (
        cache1.get("ok") is True
        and cache2.get("ok") is True
        and cache1.get("content") == "ACK-CACHE-742"
        and cache2.get("content") == "ACK-CACHE-742"
    )
    cache_reproved = (
        isinstance(scheduler.get("cache_hit_tokens"), (int, float))
        and scheduler.get("cache_hit_tokens") >= 37
        and isinstance(totals.get("l2_block_tokens_on_disk"), (int, float))
        and totals.get("l2_block_tokens_on_disk") >= 37
    )
    generation_tps = tool_bg.get("generation_tps", bg.get("generation_tps"))
    return {
        "exists": True,
        "cache_exact_pass": bool(cache_exact_pass),
        "prefix_paged_l2_cache_reproved": bool(cache_reproved),
        "tool_protocol_blocked": not bool(tool.get("tool_calls")),
        "tool_content_head": str(tool.get("content") or "")[:200],
        "long_prompt_crashed": long_prompt.get("ok") is False
        and "RemoteDisconnected" in str(long_prompt.get("error") or ""),
        "long_prompt_error": long_prompt.get("error"),
        "speed_blocked": not (
            isinstance(generation_tps, (int, float)) and generation_tps >= 40.0
        ),
        "generation_tps": generation_tps,
        "cache1_elapsed_s": cache1.get("elapsed"),
        "cache2_elapsed_s": cache2.get("elapsed"),
    }


def _cb_native_thinking_off_evidence(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    rows = data.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and row.get("name") is not None
    }

    def _row_speed(name: str) -> float | None:
        row = by_name.get(name) or {}
        usage = row.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        tokens = usage.get("completion_tokens")
        elapsed = row.get("elapsed_s")
        if not isinstance(tokens, (int, float)) or not isinstance(elapsed, (int, float)):
            return None
        if elapsed <= 0:
            return None
        return float(tokens) / float(elapsed)

    speeds = [
        value
        for value in (
            _row_speed("exact_repeat_1"),
            _row_speed("exact_repeat_2"),
        )
        if value is not None
    ]
    speed_max = max(speeds) if speeds else None
    memory_pressure = any(
        isinstance(row, dict)
        and row.get("ok") is False
        and (
            "503" in str(row.get("error") or "")
            or "Service Unavailable" in str(row.get("error") or "")
        )
        for row in rows
    ) or "working-set pressure reject" in str(data.get("log_tail") or "")
    native_cache = summary.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    return {
        "exists": True,
        "status": data.get("status"),
        "cache_exact_pass": bool(
            summary.get("exact_repeat_1") and summary.get("exact_repeat_2")
        ),
        "no_think_tags": bool(summary.get("no_think_tags")),
        "prefix_paged_l2_cache_reproved": (
            isinstance(summary.get("cache_hit_tokens"), (int, float))
            and summary.get("cache_hit_tokens") > 0
            and isinstance(summary.get("l2_tokens_on_disk"), (int, float))
            and summary.get("l2_tokens_on_disk") > 0
        ),
        "mixed_swa_native_cache": native_cache.get("cache_type") == "mixed_swa_kv",
        "storage_quantization_q8": (
            isinstance(native_cache.get("storage_quantization"), dict)
            and native_cache["storage_quantization"].get("enabled") is True
            and native_cache["storage_quantization"].get("bits") == 8
        ),
        "memory_pressure_blocked": bool(memory_pressure),
        "speed_blocked": not (
            isinstance(speed_max, (int, float)) and speed_max >= 40.0
        ),
        "speed_max_tps": speed_max,
        "texts": summary.get("texts"),
        "finish_reasons": summary.get("finish_reasons"),
        "cache_hit_tokens": summary.get("cache_hit_tokens"),
        "l2_tokens_on_disk": summary.get("l2_tokens_on_disk"),
    }


def _latest_decode_speed_evidence(data: dict[str, Any]) -> dict[str, Any]:
    speed_boundary = data.get("speed_boundary")
    if isinstance(speed_boundary, dict):
        server_tps = speed_boundary.get("current_source_server_tps")
        wall_tps = speed_boundary.get("current_source_wall_tps")
        current_artifact = speed_boundary.get("current_source_long_decode_artifact")
        speed_pass = (
            isinstance(server_tps, (int, float)) and server_tps >= 40.0
        ) or (
            isinstance(wall_tps, (int, float)) and wall_tps >= 40.0
        )
        return {
            "exists": True,
            "status": data.get("status"),
            "speed_blocked": not speed_pass,
            "classification": "current_text_runtime_release_path_speed_boundary",
            "bundle_decode_tps": server_tps,
            "wall_decode_tps": wall_tps,
            "current_source_long_decode_artifact": current_artifact,
            "stale_forced_mllm_speed_artifact": speed_boundary.get(
                "accepted_prior_speed_artifact"
            ),
            "stale_forced_mllm_generation_tps": speed_boundary.get(
                "accepted_prior_generation_tps"
            ),
            "notes": speed_boundary.get("status"),
        }

    results = data.get("results")
    result = results[0] if isinstance(results, list) and results else data
    if not isinstance(result, dict):
        return {
            "exists": True,
            "speed_blocked": True,
            "classification": "invalid_decode_speed_artifact",
        }

    bundle = result.get("bundle_sampling")
    bundle = bundle if isinstance(bundle, dict) else {}
    greedy = result.get("greedy_topk0")
    greedy = greedy if isinstance(greedy, dict) else {}
    coherency = result.get("coherency")
    coherency = coherency if isinstance(coherency, dict) else {}
    pp_rows = result.get("pp_rows")
    pp_rows = pp_rows if isinstance(pp_rows, list) else []
    health = result.get("health_after")
    health = health if isinstance(health, dict) else {}
    native_cache = health.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    generic_tq = native_cache.get("generic_turboquant_kv")
    generic_tq = generic_tq if isinstance(generic_tq, dict) else {}

    bundle_tps = bundle.get("decode_tps_wall")
    greedy_tps = greedy.get("decode_tps_wall")
    pp_speeds = [
        row.get("pp_wall_tok_s")
        for row in pp_rows
        if isinstance(row, dict) and isinstance(row.get("pp_wall_tok_s"), (int, float))
    ]
    coherency_exact = (
        coherency.get("content_head") == "READY\n17+28=45\nCERULEAN"
        and coherency.get("loopish") is False
    )
    speed_pass = (
        isinstance(bundle_tps, (int, float))
        and bundle_tps >= 40.0
        and (
            not isinstance(greedy_tps, (int, float))
            or greedy_tps >= 40.0
        )
        and coherency_exact
    )
    log_path = result.get("log_path")
    log_trace = _decode_speed_log_trace(Path(str(log_path))) if log_path else {}
    return {
        "exists": True,
        "status": result.get("status"),
        "speed_blocked": not speed_pass,
        "bundle_decode_tps": bundle_tps,
        "greedy_decode_tps": greedy_tps,
        "pp_wall_tok_s": pp_speeds,
        "coherency_exact": coherency_exact,
        "native_cache_type": native_cache.get("cache_type"),
        "generic_turboquant_kv_enabled": generic_tq.get("enabled"),
        "notes": result.get("notes"),
        "log_path": log_path,
        **log_trace,
    }


def _cb_metal_baseline_evidence(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    rows = data.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and row.get("name") is not None
    }
    probe = by_name.get("first_token_system_probe") or {}
    all_http_ok = all(
        isinstance(row, dict) and row.get("status") == 200
        for row in rows
    )
    memory_pressure_cleared = bool(
        data.get("status") == "pass"
        and summary.get("memory_pressure_cleared") is True
        and all_http_ok
        and probe.get("status") == 200
        and "503" not in str(probe.get("error") or "")
        and "Service Unavailable" not in str(probe.get("error") or "")
    )
    native_cache = summary.get("native_cache")
    native_cache = native_cache if isinstance(native_cache, dict) else {}
    storage_quantization = native_cache.get("storage_quantization")
    storage_quantization = (
        storage_quantization if isinstance(storage_quantization, dict) else {}
    )
    texts = summary.get("texts")
    texts = texts if isinstance(texts, dict) else {}
    repeat_1 = texts.get("exact_repeat_1")
    repeat_2 = texts.get("exact_repeat_2")
    return {
        "exists": True,
        "status": data.get("status"),
        "memory_pressure_cleared": memory_pressure_cleared,
        "first_token_system_probe_http_ok": probe.get("status") == 200,
        "first_token_system_probe_text": probe.get("text"),
        "cache_hit_tokens": summary.get("cache_hit_tokens"),
        "l2_tokens_on_disk": summary.get("l2_tokens_on_disk"),
        "native_cache": native_cache,
        "native_cache_type": native_cache.get("cache_type"),
        "storage_quantization_disabled": storage_quantization.get("enabled") is False,
        "cache_repeat_stable": bool(repeat_1 and repeat_1 == repeat_2),
        "cache_repeat_exact": bool(
            summary.get("exact_repeat_1") and summary.get("exact_repeat_2")
        ),
        "texts": texts,
        "boundary": (
            "This live proof clears the post-load Metal working-set 503 path only. "
            "It also proves lossless native mixed-SWA paged-cache repeat stability, "
            "but it does not clear MiMo literal exactness or tool argument exactness."
        ),
    }


def _decode_speed_log_trace(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "log_trace_present": False,
            "decode_bottleneck_classification": "missing_decode_speed_log",
        }

    fastpath_calls: list[int] = []
    compiled_shapes: list[int] = []
    decode_total_ms: list[float] = []
    decode_step_ms: list[float] = []
    decode_async_ms: list[float] = []
    decode_materialize_ms: list[float] = []
    avg_model_ms: list[float] = []
    avg_sample_ms: list[float] = []
    tight_memory_drains = 0

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Tight-memory MLLM allocator drain" in line:
            tight_memory_drains += 1
        if (
            "MiMo-V2 affine SwitchGLU decode fast path active" in line
            or "MiMo-V2 TurboQuant SwitchGLU decode fast path active" in line
        ):
            calls_match = re.search(r"\bcalls=(\d+)", line)
            shapes_match = re.search(r"\bcompiled_shapes=(\d+)", line)
            if calls_match:
                fastpath_calls.append(int(calls_match.group(1)))
            if shapes_match:
                compiled_shapes.append(int(shapes_match.group(1)))
        if "VMLINUX_DECODE_TRACE mllm" in line:
            for key, target in (
                ("avg_model_ms", avg_model_ms),
                ("avg_sample_ms", avg_sample_ms),
            ):
                match = re.search(rf"\b{key}=([0-9.]+)", line)
                if match:
                    target.append(round(float(match.group(1)), 3))
        if "VMLINUX_DECODE_TRACE_NEXT" in line:
            for key, target in (
                ("last_total_ms", decode_total_ms),
                ("last_step_ms", decode_step_ms),
                ("last_async_ms", decode_async_ms),
                ("last_materialize_ms", decode_materialize_ms),
            ):
                match = re.search(rf"\b{key}=([0-9.]+)", line)
                if match:
                    target.append(round(float(match.group(1)), 3))

    max_total_ms = max(decode_total_ms) if decode_total_ms else None
    max_async_ms = max(decode_async_ms) if decode_async_ms else None
    max_avg_model_ms = max(avg_model_ms) if avg_model_ms else None
    fastpath_active = bool(fastpath_calls)
    async_wait_dominates = bool(
        max_total_ms is not None
        and max_async_ms is not None
        and max_total_ms > 0
        and (max_async_ms / max_total_ms) >= 0.5
    )
    if fastpath_active and async_wait_dominates:
        classification = "switchglu_fastpath_active_but_metal_async_wait_dominates"
    elif async_wait_dominates:
        classification = "metal_async_wait_dominates"
    elif fastpath_active:
        classification = "switchglu_fastpath_active_without_async_trace_blocker"
    else:
        classification = "switchglu_fastpath_not_observed"

    return {
        "log_trace_present": True,
        "switchglu_fastpath_active": fastpath_active,
        "switchglu_fastpath_call_markers": fastpath_calls,
        "switchglu_fastpath_max_calls": max(fastpath_calls) if fastpath_calls else None,
        "switchglu_fastpath_compiled_shapes": compiled_shapes,
        "switchglu_fastpath_max_compiled_shapes": (
            max(compiled_shapes) if compiled_shapes else None
        ),
        "decode_next_last_total_ms": decode_total_ms,
        "decode_next_last_step_ms": decode_step_ms,
        "decode_next_last_async_ms": decode_async_ms,
        "decode_next_last_materialize_ms": decode_materialize_ms,
        "avg_model_ms": avg_model_ms,
        "avg_sample_ms": avg_sample_ms,
        "max_decode_next_last_total_ms": max_total_ms,
        "max_decode_next_last_async_ms": max_async_ms,
        "max_avg_model_ms": max_avg_model_ms,
        "async_decode_wait_dominates": async_wait_dominates,
        "tight_memory_allocator_drains": tight_memory_drains,
        "decode_bottleneck_classification": classification,
    }


def build_audit(root: Path, model_path: Path, manifest: Path) -> dict[str, Any]:
    model_parent = model_path.parent
    manifest_check = _verify_manifest(model_parent, manifest)
    bookend_quantization = _bookend_quantization_audit(model_path)
    stale_state = [
        {
            "path": str(path),
            "exists": (root / path).exists() if not path.is_absolute() else path.exists(),
        }
        for path in STALE_TARGETS
    ]

    structural = _artifact(root / STRUCTURAL_ARTIFACT)
    text_cache = _artifact(root / TEXT_CACHE_ARTIFACT)
    switchglu = _artifact(root / SWITCHGLU_ARTIFACT)
    length_sweep = _artifact(root / LENGTH_SWEEP_ARTIFACT)
    tool_failure = _artifact(root / TOOL_FAILURE_ARTIFACT)
    all_local_smoke = _artifact(root / ALL_LOCAL_SMOKE_ARTIFACT)
    jang2l_all_local_smoke = _artifact(root / JANG2L_ALL_LOCAL_SMOKE_ARTIFACT)
    kvnone_noprefix_smoke = _artifact(root / KVNONE_NOPREFIX_SMOKE_ARTIFACT)
    cache_vs_nocache = _artifact(root / CACHE_VS_NOCACHE_ARTIFACT)
    sink_mode_length = _artifact(root / SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT)
    disable_sink_length = _artifact(root / DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT)
    prompt_shape = _artifact(root / PROMPT_SHAPE_SWEEP_ARTIFACT)
    rendered_prompt = _artifact(root / RENDERED_PROMPT_ARTIFACT)
    first_token = _artifact(root / FIRST_TOKEN_ARTIFACT)
    source_vs_quant = _artifact(root / SOURCE_VS_QUANT_ARTIFACT)
    no_source_classifier = _artifact(root / NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT)
    synced_long_tool_cache = _artifact(root / SYNCED_LONG_TOOL_CACHE_ARTIFACT)
    text_route = _artifact(root / TEXT_ROUTE_ARTIFACT)
    cb_oneshot_prefill = _artifact(root / CB_ONESHOT_PREFILL_ARTIFACT)
    cb_native_thinking_off = _artifact(root / CB_NATIVE_THINKING_OFF_ARTIFACT)
    cb_metal_baseline = _artifact(root / CB_METAL_BASELINE_ARTIFACT)
    mllm_inputs_embeds_interface = _artifact(root / MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT)
    latest_decode_speed = _artifact(root / LATEST_DECODE_SPEED_ARTIFACT)
    jangtq2_source_media_proof = _artifact(root / JANGTQ2_SOURCE_MEDIA_PROOF_ARTIFACT)
    jangtq2_dev_app_video_media_proof = _artifact(
        root / JANGTQ2_DEV_APP_VIDEO_MEDIA_PROOF_ARTIFACT
    )
    jangtq2_video_cache_proof = _artifact(root / JANGTQ2_VIDEO_CACHE_PROOF_ARTIFACT)
    api_cache_responses_contract = _artifact(root / NOHEAVY_API_CACHE_CONTRACT_ARTIFACT)

    structural_pass = structural.get("status") == "pass"
    text_cache_pass = text_cache.get("exists") and _text_cache_narrow_pass(text_cache["data"])
    switchglu_data = switchglu.get("data", {}) if isinstance(switchglu.get("data"), dict) else {}
    switchglu_shape_match = (
        switchglu_data.get("got_shape") == switchglu_data.get("manual_shape")
        and isinstance(switchglu_data.get("got_shape"), list)
    )
    switchglu_numeric_match = (
        isinstance(switchglu_data.get("max_abs_diff"), (int, float))
        and switchglu_data.get("max_abs_diff") <= 0.001
        and isinstance(switchglu_data.get("mean_abs_diff"), (int, float))
        and switchglu_data.get("mean_abs_diff") <= 0.0002
    )
    switchglu_pass = switchglu.get("exists") and (
        switchglu.get("status") == "pass"
        or switchglu_data.get("status") == "pass"
        or switchglu_data.get("ok") is True
        or (switchglu_shape_match and switchglu_numeric_match)
    )
    cache_match = cache_vs_nocache.get("exists") and _cache_vs_nocache_match(
        cache_vs_nocache["data"]
    )
    length_blocked = length_sweep.get("exists") and _length_sweep_blocked(
        length_sweep["data"]
    )
    smoke_evidence = (
        _all_local_smoke_evidence(all_local_smoke["data"])
        if all_local_smoke.get("exists")
        else {"exists": False}
    )
    smoke_evidence = _ensure_smoke_bundle_identity(smoke_evidence, ALL_LOCAL_SMOKE_ARTIFACT)
    if smoke_evidence.get("exists"):
        text_cache_pass = bool(text_cache_pass or smoke_evidence.get("text_cache_narrow_pass"))
    jang2l_smoke_evidence = (
        _all_local_smoke_evidence(jang2l_all_local_smoke["data"])
        if jang2l_all_local_smoke.get("exists")
        else {"exists": False}
    )
    jang2l_smoke_evidence = _ensure_smoke_bundle_identity(
        jang2l_smoke_evidence,
        JANG2L_ALL_LOCAL_SMOKE_ARTIFACT,
    )
    kvnone_noprefix_evidence = (
        _kvnone_noprefix_exactness_evidence(kvnone_noprefix_smoke["data"])
        if kvnone_noprefix_smoke.get("exists")
        and isinstance(kvnone_noprefix_smoke.get("data"), dict)
        else {"exists": False}
    )
    synced_evidence = (
        _synced_long_tool_cache_evidence(synced_long_tool_cache["data"])
        if synced_long_tool_cache.get("exists")
        and isinstance(synced_long_tool_cache.get("data"), dict)
        else {"exists": False}
    )
    if smoke_evidence.get("exists"):
        tool_blocked = not bool(smoke_evidence.get("tool_protocol_pass"))
        artifact_exactness_blocked = not bool(
            smoke_evidence.get("artifact_exactness_pass")
        )
        exact_cache_blocked = bool(smoke_evidence.get("exact_cache_blocked"))
        speed_blocked = bool(smoke_evidence.get("speed_blocked"))
    else:
        tool_blocked = tool_failure.get("exists") and _tool_protocol_blocked(
            tool_failure["data"]
        )
        artifact_exactness_blocked = False
        exact_cache_blocked = False
        speed_blocked = True
    if synced_evidence.get("exists"):
        if not smoke_evidence.get("exists"):
            tool_blocked = tool_blocked or bool(
                synced_evidence.get("tool_protocol_blocked")
            )
        length_blocked = length_blocked or bool(synced_evidence.get("long_prompt_blocked"))
        exact_cache_blocked = exact_cache_blocked and not bool(
            synced_evidence.get("short_cache_exact_pass")
        )
    text_route_evidence = (
        _text_route_evidence(text_route["data"])
        if text_route.get("exists") and isinstance(text_route.get("data"), dict)
        else {"exists": False}
    )
    if text_route_evidence.get("exists"):
        if text_route_evidence.get("tool_call_pass"):
            tool_blocked = False
        if text_route_evidence.get("long_prompt_recall_pass") and text_route_evidence.get(
            "long_prompt_oom_cleared"
        ):
            length_blocked = False
        if text_route_evidence.get("cache_visible_pass"):
            exact_cache_blocked = exact_cache_blocked and not bool(
                text_route_evidence.get("cache_release_exact_pass")
            )
        speed_blocked = speed_blocked or bool(text_route_evidence.get("speed_blocked"))
    cb_evidence = (
        _cb_oneshot_prefill_evidence(cb_oneshot_prefill["data"])
        if cb_oneshot_prefill.get("exists")
        and isinstance(cb_oneshot_prefill.get("data"), dict)
        else {"exists": False}
    )
    if cb_evidence.get("exists"):
        if cb_evidence.get("cache_exact_pass"):
            exact_cache_blocked = False
            prompt_shape_blocked = False
        if cb_evidence.get("prefix_paged_l2_cache_reproved"):
            text_route_evidence["cache_l2_not_reproved"] = False
        if not smoke_evidence.get("exists"):
            tool_blocked = bool(cb_evidence.get("tool_protocol_blocked"))
        length_blocked = length_blocked or bool(cb_evidence.get("long_prompt_crashed"))
        speed_blocked = speed_blocked or bool(cb_evidence.get("speed_blocked"))
    cb_native_evidence = (
        _cb_native_thinking_off_evidence(cb_native_thinking_off["data"])
        if cb_native_thinking_off.get("exists")
        and isinstance(cb_native_thinking_off.get("data"), dict)
        else {"exists": False}
    )
    memory_pressure_blocked = False
    if cb_native_evidence.get("exists"):
        if cb_native_evidence.get("cache_exact_pass"):
            exact_cache_blocked = False
        if cb_native_evidence.get("prefix_paged_l2_cache_reproved"):
            text_route_evidence["cache_l2_not_reproved"] = False
        if cb_native_evidence.get("memory_pressure_blocked"):
            prompt_shape_blocked = False
            memory_pressure_blocked = True
        speed_blocked = speed_blocked or bool(cb_native_evidence.get("speed_blocked"))
    cb_metal_baseline_evidence = (
        _cb_metal_baseline_evidence(cb_metal_baseline["data"])
        if cb_metal_baseline.get("exists")
        and isinstance(cb_metal_baseline.get("data"), dict)
        else {"exists": False}
    )
    if cb_metal_baseline_evidence.get("memory_pressure_cleared"):
        memory_pressure_blocked = False
    latest_decode_speed_evidence = (
        _latest_decode_speed_evidence(latest_decode_speed["data"])
        if latest_decode_speed.get("exists")
        and isinstance(latest_decode_speed.get("data"), dict)
        else {"exists": False, "speed_blocked": True}
    )
    if latest_decode_speed_evidence.get("exists"):
        # The latest decode-speed artifact supersedes older stale smoke speed
        # evidence. Keep PP/tool/long-prompt/media blockers separate; do not
        # let an old JANG_2L smoke keep reporting 1 tok/s after packaged
        # JANGTQ2 decode has been re-proved over the 40 tok/s floor.
        speed_blocked = bool(latest_decode_speed_evidence.get("speed_blocked"))
    elif smoke_evidence.get("exists") and smoke_evidence.get("speed_blocked") is False:
        # Current all-local JANGTQ2 live smoke is the freshest decode-speed
        # evidence when the standalone speed artifact is absent. Older
        # JANG_2L/diagnostic rows still feed exactness, media, L2, and prompt
        # blockers, but they must not keep the JANGTQ2 speed row red.
        speed_blocked = False
    sink_mode_fails = sink_mode_length.get("exists") and _all_sink_diagnostic_cases_fail(
        sink_mode_length["data"]
    )
    disable_sink_fails = disable_sink_length.get("exists") and _all_sink_diagnostic_cases_fail(
        disable_sink_length["data"]
    )
    prompt_shape_evidence = (
        _mimo_prompt_shape_first_token_evidence(
            prompt_shape["data"],
            rendered_prompt["data"],
            first_token["data"],
        )
        if prompt_shape.get("exists")
        and rendered_prompt.get("exists")
        and first_token.get("exists")
        else {
            "blocked": False,
            "classification": "missing_prompt_shape_first_token_artifact",
        }
    )
    prompt_shape_blocked = bool(prompt_shape_evidence.get("blocked"))
    if memory_pressure_blocked:
        prompt_shape_blocked = False
    source_vs_quant_pass = source_vs_quant.get("exists") and _source_vs_quant_first_divergence_pass(
        source_vs_quant["data"]
    )
    source_vs_quant_policy_skip = _source_vs_quant_policy_skip(
        no_source_classifier.get("data")
        if no_source_classifier.get("exists")
        and isinstance(no_source_classifier.get("data"), dict)
        else None
    )
    source_vs_quant_requirement_satisfied = bool(
        source_vs_quant_pass or source_vs_quant_policy_skip.get("policy_skipped")
    )
    mllm_inputs_embeds_interface_pass = (
        mllm_inputs_embeds_interface.get("exists")
        and isinstance(mllm_inputs_embeds_interface.get("data"), dict)
        and mllm_inputs_embeds_interface["data"].get("status") == "pass"
    )
    api_cache_responses_evidence = (
        _api_cache_responses_contract_evidence(api_cache_responses_contract["data"])
        if api_cache_responses_contract.get("exists")
        and isinstance(api_cache_responses_contract.get("data"), dict)
        else {
            "exists": False,
            "pass": False,
            "missing_required_checks": ["missing_noheavy_api_cache_contract_artifact"],
            "classification": "missing_api_cache_responses_contract_artifact",
        }
    )
    api_cache_responses_contract_pass = bool(api_cache_responses_evidence.get("pass"))
    media_runtime_evidence = _mimo_media_runtime_evidence(
        root,
        model_path,
        all_local_smoke.get("data")
        if all_local_smoke.get("exists")
        and isinstance(all_local_smoke.get("data"), dict)
        else None,
    )
    media_route_evidence = _mimo_jangtq2_media_route_evidence(
        jangtq2_source_media_proof,
        jangtq2_dev_app_video_media_proof,
        jangtq2_video_cache_proof,
    )
    media_runtime_wired = bool(media_runtime_evidence.get("runtime_media_wired"))
    media_metadata_ok = not bool(
        media_runtime_evidence.get("metadata_overadvertises_runtime_media")
    )
    media_runtime_capabilities_safe = bool(
        media_runtime_evidence.get("runtime_capabilities_safe")
    )
    current_runtime_media_supported = bool(
        media_runtime_evidence.get("runtime_capabilities_media_supported")
    )
    text_route_media_unwired_blocked = bool(
        text_route_evidence.get("media_unwired")
    ) and not current_runtime_media_supported
    image_video_live_e2e_proven = bool(smoke_evidence.get("live_media_e2e_pass"))
    image_video_live_e2e_blocked = not image_video_live_e2e_proven
    audio_waveform_live_e2e_proven = bool(smoke_evidence.get("audio_waveform_e2e_pass"))
    audio_waveform_live_e2e_blocked = not audio_waveform_live_e2e_proven
    block_disk_l2_restart_restore_proven = bool(
        smoke_evidence.get("block_disk_l2_restart_restore_pass")
    )
    bundle_media_l2_coverage = _merge_bundle_media_l2_coverage(
        smoke_evidence,
        jang2l_smoke_evidence,
    )
    jangtq2_coverage = bundle_media_l2_coverage.get("jangtq2")
    if not isinstance(jangtq2_coverage, dict):
        jangtq2_coverage = {}
    jang2l_coverage = bundle_media_l2_coverage.get("jang2l")
    if not isinstance(jang2l_coverage, dict):
        jang2l_coverage = {}
    mimo_jangtq2_live_media_l2 = bool(
        jangtq2_coverage.get("live_media_e2e_pass")
        and jangtq2_coverage.get("block_disk_l2_restart_cache_hit")
    )
    mimo_jang2l_live_media_l2 = bool(
        jang2l_coverage.get("live_media_e2e_pass")
        and jang2l_coverage.get("block_disk_l2_restart_cache_hit")
    )
    mimo_jangtq2_l2_restart_cache_hit = bool(
        jangtq2_coverage.get("block_disk_l2_restart_cache_hit")
    )
    mimo_jangtq2_l2_restart_visible_output = bool(
        jangtq2_coverage.get("block_disk_l2_restart_visible_output_pass")
    )
    mimo_jang2l_l2_restart_cache_hit = bool(
        jang2l_coverage.get("block_disk_l2_restart_cache_hit")
    )
    mimo_jang2l_l2_restart_visible_output = bool(
        jang2l_coverage.get("block_disk_l2_restart_visible_output_pass")
    )
    mimo_jang2l_tool_long_prompt_metal_oom_blocked = bool(
        jang2l_smoke_evidence.get("metal_oom_crash")
        or jang2l_smoke_evidence.get("server_died_mid_probe")
    )
    mimo_jang2l_tight_memory_prompt_budget_blocked = bool(
        jang2l_smoke_evidence.get("typed_tight_memory_prompt_rejection")
    )
    mimo_jang2l_media_prefill_budget_blocked = bool(
        jang2l_smoke_evidence.get("typed_media_prefill_budget_rejection")
    )
    mimo_jang2l_post_media_working_set_pressure_blocked = bool(
        jang2l_smoke_evidence.get("post_media_working_set_pressure_rejection")
    )
    mimo_jang2l_media_capability_downscoped_to_text = bool(
        jang2l_smoke_evidence.get("media_capability_downscoped_to_text")
    )
    mimo_jang2l_media_capability_memory_gated = bool(
        jang2l_smoke_evidence.get("media_capability_memory_gated")
    )
    block_disk_l2_restart_restore_blocked = not bool(
        smoke_evidence.get("block_disk_l2_restart_cache_hit")
    )
    if image_video_live_e2e_proven and current_runtime_media_supported:
        media_runtime_wired = True
        media_runtime_evidence["runtime_media_wired_superseded_by_live_e2e"] = True
        media_runtime_evidence["status"] = (
            "open" if media_runtime_evidence.get("metadata_overadvertises_runtime_media") else "pass"
        )
        media_runtime_evidence["classification"] = (
            "live_image_video_runtime_wired_audio_waveform_e2e_still_open"
        )
    else:
        media_runtime_evidence["runtime_media_wired_superseded_by_live_e2e"] = False
    if media_route_evidence.get("transport_proven"):
        media_runtime_wired = True
        media_runtime_evidence["runtime_media_wired_superseded_by_route_proof"] = True
        media_runtime_evidence["media_route_evidence"] = media_route_evidence
        media_runtime_evidence["status"] = "open"
        media_runtime_evidence["classification"] = str(
            media_route_evidence.get("classification")
        )
        media_runtime_evidence["runtime_media_wired_boundary"] = str(
            media_route_evidence.get("boundary")
        )
    else:
        media_runtime_evidence["runtime_media_wired_superseded_by_route_proof"] = False
        media_runtime_evidence["media_route_evidence"] = media_route_evidence
    if image_video_live_e2e_blocked:
        media_runtime_evidence["live_media_e2e_status"] = str(
            smoke_evidence.get("live_media_e2e_status") or "missing"
        )
        media_runtime_evidence["live_media_e2e_boundary"] = (
            "Source-level MiMo media capability wiring is not a substitute for "
            "live image/audio/video server output proof."
        )
        media_runtime_evidence["live_media_e2e_failures"] = smoke_evidence.get(
            "live_media_e2e_failures"
        )
    else:
        media_runtime_evidence["live_media_e2e_status"] = "pass"
        media_runtime_evidence["live_media_e2e_boundary"] = (
            "Live MiMo object image/video, audio waveform, and post-media text "
            "recovery rows passed for the current proof."
        )
        media_runtime_evidence["live_media_e2e_labels"] = smoke_evidence.get(
            "live_media_e2e_labels"
        )
    media_runtime_evidence["audio_waveform_live_e2e_status"] = (
        "pass"
        if audio_waveform_live_e2e_proven
        else str(smoke_evidence.get("audio_waveform_e2e_status") or "missing")
    )
    media_runtime_evidence["audio_waveform_live_e2e_boundary"] = (
        "Object image/video live E2E does not clear audio waveform ingestion, "
        "tokenizer, audio-code splice, cache salt, and post-audio recovery proof."
    )
    if audio_waveform_live_e2e_proven:
        media_runtime_evidence["audio_waveform_live_e2e_boundary"] = (
            "Live MiMo audio waveform and post-audio text recovery rows passed."
        )
        media_runtime_evidence["audio_waveform_live_e2e_labels"] = smoke_evidence.get(
            "audio_waveform_e2e_labels"
        )
    else:
        media_runtime_evidence["audio_waveform_live_e2e_failures"] = smoke_evidence.get(
            "audio_waveform_e2e_failures"
        )
    if text_route_evidence.get("media_unwired") and current_runtime_media_supported:
        media_runtime_evidence["text_route_media_unwired_superseded_by_source"] = True
    else:
        media_runtime_evidence["text_route_media_unwired_superseded_by_source"] = False
    text_route_long_prompt_supersedes_length_sweep = bool(
        text_route_evidence.get("exists")
        and text_route_evidence.get("long_prompt_recall_pass")
        and text_route_evidence.get("long_prompt_oom_cleared")
    )
    if text_route_long_prompt_supersedes_length_sweep:
        length_blocked = False
    if text_route_evidence.get("exists") and text_route_evidence.get("cache_visible_pass"):
        prompt_shape_blocked = False

    stale_present = [item for item in stale_state if item["exists"]]
    release_clearance = (
        manifest_check["status"] == "pass"
        and structural_pass
        and text_cache_pass
        and switchglu_pass
        and cache_match
        and not length_blocked
        and not tool_blocked
        and not artifact_exactness_blocked
        and not exact_cache_blocked
        and not speed_blocked
        and not prompt_shape_blocked
        and not memory_pressure_blocked
        and source_vs_quant_requirement_satisfied
        and mllm_inputs_embeds_interface_pass
        and api_cache_responses_contract_pass
        and not bool(text_route_evidence.get("cache_l2_not_reproved"))
        and not block_disk_l2_restart_restore_blocked
        and not text_route_media_unwired_blocked
        and media_runtime_wired
        and media_metadata_ok
        and media_runtime_capabilities_safe
        and not image_video_live_e2e_blocked
        and not audio_waveform_live_e2e_blocked
        and mimo_jangtq2_live_media_l2
        and mimo_jangtq2_l2_restart_visible_output
        and mimo_jang2l_live_media_l2
        and not mimo_jang2l_tool_long_prompt_metal_oom_blocked
        and not mimo_jang2l_tight_memory_prompt_budget_blocked
        and not mimo_jang2l_media_prefill_budget_blocked
        and not mimo_jang2l_post_media_working_set_pressure_blocked
        and not stale_present
    )

    status = "pass" if release_clearance else "open"
    blockers = []
    if manifest_check["status"] != "pass":
        blockers.append("mimo_manifest_integrity_failed")
    if stale_present:
        blockers.append("stale_mimo_local_state_present")
    if not structural_pass:
        blockers.append("mimo_structural_verify_missing_or_failed")
    if not text_cache_pass:
        blockers.append("mimo_text_cache_narrow_proof_missing_or_failed")
    if not switchglu_pass:
        blockers.append("mimo_switchglu_selected_expert_parity_missing_or_failed")
    if not cache_match:
        blockers.append("mimo_cache_vs_nocache_next_token_missing_or_mismatch")
    if length_blocked:
        blockers.append("mimo_long_prompt_coherence_blocked")
    if tool_blocked:
        blockers.append("mimo_tool_protocol_blocked")
    if artifact_exactness_blocked:
        blockers.append("mimo_jangtq2_artifact_exactness_blocked")
    if exact_cache_blocked:
        blockers.append("mimo_exact_cache_prompt_following_blocked")
    if speed_blocked:
        blockers.append("mimo_decode_speed_below_release_target")
    if prompt_shape_blocked:
        blockers.append("mimo_system_prompt_first_token_stop_blocked")
    if memory_pressure_blocked:
        blockers.append("mimo_cb_system_prompt_working_set_pressure_blocked")
    if not source_vs_quant_requirement_satisfied:
        blockers.append("mimo_source_vs_quant_first_divergence_missing_or_failed")
    if not mllm_inputs_embeds_interface_pass:
        blockers.append("mimo_mllm_inputs_embeds_interface_missing_or_failed")
    if not api_cache_responses_contract_pass:
        blockers.append("mimo_api_cache_responses_contract_missing_or_failed")
    if text_route_evidence.get("cache_l2_not_reproved"):
        blockers.append("mimo_prefix_paged_l2_cache_not_reproved")
    if block_disk_l2_restart_restore_blocked:
        blockers.append("mimo_block_disk_l2_restart_restore_missing")
    if text_route_media_unwired_blocked:
        blockers.append("mimo_vl_audio_video_unwired")
    if image_video_live_e2e_blocked:
        blockers.append("mimo_image_video_live_e2e_missing")
    if audio_waveform_live_e2e_blocked:
        blockers.append("mimo_audio_waveform_live_e2e_missing")
    if not mimo_jangtq2_live_media_l2:
        blockers.append("mimo_jangtq2_live_media_l2_missing")
    if mimo_jangtq2_l2_restart_cache_hit and not mimo_jangtq2_l2_restart_visible_output:
        blockers.append("mimo_jangtq2_l2_restart_visible_output_blocked")
    if not mimo_jang2l_live_media_l2:
        blockers.append("mimo_jang2l_live_media_l2_missing")
    if mimo_jang2l_media_capability_downscoped_to_text:
        blockers.append("mimo_jang2l_media_capability_downscoped_to_text")
    if mimo_jang2l_media_capability_memory_gated:
        blockers.append("mimo_jang2l_media_capability_memory_gated")
    if mimo_jang2l_l2_restart_cache_hit and not mimo_jang2l_l2_restart_visible_output:
        blockers.append("mimo_jang2l_l2_restart_visible_output_blocked")
    if mimo_jang2l_tool_long_prompt_metal_oom_blocked:
        blockers.append("mimo_jang2l_tool_long_prompt_metal_oom_blocked")
    if mimo_jang2l_tight_memory_prompt_budget_blocked:
        blockers.append("mimo_jang2l_tight_memory_prompt_budget_blocked")
    if mimo_jang2l_media_prefill_budget_blocked:
        blockers.append("mimo_jang2l_media_prefill_budget_blocked")
    if mimo_jang2l_post_media_working_set_pressure_blocked:
        blockers.append("mimo_jang2l_post_media_working_set_pressure_blocked")
    if not media_runtime_wired:
        blockers.append("mimo_media_runtime_implementation_missing")
    if not media_metadata_ok:
        blockers.append("mimo_model_metadata_overadvertises_unwired_media")
    if not media_runtime_capabilities_safe:
        blockers.append("mimo_runtime_capabilities_media_status_missing_or_unsafe")

    return {
        "status": status,
        "local_release_clearance": release_clearance,
        "model_path": str(model_path),
        "manifest_integrity": manifest_check,
        "stale_local_state": {
            "cleanup_log": str(root / CLEANUP_LOG),
            "all_stale_targets_absent": not stale_present,
            "targets": stale_state,
        },
        "component_ok": {
            "manifest_integrity": manifest_check["status"] == "pass",
            "bookend_quantization_sidecars": bookend_quantization.get("status") == "pass",
            "stale_local_state_absent": not stale_present,
            "structural_verify": bool(structural_pass),
            "text_cache_narrow": bool(text_cache_pass),
            "switchglu_selected_expert_parity": bool(switchglu_pass),
            "cache_vs_nocache_next_token": bool(cache_match),
            "long_prompt_coherence": not bool(length_blocked),
            "tool_protocol": not bool(tool_blocked),
            "artifact_exactness": not bool(artifact_exactness_blocked),
            "exactness_cache_kv_quant_excluded": bool(
                kvnone_noprefix_evidence.get("exactness_not_caused_by_prefix_or_kv_quant")
            ),
            "exact_cache_prompt_following": not bool(exact_cache_blocked),
            "decode_speed_target": not bool(speed_blocked),
            "system_prompt_first_token_stop": not bool(prompt_shape_blocked),
            "cb_system_prompt_working_set_pressure": not bool(memory_pressure_blocked),
            "source_vs_quant_first_divergence": bool(source_vs_quant_pass),
            "source_vs_quant_policy_skipped": bool(
                source_vs_quant_policy_skip.get("policy_skipped")
            ),
            "source_vs_quant_requirement_satisfied": bool(
                source_vs_quant_requirement_satisfied
            ),
            "mllm_inputs_embeds_interface": bool(mllm_inputs_embeds_interface_pass),
            "api_cache_responses_contract": bool(api_cache_responses_contract_pass),
            "media_weights_preserved": bool(
                media_runtime_evidence.get("media_weights_preserved")
            ),
            "media_runtime_capabilities_safe": bool(media_runtime_capabilities_safe),
            "media_model_metadata_text_only_contract": bool(media_metadata_ok),
            "media_runtime_implementation": bool(media_runtime_wired),
            "text_route_long_prompt": bool(
                text_route_evidence.get("long_prompt_recall_pass")
                and text_route_evidence.get("long_prompt_oom_cleared")
            ),
            "text_route_tool_call": bool(text_route_evidence.get("tool_call_pass")),
            "text_route_long_prompt_supersedes_length_sweep": bool(
                text_route_long_prompt_supersedes_length_sweep
            ),
            "prefix_paged_l2_cache_reproved": not bool(
                text_route_evidence.get("cache_l2_not_reproved")
            ),
            "block_disk_l2_restart_restore": bool(
                block_disk_l2_restart_restore_proven
            ),
            "mimo_media_wired": bool(
                media_runtime_wired
                and media_metadata_ok
                and media_runtime_capabilities_safe
                and not text_route_media_unwired_blocked
            ),
            "mimo_image_video_live_e2e": bool(image_video_live_e2e_proven),
            "mimo_audio_waveform_live_e2e": bool(audio_waveform_live_e2e_proven),
            "mimo_live_media_e2e": bool(
                image_video_live_e2e_proven and audio_waveform_live_e2e_proven
            ),
            "mimo_jangtq2_live_media_l2": bool(mimo_jangtq2_live_media_l2),
            "mimo_jangtq2_l2_restart_cache_hit": bool(
                mimo_jangtq2_l2_restart_cache_hit
            ),
            "mimo_jangtq2_l2_restart_visible_output": bool(
                mimo_jangtq2_l2_restart_visible_output
            ),
            "mimo_jang2l_live_media_l2": bool(mimo_jang2l_live_media_l2),
            "mimo_jang2l_l2_restart_cache_hit": bool(
                mimo_jang2l_l2_restart_cache_hit
            ),
            "mimo_jang2l_l2_restart_visible_output": bool(
                mimo_jang2l_l2_restart_visible_output
            ),
            "mimo_jang2l_media_capability_downscoped_to_text": bool(
                mimo_jang2l_media_capability_downscoped_to_text
            ),
            "mimo_jang2l_media_capability_memory_gated": not bool(
                mimo_jang2l_media_capability_memory_gated
            ),
            "mimo_jang2l_tool_long_prompt_metal_oom": not bool(
                mimo_jang2l_tool_long_prompt_metal_oom_blocked
            ),
            "mimo_jang2l_tight_memory_prompt_budget": not bool(
                mimo_jang2l_tight_memory_prompt_budget_blocked
            ),
            "mimo_jang2l_media_prefill_budget": not bool(
                mimo_jang2l_media_prefill_budget_blocked
            ),
            "mimo_jang2l_post_media_working_set_pressure": not bool(
                mimo_jang2l_post_media_working_set_pressure_blocked
            ),
            "manual_sink_does_not_clear_length_generation": bool(sink_mode_fails),
            "disable_sink_does_not_clear_length_generation": bool(disable_sink_fails),
        },
        "latest_decode_speed_evidence": latest_decode_speed_evidence,
        "blockers": blockers,
        "artifacts": {
            "structural": str(STRUCTURAL_ARTIFACT),
            "text_cache": str(TEXT_CACHE_ARTIFACT),
            "switchglu": str(SWITCHGLU_ARTIFACT),
            "length_sweep": str(LENGTH_SWEEP_ARTIFACT),
            "tool_failure": str(TOOL_FAILURE_ARTIFACT),
            "all_local_smoke": str(ALL_LOCAL_SMOKE_ARTIFACT),
            "jang2l_all_local_smoke": str(JANG2L_ALL_LOCAL_SMOKE_ARTIFACT),
            "kvnone_noprefix_smoke": str(KVNONE_NOPREFIX_SMOKE_ARTIFACT),
            "cache_vs_nocache": str(CACHE_VS_NOCACHE_ARTIFACT),
            "sink_mode_length_diagnostic": str(SINK_MODE_LENGTH_DIAGNOSTIC_ARTIFACT),
            "disable_sink_length_diagnostic": str(DISABLE_SINK_LENGTH_DIAGNOSTIC_ARTIFACT),
            "prompt_shape_sweep": str(PROMPT_SHAPE_SWEEP_ARTIFACT),
            "rendered_prompt_compare": str(RENDERED_PROMPT_ARTIFACT),
            "first_token_probe": str(FIRST_TOKEN_ARTIFACT),
            "source_vs_quant_first_divergence": str(SOURCE_VS_QUANT_ARTIFACT),
            "no_source_exactness_classifier": str(NO_SOURCE_EXACTNESS_CLASSIFIER_ARTIFACT),
            "synced_long_tool_cache": str(SYNCED_LONG_TOOL_CACHE_ARTIFACT),
            "text_route_live_proof": str(TEXT_ROUTE_ARTIFACT),
            "cb_oneshot_prefill_live_proof": str(CB_ONESHOT_PREFILL_ARTIFACT),
            "cb_native_thinking_off_live_proof": str(CB_NATIVE_THINKING_OFF_ARTIFACT),
            "cb_metal_baseline_live_proof": str(CB_METAL_BASELINE_ARTIFACT),
            "mllm_inputs_embeds_interface": str(MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT),
            "latest_decode_speed": str(LATEST_DECODE_SPEED_ARTIFACT),
            "api_cache_responses_contract": str(NOHEAVY_API_CACHE_CONTRACT_ARTIFACT),
            "jangtq2_source_media_proof": str(JANGTQ2_SOURCE_MEDIA_PROOF_ARTIFACT),
            "jangtq2_dev_app_video_media_proof": str(
                JANGTQ2_DEV_APP_VIDEO_MEDIA_PROOF_ARTIFACT
            ),
            "jangtq2_video_cache_proof": str(JANGTQ2_VIDEO_CACHE_PROOF_ARTIFACT),
        },
        "diagnostics": {
            "cache_vs_nocache_next_token_match": bool(cache_match),
            "all_local_smoke": smoke_evidence,
            "jang2l_all_local_smoke": jang2l_smoke_evidence,
            "kvnone_noprefix_exactness_isolation": kvnone_noprefix_evidence,
            "prompt_shape_first_token": prompt_shape_evidence,
            "synced_long_tool_cache": synced_evidence,
            "text_route": text_route_evidence,
            "cb_oneshot_prefill": cb_evidence,
            "cb_native_thinking_off": cb_native_evidence,
            "cb_metal_baseline": cb_metal_baseline_evidence,
            "mllm_inputs_embeds_interface": (
                mllm_inputs_embeds_interface.get("data")
                if mllm_inputs_embeds_interface.get("exists")
                else {
                    "status": "missing",
                    "artifact": str(MLLM_INPUTS_EMBEDS_INTERFACE_ARTIFACT),
                    "classification": "missing_mimo_mllm_inputs_embeds_interface_proof",
                }
            ),
            "api_cache_responses_contract": api_cache_responses_evidence,
            "mimo_media_runtime": media_runtime_evidence,
            "mimo_jangtq2_media_route": media_route_evidence,
            "bookend_quantization": bookend_quantization,
            "source_vs_quant_first_divergence": (
                source_vs_quant.get("data")
                if source_vs_quant.get("exists")
                else {
                    "status": "missing",
                    "artifact": str(SOURCE_VS_QUANT_ARTIFACT),
                    "classification": "missing_local_source_vs_quant_first_divergence",
                }
            ),
            "source_vs_quant_policy_skip": source_vs_quant_policy_skip,
            "manual_sink_sdpa_clears_length_generation": False if sink_mode_fails else None,
            "disable_sink_clears_length_generation": False if disable_sink_fails else None,
            "sink_boundary": (
                "Manual sink SDPA and disabling SWA sink do not clear the current "
                "MiMo length-generation failures; do not classify the blocker as "
                "an MLX sink-kernel-only issue."
            ),
            "exactness_cache_boundary": (
                "The no-prefix/KV-none smoke reproduced the same MiMo literal "
                "mutations with runtime KV quantization, native cache storage "
                "quantization, prefix cache, paged cache, block-disk L2, and cache "
                "hits disabled. Do not chase prefix-cache reuse, L2 disk cache, or "
            "runtime KV quantization as the primary current MiMo exactness cause."
            ),
            "exactness_bookend_boundary": (
                "Current MiMo JANGTQ_2 exactness is not the old missing "
                "lm_head/embed_tokens sidecar failure class when "
                "bookend_quantization.status=pass; focus corrected routed "
                "TQ/quantization or runtime decode/logit proof."
            ),
        },
        "release_boundary": (
            "MiMo local artifact integrity is clean, but release remains blocked "
            "until long-prompt coherence, tool protocol/artifact exactness, "
            "decode speed, CB system prompt working-set pressure, API/cache/"
            "Responses contracts, local source-vs-quant first-divergence or an "
            "equivalent current-artifact classification, and media wiring proof "
            "pass without fake parser injection, hidden thinking-mode rewrites, "
            "forced fallback, or cache disabling."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    root = Path.cwd()
    result = build_audit(root, args.model_path, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"pass", "open"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
