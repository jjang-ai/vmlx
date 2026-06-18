#!/usr/bin/env python3
"""Fail-closed preflight for the 1.5.65 MM3 + Gemma 4 compatibility gate.

This is intentionally separate from ``scoped-release-preflight.py`` because the
1.5.63 script is a historical scoped-release record.  The 1.5.65 objective is
broader: current MiniMax-M3 strict exactness, current Gemma 4 JANG_4M VL rows,
current large MXFP4 visual rows, clean-start/autodetect evidence, lifecycle
evidence, and source version stamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRESERVED_PROOF_ROOT = ROOT / "docs/internal/release-gates/current-proof-preserved"

MM3_STRESS_GLOB = "build/live-mm3-stress-*/mm3-stress-proof.json"
GEMMA_MEDIA_GLOB = "build/live-gemma4-media-*/gemma4-media-proof.json"
CLEAN_START_GLOB = "build/live-clean-start-*/clean-start-proof.json"
LIFECYCLE_GLOB = "build/live-lifecycle-*/lifecycle-proof.json"

REQUIRED_GEMMA_MEDIA_ROWS = {
    "gemma4-e2b-jang4m-vl-current-64",
    "gemma4-e4b-jang4m-vl-current-64",
    "gemma4-12b-jang4m-vl-current-64",
    "gemma4-26b-jang4m-vl-current-64",
    "gemma4-31b-jang4m-vl-current-64",
    "gemma4-26b-mxfp4-visual-current-64",
    "gemma4-31b-mxfp4-visual-current-64",
}

REQUIRED_CLEAN_START_ROWS = {
    "mm3-reap40-d3-real-profile-gateway-visible-off",
    "gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off",
    "gemma4-26b-mxfp4-clean-start-current-64",
    "gemma4-31b-mxfp4-clean-start-current-64",
    # The JANG_4M full-size rows are intentionally required here.  The media
    # stress artifacts prove generation once settings are configured; these
    # clean-start rows prove "delete all saved sessions, start without changing
    # settings, and inspect CLI/UI/autodetect defaults" for the current bundles.
    "gemma4-12b-jang4m-clean-start-current-64",
    "gemma4-26b-jang4m-clean-start-current-64",
    "gemma4-31b-jang4m-clean-start-current-64",
}

REQUIRED_LIFECYCLE_ROWS = {
    "mm3-reap40-d3-lifecycle",
    "gemma4-e2b-mxfp4-lifecycle-vl",
    "gemma4-26b-mxfp4-lifecycle-current-64",
    "gemma4-31b-mxfp4-lifecycle-current-64",
}


def load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing proof artifact: {path}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid JSON artifact {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def text_has(text: Any, marker: str) -> bool:
    return marker in str(text or "")


def load_pass(path: Path, failures: list[str]) -> dict[str, Any]:
    data = load_json(path, failures)
    require(data.get("status") == "pass", failures, f"{path} status={data.get('status')!r}")
    require(not data.get("failures"), failures, f"{path} failures={data.get('failures')!r}")
    return data


def proof_paths(glob_pattern: str) -> list[Path]:
    paths = list(ROOT.glob(glob_pattern))
    if PRESERVED_PROOF_ROOT.exists():
        paths.extend(PRESERVED_PROOF_ROOT.glob(glob_pattern))
        if glob_pattern.startswith("build/"):
            paths.extend(PRESERVED_PROOF_ROOT.glob(glob_pattern.removeprefix("build/")))
    return sorted(paths)


def latest_by_row(glob_pattern: str) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for path in proof_paths(glob_pattern):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = str(data.get("rowName") or "")
        if data.get("status") == "pass" and not data.get("failures") and row:
            rows[row] = path
    return rows


def latest_mm3_stress(failures: list[str]) -> Path | None:
    passing: list[Path] = []
    for path in proof_paths(MM3_STRESS_GLOB):
        data = load_json(path, [])
        if data.get("status") == "pass" and not data.get("failures"):
            passing.append(path)
    if not passing:
        failures.append("missing current MM3 strict stress pass artifact")
        return None
    build_paths = [path for path in passing if "docs/internal/release-gates/current-proof-preserved" not in str(path)]
    return build_paths[-1] if build_paths else passing[-1]


def _tool_call_count(row: dict[str, Any], name: str) -> int:
    raw = row.get("toolCallsJson") or "[]"
    try:
        calls = json.loads(raw)
    except Exception:
        calls = []
    return sum(1 for c in calls if isinstance(c, dict) and c.get("toolName") == name and c.get("phase") == "calling")


def validate_mm3_stress(data: dict[str, Any], failures: list[str]) -> None:
    cfg = data.get("sessionConfigAfterStart") or {}
    cache_end = data.get("cacheEnd") or {}
    native = cache_end.get("native_cache") or {}
    stats = cache_end.get("scheduler_stats") or {}

    require(cfg.get("toolCallParser") == "minimax_m3", failures, "MM3 stress tool parser mismatch")
    require(cfg.get("reasoningParser") == "minimax_m3", failures, "MM3 stress reasoning parser mismatch")
    require(cfg.get("usePagedCache") is False, failures, f"MM3 stress usePagedCache={cfg.get('usePagedCache')!r}")
    require(cfg.get("enableDiskCache") is True, failures, "MM3 stress disk cache not enabled")
    require(cfg.get("enableJit") is False, failures, "MM3 stress JIT not forced off")
    require(native.get("schema") == "minimax_m3_msa_v1", failures, f"MM3 native schema={native.get('schema')!r}")
    require("msa_idx_keys" in set(native.get("components") or []), failures, "MM3 native cache missing msa_idx_keys")
    require((native.get("generic_turboquant_kv") or {}).get("enabled") is False, failures, "MM3 generic TQ-KV not disabled")
    require(native.get("prompt_disk_l2") is True, failures, "MM3 prompt disk L2 not enabled")
    require((stats.get("cache_hit_tokens") or 0) > 0, failures, "MM3 cache hit tokens not proven")

    mixed = ((data.get("ui") or {}).get("mixedSession") or {}).get("turns") or []
    by_label = {row.get("label"): row for row in mixed if isinstance(row, dict)}
    on = by_label.get("tool_reasoning_on") or {}
    auto = by_label.get("tool_reasoning_auto") or {}
    require(text_has(on.get("content"), "M3_MIX_TOOL_ON_DONE"), failures, "MM3 mixed reasoning-on exact marker missing")
    require(text_has(auto.get("content"), "M3_MIX_TOOL_AUTO_DONE"), failures, "MM3 mixed reasoning-auto exact marker missing")
    require(_tool_call_count(on, "run_command") == 1, failures, "MM3 mixed reasoning-on did not call run_command exactly once")
    require(_tool_call_count(auto, "run_command") == 1, failures, "MM3 mixed reasoning-auto did not call run_command exactly once")

    streaming = data.get("streaming") or {}
    require(text_has((streaming.get("chatTool") or {}).get("toolArgs"), "MM3_STREAM_CHAT_TOOL"), failures, "MM3 streaming Chat tool args exact marker missing")
    completed = (streaming.get("responsesTool") or {}).get("completedToolCalls") or []
    exact_completed = [
        c for c in completed
        if c.get("name") == "record_mm3_stream_response_label"
        and c.get("status") == "completed"
        and text_has(c.get("arguments"), "MM3_STREAM_RESP_TOOL")
    ]
    require(len(exact_completed) == 1, failures, f"MM3 streaming Responses completed exact tool calls={len(exact_completed)}")


def validate_gemma_media(row: str, data: dict[str, Any], failures: list[str]) -> None:
    cfg = data.get("sessionConfigAfterStart") or {}
    cache_end = data.get("cacheEnd") or {}
    native = cache_end.get("native_cache") or {}
    stats = cache_end.get("scheduler_stats") or {}
    defaults = data.get("generationDefaults") or {}
    session_defaults = defaults.get("session") or {}
    caps = data.get("capabilities") or {}
    modalities = {str(x).lower() for x in caps.get("modalities") or []}

    require(cfg.get("toolCallParser") == "gemma4", failures, f"{row}: tool parser mismatch")
    require(cfg.get("reasoningParser") == "gemma4", failures, f"{row}: reasoning parser mismatch")
    require(cfg.get("isMultimodal") is True, failures, f"{row}: isMultimodal not true")
    require(cfg.get("usePagedCache") is False, failures, f"{row}: paged cache not off")
    require(cfg.get("enableDiskCache") is True, failures, f"{row}: disk cache not enabled")
    require(cfg.get("kvCacheQuantization") == "auto", failures, f"{row}: kvCacheQuantization={cfg.get('kvCacheQuantization')!r}")
    require("text" in modalities and ("vision" in modalities or "image" in modalities), failures, f"{row}: modalities missing text/vision: {sorted(modalities)}")
    require(session_defaults.get("temperature") == 1, failures, f"{row}: temperature default mismatch")
    require(session_defaults.get("top_p") == 0.95, failures, f"{row}: top_p default mismatch")
    require(session_defaults.get("top_k") == 64, failures, f"{row}: top_k default mismatch")
    require(native.get("schema") == "mixed_swa_kv_v1", failures, f"{row}: native schema={native.get('schema')!r}")
    require((native.get("generic_turboquant_kv") or {}).get("enabled") is False, failures, f"{row}: generic TQ-KV unexpectedly enabled")
    require((stats.get("cache_hit_tokens") or 0) > 0, failures, f"{row}: cache hit tokens not proven")

    mixed = ((data.get("ui") or {}).get("mixedSession") or {}).get("turns") or []
    by_label = {turn.get("label"): turn for turn in mixed if isinstance(turn, dict)}
    require(text_has((by_label.get("image_reasoning_on") or {}).get("content"), "GEMMA_MIX_IMAGE_RED"), failures, f"{row}: mixed image exact marker missing")
    require(text_has((by_label.get("tool_reasoning_on") or {}).get("content"), "GEMMA_MIX_TOOL_ON_DONE"), failures, f"{row}: mixed tool-on exact marker missing")
    require(text_has((by_label.get("tool_reasoning_auto") or {}).get("content"), "GEMMA_MIX_TOOL_AUTO_DONE"), failures, f"{row}: mixed tool-auto exact marker missing")
    require(_tool_call_count(by_label.get("tool_reasoning_on") or {}, "run_command") == 1, failures, f"{row}: tool-on run_command count mismatch")
    require(_tool_call_count(by_label.get("tool_reasoning_auto") or {}, "run_command") == 1, failures, f"{row}: tool-auto run_command count mismatch")

    streaming = data.get("streaming") or {}
    require(text_has((streaming.get("chatTool") or {}).get("toolArgs"), "GEMMA_STREAM_CHAT_TOOL"), failures, f"{row}: streaming Chat tool args missing")
    completed = (streaming.get("responsesTool") or {}).get("completedToolCalls") or []
    exact_completed = [
        c for c in completed
        if c.get("name") == "record_gemma_stream_response_label"
        and c.get("status") == "completed"
        and text_has(c.get("arguments"), "GEMMA_STREAM_RESP_TOOL")
    ]
    require(len(exact_completed) == 1, failures, f"{row}: streaming Responses completed exact tool calls={len(exact_completed)}")


def validate_clean_start(row: str, data: dict[str, Any], failures: list[str]) -> None:
    cfg = data.get("sessionConfigAfterStart") or {}
    gateway = data.get("gateway") or {}
    caps = data.get("capabilities") or (gateway.get("capabilities") or {})
    cache_end = data.get("cacheEnd") or {}
    native = cache_end.get("native_cache") or (((caps.get("cache") or {}).get("native")) or {})
    status = gateway.get("status") or {}
    ui_score = ((data.get("defaultUiTurn") or {}).get("score") or {})

    require(status.get("running") is True and status.get("port") == 8080, failures, f"{row}: gateway/default port not proven")
    require(cfg.get("usePagedCache") is False, failures, f"{row}: clean-start paged cache not off")
    require(cfg.get("enableDiskCache") is True, failures, f"{row}: clean-start disk cache not enabled")
    require(cfg.get("kvCacheQuantization") == "auto", failures, f"{row}: clean-start kvCacheQuantization={cfg.get('kvCacheQuantization')!r}")
    require(ui_score.get("empty") is False and ui_score.get("loopSuspect") is False, failures, f"{row}: default UI turn empty/loop suspect")
    if row.startswith("mm3"):
        require(cfg.get("toolCallParser") == "minimax_m3", failures, f"{row}: MM3 tool parser mismatch")
        require(cfg.get("reasoningParser") == "minimax_m3", failures, f"{row}: MM3 reasoning parser mismatch")
        require(cfg.get("enableJit") is False, failures, f"{row}: MM3 JIT not off")
        require(native.get("schema") == "minimax_m3_msa_v1", failures, f"{row}: MM3 native schema mismatch")
    else:
        require(cfg.get("toolCallParser") == "gemma4", failures, f"{row}: Gemma tool parser mismatch")
        require(cfg.get("reasoningParser") == "gemma4", failures, f"{row}: Gemma reasoning parser mismatch")
        require(cfg.get("isMultimodal") is True, failures, f"{row}: Gemma isMultimodal not true")
        require(native.get("schema") == "mixed_swa_kv_v1", failures, f"{row}: Gemma native schema mismatch")


def validate_lifecycle(row: str, data: dict[str, Any], failures: list[str]) -> None:
    abort = data.get("abortTurn") or {}
    stop = data.get("stopTurn") or {}
    require(abort.get("streamingBeforeAbort") is True, failures, f"{row}: abort did not catch active stream")
    require((abort.get("abortResult") or {}).get("success") is True, failures, f"{row}: abort IPC failed")
    require((abort.get("notStreaming") or {}).get("streaming") is False, failures, f"{row}: still streaming after abort")
    require((abort.get("quiet") or {}).get("changed") is False, failures, f"{row}: changed after abort quiet wait")
    require(stop.get("streamingBeforeStop") is True, failures, f"{row}: stop did not catch active stream")
    require((stop.get("stopResult") or {}).get("success") is True, failures, f"{row}: stop IPC failed")
    require((stop.get("notStreaming") or {}).get("streaming") is False, failures, f"{row}: still streaming after stop")
    require(((stop.get("sessionAfterStop") or {}).get("status") == "stopped"), failures, f"{row}: session not stopped")
    require((stop.get("healthAfterStop") or {}).get("ok") is False, failures, f"{row}: backend still healthy after stop")
    require((stop.get("quiet") or {}).get("changed") is False, failures, f"{row}: changed after stop quiet wait")


def validate_versions(failures: list[str]) -> dict[str, str | None]:
    package = load_json(ROOT / "panel/package.json", failures)
    lock = load_json(ROOT / "panel/package-lock.json", failures)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init_py = (ROOT / "vmlx_engine/__init__.py").read_text(encoding="utf-8")
    panel_version = package.get("version")
    lock_version = lock.get("version")
    lock_root_version = ((lock.get("packages") or {}).get("") or {}).get("version")
    require(panel_version == "1.5.65", failures, f"panel/package.json version={panel_version!r}")
    require(lock_version == "1.5.65", failures, f"panel/package-lock.json version={lock_version!r}")
    require(lock_root_version == "1.5.65", failures, f"panel/package-lock root version={lock_root_version!r}")
    require('version = "1.5.65"' in pyproject, failures, "pyproject.toml is not 1.5.65")
    require('__version__ = "1.5.65"' in init_py, failures, "vmlx_engine/__init__.py is not 1.5.65")
    return {
        "panel": panel_version,
        "panel_lock": lock_version,
        "panel_lock_root": lock_root_version,
        "pyproject": "1.5.65" if 'version = "1.5.65"' in pyproject else None,
        "engine": "1.5.65" if '__version__ = "1.5.65"' in init_py else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/current-scoped-release-preflight-65.json")
    args = parser.parse_args()

    failures: list[str] = []
    manifest: dict[str, Any] = {
        "scope": "1.5.65-mm3-gemma4",
        "status": "fail",
        "failures": failures,
        "versions": validate_versions(failures),
        "proofs": {},
        "boundary": {
            "gemma_audio": "out_of_current_scope_per_user; VL only",
            "gemma_mixed_swa_tq": "generic flat TQ-KV is required to stay off unless a native mixed-SWA TQ bridge is implemented and live-proven",
            "mm3_tq": "generic TQ-KV/storage quantization forced off for native MSA idx_keys",
        },
    }

    mm3_path = latest_mm3_stress(failures)
    if mm3_path:
        manifest["proofs"]["mm3_stress"] = str(mm3_path)
        validate_mm3_stress(load_pass(mm3_path, failures), failures)

    media_rows = latest_by_row(GEMMA_MEDIA_GLOB)
    missing_media = sorted(REQUIRED_GEMMA_MEDIA_ROWS - set(media_rows))
    if missing_media:
        failures.append(f"missing Gemma media/stress pass rows: {', '.join(missing_media)}")
    manifest["proofs"]["gemma_media"] = {
        row: str(media_rows[row])
        for row in sorted(REQUIRED_GEMMA_MEDIA_ROWS & set(media_rows))
    }
    for row, path in manifest["proofs"]["gemma_media"].items():
        validate_gemma_media(row, load_pass(Path(path), failures), failures)

    clean_rows = latest_by_row(CLEAN_START_GLOB)
    missing_clean = sorted(REQUIRED_CLEAN_START_ROWS - set(clean_rows))
    if missing_clean:
        failures.append(f"missing clean-start/autodetect pass rows: {', '.join(missing_clean)}")
    manifest["proofs"]["clean_start"] = {
        row: str(clean_rows[row])
        for row in sorted(REQUIRED_CLEAN_START_ROWS & set(clean_rows))
    }
    for row, path in manifest["proofs"]["clean_start"].items():
        validate_clean_start(row, load_pass(Path(path), failures), failures)

    lifecycle_rows = latest_by_row(LIFECYCLE_GLOB)
    missing_lifecycle = sorted(REQUIRED_LIFECYCLE_ROWS - set(lifecycle_rows))
    if missing_lifecycle:
        failures.append(f"missing lifecycle pass rows: {', '.join(missing_lifecycle)}")
    manifest["proofs"]["lifecycle"] = {
        row: str(lifecycle_rows[row])
        for row in sorted(REQUIRED_LIFECYCLE_ROWS & set(lifecycle_rows))
    }
    for row, path in manifest["proofs"]["lifecycle"].items():
        validate_lifecycle(row, load_pass(Path(path), failures), failures)

    manifest["status"] = "pass" if not failures else "fail"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"status={manifest['status']}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
