#!/usr/bin/env python3
"""Fail-closed preflight for explicitly scoped public DMG builds.

The legacy release ledger still tracks broad historical family rows. For the
1.5.63 MiniMax-M3 + Gemma-VL release, this preflight validates the current
scoped evidence directly and writes a small auditable manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MM3_GATEWAY = ROOT / "build/live-clean-start-mm3-reap40-d3-real-profile-gateway-visible-off-2026-06-18T08-59-58-015Z/clean-start-proof.json"
GEMMA_E2B_GATEWAY = ROOT / "build/live-clean-start-gemma4-e2b-mxfp4-real-profile-gateway-vl-visible-off-2026-06-18T08-59-27-957Z/clean-start-proof.json"
MM3_LIFECYCLE = ROOT / "build/live-lifecycle-mm3-reap40-d3-lifecycle-2026-06-18T09-04-16-542Z/lifecycle-proof.json"
GEMMA_E2B_LIFECYCLE = ROOT / "build/live-lifecycle-gemma4-e2b-mxfp4-lifecycle-vl-2026-06-18T09-03-37-304Z/lifecycle-proof.json"

MM3_STRESS_GLOB = "build/live-mm3-stress-*/mm3-stress-proof.json"
GEMMA_MEDIA_GLOB = "build/live-gemma4-media-*/gemma4-media-proof.json"

REQUIRED_GEMMA_ROWS = {
    "gemma4-e2b-mxfp4",
    "gemma4-e4b-mxfp4",
    "gemma4-12b-mxfp4",
}

DEFERRED_GEMMA_ROWS_64 = {
    "gemma4-26b-mxfp4-visual",
    "gemma4-31b-mxfp4-visual",
}


def load_json(path: Path, failures: list[str]) -> dict:
    if not path.exists():
        failures.append(f"missing proof artifact: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid JSON artifact {path}: {exc}")
        return {}


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def pass_artifact(path: Path, failures: list[str]) -> dict:
    data = load_json(path, failures)
    require(data.get("status") == "pass", failures, f"{path} status={data.get('status')!r}")
    require(not data.get("failures"), failures, f"{path} has failures={data.get('failures')!r}")
    return data


def latest_pass_by_row(glob_pattern: str) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for path in sorted(ROOT.glob(glob_pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = str(data.get("rowName") or "")
        if data.get("status") == "pass" and row:
            rows[row] = path
    return rows


def validate_mm3_gateway(data: dict, failures: list[str]) -> None:
    cfg = data.get("sessionConfigAfterStart") or {}
    caps = data.get("capabilities") or {}
    native = ((caps.get("cache") or {}).get("native") or {})
    modalities = {str(item).lower() for item in caps.get("modalities") or []}
    gateway = data.get("gateway") or {}
    gw_status = gateway.get("status") or {}
    ui_score = ((data.get("defaultUiTurn") or {}).get("score") or {})
    cache_end = data.get("cacheEnd") or {}
    end_native = cache_end.get("native_cache") or {}
    disk_cache = cache_end.get("disk_cache") or {}

    require(data.get("expectedFamily") == "minimax_m3", failures, "MM3 gateway expectedFamily mismatch")
    require(data.get("profileMode") == "real", failures, "MM3 gateway was not real-profile")
    require(gw_status.get("running") is True and gw_status.get("port") == 8080, failures, "MM3 gateway default port not proven")
    require({"text", "vision"} <= modalities, failures, f"MM3 modalities missing text/vision: {sorted(modalities)}")
    require(cfg.get("toolCallParser") == "minimax_m3", failures, f"MM3 tool parser={cfg.get('toolCallParser')!r}")
    require(cfg.get("reasoningParser") == "minimax_m3", failures, f"MM3 reasoning parser={cfg.get('reasoningParser')!r}")
    require(cfg.get("isMultimodal") is True, failures, f"MM3 isMultimodal={cfg.get('isMultimodal')!r}")
    require(cfg.get("usePagedCache") is False, failures, f"MM3 usePagedCache={cfg.get('usePagedCache')!r}")
    require(cfg.get("enableDiskCache") is True, failures, f"MM3 enableDiskCache={cfg.get('enableDiskCache')!r}")
    require(cfg.get("kvCacheQuantization") == "auto", failures, f"MM3 kvCacheQuantization={cfg.get('kvCacheQuantization')!r}")
    require(cfg.get("enableJit") is False, failures, f"MM3 enableJit={cfg.get('enableJit')!r}")
    require(native.get("schema") == "minimax_m3_msa_v1", failures, f"MM3 capability native schema={native.get('schema')!r}")
    require(end_native.get("schema") == "minimax_m3_msa_v1", failures, f"MM3 cacheEnd native schema={end_native.get('schema')!r}")
    require((end_native.get("generic_turboquant_kv") or {}).get("enabled") is False, failures, "MM3 generic TQ-KV not disabled")
    require(end_native.get("prompt_disk_l2") is True, failures, "MM3 prompt SSD/L2 not proven")
    require((disk_cache.get("hits") or 0) > 0 and (disk_cache.get("stores") or 0) > 0, failures, "MM3 disk cache hit/store not proven")
    require(ui_score.get("empty") is False and ui_score.get("loopSuspect") is False, failures, "MM3 default UI turn empty/loop suspect")


def validate_gemma_gateway(data: dict, failures: list[str]) -> None:
    cfg = data.get("sessionConfigAfterStart") or {}
    caps = data.get("capabilities") or {}
    native = ((caps.get("cache") or {}).get("native") or {})
    modalities = {str(item).lower() for item in caps.get("modalities") or []}
    gateway = data.get("gateway") or {}
    gw_status = gateway.get("status") or {}
    defaults = caps.get("sampling_defaults") or {}
    ui_score = ((data.get("defaultUiTurn") or {}).get("score") or {})
    cache_end = data.get("cacheEnd") or {}
    end_native = cache_end.get("native_cache") or {}
    scheduler = cache_end.get("scheduler_stats") or {}

    require(data.get("expectedFamily") == "gemma4", failures, "Gemma gateway expectedFamily mismatch")
    require(data.get("profileMode") == "real", failures, "Gemma gateway was not real-profile")
    require(gw_status.get("running") is True and gw_status.get("port") == 8080, failures, "Gemma gateway default port not proven")
    require("text" in modalities and ("vision" in modalities or "image" in modalities), failures, f"Gemma modalities missing text/vision: {sorted(modalities)}")
    require(cfg.get("toolCallParser") == "gemma4", failures, f"Gemma tool parser={cfg.get('toolCallParser')!r}")
    require(cfg.get("reasoningParser") == "gemma4", failures, f"Gemma reasoning parser={cfg.get('reasoningParser')!r}")
    require(cfg.get("isMultimodal") is True, failures, f"Gemma isMultimodal={cfg.get('isMultimodal')!r}")
    require(cfg.get("usePagedCache") is False, failures, f"Gemma usePagedCache={cfg.get('usePagedCache')!r}")
    require(cfg.get("enableDiskCache") is True, failures, f"Gemma enableDiskCache={cfg.get('enableDiskCache')!r}")
    require(cfg.get("kvCacheQuantization") == "auto", failures, f"Gemma kvCacheQuantization={cfg.get('kvCacheQuantization')!r}")
    require(native.get("schema") == "mixed_swa_kv_v1", failures, f"Gemma capability native schema={native.get('schema')!r}")
    require(end_native.get("schema") == "mixed_swa_kv_v1", failures, f"Gemma cacheEnd native schema={end_native.get('schema')!r}")
    require(defaults.get("temperature") == 1 and defaults.get("top_p") == 0.95 and defaults.get("top_k") == 64, failures, f"Gemma generation defaults={defaults!r}")
    require((scheduler.get("cache_hit_tokens") or 0) > 0, failures, "Gemma cache hit tokens not proven")
    require(ui_score.get("empty") is False and ui_score.get("loopSuspect") is False, failures, "Gemma default UI turn empty/loop suspect")


def validate_lifecycle(data: dict, name: str, failures: list[str]) -> None:
    abort = data.get("abortTurn") or {}
    stop = data.get("stopTurn") or {}
    require(abort.get("streamingBeforeAbort") is True, failures, f"{name} abort did not catch active stream")
    require((abort.get("abortResult") or {}).get("success") is True, failures, f"{name} abort IPC failed")
    require((abort.get("notStreaming") or {}).get("streaming") is False, failures, f"{name} still streaming after abort")
    require((abort.get("quiet") or {}).get("changed") is False, failures, f"{name} changed after abort quiet wait")
    require(stop.get("streamingBeforeStop") is True, failures, f"{name} stop did not catch active stream")
    require((stop.get("stopResult") or {}).get("success") is True, failures, f"{name} stop IPC failed")
    require((stop.get("notStreaming") or {}).get("streaming") is False, failures, f"{name} still streaming after stop")
    require(((stop.get("sessionAfterStop") or {}).get("status") == "stopped"), failures, f"{name} session not stopped")
    require((stop.get("healthAfterStop") or {}).get("ok") is False, failures, f"{name} backend still healthy after stop")
    require((stop.get("quiet") or {}).get("changed") is False, failures, f"{name} changed after stop quiet wait")


def validate_media_rows(failures: list[str]) -> dict[str, str]:
    media_rows = latest_pass_by_row(GEMMA_MEDIA_GLOB)
    missing = sorted(REQUIRED_GEMMA_ROWS - set(media_rows))
    if missing:
        failures.append(f"missing current Gemma VL media pass artifacts for rows: {', '.join(missing)}")
    return {row: str(path) for row, path in sorted(media_rows.items()) if row in REQUIRED_GEMMA_ROWS}


def validate_mm3_stress(failures: list[str]) -> str | None:
    candidates = sorted(ROOT.glob(MM3_STRESS_GLOB))
    passing: list[Path] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "pass" and not data.get("failures"):
            passing.append(path)
    if not passing:
        failures.append("missing current MM3 stress pass artifact")
        return None
    return str(passing[-1])


def validate_versions(failures: list[str]) -> dict:
    package = load_json(ROOT / "panel/package.json", failures)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    init_py = (ROOT / "vmlx_engine/__init__.py").read_text(encoding="utf-8")
    version = package.get("version")
    require(version == "1.5.63", failures, f"panel/package.json version={version!r}")
    require('version = "1.5.63"' in pyproject, failures, "pyproject.toml is not 1.5.63")
    require('__version__ = "1.5.63"' in init_py, failures, "vmlx_engine/__init__.py is not 1.5.63")
    return {"panel": version, "pyproject": "1.5.63", "engine": "1.5.63"}


def validate_source_trace(failures: list[str]) -> dict:
    server = (ROOT / "vmlx_engine/server.py").read_text(encoding="utf-8")
    require('return ["text", "vision"]' in server and "_m3_vl_image_ok" in server, failures, "MM3 VL capability source fix missing")
    return {"m3_vl_capability": "vmlx_engine/server.py::_loaded_runtime_modalities"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="mm3_gemma_vl", choices=["mm3_gemma_vl"])
    parser.add_argument("--out", type=Path, default=ROOT / "build/current-scoped-release-preflight-mm3-gemma-vl.json")
    args = parser.parse_args()

    failures: list[str] = []
    mm3_gateway = pass_artifact(MM3_GATEWAY, failures)
    gemma_gateway = pass_artifact(GEMMA_E2B_GATEWAY, failures)
    mm3_lifecycle = pass_artifact(MM3_LIFECYCLE, failures)
    gemma_lifecycle = pass_artifact(GEMMA_E2B_LIFECYCLE, failures)

    if mm3_gateway:
        validate_mm3_gateway(mm3_gateway, failures)
    if gemma_gateway:
        validate_gemma_gateway(gemma_gateway, failures)
    if mm3_lifecycle:
        validate_lifecycle(mm3_lifecycle, "MM3", failures)
    if gemma_lifecycle:
        validate_lifecycle(gemma_lifecycle, "Gemma", failures)

    manifest = {
        "scope": args.scope,
        "status": "pass",
        "failures": failures,
        "versions": validate_versions(failures),
        "source_trace": validate_source_trace(failures),
        "required_proofs": {
            "mm3_gateway": str(MM3_GATEWAY),
            "gemma_e2b_gateway": str(GEMMA_E2B_GATEWAY),
            "mm3_lifecycle": str(MM3_LIFECYCLE),
            "gemma_e2b_lifecycle": str(GEMMA_E2B_LIFECYCLE),
        },
        "mm3_stress_proof": validate_mm3_stress(failures),
        "gemma_vl_media_proofs": validate_media_rows(failures),
        "deferred_to_1_5_64": {
            "gemma_rows": sorted(DEFERRED_GEMMA_ROWS_64),
            "note": "26B/31B visual rows and any remaining red/broad matrix rows are deferred to 1.5.64 per ASAP 1.5.63 release scope.",
        },
        "scope_note": "Gemma audio is intentionally excluded from 1.5.63 scope; scoped Gemma rows require VL/text/reasoning/tools/API/cache only.",
    }
    manifest["status"] = "pass" if not failures else "fail"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"scope={args.scope}")
    print(f"status={manifest['status']}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
