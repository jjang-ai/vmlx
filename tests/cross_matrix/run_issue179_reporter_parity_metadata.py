#!/usr/bin/env python3
"""Collect no-heavy reporter parity metadata for MiniMax-K issue #179.

This script is intended to run on the machine/session that reproduced #179. It
does not load the model. It packages the exact reporter-side hashes and
session/cancel lifecycle fields that the root-cause audit needs before it can
compare reporter state against the clean local installed-app proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_INSTALLED_SERVER = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/lib/"
    "python3.12/site-packages/vmlx_engine/server.py"
)
DEFAULT_MODEL_MANIFEST = Path(
    "build/current-issue179-minimax-k-local-model-manifest-20260527.json"
)
DEFAULT_OUT = Path("build/issue-179/reporter-parity-metadata-20260527.json")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_file_hashes(manifest: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(manifest, dict):
        return []
    rows: list[dict[str, str]] = []
    files = manifest.get("files")
    if not isinstance(files, list):
        return rows
    for row in files:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        sha = row.get("sha256")
        if isinstance(path, str) and isinstance(sha, str):
            rows.append({"path": path, "sha256": sha})
    return rows


def _session_settings_shape_ok(session: Any) -> bool:
    if not isinstance(session, dict):
        return False
    sampling = session.get("sampling")
    return (
        session.get("wireApi") == "responses"
        and session.get("detectedFamily") == "minimax"
        and session.get("route") == "/v1/responses"
        and session.get("stream") is True
        and session.get("sessionHasReasoningParser") is True
        and isinstance(sampling, dict)
        and sampling.get("temperature") == 1.0
        and sampling.get("top_p") == 0.95
        and sampling.get("top_k") == 40
        and sampling.get("max_tokens") == 4096
    )


def _raw_sse_cancel_lifecycle_shape_ok(lifecycle: Any) -> bool:
    if not isinstance(lifecycle, dict):
        return False
    request_error = lifecycle.get("request_error")
    return (
        isinstance(request_error, dict)
        and request_error.get("code") == "ECONNRESET"
        and request_error.get("fullContentLen") == 0
        and request_error.get("readerAcquired") is True
        and lifecycle.get("request_error_before_visible_content") is True
        and lifecycle.get("responses_cancel_404_after_econnreset_same_response_id")
        is True
        and lifecycle.get("responses_cancel_404_after_request_error") is True
    )


def _response_active_consistent(out: dict[str, Any]) -> bool:
    lifecycle = out.get("raw_sse_cancel_lifecycle")
    if not isinstance(lifecycle, dict):
        return False
    lifecycle_value = lifecycle.get("response_active_at_cancel")
    if lifecycle_value is None:
        return True
    return lifecycle_value == out.get("response_active_at_cancel")


def _missing_fields(out: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if out.get("capture_provenance") != "reporter_machine":
        missing.append("capture_provenance")
    if not out.get("installed_server_sha256"):
        missing.append("installed_server_sha256")
    if out.get("server_has_responses_cancel_route") is not True:
        missing.append("server_has_responses_cancel_route")
    if out.get("server_cancel_calls_engine_abort") is not True:
        missing.append("server_cancel_calls_engine_abort")
    if not out.get("model_manifest_sha256"):
        missing.append("model_manifest_sha256")
    if not out.get("model_file_hashes"):
        missing.append("model_file_hashes")
    if not out.get("chat_id"):
        missing.append("chat_id")
    if not isinstance(out.get("session_settings"), dict):
        missing.append("session_settings")
    elif not _session_settings_shape_ok(out.get("session_settings")):
        missing.append("session_settings_shape")
    if not out.get("response_id"):
        missing.append("response_id")
    if not isinstance(out.get("response_active_at_cancel"), bool):
        missing.append("response_active_at_cancel")
    if not isinstance(out.get("raw_sse_cancel_lifecycle"), dict):
        missing.append("raw_sse_cancel_lifecycle")
    elif not _raw_sse_cancel_lifecycle_shape_ok(out.get("raw_sse_cancel_lifecycle")):
        missing.append("raw_sse_cancel_lifecycle_shape")
    if (
        isinstance(out.get("response_active_at_cancel"), bool)
        and isinstance(out.get("raw_sse_cancel_lifecycle"), dict)
        and not _response_active_consistent(out)
    ):
        missing.append("response_active_at_cancel_consistency")
    return missing


def build_metadata(
    *,
    installed_server_path: Path,
    model_manifest_path: Path,
    capture_provenance: str,
    chat_id: str,
    session_settings_path: Path | None,
    response_id: str,
    response_active_at_cancel: bool | None,
    raw_sse_cancel_lifecycle_path: Path | None,
) -> dict[str, Any]:
    server = read_text(installed_server_path)
    manifest = read_json(model_manifest_path)
    session_settings = read_json(session_settings_path)
    raw_lifecycle = read_json(raw_sse_cancel_lifecycle_path)
    out: dict[str, Any] = {
        "installed_server_path": str(installed_server_path),
        "capture_provenance": capture_provenance or None,
        "installed_server_sha256": sha256_file(installed_server_path),
        "server_has_responses_cancel_route": (
            '@app.post("/v1/responses/{response_id}/cancel"' in server
        ),
        "server_cancel_calls_engine_abort": (
            "async def cancel_response" in server
            and "await _engine.abort_request(response_id)" in server
        ),
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "model_file_hashes": _model_file_hashes(manifest),
        "chat_id": chat_id or None,
        "session_settings": session_settings if isinstance(session_settings, dict) else None,
        "response_id": response_id or None,
        "response_active_at_cancel": response_active_at_cancel,
        "raw_sse_cancel_lifecycle": (
            raw_lifecycle if isinstance(raw_lifecycle, dict) else None
        ),
    }
    missing = _missing_fields(out)
    out["missing_fields"] = missing
    out["status"] = "pass" if not missing else "open"
    return out


def write_metadata(out_path: Path, metadata: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_response_active_at_cancel(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-server", type=Path, default=DEFAULT_INSTALLED_SERVER)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument(
        "--capture-provenance",
        default="",
        help="Must be reporter_machine for release parity; local templates stay open.",
    )
    parser.add_argument("--chat-id", default="")
    parser.add_argument("--session-settings-json", type=Path)
    parser.add_argument("--response-id", default="")
    parser.add_argument(
        "--response-active-at-cancel",
        type=_parse_response_active_at_cancel,
        default=None,
    )
    parser.add_argument("--raw-sse-cancel-lifecycle-json", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    metadata = build_metadata(
        installed_server_path=args.installed_server,
        model_manifest_path=args.model_manifest,
        capture_provenance=args.capture_provenance,
        chat_id=args.chat_id,
        session_settings_path=args.session_settings_json,
        response_id=args.response_id,
        response_active_at_cancel=args.response_active_at_cancel,
        raw_sse_cancel_lifecycle_path=args.raw_sse_cancel_lifecycle_json,
    )
    write_metadata(args.out, metadata)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": metadata["status"],
                "missing_fields": metadata["missing_fields"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
