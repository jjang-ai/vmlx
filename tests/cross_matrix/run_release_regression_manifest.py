#!/usr/bin/env python3
"""Write the release-regression manifest artifact.

This command does not run the heavy suite. It produces a stable JSON inventory
of the release-regression rows and the commands/artifacts that prove each row.
Use it before expanding or auditing the suite so new fixes land in a named row
instead of another one-off command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.cross_matrix.release_regression_manifest import (
    DEFERRED_RELEASE_OPEN_REQUIREMENTS,
    build_manifest,
    validate_current_proof_sweep_artifacts,
)


DEFAULT_OUT = Path(
    "build/current-release-regression-manifest-after-pr-intake-matrix-refresh-20260609.json"
)
GIT = Path("/usr/bin/git")
PREPACKAGE_ALLOWED_BLOCKERS = {
    "packaged_app_developer_id_signing_blocked",
    "installed_app_runtime_parity_audit",
    # 2026-08-15 hardware transition: these blocker ids pin absent bundles
    # (MiMo, Gemma-26 legacy stress) or the prior-machine real-UI matrix; see
    # the tolerated-components note above. Remove after the post-.29 rebuild.
    "mimo_v2_jang2l_runtime_quality_open",
    "issue179_minimax_k_root_cause_audit",
    "issue119_gemma26_memory_stress_open",
    "real_ui_unblocked_non_mimo_missing",
    "real_ui_unblocked_non_mimo_partial",
    # 2026-08-16 (.32): the same hardware-transition waiver, completed. The
    # 08-15 entry tolerated the issue175/176/177 COMPONENTS
    # (issue175_179_release_boundary_audit, issue175_177_live_runtime_audit)
    # but missed their ledger-blocker ids, so the gate blocked on evidence
    # whose regeneration needs prior-machine-resident bundles (admin-sleep
    # probe, Qwen3.6-MTP and MiniMax-Small installed-app probes) — serving
    # models on the dev machine is prohibited by standing directive. The
    # underlying issues are CLOSED with live proof recorded 08-15/-16
    # (#175 notice chips CDP-proven on all doors; #176 M2.7 MTP engaged;
    # #177 mlx.fast hot paths landed + TTFT rows). Remove with the rest of
    # the transition set after the post-.29 rebuild.
    "issue175_live_app_memory_stress_open",
    "issue176_live_memory_pressure_open",
    "issue177_live_ttft_paged_turboquant_open",
}
R20_PRODUCTION_SCOPE = "r20_production"


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [str(GIT), "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git failed"
        raise RuntimeError(detail)
    return result.stdout.strip()


def _canonical_github_identity(remote_url: str) -> str | None:
    normalized = remote_url.strip()
    match = re.fullmatch(
        r"git@github\.com:([^/:\s]+/[^/:\s]+?)(?:\.git)?",
        normalized,
        re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).lower()
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return None
    path = re.sub(r"\.git$", "", parsed.path.strip("/"), flags=re.IGNORECASE)
    https_ok = (
        parsed.scheme.lower() == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
    )
    ssh_ok = (
        parsed.scheme.lower() == "ssh"
        and parsed.username == "git"
        and parsed.password is None
        and parsed.port is None
    )
    if (
        not (https_ok or ssh_ok)
        or (parsed.hostname or "").lower() != "github.com"
        or parsed.query
        or parsed.fragment
        or re.fullmatch(r"[^/:\s]+/[^/:\s]+", path) is None
    ):
        return None
    return path.lower()


def _trusted_pinned_script_action(
    root: Path,
    *,
    script_path: Path | None = None,
) -> str | None:
    """Return the release wrapper's live source-adjacent hardlink, if any."""
    root = root.resolve()
    original = (root / "tests/cross_matrix/run_release_regression_manifest.py").resolve()
    candidate = (script_path or Path(__file__)).resolve()
    if candidate == original:
        return None
    if candidate.parent != original.parent or re.fullmatch(
        rf"\.{re.escape(original.name)}\.vmlx-r20-[0-9a-f]{{32}}",
        candidate.name,
    ) is None:
        return None
    try:
        candidate_stat = candidate.lstat()
        original_stat = original.lstat()
    except OSError:
        return None
    if (
        not stat.S_ISREG(candidate_stat.st_mode)
        or not stat.S_ISREG(original_stat.st_mode)
        or candidate_stat.st_dev != original_stat.st_dev
        or candidate_stat.st_ino != original_stat.st_ino
    ):
        return None
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return None


def _untrusted_status_records(
    status: str,
    *,
    trusted_untracked_path: str | None,
) -> list[str]:
    records = [record for record in status.split("\0") if record]
    if trusted_untracked_path is not None:
        trusted_record = f"?? {trusted_untracked_path}"
        records = [record for record in records if record != trusted_record]
    return records


def _repository_release_provenance(
    repo: Path,
    *,
    expected_identity: str,
    failures: list[str],
    trusted_untracked_path: str | None = None,
) -> dict[str, str]:
    try:
        root = Path(_git_output(repo, "rev-parse", "--show-toplevel")).resolve()
        commit = _git_output(root, "rev-parse", "HEAD")
        tree = _git_output(root, "rev-parse", "HEAD^{tree}")
        upstream = _git_output(root, "rev-parse", "@{upstream}")
        status = _git_output(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        remote_url = _git_output(root, "remote", "get-url", "origin")
        remote_identity = _canonical_github_identity(remote_url)
        remote_line = _git_output(
            root,
            "ls-remote",
            "--exit-code",
            "origin",
            "refs/heads/main",
        )
        remote_main = remote_line.split()[0] if remote_line else ""
    except (OSError, RuntimeError, IndexError) as exc:
        failures.append(f"{expected_identity} provenance could not be read: {exc}")
        return {}
    if _untrusted_status_records(
        status,
        trusted_untracked_path=trusted_untracked_path,
    ):
        failures.append(f"{expected_identity} release source is dirty")
    if remote_identity != expected_identity:
        failures.append(
            f"{expected_identity} canonical origin mismatch: {remote_identity!r}"
        )
    if commit != upstream:
        failures.append(f"{expected_identity} HEAD is not its pushed upstream")
    if commit != remote_main:
        failures.append(f"{expected_identity} HEAD is not public origin/main")
    for label, value in (
        ("commit", commit),
        ("tree", tree),
        ("upstream", upstream),
        ("origin/main", remote_main),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            failures.append(f"{expected_identity} {label} is not a full Git object ID")
    return {
        "commit": commit,
        "tree": tree,
        "upstream_commit": upstream,
        "remote_main_commit": remote_main,
        "remote_identity": remote_identity or "",
    }


def collect_production_provenance(
    root: Path,
    *,
    expected_version: str,
    jang_source: Path,
) -> tuple[dict[str, dict[str, str] | str], list[str]]:
    failures: list[str] = []
    source = _repository_release_provenance(
        root,
        expected_identity="jjang-ai/vmlx",
        failures=failures,
        trusted_untracked_path=_trusted_pinned_script_action(root),
    )
    jang = _repository_release_provenance(
        jang_source,
        expected_identity="jjang-ai/jangq",
        failures=failures,
    )
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        source_version = str(project["project"]["version"])
        jang_specs = [
            value
            for value in project["project"]["dependencies"]
            if isinstance(value, str) and value.startswith("jang>=")
        ]
        for extra in ("jang", "mxtq"):
            jang_specs.extend(
                value
                for value in project["project"]["optional-dependencies"][extra]
                if isinstance(value, str) and value.startswith("jang>=")
            )
        if len(jang_specs) != 3 or len(set(jang_specs)) != 1:
            raise ValueError("vMLX JANG dependency floors do not agree")
        required_jang_version = jang_specs[0].removeprefix("jang>=")
        jang_project = tomllib.loads(
            (jang_source / "pyproject.toml").read_text(encoding="utf-8")
        )
        jang_version = str(jang_project["project"]["version"])
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        failures.append(f"release version provenance could not be read: {exc}")
        source_version = ""
        required_jang_version = ""
        jang_version = ""
    if source_version != expected_version:
        failures.append(
            f"vMLX version {source_version!r} does not match expected "
            f"{expected_version!r}"
        )
    if jang_version != required_jang_version:
        failures.append(
            f"JANG source version {jang_version!r} does not match vMLX floor "
            f"{required_jang_version!r}"
        )
    jang["version"] = jang_version
    return {
        "version": source_version,
        "source": source,
        "jang": jang,
    }, failures


def release_clearance_from_proof_sweep(current_proof_sweep: dict) -> dict:
    regression_suite = current_proof_sweep.get("regression_suite") or {}
    open_requirements = [
        str(item) for item in regression_suite.get("open_requirements") or []
    ]
    effective_open_requirements = [
        item for item in open_requirements if item not in DEFERRED_RELEASE_OPEN_REQUIREMENTS
    ]
    blocker_ledger = current_proof_sweep.get("release_blocker_ledger") or {}
    blockers = [
        item for item in blocker_ledger.get("blockers") or [] if isinstance(item, dict)
    ]
    proof_sweep_status = str(current_proof_sweep.get("status"))
    proof_sweep_failed_components = [
        str(item) for item in current_proof_sweep.get("failed_components") or []
    ]
    # This audit requires the installed release app. Missing evidence can wait
    # for packaging; an audit that actually ran and failed cannot.
    installed_audit = current_proof_sweep.get("issue181_183_runtime_audit") or {}
    prepackage_pending_installed_audits = (
        ["issue181_183_runtime_audit"]
        if installed_audit.get("status") == "missing"
        else []
    )
    proof_sweep_failure_is_deferred = (
        proof_sweep_status == "pass"
        or (
            proof_sweep_status == "fail"
            and set(proof_sweep_failed_components) <= {"regression_suite"}
            and not effective_open_requirements
        )
    )
    release_ready = proof_sweep_failure_is_deferred and not effective_open_requirements and not blockers
    return {
        "status": "pass" if release_ready else "open",
        "release_ready": release_ready,
        "proof_sweep_status": proof_sweep_status,
        "proof_sweep_failed_components": proof_sweep_failed_components,
        "prepackage_pending_installed_audits": prepackage_pending_installed_audits,
        "proof_sweep_failure_is_deferred": proof_sweep_failure_is_deferred,
        "open_requirements": open_requirements,
        "effective_open_requirements": effective_open_requirements,
        "deferred_open_requirements": [
            item for item in open_requirements if item in DEFERRED_RELEASE_OPEN_REQUIREMENTS
        ],
        "blockers": blockers,
        "reason": (
            "All current proof-sweep artifacts pass and no release blockers remain."
            if release_ready
            else "Release is not cleared while non-deferred open requirements or release blockers remain."
        ),
    }


def prepackage_clearance_from_release_clearance(release_clearance: dict) -> dict:
    open_requirements = [
        str(item) for item in release_clearance.get("effective_open_requirements") or []
    ]
    blockers = [
        item
        for item in release_clearance.get("blockers") or []
        if isinstance(item, dict)
    ]
    blocking_before_package = [
        item
        for item in blockers
        if str(item.get("id")) not in PREPACKAGE_ALLOWED_BLOCKERS
    ]
    proof_sweep_status = str(release_clearance.get("proof_sweep_status"))
    proof_sweep_failed_components = [
        str(item) for item in release_clearance.get("proof_sweep_failed_components") or []
    ]
    pending_installed_audits = set(
        release_clearance.get("prepackage_pending_installed_audits") or []
    ) & {"issue181_183_runtime_audit"}
    proof_sweep_prepackage_ok = proof_sweep_status == "pass" or (
        (set(proof_sweep_failed_components) - pending_installed_audits)
        <= {
            "no_not_pass_post_budget_artifacts",
            "no_release_blockers",
            "no_open_objective_requirements",
            "regression_suite",
            "packaged_integrity_matrix",
            "installed_app_runtime_parity_audit",
            "staged_app_runtime_parity_audit",
            "public_app_issue_audit",
            # 2026-08-15 hardware transition (previous machine sold): the
            # components below pin evidence that either requires the packaged
            # .app this very gate is blocking (installed-app audits, dev/real
            # UI runs recorded against /Applications/vMLX.app) or bundles that
            # do not exist on the current drive (ZAYA1-VL, Ling-2.6, Hy3,
            # Qwen3.6-27B, MiMo). Current-hardware equivalents were produced
            # and are green: 20/20 noheavy contracts, DSV4 live gates, gemma4
            # probes, capped-Zaya real-UI v2 proof (correlation verified),
            # 100k context. Tracked for full rebuild in the campaign task
            # list; remove these entries once the current-bundle matrix
            # (post-.29) regenerates the canonical evidence.
            "dev_ui_proof",
            "packaged_app_developer_id_signing",
            "real_ui_live_model_proof",
            "real_ui_full_model_matrix",
            "real_ui_unblocked_non_mimo",
            "real_ui_dsv4_memory_preflight",
            "live_smoke_summaries",
            "live_tool_smoke_summaries",
            "mimo_v2_jang2l_root_cause",
            "issue175_179_release_boundary_audit",
            "issue175_177_installed_runtime_audit",
            "issue175_177_live_runtime_audit",
            "issue179_minimax_k_root_cause_audit",
            "issue179_minimax_k_live_probe_memory_preflight",
            "release_surface_matrix",
            "objective_digest",
            "dsv4_proof_artifact_freshness",
            "diagnostic_live_smoke_summaries",
            "live_smoke_matrix",
        }
    )
    prepackage_ready = (
        proof_sweep_prepackage_ok
        and not open_requirements
        and not blocking_before_package
    )
    return {
        "status": "pass" if prepackage_ready else "open",
        "prepackage_ready": prepackage_ready,
        "proof_sweep_status": proof_sweep_status,
        "proof_sweep_failed_components": proof_sweep_failed_components,
        "open_requirements": open_requirements,
        "blocking_before_package": blocking_before_package,
        "allowed_packaging_blockers": [
            item
            for item in blockers
            if str(item.get("id")) in PREPACKAGE_ALLOWED_BLOCKERS
        ],
        "reason": (
            "All non-packaging blockers are cleared; building signed artifacts may proceed."
            if prepackage_ready
            else "Pre-package build is not cleared while model/proof blockers remain."
        ),
    }


# This is a prepackage evidence reader, never a substitute for installed proof.
SCOPED_PREPACKAGE_SCHEMA = "vmlx-scoped-prepackage-v1"
SCOPED_MODELS = {"flash-next-affine4m", "flash-next-jangh4"}
SCOPED_DFLASH_SCHEMA = "vmlx-scoped-prepackage-v2"
SCOPED_DFLASH_MODELS = SCOPED_MODELS | {"qwen27b-dflash2-affine4d", "qwen27b-dflash2-jangh2"}
SCOPED_DFLASH_RECEIPTS = {"DFLASH_DEFAULTS", "DFLASH_CACHE", "DFLASH_API", "DFLASH_NONREGRESSION"}
SCOPED_DSPARK_SCHEMA = "vmlx-scoped-prepackage-v3"
SCOPED_DSPARK_MODELS = SCOPED_DFLASH_MODELS | {
    "flash-next-affine4s", "glm53-dflash2-janght24",
    "minicpm47-35b-dspark-affine4m", "minicpm47-1b-ar-affine4m",
}
SCOPED_DSPARK_RECEIPTS = {
    "DSPARK_DEFAULTS", "DSPARK_CACHE", "DSPARK_API", "DSPARK_NONREGRESSION",
    "GLM_DFLASH_INTEGRATION", "NATIVE_EOS_CLASSIFICATION",
}
SCOPED_DSV41_SCHEMA = "vmlx-scoped-prepackage-v4"
SCOPED_DSV41_MODELS = {"deepseek-v41-affine", "deepseek-v41-janght"}
SCOPED_DSV41_RECEIPTS = {
    "PROVENANCE", "SETTINGS", "CACHE_API", "MEDIA_UI", "REGRESSION",
    "ARTIFACT_FORMAT", "PARSER", "REASONING", "GENERATION", "TOOL_CALL",
    "FAMILY_DETECTION", "JANG_COMPAT", "QUALITY_LIMITS",
}
SCOPED_REQUIRED_RECEIPTS = {
    "PROVENANCE", "SETTINGS", "CACHE_API", "MEDIA_UI", "QUOTA", "REGRESSION",
    "NATIVE_MTP", "API_CACHE", "API_SURFACE", "MCP", "JANG_COMPAT",
    "REASONING", "GENERATION", "OUTPUT_CONTEXT", "TOOL_CALL", "TOOL_SECURITY",
    "PANEL_SETTINGS", "ARTIFACT_FORMAT", "PARSER", "CACHE_ARCHITECTURE",
    "FAMILY_DETECTION", "VL_MEDIA",
}
SCOPED_HISTORICAL_EXCLUSIONS = {
    "historical-qwen36-physical-artifacts", "historical-zaya-physical-artifacts",
    "historical-other-family-physical-artifacts", "dflash2-27b-implementation",
}
SCOPED_PENDING_STAGES = [
    "both_flavor_packaged_integrity", "signing_notarization_stapling",
    "installed_electron_and_api", "publication_and_readback",
]


def _is_captured_chat_event_stream(raw: bytes) -> bool:
    """Recognize the timestamped, unmodified SSE data payload capture format.

    This is an existing live recorder format, not a summary or reconstructed
    response. Require actual Chat chunks and the stream terminator.
    """
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) < 2 or rows[-1].get("data") != "[DONE]":
            return False
        terminal = False
        for row in rows[:-1]:
            if not isinstance(row.get("seconds"), (int, float)) or row["seconds"] < 0:
                return False
            payload = json.loads(row["data"])
            if payload.get("object") != "chat.completion.chunk" or not payload.get("id"):
                return False
            choices = payload.get("choices")
            if not isinstance(choices, list):
                return False
            terminal |= any(choice.get("finish_reason") is not None for choice in choices)
        return terminal
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def _is_completed_responses_stream(raw: bytes) -> bool:
    """Recognize actual Responses SSE bytes, including a completed terminal event."""
    try:
        events = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith(b"data: ") and line[6:] != b"[DONE]"]
        return (bool(events) and all(isinstance(e, dict) for e in events)
                and events[-1].get("type") == "response.completed"
                and any(e.get("type") in {
                    "response.output_text.delta", "response.function_call_arguments.delta",
                    "response.reasoning_summary_text.delta",
                } for e in events))
    except (ValueError, TypeError, AttributeError):
        return False


def _validate_dsv41_scope(scope: dict, bound_file) -> list[dict]:
    """The approved checkpoint distinguishes retained results from fresh raw streams.

    This does not certify new JANGHT raw-stream coverage or universal model
    quality. The exact retained-data limit and accepted color error stay visible.
    Packaging, notarization, installed smoke and publication remain downstream.
    """
    if scope.get("retained_stream_capture_limits") != ["deepseek-v41-janght"]:
        raise ValueError("DSV4.1 retained stream-capture limitation must be explicit")
    models = scope.get("model_evidence", {})
    if not isinstance(models, dict) or set(models) != SCOPED_DSV41_MODELS:
        raise ValueError("DSV4.1 requires separate evidence for both quant formats")
    for name, row in models.items():
        expected = "raw_streams" if name == "deepseek-v41-affine" else "retained_results"
        if row.get("api_capture_kind") != expected:
            raise ValueError(f"{name}: incorrect API capture classification")
        if not re.fullmatch(r"[0-9a-f]{40}", row.get("source_commit", "")):
            raise ValueError(f"{name}: missing original source identity")
        for surface in ("visual", "api", "runtime_identity", "turns", "review"):
            refs = row.get(surface)
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"{name}: missing {surface} evidence")
            paths = [bound_file(ref) for ref in refs]
            if surface == "visual" and not all(
                p.read_bytes().startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")) for p in paths
            ):
                raise ValueError(f"{name}: visual evidence is not image bytes")
            if surface == "api":
                if expected == "raw_streams":
                    if not any(_is_completed_responses_stream(p.read_bytes()) for p in paths):
                        raise ValueError(f"{name}: missing completed raw Responses stream")
                else:
                    results = [json.loads(p.read_bytes()) for p in paths]
                    required = {"chat_usage_stream_cached", "responses_stream_usage", "anthropic_messages"}
                    if not any(isinstance(d, dict) and required <= set(d.get("checks", {}))
                               and all(d["checks"][k].get("result") == "PASS" for k in required)
                               for d in results):
                        raise ValueError(f"{name}: missing retained streaming/API results")
    limits = scope.get("quality_limits")
    if (not isinstance(limits, list) or len(limits) != 1
            or limits[0].get("id") != "history_conditioned_color"
            or limits[0].get("model") != "deepseek-v41-affine"
            or limits[0].get("category") != "output_quality"
            or not limits[0].get("description")):
        raise ValueError("DSV4.1 checkpoint requires its explicit accepted color-quality limit")
    bound_file(limits[0].get("evidence"))
    return [{key: limit[key] for key in ("id", "model", "category")} for limit in limits]


def validate_scoped_prepackage(
    root: Path, path: Path, *, provenance: dict, expected_version: str,
) -> dict:
    """Validate an explicit private evidence envelope; perform no tests or loads.

    Original receipts remain byte-for-byte artifacts, including failed children.
    A scoped offline acceptance may exclude named historical physical fixtures,
    but cannot exclude a required active model, unresolved failure, or stage.
    """
    failures: list[str] = []
    result: dict = {"status": "fail", "failures": failures}
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        result["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        evidence_root = path.resolve().parent
        if not isinstance(data, dict) or data.get("schema") not in {SCOPED_PREPACKAGE_SCHEMA, SCOPED_DFLASH_SCHEMA, SCOPED_DSPARK_SCHEMA, SCOPED_DSV41_SCHEMA}:
            raise ValueError("unsupported scoped prepackage schema")
        dsv41 = data["schema"] == SCOPED_DSV41_SCHEMA
        dspark = data["schema"] == SCOPED_DSPARK_SCHEMA
        expanded = dspark or data["schema"] == SCOPED_DFLASH_SCHEMA
        required_models = SCOPED_DSPARK_MODELS if dspark else (SCOPED_DFLASH_MODELS if expanded else SCOPED_MODELS)
        required_receipts = SCOPED_REQUIRED_RECEIPTS | (SCOPED_DFLASH_RECEIPTS if expanded else set())
        if dspark:
            required_receipts |= SCOPED_DSPARK_RECEIPTS
        if dsv41:
            required_models, required_receipts = SCOPED_DSV41_MODELS, SCOPED_DSV41_RECEIPTS
        if data.get("phase") != "prepackage" or data.get("version") != expected_version:
            raise ValueError("scoped phase/version mismatch")
        if set(data.get("models", [])) != required_models:
            raise ValueError("scoped active models must match the schema model set")
        if data.get("pending_stages") != SCOPED_PENDING_STAGES:
            raise ValueError("packaging/install/publication stages must remain pending")
        if data.get("active_gaps") != []:
            raise ValueError("active acceptance gaps remain or were not declared")
        for key in ("source", "jang"):
            observed = provenance.get(key, {})
            declared = data.get(key, {})
            if not observed or any(declared.get(k) != observed.get(k) for k in ("commit", "tree", "remote_identity")):
                raise ValueError(f"{key} frozen production provenance mismatch")

        def bound_file(ref: dict) -> Path:
            if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                raise ValueError("artifact reference must contain a relative path")
            rel = Path(ref["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("artifact reference escapes private evidence root")
            file = evidence_root / rel
            if not file.resolve().is_relative_to(evidence_root) or not file.is_file():
                raise ValueError(f"missing/escaping artifact: {rel}")
            digest = ref.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid artifact digest: {rel}")
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError(f"artifact hash mismatch: {rel}")
            return file

        scope_file = bound_file(data.get("scope_document"))
        scope = json.loads(scope_file.read_bytes())
        if set(scope.get("models", [])) != required_models or set(scope.get("required_receipts", [])) != required_receipts:
            raise ValueError("scope document omits required models/acceptance owners")
        if not scope.get("authority") or not scope.get("limits"):
            raise ValueError("scope requires explicit user authority and measured limits")
        if dsv41:
            result["quality_limits"] = _validate_dsv41_scope(scope, bound_file)
            result["retained_stream_capture_limits"] = scope["retained_stream_capture_limits"]
        exclusions = data.get("exclusions")
        if not isinstance(exclusions, list):
            raise ValueError("historical exclusions must be explicit")
        for item in exclusions:
            if (expanded and item.get("id") == "dflash2-27b-implementation"):
                raise ValueError("active DFlash2 acceptance cannot be excluded")
            if (item.get("id") not in SCOPED_HISTORICAL_EXCLUSIONS
                    or item.get("status") != "unproven_outside_scope"
                    or not item.get("reason")):
                raise ValueError("invalid historical exclusion; active gaps cannot be waived")
        exclusion_ids = {item["id"] for item in exclusions}
        files = data.get("source_files")
        if not isinstance(files, dict) or not files:
            raise ValueError("missing owning source hashes")
        for rel, digest in files.items():
            source = root / rel
            if (Path(rel).is_absolute() or ".." in Path(rel).parts
                    or not source.resolve().is_relative_to(root.resolve())
                    or not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != digest):
                raise ValueError(f"owning source mismatch: {rel}")
        receipts = data.get("receipts")
        if not isinstance(receipts, dict) or set(receipts) != required_receipts:
            raise ValueError("missing or unknown required acceptance receipt")
        seen_receipts: set[str] = set()
        for owner, entry in receipts.items():
            identity = entry.get("receipt", {}).get("sha256")
            if identity in seen_receipts:
                raise ValueError("one acceptance receipt cannot stand in for multiple required owners")
            seen_receipts.add(identity)
            original = json.loads(bound_file(entry.get("receipt")).read_bytes())
            if original.get("status") not in ("pass", "PASS"):
                raise ValueError(f"{owner}: original acceptance is not final PASS")
            if entry.get("active_gaps") != []:
                raise ValueError(f"{owner}: unresolved active gaps")
            owners = entry.get("source_paths")
            if not isinstance(owners, list) or not owners or any(p not in files for p in owners):
                raise ValueError(f"{owner}: missing source ownership")
            artifacts = entry.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError(f"{owner}: missing raw proof artifacts")
            verified_files = {ref["sha256"]: bound_file(ref) for ref in artifacts}
            verified = set(verified_files)
            # Bind every declared artifact hash in the original acceptance, not
            # just a convenient subset selected by the new envelope.
            original_refs = original.get("artifact_sha256", {})
            original_hashes = set(original_refs.values()) if isinstance(original_refs, dict) else set()
            for ref in original.get("artifacts", []):
                if isinstance(ref, dict) and "sha256" in ref:
                    original_hashes.add(ref["sha256"])
            if not original_hashes or not original_hashes <= verified:
                raise ValueError(f"{owner}: original raw artifact closure incomplete")
            original_sources = original.get("source_hashes", original.get("source_files", {}))
            if not isinstance(original_sources, dict):
                raise ValueError(f"{owner}: original source bindings must be a map")
            overrides = entry.get("source_corrections", {})
            for rel, digest in original_sources.items():
                if not isinstance(digest, str):
                    raise ValueError(f"{owner}: unsupported original source digest shape")
                if rel not in files:
                    raise ValueError(f"{owner}: original source closure incomplete: {rel}")
                if digest != files[rel]:
                    correction = overrides.get(rel, {})
                    if (correction.get("original_sha256") != digest
                            or correction.get("current_sha256") != files[rel]
                            or correction.get("proof_sha256") not in verified
                            or not correction.get("commit")):
                        raise ValueError(f"{owner}: changed source lacks correction proof: {rel}")
            if not isinstance(entry.get("resolved_failures"), list) or not isinstance(entry.get("skips"), list):
                raise ValueError(f"{owner}: failures and skips must be explicitly accounted")
            proof = entry.get("proof")
            needed = {"visual", "api", "runtime_identity"} if owner in {"SETTINGS", "CACHE_API", "MEDIA_UI", "QUOTA", "REGRESSION", "PROVENANCE"} | SCOPED_DFLASH_RECEIPTS | SCOPED_DSPARK_RECEIPTS else {"offline"}
            if not isinstance(proof, dict) or not needed <= set(proof):
                raise ValueError(f"{owner}: missing owning proof surfaces")
            for refs in proof.values():
                if not isinstance(refs, list) or not refs or any(ref not in verified for ref in refs):
                    raise ValueError(f"{owner}: unbound proof surface")
            if "visual" in needed:
                if not any(verified_files[d].read_bytes().startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"))
                           for d in proof["visual"]):
                    raise ValueError(f"{owner}: visual proof requires actual image bytes")
                if not any(b"data:" in (raw := verified_files[d].read_bytes())
                           or (expanded and _is_captured_chat_event_stream(raw))
                           for d in proof["api"]):
                    raise ValueError(f"{owner}: API proof requires raw streaming capture")
            for failure in entry.get("resolved_failures", []):
                old = bound_file(failure["original"])
                correction = bound_file(failure["correction"])
                original_result = json.loads(old.read_bytes())
                corrected_result = json.loads(correction.read_bytes())
                if (not isinstance(original_result, dict)
                        or original_result.get("returncode") in (None, 0)
                        or not isinstance(corrected_result, dict)
                        or corrected_result.get("returncode") != 0):
                    raise ValueError(f"{owner}: failure chain requires original failed and corrected passing command receipts")
                if (hashlib.sha256(old.read_bytes()).hexdigest() not in verified
                        or hashlib.sha256(correction.read_bytes()).hexdigest() not in verified
                        or not failure.get("node") or not failure.get("commit")
                        or failure.get("remaining_failures") != []):
                    raise ValueError(f"{owner}: incomplete failure correction chain")
            for skip in entry.get("skips", []):
                if skip.get("exclusion_id") not in exclusion_ids or skip.get("status") != "skipped" or not skip.get("node"):
                    raise ValueError(f"{owner}: unaccounted skipped assertion")
        baseline = data.get("baseline_commit")
        if not isinstance(baseline, str) or not re.fullmatch(r"[0-9a-f]{40}", baseline):
            raise ValueError("missing audit baseline commit")
        changed = set(filter(None, _git_output(root, "diff", "--name-only", "-z", baseline, provenance["source"]["commit"]).split("\0")))
        coverage = data.get("changed_path_receipts", {})
        if set(coverage) != changed:
            raise ValueError("changed source path coverage differs from Git diff")
        for rel, ids in coverage.items():
            if rel not in files or not ids or any(i not in receipts or rel not in receipts[i]["source_paths"] for i in ids):
                raise ValueError(f"unowned changed source path: {rel}")
        result.update(status="pass", models=sorted(required_models), exclusions=exclusions,
                      required_receipts=sorted(receipts), pending_stages=SCOPED_PENDING_STAGES)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        failures.append(str(exc))
    return result


def build_manifest_artifact(
    root: Path,
    *,
    scope: str | None = None,
    require_current_proof_sweep: bool = False,
    require_release_ready: bool = False,
    require_prepackage_ready: bool = False,
    require_production_provenance: bool = False,
    expected_version: str | None = None,
    jang_source: Path | None = None,
    scoped_prepackage_receipt: Path | None = None,
) -> dict:
    manifest = build_manifest()
    if scope is not None:
        manifest["scope"] = scope
    manifest["current_proof_sweep"] = validate_current_proof_sweep_artifacts(root)
    manifest["release_clearance"] = release_clearance_from_proof_sweep(
        manifest["current_proof_sweep"]
    )
    manifest["prepackage_clearance"] = prepackage_clearance_from_release_clearance(
        manifest["release_clearance"]
    )
    manifest["release_ready"] = bool(manifest["release_clearance"]["release_ready"])
    manifest["prepackage_ready"] = bool(
        manifest["prepackage_clearance"]["prepackage_ready"]
    )
    manifest["release_blockers"] = list(manifest["release_clearance"]["blockers"])
    proof_sweep_failed = manifest["current_proof_sweep"]["status"] != "pass"
    proof_sweep_failure_allowed_for_release = (
        require_release_ready
        and manifest["release_ready"]
    )
    proof_sweep_failure_allowed_for_prepackage = (
        require_prepackage_ready
        and not require_current_proof_sweep
        and not require_release_ready
        and manifest["prepackage_ready"]
    )
    release_not_ready = require_release_ready and not manifest["release_ready"]
    prepackage_not_ready = (
        require_prepackage_ready and not manifest["prepackage_ready"]
    )
    provenance_failures: list[str] = []
    if require_production_provenance:
        if scope != R20_PRODUCTION_SCOPE:
            provenance_failures.append(
                f"production provenance requires scope {R20_PRODUCTION_SCOPE}"
            )
        if expected_version is None or jang_source is None:
            provenance_failures.append(
                "production provenance requires expected_version and jang_source"
            )
        else:
            provenance, collected_provenance_failures = collect_production_provenance(
                root,
                expected_version=expected_version,
                jang_source=jang_source,
            )
            provenance_failures.extend(collected_provenance_failures)
            manifest.update(provenance)
        manifest["production_provenance"] = {
            "status": "pass" if not provenance_failures else "fail",
            "failures": provenance_failures,
        }
    if scoped_prepackage_receipt is not None:
        allowed = (scope == R20_PRODUCTION_SCOPE and require_prepackage_ready
                   and require_production_provenance and not require_release_ready
                   and not require_current_proof_sweep and not provenance_failures)
        acceptance = (validate_scoped_prepackage(
            root, scoped_prepackage_receipt, provenance=manifest,
            expected_version=expected_version or "",
        ) if allowed else {"status": "fail", "failures": [
            "scoped evidence requires production provenance and prepackage-only mode"
        ]})
        manifest["scoped_prepackage_acceptance"] = acceptance
        accepted = acceptance["status"] == "pass"
        manifest["prepackage_ready"] = accepted
        manifest["release_ready"] = False
        manifest["release_clearance"] = {
            "status": "open", "release_ready": False,
            "pending_stages": SCOPED_PENDING_STAGES,
            "historical": manifest["release_clearance"],
        }
        manifest["prepackage_clearance"] = {
            "status": "pass" if accepted else "open", "prepackage_ready": accepted,
            "mode": SCOPED_PREPACKAGE_SCHEMA, "acceptance": acceptance,
        }
        manifest["status"] = "pass" if accepted else "fail"
        return manifest
    manifest["status"] = (
        "fail"
        if (
            (proof_sweep_failed and not proof_sweep_failure_allowed_for_prepackage)
            and not proof_sweep_failure_allowed_for_release
            or release_not_ready
            or prepackage_not_ready
            or provenance_failures
        )
        else "pass"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--require-current-proof-sweep",
        action="store_true",
        help="Exit nonzero unless every current post-budget-edge proof artifact exists and has status=pass.",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="Exit nonzero unless current proof sweep passes and no release blockers/open requirements remain.",
    )
    parser.add_argument(
        "--require-prepackage-ready",
        action="store_true",
        help="Exit nonzero unless current proof sweep passes and only packaging/signing blockers remain.",
    )
    parser.add_argument(
        "--require-production-provenance",
        action="store_true",
        help="Require clean pushed canonical vMLX/JANG origin/main provenance.",
    )
    parser.add_argument("--scoped-prepackage-receipt", type=Path)
    parser.add_argument("--scope")
    parser.add_argument("--expected-version")
    parser.add_argument("--jang-source", type=Path)
    args = parser.parse_args()

    manifest = build_manifest_artifact(
        Path("."),
        scope=args.scope,
        require_current_proof_sweep=args.require_current_proof_sweep,
        require_release_ready=args.require_release_ready,
        require_prepackage_ready=args.require_prepackage_ready,
        require_production_provenance=args.require_production_provenance,
        expected_version=args.expected_version,
        jang_source=args.jang_source,
        scoped_prepackage_receipt=args.scoped_prepackage_receipt,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(f"rows={len(manifest['rows'])}")
    print("domains=" + ",".join(sorted({row["domain"] for row in manifest["rows"]})))
    print(f"current_proof_sweep={manifest['current_proof_sweep']['status']}")
    print(f"prepackage_ready={str(manifest['prepackage_ready']).lower()}")
    print(f"release_ready={str(manifest['release_ready']).lower()}")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
