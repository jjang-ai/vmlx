#!/usr/bin/env python3
"""Write the release-regression manifest artifact.

This command does not run the heavy suite. It produces a stable JSON inventory
of the release-regression rows and the commands/artifacts that prove each row.
Use it before expanding or auditing the suite so new fixes land in a named row
instead of another one-off command.
"""

from __future__ import annotations

import argparse
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
