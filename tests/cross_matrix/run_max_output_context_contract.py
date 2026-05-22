#!/usr/bin/env python3
"""Run no-heavy max-output/max-context separation contracts.

This gate protects the release regression Eric called out: server default
output caps, per-chat/API output overrides, and prompt/context caps must stay
separate across panel launch, request building, persisted-session migration,
and server request resolution.
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


DEFAULT_OUT = Path("build/current-max-output-context-contract-20260521.json")

SOURCE_HASH_FILES = (
    "vmlx_engine/server.py",
    "vmlx_engine/api/anthropic_adapter.py",
    "vmlx_engine/api/ollama_adapter.py",
    "tests/test_engine_audit.py",
    "tests/test_ollama_adapter.py",
    "tests/test_max_output_context_contract.py",
    "panel/src/main/chat-override-policy.ts",
    "panel/src/main/sessions.ts",
    "panel/src/main/ipc/chat.ts",
    "panel/src/shared/sessionConfigMigrations.ts",
    "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
    "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
    "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
    "panel/src/renderer/src/components/chat/ChatSettings.tsx",
    "panel/src/renderer/src/i18n/locales/en.json",
    "panel/tests/settings-flow.test.ts",
    "panel/tests/database-migrations.test.ts",
    "panel/tests/request-builder.test.ts",
    "panel/tests/chat-override-policy.test.ts",
    "panel/tests/coding-tools-config-save.test.ts",
    "panel/tests/dsv4-request-budget.test.ts",
    "panel/tests/api-gateway-ollama-behavior.test.ts",
    "tests/cross_matrix/run_max_output_context_contract.py",
)

REQUIRED_MAX_OUTPUT_CONTEXT_TEST_MARKERS = (
    # Engine/API resolution. These prove server --max-tokens is an omitted
    # request default, not a ceiling over explicit chat/API output caps.
    "test_request_output_caps_override_server_default_without_touching_context_cap",
    "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap",
    "test_explicit_startup_max_tokens_is_default_not_request_ceiling",
    "test_request_output_caps_can_go_below_or_above_startup_default",
    "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap",
    "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap",
    "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap",
    "test_explicit_server_max_tokens_overrides_bundle_max_new_tokens",
    "test_omitted_server_max_tokens_uses_bundle_max_new_tokens",
    "test_omitted_server_max_tokens_without_bundle_default_is_bounded",
    "test_prompt_context_aliases_clamp_without_rewriting_output_caps",
    "omits malformed Ollama context values instead of poisoning max_prompt_tokens",
    "test_max_tokens_resolution_contract_applies_to_every_registered_family",
    "test_wake_reload_preserves_max_tokens_explicitness",
    "test_cli_serve_implicit_max_tokens_uses_bounded_fallback",
    "test_anthropic_messages_omitted_max_tokens_uses_bundle_default",
    "test_ollama_chat_omits_non_positive_num_predict_sentinels",
    "test_ollama_generate_omits_non_positive_num_predict_sentinels",
    "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap",
    "omits malformed Ollama num_predict values instead of poisoning max_tokens",
    # Panel launch/settings. These catch UI confusion between response length
    # and prompt/context length before a session can relaunch with stale state.
    "surfaces Max Output Tokens separately from Max Context Tokens",
    "does not synthesize a huge max tokens flag when set to 0 (model/server default)",
    "does not copy model max_new_tokens into hidden startup maxTokens config",
    "database clears legacy session maxTokens before settings UI or launch can reuse them",
    "changing maxTokens produces different CLI output",
    "maxContextLength emits max prompt/context CLI flag when explicitly set",
    "clears legacy session maxTokens=32768 before launch can reuse it",
    # Chat/API request builders. These prove per-chat output caps stay request
    # scoped and do not become server context caps or poisoned defaults.
    "chat settings expose per-chat max tokens without hidden DSV4 floors",
    "keeps per-chat maxTokens as output budget only, never prompt context",
    "keeps Responses maxTokens as output budget only, never prompt context",
    "omits invalid persisted maxTokens values instead of poisoning Chat Completions",
    "omits invalid persisted maxTokens values instead of poisoning Responses",
    "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap",
    "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults",
    "chat maxTokens save path cannot mutate session startup maxTokens",
    "persisted chat maxTokens cannot relaunch server with a new startup maxTokens",
    "server startup maxTokens and chat maxTokens remain independent when both are set",
    "per-chat maxTokens below or above the server startup default remain request scoped",
    "Auto chat maxTokens omits per-request output caps so server default can apply",
    "default profiles cannot make maxTokens sticky on clean new chats",
    "new chats preserve model-owned maxTokens while refusing inherited output caps",
    "coding tool configs keep output limit separate from context fallback",
    "does not synthesize a DSV4 max token budget when the request leaves it to server/model defaults",
    "preserves DSV4 Responses max_output_tokens for Max thinking",
    "omits unset and disabled sampling sentinels without dropping explicit overrides",
)

PANEL_PATTERN = (
    "maxTokens|Max Tokens|Max Output|Max Context|max context|max output|"
    "max_tokens|max_output_tokens|32768|server default output|per-chat|"
    "omits max_tokens|max token budget|output budgets|output limit|"
    "context fallback|coding tool"
)

COMMANDS: dict[str, tuple[Path, list[str]]] = {
    "engine_output_context_resolution": (
        Path("."),
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-vv",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_request_output_caps_override_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_explicit_startup_max_tokens_is_default_not_request_ceiling",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_request_output_caps_can_go_below_or_above_startup_default",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_prompt_context_aliases_clamp_without_rewriting_output_caps",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_explicit_server_max_tokens_overrides_bundle_max_new_tokens",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_omitted_server_max_tokens_uses_bundle_max_new_tokens",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_omitted_server_max_tokens_without_bundle_default_is_bounded",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_max_tokens_resolution_contract_applies_to_every_registered_family",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_wake_reload_preserves_max_tokens_explicitness",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_cli_serve_implicit_max_tokens_uses_bounded_fallback",
            "tests/test_engine_audit.py::TestServerSamplingResolution::test_anthropic_messages_omitted_max_tokens_uses_bundle_default",
            "tests/test_ollama_adapter.py::test_ollama_generate_default_uses_chat_template_request_shape",
            "tests/test_ollama_adapter.py::test_ollama_chat_omits_non_positive_num_predict_sentinels",
            "tests/test_ollama_adapter.py::test_ollama_generate_omits_non_positive_num_predict_sentinels",
            "tests/test_max_output_context_contract.py",
        ],
    ),
    "panel_output_context_wiring": (
        Path("panel"),
        [
            "npx",
            "vitest",
            "run",
            "tests/settings-flow.test.ts",
            "tests/chat-settings-compatibility.test.ts",
            "tests/database-migrations.test.ts",
            "tests/request-builder.test.ts",
            "tests/chat-override-policy.test.ts",
            "tests/coding-tools-config-save.test.ts",
            "tests/dsv4-request-budget.test.ts",
            "tests/api-gateway-ollama-behavior.test.ts",
            "--testNamePattern",
            PANEL_PATTERN,
            "--reporter=verbose",
        ],
    ),
}


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
        "stdout_tail": proc.stdout.splitlines()[-80:],
    }


def build_artifact(root: Path) -> dict[str, Any]:
    results = {
        name: _run(root, name, cwd_rel, cmd)
        for name, (cwd_rel, cmd) in COMMANDS.items()
    }
    failed = [name for name, result in results.items() if result["returncode"] != 0]
    stdout = "\n".join(str(result.get("stdout", "")) for result in results.values())
    missing_markers = [
        marker
        for marker in REQUIRED_MAX_OUTPUT_CONTEXT_TEST_MARKERS
        if marker not in stdout
    ]
    engine_passed = results["engine_output_context_resolution"]["counts"]["passed"] or 0
    panel_passed = results["panel_output_context_wiring"]["counts"]["passed"] or 0
    checks = {
        "server_default_output_cap_uses_max_tokens": (
            not failed
            and "test_explicit_server_max_tokens_overrides_bundle_max_new_tokens" not in missing_markers
            and "test_omitted_server_max_tokens_uses_bundle_max_new_tokens" not in missing_markers
        ),
        "startup_output_cap_is_default_not_request_ceiling": (
            not failed
            and "test_explicit_startup_max_tokens_is_default_not_request_ceiling" not in missing_markers
            and "test_request_output_caps_can_go_below_or_above_startup_default" not in missing_markers
        ),
        "request_output_caps_can_go_below_or_above_startup_default": (
            not failed
            and "test_request_output_caps_can_go_below_or_above_startup_default" not in missing_markers
        ),
        "legacy_completions_output_cap_overrides_server_default": (
            not failed
            and "test_legacy_completions_output_cap_overrides_server_default_without_touching_context_cap" not in missing_markers
            and "test_legacy_completions_streaming_output_cap_overrides_server_default_without_touching_context_cap" not in missing_markers
        ),
        "chat_max_tokens_overrides_server_default_per_request": (
            not failed
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "keeps per-chat maxTokens as output budget only, never prompt context" not in missing_markers
        ),
        "responses_max_output_tokens_overrides_server_default_per_request": (
            not failed
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_chat_and_responses_streaming_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "keeps Responses maxTokens as output budget only, never prompt context" not in missing_markers
        ),
        "anthropic_messages_preserves_bundle_and_explicit_output_caps": (
            not failed
            and "test_anthropic_messages_omitted_max_tokens_uses_bundle_default" not in missing_markers
            and "test_anthropic_messages_streaming_max_tokens_overrides_server_default_without_touching_context_cap" not in missing_markers
        ),
        "ollama_num_predict_maps_only_positive_output_caps": (
            not failed
            and "test_ollama_chat_omits_non_positive_num_predict_sentinels" not in missing_markers
            and "test_ollama_generate_omits_non_positive_num_predict_sentinels" not in missing_markers
            and "test_ollama_streaming_num_predict_overrides_server_default_without_touching_context_cap" not in missing_markers
            and "omits malformed Ollama num_predict values instead of poisoning max_tokens" not in missing_markers
            and "omits unset and disabled sampling sentinels without dropping explicit overrides" not in missing_markers
        ),
        "prompt_context_caps_do_not_rewrite_output_cap": (
            not failed
            and "test_request_output_caps_override_server_default_without_touching_context_cap" not in missing_markers
            and "test_prompt_context_aliases_clamp_without_rewriting_output_caps" not in missing_markers
            and "omits malformed Ollama context values instead of poisoning max_prompt_tokens" not in missing_markers
            and "maxContextLength emits max prompt/context CLI flag when explicitly set" not in missing_markers
        ),
        "panel_server_default_output_maps_to_max_tokens": (
            not failed
            and "surfaces Max Output Tokens separately from Max Context Tokens" not in missing_markers
            and "changing maxTokens produces different CLI output" not in missing_markers
        ),
        "panel_max_context_maps_to_max_prompt_tokens": (
            not failed
            and "surfaces Max Output Tokens separately from Max Context Tokens" not in missing_markers
            and "maxContextLength emits max prompt/context CLI flag when explicitly set" not in missing_markers
        ),
        "stale_32768_session_output_caps_are_migrated": (
            not failed
            and "database clears legacy session maxTokens before settings UI or launch can reuse them" not in missing_markers
            and "clears legacy session maxTokens=32768 before launch can reuse it" not in missing_markers
        ),
        "chat_output_cap_remains_per_chat_override": (
            not failed
            and "chat settings expose per-chat max tokens without hidden DSV4 floors" not in missing_markers
            and "chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap" not in missing_markers
            and "chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults" not in missing_markers
            and "persisted chat maxTokens cannot relaunch server with a new startup maxTokens" not in missing_markers
            and "per-chat maxTokens below or above the server startup default remain request scoped" not in missing_markers
        ),
        "request_builders_omit_auto_output_cap": (
            not failed
            and "does not synthesize a DSV4 max token budget when the request leaves it to server/model defaults" not in missing_markers
            and "omits invalid persisted maxTokens values instead of poisoning Chat Completions" not in missing_markers
            and "omits invalid persisted maxTokens values instead of poisoning Responses" not in missing_markers
        ),
        "new_chat_output_caps_are_not_inherited_or_made_sticky": (
            not failed
            and "default profiles cannot make maxTokens sticky on clean new chats" not in missing_markers
            and "new chats preserve model-owned maxTokens while refusing inherited output caps" not in missing_markers
            and "chat maxTokens save path cannot mutate session startup maxTokens" not in missing_markers
        ),
        "all_family_max_token_precedence_stays_uniform": (
            not failed
            and "test_max_tokens_resolution_contract_applies_to_every_registered_family" not in missing_markers
            and "test_omitted_server_max_tokens_without_bundle_default_is_bounded" not in missing_markers
        ),
        "wake_reload_and_cli_preserve_explicitness": (
            not failed
            and "test_wake_reload_preserves_max_tokens_explicitness" not in missing_markers
            and "test_cli_serve_implicit_max_tokens_uses_bounded_fallback" not in missing_markers
        ),
        "legacy_count_floor_still_nontrivial": (
            not failed and engine_passed >= 14 and panel_passed >= 28
        ),
        "all_required_max_output_context_markers_present": not failed and not missing_markers,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed": failed,
        "missing_markers": missing_markers,
        "source_hashes": {
            rel: _sha256(root / rel)
            for rel in SOURCE_HASH_FILES
            if (root / rel).exists()
        },
        "results": {
            name: {key: value for key, value in result.items() if key != "stdout"}
            for name, result in results.items()
        },
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
    print("missing_markers=" + json.dumps(artifact["missing_markers"]))
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
