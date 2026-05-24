from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_NON_SERVE_DIAGNOSTIC_FLAGS = {
    # sessions.ts still parses old log lines from `python -m vmlx_engine.server
    # --model <path>` so the UI can recover a model path from historical output.
    # It is not emitted by the current `vmlx_engine.cli serve` launcher.
    "--model",
}


def _serve_cli_flags() -> set[str]:
    source = (ROOT / "vmlx_engine" / "cli.py").read_text(encoding="utf-8")
    serve_block = source[
        source.index('serve_parser = subparsers.add_parser("serve"'):
        source.index('bench_parser = subparsers.add_parser("bench"')
    ]
    return set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', serve_block))


def _panel_source_flags() -> dict[str, set[str]]:
    rels = (
        "panel/src/main/sessions.ts",
        "panel/src/renderer/src/components/sessions/SessionSettings.tsx",
        "panel/src/renderer/src/components/sessions/SessionConfigForm.tsx",
        "panel/src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
        "panel/src/renderer/src/components/sessions/CreateSession.tsx",
    )
    return {
        rel: set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', (ROOT / rel).read_text(encoding="utf-8")))
        for rel in rels
    }


def _constant_flag_set(rel: str, const_name: str) -> set[str]:
    source = (ROOT / rel).read_text(encoding="utf-8")
    start = source.index(f"const {const_name}")
    end = source.index("])", start)
    block = source[start:end]
    return set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', block))


def test_panel_serve_flags_are_registered_engine_cli_flags() -> None:
    """The app must not emit or preview serve flags argparse cannot accept."""

    serve_flags = _serve_cli_flags()
    panel_flags = _panel_source_flags()
    missing = {}
    for rel, flags in panel_flags.items():
        unsupported = sorted(flags - serve_flags - LEGACY_NON_SERVE_DIAGNOSTIC_FLAGS)
        if unsupported:
            missing[rel] = unsupported
    assert missing == {}


def test_dsv4_advanced_args_blocklist_names_real_serve_flags() -> None:
    """DSV4 stale-arg sanitizer should track real serve flags, not dead names."""

    source = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    start = source.index("const DSV4_ADDITIONAL_ARG_BLOCKLIST")
    end = source.index("])", start)
    block = source[start:end]
    blocked = set(re.findall(r'["\'](--[a-z0-9][a-z0-9-]*)["\']', block))
    assert blocked, "DSV4 blocklist unexpectedly empty"
    assert sorted(blocked - _serve_cli_flags()) == []


def test_runtime_and_preview_additional_arg_filters_share_blocklists() -> None:
    """Runtime launch and UI command preview must strip the same stale flags."""

    names = (
        "ADDITIONAL_ARG_VALUE_FLAGS",
        "IMAGE_ADDITIONAL_ARG_BLOCKLIST",
        "DSV4_ADDITIONAL_ARG_BLOCKLIST",
    )
    runtime_rel = "panel/src/main/sessions.ts"
    preview_rel = "panel/src/renderer/src/components/sessions/SessionSettings.tsx"
    for name in names:
        assert _constant_flag_set(preview_rel, name) == _constant_flag_set(runtime_rel, name)


def test_panel_cli_flag_contract_covers_dsv4_cache_and_output_boundaries() -> None:
    """Pin the risky rows Eric called out so this file stays purpose-built."""

    sessions = (ROOT / "panel" / "src" / "main" / "sessions.ts").read_text(encoding="utf-8")
    assert "--dsv4-enable-prefix-cache" in sessions
    assert "--use-paged-cache" in sessions
    assert "--enable-block-disk-cache" in sessions
    assert "--kv-cache-quantization" in sessions
    assert "--max-tokens" in sessions
    assert "--max-prompt-tokens" in sessions
    assert "--native-mtp-depth" in sessions
    assert "--native-mtp-sampling-policy" in sessions
