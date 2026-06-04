from pathlib import Path

from tests.cross_matrix.run_remote_max2_dsv4_real_ui_guard import (
    build_guard_result,
    remote_real_ui_command,
)


def test_remote_max2_dsv4_real_ui_guard_builds_responses_tool_cache_command():
    command = remote_real_ui_command(
        worktree=Path("/Users/example/mlx/vllm-mlx-codex-proof-1554"),
        node=Path("/opt/homebrew/bin/node"),
        app=Path("/Users/example/mlx/vllm-mlx-codex-proof-1554/panel/release/sequoia-app/mac-arm64/vMLX.app"),
        model=Path("/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"),
        proof_basename="dsv4-ui-proof",
        max_tokens=512,
        max_tool_iterations=8,
        cache_expect_regex="Prefix Cache|Paged KV Cache",
    )

    assert command.startswith("cd /Users/example/mlx/vllm-mlx-codex-proof-1554 &&")
    assert "VMLINUX_REAL_UI_WIRE_API=responses" in command
    assert "VMLINUX_REAL_UI_BUILTIN_TOOLS=1" in command
    assert "VMLINUX_REAL_UI_MAX_TOOL_ITERATIONS=8" in command
    assert "VMLINUX_REAL_UI_MAX_TOKENS=512" in command
    assert "VMLINUX_REAL_UI_CHECK_SERVER_CACHE_CONTROLS=1" in command
    assert "DeepSeek-V4-Flash-JANGTQ-K" in command
    assert "/opt/homebrew/bin/node panel/scripts/live-real-ui-model-proof.mjs" in command


def test_remote_max2_dsv4_real_ui_guard_skips_memory_and_app_blockers():
    result = build_guard_result(
        readiness={
            "launch_allowed": False,
            "launch_blockers": ["insufficient_memory"],
        },
        prereqs={
            "node_present": True,
            "node_version": "v24.0.0",
            "script_present": True,
            "app_present": False,
        },
        command="cd /proof && node panel/scripts/live-real-ui-model-proof.mjs",
        dry_run=False,
    )

    assert result["status"] == "skipped_not_ready"
    assert result["launch_decision"] == "do_not_launch"
    assert result["launch_blockers"] == ["insufficient_memory", "staged_app_missing"]


def test_remote_max2_dsv4_real_ui_guard_dry_run_never_launches():
    result = build_guard_result(
        readiness={"launch_allowed": True, "launch_blockers": []},
        prereqs={
            "node_present": True,
            "node_version": "v24.0.0",
            "script_present": True,
            "app_present": True,
        },
        command="cd /proof && node panel/scripts/live-real-ui-model-proof.mjs",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["launch_decision"] == "dry_run_no_launch"
    assert result["launch_allowed"] is False
