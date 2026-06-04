from pathlib import Path

from tests.cross_matrix.run_remote_max2_dsv4_exactness_guard import (
    build_guard_result,
    remote_exactness_command,
)


def test_remote_max2_dsv4_exactness_guard_builds_current_worktree_command():
    command = remote_exactness_command(
        worktree=Path("/Users/example/mlx/vllm-mlx-codex-proof-1554"),
        python=Path("/Users/example/mlx/vllm-mlx/.venv/bin/python"),
        model=Path("/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"),
        remote_out=Path("build/current-dsv4-route-mode-code-exactness-max2.json"),
        cases="chat_off_rep1,responses_off_rep1",
        port=8897,
        timeout=420,
        request_timeout=300,
        min_free_gb=120.0,
    )

    assert command.startswith("cd /Users/example/mlx/vllm-mlx-codex-proof-1554 &&")
    assert "/Users/example/mlx/vllm-mlx/.venv/bin/python" in command
    assert "tests/cross_matrix/run_dsv4_route_mode_code_exactness.py" in command
    assert "--model /Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K" in command
    assert "--cases chat_off_rep1,responses_off_rep1" in command
    assert "--min-free-gb 120.0" in command


def test_remote_max2_dsv4_exactness_guard_skips_when_readiness_blocks():
    result = build_guard_result(
        readiness={
            "launch_allowed": False,
            "launch_decision": "do_not_launch",
            "launch_blockers": ["insufficient_memory"],
        },
        command="cd /proof && python run.py",
        dry_run=False,
    )

    assert result["status"] == "skipped_not_ready"
    assert result["launch_decision"] == "do_not_launch"
    assert result["readiness"]["launch_blockers"] == ["insufficient_memory"]


def test_remote_max2_dsv4_exactness_guard_dry_run_never_launches():
    result = build_guard_result(
        readiness={"launch_allowed": True, "launch_decision": "launch_allowed"},
        command="cd /proof && python run.py",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["launch_decision"] == "dry_run_no_launch"
