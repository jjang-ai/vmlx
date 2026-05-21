from __future__ import annotations

import json


def _write_fake_bundle(path, *, family: str = "qwen3_5"):
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": family,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    (path / "jang_config.json").write_text(
        json.dumps(
            {
                "runtime": {"bundle_has_mtp": True, "mtp_layers": 1},
                "mtp": {"enabled": True, "kept": True, "num_layers": 1},
                "capabilities": {"family": family, "cache_type": "kv"},
            }
        ),
        encoding="utf-8",
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "mtp.0.layers.0.self_attn.q_proj.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )


def test_common_depth_aliases_clamp_and_default():
    from vmlx_engine.native_mtp_examples.mtp_runtime_common import (
        clamp_native_mtp_depth,
        effective_depth_from_env,
    )

    assert clamp_native_mtp_depth(0) == 1
    assert clamp_native_mtp_depth(99) == 3
    assert effective_depth_from_env({}) == (3, "default")
    assert effective_depth_from_env({"VMLINUX_NATIVE_MTP_DEPTH": "2"}) == (
        2,
        "VMLINUX_NATIVE_MTP_DEPTH",
    )
    assert effective_depth_from_env({"VMLX_NATIVE_MTP_DEPTH": "8"}) == (
        3,
        "VMLX_NATIVE_MTP_DEPTH",
    )


def test_common_tuning_depth_uses_validated_sidecar(tmp_path):
    from vmlx_engine.native_mtp_examples.mtp_runtime_common import tuning_depth

    (tmp_path / "vmlx_mtp_tuning.json").write_text(
        json.dumps(
            {
                "native_mtp": {
                    "validated": True,
                    "output_equivalent": True,
                    "best_depth": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    assert tuning_depth(tmp_path) == (2, "vmlx_mtp_tuning.json:native_mtp.best_depth")


def test_inspect_metadata_preserves_qwen_supported_and_dsv4_unwired(tmp_path):
    from vmlx_engine.native_mtp_examples.inspect_mtp_metadata import inspect_path

    qwen = tmp_path / "qwen"
    qwen.mkdir()
    _write_fake_bundle(qwen, family="qwen3_5")
    dsv4 = tmp_path / "dsv4"
    dsv4.mkdir()
    _write_fake_bundle(dsv4, family="deepseek_v4")

    qwen_status = inspect_path(qwen)["status"]
    dsv4_status = inspect_path(dsv4)["status"]

    assert qwen_status["artifact_available"] is True
    assert qwen_status["runtime_supported"] is True
    assert qwen_status["runtime_available"] is True
    assert qwen_status["effective_depth"] == 3
    assert dsv4_status["artifact_available"] is True
    assert dsv4_status["runtime_supported"] is False
    assert dsv4_status["runtime_available"] is False
    assert dsv4_status["status"] == "weights_present_runtime_unwired"


def test_env_matrix_builds_default_depth_and_clamped_alias_rows(tmp_path):
    import pytest

    from vmlx_engine.native_mtp_examples.env_flag_matrix import parse_depths
    from vmlx_engine.native_mtp_examples.mtp_runtime_common import command_matrix_rows

    rows = command_matrix_rows(tmp_path, depths=[0, 2, 9], include_disabled=True)

    labels = [row["label"] for row in rows]
    assert labels == [
        "native_mtp_default_depth",
        "native_mtp_d1",
        "native_mtp_d2",
        "native_mtp_d3",
        "native_mtp_disabled",
    ]
    d2 = rows[2]
    assert d2["env"]["VMLINUX_NATIVE_MTP_DEPTH"] == "2"
    assert d2["env"]["VMLX_NATIVE_MTP_DEPTH"] == "2"
    assert rows[-1]["env"]["VMLINUX_NATIVE_MTP"] == "0"
    with pytest.raises(Exception, match="depth values must be integers"):
        parse_depths("1,nope")


def test_server_command_reports_validated_tuning_without_setting_depth_env(tmp_path):
    from vmlx_engine.native_mtp_examples.mtp_runtime_common import build_server_command

    (tmp_path / "vmlx_mtp_tuning.json").write_text(
        json.dumps(
            {
                "native_mtp": {
                    "validated": True,
                    "output_equivalent": True,
                    "best_depth": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    plan = build_server_command(tmp_path)

    assert plan["native_mtp_depth"] == 2
    assert plan["native_mtp_depth_source"] == "vmlx_mtp_tuning.json:native_mtp.best_depth"
    assert "VMLINUX_NATIVE_MTP_DEPTH" not in plan["env"]


def test_parse_mtp_logs_detects_activation_accept_and_depth_telemetry():
    from vmlx_engine.native_mtp_examples.parse_mtp_logs import parse_log_lines

    report = parse_log_lines(
        [
            "INFO MTP path activated for request abc depth=3",
            "INFO MTP[abc] accept=2/3 token=42",
            "INFO MLLM MTP[abc] finish=stop cycles=4 accepted=10/12 (83.3%) emits[init=1,draft=8,bonus=2,verify=4]",
            "INFO MLLM MTP[abc] accept_by_depth[d1=4/4,d2=3/4,d3=3/4] forwards[seed_main=1,verify_main=4,replay_main=0,mtp=12]",
        ]
    )

    assert report["mtp_path_activated"] is True
    assert report["request_count"] == 1
    row = report["requests"][0]
    assert row["accepted_tokens"] == 10
    assert row["drafted_tokens"] == 12
    assert row["accept_events"][0]["acceptance_rate"] == 2 / 3
    assert row["accepted_by_depth"] == [4, 3, 3]
    assert row["acceptance_by_depth"] == [1.0, 0.75, 0.75]


def test_parse_mtp_logs_detects_text_batch_generator_finish():
    from vmlx_engine.native_mtp_examples.parse_mtp_logs import parse_log_lines

    report = parse_log_lines(
        [
            "INFO MTP[7] finish=stop tokens=11 cycles=5 accept=4/5 (80.0%) emits[init=2,draft=4,bonus=2,verify=5]",
        ]
    )

    assert report["acceptance_telemetry_present"] is True
    row = report["requests"][0]
    assert row["finish"] == "stop"
    assert row["tokens"] == 11
    assert row["accepted_tokens"] == 4
    assert row["drafted_tokens"] == 5


def test_parse_mtp_logs_summarizes_accept_only_telemetry():
    from vmlx_engine.native_mtp_examples.parse_mtp_logs import parse_log_lines

    report = parse_log_lines(["INFO MTP[req] accept=2/3 token=42"])

    assert report["totals"]["accepted_tokens"] == 2
    assert report["totals"]["drafted_tokens"] == 3
    assert report["totals"]["acceptance_rate"] == 2 / 3


def test_parse_mtp_logs_requirement_flags_fail_closed(tmp_path, capsys):
    from vmlx_engine.native_mtp_examples.parse_mtp_logs import main

    empty_log = tmp_path / "empty.log"
    empty_log.write_text("", encoding="utf-8")

    assert main(["--require-active", str(empty_log)]) == 2
    captured = capsys.readouterr()
    assert "Missing required native MTP activation log" in captured.err

    assert main(["--require-acceptance", str(empty_log)]) == 3
    captured = capsys.readouterr()
    assert "Missing required native MTP acceptance telemetry" in captured.err


def test_server_smoke_is_dry_run_and_live_guard_blocks(tmp_path):
    import pytest

    from vmlx_engine.native_mtp_examples.mtp_runtime_common import require_live_run_allowed
    from vmlx_engine.native_mtp_examples.server_smoke import build_smoke_plan

    plan = build_smoke_plan(tmp_path, depth=3, model_name="fake-mtp")

    assert plan["dry_run"] is True
    assert plan["no_model_load"] is True
    assert "serve" in plan["server"]["command"]
    assert plan["server"]["env"]["VMLINUX_NATIVE_MTP_DEPTH"] == "3"
    assert plan["checks"][0][-1].endswith("/health")
    assert "evil;name" in build_smoke_plan(tmp_path, model_name="evil;name")["checks"][1][-1]
    with pytest.raises(SystemExit):
        require_live_run_allowed({}, allow_live=True)
