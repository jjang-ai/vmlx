from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def test_dsv4_default_cache_tool_loop_gate_resolves_current_python_candidate(
    tmp_path,
):
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
        resolve_default_python,
    )

    stale = tmp_path / "panel/release/mac-arm64/vMLX.app/python3"
    current = tmp_path / "panel/release/sequoia-app/mac-arm64/vMLX.app/python3"
    current.parent.mkdir(parents=True)
    current.write_text("#!/bin/sh\n", encoding="utf-8")

    assert resolve_default_python((stale, current)) == current


def test_dsv4_default_cache_tool_loop_gate_dry_run_pins_default_cache_flags():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import run

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            disable_prefix_cache=False,
        )
    )

    assert result["status"] == "dry_run"
    assert "--dsv4-enable-prefix-cache" in result["cmd"]
    assert "--use-paged-cache" in result["cmd"]
    assert "--enable-block-disk-cache" in result["cmd"]
    assert "--disable-prefix-cache" not in result["cmd"]
    assert "--kv-cache-quantization" not in result["cmd"]
    assert "--tool-call-parser" in result["cmd"]
    assert "dsml" in result["cmd"]
    assert "--reasoning-parser" in result["cmd"]
    assert "deepseek_r1" in result["cmd"]
    assert result["env"]["DSV4_LONG_CTX"] == "1"
    assert result["env"]["DSV4_POOL_QUANT"] == "0"


def test_dsv4_default_cache_tool_loop_gate_dry_run_can_disable_cache_for_diagnostic_ab():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import run

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            disable_prefix_cache=True,
        )
    )

    assert result["status"] == "dry_run"
    assert "--disable-prefix-cache" in result["cmd"]
    assert "--dsv4-enable-prefix-cache" not in result["cmd"]
    assert "--use-paged-cache" not in result["cmd"]
    assert "--enable-block-disk-cache" not in result["cmd"]
    assert result["diagnostic_cache_mode"] == "disabled"


def test_dsv4_default_cache_tool_loop_gate_dry_run_pins_code_body_tool_probe():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
        EXPECTED_CODE_TOOL_CONTENT,
        EXPECTED_CODE_TOOL_PATH,
        run,
    )

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            disable_prefix_cache=False,
        )
    )

    assert result["code_tool_probe"]["path"] == EXPECTED_CODE_TOOL_PATH
    assert result["code_tool_probe"]["expected_content"] == EXPECTED_CODE_TOOL_CONTENT
    assert result["code_tool_probe"]["actual_content"] is None
    assert result["code_tool_probe"]["exact"] is False
    assert "THREE.WebGLRenderer" in EXPECTED_CODE_TOOL_CONTENT
    assert "THREE.MeshBasicMaterial" in EXPECTED_CODE_TOOL_CONTENT


def test_dsv4_default_cache_tool_loop_gate_dry_run_surfaces_code_round_controls_and_diff():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
        EXPECTED_CODE_TOOL_CONTENT,
        build_code_tool_probe,
        run,
    )

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            max_output_tokens=768,
            enable_thinking=False,
            disable_prefix_cache=False,
        )
    )

    assert result["code_round_request_controls"] == {
        "round": 2,
        "model": "dsv4-default-cache-tools",
        "store": True,
        "stream": False,
        "max_output_tokens": 768,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools_enabled": True,
        "tool_choice": "auto",
    }

    probe = build_code_tool_probe(
        EXPECTED_CODE_TOOL_CONTENT.replace("WebGLRenderer", "WebRenderer").removesuffix(";")
    )
    assert probe["exact"] is False
    assert probe["first_difference"]["expected_char"] == "G"
    assert probe["first_difference"]["actual_char"] == "R"
    assert "THREE.WebGLRenderer" in probe["missing_expected_fragments"]
    assert "THREE.WebRenderer" in probe["corrupt_identifier_patterns"]
    assert "renderer.render(scene, camera)" not in probe["corrupt_identifier_patterns"]

    prompt_guard_probe = build_code_tool_probe(
        EXPECTED_CODE_TOOL_CONTENT.replace(
            "THREE.PerspectiveCamera",
            "THREE.PPerspectiveCamera",
        ).replace("THREE.BoxGeometry", "THREE.BBoxGeometry")
    )
    assert prompt_guard_probe["missing_expected_fragments"] == [
        "THREE.PerspectiveCamera"
    ]
    assert prompt_guard_probe["corrupt_identifier_patterns"] == [
        "THREE.PPerspectiveCamera",
        "THREE.BBoxGeometry",
    ]

    exact_probe = build_code_tool_probe(EXPECTED_CODE_TOOL_CONTENT)
    assert exact_probe["exact"] is True
    assert exact_probe["corrupt_identifier_patterns"] == []


def test_dsv4_default_cache_tool_loop_gate_dry_run_can_select_copy_block_code_prompt():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
        EXPECTED_CODE_TOOL_CONTENT,
        run,
    )

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            max_output_tokens=768,
            enable_thinking=False,
            disable_prefix_cache=False,
            code_prompt_variant="copy_block",
        )
    )

    assert result["diagnostic_code_prompt_variant"] == "copy_block"
    assert "Copy this exact JavaScript into" in result["diagnostic_code_prompt"]
    assert EXPECTED_CODE_TOOL_CONTENT in result["diagnostic_code_prompt"]
    assert "Preserve every character" in result["diagnostic_code_prompt"]


def test_dsv4_default_cache_tool_loop_gate_dry_run_pins_output_budget():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import run

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            max_output_tokens=768,
            disable_prefix_cache=False,
        )
    )

    assert result["request_max_output_tokens"] == 768
    assert "--max-tokens" in result["cmd"]
    assert result["cmd"][result["cmd"].index("--max-tokens") + 1] == "768"


def test_dsv4_default_cache_tool_loop_gate_dry_run_can_request_thinking_mode():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import run

    result = run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=0,
            dry_run=True,
            pool_quant=False,
            max_output_tokens=320,
            enable_thinking=True,
            disable_prefix_cache=False,
        )
    )

    assert result["request_thinking_mode"] == {
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_dsv4_default_cache_tool_loop_memory_snapshot_labels_binary_units(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate as gate

    class FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(
                total=128 * 1024**3,
                available=113.59 * 1024**3,
                percent=11.3,
            )

    monkeypatch.setitem(__import__("sys").modules, "psutil", FakePsutil)

    snapshot = gate.resource_snapshot("preflight")

    assert snapshot["system_memory"]["unit"] == "GiB"
    assert snapshot["system_memory"]["available_gib"] == 113.59
    assert snapshot["system_memory"]["total_gib"] == 128.0
    assert snapshot["system_memory"]["available_gb"] == 113.59
    assert snapshot["system_memory"]["total_gb"] == 128.0


def test_dsv4_default_cache_tool_loop_gate_memory_preflight_skips_before_spawn(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate as gate

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 8.0},
        },
    )

    result = gate.run(
        Namespace(
            model="/models/dsv4",
            python=Path("/python"),
            port=8899,
            out=Path("unused.json"),
            timeout=1,
            request_timeout=1,
            min_free_gb=120,
            dry_run=False,
            pool_quant=False,
            disable_prefix_cache=False,
        )
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_free_memory"
    assert "cmd" in result
    assert "--dsv4-enable-prefix-cache" in result["cmd"]


def test_dsv4_default_cache_tool_loop_response_diagnostics_capture_incomplete_state():
    from tests.cross_matrix.run_dsv4_default_cache_tool_loop_gate import (
        response_diagnostics,
    )

    resp = {
        "id": "resp_test",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "incomplete",
                "finish_reason": "length",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "name": "write_file",
            },
        ],
    }

    assert response_diagnostics(resp) == {
        "response_status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_item_statuses": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "incomplete",
                "finish_reason": "length",
                "name": None,
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "finish_reason": None,
                "name": "write_file",
            },
        ],
    }


def test_dsv4_responses_one_tool_stop_gate_keeps_tools_available_on_final_turn():
    import inspect
    from tests.cross_matrix import run_dsv4_responses_one_tool_stop_gate as gate

    src = inspect.getsource(gate.run)

    assert "No more tools. Reply exactly DONE." in src
    assert '"previous_response_id": round1.get("id")' in src
    assert '"tools": TOOLS' in src
    assert '"tool_choice": "auto"' in src
    assert "round2_no_function_calls" in src
    assert "round2_tools_still_available" in src
    assert "native_cache" in src
