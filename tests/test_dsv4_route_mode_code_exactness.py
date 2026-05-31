from tests.cross_matrix.run_dsv4_route_mode_code_exactness import (
    CASE_NAMES,
    EXACT_CODE,
    analyze_content,
    build_cmd,
    case_body,
    dry_run,
    effective_prompt_diagnostics,
    normalize_python_executable,
    normalize_code_content,
    prompt_diagnostics,
    run,
    selected_case_names,
)


def test_dsv4_code_exactness_probe_has_explicit_rep1_controls():
    route, chat_body = case_body("chat_off_rep1")
    assert route == "chat"
    assert chat_body["enable_thinking"] is False
    assert chat_body["repetition_penalty"] == 1.0
    assert chat_body["skip_prefix_cache"] is True

    route, responses_body = case_body("responses_off_rep1")
    assert route == "responses"
    assert responses_body["enable_thinking"] is False
    assert responses_body["repetition_penalty"] == 1.0
    assert responses_body["skip_prefix_cache"] is True


def test_dsv4_code_exactness_probe_has_true_direct_off_no_punct_variants():
    route, chat_body = case_body("chat_off_no_punct_rep1")
    assert route == "chat"
    assert chat_body["enable_thinking"] is False
    assert chat_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat_body["repetition_penalty"] == 1.0
    assert chat_body["skip_prefix_cache"] is True
    assert chat_body["messages"][0]["content"].startswith(
        "Return exactly this JavaScript code and no markdown fences\n"
    )

    route, responses_body = case_body("responses_off_no_punct_rep1")
    assert route == "responses"
    assert responses_body["enable_thinking"] is False
    assert responses_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert responses_body["repetition_penalty"] == 1.0
    assert responses_body["skip_prefix_cache"] is True
    assert responses_body["input"][0]["content"].startswith(
        "Return exactly this JavaScript code and no markdown fences\n"
    )


def test_dsv4_code_exactness_probe_has_bundle_default_sampling_cases():
    route, chat_body = case_body("chat_off_bundle_defaults")
    assert route == "chat"
    assert chat_body["enable_thinking"] is False
    assert chat_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat_body["skip_prefix_cache"] is True
    assert "temperature" not in chat_body
    assert "top_p" not in chat_body
    assert "repetition_penalty" not in chat_body

    route, responses_body = case_body("responses_off_bundle_defaults")
    assert route == "responses"
    assert responses_body["enable_thinking"] is False
    assert responses_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert responses_body["skip_prefix_cache"] is True
    assert "temperature" not in responses_body
    assert "top_p" not in responses_body
    assert "repetition_penalty" not in responses_body


def test_dsv4_code_exactness_probe_has_thinking_on_controls_for_chat_and_responses():
    route, chat_body = case_body("chat_on_rep1")
    assert route == "chat"
    assert chat_body["enable_thinking"] is True
    assert chat_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert chat_body["repetition_penalty"] == 1.0
    assert chat_body["skip_prefix_cache"] is True

    route, responses_body = case_body("responses_on")
    assert route == "responses"
    assert responses_body["enable_thinking"] is True
    assert responses_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "repetition_penalty" not in responses_body
    assert responses_body["skip_prefix_cache"] is True

    route, responses_rep_body = case_body("responses_on_rep1")
    assert route == "responses"
    assert responses_rep_body["enable_thinking"] is True
    assert responses_rep_body["chat_template_kwargs"] == {"enable_thinking": True}
    assert responses_rep_body["repetition_penalty"] == 1.0
    assert responses_rep_body["skip_prefix_cache"] is True


def test_dsv4_code_exactness_probe_records_prompt_rail_diagnostics():
    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens=False):
            assert add_special_tokens is False
            suffix = [128804]
            if prompt.endswith("<｜Assistant｜><think>"):
                suffix.append(128821)
            elif prompt.endswith("<｜Assistant｜></think>"):
                suffix.append(128822)
            return [11, 22, *suffix]

    def fake_renderer(messages, *, enable_thinking, reasoning_effort, model_path):
        assert messages[0]["role"] == "user"
        assert reasoning_effort is None
        suffix = "<｜Assistant｜><think>" if enable_thinking else "<｜Assistant｜></think>"
        return "<｜begin▁of▁sentence｜><｜User｜>copy code" + suffix

    route, chat_off = case_body("chat_off")
    off_diag = prompt_diagnostics(
        route,
        chat_off,
        model_path="/unused/model",
        renderer=fake_renderer,
        tokenizer=FakeTokenizer(),
    )
    assert off_diag["assistant_suffix_kind"] == "thinking_closed"
    assert off_diag["enable_thinking"] is False
    assert off_diag["prompt_endswith_assistant_think_close"] is True
    assert off_diag["token_diagnostics"]["available"] is True
    assert off_diag["token_diagnostics"]["tail_ids"][-2:] == [128804, 128822]
    assert off_diag["token_diagnostics"]["endswith_think_close_id"] is True

    route, chat_on = case_body("chat_on")
    on_diag = prompt_diagnostics(
        route,
        chat_on,
        model_path="/unused/model",
        renderer=fake_renderer,
        tokenizer=FakeTokenizer(),
    )
    assert on_diag["assistant_suffix_kind"] == "thinking_open"
    assert on_diag["enable_thinking"] is True
    assert on_diag["prompt_endswith_assistant_think_open"] is True
    assert on_diag["token_diagnostics"]["tail_ids"][-2:] == [128804, 128821]
    assert on_diag["token_diagnostics"]["endswith_think_open_id"] is True


def test_dsv4_code_exactness_probe_separates_requested_from_server_effective_rail():
    def fake_renderer(messages, *, enable_thinking, reasoning_effort, model_path):
        assert messages[0]["role"] == "user"
        suffix = "<｜Assistant｜><think>" if enable_thinking else "<｜Assistant｜></think>"
        return "<｜begin▁of▁sentence｜><｜User｜>copy code" + suffix

    route, chat_off = case_body("chat_off")
    requested_diag = prompt_diagnostics(
        route,
        chat_off,
        model_path="/unused/model",
        renderer=fake_renderer,
    )
    effective_diag = effective_prompt_diagnostics(
        route,
        chat_off,
        model_path="/unused/model",
        renderer=fake_renderer,
    )

    assert requested_diag["assistant_suffix_kind"] == "thinking_closed"
    assert requested_diag["enable_thinking"] is False
    assert effective_diag["assistant_suffix_kind"] == "thinking_closed"
    assert effective_diag["enable_thinking"] is False
    assert effective_diag["dsv4_policy_reason"] == "thinking_disabled"


def test_dsv4_code_exactness_effective_prompt_uses_top_level_thinking_precedence():
    def fake_renderer(messages, *, enable_thinking, reasoning_effort, model_path):
        suffix = "<｜Assistant｜><think>" if enable_thinking else "<｜Assistant｜></think>"
        return "<｜begin▁of▁sentence｜><｜User｜>copy code" + suffix

    route, chat_body = case_body("chat_off")
    chat_body["enable_thinking"] = False
    chat_body["chat_template_kwargs"] = {"enable_thinking": True}

    effective_diag = effective_prompt_diagnostics(
        route,
        chat_body,
        model_path="/unused/model",
        renderer=fake_renderer,
    )

    assert effective_diag["requested_enable_thinking"] is False
    assert effective_diag["enable_thinking"] is False
    assert effective_diag["assistant_suffix_kind"] == "thinking_closed"
    assert effective_diag["dsv4_policy_reason"] == "thinking_disabled"


def test_dsv4_code_exactness_probe_records_completion_chat_rail_effective_prompt():
    def fake_renderer(messages, *, enable_thinking, reasoning_effort, model_path):
        assert messages[0]["role"] == "user"
        assert "Return exactly this JavaScript code" in messages[0]["content"]
        assert reasoning_effort is None
        suffix = "<｜Assistant｜><think>" if enable_thinking else "<｜Assistant｜></think>"
        return "<｜begin▁of▁sentence｜><｜User｜>" + messages[0]["content"] + suffix

    route, completion_body = case_body("legacy_completion_raw")
    requested_diag = prompt_diagnostics(
        route,
        completion_body,
        model_path="/unused/model",
        renderer=fake_renderer,
    )
    effective_diag = effective_prompt_diagnostics(
        route,
        completion_body,
        model_path="/unused/model",
        renderer=fake_renderer,
    )

    assert requested_diag["assistant_suffix_kind"] == "none"
    assert effective_diag["assistant_suffix_kind"] == "thinking_closed"
    assert effective_diag["enable_thinking"] is False
    assert effective_diag["dsv4_policy_reason"] == "thinking_not_requested"


def test_dsv4_code_exactness_probe_separates_fence_from_identifier_corruption():
    fenced_exact = f"```javascript\n{EXACT_CODE}\n```"
    fenced_analysis = analyze_content(fenced_exact)

    assert normalize_code_content(fenced_exact) == EXACT_CODE
    assert fenced_analysis["exact"] is False
    assert fenced_analysis["normalized_exact"] is True
    assert fenced_analysis["has_markdown_fence"] is True
    assert fenced_analysis["missing"] == []
    assert fenced_analysis["corrupt_patterns"] == []

    corrupt = fenced_exact.replace("WebGLRenderer", "WebWebGLRenderer")
    corrupt_analysis = analyze_content(corrupt)

    assert corrupt_analysis["exact"] is False
    assert corrupt_analysis["normalized_exact"] is False
    assert "THREE.WebGLRenderer" in corrupt_analysis["missing"]
    assert "WebWebGLRenderer" in corrupt_analysis["corrupt_patterns"]


def test_dsv4_code_exactness_probe_records_near_miss_identifier_candidates():
    corrupt = EXACT_CODE.replace("THREE.PerspectiveCamera", "THREE.PerscpectiveCamera")
    corrupt = corrupt.replace("THREE.BoxGeometry", "THREE.BBoxGeometry")
    analysis = analyze_content(corrupt)

    assert "THREE.PerspectiveCamera" in analysis["missing"]
    assert "THREE.BoxGeometry" in analysis["missing"]
    assert {
        "required": "THREE.PerspectiveCamera",
        "candidate": "THREE.PerscpectiveCamera",
    } in analysis["identifier_corruption_candidates"]
    assert {
        "required": "THREE.BoxGeometry",
        "candidate": "THREE.BBoxGeometry",
    } in analysis["identifier_corruption_candidates"]


def test_dsv4_code_exactness_probe_dry_run_records_all_prompt_rails():
    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        max_tokens = 512
        dry_run = True

    artifact = dry_run(Args())

    assert artifact["status"] == "dry_run"
    assert artifact["dry_run"] is True
    assert artifact["case_count"] == len(CASE_NAMES)
    cases = {case["name"]: case for case in artifact["cases"]}
    assert set(cases) == set(CASE_NAMES)
    assert cases["chat_off"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert cases["chat_off"]["requested_prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert cases["chat_off"]["effective_prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert (
        cases["chat_off"]["effective_prompt_diagnostics"]["dsv4_policy_reason"]
        == "thinking_disabled"
    )
    assert cases["chat_off_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert (
        cases["chat_off_no_punct_rep1"]["prompt_diagnostics"][
            "assistant_suffix_kind"
        ]
        == "thinking_closed"
    )
    assert (
        cases["chat_off_no_punct_rep1"]["effective_prompt_diagnostics"][
            "dsv4_policy_reason"
        ]
        == "thinking_disabled"
    )
    assert (
        cases["chat_off_no_punct_rep1"]["request_overrides"][
            "repetition_penalty"
        ]
        == 1.0
    )
    assert cases["responses_off"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert cases["responses_off"]["effective_prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert cases["responses_off_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert (
        cases["responses_off_no_punct_rep1"]["effective_prompt_diagnostics"][
            "dsv4_policy_reason"
        ]
        == "thinking_disabled"
    )
    assert (
        cases["responses_off_no_punct_rep1"]["request_overrides"][
            "repetition_penalty"
        ]
        == 1.0
    )
    assert cases["chat_on"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["chat_on_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["chat_on_rep1"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["responses_on"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["responses_on_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["responses_on_rep1"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["legacy_completion_raw"]["prompt_diagnostics"]["assistant_suffix_kind"] == "none"
    assert (
        cases["legacy_completion_raw"]["effective_prompt_diagnostics"][
            "assistant_suffix_kind"
        ]
        == "thinking_closed"
    )
    assert (
        cases["legacy_completion_raw"]["effective_prompt_diagnostics"][
            "dsv4_policy_reason"
        ]
        == "thinking_not_requested"
    )


def test_dsv4_code_exactness_probe_preserves_venv_python_symlink(tmp_path):
    venv_bin = tmp_path / ".venv" / "bin"
    base_bin = tmp_path / "base" / "bin"
    venv_bin.mkdir(parents=True)
    base_bin.mkdir(parents=True)
    base_python = base_bin / "python3.13"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base_python)

    normalized = normalize_python_executable(".venv/bin/python", cwd=tmp_path)
    assert normalized == venv_python
    assert normalized.resolve() == base_python

    class Args:
        python = venv_python
        model = "/unused/model"
        port = 8861
        timeout = 420
        max_tokens = 512

    assert build_cmd(Args())[0] == str(venv_python)


def test_dsv4_code_exactness_probe_case_filter_rejects_unknown_cases():
    class Args:
        cases = "chat_max,legacy_completion_raw"

    assert selected_case_names(Args()) == ("chat_max", "legacy_completion_raw")

    class BadArgs:
        cases = "chat_max,not_real"

    try:
        selected_case_names(BadArgs())
    except ValueError as exc:
        assert "not_real" in str(exc)
    else:
        raise AssertionError("unknown DSV4 exactness case should fail")


def test_dsv4_code_exactness_probe_dry_run_honors_case_filter():
    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        max_tokens = 512
        dry_run = True
        base_url = "http://127.0.0.1:8886"
        cases = "chat_max,legacy_completion_raw"

    artifact = dry_run(Args())

    assert artifact["case_count"] == 2
    assert artifact["selected_cases"] == ["chat_max", "legacy_completion_raw"]
    assert [case["name"] for case in artifact["cases"]] == [
        "chat_max",
        "legacy_completion_raw",
    ]


def test_dsv4_code_exactness_probe_live_launch_honors_case_filter(monkeypatch):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    requested_paths: list[str] = []

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = "chat_max,legacy_completion_raw"

    class FakeProc:
        stdout = []

        def poll(self):
            return None

        def send_signal(self, _signal):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(gate.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(gate, "wait_health", lambda port, proc, timeout: {"status": "healthy"})
    monkeypatch.setattr(gate, "terminate", lambda proc: None)

    def fake_http_json(method, url, body=None, timeout=0):
        requested_paths.append(url.rsplit("/", 1)[-1])
        if url.endswith("/v1/completions"):
            return 200, {
                "choices": [{"text": gate.EXACT_CODE, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        return 200, {
            "choices": [
                {"message": {"content": gate.EXACT_CODE}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(gate, "http_json", fake_http_json)

    artifact = run(Args())

    assert artifact["case_count"] == 2
    assert [case["name"] for case in artifact["cases"]] == [
        "chat_max",
        "legacy_completion_raw",
    ]
    assert requested_paths == ["completions", "completions"]


def test_dsv4_code_exactness_probe_memory_preflight_skips_before_spawn(monkeypatch):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = "chat_max"
        min_free_gb = 120.0

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 74.0},
        },
        raising=False,
    )

    def fail_popen(*args, **kwargs):
        raise AssertionError("route exactness probe should not spawn under memory preflight")

    monkeypatch.setattr(gate.subprocess, "Popen", fail_popen)

    artifact = run(Args())

    assert artifact["status"] == "skipped"
    assert artifact["reason"] == "insufficient_free_memory"
    assert artifact["required_available_gb"] == 120.0
    assert artifact["available_gb"] == 74.0
    assert artifact["memory_gap_gb"] == 46.0
    assert artifact["did_not_launch"] is True
    assert artifact["launch_decision"] == "do_not_launch"
    assert artifact["selected_cases"] == ["chat_max"]
    assert artifact["telemetry"][0]["system_memory"]["available_gb"] == 74.0


def test_dsv4_code_exactness_probe_preflight_only_never_spawns_when_ready(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = ""
        min_free_gb = 120.0
        memory_preflight_only = True

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 130.0, "total_gb": 192.0},
        },
        raising=False,
    )

    def fail_popen(*args, **kwargs):
        raise AssertionError("preflight-only mode must not spawn DSV4")

    monkeypatch.setattr(gate.subprocess, "Popen", fail_popen)

    artifact = run(Args())

    assert artifact["status"] == "ready_to_launch"
    assert artifact["reason"] == "memory_preflight_floor_met"
    assert artifact["required_available_gb"] == 120.0
    assert artifact["available_gb"] == 130.0
    assert artifact["memory_gap_gb"] == 0.0
    assert artifact["did_not_launch"] is True
    assert artifact["launch_decision"] == "launch_allowed"
    assert artifact["case_count"] == len(CASE_NAMES)
    assert artifact["selected_cases"] == list(CASE_NAMES)


def test_dsv4_code_exactness_probe_uses_vm_stat_floor_over_psutil_available(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = "chat_max"
        min_free_gb = 120.0
        memory_preflight_only = True

    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                         9000000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 130.0, "total_gb": 192.0},
        },
        raising=False,
    )
    monkeypatch.setattr(gate, "_run_text", lambda cmd: vm_stat, raising=False)

    def fail_popen(*args, **kwargs):
        raise AssertionError("preflight-only mode must not spawn DSV4")

    monkeypatch.setattr(gate.subprocess, "Popen", fail_popen)

    artifact = run(Args())

    assert artifact["status"] == "skipped"
    assert artifact["reason"] == "insufficient_vm_stat_memory"
    assert artifact["available_gb"] == 130.0
    assert artifact["psutil_available_gap_gb"] == 0.0
    assert artifact["free_plus_speculative_purgeable_gb"] < 120.0
    assert artifact["strict_vm_stat_memory_gap_gb"] > 0
    assert artifact["launch_decision"] == "do_not_launch"
    assert "insufficient_memory" in artifact["launch_blockers"]


def test_dsv4_code_exactness_probe_memory_preflight_records_process_context(
    monkeypatch,
):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = "chat_max"
        min_free_gb = 120.0
        memory_preflight_only = True

    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                                  1.
Pages inactive:                           200000.
Pages speculative:                         50000.
Pages purgeable:                           25000.
"""
    top_processes = """  PID      RSS COMMAND
 1001 10485760 /Applications/vMLX.app/Contents/MacOS/vMLX
 1002  2097152 /usr/bin/python model-worker
"""
    active_processes = """1234 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4
5678 .venv/bin/python harmless.py
"""

    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 130.0, "total_gb": 192.0},
        },
        raising=False,
    )

    def fake_run_text(cmd):
        joined = " ".join(cmd)
        if cmd == ["vm_stat"]:
            return vm_stat
        if "ps -axo" in joined:
            return top_processes
        if "pgrep -af" in joined:
            return active_processes
        return ""

    monkeypatch.setattr(gate, "_run_text", fake_run_text, raising=False)

    def fail_popen(*args, **kwargs):
        raise AssertionError("preflight-only mode must not spawn DSV4")

    monkeypatch.setattr(gate.subprocess, "Popen", fail_popen)

    artifact = run(Args())

    assert artifact["top_memory_processes"][0]["pid"] == 1001
    assert artifact["top_memory_processes"][0]["rss_gb"] == 10.0
    assert artifact["top_memory_processes"][0]["command"].endswith("vMLX")
    assert artifact["active_heavy_process_count"] == 1
    assert artifact["active_heavy_processes"][0]["pid"] == 1234
    assert "active_heavy_process" in artifact["launch_blockers"]
    assert artifact["commands"]["top_memory_processes"].startswith("ps ")
    assert "pgrep -af" in artifact["commands"]["active_heavy_processes"]


def test_dsv4_code_exactness_probe_ignores_own_preflight_process():
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    own_pid = 4321
    active = f"""{own_pid} .venv/bin/python tests/cross_matrix/run_dsv4_route_mode_code_exactness.py --memory-preflight-only
5678 .venv/bin/python tests/cross_matrix/run_runtime_memory_stress_probe.py --row gemma4
"""

    processes = gate.parse_active_heavy_processes(active, ignore_pids={own_pid})

    assert [process["pid"] for process in processes] == [5678]


def test_dsv4_code_exactness_probe_main_treats_memory_skip_as_artifact_not_failure(
    monkeypatch,
    tmp_path,
):
    import tests.cross_matrix.run_dsv4_route_mode_code_exactness as gate

    class Args:
        python = "/tmp/python"
        model = "/unused/model"
        port = 8861
        timeout = 420
        request_timeout = 5
        max_tokens = 512
        dry_run = False
        base_url = None
        cases = "chat_max"
        min_free_gb = 120.0
        out = str(tmp_path / "preflight.json")

    monkeypatch.setattr(gate, "parse_args", lambda: Args())
    monkeypatch.setattr(
        gate,
        "resource_snapshot",
        lambda name, proc=None: {
            "name": name,
            "system_memory": {"available_gb": 74.0},
        },
    )

    def fail_popen(*args, **kwargs):
        raise AssertionError("route exactness probe should not spawn under memory preflight")

    monkeypatch.setattr(gate.subprocess, "Popen", fail_popen)

    assert gate.main() == 0
