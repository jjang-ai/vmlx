from tests.cross_matrix.run_dsv4_route_mode_code_exactness import (
    CASE_NAMES,
    EXACT_CODE,
    analyze_content,
    case_body,
    dry_run,
    normalize_code_content,
    prompt_diagnostics,
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
    )
    assert off_diag["assistant_suffix_kind"] == "thinking_closed"
    assert off_diag["enable_thinking"] is False
    assert off_diag["prompt_endswith_assistant_think_close"] is True

    route, chat_on = case_body("chat_on")
    on_diag = prompt_diagnostics(
        route,
        chat_on,
        model_path="/unused/model",
        renderer=fake_renderer,
    )
    assert on_diag["assistant_suffix_kind"] == "thinking_open"
    assert on_diag["enable_thinking"] is True
    assert on_diag["prompt_endswith_assistant_think_open"] is True


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
    assert cases["chat_off_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["responses_off"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_closed"
    assert cases["responses_off_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["chat_on"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["chat_on_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["chat_on_rep1"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["responses_on"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["responses_on_rep1"]["request_overrides"]["repetition_penalty"] == 1.0
    assert cases["responses_on_rep1"]["prompt_diagnostics"]["assistant_suffix_kind"] == "thinking_open"
    assert cases["legacy_completion_raw"]["prompt_diagnostics"]["assistant_suffix_kind"] == "none"
