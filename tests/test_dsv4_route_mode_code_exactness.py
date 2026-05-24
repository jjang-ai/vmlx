from tests.cross_matrix.run_dsv4_route_mode_code_exactness import (
    case_body,
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
