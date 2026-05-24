from tests.cross_matrix.run_dsv4_route_mode_code_exactness import case_body


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
