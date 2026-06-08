from vmlx_engine.reasoning.think_xml_parser import ThinkXmlReasoningParser


def test_think_xml_no_tags_with_think_in_prompt_stays_visible_content():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning(
        "The answer is ready.\n\nFINAL=OK"
    )

    assert reasoning is None
    assert content == "The answer is ready.\n\nFINAL=OK"


def test_think_xml_streaming_no_tags_with_think_in_prompt_stays_visible_content():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    delta = parser.extract_reasoning_streaming("", "FINAL=OK", "FINAL=OK")

    assert delta is not None
    assert delta.reasoning is None
    assert delta.content == "FINAL=OK"


def test_think_xml_explicit_tags_still_extract_reasoning():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("<think>brief</think>FINAL=OK")

    assert reasoning == "brief"
    assert content == "FINAL=OK"


def test_think_xml_only_end_tag_still_extracts_implicit_reasoning():
    parser = ThinkXmlReasoningParser()
    parser.reset_state(think_in_prompt=True)

    reasoning, content = parser.extract_reasoning("brief</think>FINAL=OK")

    assert reasoning == "brief"
    assert content == "FINAL=OK"
