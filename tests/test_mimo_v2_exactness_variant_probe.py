import json

from tests.cross_matrix.run_mimo_v2_exactness_variant_probe import (
    build_cases,
    classify_case,
)


def test_mimo_exactness_probe_builds_literal_json_and_tool_cases():
    cases = build_cases(model="mimo-test")
    labels = [case["label"] for case in cases]

    assert labels == [
        "plain_exact_blue_cat",
        "plain_exact_sentinel",
        "json_blue_cat",
        "json_sentinel",
        "tool_blue_cat",
        "tool_sentinel_json_call",
    ]
    by_label = {case["label"]: case for case in cases}
    assert by_label["plain_exact_sentinel"]["payload"]["model"] == "mimo-test"
    assert "B7-CAT-09" in by_label["json_sentinel"]["payload"]["messages"][0]["content"]
    assert by_label["tool_sentinel_json_call"]["payload"]["tool_choice"] == "required"


def test_mimo_exactness_probe_accepts_exact_outputs():
    cases = {case["label"]: case for case in build_cases(model="mimo-test")}

    plain = classify_case(
        cases["plain_exact_sentinel"],
        {"code": 200, "body": {"choices": [{"text": "B7-CAT-09"}]}},
    )
    json_row = classify_case(
        cases["json_sentinel"],
        {
            "code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"status":"ok","value":"B7-CAT-09","count":3}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 20},
            },
        },
    )
    tool = classify_case(
        cases["tool_sentinel_json_call"],
        {
            "code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps({"value": "B7-CAT-09"})
                                    }
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        },
    )

    assert plain["pass"] is True
    assert json_row["pass"] is True
    assert tool["pass"] is True


def test_mimo_exactness_probe_rejects_literal_mutation_without_repair():
    cases = {case["label"]: case for case in build_cases(model="mimo-test")}

    plain = classify_case(
        cases["plain_exact_sentinel"],
        {"code": 200, "body": {"choices": [{"text": "B7CAT-09"}]}},
    )
    json_row = classify_case(
        cases["json_sentinel"],
        {
            "code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"status":"ok","value":"B7CAT-09","count":3}'
                        }
                    }
                ]
            },
        },
    )
    tool = classify_case(
        cases["tool_sentinel_json_call"],
        {
            "code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": json.dumps({"value": "B7CAT-09"})
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
        },
    )

    assert plain["pass"] is False
    assert plain["content"] == "B7CAT-09"
    assert json_row["pass"] is False
    assert json_row["parsed"] == {"status": "ok", "value": "B7CAT-09", "count": 3}
    assert tool["pass"] is False
    assert tool["parsed"] == {"value": "B7CAT-09"}
