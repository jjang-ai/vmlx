"""Keep the API quality gate's scorer independent of model output."""

import importlib.util
from pathlib import Path

import pytest


def gate():
    path = Path(__file__).resolve().parents[1] / "bench/qwen4_quality_gate.py"
    spec = importlib.util.spec_from_file_location("qwen4_quality_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_scorer_distinguishes_false_and_zero():
    case = {"kind": "json", "expected": {"flag": False}}
    assert gate().grade(case, {"content": '{"flag": false}'})[0]
    assert not gate().grade(case, {"content": '{"flag": 0}'})[0]


def test_code_scorer_executes_behavior_checks():
    case = {"kind": "code", "name": "increment", "checks": [([0], 1), ([-3], -2)]}
    assert gate().grade(case, {"content": "def increment(x):\n return x + 1"})[0]
    assert not gate().grade(case, {"content": "def increment(x):\n return 1"})[0]


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef f(): return 1",
        "def f(): return (1).__class__",
        "def f(): return __builtins__",
    ],
)
def test_code_scorer_rejects_unrestricted_execution(code):
    assert not gate().grade(
        {"kind": "code", "name": "f", "checks": [([], 1)]}, {"content": code}
    )[0]


def test_tool_scorer_checks_name_and_arguments():
    case = {"kind": "tool", "expected": {"city": "Oslo", "unit": "celsius"}}
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Oslo","unit":"celsius"}',
                }
            }
        ]
    }
    assert gate().grade(case, message)[0]
    message["tool_calls"][0]["function"]["name"] = "wrong_tool"
    assert not gate().grade(case, message)[0]
