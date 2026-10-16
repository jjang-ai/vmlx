"""Nemotron-3.5 stamps capabilities.reasoning_modes {on, medium, off}; its template reads enable_thinking + medium_effort,
never reasoning_effort. The stamped effort policy must translate efforts into those kwargs on every API surface."""
import json
from types import SimpleNamespace

import pytest

from vmlx_engine import server

MODES = {"on": {"enable_thinking": True}, "medium": {"enable_thinking": True, "medium_effort": True},
         "off": {"enable_thinking": False}}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    (tmp_path / "jang_config.json").write_text(json.dumps({"capabilities": {"family": "nemotron_h", "reasoning_modes": MODES}}))
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    server._read_bundle_json.cache_clear() if hasattr(server._read_bundle_json, "cache_clear") else None
    return str(tmp_path)


def run(bundle, effort=None, enable_thinking=None, request=None):
    chat, ct = {}, {}
    if effort is not None:
        chat["reasoning_effort"] = effort
        ct["reasoning_effort"] = effort
    if enable_thinking is not None:
        ct["enable_thinking"] = enable_thinking
        chat["enable_thinking"] = enable_thinking
    server._apply_stamped_effort_policy(chat, ct, model_key=bundle, request=request)
    return chat, ct


def test_contract_advertises_medium_and_high(bundle):
    assert server._stamped_reasoning_effort_contract(bundle) == (("medium", "high"), "high")


def test_medium_sets_medium_effort(bundle):
    chat, ct = run(bundle, "medium")
    assert ct["medium_effort"] is True and ct["enable_thinking"] is True and chat["enable_thinking"] is True


def test_high_is_template_default_and_clears_medium(bundle):
    _, ct = run(bundle, "high")
    assert ct["enable_thinking"] is True and ct.get("medium_effort") is False


@pytest.mark.parametrize("effort", ["low", "minimal"])
def test_below_medium_takes_nearest_tier(bundle, effort):
    _, ct = run(bundle, effort)
    assert ct["medium_effort"] is True


def test_explicit_thinking_off_wins(bundle):
    _, ct = run(bundle, "medium", enable_thinking=False)
    assert ct["enable_thinking"] is False and "medium_effort" not in ct


def test_inferred_medium_from_generic_thinking_mode_is_template_default(bundle):
    req = SimpleNamespace(reasoning_effort="medium", thinking_mode="thinking", reasoning=None, chat_template_kwargs=None)
    chat, ct = run(bundle, "medium", request=req)
    assert "medium_effort" not in ct and "reasoning_effort" not in ct and "reasoning_effort" not in chat


def test_explicit_medium_with_request_is_medium(bundle):
    req = SimpleNamespace(reasoning_effort="medium", thinking_mode=None, reasoning=None, chat_template_kwargs=None)
    _, ct = run(bundle, "medium", request=req)
    assert ct["medium_effort"] is True


def test_no_effort_leaves_template_default(bundle):
    chat, ct = run(bundle)
    assert ct == {} and chat == {}
