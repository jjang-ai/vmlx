"""Each speculative lane owns its checkpoint policy, independent of other lanes."""
import pytest

from vmlx_engine import mimo_session
from vmlx_engine.dspark import session as dspark_session


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 100 if token == "<|im_start|>" else -1

    def encode(self, text, **kwargs):
        assert text == "system"
        return [101]


PROMPT = [100, 101, 7, 7, 100, 102] + [7] * 34 + [100, 103]


@pytest.fixture(autouse=True)
def policies(monkeypatch):
    monkeypatch.setenv("VMLX_MIMO_DFLASH_PREFIX_REUSE", "1")
    monkeypatch.setenv("VMLX_MIMO_DFLASH_CHUNK_SNAPSHOT_EVERY", "16")
    monkeypatch.setenv("VMLX_DSPARK_PREFIX_REUSE", "1")
    monkeypatch.setenv("VMLX_DSPARK_CHUNK_SNAPSHOT_EVERY", "8")
    monkeypatch.setattr(mimo_session, "SSD", object())
    monkeypatch.setattr(dspark_session, "SSD", None)


def test_mimo_reuse_does_not_depend_on_dspark_switch(monkeypatch):
    expected = mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0)
    assert expected == [("system", 4), ("chunk", 16), ("chunk", 32), ("boundary", 40)]
    monkeypatch.setenv("VMLX_DSPARK_PREFIX_REUSE", "0")
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0) == expected


def test_mimo_does_not_inherit_dspark_ssd_checkpoints(monkeypatch):
    from vmlx_engine import glm_dflash2

    monkeypatch.setattr(glm_dflash2, "ssd_only_ram_release_enabled", lambda: True)
    expected = mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0)
    monkeypatch.setattr(dspark_session, "SSD", object())
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0) == expected
    monkeypatch.setenv("VMLX_MIMO_DFLASH_CHUNK_SNAPSHOT_EVERY", "0")
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0) == [("system", 4), ("boundary", 40)]


@pytest.mark.parametrize("restored,expected", [
    (0, [("system", 4), ("chunk", 16), ("chunk", 32), ("boundary", 40)]),
    (20, [("chunk", 32), ("boundary", 40)]),
    (40, []),
])
def test_mimo_own_policy_and_restored_boundary(monkeypatch, restored, expected):
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, restored) == expected
    monkeypatch.setenv("VMLX_MIMO_DFLASH_PREFIX_REUSE", "0")
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, restored) == []


def test_mimo_ram_only_has_no_chunk_checkpoints(monkeypatch):
    monkeypatch.setattr(mimo_session, "SSD", None)
    assert mimo_session.snapshot_cuts(Tokenizer(), PROMPT, 0) == [("system", 4), ("boundary", 40)]
