"""With the DSpark draft resident, DSV4.1 prefill chunks are capped at 1024 tokens (Metal working set)."""
from vmlx_engine.utils import dsv41_dspark_generator as G


def test_draft_prefill_step(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_DSPARK_PREFILL_STEP", raising=False)
    assert G._draft_prefill_step(2048) == 1024
    assert G._draft_prefill_step(512) == 512
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "2048")
    assert G._draft_prefill_step(2048) == 2048
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "junk")
    assert G._draft_prefill_step(2048) == 1024
