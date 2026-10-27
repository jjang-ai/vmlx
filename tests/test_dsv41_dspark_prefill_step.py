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


def test_short_prompt_runs_as_one_chunk(monkeypatch):
    """A prompt (or uncached tail) of up to 2048 tokens on a short session is one chunk: each chunk passes over all
    routed experts, so 1024 + 380 paid that pass twice. Longer prompts and long sessions keep the step."""
    monkeypatch.delenv("VMLX_DSV41_DSPARK_PREFILL_STEP", raising=False)
    assert G._prompt_chunk(1400, 0, 1024) == 1400
    assert G._prompt_chunk(2048, 0, 1024) == 2048
    assert G._prompt_chunk(900, 0, 1024) == 1024        # already one chunk at the step
    assert G._prompt_chunk(2049, 0, 1024) == 1024       # longer prompts: the step
    assert G._prompt_chunk(1400, 64500, 1024) == 1024   # long session: the step (working set)
    assert G._prompt_chunk(1400, 64000, 1024) == 1400
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "512")
    assert G._prompt_chunk(1400, 0, 512) == 512         # an explicit step always wins
