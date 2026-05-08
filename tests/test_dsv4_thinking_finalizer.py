from vmlx_engine.utils.dsv4_batch_generator import (
    DSV4_ASSISTANT_ID,
    DSV4BatchGenerator,
    DSV4_EOS_ID,
    DSV4_USER_ID,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    _Request,
)


def _generator():
    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.stop_tokens = {DSV4_EOS_ID, 128803, 128804}
    return gen


def _request(prompt_tokens, *, out_tokens=None, max_tokens=32):
    return _Request(
        uid=1,
        prompt_tokens=list(prompt_tokens),
        context_tokens=list(prompt_tokens),
        cache=[],
        out_tokens=list(out_tokens or []),
        max_tokens=max_tokens,
    )


def test_dsv4_generator_forces_close_before_stop_inside_thinking(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "7")
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], max_tokens=32)

    token, finish = gen._maybe_force_think_close(req, DSV4_EOS_ID)

    assert token == THINK_CLOSE_ID
    assert finish is None
    assert req.forced_think_close is True
    assert req.forced_think_close_at == 0
    assert req.finalizer_max_tokens == 8

    req.out_tokens.append(token)
    gen._update_finish_reason_after_token(req, token)
    assert req.finish_reason is None


def test_dsv4_generator_forces_close_before_length_inside_thinking(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "5")
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], max_tokens=1)

    token, finish = gen._maybe_force_think_close(req, 42)

    assert token == THINK_CLOSE_ID
    assert finish is None
    assert req.finalizer_max_tokens == 6


def test_dsv4_generator_does_not_force_direct_rail():
    gen = _generator()
    req = _request([101, THINK_CLOSE_ID], max_tokens=32)

    token, finish = gen._maybe_force_think_close(req, DSV4_EOS_ID)

    assert token == DSV4_EOS_ID
    assert finish is None
    assert req.forced_think_close is False


def test_dsv4_generator_does_not_force_after_natural_close():
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], out_tokens=[11, THINK_CLOSE_ID], max_tokens=32)

    token, finish = gen._maybe_force_think_close(req, DSV4_EOS_ID)

    assert token == DSV4_EOS_ID
    assert finish is None
    assert req.forced_think_close is False


def test_dsv4_generator_finalizer_budget_bounds_extra_tokens(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "3")
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], max_tokens=1)

    token, _ = gen._maybe_force_think_close(req, 42)
    req.out_tokens.append(token)
    gen._update_finish_reason_after_token(req, token)
    assert req.finish_reason is None

    req.out_tokens.extend([201, 202, 203])
    gen._update_finish_reason_after_token(req, 203)
    assert req.finish_reason == "length"


def test_dsv4_generator_always_adds_role_boundary_stop_ids():
    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.model = None
    gen.max_tokens = 8
    gen.sampler = None
    gen.fallback_sampler = None
    gen.logits_processors = []
    gen._warmed_up = True
    gen.stop_tokens = set()

    DSV4BatchGenerator.__init__(
        gen,
        model=None,
        stop_tokens=[[DSV4_EOS_ID]],
    )

    assert DSV4_USER_ID in gen.stop_tokens
    assert DSV4_ASSISTANT_ID in gen.stop_tokens
