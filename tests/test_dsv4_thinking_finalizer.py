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


def test_dsv4_generator_uses_caller_max_tokens_even_with_legacy_finalizer_env(monkeypatch):
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "7")
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], max_tokens=32)

    assert gen._effective_max_tokens(req) == 32

    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "8192")
    assert gen._effective_max_tokens(req) == 32


def test_dsv4_finish_reason_uses_stop_and_length_without_token_rewrite():
    gen = _generator()

    stop_req = _request([101, THINK_OPEN_ID], max_tokens=32)
    stop_req.out_tokens.append(DSV4_EOS_ID)
    gen._update_finish_reason_after_token(stop_req, DSV4_EOS_ID)
    assert stop_req.finish_reason == "stop"
    assert stop_req.out_tokens[-1] == DSV4_EOS_ID

    length_req = _request([101, THINK_OPEN_ID], out_tokens=[42], max_tokens=1)
    gen._update_finish_reason_after_token(length_req, 42)
    assert length_req.finish_reason == "length"
    assert length_req.out_tokens[-1] == 42


def test_dsv4_request_has_no_force_close_bookkeeping():
    req = _request([101, THINK_OPEN_ID], max_tokens=32)

    assert not hasattr(req, "forced_think_close")
    assert not hasattr(req, "forced_think_close_at")


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


# ─── DSV4 thinking contract ────────────────────────────────────────────────
# Max thinking is a real API rail, not an opt-in compatibility trap. The
# generator must not force a close/continuation rail behind the model.


def test_dsv4_normalize_effort_passes_max_without_env_guard(monkeypatch):
    from vmlx_engine.server import _normalize_dsv4_reasoning_effort
    monkeypatch.setenv("VMLX_DSV4_RAW_MAX", "0")
    assert _normalize_dsv4_reasoning_effort("max") == "max"
    assert _normalize_dsv4_reasoning_effort("high") == "high"
    assert _normalize_dsv4_reasoning_effort("low") == "high"
    assert _normalize_dsv4_reasoning_effort(None) is None
    assert _normalize_dsv4_reasoning_effort("invalid") is None


def test_dsv4_encoder_adapter_max_passes_through_without_env_guard(monkeypatch):
    from vmlx_engine.loaders import dsv4_chat_encoder

    monkeypatch.setenv("VMLX_DSV4_RAW_MAX", "0")

    assert dsv4_chat_encoder._resolve_mode_and_effort(True, "max") == (
        "thinking",
        "max",
    )


def test_dsv4_normalize_effort_ignores_raw_max_env_aliases(monkeypatch):
    from vmlx_engine.server import _normalize_dsv4_reasoning_effort
    for val in ("1", "true", "TRUE", "yes", "True", "0", "false", "no", "", "off"):
        monkeypatch.setenv("VMLX_DSV4_RAW_MAX", val)
        assert _normalize_dsv4_reasoning_effort("max") == "max", f"failed for {val!r}"


def test_dsv4_generator_does_not_force_close_even_with_legacy_finalizer_env(monkeypatch):
    monkeypatch.delenv("VMLX_DSV4_FINALIZER_TOKENS", raising=False)
    gen = _generator()
    req = _request([101, THINK_OPEN_ID], max_tokens=1)

    assert gen._effective_max_tokens(req) == 1
    assert not hasattr(gen, "_maybe_force_think_close")

    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "8192")
    assert gen._effective_max_tokens(req) == 1
    assert not hasattr(gen, "_maybe_force_think_close")
