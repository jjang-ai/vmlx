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


# ─── DSV4 max-thinking fix slice (claude 2026-05-09) ──────────────────────
# Eric directive: "i realy nwat to be able to use dsv4 flash coherently long
# form max thinking man". Audit at docs/internal/AUDIT_dsv4_flash_long_form_max_thinking_2026_05_09.md
#
# Fix slice (A + B + D combined):
#   A. VMLX_DSV4_RAW_MAX=1 opt-in lets users actually request reasoning_effort=max
#      (default still downgrades to high for safety)
#   B. Default finalizer budget bumped from 512 -> 2048 so visible answers after
#      forced </think> are substantive
#   D. Capabilities API surfaces a compat warning when max is downgraded


def test_dsv4_normalize_effort_default_downgrades_max_to_high(monkeypatch):
    """A: default behavior unchanged. max -> high without env opt-in."""
    monkeypatch.delenv("VMLX_DSV4_RAW_MAX", raising=False)
    from vmlx_engine.server import _normalize_dsv4_reasoning_effort
    assert _normalize_dsv4_reasoning_effort("max") == "high"
    assert _normalize_dsv4_reasoning_effort("high") == "high"
    assert _normalize_dsv4_reasoning_effort("low") == "high"
    assert _normalize_dsv4_reasoning_effort(None) is None
    assert _normalize_dsv4_reasoning_effort("invalid") is None


def test_dsv4_normalize_effort_raw_max_env_passes_through(monkeypatch):
    """A: with VMLX_DSV4_RAW_MAX=1, max passes through as max."""
    from vmlx_engine.server import _normalize_dsv4_reasoning_effort
    monkeypatch.setenv("VMLX_DSV4_RAW_MAX", "1")
    assert _normalize_dsv4_reasoning_effort("max") == "max"
    # Other values still downgrade (env only un-blocks max, not low/medium)
    assert _normalize_dsv4_reasoning_effort("low") == "high"
    assert _normalize_dsv4_reasoning_effort("high") == "high"


def test_dsv4_encoder_adapter_raw_max_env_passes_through(monkeypatch):
    """A: server raw-max opt-in must survive the tokenizer shim adapter."""
    from vmlx_engine.loaders import dsv4_chat_encoder

    monkeypatch.setenv("VMLX_DSV4_RAW_MAX", "1")

    assert dsv4_chat_encoder._resolve_mode_and_effort(True, "max") == (
        "thinking",
        "max",
    )


def test_dsv4_effort_route_label_distinguishes_raw_max_from_downgrade():
    from vmlx_engine.server import _dsv4_reasoning_effort_route_label

    assert _dsv4_reasoning_effort_route_label("max", "max") == "raw_max"
    assert _dsv4_reasoning_effort_route_label("max", "high") == "downgraded"
    assert _dsv4_reasoning_effort_route_label("high", "high") is None


def test_dsv4_normalize_effort_raw_max_env_truthy_aliases(monkeypatch):
    """A: env accepts truthy aliases (true, yes) like other DSV4 envs."""
    from vmlx_engine.server import _normalize_dsv4_reasoning_effort
    for val in ("1", "true", "TRUE", "yes", "True"):
        monkeypatch.setenv("VMLX_DSV4_RAW_MAX", val)
        assert _normalize_dsv4_reasoning_effort("max") == "max", f"failed for {val!r}"
    for val in ("0", "false", "no", "", "off"):
        monkeypatch.setenv("VMLX_DSV4_RAW_MAX", val)
        assert _normalize_dsv4_reasoning_effort("max") == "high", f"failed for {val!r}"


def test_dsv4_finalizer_budget_default_bumped_to_2048(monkeypatch):
    """B: default visible-answer budget after forced </think> is 2048
    (was 512). Lets long-thinking turns still produce substantive answers."""
    monkeypatch.delenv("VMLX_DSV4_FINALIZER_TOKENS", raising=False)
    gen = _generator()
    assert gen._finalizer_budget() == 2048


def test_dsv4_finalizer_budget_env_override_still_works(monkeypatch):
    """B: env override path preserved for users wanting smaller/larger budgets."""
    gen = _generator()
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "256")
    assert gen._finalizer_budget() == 256
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "8192")
    assert gen._finalizer_budget() == 8192
    monkeypatch.setenv("VMLX_DSV4_FINALIZER_TOKENS", "garbage")
    assert gen._finalizer_budget() == 2048  # falls back to bumped default


def test_dsv4_compat_warning_emitted_when_max_downgraded(monkeypatch):
    """D: capabilities surface honestly notes max-downgrade unless env opt-in."""
    monkeypatch.delenv("VMLX_DSV4_RAW_MAX", raising=False)
    from vmlx_engine.server import _dsv4_max_compat_warning
    warning = _dsv4_max_compat_warning()
    assert warning is not None
    assert "max" in warning.lower()
    assert "high" in warning.lower() or "downgrad" in warning.lower()


def test_dsv4_compat_warning_silenced_when_raw_max_enabled(monkeypatch):
    """D: when user opts into raw max, no compat warning."""
    monkeypatch.setenv("VMLX_DSV4_RAW_MAX", "1")
    from vmlx_engine.server import _dsv4_max_compat_warning
    assert _dsv4_max_compat_warning() is None
