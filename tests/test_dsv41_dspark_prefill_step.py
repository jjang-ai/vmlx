"""DSV4.1 DSpark prefill chunk width: chosen per request from MLX's allocator headroom (PrefillChunkChooser).

Fake allocator numbers throughout: the GC line and active memory are injected, nothing touches Metal except the
peak-counter calls of begin()/end(), which the tests bypass by passing the peak explicitly."""
import pytest

from vmlx_engine.utils import dsv41_dspark_generator as G

GiB = 1 << 30


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("VMLX_DSV41_DSPARK_PREFILL_STEP", raising=False)
    monkeypatch.setattr(G.PrefillChunkChooser, "_ratios", [])
    yield


def chooser(headroom_gib, configured=2048, line_gib=211.58):
    line = int(line_gib * GiB)
    return G.PrefillChunkChooser(configured, gc_line=lambda: line, active=lambda: line - int(headroom_gib * GiB))


def chunks_of(body, width):
    out, i = [], 0
    while i < body:
        out.append(min(width, body - i))
        i += width
    return out


# --- the inherited AR prefill (VL turns) keeps the fixed step --------------------------------------------------------
def test_ar_prefill_step_keeps_the_fixed_rule(monkeypatch):
    assert G._draft_prefill_step(2048) == 1024
    assert G._draft_prefill_step(512) == 512
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "2048")
    assert G._draft_prefill_step(2048) == 2048
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "junk")
    assert G._draft_prefill_step(2048) == 1024


# --- env override: the diagnostic pin wins over everything ------------------------------------------------------------
def test_env_override_pins_the_width(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "512")
    for headroom in (0.5, 6.2, 116.0):
        c = chooser(headroom)
        assert c.choose(1400, 0) == 512
        assert c.choose(100_000, 0) == 512
        assert c.last["rule"] == "override"
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "4096")
    assert chooser(116.0).choose(32_845, 0) == 4096  # a pin is not clipped to the chooser's range


def test_bad_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "junk")
    c = chooser(116.0)
    assert c.choose(12_000, 0) == 4_000
    assert c.last["rule"] == "adaptive"


# --- failed detection: the fixed rule of vMLX <= 1.6.82, never something stricter -------------------------------------
@pytest.mark.parametrize("gc_line, active", [
    (lambda: (_ for _ in ()).throw(RuntimeError("no device_info")), lambda: 100 * GiB),
    (lambda: None, lambda: 100 * GiB),
    (lambda: 0, lambda: 100 * GiB),
    (lambda: 200 * GiB, lambda: 0),
    (lambda: 200 * GiB, lambda: (_ for _ in ()).throw(RuntimeError("no active memory"))),
])
def test_failed_detection_keeps_the_old_rule(gc_line, active):
    c = G.PrefillChunkChooser(2048, gc_line=gc_line, active=active)
    assert c.choose(1400, 0) == 1400          # old one-chunk rule
    assert c.last["rule"] == "fallback"
    assert c.choose(2048, 0) == 2048
    assert c.choose(900, 0) == 1024           # already one chunk at the step
    assert c.choose(2049, 0) == 1024          # longer prompts: the step
    assert c.choose(1400, 64_500) == 1024     # long session: the step
    assert c.choose(100_000, 0) == 1024
    c512 = G.PrefillChunkChooser(512, gc_line=gc_line, active=active)
    assert c512.choose(100_000, 0) == 512     # a smaller configured step is kept


# --- the rule ---------------------------------------------------------------------------------------------------------
def test_plenty_of_room_runs_the_widest_width_in_equal_chunks():
    c = chooser(116.0)                         # JANGHT on the 256 GB Ultra
    assert c.choose(12_000, 0) == 4_000       # 11,999 body rows -> 3 chunks of <= 4096, equal
    assert c.last["widest"] == G._MAX_CHUNK == 4096
    assert c.choose(32_845, 0) == 3_650       # 32,844 body rows -> 9 chunks of <= 4096, equal


def test_a_prompt_that_fits_runs_as_one_chunk():
    c = chooser(116.0)
    assert c.choose(1_382, 0) == 1_381        # the whole body in one chunk
    assert c.choose(4_097, 0) == 4_096
    assert c.choose(4_098, 0) == 2_049        # 4,097 body rows: two equal chunks, no 1-row tail


def test_one_chunk_only_when_the_rule_would_pick_that_width():
    tight = chooser(6.2)                       # JANG_2K on the Ultra: ~6.2 GiB under the GC line at rest
    w = tight.widest(int(6.2 * GiB), 1_900)
    assert w < 1_899
    assert tight.choose(1_900, 0) == 950      # two equal chunks, not one 1,899-row chunk
    assert tight.choose(1_382, 0) == 1_381    # fits under the widest: one chunk


def test_width_follows_the_headroom():
    widths = [chooser(h).widest(int(h * GiB), 12_000) for h in (4.0, 5.0, 6.2, 7.2, 8.0, 20.0)]
    assert widths == sorted(widths)
    assert widths[-1] == G._MAX_CHUNK
    assert widths[0] == G._MIN_CHUNK


def test_width_narrows_as_the_context_grows():
    c = chooser(7.2)                           # JANGHT on the 128 GB Max
    ws = [c.widest(int(7.2 * GiB), ctx) for ctx in (1_000, 32_000, 100_000, 262_144, 524_288, 1_048_576)]
    assert ws == sorted(ws, reverse=True)
    assert ws[0] > ws[-1]


def test_no_room_keeps_the_fixed_step_never_refuses():
    for headroom in (2.0, 0.5, 0.0, -10.0):
        c = chooser(headroom)
        w = c.choose(100_000, 0)
        assert c.last["widest"] == G._MIN_CHUNK == 1024  # the fixed rule's step: never narrower
        assert w == 1021                       # 99,999 body rows in 98 equal chunks of <= 1024
        assert c.last["rule"] == "adaptive"
    assert chooser(-10.0).choose(10_000_000, 0) <= G._MIN_CHUNK  # any length: a width, never an error


def test_a_step_set_below_the_default_bounds_the_width():
    assert chooser(116.0, configured=1024).choose(32_845, 0) <= 1024
    assert chooser(116.0, configured=1024).choose(1_382, 0) == 691  # 1,381 rows in two chunks of <= 1024
    assert chooser(116.0, configured=256).choose(32_845, 0) <= 256
    assert chooser(116.0, configured=2047).last is None
    assert chooser(116.0, configured=2047).choose(32_845, 0) <= 2047
    # the default (2048), unset, or larger: the chooser's own range
    for cfg in (2048, 0, None, 8192):
        c = chooser(116.0, configured=cfg)
        assert c.choose(32_845, 0) == 3_650 and c.last["widest"] == G._MAX_CHUNK
    assert chooser(6.2, configured=8192).choose(32_845, 0) < 2048   # still bounded by the headroom


@pytest.mark.parametrize("n", [2, 3, 513, 1_025, 1_900, 6_563, 12_000, 32_845, 100_190, 262_144])
@pytest.mark.parametrize("headroom", [4.0, 6.2, 7.2, 116.0])
def test_chunks_are_near_equal_never_a_tiny_tail(n, headroom):
    c = chooser(headroom)
    w = c.choose(n, 0)
    body = n - 1
    parts = chunks_of(body, w)
    assert sum(parts) == body
    assert w <= max(c.last["widest"], 1)
    assert min(parts) >= max(parts) - len(parts)  # equal up to rounding
    assert len(parts) == -(-body // c.last["widest"])  # the fewest chunks the widest width allows


def test_single_token_prompt():
    c = chooser(116.0)
    assert c.choose(1, 0) == 4096             # no body rows: any width, the forward is the last token alone
    assert chooser(6.2).choose(1, 500_000) >= 1


# --- learning -----------------------------------------------------------------------------------------------------------
def test_measured_peak_scales_the_prediction():
    c = chooser(7.2)
    before = c.widest(int(7.2 * GiB), 32_000)
    w = c.choose(32_000, 0)
    pred = c.predicted_transient(w, 32_000)
    c.end(100 * GiB, w, 32_000, 0, peak=100 * GiB + int(1.5 * pred))  # 1.5x the model
    assert c.scale() == pytest.approx(1.5, rel=1e-6)
    after = c.widest(int(7.2 * GiB), 32_000)
    assert after < before
    assert c.describe()["ratios"] == [1.5]


def test_learned_scale_is_process_wide_and_windowed():
    a = chooser(7.2)
    a.choose(32_000, 0)
    a.end(0, 2048, 32_000, 0, peak=int(0.5 * a.predicted_transient(2048, 32_000)))
    b = chooser(7.2)                           # a rebuilt generator sees the same scale
    assert b.scale() == pytest.approx(0.5, rel=1e-6)
    for _ in range(G._SCALE_WINDOW):
        b.choose(32_000, 0)
        b.end(0, 2048, 32_000, 0, peak=int(0.9 * b.predicted_transient(2048, 32_000)))
    assert len(G.PrefillChunkChooser._ratios) == G._SCALE_WINDOW
    assert b.scale() == pytest.approx(0.9, rel=1e-3)  # the 0.5 aged out; the largest recent ratio wins


def test_no_learning_from_short_prefills_overrides_or_fallbacks(monkeypatch):
    c = chooser(7.2)
    c.choose(200, 0)
    c.end(0, 199, 200, 0, peak=50 * GiB)          # < _OBSERVE_MIN_ROWS rows
    assert G.PrefillChunkChooser._ratios == []
    c.end(None, 2048, 32_000, 0, peak=50 * GiB)   # begin() failed
    assert G.PrefillChunkChooser._ratios == []
    monkeypatch.setenv("VMLX_DSV41_DSPARK_PREFILL_STEP", "1024")
    c.choose(32_000, 0)
    c.end(0, 1024, 32_000, 0, peak=50 * GiB)      # pinned width: not the chooser's decision
    assert G.PrefillChunkChooser._ratios == []
    monkeypatch.delenv("VMLX_DSV41_DSPARK_PREFILL_STEP")
    f = G.PrefillChunkChooser(2048, gc_line=lambda: None, active=lambda: 1)
    f.choose(32_000, 0)
    f.end(0, 1024, 32_000, 0, peak=50 * GiB)      # fallback rule
    assert G.PrefillChunkChooser._ratios == []


def test_zero_or_negative_measurement_is_ignored():
    c = chooser(7.2)
    c.choose(32_000, 0)
    c.end(10 * GiB, 2048, 32_000, 0, peak=10 * GiB)
    assert G.PrefillChunkChooser._ratios == []


def test_module_under_test_is_this_tree():
    import os
    assert os.path.dirname(os.path.dirname(os.path.dirname(G.__file__))) == os.path.dirname(os.path.dirname(__file__))


def test_real_device_numbers_are_read_when_available():
    line = G._gc_line_bytes()
    assert line is None or line > 0
    assert G._active_bytes() >= 0


def test_learned_scale_has_a_floor():
    c = chooser(7.2)
    c.choose(32_000, 0)
    c.end(0, 2048, 32_000, 0, peak=int(0.01 * c.predicted_transient(2048, 32_000)))  # a misread peak
    assert c.scale() == G._SCALE_FLOOR


def test_never_narrower_than_the_fixed_step_unless_configured():
    for headroom in (0.0, 3.0, 5.0, 6.2, 7.2):
        for ctx in (1_000, 100_190, 524_288, 1_048_576):
            assert chooser(headroom).widest(int(headroom * GiB), ctx) >= 1024
    assert chooser(0.0, configured=512).widest(0, 1_048_576) == 512
