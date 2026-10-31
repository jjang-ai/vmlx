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


def chooser(headroom_gib, configured=2048, line_gib=211.58, max_chunk=4096):
    """An Ultra-class chooser (two dies: widest 4096) unless max_chunk says otherwise."""
    line = int(line_gib * GiB)
    return G.PrefillChunkChooser(configured, gc_line=lambda: line, active=lambda: line - int(headroom_gib * GiB),
                                 max_chunk=max_chunk)


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
    assert c.choose(12_000, 0) == 4_096
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
    assert c.choose(12_000, 0) == 4_096       # 11,999 body rows -> 3 chunks of <= 4096 (4096 + 4096 + 3807)
    assert c.last["widest"] == 2 * G._DIE_MAX_CHUNK == 4096
    assert c.choose(32_845, 0) == 3_712       # 32,844 body rows -> 9 chunks: share 3650 rounded up to 128


def test_a_prompt_that_fits_runs_as_one_chunk():
    c = chooser(116.0)
    assert c.choose(1_382, 0) == 1_381        # the whole body in one chunk
    assert c.choose(4_097, 0) == 4_096
    assert c.choose(4_098, 0) == 2_176        # 4,097 body rows: two chunks (2176 + 1921), no 1-row tail


def test_one_chunk_only_when_the_rule_would_pick_that_width():
    tight = chooser(6.2)                       # JANG_2K on the Ultra: ~6.2 GiB under the GC line at rest
    w = tight.widest(int(6.2 * GiB), 1_900)
    assert w < 1_899
    assert tight.choose(1_900, 0) == 1_024    # two chunks (1024 + 875), not one 1,899-row chunk
    assert tight.choose(1_382, 0) == 1_381    # fits under the widest: one chunk


def test_width_follows_the_headroom():
    widths = [chooser(h).widest(int(h * GiB), 12_000) for h in (4.0, 5.0, 6.2, 7.2, 8.0, 20.0)]
    assert widths == sorted(widths)
    assert widths[-1] == 2 * G._DIE_MAX_CHUNK
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
        assert w == 1024                       # 99,999 body rows in 98 chunks of <= 1024
        assert c.last["rule"] == "adaptive"
    assert chooser(-10.0).choose(10_000_000, 0) <= G._MIN_CHUNK  # any length: a width, never an error


def test_a_step_set_below_the_default_bounds_the_width():
    assert chooser(116.0, configured=1024).choose(32_845, 0) <= 1024
    assert chooser(116.0, configured=1024).choose(1_382, 0) == 768  # 1,381 rows: 768 + 613
    assert chooser(116.0, configured=256).choose(32_845, 0) <= 256
    assert chooser(116.0, configured=2047).last is None
    assert chooser(116.0, configured=2047).choose(32_845, 0) <= 2047
    # the default (2048), unset, or larger: the chooser's own range
    for cfg in (2048, 0, None, 8192):
        c = chooser(116.0, configured=cfg)
        assert c.choose(32_845, 0) == 3_712 and c.last["widest"] == 2 * G._DIE_MAX_CHUNK
    assert chooser(6.2, configured=8192).choose(32_845, 0) < 2048   # still bounded by the headroom


@pytest.mark.parametrize("n", [2, 3, 513, 1_025, 1_900, 6_563, 12_000, 32_845, 100_190, 262_144])
@pytest.mark.parametrize("headroom", [4.0, 6.2, 7.2, 116.0])
def test_chunks_fewest_aligned_never_wider_than_widest(n, headroom):
    c = chooser(headroom)
    w = c.choose(n, 0)
    body = n - 1
    parts = chunks_of(body, w)
    widest = c.last["widest"]
    assert sum(parts) == body
    assert w <= widest
    if body <= widest:
        assert parts == [body]                   # fits: one chunk
    else:
        top = widest // G._ALIGN * G._ALIGN
        assert w % G._ALIGN == 0                 # aligned width
        assert len(parts) == -(-body // top)     # the fewest aligned chunks
        assert parts[-1] >= 1


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
    a.end(0, 2048, 32_000, 0, peak=int(1.3 * a.predicted_transient(2048, 32_000)))
    b = chooser(7.2)                           # a rebuilt generator sees the same scale
    assert b.scale() == pytest.approx(1.3, rel=1e-6)
    for _ in range(G._SCALE_WINDOW):
        b.choose(32_000, 0)
        b.end(0, 2048, 32_000, 0, peak=int(1.1 * b.predicted_transient(2048, 32_000)))
    assert len(G.PrefillChunkChooser._ratios) == G._SCALE_WINDOW
    assert b.scale() == pytest.approx(1.1, rel=1e-3)  # the 1.3 aged out


def test_learned_scale_is_the_median_so_one_sawtooth_peak_does_not_stick():
    c = chooser(6.2)
    for r in (1.02, 1.03, 1.25, 1.04):         # one request whose worst chunk sat on a transient peak
        c.choose(32_000, 0)
        c.end(0, 1369, 32_000, 0, peak=int(r * c.predicted_transient(1369, 32_000)))
    assert c.scale() == pytest.approx(1.035, abs=2e-3)


def test_learned_scale_never_goes_below_the_model():
    c = chooser(6.2)
    before = c.widest(int(6.2 * GiB), 100_190)
    for _ in range(G._SCALE_WINDOW):          # short prompts where the model has slack: ratios ~0.9
        c.choose(12_000, 0)
        c.end(0, 1536, 12_000, 0, peak=int(0.9 * c.predicted_transient(1536, 12_000)))
    assert c.scale() == 1.0
    assert c.widest(int(6.2 * GiB), 100_190) == before   # a 100k prompt is not made wider by short-prompt slack
    c.end(0, 2048, 32_000, 0, peak=int(0.01 * c.predicted_transient(2048, 32_000)))  # a misread peak
    assert c.scale() == 1.0


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


def test_never_narrower_than_the_fixed_step_unless_configured():
    for headroom in (0.0, 3.0, 5.0, 6.2, 7.2):
        for ctx in (1_000, 100_190, 524_288, 1_048_576):
            assert chooser(headroom).widest(int(headroom * GiB), ctx) >= 1024
    assert chooser(0.0, configured=512).widest(0, 1_048_576) == 512


def test_split_examples():
    assert G._split(1_899, 1_614) == 1_024      # 1024 + 875 (the unaligned 950 + 949 was 2 % slower)
    assert G._split(1_381, 1_617) == 1_381      # fits: one chunk
    assert G._split(11_999, 1_547) == 1_536     # 7 x 1536 + 1247
    assert G._split(100_189, 1_024) == 1_024
    assert G._split(5, 1_024) == 5
    assert G._split(3_000, 100) == 100          # a width under _ALIGN is used as is


@pytest.mark.parametrize("name, widest", [("Apple M5 Ultra", 4096), ("Apple M5 Max", 2048), ("Apple M4 Pro", 2048),
                                          ("", 2048)])
def test_widest_width_per_gpu_die(monkeypatch, name, widest):
    monkeypatch.setattr(G, "_device_info", lambda: {"device_name": name, "max_recommended_working_set_size": 1 << 40})
    assert G._gpu_max_chunk() == widest
    assert G.PrefillChunkChooser(2048, gc_line=lambda: 1 << 40, active=lambda: 1 << 30).choose(32_845, 0) <= widest


def test_unreadable_device_name_keeps_one_die(monkeypatch):
    def boom():
        raise RuntimeError("no device")
    monkeypatch.setattr(G, "_device_info", boom)
    assert G._gpu_max_chunk() == G._DIE_MAX_CHUNK


def test_max_class_gpu_with_room_runs_2048_rows():
    c = chooser(7.2, line_gib=102.14, max_chunk=2048)     # JANGHT on the 128 GB M5 Max
    assert c.choose(12_000, 0) == 2048                   # 11,999 rows: 6 x 2048 (5 x 2048 + 1759)
    assert c.choose(32_845, 0) == 2048
    assert c.choose(1_900, 0) == 1_899                   # fits: one chunk
