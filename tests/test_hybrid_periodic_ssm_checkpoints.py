# SPDX-License-Identifier: Apache-2.0
"""Tail-periodic hybrid SSM checkpoints (VMLX_HYBRID_CHECKPOINT_*).

Workload this exists for: a client (SillyTavern) that keeps the whole chat in
one long user message rewrites the prompt ~10K tokens before its end on every
turn. The paged KV chain still matches up to the divergence, but without an
SSM companion checkpoint below it the hybrid hit is unusable and the whole
~70K prompt re-prefills. These tests pin:

* the pure boundary planner (absolute, block-aligned, tail-limited, capped,
  never past the clean boundary, never past media, resumed-prefill offsets);
* the policy resolution (off by default, env first, config fallback only in
  explicit ``hybrid.ssm_recompute: checkpoint`` mode);
* the generator hook that merges them into ``_ssm_capture_boundaries_for``;
* the store side (SSD-only, nearest-first, backpressure instead of drops);
* an end-to-end run of the REAL chunked prefill lane on a tiny real
  Qwen3.5 GatedDeltaNet hybrid, publishing to a real SSD companion store and
  resuming a diverged second prompt from the stored checkpoint after a
  simulated restart.
"""

import os
import sys
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vmlx_engine.utils.hybrid_periodic_checkpoints import (  # noqa: E402
    DEFAULT_MAX_CHECKPOINTS,
    DEFAULT_TAIL_TOKENS,
    ENV_INTERVAL,
    ENV_MAX,
    ENV_TAIL,
    HARD_MAX_CHECKPOINTS,
    config_checkpoint_interval,
    plan_periodic_checkpoints,
    resolve_periodic_checkpoint_policy,
)

_ENV = (
    ENV_INTERVAL,
    ENV_TAIL,
    ENV_MAX,
    "VMLX_DISABLE_SSM_INLINE_CAPTURE",
    "VMLX_HYBRID_SSM_RECOMPUTE",
    "VMLX_ALLOW_HYBRID_CHUNKED_PREFILL",
    "VMLINUX_ALLOW_HYBRID_CHUNKED_PREFILL",
    "VMLX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT",
    "VMLINUX_ENABLE_NATIVE_MTP_HYBRID_TEXT_SPLIT",
    "VMLINUX_TIGHT_MEMORY_PREFILL_STEP_SIZE",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    from vmlx_engine.config import reset_config

    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Pure planner
# ---------------------------------------------------------------------------


def _plan(**overrides):
    args = dict(
        base_tokens=0,
        seq_len=71743,
        upper_local=71743 - 1 - 5,
        interval=2048,
        tail=32768,
        max_count=16,
        block_size=64,
    )
    args.update(overrides)
    return plan_periodic_checkpoints(**args)


def test_cold_70k_prompt_gets_sixteen_block_aligned_tail_checkpoints():
    # The measured Flash-Next prompt: 71,743 tokens, gen-prompt suffix 5.
    local = _plan()
    assert local == list(range(40960, 71681, 2048))
    assert len(local) == 16
    assert all(p % 64 == 0 for p in local)
    # Inside the last 32K before the clean boundary, never past it.
    assert local[0] >= 71737 - 32768
    assert local[-1] <= 71737


def test_resumed_prefill_plans_absolute_positions_and_returns_local_offsets():
    # A turn that resumed at a block-aligned checkpoint that is NOT a multiple
    # of the interval: positions stay on the absolute 2048 grid (the same keys
    # any other turn sharing the prefix would produce).
    base = 58368
    seq_len = 71743 - base
    local = _plan(base_tokens=base, seq_len=seq_len, upper_local=seq_len - 1 - 5)
    absolute = [base + b for b in local]
    assert absolute == list(range(59392, 71737 + 1, 2048))
    assert all(0 < b < seq_len - 1 for b in local)
    assert all(p % 2048 == 0 for p in absolute)


def test_interval_rounds_up_to_block_and_clean_boundary_is_a_hard_ceiling():
    # 2000 is not a block multiple: a checkpoint at 2000 could never pair with
    # a block-aligned KV trim, so the step becomes 2048.
    assert _plan(seq_len=5000, upper_local=4095, interval=2000, tail=0) == [2048]
    # Exactly on the clean boundary is allowed (the generator de-duplicates it
    # against the existing clean capture); one token past is not.
    assert _plan(seq_len=5000, upper_local=4096, tail=0) == [2048, 4096]
    # The final prompt token is forwarded alone for logits, so the last
    # possible boundary is seq_len - 1 (here 4095: 4096 is out).
    assert _plan(seq_len=4096, upper_local=9999, tail=0) == [2048]
    assert _plan(seq_len=4097, upper_local=9999, tail=0) == [2048, 4096]


def test_cap_thins_evenly_across_the_window_and_single_survivor_is_nearest_end():
    plan = _plan(seq_len=20000, upper_local=10245, interval=1024, tail=0, max_count=4)
    # Ten candidates 1024..10240; span-preserving thinning keeps both ends.
    assert plan == [1024, 4096, 7168, 10240]
    assert _plan(seq_len=20000, upper_local=10245, interval=1024, tail=0, max_count=1) == [10240]


def test_tail_window_media_limit_and_degenerate_inputs():
    # Tail window: only the last 4096 tokens before the clean boundary.
    assert _plan(seq_len=20000, upper_local=10245, interval=1024, tail=4096) == [
        7168, 8192, 9216, 10240
    ]
    # Media: the state at p covers [0, p); the first placeholder caps p.
    assert _plan(
        seq_len=20000, upper_local=10245, interval=1024, tail=0, safe_abs_limit=5000
    ) == [1024, 2048, 3072, 4096]
    for bad in (
        dict(interval=0),
        dict(max_count=0),
        dict(block_size=0),
        dict(seq_len=1),
        dict(upper_local=0),
        dict(safe_abs_limit=0),
    ):
        assert _plan(**bad) == [], bad
    # Nothing at or below the already-restored prefix.
    assert _plan(base_tokens=71680, seq_len=63, upper_local=57) == []


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_is_off_by_default_and_env_driven():
    policy = resolve_periodic_checkpoint_policy(environ={})
    assert not policy.enabled
    assert policy.source == "disabled"
    assert policy.tail == DEFAULT_TAIL_TOKENS
    assert policy.max_count == DEFAULT_MAX_CHECKPOINTS

    policy = resolve_periodic_checkpoint_policy(
        environ={ENV_INTERVAL: "2048", ENV_TAIL: "16384", ENV_MAX: "8"}
    )
    assert policy.enabled and policy.source == "env"
    assert (policy.interval, policy.tail, policy.max_count) == (2048, 16384, 8)

    # Junk is a no-op, never a crash; MAX is clamped; MAX<=0 disables.
    assert not resolve_periodic_checkpoint_policy(environ={ENV_INTERVAL: "lots"}).enabled
    assert (
        resolve_periodic_checkpoint_policy(
            environ={ENV_INTERVAL: "2048", ENV_MAX: "1000"}
        ).max_count
        == HARD_MAX_CHECKPOINTS
    )
    assert not resolve_periodic_checkpoint_policy(
        environ={ENV_INTERVAL: "2048", ENV_MAX: "0"}
    ).enabled


def test_policy_config_fallback_only_when_env_unset():
    policy = resolve_periodic_checkpoint_policy(environ={}, config_interval=lambda: 1024)
    assert policy.enabled and policy.source == "config" and policy.interval == 1024
    # An explicit env "0" is an explicit off, even when config asks for it.
    policy = resolve_periodic_checkpoint_policy(
        environ={ENV_INTERVAL: "0"}, config_interval=lambda: 1024
    )
    assert not policy.enabled
    # A broken config source is ignored.
    def _boom():
        raise RuntimeError("no config")

    assert not resolve_periodic_checkpoint_policy(environ={}, config_interval=_boom).enabled


def test_config_system_interval_requires_explicit_checkpoint_mode(monkeypatch):
    from vmlx_engine.config import reset_config

    # Defaults: ssm_recompute=full, checkpoint_interval_tokens=512 -> off.
    assert config_checkpoint_interval() is None
    monkeypatch.setenv("VMLX_HYBRID_SSM_RECOMPUTE", "checkpoint")
    reset_config()
    assert config_checkpoint_interval() == 512


# ---------------------------------------------------------------------------
# Generator hook (no model)
# ---------------------------------------------------------------------------


def _bare_generator(*, block_size=64, disk=True, image_token_id=None):
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._is_hybrid = True
    gen.block_aware_cache = SimpleNamespace(block_size=block_size)
    gen._ssm_state_cache = SimpleNamespace(disk_enabled=disk)
    config = {} if image_token_id is None else {"image_token_id": image_token_id}
    gen.model = SimpleNamespace(config=config)
    gen.language_model = SimpleNamespace(config={})
    # Keep the config-system fallback out of these tests.
    gen._periodic_ckpt_config_interval = None
    return gen


def _request(n_tokens=71743, cached=0, gpl=5, **extra):
    return SimpleNamespace(
        request_id="periodic-req",
        _original_token_ids=list(range(n_tokens)),
        _cached_tokens=cached,
        _gen_prompt_len=gpl,
        **extra,
    )


def test_capture_boundaries_unchanged_when_disabled():
    gen = _bare_generator()
    req = _request()
    assert gen._ssm_capture_boundaries_for(req, 71743, False, 71737) == [71680, 71737]
    assert req._periodic_ssm_checkpoint_abs == ()


def test_capture_boundaries_add_tail_checkpoints_when_enabled(monkeypatch):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator()
    req = _request()
    bounds = gen._ssm_capture_boundaries_for(req, 71743, False, 71737)
    assert bounds == sorted(set(range(40960, 71681, 2048)) | {71680, 71737})
    # 71680 is ALSO the existing block-aligned clean boundary: it stays a
    # clean checkpoint (scheduler handoff, RAM-eligible), not a periodic one.
    assert req._periodic_ssm_checkpoint_abs == tuple(range(40960, 71680, 2048))
    assert req._periodic_ssm_checkpoint_plan["interval"] == 2048
    assert req._periodic_ssm_checkpoint_plan["clean_boundary"] == 71737


def test_capture_boundaries_use_absolute_positions_after_resume(monkeypatch):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator()
    base = 58368
    seq_len = 71743 - base
    clean = seq_len - 1 - 5
    req = _request(cached=base)
    bounds = gen._ssm_capture_boundaries_for(req, seq_len, False, clean)
    assert req._periodic_ssm_checkpoint_abs == tuple(range(59392, 71680, 2048))
    assert [base + b for b in bounds] == sorted(
        set(range(59392, 71680, 2048)) | {71680, 71737}
    )


@pytest.mark.parametrize(
    "case",
    ["bypass", "no_disk_tier", "no_block_cache", "killswitch", "gpl_zero", "images"],
)
def test_periodic_checkpoints_stay_off_where_capture_is_unsafe_or_useless(monkeypatch, case):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator(disk=case != "no_disk_tier", block_size=0 if case == "no_block_cache" else 64)
    req = _request(_bypass_prefix_cache=(case == "bypass"))
    clean = 0 if case == "gpl_zero" else 71737
    if case == "killswitch":
        monkeypatch.setenv("VMLX_DISABLE_SSM_INLINE_CAPTURE", "1")
    bounds = gen._ssm_capture_boundaries_for(req, 71743, case == "images", clean)
    assert not (set(bounds) & set(range(40960, 71680, 2048)))
    assert not getattr(req, "_periodic_ssm_checkpoint_abs", ())


def test_periodic_checkpoints_never_absorb_media(monkeypatch):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator(image_token_id=7)
    tokens = list(range(100, 100 + 71743))
    tokens[50000] = 7  # an image placeholder deep in the history
    req = _request()
    req._original_token_ids = tokens
    gen._ssm_capture_boundaries_for(req, 71743, False, 71737)
    assert req._periodic_ssm_checkpoint_abs == tuple(range(40960, 50000, 2048))


def test_stale_plan_is_cleared_by_the_next_planning_call(monkeypatch):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator()
    req = _request()
    gen._ssm_capture_boundaries_for(req, 71743, False, 71737)
    assert req._periodic_ssm_checkpoint_abs
    monkeypatch.setenv(ENV_INTERVAL, "0")
    gen._ssm_capture_boundaries_for(req, 71743, False, 71737)
    assert req._periodic_ssm_checkpoint_abs == ()


# ---------------------------------------------------------------------------
# Store side
# ---------------------------------------------------------------------------


class _RecordingCache:
    def __init__(self, room=True):
        self.calls = []
        self.room = room

    def wait_for_disk_room(self, layers, timeout=5.0):
        self.calls.append(("wait", round(timeout, 3) > 0))
        return self.room

    def store(self, tokens, n, layers, *, is_complete, cache_extra_keys, retain_in_ram):
        self.calls.append(("store", n, is_complete, retain_in_ram, cache_extra_keys))
        self.last_store_accepted = True


def _checkpoints(*positions):
    return [(p, list(range(p)), [object()]) for p in positions]


def test_split_keeps_clean_checkpoints_out_of_the_periodic_set():
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    req = SimpleNamespace(_periodic_ssm_checkpoint_abs=(40960, 43008))
    base, periodic = MLLMBatchGenerator._split_periodic_ssm_checkpoints(
        req, _checkpoints(40960, 43008, 71680, 71737)
    )
    assert [cp[0] for cp in base] == [71680, 71737]
    assert [cp[0] for cp in periodic] == [40960, 43008]
    # No plan -> everything is a clean checkpoint, exactly as upstream.
    base, periodic = MLLMBatchGenerator._split_periodic_ssm_checkpoints(
        SimpleNamespace(), _checkpoints(71680, 71737)
    )
    assert [cp[0] for cp in base] == [71680, 71737] and periodic == []


def test_store_publishes_nearest_first_ssd_only_and_logs_one_line(caplog):
    import logging

    gen = _bare_generator()
    gen._ssm_state_cache = _RecordingCache()
    req = SimpleNamespace(
        request_id="r1",
        _periodic_ssm_checkpoint_abs=(40960, 43008, 45056),
        _periodic_ssm_checkpoint_plan={"interval": 2048, "tail": 32768, "max": 16,
                                        "source": "env", "cached_tokens": 0,
                                        "clean_boundary": 71737},
    )
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        stored = gen._store_periodic_ssm_checkpoints(
            req, _checkpoints(40960, 43008, 45056), {"salt": 1}
        )
    assert stored == [40960, 43008, 45056]
    stores = [c for c in gen._ssm_state_cache.calls if c[0] == "store"]
    assert [c[1] for c in stores] == [45056, 43008, 40960]
    assert all(c[2] is True and c[3] is False and c[4] == {"salt": 1} for c in stores)
    # Backpressure is consulted before every store.
    assert [c[0] for c in gen._ssm_state_cache.calls] == ["wait", "store"] * 3
    lines = [r.getMessage() for r in caplog.records if "periodic SSM checkpoints" in r.getMessage()]
    assert len(lines) == 1
    assert "stored 3/3 at [40960, 43008, 45056]" in lines[0]
    assert "interval=2048" in lines[0]
    # The plan is consumed.
    assert req._periodic_ssm_checkpoint_abs == ()


def test_store_skips_instead_of_dropping_terminal_entries_when_writer_stalls(caplog):
    import logging

    gen = _bare_generator()
    gen._ssm_state_cache = _RecordingCache(room=False)
    req = SimpleNamespace(request_id="r2")
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        stored = gen._store_periodic_ssm_checkpoints(
            req, _checkpoints(40960, 43008), None
        )
    assert stored == []
    assert not [c for c in gen._ssm_state_cache.calls if c[0] == "store"]
    assert any("skipped [40960, 43008]" in r.getMessage() for r in caplog.records)


def test_store_reports_l2_refusal_as_skipped(caplog):
    import logging

    class _RefusingCache(_RecordingCache):
        def store(self, *args, **kwargs):
            super().store(*args, **kwargs)
            self.last_store_accepted = False

    gen = _bare_generator()
    gen._ssm_state_cache = _RefusingCache()
    req = SimpleNamespace(request_id="r3")
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        assert gen._store_periodic_ssm_checkpoints(req, _checkpoints(40960), None) == []
    assert any(
        "stored 0/1 at []" in r.getMessage() and "skipped [40960]" in r.getMessage()
        for r in caplog.records
    )


def test_companion_store_l2_only_does_not_touch_ram_and_reports_acceptance(tmp_path):
    from mlx_lm.models.cache import ArraysCache

    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    layer = ArraysCache(size=2)
    layer.cache = [mx.ones((1, 3, 8)), mx.ones((1, 2, 4, 4))]
    disk = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    try:
        cache = SSMCompanionCache(max_entries=8, disk_store=disk)
        tokens = list(range(256))
        assert cache.store(tokens, 128, [layer], retain_in_ram=False) is None
        assert cache.last_store_accepted is True
        assert cache.size == 0  # never entered the RAM LRU
        cache.store(tokens, 192, [layer])
        assert cache.last_store_accepted is True
        assert cache.size == 1
        assert disk.wait_for_pending(timeout=10)
        assert sorted(disk.candidate_lengths(256)) == [128, 192]
        # No L2 tier: an L2-only store is a no-op, not a silent RAM insert.
        ram_only = SSMCompanionCache(max_entries=8, disk_store=None)
        ram_only._disk = None
        ram_only.store(tokens, 128, [layer], retain_in_ram=False)
        assert ram_only.last_store_accepted is False
        assert ram_only.size == 0
    finally:
        disk.shutdown(timeout=10)


def test_disk_wait_for_pending_room_waits_for_the_writer(tmp_path):
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    disk = SSMCompanionDiskStore(
        directory=tmp_path / "ssm", budget_bytes=1 << 30, max_pending_write_bytes=1000
    )
    try:
        assert disk.wait_for_pending_room(600, timeout=0.1)
        assert disk._reserve_pending_bytes(600)
        # 600 + 600 > 1000 while the first payload is pending.
        assert not disk.wait_for_pending_room(600, timeout=0.05)

        def _finish():
            time.sleep(0.1)
            disk._release_pending_bytes(600)
            with disk._write_condition:
                disk._write_condition.notify_all()

        t = threading.Thread(target=_finish)
        t.start()
        started = time.monotonic()
        assert disk.wait_for_pending_room(600, timeout=5.0)
        assert time.monotonic() - started < 4.0
        t.join()
        # Oversized single payloads get the exclusive admission store() grants.
        assert disk.wait_for_pending_room(10_000, timeout=0.05)
    finally:
        disk.shutdown(timeout=10)
    assert not disk.wait_for_pending_room(1, timeout=0.05)


# ---------------------------------------------------------------------------
# End to end: real tiny hybrid, real chunked lane, real SSD companion store
# ---------------------------------------------------------------------------


def _tiny_hybrid_generator():
    from tests.test_hybrid_chunked_prefill_equivalence import _build_lm

    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    lm = _build_lm("qwen3_5")

    class _VLM:
        def __init__(self):
            self.language_model = lm
            self.config = {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text"},
            }

    gen = MLLMBatchGenerator(
        model=_VLM(),
        processor=object(),
        prefill_step_size=64,
        enable_prefix_cache=False,
    )
    return gen, lm


def _abs_positions(start, end):
    pos = mx.arange(start, end, dtype=mx.int32).reshape(1, end - start)
    return mx.broadcast_to(pos[None, ...], (3, 1, end - start))


def _last_logits(lm, tokens, cache, start):
    out = lm(
        mx.array([tokens], dtype=mx.int32),
        cache=cache,
        position_ids=_abs_positions(start, start + len(tokens)),
    )
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    return logits[:, -1, :].astype(mx.float32)


def test_tiny_hybrid_chunked_prefill_captures_stores_and_resumes(monkeypatch, tmp_path):
    from mlx_lm.models.cache import KVCache

    from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    monkeypatch.setenv("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", "1")
    monkeypatch.setenv(ENV_INTERVAL, "128")
    monkeypatch.setenv(ENV_TAIL, "384")
    gen, lm = _tiny_hybrid_generator()
    assert gen._is_hybrid
    kv_positions = set(gen._hybrid_kv_positions)
    assert kv_positions  # full-attention layers 3 and 7

    disk = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    gen._ssm_state_cache = SSMCompanionCache(max_entries=0, disk_store=disk)
    gen.block_aware_cache = SimpleNamespace(block_size=64)
    gen._periodic_ckpt_config_interval = None

    mx.random.seed(11)
    prompt = mx.random.randint(0, 199, (600,)).tolist()
    req = MLLMBatchRequest(
        uid=0,
        request_id="periodic-e2e",
        prompt="",
        input_ids=mx.array([prompt], dtype=mx.int32),
        temperature=0.0,
    )
    req._original_token_ids = list(prompt)
    req._gen_prompt_len = 5
    req._cached_tokens = 0
    live_cache = lm.make_cache()
    gen._run_vision_encoding_inner(req, cache=live_cache)

    # clean = 600-1-5 = 594, block-aligned 576; tail 384 -> periodic from
    # ceil((594-384)/128)*128 = 256 on the absolute 128 grid.
    captured = [int(cp[0]) for cp in req._inline_ssm_checkpoints]
    assert captured == [256, 384, 512, 576, 594]
    assert req._periodic_ssm_checkpoint_abs == (256, 384, 512)

    base, periodic = gen._split_periodic_ssm_checkpoints(req, req._inline_ssm_checkpoints)
    assert [cp[0] for cp in base] == [576, 594]
    for boundary, tokens, layers in base:
        gen._ssm_state_cache.store(tokens, boundary, layers, is_complete=True)
    # Control: with only the upstream end-of-prompt checkpoints, a next turn
    # that diverges at 450 has nothing to resume from (both sit past 448).
    assert disk.wait_for_pending(timeout=30)
    diverged = prompt[:450] + [(t + 1) % 199 for t in prompt[450:]]
    upstream_only = SSMCompanionCache(max_entries=0, disk_store=disk)
    assert upstream_only.fetch_longest_prefix(diverged, 448) is None
    assert gen._store_periodic_ssm_checkpoints(req, periodic, None) == [256, 384, 512]
    assert disk.wait_for_pending(timeout=30)
    disk.shutdown(timeout=30)

    # Next turn: shares the first 450 tokens, then diverges. The paged chain
    # matches whole blocks up to 448; a FRESH process (new store, no RAM
    # index) must find the 384 periodic checkpoint through the disk scan.
    mx.random.seed(12)
    prompt2 = prompt[:450] + mx.random.randint(0, 199, (150,)).tolist()
    assert prompt2[450] != prompt[450] or prompt2[451] != prompt[451]
    disk2 = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    try:
        cold_store = SSMCompanionCache(max_entries=0, disk_store=disk2)
        hit = cold_store.fetch_longest_prefix(prompt2, 448)
        assert hit is not None
        ck_len, ck_states, ck_complete = hit
        assert ck_len == 384 and ck_complete is True
        assert cold_store.last_prefix_lookup["source"] == "partial_boundary_disk_l2"

        # Resume exactly the way the paged path pairs them: attention KV
        # trimmed to the checkpoint, recurrent state from the companion.
        resumed = []
        recurrent = iter(ck_states)
        for idx, layer in enumerate(live_cache):
            if idx in kv_positions:
                clone = KVCache()
                clone.keys = layer.keys[..., :ck_len, :]
                clone.values = layer.values[..., :ck_len, :]
                clone.offset = ck_len
                resumed.append(clone)
            else:
                resumed.append(next(recurrent))
        warm = _last_logits(lm, prompt2[ck_len:], resumed, ck_len)
        cold = _last_logits(lm, prompt2, lm.make_cache(), 0)
        # The chunked lane is bit-exact on this family (see
        # test_hybrid_chunked_prefill_equivalence.py), so the resumed turn
        # must reproduce the cold prefill exactly, not approximately.
        assert mx.array_equal(warm, cold).item()
    finally:
        disk2.shutdown(timeout=30)


def test_tiny_hybrid_resumed_one_shot_lane_captures_absolute_periodic_state(
    monkeypatch, tmp_path, caplog
):
    """A RESUMED turn (cached prefix > 0) on the default one-shot lane.

    On Flash-Next a resumed ~12K tail stays under the one-shot attention
    guard, so this lane must also leave periodic checkpoints behind or the
    turn after it falls back to an older, shorter checkpoint. The checkpoint
    keys are absolute (cached prefix + local offset) and the captured state
    must equal a cold prefill of that same prefix.
    """
    import logging

    from vmlx_engine.mllm_batch_generator import MLLMBatchRequest
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    monkeypatch.setenv(ENV_INTERVAL, "128")
    monkeypatch.setenv(ENV_TAIL, "384")
    gen, lm = _tiny_hybrid_generator()
    kv_positions = set(gen._hybrid_kv_positions)
    disk = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    gen._ssm_state_cache = SSMCompanionCache(max_entries=0, disk_store=disk)
    gen.block_aware_cache = SimpleNamespace(block_size=64)
    gen._periodic_ckpt_config_interval = None

    mx.random.seed(21)
    prompt = mx.random.randint(0, 199, (700,)).tolist()
    base = 384
    # The restored prefix: attention KV and recurrent state at 384, exactly
    # what a RESUME from a 384 checkpoint hands the prefill.
    restored = lm.make_cache()
    _last_logits(lm, prompt[:base], restored, 0)
    req = MLLMBatchRequest(
        uid=0,
        request_id="periodic-resumed",
        prompt="",
        input_ids=mx.array([prompt[base:]], dtype=mx.int32),
        temperature=0.0,
    )
    req._original_token_ids = list(prompt)
    req._gen_prompt_len = 5
    req._cached_tokens = base
    try:
        with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
            logits = gen._run_vision_encoding_inner(req, cache=restored)
        assert "Hybrid prefill path=one-shot" in caplog.text
        # clean = 384 + (316-1-5) = 694; local block-aligned 256 -> abs 640;
        # periodic on the absolute 128 grid inside [310, 694] past the
        # restored 384: 512 and 640, and 640 is already a clean boundary.
        assert [int(cp[0]) for cp in req._inline_ssm_checkpoints] == [512, 640, 694]
        assert req._periodic_ssm_checkpoint_abs == (512,)

        cold = lm.make_cache()
        _last_logits(lm, prompt[:512], cold, 0)
        (cp_512,) = [cp for cp in req._inline_ssm_checkpoints if int(cp[0]) == 512]
        assert cp_512[1] == prompt[:512]
        cold_recurrent = [
            layer for idx, layer in enumerate(cold) if idx not in kv_positions
        ]
        assert len(cp_512[2]) == len(cold_recurrent)
        for got, want in zip(cp_512[2], cold_recurrent):
            for a, b in zip(got.cache, want.cache):
                assert (a is None) == (b is None)
                if a is not None:
                    assert mx.array_equal(a, b).item()

        # The warm turn's own answer is unchanged by the extra split.
        cold_full = _last_logits(lm, prompt, lm.make_cache(), 0)
        warm_full = logits[:, -1, :].astype(mx.float32)
        assert mx.array_equal(warm_full, cold_full).item()

        _, periodic = gen._split_periodic_ssm_checkpoints(
            req, req._inline_ssm_checkpoints
        )
        assert gen._store_periodic_ssm_checkpoints(req, periodic, None) == [512]
        assert disk.wait_for_pending(timeout=30)
        assert 512 in disk.candidate_lengths(700)
    finally:
        disk.shutdown(timeout=30)


def test_process_prompts_publishes_clean_checkpoints_first_and_keeps_periodic_out_of_handoff():
    """Ordering contract of the post-prefill store block in _process_prompts.

    The terminal clean/required companions keep their upstream write order
    (they must never queue behind a burst of periodic entries), the periodic
    entries follow, and only the clean ones are handed to the scheduler's
    clean-boundary assembly (which matches the store boundary exactly), so the
    periodic Metal snapshots are released before decode instead of being held
    until the request finishes.
    """
    import inspect

    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

    src = inspect.getsource(MLLMBatchGenerator._process_prompts)
    split = src.index("self._split_periodic_ssm_checkpoints(req, _inline_checkpoints)")
    clean_loop = src.index(
        "for _inline_boundary, _inline_tokens, _inline_layers in _inline_checkpoints:"
    )
    periodic_store = src.index("self._store_periodic_ssm_checkpoints(", clean_loop)
    handoff = src.index("req._clean_boundary_recurrent = list(_inline_checkpoints)")
    assert split < clean_loop < periodic_store < handoff
    # Clean capture missing: periodic entries are still published and the
    # upstream fallback (deferred clean re-derive) still runs afterwards.
    fallback_store = src.index("if _periodic_checkpoints:", handoff)
    assert src.index("if _media_context_for_ssm:", fallback_store) > fallback_store


# ---------------------------------------------------------------------------
# Review hardening: the opt-in feature must never fail a turn
# ---------------------------------------------------------------------------


def test_store_consumes_snapshot_list_and_survives_a_raising_backpressure_hook(caplog):
    """A broken waiter costs its entry, not the request.

    ``_store_periodic_ssm_checkpoints`` runs inside the per-request prefill
    ``try`` of ``_process_prompts``; an escaping exception there is handled as
    a prefill failure for a turn that already sampled its first token.
    """
    import logging

    class _BrokenWaiterCache(_RecordingCache):
        def wait_for_disk_room(self, layers, timeout=5.0):
            raise RuntimeError("accounting exploded")

    gen = _bare_generator()
    gen._ssm_state_cache = _BrokenWaiterCache()
    req = SimpleNamespace(request_id="r4")
    snapshots = _checkpoints(40960, 43008)
    with caplog.at_level(logging.INFO, logger="vmlx_engine.mllm_batch_generator"):
        assert gen._store_periodic_ssm_checkpoints(req, snapshots, None) == []
    assert snapshots == []  # references released, not held until return
    assert not [c for c in gen._ssm_state_cache.calls if c[0] == "store"]
    assert any("stored 0/2 at []" in r.getMessage() for r in caplog.records)


def test_companion_disk_room_estimate_failure_is_a_refusal_not_an_exception(tmp_path):
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    class _ExplodingLayer:
        @property
        def resident_nbytes(self):
            raise RuntimeError("no accounting")

    disk = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    try:
        cache = SSMCompanionCache(max_entries=0, disk_store=disk)
        assert cache.wait_for_disk_room([_ExplodingLayer()], timeout=0.05) is False
    finally:
        disk.shutdown(timeout=10)


def test_planning_failure_degrades_to_upstream_boundaries(monkeypatch):
    monkeypatch.setenv(ENV_INTERVAL, "2048")
    gen = _bare_generator()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("planner exploded")

    gen._periodic_ssm_capture_boundaries = _boom
    req = _request()
    req._periodic_ssm_checkpoint_abs = (40960,)
    assert gen._ssm_capture_boundaries_for(req, 71743, False, 71737) == [71680, 71737]
    assert req._periodic_ssm_checkpoint_abs == ()


# ---------------------------------------------------------------------------
# End to end on the target family: tiny real qwen4_exp (GDN + QSA + PLE)
# ---------------------------------------------------------------------------


def test_tiny_qwen4_exp_chunked_prefill_periodic_checkpoints_resume_exactly(
    monkeypatch, tmp_path
):
    """Qwen3.8-Flash-Next's own layer mix, through the real chunked lane.

    The model this feature targets is qwen4_exp: GatedDeltaNet ArraysCache
    layers (one carrying the 4-entry PLE n-gram/conv aux state) plus QSA
    MiniMaxM3SparseCache attention layers. Pins, on that family:

    * periodic keys stay on the ABSOLUTE grid for a cold prefill and for a
      resumed prefill whose restored prefix is NOT block-aligned (100);
    * every captured periodic state equals a one-shot prefill of its prefix;
    * after an SSD round trip and a simulated restart, a prompt diverging
      at 450 finds the 384 checkpoint through the disk length scan, and the
      resumed turn (QSA KV sliced to 384 + restored recurrent state) matches
      a cold prefill of the diverged prompt.
    """
    from copy import deepcopy

    import numpy as np

    from tests.test_qwen4_exp_runtime import _randomize, _tiny_args
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
    from vmlx_engine.models.minimax_m3.cache import (
        MiniMaxM3SparseCache,
        clone_minimax_m3_sparse,
    )
    from vmlx_engine.models.qwen4_exp.language import LanguageModel
    from vmlx_engine.utils.ssm_companion_cache import SSMCompanionCache
    from vmlx_engine.utils.ssm_companion_disk_store import SSMCompanionDiskStore

    monkeypatch.setenv("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", "1")
    monkeypatch.setenv(ENV_INTERVAL, "128")
    monkeypatch.setenv(ENV_TAIL, "384")
    args = _tiny_args()
    lm = LanguageModel(args)
    _randomize(lm)
    mx.eval(lm.parameters())

    class _VLM:
        def __init__(self):
            self.language_model = lm
            self.config = {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
            }

    gen = MLLMBatchGenerator(
        model=_VLM(), processor=object(), prefill_step_size=64,
        enable_prefix_cache=False,
    )
    assert gen._is_hybrid
    kv_positions = set(gen._hybrid_kv_positions)
    template = lm.make_cache()
    assert {i for i, c in enumerate(template) if isinstance(c, MiniMaxM3SparseCache)} == kv_positions
    assert any(
        len(c.cache) == 4 for i, c in enumerate(template) if i not in kv_positions
    ), "tiny qwen4_exp must exercise the PLE aux recurrent state"
    gen.block_aware_cache = SimpleNamespace(block_size=64)
    gen._periodic_ckpt_config_interval = None

    rng = np.random.default_rng(5)
    prompt = rng.integers(8, args.vocab_size, size=(600,)).tolist()
    gpl = 5

    def _prefill(base, store):
        gen._ssm_state_cache = store
        live = lm.make_cache()
        if base:
            lm(mx.array([prompt[:base]]), cache=live)
            mx.eval([c.state for c in live])
        req = MLLMBatchRequest(
            uid=0, request_id=f"qwen4-periodic-{base}", prompt="",
            input_ids=mx.array([prompt[base:]], dtype=mx.int32), temperature=0.0,
        )
        req._original_token_ids = list(prompt[:-gpl])  # gpl-stripped, as served
        req._gen_prompt_len = gpl
        req._cached_tokens = base
        gen._run_vision_encoding_inner(req, cache=live)
        return req, live

    def _recurrent_at(length):
        cache = lm.make_cache()
        lm(mx.array([prompt[:length]]), cache=cache)
        return [c for i, c in enumerate(cache) if i not in kv_positions]

    def _assert_states_equal(got_layers, want_layers):
        assert len(got_layers) == len(want_layers)
        for got, want in zip(got_layers, want_layers):
            assert len(got.cache) == len(want.cache)
            for a, b in zip(got.cache, want.cache):
                assert (a is None) == (b is None)
                if a is not None:
                    assert a.shape == b.shape
                    assert mx.allclose(
                        a.astype(mx.float32), b.astype(mx.float32), atol=1e-5, rtol=1e-5
                    ).item()

    disk = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    store = SSMCompanionCache(max_entries=0, disk_store=disk)
    # clean = 600-1-5 = 594 absolute in both runs; tail 384 -> grid from 256.
    cold_req, cold_live = _prefill(0, store)
    assert [int(cp[0]) for cp in cold_req._inline_ssm_checkpoints] == [256, 384, 512, 576, 594]
    assert cold_req._periodic_ssm_checkpoint_abs == (256, 384, 512)
    resumed_req, _ = _prefill(100, store)
    # Local block alignment of the clean boundary gives 548 here (upstream
    # behaviour); the periodic keys stay on the absolute 128 grid.
    assert resumed_req._periodic_ssm_checkpoint_abs == (256, 384, 512)
    for req in (cold_req, resumed_req):
        for boundary, tokens, layers in req._inline_ssm_checkpoints:
            if boundary in req._periodic_ssm_checkpoint_abs:
                assert tokens == prompt[:boundary]
                _assert_states_equal(layers, _recurrent_at(boundary))

    _, periodic = gen._split_periodic_ssm_checkpoints(
        cold_req, cold_req._inline_ssm_checkpoints
    )
    assert gen._store_periodic_ssm_checkpoints(cold_req, periodic, None) == [256, 384, 512]
    assert disk.wait_for_pending(timeout=30)
    disk.shutdown(timeout=30)

    prompt2 = prompt[:450] + rng.integers(8, args.vocab_size, size=(150,)).tolist()
    assert prompt2[450:] != prompt[450:]
    disk2 = SSMCompanionDiskStore(directory=tmp_path / "ssm", budget_bytes=1 << 30)
    try:
        fresh = SSMCompanionCache(max_entries=0, disk_store=disk2)
        hit = fresh.fetch_longest_prefix(prompt2, 448)
        assert hit is not None
        ck_len, ck_states, ck_complete = hit
        assert (ck_len, ck_complete) == (384, True)
        assert fresh.last_prefix_lookup["source"] == "partial_boundary_disk_l2"
        _assert_states_equal(ck_states, _recurrent_at(384))

        resumed = []
        recurrent = iter(ck_states)
        for idx, layer in enumerate(cold_live):
            if idx in kv_positions:
                clone = clone_minimax_m3_sparse(layer, length=ck_len)
                assert clone is not None and int(clone.offset) == ck_len
                resumed.append(clone)
            else:
                resumed.append(deepcopy(next(recurrent)))
        warm = lm(mx.array([prompt2[ck_len:]]), cache=resumed).logits[:, -1, :]
        cold = lm(mx.array([prompt2]), cache=lm.make_cache()).logits[:, -1, :]
        mx.eval(warm, cold)
        assert float(mx.max(mx.abs(warm - cold)).item()) < 1e-4
    finally:
        disk2.shutdown(timeout=30)
