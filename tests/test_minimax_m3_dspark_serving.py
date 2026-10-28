# SPDX-License-Identifier: Apache-2.0
"""DSpark serving lane for MiniMax-M3 / M3.1: drafter admission (release MXFP8 layout and pre-quantized per-module
entries), the shared runtime loader (target families, Adaptive ladder, stamps advisory, env levers), the text-lane
attach, and DSparkTextGenerator == SingleBatchGenerator (the plain M3 serving path) on the same requests.

Models and drafters are the tiny synthetic ones of tests/test_minimax_m3_dspark.py (no real weights)."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from tests.test_minimax_m3_dspark import (  # noqa: E402,F401
    D, DRAFT_BLOCK, LAYERS, MX8, _cache_problems, _config, _drafter_config, _every_fifth, _exact_defaults, _ids,
    _model, _save_release_drafter, _Scripted, release_drafter_dir, target,
)
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402


def _bundle(tmp_path, *, stamp=None, drafter_cfg=None):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config()))
    if stamp is not None:
        (root / "jang_config.json").write_text(json.dumps({"speculative": stamp}))
    _save_release_drafter(root / "dspark", drafter_cfg)
    return root


@pytest.mark.parametrize("stamp, extra", [
    ({"method": "dspark", "max_draft": 2}, {}),
    ({"method": "dspark", "recommended_num_drafts": 1}, {"recommended_num_drafts": 1}),
    (None, {}),
])
def test_bundle_stamps_never_cap_the_ladder(tmp_path, target, stamp, extra, monkeypatch):
    from vmlx_engine.dspark.runtime import load_dspark_runtime

    root = _bundle(tmp_path, stamp=stamp, drafter_cfg=_drafter_config(**extra))
    rt = load_dspark_runtime(target.model, root)
    assert isinstance(rt["target"], MiniMaxM3DSparkTarget)
    assert rt["max_draft"] == 2 * DRAFT_BLOCK and rt["fixed_width"] is None
    # Only explicit env levers limit or pin the width (diagnostics).
    monkeypatch.setenv("VMLX_DSPARK_MAX_DRAFT", "3")
    monkeypatch.setenv("VMLX_DSPARK_WIDTH", "2")
    rt = load_dspark_runtime(target.model, root)
    assert rt["max_draft"] == 3 and rt["fixed_width"] == 2


# ----------------------------------------------------------------------------------------------- serving
def test_validate_bundled_draft_admits_the_minimax_layouts(tmp_path, release_drafter_dir):
    pytest.importorskip("jang_tools.format.aligned_safetensors")
    from vmlx_engine.bundled_draft import validate_bundled_draft
    from vmlx_engine.dspark import write_quantized_minimax_drafter

    root = _bundle(tmp_path)
    validate_bundled_draft(root / "dspark", root)                                   # as released (MXFP8)
    q = write_quantized_minimax_drafter(release_drafter_dir, tmp_path / "q")         # fc affine 8, layers affine 4
    quant = json.loads((q / "config.json").read_text())["quantization"]
    assert quant["fc"] == {"group_size": 64, "bits": 8} and quant["bits"] == 4      # per-module fc entry != top level
    validate_bundled_draft(q, root)
    mx8 = write_quantized_minimax_drafter(release_drafter_dir, tmp_path / "mx8",
                                          plan={"fc": MX8, "layers.*": None, "markov_head.*": None,
                                                "confidence_head.*": None})
    validate_bundled_draft(mx8, root)

    def broken(mutate, error):
        d = tmp_path / f"bad{len(list(tmp_path.iterdir()))}"
        _save_release_drafter(d, mutate(_drafter_config()))
        with pytest.raises(error):
            validate_bundled_draft(d, root)

    def taps(c):
        c["text_config"]["dspark_target_layer_ids"] = [1, LAYERS]
        return c

    def hidden(c):
        c["text_config"]["hidden_size"] = D // 2
        return c

    broken(taps, ValueError)
    with pytest.raises(ValueError):
        validate_bundled_draft(_save_release_drafter(tmp_path / "hid", hidden(_drafter_config())), root)
    cfg = json.loads((q / "config.json").read_text())
    cfg["quantization"]["fc"] = {"group_size": 64, "bits": 4}                       # wrong per-module entry
    (q / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        validate_bundled_draft(q, root)


def test_runtime_lanes_and_attach_outside_the_parameter_tree(tmp_path, monkeypatch):
    from vmlx_engine.dspark.runtime import attach_text_lane, load_dspark_runtime
    from vmlx_engine.models.mllm import _load_dspark_runtime

    model = _model(seed=9)
    root = _bundle(tmp_path)
    assert _load_dspark_runtime(model, root) is None          # the multimodal lane stays MiniCPM-only
    before = sorted(k for k, _ in tree_flatten(model.parameters()))
    rt = attach_text_lane(model, str(root))
    assert rt is not None and model._vmlx_dspark_runtime is rt
    assert sorted(k for k, _ in tree_flatten(model.parameters())) == before
    monkeypatch.setenv("VMLX_DSPARK", "0")
    assert load_dspark_runtime(model, root) is None
    monkeypatch.delenv("VMLX_DSPARK")
    (root / "dspark" / "model.safetensors").write_bytes(b"\0" * 16)
    fresh = _model(seed=9)
    assert attach_text_lane(fresh, str(root)) is None and getattr(fresh, "_vmlx_dspark_runtime", None) is None


def _drive(gen, prompts, *, caches=None, contexts=None, samplers=None, max_tokens=24, processors=None):
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts), caches=caches, all_tokens=contexts,
                      samplers=samplers, logits_processors=processors)
    first, = gen.next()[0]
    tokens, finish, snapshot = [first.token], first.finish_reason, first.prompt_cache_snapshot
    while finish is None:
        resp, = gen.next()[1]
        tokens.append(resp.token)
        finish = resp.finish_reason
    return uids[0], tokens, finish, snapshot


def _greedy():
    from vmlx_engine.sampling import make_minimax_m3_sampler

    sampler = make_minimax_m3_sampler(temp=0.0)
    sampler._vmlx_greedy = True
    return sampler


def test_text_generator_equals_single_batch_generator(tmp_path, release_drafter_dir):
    """The serving lane: DSparkTextGenerator == SingleBatchGenerator (the plain M3 path) on the same requests: tokens,
    finish reasons, the prompt snapshot it hands to the prefix cache, a prefix-cache hit continued on a restored
    cache, and requests that must take the plain path (logits processors)."""
    from vmlx_engine.dspark.runtime import load_dspark_runtime
    from vmlx_engine.models.minimax_m3.cache import clone_minimax_m3_sparse
    from vmlx_engine.utils.dspark_text_generator import DSparkTextGenerator
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    model = _model(seed=11)
    root = _bundle(tmp_path)
    rt = load_dspark_runtime(model, root)
    object.__setattr__(model, "_vmlx_dspark_runtime", rt)
    prompt = _ids(645, 61)
    plain = SingleBatchGenerator(model, sampler=_greedy(), max_tokens=24, prefill_step_size=256, stop_tokens=[])
    _, ref_tokens, ref_finish, ref_snap = _drive(plain, [prompt])
    # Script the drafter over the plain path's continuation so rounds accept partially.
    rt["drafter"] = _Scripted(rt["drafter"], prompt + ref_tokens, _every_fifth)
    gen = DSparkTextGenerator(model, sampler=_greedy(), max_tokens=24, prefill_step_size=256, stop_tokens=[])
    _, tokens, finish, snap = _drive(gen, [prompt])
    assert tokens == ref_tokens and finish == ref_finish == "length"
    assert ref_snap is not None and snap is not None and not _cache_problems(snap, ref_snap)
    stats = gen.last_dspark_stats
    assert stats["rounds"] > 0 and 0 < stats["accepted"] < stats["drafted"]

    # Prefix-cache hit: restore the prompt snapshot minus a tail, feed the rest (the scheduler's contract).
    cached = 600

    def restored(snapshot):
        out = []
        for c in snapshot:
            if hasattr(c, "idx_keys"):
                out.append(clone_minimax_m3_sparse(c, cached, copy_fn=lambda a: mx.array(a)))
            else:
                from mlx_lm.models.cache import KVCache
                k, v = c.state
                fresh = KVCache()
                fresh.state = (mx.array(k[..., :cached, :]), mx.array(v[..., :cached, :]))
                out.append(fresh)
        return out

    plain = SingleBatchGenerator(model, sampler=_greedy(), max_tokens=24, prefill_step_size=256, stop_tokens=[])
    _, hit_ref, _, _ = _drive(plain, [prompt[cached:]], caches=[restored(ref_snap)], contexts=[prompt[:cached]])
    drafter = rt["drafter"].inner
    _, hit_tokens, _, _ = _drive(gen, [prompt[cached:]], caches=[restored(snap)], contexts=[prompt[:cached]])
    assert hit_tokens == hit_ref and len(hit_ref) == 24
    # The previous request's drafter context covered the restored prefix: carried over. It starts where the cold
    # prompt's kept window started (the drafter's sliding window of 64 rows before the prompt end).
    window = drafter.sliding_window
    assert drafter.ctx_base == len(prompt) - window == gen._draft_memory[2]
    # Without a covering context the drafter starts empty at the restored position; the output is unchanged.
    gen._draft_memory = None
    _, cold_draft, _, _ = _drive(gen, [prompt[cached:]], caches=[restored(snap)], contexts=[prompt[:cached]])
    assert cold_draft == hit_ref and drafter.ctx_base == cached

    # Requests with logits processors take the plain path (and still match it).
    def bump(tokens_so_far, logits):
        return logits
    _, proc_tokens, _, _ = _drive(gen, [prompt], processors=[[bump]])
    assert proc_tokens == ref_tokens and gen._request is None


def test_text_generator_stops_like_the_plain_path(tmp_path):
    from vmlx_engine.dspark.runtime import load_dspark_runtime
    from vmlx_engine.utils.dspark_text_generator import DSparkTextGenerator
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    model = _model(seed=12)
    rt = load_dspark_runtime(model, _bundle(tmp_path))
    object.__setattr__(model, "_vmlx_dspark_runtime", rt)
    prompt = _ids(333, 71)
    free = _drive(SingleBatchGenerator(model, sampler=_greedy(), max_tokens=30, prefill_step_size=128,
                                       stop_tokens=[]), [prompt], max_tokens=30)[1]
    stop = free[9]
    for max_tokens in (1, 30):
        plain = SingleBatchGenerator(model, sampler=_greedy(), max_tokens=max_tokens, prefill_step_size=128,
                                     stop_tokens=[stop])
        ref = _drive(plain, [prompt], max_tokens=max_tokens)
        rt["drafter"] = _Scripted(getattr(rt["drafter"], "inner", rt["drafter"]), prompt + free, lambda p: False)
        gen = DSparkTextGenerator(model, sampler=_greedy(), max_tokens=max_tokens, prefill_step_size=128,
                                  stop_tokens=[stop])
        got = _drive(gen, [prompt], max_tokens=max_tokens)
        assert got[1:3] == ref[1:3]


@pytest.mark.parametrize("attached", [True, False])
def test_scheduler_picks_the_dspark_text_generator_only_when_a_runtime_is_attached(monkeypatch, attached):
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
    from vmlx_engine.request import SamplingParams
    from vmlx_engine.scheduler import Scheduler, SchedulerConfig

    class _FakeM3Model:
        def make_cache(self):
            return [MiniMaxM3SparseCache()]

    model = _FakeM3Model()
    if attached:
        model._vmlx_dspark_runtime = {"target": None, "drafter": None, "max_draft": 8, "cost_per_row": 0.12}
    scheduler = object.__new__(Scheduler)
    scheduler.model = model
    scheduler.config = SchedulerConfig(max_num_seqs=1)
    scheduler.memory_aware_cache = None
    scheduler.disk_cache = None
    scheduler.block_aware_cache = None
    scheduler._long_repetition_context = False
    scheduler._uses_m3_msa_cache = True
    monkeypatch.setattr(Scheduler, "_get_stop_tokens", lambda _self: set())
    gen = scheduler._create_batch_generator(SamplingParams(max_tokens=8, temperature=0.0))
    assert type(gen).__name__ == ("DSparkTextGenerator" if attached else "SingleBatchGenerator")
    assert getattr(gen.sampler, "_vmlx_greedy", False)


def test_text_generator_sampled_requests(tmp_path, monkeypatch):
    """Sampled requests: DSpark draws every token with the request's own sampler (sample-then-match);
    VMLX_DSPARK_SAMPLED=0 sends them to the plain path, which then equals SingleBatchGenerator draw for draw."""
    from vmlx_engine.dspark.runtime import load_dspark_runtime
    from vmlx_engine.sampling import make_minimax_m3_sampler
    from vmlx_engine.utils.dspark_text_generator import DSparkTextGenerator
    from vmlx_engine.utils.single_batch_generator import SingleBatchGenerator

    model = _model(seed=13)
    rt = load_dspark_runtime(model, _bundle(tmp_path))
    object.__setattr__(model, "_vmlx_dspark_runtime", rt)
    prompt = _ids(300, 91)

    def seeded():
        return make_minimax_m3_sampler(temp=0.8, top_p=0.95, seed=7)

    ref = _drive(SingleBatchGenerator(model, sampler=seeded(), max_tokens=20, prefill_step_size=128, stop_tokens=[]),
                 [prompt], max_tokens=20)[1]
    monkeypatch.setenv("VMLX_DSPARK_SAMPLED", "0")
    gen = DSparkTextGenerator(model, sampler=seeded(), max_tokens=20, prefill_step_size=128, stop_tokens=[])
    assert _drive(gen, [prompt], max_tokens=20)[1] == ref and gen.last_dspark_stats is None
    monkeypatch.delenv("VMLX_DSPARK_SAMPLED")
    gen = DSparkTextGenerator(model, sampler=seeded(), max_tokens=20, prefill_step_size=128, stop_tokens=[])
    tokens = _drive(gen, [prompt], max_tokens=20)[1]
    assert len(tokens) == 20 and all(0 <= t < 1024 for t in tokens)
    assert gen.last_dspark_stats is not None and gen.last_dspark_stats["tokens"] == 20
