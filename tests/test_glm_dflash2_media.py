"""GLM-5.3-Flash DFlash2 lane with an IMAGE plan (vmlx_engine.glm_dflash2): prefill from merged input embeddings,
prefix reuse keyed per image.

Fixture: the tiny random glm5_next of test_glm_dflash2_lossless (KDA + DSA layers, mHC streams). An "image" is a run
of placeholder ids whose input rows are replaced by random feature vectors, exactly like
glm5_next.vlm.Model.get_input_embeddings scatters vision features; the plan's key ids replace that run with ids
derived from the image (here: a per-image base above the int32 range).
Oracles: speculative tokens equal greedy AR over the SAME input embeddings (near-ties excepted) and every emitted
token is the argmax of ONE cache-free forward over the prompt embeddings + emitted tokens; warm (restored) runs equal
cold runs token for token; the vision tower (plan materialization) does not run when the restored prefix already
covers every image, and runs when it does not.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("dflash.model_mlx")

from tests.test_glm_dflash2_lossless import TAPS, TOL, ScriptedDraft, _target  # noqa: E402

from vmlx_engine.glm_dflash2 import GLMDFlashTarget, SpecStats, generate_dflash2  # noqa: E402

IMG = 3                                         # placeholder id (the real one is config.image_token_id)


def _prompt(text_a, n_img, text_b):
    ids = list(text_a) + [IMG] * n_img + list(text_b)
    return ids, len(text_a), len(text_a) + n_img


def _plan(model, ids, first, end, image_seed, calls, key_base=None):
    """An image plan as MLXMultimodalLM._dflash2_media_plan returns it (GLM: embeddings only)."""
    def materialize():
        calls.append(image_seed)
        mx.random.seed(image_seed)
        emb = model.model.embed_tokens(mx.array([ids]))
        feats = mx.random.normal((1, end - first, emb.shape[-1])).astype(emb.dtype) * 0.5
        emb = mx.concatenate([emb[:, :first], feats, emb[:, end:]], axis=1)
        mx.eval(emb)
        return {"embeds": emb, "positions": None, "rope_delta": 0}

    key = None
    if key_base is not None:
        key = list(ids)
        key[first:end] = [key_base + j for j in range(end - first)]
    return {"prompt_tokens": list(ids), "key_tokens": key, "materialize": materialize, "embeds": None,
            "positions": None, "rope_delta": None, "media_end": end, "first_media_index": first}


def _embeds(plan):
    return plan["materialize"]()["embeds"]


def _ar_embeds(model, embeds, n):
    c = model.make_cache()
    lg = model(None, cache=c, inputs_embeds=embeds)[0, -1]
    toks, rows = [], []
    for _ in range(n):
        t = int(mx.argmax(lg).item()); toks.append(t); rows.append(np.array(lg.astype(mx.float32)))
        lg = model(mx.array([[t]]), cache=c)[0, -1]
    return toks, rows


def _check(model, embeds, got, ref, rows):
    div = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
    if div is not None:
        margin = float(rows[div][ref[div]] - rows[div][got[div]])
        assert margin < TOL, f"token {got[div]} != AR {ref[div]} at {div}, AR margin {margin:.3f}"
    full = mx.concatenate([embeds, model.model.embed_tokens(mx.array([got]))], axis=1)
    lg = np.array(model(None, inputs_embeds=full)[0].astype(mx.float32))[embeds.shape[1] - 1:][: len(got)]
    bad = [(i, float(lg[i].max() - lg[i][t])) for i, t in enumerate(got) if int(lg[i].argmax()) != t]
    assert all(m < TOL for _, m in bad) and len(bad) <= 3, f"not the target's argmax over the image embeddings: {bad}"


def _gen(model, target, plan, n, store=None, key=True):
    ref, _ = _ar_embeds(model, _embeds(plan), n)
    st = SpecStats()
    got = list(generate_dflash2(target, ScriptedDraft(ref, 4), plan["prompt_tokens"], n, eos=set(), stats=st,
                                prefix_store=store, media=plan, key_ids=plan["key_tokens"] if key else None))
    return got, st


def test_image_prefill_is_lossless_and_uses_the_embeddings():
    model = _target(11)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids, first, end = _prompt(range(5, 22), 9, range(30, 41))          # 37 tokens, image rows 17..25
        calls = []
        plan = _plan(model, ids, first, end, 101, calls, key_base=2 ** 31 + 7)
        emb = _embeds(plan)
        ref, rows = _ar_embeds(model, emb, 24)
        calls.clear()
        st = SpecStats()
        got = list(generate_dflash2(target, ScriptedDraft(ref, 4), ids, 24, eos=set(), stats=st, media=plan,
                                    key_ids=plan["key_tokens"]))
        assert calls == [101] and st.vision_tower
        _check(model, emb, got, ref, rows)
        # fail closed: the image rows must change the answer (otherwise this test proves nothing about embeddings)
        text_ref, _ = _ar_embeds(model, model.model.embed_tokens(mx.array([ids])), 24)
        assert text_ref != ref
    finally:
        target.close()


def test_text_after_image_restores_without_running_the_vision_tower():
    from vmlx_engine.glm_dflash2 import PrefixStore
    model = _target(12)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids1, first, end = _prompt(range(5, 22), 9, range(30, 41))
        ids2 = ids1 + list(range(60, 77))                                   # the follow-up turn: same image, more text
        c1, c2, cold_calls = [], [], []
        p1 = _plan(model, ids1, first, end, 101, c1, key_base=2 ** 31 + 7)
        p2 = _plan(model, ids2, first, end, 101, c2, key_base=2 ** 31 + 7)
        cold, _ = _gen(model, target, _plan(model, ids2, first, end, 101, cold_calls, key_base=2 ** 31 + 7), 20)
        store = PrefixStore(4)
        _gen(model, target, p1, 20, store)
        c2.clear()
        got, st = _gen(model, target, p2, 20, store)
        assert c2 == [101]                     # only _gen's own reference computation; the lane did not materialize
        assert st.cached_tokens == len(ids1) and not st.vision_tower
        assert got == cold
    finally:
        target.close()


def test_a_different_image_does_not_reuse_through_it(monkeypatch):
    """Same text, different image: restore the last boundary snapshot BEFORE the image (16), never the one after it
    (32), and prefill the image rows from the new embeddings. Tokens equal a cold run."""
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    model = _target(13)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids, first, end = _prompt(range(5, 25), 9, range(30, 41))          # image rows 20..28, boundaries 16 / 32
        a, b, cold_calls = [], [], []
        pa = _plan(model, ids, first, end, 101, a, key_base=2 ** 31 + 7)
        pb = _plan(model, ids, first, end, 202, b, key_base=2 ** 31 + 9000)
        cold, _ = _gen(model, target, _plan(model, ids, first, end, 202, cold_calls, key_base=2 ** 31 + 9000), 16)
        store = GD.PrefixStore(12)
        _gen(model, target, pa, 16, store)
        b.clear()
        got, st = _gen(model, target, pb, 16, store)
        assert st.cached_tokens == 16 and st.vision_tower and b == [202, 202]
        assert got == cold
    finally:
        target.close()


def test_unkeyed_images_reuse_only_text_before_the_first_image(monkeypatch):
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    model = _target(14)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids, first, end = _prompt(range(5, 25), 9, range(30, 41))
        calls = []
        plan = _plan(model, ids, first, end, 101, calls, key_base=None)
        store = GD.PrefixStore(12)
        store.last_persistence = "stale receipt from an earlier request"
        _gen(model, target, plan, 8, store, key=False)
        assert sorted(len(k) for k in store.entries) == [16]              # nothing at or after the image
        assert store.last_persistence is None
        got, st = _gen(model, target, plan, 8, store, key=False)
        assert st.cached_tokens == 16
    finally:
        target.close()


def test_image_prefix_survives_restart_on_the_ssd_tier(tmp_path):
    from vmlx_engine.glm_dflash2 import PrefixDiskTier, PrefixStore
    model = _target(15)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids1, first, end = _prompt(range(5, 22), 9, range(30, 41))
        ids2 = ids1 + list(range(60, 77))
        key_base = 2 ** 31 + 7                                             # above int32
        cold, _ = _gen(model, target, _plan(model, ids2, first, end, 101, [], key_base=key_base), 20)
        first_store = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        _gen(model, target, _plan(model, ids1, first, end, 101, [], key_base=key_base), 20, first_store)
        first_store.disk.flush()
        first_store.disk.close()
        restarted = PrefixStore(4, disk=PrefixDiskTier(tmp_path, 2 ** 30))
        calls = []
        p2 = _plan(model, ids2, first, end, 101, calls, key_base=key_base)
        got, st = _gen(model, target, p2, 20, restarted)
        assert st.cached_tokens == len(ids1) and st.cache_source == "disk" and not st.vision_tower
        assert got == cold
        restarted.disk.close()
    finally:
        target.close()


def test_restore_at_a_boundary_inside_the_image_prefills_the_rest_of_it(monkeypatch):
    """A boundary snapshot can fall INSIDE an image run (16 here, image rows 12..24). A later prompt with the same
    image and a different tail restores that boundary and must prefill the remaining image rows from the plan's
    embeddings (not their placeholder ids). Tokens equal a cold run."""
    import vmlx_engine.glm_dflash2 as GD
    monkeypatch.setattr(GD, "PREFIX_SNAPSHOT_EVERY", 16)
    model = _target(16)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids_a, first, end = _prompt(range(5, 17), 12, range(30, 36))       # image 12..24, tail to 30
        ids_b = ids_a[:end] + list(range(80, 92))                            # same image, different tail
        pa = _plan(model, ids_a, first, end, 101, [], key_base=2 ** 31 + 7)
        cold_calls, calls = [], []
        cold, _ = _gen(model, target, _plan(model, ids_b, first, end, 101, cold_calls, key_base=2 ** 31 + 7), 16)
        store = GD.PrefixStore(12)
        _gen(model, target, pa, 16, store)
        pb = _plan(model, ids_b, first, end, 101, calls, key_base=2 ** 31 + 7)
        got, st = _gen(model, target, pb, 16, store)
        assert st.cached_tokens == 16 and st.vision_tower
        assert got == cold
    finally:
        target.close()


def test_prefill_compute_clock_excludes_restore_and_covers_the_vision_tower():
    """usage.vmlx_prefill (chat footer pp/s) on the GLM DFlash2 lane: the compute clock starts after the prefix
    lookup/restore and still covers the vision tower + suffix prefill."""
    from vmlx_engine.glm_dflash2 import PrefixStore
    model = _target(14)
    target = GLMDFlashTarget(model, TAPS)
    try:
        ids1, first, end = _prompt(range(5, 22), 9, range(30, 41))
        ids2 = ids1 + list(range(60, 77))
        store = PrefixStore(4)
        _, cold = _gen(model, target, _plan(model, ids1, first, end, 101, [], key_base=2 ** 31 + 7), 8, store)
        assert cold.vision_tower and 0 < cold.prefill_compute_s <= cold.prefill_s
        _, warm = _gen(model, target, _plan(model, ids2, first, end, 101, [], key_base=2 ** 31 + 7), 8, store)
        assert warm.cached_tokens == len(ids1) and not warm.vision_tower
        assert 0 < warm.prefill_compute_s <= warm.prefill_s
    finally:
        target.close()


def test_stream_chunks_carry_the_prefill_receipt(monkeypatch):
    import mlx_lm.tokenizer_utils as tu
    import vmlx_engine.glm_dflash2 as glm

    class Detok:
        last_segment = ""
        def reset(self): pass
        def add_token(self, t): self.last_segment = "x"
        def finalize(self): pass

    class FakeWrapper:
        eos_token_ids = {99}
        detokenizer = Detok()

    def rounds(target, draft, prompt_ids, max_tokens, eos, *a, stats=None, **k):
        stats.prompt_tokens, stats.cached_tokens, stats.prefill_compute_s = 40, 30, 0.5
        stats.vision_tower = True
        yield [7], 0
        yield [8, 99], 3

    monkeypatch.setattr(tu, "TokenizerWrapper", FakeWrapper)
    monkeypatch.setattr(glm, "target_for", lambda model, ids: object())
    monkeypatch.setattr(glm, "_dflash2_rounds", rounds)
    monkeypatch.setenv("VMLX_GLM_DFLASH2_PREFIX_CACHE", "0")
    draft = type("D", (), {"config": type("C", (), {"target_layer_ids": [1]})()})()
    chunks = list(glm.stream_glm_dflash2(object(), FakeWrapper(), draft, "", max_tokens=8, temperature=0.0,
                                         prompt_tokens=list(range(40))))
    want = {"tokens": 10, "seconds": 0.5, "scope": "model_prefill_and_prompt_state", "path": "glm_dflash2_with_vision"}
    assert [c.prefill_usage for c in chunks] == [want, want]
