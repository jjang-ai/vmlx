# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter fine-tuning pipeline (vmlx_engine.dspark.finetune) on a tiny synthetic MiniMax-M3 + drafter.

Checked:
  - records: sharded write / memory-mapped read round trip, resume (manifest), layout clash refused;
  - data: generated records carry exactly the serving rows (prefill keep_last rows + one-row ar_step rows, bit-equal)
    and the greedy tokens as labels; the teacher pass equals the served prefill; assistant spans of the M3 template;
  - objective: the batched multi-anchor training forward equals the runtime DSparkDrafter block forward (bf16
    rounding) with the sliding window active; its candidate pick equals the runtime chain's draft; the mask and
    on-path row validity; MXFP8 fake quantization is the identity on the release's stored values;
  - offline replay == the live iter_dspark loop (rounds, drafts, accepts) at a fixed width and with Adaptive widths;
  - training: loss falls, checkpoint resume reproduces an uninterrupted run bit for bit;
  - export: an untouched drafter re-exports to identical tensors; a trained one keeps the release layout (names,
    dtypes, shapes, index, config bytes), loads through load_drafter / validate_bundled_draft / load_dspark_runtime and
    carries the trained values (MXFP8 round trip / BF16); bf16 export loads as float linears;
  - (slow) the end-to-end synthetic proof: fine-tuning on the bundle beats the shipped drafter and a same-budget
    source-data control on the bundle's held-out acceptance; greedy DSpark stays identical to AR.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.dspark.drafter import load_drafter  # noqa: E402
from vmlx_engine.dspark.finetune import datagen as DG  # noqa: E402
from vmlx_engine.dspark.finetune import objective as O  # noqa: E402
from vmlx_engine.dspark.finetune import records as R  # noqa: E402
from vmlx_engine.dspark.finetune import synthetic as SY  # noqa: E402
from vmlx_engine.dspark.finetune.evaluate import replay  # noqa: E402
from vmlx_engine.dspark.finetune.target_head import TargetHead, save_target_head  # noqa: E402
from vmlx_engine.dspark.generate import DSparkStats, generate_ar, generate_dspark  # noqa: E402
from vmlx_engine.dspark.minimax_target import MiniMaxM3DSparkTarget  # noqa: E402

WINDOW = 64
GEN = 48


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in ("VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH", "VMLX_DSPARK_PIPELINE",
                 "VMLX_DSPARK_MARKOV_CANDIDATES", "VMLX_DSPARK_MEASURED_WIDTH", "VMLX_DSPARK_DRAFT_WINDOW"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def bundle_model():
    return SY.target_model(0, quantize=True, scale=0.6)


@pytest.fixture(scope="module")
def target(bundle_model):
    return MiniMaxM3DSparkTarget(bundle_model)


@pytest.fixture(scope="module")
def release_dir(tmp_path_factory):
    return SY.write_release_drafter(tmp_path_factory.mktemp("ft") / "dspark", seed=3)


@pytest.fixture(scope="module")
def spec(release_dir):
    return O.Spec.from_drafter(load_drafter(release_dir))


def _meta(spec, topk=0):
    return {"taps": list(spec.taps), "hidden": spec.hidden, "tap_dtype": "bfloat16", "topk": topk,
            "vocab_size": spec.vocab, "window": spec.window}


@pytest.fixture(scope="module")
def prompts():
    return SY.random_prompts(6, 101, lo=90, hi=180)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, target, spec, prompts):
    out = tmp_path_factory.mktemp("ftdata") / "data"
    w = R.ShardWriter(out, _meta(spec, topk=4), shard_bytes=1 << 20)
    save_target_head(target, out)
    runner = DG.TargetRunner(target, spec.taps, prefill_step=256, topk=4)
    items = [{"id": f"r{i}", "domain": "d0" if i % 2 else "d1", "set": "calib" if i < 4 else "ref", "ids": p}
             for i, p in enumerate(prompts)]
    DG.run(items, runner, w, mode="generate", prompt_ctx=WINDOW, max_new=GEN, eos=set(), log=lambda *a: None)
    return out


# ----------------------------------------------------------------------------------------------- records
def test_records_roundtrip_resume_and_layout_guard(tmp_path):
    rng = np.random.default_rng(0)
    meta = {"taps": [1, 3], "hidden": 8, "tap_dtype": "bfloat16", "topk": 2, "vocab_size": 50}
    w = R.ShardWriter(tmp_path, meta, shard_bytes=600)
    want = {}
    for i in range(5):
        n, ts = int(rng.integers(5, 12)), int(rng.integers(0, 4))
        arr = {"tokens": rng.integers(0, 50, n), "labels": rng.integers(0, 50, n), "gen": rng.integers(0, 2, n),
               "taps": rng.integers(0, 65535, (n - ts, 16)).astype(np.uint16), "picks": rng.integers(0, 50, n),
               "topk_ids": rng.integers(0, 50, (n, 2)), "topk_logprobs": rng.standard_normal((n, 2))}
        w.add(R.Record(id=f"x{i}", n=n, tap_start=ts, domain="a", set="calib", n_gen=int(arr["gen"].sum())), **arr)
        want[f"x{i}"] = (ts, arr)
    w.close()
    assert len(list((tmp_path / "shards").glob("shard-*.safetensors"))) >= 2            # the size cap split shards
    assert w.done_ids() == set(want)
    ds = R.Dataset(tmp_path)
    assert {r.id for r in ds.records} == set(want)
    for rec in ds.records:
        ts, arr = want[rec.id]
        assert rec.tap_start == ts
        assert np.array_equal(ds.tokens(rec), arr["tokens"]) and np.array_equal(ds.labels(rec), arr["labels"])
        assert np.array_equal(ds.gen(rec), arr["gen"]) and np.array_equal(ds.taps(rec, ts, rec.n), arr["taps"])
        assert np.array_equal(ds.taps(rec, rec.n - 1, rec.n), arr["taps"][-1:])
        assert np.array_equal(ds.per_position(rec, "topk_ids"), arr["topk_ids"])
        with pytest.raises(IndexError):
            ds.taps(rec, ts - 1, rec.n) if ts else ds.taps(rec, 0, rec.n + 1)
    w2 = R.ShardWriter(tmp_path, meta)                                                    # resume: same layout
    assert w2.done_ids() == set(want) and w2._next == len(list((tmp_path / "shards").glob("shard-*.json")))
    with pytest.raises(ValueError, match="another layout"):
        R.ShardWriter(tmp_path, {**meta, "hidden": 16})


def test_assistant_mask_and_anchor_positions():
    H, E = DG.M3_ASSISTANT_HEADER, DG.M3_TURN_END
    user = [200019, 3995, 10, 5, 6, E, 10]
    ids = [200034] + user + list(H) + [7, 8, 9, E, 10] + user + list(H) + [11, 12]
    g = DG.assistant_mask(ids)
    first = 1 + len(user) + len(H)
    second = first + 5 + len(user) + len(H)
    expect = np.zeros(len(ids), np.uint8)
    expect[first:first + 4] = 1                      # 7 8 9 [e~[
    expect[second:second + 2] = 1                    # 11 12 (unterminated)
    assert np.array_equal(g, expect)
    assert R.anchor_positions(g).tolist() == [first, first + 1, first + 2, second]


# ----------------------------------------------------------------------------------------------- data
def test_generated_records_carry_the_serving_rows(target, spec, data_dir, prompts):
    ds = R.Dataset(data_dir)
    assert len(ds.records) == len(prompts)
    for rec, prompt in zip(sorted(ds.records, key=lambda r: int(r.id[1:])), prompts):
        P = len(prompt)
        assert rec.n == P + GEN and rec.n_gen == GEN and rec.tap_start == P - WINDOW
        tokens = ds.tokens(rec)
        ref = generate_ar(target, prompt, GEN, set(), prefill_step=256)
        assert tokens[P:].tolist() == ref and np.array_equal(ds.labels(rec)[P:], tokens[P:])
        cache = target.make_cache()
        _, f = target.prefill(mx.array([prompt]), cache, spec.taps, 256, keep_last=WINDOW)
        rows = [f[0]]
        for t in ref:
            _, f1 = target.ar_step(mx.array([t]), cache, spec.taps)
            rows.append(f1[0])
        want = np.array(mx.view(mx.concatenate(rows), mx.uint16))
        assert np.array_equal(np.asarray(ds.taps(rec, rec.tap_start, rec.n)), want)
        picks = ds.per_position(rec, "picks")
        assert np.array_equal(picks[P - 1:-1], tokens[P:])                       # row j's argmax = token j + 1
        tk = ds.per_position(rec, "topk_ids")
        assert np.array_equal(tk[P - 1:-1, 0], tokens[P:])                        # top-1 of the stored top-K


def test_teacher_pass_equals_the_served_prefill(target, spec, prompts):
    ids = prompts[0] + prompts[1]
    runner = DG.TargetRunner(target, spec.taps, prefill_step=128)
    taps, picks, _, _ = runner.teacher_pass(ids, 0)
    cache = target.make_cache()
    logits, feats = target.prefill(mx.array([ids]), cache, spec.taps, 128)
    assert np.array_equal(taps, np.array(mx.view(feats[0], mx.uint16)))
    assert int(picks[-1]) == int(mx.argmax(logits).item())


# ----------------------------------------------------------------------------------------------- objective
def test_training_forward_equals_the_runtime_drafter(release_dir, spec, data_dir):
    ds = R.Dataset(data_dir)
    head = TargetHead(data_dir)
    dr = load_drafter(release_dir)
    params, mxfp8 = O.drafter_params(dr)
    rec = ds.records[0]
    anchors = R.anchor_positions(ds.gen(rec))[::5]
    assert anchors.max() - rec.tap_start > WINDOW + spec.block      # the sliding window cuts the context
    b = O.make_batch(ds, rec, anchors, spec.block, spec)
    hid, base, bias, _ = O.forward(params, spec, head, b)
    picks = O.pick_tokens(base, bias, 32)
    mx.eval(hid, picks)
    toks = ds.tokens(rec)
    mask_embed = head.embed(mx.array([[spec.mask_token_id]]))
    worst, same = 0.0, 0
    for m, p in enumerate(anchors.tolist()):
        dr.reset(base=rec.tap_start)
        dr.add_context(mx.view(mx.array(np.asarray(ds.taps(rec, rec.tap_start, p))), mx.bfloat16)[None])
        anchor = mx.array([int(toks[p])])
        blk = mx.concatenate([head.embed(anchor.reshape(1, 1)),
                              mx.broadcast_to(mask_embed, (1, spec.block - 1, spec.hidden))], axis=1)
        h = dr(blk)
        rel = mx.abs(h[0].astype(mx.float32) - hid[m].astype(mx.float32)) / (mx.abs(h[0].astype(mx.float32)) + 1.0)
        worst = max(worst, float(rel.max()))
        drafts, _ = dr.draft(h, anchor, head.head, 1)
        same += int(drafts[0].item()) == int(picks[m, 0].item())
    assert worst < 2 ** -6                                           # bf16 rounding, not a different function
    assert same >= len(anchors) - 1


def test_attention_mask_and_on_path_rows(spec):
    m = np.asarray(O.attention_mask(np.array([10, 12]), 3, ctx_start=4, ctx_len=8, window=5))
    # row (anchor 10, i=0) at q=10: context 6..9 (> 10-5, < 10); row (10, 2) at q=12: 8..9; block 0 only
    assert m[0, :8].tolist() == [False, False, True, True, True, True, False, False]
    assert m[2, :8].tolist() == [False] * 4 + [True, True, False, False]
    assert m[3, :8].tolist() == [False] * 4 + [True] * 4                    # anchor 12, q=12: 8..11
    assert m[0, 8:].tolist() == [True] * 3 + [False] * 3 and m[4, 8:].tolist() == [False] * 3 + [True] * 3

    class FakeDS:
        width = spec.hidden * len(spec.taps)
        topk = 0

        def __init__(self):
            self.t = np.arange(20, dtype=np.int32)
            self.l = self.t.copy()
            self.l[13] = 99                                                  # the target would not write token 13
            self.g = (np.arange(20) >= 8).astype(np.uint8)
            self.g[17:] = 0                                                  # the span ends at 16

        def tokens(self, r):
            return self.t

        def labels(self, r):
            return self.l

        def gen(self, r):
            return self.g

        def taps(self, r, a, b):
            return np.zeros((b - a, self.width), np.uint16)

    rec = R.Record(id="f", n=20, tap_start=0)
    b = O.make_batch(FakeDS(), rec, np.array([10, 14]), 4, spec, on_path=True)
    valid = np.asarray(b.valid)
    # anchor 10: rows predict 11, 12, 13, 14; row 2 predicts 13 (label 99, fine), row 3 conditions on token 13 != label
    assert valid[0].tolist() == [True, True, True, False]
    # anchor 14: rows predict 15, 16, 17 (not generated), 18
    assert valid[1].tolist() == [True, True, False, False]
    assert np.asarray(b.prev)[0].tolist() == [10, 11, 12, 99] and np.asarray(b.labels)[0, :3].tolist() == [11, 12, 99]
    assert np.asarray(b.block_tokens)[1].tolist() == [14] + [spec.mask_token_id] * 3
    off = np.asarray(O.make_batch(FakeDS(), rec, np.array([10]), 4, spec, on_path=False).valid)
    assert off[0].tolist() == [True] * 4


def test_fake_mxfp8_is_the_identity_on_stored_values(release_dir):
    params, mxfp8 = O.drafter_params(load_drafter(release_dir))
    assert len(mxfp8) == 14
    for path in mxfp8:
        w = params[path + ".weight"].astype(mx.float32)
        assert bool(mx.array_equal(O.fake_mxfp8(w), w).item()), path


# ----------------------------------------------------------------------------------------------- replay
class _WidthOnly:
    def __new__(cls):
        from vmlx_engine.dspark.finetune.proof import WidthOnly

        return WidthOnly()


@pytest.mark.parametrize("width", [2, "adaptive"])
def test_replay_equals_the_live_loop(target, release_dir, data_dir, prompts, width):
    """Offline replay over recorded greedy data == iter_dspark on the target: same rounds, drafts and accepts (the
    recorded tap rows are the rows the server feeds; the drafter, width policy and Markov chain are the runtime's)."""
    ds = R.Dataset(data_dir)
    head = TargetHead(data_dir)
    dr = load_drafter(release_dir)
    top = 2 * int(dr.block)
    got = replay(dr, head, ds, width=width, max_draft=(top if width == "adaptive" else width),
                 room_rule="max_new")["all"]
    live = DSparkStats()
    for p in prompts:
        st = DSparkStats()
        if width == "adaptive":
            out = generate_dspark(target, dr, p, GEN, set(), prefill_step=256, stats=st, max_draft=top,
                                  ar_fallback=_WidthOnly())
        else:
            out = generate_dspark(target, dr, p, GEN, set(), prefill_step=256, stats=st, max_draft=width,
                                  fixed_width=width)
        assert out == generate_ar(target, p, GEN, set(), prefill_step=256)
        live.rounds, live.drafted, live.accepted = (live.rounds + st.rounds, live.drafted + st.drafted,
                                                   live.accepted + st.accepted)
    assert live.accepted > 0                                          # not a vacuous comparison
    assert (got["rounds"], got["drafted"], got["accepted"]) == (live.rounds, live.drafted, live.accepted)


# ----------------------------------------------------------------------------------------------- train / export
def _cfg(data_dir, release_dir, out, **kw):
    from vmlx_engine.dspark.finetune.train import TrainConfig

    base = dict(data=str(data_dir), drafter=str(release_dir), out=str(out), steps=6, lr=1e-3, warmup=2,
                anchors=16, windows_per_step=2, eval_every=0, ckpt_every=3, log_every=3, eval_sets=())
    base.update(kw)
    return TrainConfig(**base)


def test_training_lowers_the_loss_and_resume_is_exact(tmp_path, data_dir, release_dir):
    from vmlx_engine.dspark.finetune.train import train

    a = train(_cfg(data_dir, release_dir, tmp_path / "a", steps=6), log=lambda *x: None)
    log = [json.loads(line) for line in (tmp_path / "a" / "train_log.jsonl").read_text().splitlines()]
    losses = [r["ce"] for r in log if "ce" in r and "eval" not in r]
    assert losses[-1] < losses[0]
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    shutil.rmtree(tmp_path / "b" / "final")
    shutil.rmtree(tmp_path / "b" / "ckpt" / "step-0000006")
    cfg_b = _cfg(data_dir, release_dir, tmp_path / "b", steps=6)
    b = train(cfg_b, log=lambda *x: None)                               # resumes at step 3
    for k, v in a["train"].items():
        assert bool(mx.array_equal(v, b["train"][k]).item()), k


def test_export_untouched_is_identical_and_trained_keeps_the_release_layout(tmp_path, release_dir, bundle_model):
    from vmlx_engine.bundled_draft import validate_bundled_draft
    from vmlx_engine.dspark.finetune.export import export_dspark
    from vmlx_engine.dspark.runtime import load_dspark_runtime

    params, _ = O.drafter_params(load_drafter(release_dir))
    rep = export_dspark(release_dir, params, tmp_path / "same")
    assert not rep["changed"] and rep["unchanged"] == rep["tensors"]
    src_shard = release_dir / "model-dspark-00001.safetensors"
    assert (tmp_path / "same" / src_shard.name).read_bytes() == src_shard.read_bytes()
    assert (tmp_path / "same" / "config.json").read_bytes() == (release_dir / "config.json").read_bytes()

    rng = np.random.default_rng(5)
    moved = dict(params)
    for k in ("fc.weight", "layers.0.self_attn.q_proj.weight", "layers.1.mlp.down_proj.weight",
              "markov_head.markov_w1.weight", "norm.weight"):
        moved[k] = params[k].astype(mx.float32) + mx.array(rng.standard_normal(params[k].shape) * 0.01,
                                                           dtype=mx.float32)
    rep = export_dspark(release_dir, moved, tmp_path / "trained")
    assert sorted(rep["changed"]) == sorted(k.rsplit(".", 1)[0] + ".weight" for k in
                                            ("fc.weight", "layers.0.self_attn.q_proj.weight",
                                             "layers.1.mlp.down_proj.weight", "markov_head.markov_w1.weight",
                                             "norm.weight"))
    src_h, _ = R.read_safetensors_header(src_shard)
    dst_h, _ = R.read_safetensors_header(tmp_path / "trained" / src_shard.name)
    assert {k: (v["dtype"], v["shape"]) for k, v in src_h.items()} == \
        {k: (v["dtype"], v["shape"]) for k, v in dst_h.items()}
    assert (tmp_path / "trained" / "config.json").read_bytes() == (release_dir / "config.json").read_bytes()
    idx = json.loads((tmp_path / "trained" / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(dst_h) and idx["metadata"]["total_size"] == rep["bytes"]
    got, _ = O.drafter_params(load_drafter(tmp_path / "trained"))
    assert bool(mx.array_equal(got["fc.weight"], moved["fc.weight"].astype(mx.bfloat16)).item())
    q = "layers.0.self_attn.q_proj.weight"
    assert bool(mx.array_equal(got[q].astype(mx.float32), O.fake_mxfp8(moved[q])).item())
    assert not bool(mx.array_equal(got[q], params[q]).item())
    untouched = "layers.1.self_attn.k_proj.weight"
    assert bool(mx.array_equal(got[untouched], params[untouched]).item())
    root = SY.bundle_dir(tmp_path / "bundle", tmp_path / "trained")
    validate_bundled_draft(root / "dspark", root)
    rt = load_dspark_runtime(bundle_model, root)
    assert rt is not None and rt["drafter"].block == 4 and rt["drafter"].target_layer_ids == [1, 3]

    export_dspark(release_dir, moved, tmp_path / "bf16", fmt="bf16")
    d = load_drafter(tmp_path / "bf16")
    assert type(d.layers[0].self_attn.q_proj) is nn.Linear                          # unpaired weight -> float
    assert isinstance(d.layers[1].self_attn.k_proj, nn.QuantizedLinear)              # untouched stays MXFP8
    assert bool(mx.array_equal(d.layers[0].self_attn.q_proj.weight, moved[q].astype(mx.bfloat16)).item())
    with pytest.raises(FileExistsError):
        export_dspark(release_dir, moved, tmp_path / "bf16")


@pytest.mark.slow
def test_end_to_end_synthetic_proof(tmp_path):
    """The pipeline end to end (proof.run, reduced): fine-tuning the source-trained drafter on the bundle's data beats
    the shipped drafter AND a control fine-tuned on as much new source data, on the bundle's held-out tokens per
    round (Adaptive) and fixed-width acceptance; the exported drafter loads through load_dspark_runtime and greedy
    DSpark equals plain decoding."""
    from vmlx_engine.dspark.finetune import proof

    r = proof.run(tmp_path, n_train=48, n_eval=12, steps_src=600, steps_ft=300, live_prompts=4, log=lambda *a: None)
    tpr = {k: r[k]["bundle_adaptive"]["tokens_per_round"] for k in ("shipped", "control", "tuned")}
    acc = {k: r[k]["bundle_fixed_4"]["acceptance"] for k in ("shipped", "control", "tuned")}
    assert tpr["tuned"] > tpr["control"] and tpr["tuned"] > tpr["shipped"]
    assert acc["tuned"] > acc["control"] and acc["tuned"] > acc["shipped"]
    for tag in ("shipped", "tuned"):
        assert all(v["identical_to_ar"] for v in r["live"][tag].values())
    assert r["live"]["tuned"]["adaptive"]["tokens_per_round"] > r["live"]["shipped"]["adaptive"]["tokens_per_round"]
