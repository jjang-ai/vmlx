"""End-to-end synthetic proof of the fine-tune pipeline (no real weights; runs in minutes on any Apple GPU).

The real situation in miniature: a "source" target (tiny MiniMax-M3, BF16) and its "bundle" (the same weights
quantized per module like an M3.1 JANGHT bundle; teacher-forced top-1 agreement with the source ~0.84 at the default
weight scale, the real bundle's is ~0.86). A drafter trained on the SOURCE plays the shipped drafter; it starts from a
random release-layout drafter (deliberately mismatched). Then the pipeline as it runs on the real bundle:

1. ``generate`` data on the bundle (greedy continuations, exact serving taps), train and held-out prompts;
2. fine-tune the shipped drafter on the bundle's data (QAT MXFP8, as default);
3. export in the release layout, load through ``load_dspark_runtime`` from a bundle directory;
4. acceptance on the bundle's held-out data: offline replay (Adaptive + fixed widths) and the live ``iter_dspark``
   loop, shipped vs fine-tuned; greedy output must equal plain decoding for both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _log(msg):
    print(msg, flush=True)


def _gen(target, prompts, out_dir, *, set_name, prompt_ctx, max_new, writer=None, spec=None, head_from=None):
    from . import datagen as DG
    from .records import ShardWriter
    from .target_head import save_target_head

    if writer is None:
        writer = ShardWriter(out_dir, {"taps": list(spec.taps), "hidden": spec.hidden, "tap_dtype": "bfloat16",
                                       "topk": 0, "vocab_size": spec.vocab, "window": spec.window,
                                       "mode": "generate", "synthetic": True}, shard_bytes=8 << 20)
        save_target_head(target, out_dir)
    runner = DG.TargetRunner(target, spec.taps, prefill_step=256)
    items = [{"id": f"{set_name}{i:03d}", "domain": "syn", "set": set_name, "ids": p} for i, p in enumerate(prompts)]
    tot = DG.run(items, runner, writer, mode="generate", prompt_ctx=prompt_ctx, max_new=max_new, eos=set(),
                 log=lambda *a: None)
    return writer, tot


class WidthOnly:
    """``ARFallback`` without the wall-clock AR fallback (the tiny target's plain decode is cheaper than any round)."""

    def __new__(cls):
        from ..generate import ARFallback

        class _W(ARFallback):
            def next_is_ar(self) -> bool:
                return False

        return _W()


def live_acceptance(runtime: dict, target, prompts, max_new: int, *, width=None) -> dict:
    """Greedy ``iter_dspark`` on ``prompts`` through a loaded runtime: identity vs ``generate_ar`` and DSparkStats."""
    from ..generate import DSparkStats, generate_ar, generate_dspark

    drafted = accepted = rounds = tokens = 0
    identical = True
    for p in prompts:
        ref = generate_ar(target, p, max_new, set(), prefill_step=256)
        st = DSparkStats()
        if width is None:
            out = generate_dspark(target, runtime["drafter"], p, max_new, set(), prefill_step=256, stats=st,
                                  max_draft=runtime["max_draft"], ar_fallback=WidthOnly())
        else:
            out = generate_dspark(target, runtime["drafter"], p, max_new, set(), prefill_step=256, stats=st,
                                  max_draft=int(width), fixed_width=int(width))
        identical &= out == ref
        drafted, accepted, rounds, tokens = drafted + st.drafted, accepted + st.accepted, rounds + st.rounds, \
            tokens + st.tokens - 1 - st.ar_tokens
    return {"identical_to_ar": bool(identical), "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "acceptance": round(accepted / drafted, 4) if drafted else None,
            "tokens_per_round": round(tokens / rounds, 4) if rounds else None}


def run(out: str | Path, *, scale: float = 0.6, n_train: int = 64, n_eval: int = 16, max_new: int = 128,
        steps_src: int = 800, steps_ft: int = 400, lr_src: float = 1e-3, lr_ft: float = 3e-4, anchors: int = 64,
        windows: int = 4, live_prompts: int = 8, seed: int = 0, train_overrides: dict | None = None,
        log=_log) -> dict:
    """See the module docstring. Arms (all evaluated on the BUNDLE's held-out data):
    random drafter / shipped (random -> source data A) / tuned (shipped -> bundle data on prompts B) / control
    (shipped -> source data on the same prompts B, same steps: the extra training without the target change)."""
    import mlx.core as mx  # noqa: F401  (imports MLX before the model classes)

    from ..drafter import load_drafter
    from ..minimax_target import MiniMaxM3DSparkTarget
    from ..runtime import load_dspark_runtime
    from . import synthetic as SY
    from .evaluate import replay, teacher_forced
    from .export import export_dspark
    from .objective import Spec
    from .records import Dataset
    from .target_head import TargetHead
    from .train import TrainConfig, load_trained, train

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    over = dict(train_overrides or {})
    report = {"config": {"scale": scale, "n_train": n_train, "n_eval": n_eval, "max_new": max_new,
                         "steps_src": steps_src, "steps_ft": steps_ft, "lr_src": lr_src, "lr_ft": lr_ft,
                         "anchors": anchors, "windows_per_step": windows, "train_overrides": over}}
    src_model = SY.target_model(seed, quantize=False, scale=scale)
    bun_model = SY.target_model(seed, quantize=True, scale=scale)
    src, bun = MiniMaxM3DSparkTarget(src_model), MiniMaxM3DSparkTarget(bun_model)
    d0 = SY.write_release_drafter(out / "drafter_random", seed=seed)
    spec = Spec.from_drafter(load_drafter(d0))
    prompts = {"calib": SY.random_prompts(n_train, seed + 11), "ft": SY.random_prompts(n_train, seed + 13),
               "ref": SY.random_prompts(n_eval, seed + 12)}
    prompt_ctx = spec.window

    log("[proof] generating source and bundle data")
    for name, tgt, sets in (("data_src", src, ("calib", "ft", "ref")), ("data_bundle", bun, ("ft", "ref"))):
        if (out / name / "manifest.jsonl").is_file():
            continue
        w = None
        for set_name in sets:
            w, tot = _gen(tgt, prompts[set_name], out / name, set_name=set_name, prompt_ctx=prompt_ctx,
                          max_new=max_new, writer=w, spec=spec)
            report[f"gen_{name}_{set_name}"] = tot
    head = TargetHead(out / "data_bundle")
    src_head = TargetHead(out / "data_src")
    ev_bun = Dataset(out / "data_bundle", sets=["ref"])
    ev_src = Dataset(out / "data_src", sets=["ref"])

    def evaluate_dir(path, tag):
        d = load_drafter(path)
        res = {"bundle_adaptive": replay(d, head, ev_bun)["all"],
               "source_adaptive": replay(d, src_head, ev_src)["all"]}
        for k in (1, 2, spec.block, 2 * spec.block):
            res[f"bundle_fixed_{k}"] = replay(d, head, ev_bun, width=k)["all"]
        res["bundle_teacher_forced"] = teacher_forced(_params(path), spec, head, ev_bun)["all"]
        ba = res["bundle_adaptive"]
        log(f"[proof] {tag}: bundle Adaptive tok/round {ba['tokens_per_round']} (acceptance {ba['acceptance']}), "
            f"fixed {spec.block} acceptance {res[f'bundle_fixed_{spec.block}']['acceptance']} | source Adaptive "
            f"tok/round {res['source_adaptive']['tokens_per_round']}")
        return res

    def stage(name, data, drafter, steps, lr, sets, seed_):
        cfg = TrainConfig(data=str(out / data), drafter=str(drafter), out=str(out / f"run_{name}"), steps=steps,
                          lr=lr, warmup=max(10, steps // 20), anchors=anchors, windows_per_step=windows,
                          train_sets=sets, eval_sets=("ref",), eval_every=max(1, steps // 4), ckpt_every=0,
                          log_every=max(1, steps // 4), seed=seed_, **over)
        if not (out / f"run_{name}" / "final" / "train.safetensors").is_file():
            train(cfg, log=log)
        dst = out / f"drafter_{name}"
        if not dst.is_dir():
            rep = export_dspark(drafter, load_trained(out / f"run_{name}"), dst)
            report[f"export_{name}"] = {"tensors": rep["tensors"], "changed": len(rep["changed"]),
                                        "unchanged": rep["unchanged"], "bytes": rep["bytes"]}
        return dst

    log("[proof] stage A: the shipped drafter = random release drafter trained on SOURCE data (prompts A)")
    shipped = stage("shipped", "data_src", d0, steps_src, lr_src, ("calib",), seed)
    log("[proof] stage B: fine-tune the shipped drafter on the BUNDLE's data (prompts B)")
    tuned = stage("tuned", "data_bundle", shipped, steps_ft, lr_ft, ("ft",), seed + 1)
    log("[proof] control: fine-tune the shipped drafter on SOURCE data of the same prompts B, same steps")
    control = stage("control", "data_src", shipped, steps_ft, lr_ft, ("ft",), seed + 1)
    for tag, path in (("random", d0), ("shipped", shipped), ("tuned", tuned), ("control", control)):
        report[tag] = evaluate_dir(path, tag)

    log("[proof] live iter_dspark through load_dspark_runtime (bundle target)")
    live = {}
    for tag, path in (("shipped", shipped), ("tuned", tuned)):
        root = SY.bundle_dir(out / f"bundle_{tag}", path)
        rt = load_dspark_runtime(bun_model, root)
        assert rt is not None, "the runtime did not load the exported drafter"
        rt["target"] = bun
        live[tag] = {"adaptive": live_acceptance(rt, bun, prompts["ref"][:live_prompts], max_new),
                     f"fixed_{spec.block}": live_acceptance(rt, bun, prompts["ref"][:live_prompts], max_new,
                                                            width=spec.block)}
        log(f"[proof] live {tag}: {json.dumps(live[tag])}")
    report["live"] = live
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "proof_report.json").write_text(json.dumps(report, indent=1, default=_np))
    log(summary(report, spec.block))
    return report


def summary(report: dict, block: int) -> str:
    rows = ["| drafter | bundle Adaptive tok/round | bundle Adaptive acceptance | bundle fixed-%d acceptance | "
            "bundle fixed-%d tok/round | bundle TF mean L | source Adaptive tok/round |" % (block, block),
            "|---|---|---|---|---|---|---|"]
    for tag in ("random", "shipped", "control", "tuned"):
        r = report.get(tag)
        if not r:
            continue
        fx = r[f"bundle_fixed_{block}"]
        rows.append(f"| {tag} | {r['bundle_adaptive']['tokens_per_round']} | {r['bundle_adaptive']['acceptance']} | "
                    f"{fx['acceptance']} | {fx['tokens_per_round']} | {r['bundle_teacher_forced']['mean_L']} | "
                    f"{r['source_adaptive']['tokens_per_round']} |")
    return "\n".join(rows)


def _params(path):
    from ..drafter import load_drafter
    from .objective import drafter_params

    return drafter_params(load_drafter(path))[0]


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
