"""CLI: python -m vmlx_engine.dspark.finetune {gen,train,eval,export,info,synth} ...

gen     run the target bundle over prompts, write training records (resumable: re-run the same command)
train   fine-tune the bundle's drafter on records (resumable from the run's last checkpoint)
eval    offline acceptance of one or more drafter directories on held-out records (replay + teacher-forced)
export  write a run's weights as a dspark/ directory in the release format
info    summarize a data directory
synth   the end-to-end synthetic proof (tiny target, no real weights)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _csv(v, typ=str):
    return tuple(typ(x) for x in str(v).split(",") if x != "") if v else ()


def _drafter_cfg(bundle: Path) -> dict:
    from ..drafter import minimax_config

    spec = {}
    stamp = bundle / "jang_config.json"
    if stamp.is_file():
        spec = json.loads(stamp.read_text()).get("speculative") or {}
    path = bundle / str(spec.get("drafter_path") or "dspark") / "config.json"
    return minimax_config(json.loads(path.read_text()))


def cmd_gen(a) -> None:
    import mlx.core as mx

    from ..minimax_target import MiniMaxM3DSparkTarget
    from . import datagen as DG
    from .records import ShardWriter
    from .target_head import save_target_head

    bundle = Path(a.bundle).expanduser()
    dcfg = _drafter_cfg(bundle)
    taps = list(dcfg["dflash_config"]["target_layer_ids"])
    window = int(dcfg["sliding_window"])
    prompt_ctx = int(a.prompt_ctx) if a.prompt_ctx is not None else window
    items = DG.load_items(a.prompts, a.mode, sets=_csv(a.sets), domains=_csv(a.domains), limit=a.limit,
                          max_len=a.max_len, cut_turns=a.cut_turns)
    prefix = a.id_prefix if a.id_prefix is not None else ("tf:" if a.mode == "teacher" else
                                                          "gs:" if a.sample else "gg:")
    for it in items:
        it["id"] = prefix + it["id"]          # generate / teacher / sampled records of one row never collide
        if a.set_name:
            it["set"] = a.set_name
    print(f"[gen] {len(items)} items from {a.prompts} ({a.mode}); taps {taps}, window {window}, prompt_ctx {prompt_ctx}")
    if a.dry_run:
        tot = sum(len(it["ids"]) for it in items)
        print(f"[gen] dry run: {tot} prompt / sequence tokens")
        return
    mx.set_cache_limit(int(a.cache_gb * (1 << 30)))
    from ..utils import jang_loader

    t0 = time.perf_counter()
    model, _ = jang_loader.load_jang_model(str(bundle))
    target = MiniMaxM3DSparkTarget(model)
    print(f"[gen] bundle loaded in {time.perf_counter() - t0:.1f} s")
    out = Path(a.out)
    meta = {"taps": taps, "hidden": int(dcfg["hidden_size"]), "tap_dtype": "bfloat16", "topk": int(a.topk),
            "vocab_size": int(dcfg["vocab_size"]), "window": window, "block": int(dcfg["block_size"]),
            "bundle": str(bundle), "prefill_step": int(a.prefill_step)}
    writer = ShardWriter(out, meta, shard_bytes=int(a.shard_gb * (1 << 30)))
    if not (out / "target_head.safetensors").is_file():
        save_target_head(target, out)
    sampler = None
    if a.sample:
        from mlx_lm.sample_utils import make_sampler

        temp, top_p = (float(x) for x in a.sample.split(","))
        sampler = make_sampler(temp=temp, top_p=top_p)
    runner = DG.TargetRunner(target, taps, prefill_step=a.prefill_step, topk=a.topk)
    eos = set(_csv(a.eos, int))
    tot = DG.run(items, runner, writer, mode=a.mode, prompt_ctx=prompt_ctx, max_new=a.max_new, eos=eos,
                 sampler=sampler, seed=a.seed)
    print(f"[gen] done: {json.dumps(tot)}")
    with open(out / "gen_runs.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "args": vars(a), "totals": tot}) + "\n")


def cmd_train(a) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig(data=a.data, drafter=a.drafter, out=a.out, head=a.head, steps=a.steps, lr=a.lr,
                      min_lr_frac=a.min_lr_frac, warmup=a.warmup, weight_decay=a.weight_decay, clip=a.clip,
                      anchors=a.anchors, windows_per_step=a.windows, spans=_csv(a.spans, int), pos_decay=a.pos_decay,
                      conf_weight=a.conf_weight, soft_weight=a.soft_weight, base_weight=a.base_weight,
                      candidates=a.candidates, on_path=not a.all_rows, trainable=_csv(a.trainable) or ("*",),
                      qat=not a.no_qat, train_sets=_csv(a.train_sets), eval_sets=_csv(a.eval_sets),
                      domains=_csv(a.domains), modes=_csv(a.modes), eval_modes=_csv(a.eval_modes),
                      eval_every=a.eval_every,
                      eval_anchors_per_record=a.eval_anchors_per_record, ckpt_every=a.ckpt_every, seed=a.seed,
                      log_every=a.log_every)
    train(cfg)


def cmd_export(a) -> None:
    from .export import export_dspark
    from .train import load_trained

    params = load_trained(a.run, a.src)
    rep = export_dspark(a.src, params, a.out, fmt=a.fmt)
    print(json.dumps({k: (len(v) if k == "changed" else v) for k, v in rep.items()}))
    if a.check:
        from ..drafter import load_drafter

        d = load_drafter(a.out)
        print(f"[export] load_drafter OK: block {d.block}, taps {d.target_layer_ids}, window {d.sliding_window}")


def cmd_eval(a) -> None:
    from ..drafter import load_drafter
    from .evaluate import est_speedup, fmt_table, replay, teacher_forced
    from .objective import Spec, drafter_params
    from .records import Dataset
    from .target_head import TargetHead

    ds = Dataset(a.data, sets=_csv(a.sets) or None, domains=_csv(a.domains) or None, modes=_csv(a.modes) or None)
    if a.limit:
        ds.records = ds.records[:a.limit]
    head = TargetHead(a.head or a.data)
    widths = [w if w == "adaptive" else int(w) for w in _csv(a.width)]
    report = {"data": a.data, "sets": a.sets, "records": len(ds.records), "drafters": {}}
    for path in a.drafter:
        d = load_drafter(path)
        res = {}
        for w in widths:
            t0 = time.perf_counter()
            r = replay(d, head, ds, width=w, candidates=a.candidates)
            for dom, summ in r.items():
                summ["est_speedup_vs_ar"] = est_speedup(summ, drafter_ar=a.drafter_ar)
            res[f"replay_{w}"] = r
            print(fmt_table(r, f"\n{path} replay width={w} ({time.perf_counter() - t0:.1f} s)",
                            ("acceptance", "tokens_per_round", "rounds", "est_speedup_vs_ar")))
        if a.teacher_forced:
            spec = Spec.from_drafter(d)
            res["teacher_forced"] = teacher_forced(drafter_params(d)[0], spec, head, ds,
                                                   candidates=a.candidates,
                                                   max_anchors_per_record=a.tf_anchors_per_record)
            print(json.dumps({k: {"mean_L": v["mean_L"], "top1_by_row": v["top1_by_row"]}
                              for k, v in res["teacher_forced"].items()}))
        report["drafters"][path] = res
    report["cost_model"] = {"verify": "M31_VERIFY_AR_20261028 (01-LOG 2026-10-28 20:15, m5-max, R8)",
                            "drafter_ar": a.drafter_ar}
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"[eval] wrote {a.out}")


def cmd_info(a) -> None:
    from .records import Dataset

    ds = Dataset(a.data)
    s = ds.summary()
    rows = sum(r.n - r.tap_start for r in ds.records)
    s["tap_rows"] = rows
    s["tap_gb"] = round(rows * ds.width * 2 / 1e9, 2)
    s["by_set"] = {}
    for r in ds.records:
        b = s["by_set"].setdefault(r.set or "?", {"records": 0, "gen": 0})
        b["records"] += 1
        b["gen"] += r.n_gen
    s["meta"] = ds.meta
    print(json.dumps(s, indent=1))


def cmd_synth(a) -> None:
    from . import proof

    over = json.loads(a.train_overrides) if a.train_overrides else None
    proof.run(a.out, scale=a.scale, n_train=a.n_train, n_eval=a.n_eval, max_new=a.max_new, steps_src=a.steps_src,
              steps_ft=a.steps_ft, seed=a.seed, train_overrides=over)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vmlx_engine.dspark.finetune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="run the target, write training records")
    g.add_argument("--bundle", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--prompts", required=True, help="jsonl of token-id rows (prompt_ids or ids)")
    g.add_argument("--mode", choices=("generate", "teacher"), default="generate")
    g.add_argument("--sets", default="", help="keep rows whose 'set' is one of these (csv)")
    g.add_argument("--set-name", default="", help="store every record under this set name instead of its own")
    g.add_argument("--id-prefix", default=None,
                   help="record id prefix (default tf: teacher, gg: greedy, gs: sampled), so modes share a directory")
    g.add_argument("--domains", default="")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--max-new", type=int, default=1024)
    g.add_argument("--max-len", type=int, default=None, help="teacher: truncate rows")
    g.add_argument("--cut-turns", type=int, default=0, help="generate: prompts at the last N assistant headers of chat rows")
    g.add_argument("--prompt-ctx", type=int, default=None, help="prompt tap rows kept (default: the drafter window)")
    g.add_argument("--topk", type=int, default=0, help="also store the target's top-K per row (soft targets)")
    g.add_argument("--sample", default="", help="TEMP,TOP_P: sampled continuations (labels stay the argmax)")
    g.add_argument("--eos", default="200020")
    g.add_argument("--prefill-step", type=int, default=2048)
    g.add_argument("--shard-gb", type=float, default=1.0)
    g.add_argument("--cache-gb", type=float, default=4.0, help="mx.set_cache_limit during generation")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(fn=cmd_gen)

    t = sub.add_parser("train", help="fine-tune a drafter")
    t.add_argument("--data", required=True)
    t.add_argument("--drafter", required=True, help="the shipped dspark/ directory (initial weights)")
    t.add_argument("--out", required=True)
    t.add_argument("--head", default=None, help="directory with target_head.* (default: --data)")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--min-lr-frac", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--anchors", type=int, default=64, help="anchors per window (M)")
    t.add_argument("--windows", type=int, default=4, help="windows per optimizer step (accumulation)")
    t.add_argument("--spans", default="", help="block lengths to train, csv (default: the drafter's block)")
    t.add_argument("--pos-decay", type=float, default=0.0)
    t.add_argument("--conf-weight", type=float, default=1.0)
    t.add_argument("--soft-weight", type=float, default=0.0)
    t.add_argument("--base-weight", type=float, default=0.0)
    t.add_argument("--candidates", type=int, default=-1)
    t.add_argument("--all-rows", action="store_true", help="train off-path rows too (teacher data)")
    t.add_argument("--trainable", default="*", help="csv of fnmatch patterns, or all / adapter / heads")
    t.add_argument("--no-qat", action="store_true", help="no MXFP8 fake quantization (export --fmt bf16)")
    t.add_argument("--train-sets", default="calib")
    t.add_argument("--eval-sets", default="ref")
    t.add_argument("--domains", default="")
    t.add_argument("--modes", default="", help="record modes to train on: generate,generate-sampled,teacher")
    t.add_argument("--eval-modes", default="", help="record modes of the in-training held-out eval")
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--eval-anchors-per-record", type=int, default=512)
    t.add_argument("--ckpt-every", type=int, default=200)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="offline acceptance of drafter directories")
    e.add_argument("--data", required=True)
    e.add_argument("--drafter", action="append", required=True)
    e.add_argument("--head", default=None)
    e.add_argument("--sets", default="ref")
    e.add_argument("--domains", default="")
    e.add_argument("--modes", default="", help="generate (exact serving data) / generate-sampled / teacher")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--width", default="adaptive", help="csv of 'adaptive' and fixed widths")
    e.add_argument("--candidates", type=int, default=None)
    e.add_argument("--teacher-forced", action="store_true")
    e.add_argument("--tf-anchors-per-record", type=int, default=None)
    e.add_argument("--drafter-ar", type=float, default=0.3, help="drafter cost per round in AR steps (estimate)")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    x = sub.add_parser("export", help="write a run as a release-format dspark/ directory")
    x.add_argument("--run", required=True)
    x.add_argument("--src", required=True, help="the source dspark/ directory (layout, config, frozen tensors)")
    x.add_argument("--out", required=True)
    x.add_argument("--fmt", choices=("mxfp8", "bf16"), default="mxfp8")
    x.add_argument("--check", action="store_true", help="load the result with load_drafter")
    x.set_defaults(fn=cmd_export)

    i = sub.add_parser("info", help="summarize a data directory")
    i.add_argument("--data", required=True)
    i.set_defaults(fn=cmd_info)

    s = sub.add_parser("synth", help="end-to-end synthetic proof")
    s.add_argument("--out", required=True)
    s.add_argument("--scale", type=float, default=0.6)
    s.add_argument("--n-train", type=int, default=64)
    s.add_argument("--n-eval", type=int, default=16)
    s.add_argument("--max-new", type=int, default=128)
    s.add_argument("--steps-src", type=int, default=800)
    s.add_argument("--steps-ft", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--train-overrides", default="", help="JSON of TrainConfig fields for every stage")
    s.set_defaults(fn=cmd_synth)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
