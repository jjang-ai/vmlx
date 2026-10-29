"""MLX training loop for DSpark drafter fine-tuning (see ``objective`` for the loss).

Initialized from the shipped drafter (any directory ``load_drafter`` reads: the release layout or a pre-quantized
copy). Trainable tensors live as fp32 master copies, frozen ones as stored (BF16 / dequantized); compute is bf16.
AdamW with linear warmup + cosine decay, gradient accumulation over several sequence windows per step, global-norm
clipping, checkpoints (trainable tensors, optimizer state, data RNG) every ``ckpt_every`` steps with resume, held-out
teacher-forced acceptance every ``eval_every`` steps, a JSONL log.
"""

from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .evaluate import teacher_forced
from .objective import Spec, default_candidates, drafter_params, loss_fn, make_batch
from .records import Dataset, anchor_positions
from .target_head import TargetHead

PRESETS = {
    "all": ("*",),
    # The pieces that adapt the drafter to new taps and a new head without touching the decoder layers.
    "adapter": ("fc.*", "hidden_norm.*", "norm.*", "markov_head.*", "confidence_head.*", "layers.*norm.*"),
    "heads": ("markov_head.*", "confidence_head.*"),
}


@dataclass
class TrainConfig:
    data: str
    drafter: str
    out: str
    head: Optional[str] = None              # directory with target_head.* (default: the data directory)
    steps: int = 1000
    lr: float = 2e-5
    min_lr_frac: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.0
    clip: float = 1.0
    anchors: int = 64                       # anchors per sequence window (M)
    windows_per_step: int = 4               # gradient accumulation
    spans: tuple = ()                       # block lengths to train (default: the drafter's block); sampled uniformly
    pos_decay: float = 0.0                  # row weight exp(-i / pos_decay); 0 = uniform
    conf_weight: float = 1.0
    base_weight: float = 0.0                # CE on the base logits (no Markov bias); see objective.loss_fn
    candidates: int = -1                    # Markov candidates of the runtime pick (-1: the runtime default)
    soft_weight: float = 0.0                # blend of the target top-K distribution (needs --topk data)
    on_path: bool = True
    trainable: tuple = ("*",)               # fnmatch patterns over live tensor names, or one PRESETS key
    qat: bool = True                        # fake-quantize MXFP8-stored linears (export stays MXFP8)
    train_sets: tuple = ("calib",)
    eval_sets: tuple = ("ref",)
    domains: tuple = ()
    modes: tuple = ()                       # record modes to train on (generate / generate-sampled / teacher)
    eval_modes: tuple = ()
    eval_every: int = 200
    eval_anchors_per_record: int = 512
    ckpt_every: int = 200
    keep_ckpts: int = 2
    seed: int = 0
    log_every: int = 10
    extra: dict = field(default_factory=dict)


def _select(names, patterns) -> set:
    pats = PRESETS.get(patterns[0], patterns) if len(patterns) == 1 else patterns
    return {n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)}


class WindowSampler:
    """Random anchor windows: a record (weighted by its anchor count), then up to M consecutive anchors."""

    def __init__(self, ds: Dataset, m: int, seed: int):
        self.ds, self.m = ds, int(m)
        self.rng = np.random.default_rng(seed)
        self.items = []
        for rec in ds.records:
            a = anchor_positions(ds.gen(rec))
            if len(a):
                self.items.append((rec, a))
        if not self.items:
            raise ValueError("no anchor positions in the training records")
        w = np.array([len(a) for _, a in self.items], np.float64)
        self.p = w / w.sum()

    def __call__(self):
        rec, a = self.items[int(self.rng.choice(len(self.items), p=self.p))]
        i = int(self.rng.integers(0, len(a)))
        start = int(a[i])
        sel = a[(a >= start) & (a < start + self.m)]
        if len(sel) < max(1, self.m // 4) and i > 0:            # near the end of a span: take the window before
            sel = a[(a > start - self.m) & (a <= start)]
        return rec, sel


def _schedule(cfg: TrainConfig):
    warm = max(1, int(cfg.warmup))
    decay = max(1, int(cfg.steps) - warm)
    return optim.join_schedules([optim.linear_schedule(cfg.lr * 1e-3, cfg.lr, warm),
                                 optim.cosine_decay(cfg.lr, decay, end=cfg.lr * cfg.min_lr_frac)], [warm])


def setup(cfg: TrainConfig):
    from ..drafter import load_drafter

    drafter = load_drafter(cfg.drafter)
    spec = Spec.from_drafter(drafter)
    params, mxfp8 = drafter_params(drafter)
    del drafter
    names = _select(params, tuple(cfg.trainable))
    if not names:
        raise ValueError(f"no tensor matches trainable={cfg.trainable}")
    train = {k: params[k].astype(mx.float32) for k in sorted(names)}
    frozen = {k: v for k, v in params.items() if k not in names}
    fq = frozenset(m for m in mxfp8 if f"{m}.weight" in names) if cfg.qat else frozenset()
    head = TargetHead(cfg.head or str(cfg.data).split(",")[0])
    return spec, train, frozen, fq, head, mxfp8


def _save_ckpt(out: Path, step: int, train: dict, opt, sampler: WindowSampler, keep: int) -> Path:
    d = out / "ckpt" / f"step-{step:07d}"
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(d / "train.safetensors"), train)
    mx.save_safetensors(str(d / "optim.safetensors"), _flat_state(opt.state))
    (d / "state.json").write_text(json.dumps({"step": step, "rng": sampler.rng.bit_generator.state}))
    ckpts = sorted((out / "ckpt").glob("step-*"))
    for old in ckpts[:-keep] if keep else []:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    return d


def _flat_state(state: dict) -> dict:
    """Optimizer state of a flat parameter dict (dotted keys) -> one level of ``key|slot`` names."""
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out.update({f"{k}|{s}": a for s, a in v.items()})
        elif isinstance(v, mx.array):
            out[k] = v
    return out


def _unflat_state(flat: dict) -> dict:
    state = {}
    for k, v in flat.items():
        if "|" in k:
            name, slot = k.split("|", 1)
            state.setdefault(name, {})[slot] = v
        else:
            state[k] = v
    return state


def _latest(out: Path) -> Optional[Path]:
    ckpts = sorted(p for p in (out / "ckpt").glob("step-*") if (p / "state.json").is_file())
    return ckpts[-1] if ckpts else None


def merged_params(train: dict, frozen: dict) -> dict:
    return {**frozen, **train}


def train(cfg: TrainConfig, log=print) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1, default=list))
    spec, train_p, frozen, fq, head, _ = setup(cfg)
    ds = Dataset(cfg.data, sets=cfg.train_sets or None, domains=cfg.domains or None, modes=cfg.modes or None)
    ev = Dataset(cfg.data, sets=cfg.eval_sets or None, domains=cfg.domains or None,
                 modes=cfg.eval_modes or None) if cfg.eval_sets else None
    sampler = WindowSampler(ds, cfg.anchors, cfg.seed)
    spans = tuple(int(s) for s in (cfg.spans or (spec.block,)))
    cands = default_candidates() if cfg.candidates < 0 else int(cfg.candidates)
    opt = optim.AdamW(learning_rate=_schedule(cfg), weight_decay=cfg.weight_decay)
    opt.init(train_p)
    start = 0
    ck = _latest(out)
    if ck is not None:
        loaded = mx.load(str(ck / "train.safetensors"))
        # The order of the parameter dict is the order of every tree reduction (the clip norm): keep setup's order.
        train_p = {k: loaded[k] for k in train_p}
        opt.state = _unflat_state(mx.load(str(ck / "optim.safetensors")))
        st = json.loads((ck / "state.json").read_text())
        start = int(st["step"])
        sampler.rng.bit_generator.state = st["rng"]
        log(f"[train] resumed from {ck} at step {start}")
    n_train = sum(int(v.size) for v in train_p.values())
    log(f"[train] {len(train_p)} trainable tensors ({n_train / 1e6:.1f} M params), {len(frozen)} frozen, "
        f"fake-quant MXFP8 on {len(fq)} linears; {len(sampler.items)} train records, "
        f"{sum(len(a) for _, a in sampler.items)} anchors; spans {spans}")
    logf = open(out / "train_log.jsonl", "a")
    history = []

    def evaluate(step):
        if ev is None or not ev.records:
            return None
        t0 = time.perf_counter()
        res = teacher_forced(merged_params(train_p, frozen), spec, head, ev, anchors_per_batch=cfg.anchors,
                             fq=fq, on_path=cfg.on_path, max_anchors_per_record=cfg.eval_anchors_per_record,
                             candidates=cands)
        row = {"step": step, "eval": res, "eval_s": round(time.perf_counter() - t0, 2)}
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        a = res["all"]
        log(f"[eval] step {step}: mean_L {a['mean_L']} ce {a['ce']} top1/row {a['top1_by_row']} "
            f"acc@{spec.block} {a['fixed_width'].get(spec.block, {}).get('acceptance')}")
        history.append(row)
        return res

    if start == 0:
        evaluate(0)
    t_last = time.perf_counter()
    rows_since = 0
    for step in range(start + 1, cfg.steps + 1):
        grads = None
        agg = {"loss": 0.0, "ce": 0.0, "bce": 0.0, "rows": 0, "correct": 0}
        for _ in range(cfg.windows_per_step):
            rec, anchors = sampler()
            span = spans[int(sampler.rng.integers(0, len(spans)))]
            b = make_batch(ds, rec, anchors, span, spec, on_path=cfg.on_path, pos_decay=cfg.pos_decay,
                           soft=cfg.soft_weight > 0)

            def f(tp):
                return loss_fn(tp, frozen, spec, head, b, fq, conf_weight=cfg.conf_weight,
                               soft_weight=cfg.soft_weight, base_weight=cfg.base_weight, candidates=cands)

            (loss, aux), g = mx.value_and_grad(f)(train_p)
            grads = g if grads is None else tree_map(mx.add, grads, g)
            mx.eval(grads, loss, aux["ce"], aux["bce"], aux["correct"])
            agg["loss"] += float(loss)
            agg["ce"] += float(aux["ce"])
            agg["bce"] += float(aux["bce"])
            agg["rows"] += int(np.asarray(b.valid).sum())
            agg["correct"] += int(np.asarray(aux["correct"]).sum())
        grads = tree_map(lambda g: g / cfg.windows_per_step, grads)
        grads, gnorm = optim.clip_grad_norm(grads, cfg.clip)
        train_p = opt.apply_gradients(grads, train_p)
        mx.eval(train_p, opt.state)
        rows_since += agg["rows"]
        if step % cfg.log_every == 0 or step == cfg.steps:
            dt = time.perf_counter() - t_last
            w = cfg.windows_per_step
            row = {"step": step, "loss": round(agg["loss"] / w, 5), "ce": round(agg["ce"] / w, 5),
                   "bce": round(agg["bce"] / w, 5), "top1": round(agg["correct"] / max(agg["rows"], 1), 4),
                   "gnorm": round(float(gnorm), 4), "lr": float(opt.learning_rate),
                   "rows_per_s": round(rows_since / dt, 1), "s_per_step": round(dt / cfg.log_every, 3)}
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            log(f"[train] {json.dumps(row)}")
            t_last, rows_since = time.perf_counter(), 0
        if cfg.eval_every and step % cfg.eval_every == 0 and step != cfg.steps:
            evaluate(step)
        if cfg.ckpt_every and step % cfg.ckpt_every == 0:
            _save_ckpt(out, step, train_p, opt, sampler, cfg.keep_ckpts)
    final = out / "final"
    final.mkdir(exist_ok=True)
    mx.save_safetensors(str(final / "train.safetensors"), train_p)
    res = evaluate(cfg.steps)
    logf.close()
    return {"spec": spec, "train": train_p, "frozen": frozen, "fq": fq, "head": head, "final_eval": res,
            "history": history}


def load_trained(cfg_or_out, drafter_dir: Optional[str] = None) -> dict:
    """All drafter tensors (live module paths, float) of a finished or checkpointed run: the shipped drafter's values
    with the run's trained tensors on top."""
    from ..drafter import load_drafter

    out = Path(cfg_or_out)
    cfg = json.loads((out / "train_config.json").read_text())
    src = drafter_dir or cfg["drafter"]
    params, _ = drafter_params(load_drafter(src))
    path = out / "final" / "train.safetensors"
    if not path.is_file():
        ck = _latest(out)
        if ck is None:
            raise FileNotFoundError(f"{out}: no final weights or checkpoint")
        path = ck / "train.safetensors"
    params.update(dict(mx.load(str(path))))
    return params
