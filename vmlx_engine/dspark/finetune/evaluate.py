"""Offline DSpark acceptance on recorded target data (no target model needed).

Two meters:

- ``teacher_forced``: the training forward over every anchor of the held-out records -> per anchor the accepted-prefix
  length L (leading block rows whose argmax is the target's token; with teacher-forced previous tokens these are
  exactly the drafts a greedy verify accepts), per-row top-1 / CE, and the serving ROUNDS simulated from L at every
  fixed width k <= the trained block (a round at anchor p accepts min(L(p), k) drafts and moves to p + accepted + 1).
- ``replay``: the serving rounds through the runtime's own ``DSparkDrafter`` (context cache fed with the recorded tap
  rows of the committed tokens, sliding window, Markov candidates, blocks longer than the trained block when the width
  exceeds it) with the runtime's Adaptive width policy (``generate.ARFallback`` width choice, no wall-clock AR fallback)
  or a fixed width. Acceptance = accepted / drafted, tokens per round = (accepted + 1) per round, as ``DSparkStats``.

On greedy generations of the target both meters are the greedy-verify acceptance up to the arithmetic of the recorded
taps (one teacher-forced pass; the server feeds verify rows: same model, multi-row vs one-row kernels). On
teacher-forced conversations the labels are the target's argmax given the conversation's own tokens.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import mlx.core as mx
import numpy as np

from .objective import accepted_lengths, default_candidates, forward, make_batch, pick_tokens
from .records import anchor_positions


def gen_spans(gen: np.ndarray) -> list:
    """Maximal runs [s, e) of target-produced positions with at least two positions (one anchor + one label)."""
    g = np.asarray(gen, bool)
    out, i, n = [], 0, len(g)
    while i < n:
        if g[i]:
            j = i
            while j < n and g[j]:
                j += 1
            if j - i >= 2:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def simulate_rounds(L: dict, spans: list, k: int) -> dict:
    """Serving rounds at fixed width ``k`` from per-anchor accepted lengths ``L`` (position -> L)."""
    rounds = drafted = accepted = 0
    for s, e in spans:
        p = s
        while p + 1 < e:
            kk = min(k, e - 1 - p)
            a = min(int(L.get(p, 0)), kk)
            rounds += 1
            drafted += kk
            accepted += a
            p += a + 1
    return {"rounds": rounds, "drafted": drafted, "accepted": accepted,
            "acceptance": round(accepted / drafted, 4) if drafted else None,
            "tokens_per_round": round((accepted + rounds) / rounds, 4) if rounds else None}


def teacher_forced(p: dict, spec, head, ds, records=None, *, anchors_per_batch: int = 64, span: Optional[int] = None,
                   fq=frozenset(), on_path: bool = True, max_anchors_per_record: Optional[int] = None,
                   candidates: Optional[int] = None) -> dict:
    """Held-out metrics from the training forward (see the module docstring). A row is correct when the token the
    serving chain would draft there (``pick_tokens``: Markov rescoring of the top-``candidates`` base logits, the
    runtime default) is the target's label."""
    span = int(span or spec.block)
    candidates = default_candidates() if candidates is None else int(candidates)
    records = ds.records if records is None else records
    by_dom = defaultdict(lambda: {"L": [], "ce": 0.0, "rows": 0, "top1": np.zeros(span), "valid": np.zeros(span),
                                  "sim": defaultdict(lambda: np.zeros(3, np.int64))})
    for rec in records:
        anchors = anchor_positions(ds.gen(rec))
        if max_anchors_per_record:
            anchors = anchors[:max_anchors_per_record]
        if not len(anchors):
            continue
        L = {}
        d = by_dom[rec.domain or "?"]
        for i in range(0, len(anchors), anchors_per_batch):
            a = anchors[i:i + anchors_per_batch]
            b = make_batch(ds, rec, a, span, spec, on_path=on_path)
            _, base, bias, _ = forward(p, spec, head, b, fq)
            logits = base + bias
            lse = mx.logsumexp(logits, axis=-1)
            ce = lse - mx.take_along_axis(logits, b.labels[..., None], axis=-1)[..., 0]
            correct = (pick_tokens(base, bias, candidates) == b.labels) & b.valid
            mx.eval(ce, correct)
            valid = np.asarray(b.valid)
            corr = np.asarray(correct)
            d["ce"] += float((np.asarray(ce) * valid).sum())
            d["rows"] += int(valid.sum())
            d["top1"] += corr.sum(0)
            d["valid"] += valid.sum(0)
            for pos, l in zip(a.tolist(), accepted_lengths(correct, b.valid).tolist()):
                L[pos] = l
        d["L"].extend(L.values())
        spans = gen_spans(ds.gen(rec))
        for k in range(1, span + 1):
            r = simulate_rounds(L, spans, k)
            d["sim"][k] += np.array([r["rounds"], r["drafted"], r["accepted"]])
    out, tot = {}, {"L": [], "ce": 0.0, "rows": 0, "top1": np.zeros(span), "valid": np.zeros(span),
                    "sim": defaultdict(lambda: np.zeros(3, np.int64))}
    for dom, d in sorted(by_dom.items()):
        out[dom] = _summarize(d, span)
        tot["L"] += d["L"]
        tot["ce"] += d["ce"]
        tot["rows"] += d["rows"]
        tot["top1"] += d["top1"]
        tot["valid"] += d["valid"]
        for k, v in d["sim"].items():
            tot["sim"][k] += v
    out["all"] = _summarize(tot, span)
    return out


def _summarize(d: dict, span: int) -> dict:
    sim = {}
    for k, (r, dr, ac) in sorted(d["sim"].items()):
        sim[k] = {"acceptance": round(ac / dr, 4) if dr else None, "tokens_per_round": round((ac + r) / r, 4) if r else None}
    return {"anchors": len(d["L"]), "mean_L": round(float(np.mean(d["L"])), 4) if d["L"] else None,
            "ce": round(d["ce"] / max(d["rows"], 1), 4),
            "top1_by_row": [round(float(t / v), 4) if v else None for t, v in zip(d["top1"], d["valid"])],
            "fixed_width": sim}


class _WidthOnly:
    """The runtime's Adaptive width choice (``ARFallback.planned_width`` from the per-position acceptance history)
    without the wall-clock AR fallback (offline there is no clock)."""

    def __init__(self):
        from ..generate import ARFallback

        self.inner = ARFallback()

    def width(self, max_draft: int, cost_per_row: float) -> int:
        return int(self.inner.planned_width(max_draft, cost_per_row))

    def record(self, drafted: int, accepted: int) -> None:
        self.inner.record_acceptance(drafted, accepted)


def replay(drafter, head, ds, records=None, *, width="adaptive", max_draft: Optional[int] = None,
           cost_per_row: float = 0.12, candidates: Optional[int] = None, room_rule: str = "labels") -> dict:
    """Serving rounds through the runtime ``DSparkDrafter`` (see the module docstring). ``width``: "adaptive" or an
    int. ``max_draft``: the ladder top (default the runtime's: twice the trained block). ``room_rule``: "labels"
    drafts up to every recorded label left in the span; "max_new" mirrors ``iter_dspark`` stopping at
    ``max_new`` = the span's length (a round drafts at most room - 1, a zero-draft round when one token is left)."""
    records = ds.records if records is None else records
    top = int(max_draft or 2 * int(drafter.block))
    mask_embed = head.embed(mx.array([[drafter.mask_token_id]]))
    by_dom = defaultdict(lambda: {"rounds": 0, "drafted": 0, "accepted": 0, "widths": defaultdict(int),
                                  "pos_drafted": np.zeros(top, np.int64), "pos_accepted": np.zeros(top, np.int64)})
    for rec in records:
        tokens, labels, gen = ds.tokens(rec), ds.labels(rec), ds.gen(rec)
        d = by_dom[rec.domain or "?"]
        policy = _WidthOnly() if width == "adaptive" else None
        drafter.reset(base=rec.tap_start)
        cur = rec.tap_start

        def feed(a: int, b: int):
            if b > a:
                drafter.add_context(mx.view(mx.array(np.asarray(ds.taps(rec, a, b))), mx.bfloat16)[None])

        for s, e in gen_spans(gen):
            feed(cur, s)
            p = s
            while p + 1 < e:
                cap = e - 1 - p if room_rule == "labels" else e - p - 2
                if cap < 1:                                   # iter_dspark: one token left -> a zero-draft round
                    d["rounds"] += 1
                    feed(p, p + 1)
                    p += 1
                    continue
                k = policy.width(top, cost_per_row) if policy is not None else int(width)
                k = max(1, min(k, cap, top))
                span = max(int(drafter.block), k)
                anchor = mx.array([int(tokens[p])], dtype=mx.int32)
                blk = mx.concatenate([head.embed(anchor.reshape(1, 1)),
                                      mx.broadcast_to(mask_embed, (1, span - 1, mask_embed.shape[-1]))], axis=1)
                hidden = drafter(blk)
                drafts, _ = drafter.draft(hidden, anchor, head.head, k, candidates=candidates)
                dl = drafts.tolist()
                a = 0
                while a < k and dl[a] == int(labels[p + 1 + a]):
                    a += 1
                d["rounds"] += 1
                d["drafted"] += k
                d["accepted"] += a
                d["widths"][k + 1] += 1
                d["pos_drafted"][:k] += 1
                d["pos_accepted"][:a] += 1
                if policy is not None:
                    policy.record(k, a)
                feed(p, p + a + 1)
                p += a + 1
            cur = p
    out, tot = {}, {"rounds": 0, "drafted": 0, "accepted": 0, "widths": defaultdict(int),
                    "pos_drafted": np.zeros(top, np.int64), "pos_accepted": np.zeros(top, np.int64)}
    for dom, d in sorted(by_dom.items()):
        out[dom] = _round_summary(d)
        for k in ("rounds", "drafted", "accepted", "pos_drafted", "pos_accepted"):
            tot[k] = tot[k] + d[k]
        for w, c in d["widths"].items():
            tot["widths"][w] += c
    out["all"] = _round_summary(tot)
    return out


def _round_summary(d: dict) -> dict:
    r, dr, ac = d["rounds"], d["drafted"], d["accepted"]
    pd, pa = d["pos_drafted"], d["pos_accepted"]
    return {"rounds": r, "drafted": dr, "accepted": ac, "acceptance": round(ac / dr, 4) if dr else None,
            "tokens_per_round": round((ac + r) / r, 4) if r else None,
            "accept_by_position": [round(float(a / x), 4) for a, x in zip(pa, pd) if x],
            "widths": dict(sorted(d["widths"].items()))}


# Verify cost of R rows in AR-step units, MEASURED on the real MiniMax-M3.1-Flash-JANGHT1.9 bundle (campaign 01-LOG,
# 2026-10-28 20:15, erics-m5-max, R8 runtime before the fused decode / verify kernels, scripts/mm31t/profile_verify.py:
# 30.4 ms for one row). Re-measure on the runtime you ship; rows between measured points are interpolated.
M31_VERIFY_AR_20261028 = {1: 1.00, 2: 1.50, 3: 1.88, 4: 2.32, 5: 2.81, 6: 3.22, 8: 4.09, 10: 5.00, 12: 5.87, 16: 7.80}


def verify_cost(rows: int, table: dict = M31_VERIFY_AR_20261028) -> float:
    pts = sorted(table.items())
    if rows <= pts[0][0]:
        return pts[0][1]
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if rows <= r1:
            return c0 + (c1 - c0) * (rows - r0) / (r1 - r0)
    (r0, c0), (r1, c1) = pts[-2], pts[-1]
    return c1 + (c1 - c0) / (r1 - r0) * (rows - r1)


def est_speedup(summary: dict, *, drafter_ar: float = 0.3, table: dict = M31_VERIFY_AR_20261028) -> Optional[float]:
    """ESTIMATE of DSpark decode speed vs plain decoding for a ``replay`` summary: committed tokens / (per round: the
    drafter, ``drafter_ar`` AR steps, + a verify of its rows from ``table``). Rounds only (Adaptive's AR fallback
    would take AR steps where this is < 1). A cost model, not a measurement."""
    widths = summary.get("widths") or {}
    rounds = summary.get("rounds") or 0
    if not rounds:
        return None
    zero = rounds - sum(widths.values())
    cost = zero * (drafter_ar + verify_cost(1, table)) + sum(
        c * (drafter_ar + verify_cost(int(r), table)) for r, c in widths.items())
    return round((summary["accepted"] + rounds) / cost, 4)


def compare(before: dict, after: dict, key: str = "acceptance") -> dict:
    """Per domain before / after / delta of one metric of two ``replay`` results."""
    out = {}
    for dom in sorted(set(before) | set(after)):
        b, a = (before.get(dom) or {}).get(key), (after.get(dom) or {}).get(key)
        out[dom] = {"before": b, "after": a, "delta": round(a - b, 4) if a is not None and b is not None else None}
    return out


def fmt_table(results: dict, title: str, keys: Sequence[str] = ("acceptance", "tokens_per_round", "rounds")) -> str:
    lines = [title, "| domain | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for dom, r in results.items():
        lines.append(f"| {dom} | " + " | ".join(str(r.get(k)) for k in keys) + " |")
    return "\n".join(lines)
