"""Training data for DSpark fine-tuning: run the TARGET (the quantized bundle) and record what the drafter consumes.

Per sequence (``records.Record``): token ids, the target's own label at every position, which positions the target
generated, and the drafter's tap features (``target_layer_ids`` residual rows, bf16) for every position the drafter's
sliding window can read.

Modes:
- ``generate``: a prompt (already chat-formatted with the bundle's template, as token ids) -> the target's continuation,
  greedy (argmax, the tokens ``generate_ar`` produces) or sampled, recorded on exactly the arithmetic serving feeds the
  drafter: the prompt's tap rows from ``target.prefill`` (served chunking, last ``prompt_ctx`` rows), every generated
  token's row from a one-row ``target.ar_step`` (bit-equal to the served decode step and to a DSpark verify row).
  Labels on the generated part: the target's argmax at each step (= the generated tokens when greedy).
- ``teacher``: an existing conversation (e.g. the corpus' native traces) -> one teacher-forced pass of the target's own
  forward (``MiniMaxM3DSparkTarget._stock`` with ``return_aux`` taps, ``prefill_step`` chunks); the generated positions
  are the assistant spans (after the ``]~b]ai\\n`` header through the closing ``[e~[``) and the labels are the
  target's argmax at each of them. Prefill speed (no decoding): ~10-30x more tokens per hour than ``generate``; the
  assistant rows are on multi-row (prefill) arithmetic and the prefix is the conversation's, not the target's.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from .records import Record, ShardWriter

# MiniMax-M3 / M3.1 chat template tokens (tokenizer of the M3.1 release): ']~b]' 'ai' '\n' opens an assistant message,
# '[e~[' closes every message (and is the EOS the model emits).
M3_ASSISTANT_HEADER = (200019, 1361, 10)
M3_TURN_END = 200020


# ----------------------------------------------------------------------------------------------- prompts / spans
def assistant_mask(ids: Sequence[int], header: Sequence[int] = M3_ASSISTANT_HEADER, end: int = M3_TURN_END) -> np.ndarray:
    """1 at every token the model writes in an assistant message: after each ``header`` up to and including the next
    ``end`` (or the end of the sequence)."""
    ids = np.asarray(ids, np.int64)
    gen = np.zeros(len(ids), np.uint8)
    h = len(header)
    i, inside = 0, False
    while i < len(ids):
        if not inside and i + h <= len(ids) and all(int(ids[i + j]) == header[j] for j in range(h)):
            inside = True
            i += h
            continue
        if inside:
            gen[i] = 1
            if int(ids[i]) == end:
                inside = False
        i += 1
    return gen


def _interleave(rows: list, key: str = "domain") -> list:
    """Round-robin over domains, so a partial run covers every domain."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "?"].append(r)
    out, queues = [], [list(v) for _, v in sorted(groups.items())]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def load_items(path: str | Path, mode: str, *, sets=None, domains=None, limit: Optional[int] = None,
               max_len: Optional[int] = None, kinds=("chat",), cut_turns: int = 0,
               header: Sequence[int] = M3_ASSISTANT_HEADER) -> list:
    """Work items ``{id, domain, set, ids, [gen]}`` from a jsonl of token-id rows.

    ``generate``: rows with ``prompt_ids`` (a prompt that ends at the assistant generation point), or rows with ``ids``
    cut at assistant headers (``cut_turns`` > 0: the last ``cut_turns`` assistant turns of each chat row each become a
    prompt; with ``gen_from`` the row's own prompt ``ids[:gen_from]``).
    ``teacher``: rows with ``ids`` (``kind`` in ``kinds``), ``gen`` = their assistant spans; ``max_len`` truncates."""
    sets, domains = (set(sets) if sets else None), (set(domains) if domains else None)
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if sets and r.get("set") not in sets or domains and r.get("domain") not in domains:
                continue
            base = {"id": r["id"], "domain": r.get("domain", ""), "set": r.get("set", ""),
                    "source": r.get("source", "")}
            if mode == "generate":
                if "prompt_ids" in r:
                    rows.append({**base, "ids": list(r["prompt_ids"])})
                elif "ids" in r and r.get("gen_from") is not None and not cut_turns:
                    rows.append({**base, "ids": list(r["ids"][:int(r["gen_from"])])})
                elif "ids" in r and cut_turns and r.get("kind", "chat") in kinds:
                    ids = list(r["ids"])
                    h = len(header)
                    cuts = [i + h for i in range(len(ids) - h + 1) if tuple(ids[i:i + h]) == tuple(header)]
                    for c in cuts[-cut_turns:]:
                        rows.append({**base, "id": f"{r['id']}@{c}", "ids": ids[:c]})
            elif mode == "teacher":
                if "ids" not in r or r.get("kind", "chat") not in kinds:
                    continue
                ids = list(r["ids"])[: max_len or None]
                gen = assistant_mask(ids, header)
                if gen.sum() < 2:
                    continue
                rows.append({**base, "ids": ids, "gen": gen})
            else:
                raise ValueError(f"mode {mode!r}")
    rows = _interleave(rows)
    return rows[:limit] if limit else rows


# ----------------------------------------------------------------------------------------------- target runs
class TargetRunner:
    """Runs a DSpark target adapter (``MiniMaxM3DSparkTarget``) for data generation."""

    def __init__(self, target, taps: Sequence[int], *, prefill_step: int = 2048, topk: int = 0):
        self.target = target
        self.taps = tuple(int(t) for t in taps)
        self.step = int(prefill_step)
        self.topk = int(topk)

    def teacher_pass(self, ids: Sequence[int], tap_start: int):
        """One causal pass over ``ids`` (chunks of ``prefill_step``, fresh cache). Returns (taps uint16 [n - tap_start, F]
        for positions tap_start.., picks int32 [n] = argmax of row j (the target's token for position j + 1), top-K ids
        int32 [n, K] / log-probs float16 [n, K] or None)."""
        import mlx.core as mx

        t = self.target
        n = len(ids)
        cache = t.make_cache()
        picks = np.empty(n, np.int32)
        topk_ids = np.empty((n, self.topk), np.int32) if self.topk else None
        topk_lp = np.empty((n, self.topk), np.float16) if self.topk else None
        taps = []
        a = 0
        while a < n:
            b = min(n, a + self.step)
            if n - b == 1 and b - a > 1:
                b -= 1                                    # never a lone final row (the served chunking rule)
            logits, feats = t._stock(mx.array([list(ids[a:b])]), cache, self.taps, logits=True)
            lg = logits[0]
            pk = mx.argmax(lg, axis=-1)
            outs = [pk]
            if self.topk:
                idx, val = self._topk(lg)
                outs += [idx, val]
            lo = max(a, int(tap_start))
            keep = feats[0, lo - a:] if lo < b else None
            if keep is not None:
                outs.append(keep)
            mx.eval(outs, [c.state for c in cache])
            picks[a:b] = np.array(pk).astype(np.int32)
            if self.topk:
                topk_ids[a:b] = np.array(idx).astype(np.int32)
                topk_lp[a:b] = np.array(val).astype(np.float16)
            if keep is not None:
                taps.append(np.array(mx.view(keep, mx.uint16)))
            a = b
        width = len(self.taps) * int(t.args.hidden_size) if hasattr(t, "args") else None
        out = np.concatenate(taps) if taps else np.zeros((0, width or 0), np.uint16)
        return out, picks, topk_ids, topk_lp

    def _topk(self, lg):
        import mlx.core as mx

        lf = lg.astype(mx.float32)
        lse = mx.logsumexp(lf, axis=-1, keepdims=True)
        idx = mx.argpartition(-lf, kth=self.topk - 1, axis=-1)[..., :self.topk]
        val = mx.take_along_axis(lf, idx, axis=-1) - lse
        order = mx.argsort(-val, axis=-1)
        return mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(val, order, axis=-1)

    def generate(self, prompt: Sequence[int], max_new: int, eos: set, *, keep_prompt: int, sampler=None,
                 seed: int = 0):
        """The target's continuation with the drafter's tap rows, on the arithmetic serving feeds the drafter:
        the prompt through ``target.prefill`` (the served chunking; its last ``keep_prompt`` tap rows), every
        generated token through ``target.ar_step`` (one-row decode steps: bit-equal to the served decode and to
        DSpark verify rows). Greedy (argmax, = ``generate_ar``) or sampled (``sampler``: log-probs -> token).
        Steps are pipelined (step i + 1 queued before token i is read); the step that consumes the last token runs
        too, so every position has its tap row.

        Returns (generated tokens, tap rows uint16 [min(P, keep_prompt) + G, F], argmax pick per generated
        position int32 [G], top-K ids / log-probs per generated position or None)."""
        import mlx.core as mx

        t = self.target
        mx.random.seed(int(seed))
        cache = t.make_cache()
        logits, feats = t.prefill(mx.array([list(prompt)]), cache, self.taps, self.step,
                                  keep_last=int(keep_prompt) or None)
        rows = [np.array(mx.view(feats[0], mx.uint16))]

        def choose(lg):
            pick = mx.argmax(lg, axis=-1).reshape(1)
            if sampler is None:
                return pick, pick
            lp = lg.astype(mx.float32)
            return sampler((lp - mx.logsumexp(lp))[None]).reshape(1).astype(pick.dtype), pick

        tok, pick = choose(logits)
        extra = self._topk(logits) if self.topk else ()
        out, picks, tk_ids, tk_lp = [], [], [], []
        while True:
            nxt_logits, nxt_feats = t.ar_step(tok, cache, self.taps)
            mx.async_eval(nxt_logits, nxt_feats)
            out.append(int(tok.item()))
            picks.append(int(pick.item()))
            if self.topk:
                tk_ids.append(np.array(extra[0]))
                tk_lp.append(np.array(extra[1]))
            mx.eval(nxt_feats)
            rows.append(np.array(mx.view(nxt_feats[0], mx.uint16)))
            if len(out) >= max_new or out[-1] in eos:
                break
            tok, pick = choose(nxt_logits)
            extra = self._topk(nxt_logits) if self.topk else ()
        taps = np.concatenate(rows)
        topk = (np.stack(tk_ids).astype(np.int32), np.stack(tk_lp).astype(np.float16)) if self.topk else (None, None)
        return out, taps, np.asarray(picks, np.int32), topk[0], topk[1]


def make_record(item: dict, runner: TargetRunner, *, mode: str, prompt_ctx: int, max_new: int = 0, eos=(),
                sampler=None, seed: int = 0) -> tuple[Record, dict]:
    """Run one work item; returns (Record, arrays for ``ShardWriter.add``)."""
    t0 = time.perf_counter()
    if mode == "generate":
        prompt = list(item["ids"])
        out, taps, gpicks, gk_ids, gk_lp = runner.generate(prompt, max_new, set(eos), keep_prompt=prompt_ctx,
                                                           sampler=sampler, seed=seed)
        P, G = len(prompt), len(out)
        ids = prompt + list(out)
        n = len(ids)
        tokens = np.asarray(ids, np.int32)
        gen = np.zeros(n, np.uint8)
        gen[P:] = 1
        picks = np.full(n, -1, np.int32)
        picks[P - 1:n - 1] = gpicks                       # row j's argmax is the target's token for j + 1
        labels = tokens.copy()
        labels[P:] = gpicks                               # greedy: == the generated tokens; sampled: the argmax
        tap_start = n - int(taps.shape[0])
        topk_ids = topk_lp = None
        if runner.topk:
            topk_ids = np.zeros((n, runner.topk), np.int32)
            topk_lp = np.zeros((n, runner.topk), np.float16)
            topk_ids[P - 1:n - 1], topk_lp[P - 1:n - 1] = gk_ids.reshape(G, -1), gk_lp.reshape(G, -1)
        extra = {"gen_s": round(time.perf_counter() - t0, 3), "pass_s": 0.0, "prompt": P}
        sampled = sampler is not None
    else:
        ids = list(item["ids"])
        n = len(ids)
        gen = np.asarray(item["gen"], np.uint8)
        first = int(np.argmax(gen)) if gen.any() else n
        tap_start = max(0, first - int(prompt_ctx))
        taps, picks, topk_ids, topk_lp = runner.teacher_pass(ids, tap_start)
        tokens = np.asarray(ids, np.int32)
        labels = np.empty_like(tokens)
        labels[0] = tokens[0]
        labels[1:] = picks[:-1]
        g = gen.astype(bool)
        g[0] = False
        extra = {"gen_s": 0.0, "pass_s": round(time.perf_counter() - t0, 3),
                 "corpus_vs_target_top1": round(float(np.mean(labels[g] == tokens[g])), 5) if g.any() else None}
        sampled = False
    rec = Record(id=item["id"], n=n, tap_start=tap_start, domain=item.get("domain", ""),
                 set=item.get("set", ""), mode=mode + ("-sampled" if sampled else ""),
                 n_gen=int(gen.sum()), extra=extra)
    arrays = {"tokens": tokens, "labels": labels, "gen": gen, "taps": taps, "picks": picks,
              "topk_ids": topk_ids, "topk_logprobs": topk_lp}
    return rec, arrays


def run(items: list, runner: TargetRunner, writer: ShardWriter, *, mode: str, prompt_ctx: int, max_new: int = 0,
        eos=(), sampler=None, seed: int = 0, log=print) -> dict:
    """Generate every item not yet in the writer's manifest; flushes at the end. Returns throughput totals."""
    done = writer.done_ids()
    todo = [it for it in items if it["id"] not in done]
    log(f"[gen] {len(items)} items, {len(done)} already done, {len(todo)} to run")
    tot = {"records": 0, "tokens": 0, "gen_tokens": 0, "gen_s": 0.0, "pass_s": 0.0, "tap_rows": 0}
    t_start = time.perf_counter()
    for i, item in enumerate(todo):
        rec, arrays = make_record(item, runner, mode=mode, prompt_ctx=prompt_ctx, max_new=max_new, eos=eos,
                                  sampler=sampler, seed=seed + i)
        writer.add(rec, **arrays)
        tot["records"] += 1
        tot["tokens"] += rec.n
        tot["gen_tokens"] += rec.n_gen
        tot["gen_s"] += rec.extra["gen_s"]
        tot["pass_s"] += rec.extra["pass_s"]
        tot["tap_rows"] += rec.n - rec.tap_start
        el = time.perf_counter() - t_start
        log(f"[gen] {i + 1}/{len(todo)} {rec.id} dom={rec.domain} n={rec.n} gen={rec.n_gen} "
            f"gen_s={rec.extra['gen_s']} pass_s={rec.extra['pass_s']} | total gen tok {tot['gen_tokens']} "
            f"{tot['gen_tokens'] / max(el, 1e-9):.1f} gen tok/s, ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min")
    writer.close()
    tot["wall_s"] = round(time.perf_counter() - t_start, 2)
    tot["tap_gb"] = round(tot["tap_rows"] * writer.width * 2 / 1e9, 3)
    return tot


def iter_prompt_rows(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            yield json.loads(line)
