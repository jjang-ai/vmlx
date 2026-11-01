# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 MSA: block-sparse attention paths vs the dense-mask path, synthetic tensors (no model, no weights).

Decode / verify rows: per query position p (keys 0..p), every path gets the SAME selection from the model's
Indexer.select and the same q / K / V; the dense path attends all keys under the [B, G, 1, Sk] mask, the sparse
paths ("decode" kernel, op-level "gather") only the selected blocks. Prefill chunks (--prefill L): an L-row chunk
ending at Sk through the chunk kernel ("kernel": matrix units for bf16, simdgroup matrices for fp32) vs dense.

Per (dtype, groups, Sk, path), over positions:
  rel   ||o_path - o_dense|| / ||o_dense|| over the H * D output row: median / p99 / max
  ref   the same against float64 attention (the dense path's own distance from float64 is printed as "dense")
  top1  argmax agreement with the dense path after a random o_proj (H*D -> 1024) and head (1024 -> 4096)

Negative controls replace the selection fed to the sparse path; each must fail the tolerance the real one passes:
  shared     every group uses group 0's blocks
  off_by_one every block id + 1 (mod n_blocks)
  no_local   the query's own block swapped for an unselected one

    python bench/minimax_m3_msa_equivalence.py                  # decode / verify rows
    python bench/minimax_m3_msa_equivalence.py --prefill 512     # also a 512-row chunk at each Sk
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mlx.core as mx

H, D, IDX_D, BLOCK, TOPK = 64, 128, 128, 128, 16
SIZES = (2047, 2048, 2049, 2853, 3011, 8191 + 77)


def _attention(groups: int):
    from vmlx_engine.models.minimax_m3.minimax_m3 import Attention, ModelArgs

    args = ModelArgs(hidden_size=256, num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=groups,
                     head_dim=D, rotary_dim=64, index_n_heads=groups, index_head_dim=IDX_D,
                     index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1,
                     sparse_attention_freq=[1])
    return Attention(args, 0)


def tensors(sk: int, groups: int, dtype, seed: int = 0, rows: int = 0):
    """q [1, H, rows or sk, D], k / v [1, G, sk, D], idx_q [1, G, ., IDX_D], idx_k [1, 1, sk, IDX_D].

    Index keys carry a slowly varying component so neighbouring blocks score alike (closer to a trained indexer
    than white noise) while groups still pick different blocks."""
    rng = np.random.default_rng(seed)
    n = rows or sk
    drift = np.cumsum(rng.standard_normal((sk, IDX_D)) * 0.05, axis=0)
    ik = rng.standard_normal((sk, IDX_D)) + drift
    k = rng.standard_normal((groups, sk, D))
    v = rng.standard_normal((groups, sk, D))
    q = rng.standard_normal((H, n, D)) * 0.5
    iq = rng.standard_normal((groups, n, IDX_D))
    f = lambda a: mx.array(a.astype(np.float32)).astype(dtype)[None]  # noqa: E731
    return f(q), f(k), f(v), f(iq), f(ik[None])


def control(sel, kind: str):
    """A wrong selection derived from ``sel`` (see module docstring)."""
    from vmlx_engine.models.minimax_m3.msa import MSASelection

    top = np.array(sel.top_idx).astype(np.int64)                      # [B, G, Sq, keep]
    nb = sel.n_blocks
    if kind == "shared":
        top = np.broadcast_to(top[:, :1], top.shape).copy()
    elif kind == "off_by_one":
        top = (top + 1) % nb
    elif kind == "no_local":
        Sq = top.shape[2]
        for r in range(Sq):
            local = (sel.offset + r) // sel.block
            for b in range(top.shape[0]):
                for g in range(top.shape[1]):
                    row = top[b, g, r]
                    spare = next(x for x in range(nb) if x not in set(row.tolist()))
                    row[row == local] = spare
    else:
        raise ValueError(kind)
    return MSASelection(mx.array(top.astype(np.uint32)), sel.offset, sel.sk, sel.block)


def _run_path(attn, path, q, k, v, sel, mask=None):
    os.environ["VMLX_MINIMAX_MSA"] = path
    try:
        out = attn.attend(q, k, v, sel, mask, None)
        mx.eval(out)
        return out
    finally:
        os.environ.pop("VMLX_MINIMAX_MSA", None)


def _heads(groups: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    o_proj = mx.array((rng.standard_normal((1024, H * D)) / np.sqrt(H * D)).astype(np.float32))
    head = mx.array((rng.standard_normal((4096, 1024)) / np.sqrt(1024)).astype(np.float32))
    return o_proj, head


def _reference(q, k, v, top_idx, offset: int, scale: float) -> np.ndarray:
    """float64 MSA attention [rows, H * D] from the selection's block ids."""
    Q = np.array(q.astype(mx.float32), dtype=np.float64)[0]
    K = np.array(k.astype(mx.float32), dtype=np.float64)[0]
    V = np.array(v.astype(mx.float32), dtype=np.float64)[0]
    G, rows, sk = K.shape[0], Q.shape[1], K.shape[1]
    hg = H // G
    pos = np.arange(sk)
    top = np.array(top_idx).astype(np.int64)[0]
    out = np.zeros((rows, H, D))
    with np.errstate(all="ignore"):         # Accelerate's matmul raises spurious FP flags on finite inputs
        for r in range(rows):
            for g in range(G):
                allowed = (pos <= offset + r) & np.isin(pos // BLOCK, top[g, r])
                s = Q[g * hg:(g + 1) * hg, r] @ K[g, allowed].T * scale
                pr = np.exp(s - s.max(-1, keepdims=True))
                out[r, g * hg:(g + 1) * hg] = (pr / pr.sum(-1, keepdims=True)) @ V[g, allowed]
    assert np.isfinite(out).all()
    return out.reshape(rows, H * D)


def _rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def _dist(rel):
    return {"min": float(rel.min()), "median": float(np.median(rel)), "p99": float(np.percentile(rel, 99)),
            "max": float(rel.max())}


def _stats(dense, out, ref, o_proj, head):
    a, b = np.stack(dense), np.stack(out)
    la = np.array(mx.array(a.astype(np.float32)) @ o_proj.T @ head.T)
    lb = np.array(mx.array(b.astype(np.float32)) @ o_proj.T @ head.T)
    res = {"n": int(a.shape[0]), "rel": _dist(_rel(b, a)), "top1": float((la.argmax(-1) == lb.argmax(-1)).mean())}
    if ref is not None:
        res["ref"] = _dist(_rel(b, np.stack(ref)))
    return res


def _rows(o):
    a = np.array(o.astype(mx.float32), dtype=np.float64)[0]
    return a.transpose(1, 0, 2).reshape(a.shape[1], -1)


def decode_positions(sk: int, n_random: int, seed: int):
    """Sparse query positions (keys 0..p with p + 1 > 2048): the last rows of a context of length sk plus random."""
    rng = np.random.default_rng(seed)
    pos = {sk - 1, sk - 2, sk - 3}
    if sk - 1 >= TOPK * BLOCK:
        pos.update(int(x) for x in rng.integers(TOPK * BLOCK, sk, n_random))
    return sorted(p for p in pos if p >= TOPK * BLOCK)


def compare_decode(groups: int, dtype, sk: int, paths=("decode", "gather"), n_random: int = 24, seed: int = 0,
                   controls=()):
    attn = _attention(groups)
    q, k, v, iq, ik = tensors(sk, groups, dtype, seed)
    o_proj, head = _heads(groups)
    acc = {}
    for p in decode_positions(sk, n_random, seed):
        sel = attn.indexer.select(iq[:, :, p:p + 1], ik[:, :, :p + 1], p)
        qp, kp, vp = q[:, :, p:p + 1], k[:, :, :p + 1], v[:, :, :p + 1]
        dense = _rows(_run_path(attn, "dense", qp, kp, vp, sel))[0]
        ref = _reference(qp, kp, vp, sel.top_idx, p, attn.scale)[0]
        acc.setdefault("dense", ([], [], []))
        acc["dense"][0].append(dense), acc["dense"][1].append(dense), acc["dense"][2].append(ref)
        for path in paths:
            for name in ("ref",) + tuple(controls):
                s = sel if name == "ref" else control(sel, name)
                o = _rows(_run_path(attn, path, qp, kp, vp, s))[0]
                key = path if name == "ref" else f"{path}:{name}"
                acc.setdefault(key, ([], [], []))
                acc[key][0].append(dense), acc[key][1].append(o), acc[key][2].append(ref)
    if not acc:
        return {"sparse_positions": 0}
    res = {"sparse_positions": len(acc["dense"][0])}
    for key, (d, o, r) in acc.items():
        res[key] = _stats(d, o, r, o_proj, head)
    return res


def compare_prefill(groups: int, dtype, sk: int, rows: int, seed: int = 0, controls=(), with_ref: bool = True):
    attn = _attention(groups)
    q, k, v, iq, ik = tensors(sk, groups, dtype, seed, rows=rows)
    o_proj, head = _heads(groups)
    offset = sk - rows
    sel = attn.indexer.select(iq, ik, offset)
    dense = list(_rows(_run_path(attn, "dense", q, k, v, sel, "causal")))
    ref = list(_reference(q, k, v, sel.top_idx, offset, attn.scale)) if with_ref else None
    res = {"sparse_positions": rows, "dense": _stats(dense, dense, ref, o_proj, head)}
    for name in ("ref",) + tuple(controls):
        s = sel if name == "ref" else control(sel, name)
        o = list(_rows(_run_path(attn, "kernel", q, k, v, s, "causal")))
        res["kernel" if name == "ref" else f"kernel:{name}"] = _stats(dense, o, ref, o_proj, head)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--controls", nargs="*", default=["shared", "off_by_one", "no_local"])
    ap.add_argument("--prefill", type=int, default=0, help="also compare an L-row prefill chunk ending at each Sk")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    report = []
    for dt in a.dtypes:
        dtype = getattr(mx, dt)
        for g in a.groups:
            for sk in a.sizes:
                r = compare_decode(g, dtype, sk, controls=a.controls)
                r.update({"kind": "decode", "dtype": dt, "groups": g, "sk": sk})
                report.append(r)
                _print(r)
                if a.prefill and sk > max(a.prefill, TOPK * BLOCK):
                    r = compare_prefill(g, dtype, sk, a.prefill, controls=a.controls)
                    r.update({"kind": f"prefill{a.prefill}", "dtype": dt, "groups": g, "sk": sk})
                    report.append(r)
                    _print(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


def _print(r):
    head = f"{r['kind']:11s} {r['dtype']:8s} G={r['groups']} Sk={r['sk']:5d} n={r['sparse_positions']:3d}"
    if not r["sparse_positions"]:
        print(head, " (every block visible: plain causal attention on every path)")
        return
    for key, st in r.items():
        if not isinstance(st, dict) or "rel" not in st:
            continue
        line = f"{head}  {key:22s} vs dense: min {st['rel']['min']:.2e} med {st['rel']['median']:.2e} " \
               f"p99 {st['rel']['p99']:.2e} max {st['rel']['max']:.2e} top1 {st['top1']:.3f}"
        if "ref" in st:
            line += f" | vs f64: med {st['ref']['median']:.2e} max {st['ref']['max']:.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
