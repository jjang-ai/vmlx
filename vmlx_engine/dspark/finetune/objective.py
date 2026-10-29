"""The DSpark block-drafting objective, as a batched training forward over many anchors of one sequence.

The serving drafter (``drafter.DSparkDrafter``) drafts one block per round:

- context: target tap rows of every position before the anchor -> ``fc`` -> ``hidden_norm``; each draft layer projects
  it with its own ``k_proj`` / ``v_proj`` (+ ``k_norm``, RoPE at the context positions);
- block: ``[anchor, mask x (S - 1)]`` embedded by the target, positions ``p .. p + S - 1`` (anchor at p);
- each layer: queries = the block rows; keys = context + the whole block (non-causal); sliding window: the row at
  position q reads context positions > q - window;
- logits of row i = target head(norm(h_i)) + Markov bias ``markov_w2 @ markov_w1[prev_i]`` (prev_0 = anchor,
  prev_i = the token drafted at i - 1); row i predicts position p + i + 1;
- confidence of row i = sigmoid(proj([h_i ; markov_w1[prev_i]])).

The objective trains exactly that function. A draft is accepted only while every earlier draft was, so with
teacher-forced previous tokens (prev_i = the target's token at p + i) row i's logits are the logits the serving chain
computes whenever row i matters. Loss = per-row cross-entropy against the target's token at p + i + 1 (rows whose
label is not a target-generated position are masked; optional per-position decay), plus a BCE term training the
confidence head to predict "row i's argmax is the target's token" (its runtime meaning: P(draft i accepted | the drafts
before it were)), with stopped gradients into the drafter body. MiniMax's original training recipe is not in the
release; this objective is derived from the runtime forward (documented in the campaign's 08-DSPARK-FINETUNE.md).

Many anchors run in ONE forward: the context K/V are computed once for the rows the anchors need, the M blocks are
laid out as M * S query rows, and a boolean mask gives each row its own visible context window plus its own block.
Weights are fp32 master copies; compute is bf16 like the runtime; linears the release stores as MXFP8 can be
fake-quantized (straight-through) so the exported MXFP8 drafter is the one that was trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


@dataclass(frozen=True)
class Spec:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int
    rope_theta: float
    eps: float
    window: int
    block: int
    mask_token_id: int
    taps: tuple
    vocab: int
    rank: int
    alpha: float
    limit: float

    @classmethod
    def from_drafter(cls, d) -> "Spec":
        cfg = d.cfg
        return cls(hidden=int(cfg["hidden_size"]), layers=len(d.layers), heads=int(cfg["num_attention_heads"]),
                   kv_heads=int(cfg["num_key_value_heads"]), head_dim=int(d.head_dim), rope_dims=int(d.rope_dims),
                   rope_theta=float(d.rope_theta), eps=float(cfg["rms_norm_eps"]), window=int(d.sliding_window),
                   block=int(d.block), mask_token_id=int(d.mask_token_id), taps=tuple(d.target_layer_ids),
                   vocab=int(cfg["vocab_size"]), rank=int(d.markov_head.markov_w1.weight.shape[1]),
                   alpha=float(cfg.get("swiglu_alpha", 1.702)), limit=float(cfg.get("swiglu_limit", 7.0)))


# ----------------------------------------------------------------------------------------------- parameters
def drafter_params(drafter) -> tuple[dict, set]:
    """Every drafter tensor as a float array under its live module path (quantized modules dequantized exactly, BF16),
    and the set of linear module paths stored as MXFP8."""
    from mlx.utils import tree_flatten

    from ..drafter import _float_weight

    params, mxfp8 = {}, set()
    for path, m in drafter.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            params[path + ".weight"] = _float_weight(m)
            if str(getattr(m, "mode", "affine")) == "mxfp8":
                mxfp8.add(path)
        elif isinstance(m, (nn.Linear, nn.Embedding)):
            params[path + ".weight"] = m.weight
        else:
            continue
        if "bias" in m:
            params[path + ".bias"] = m["bias"]
    for k, v in tree_flatten(drafter.parameters()):
        if k.endswith("norm.weight"):
            params.setdefault(k, v)
    mx.eval(params)
    return params, mxfp8


def fake_mxfp8(w: mx.array) -> mx.array:
    """MXFP8 round trip in the forward, identity in the backward (straight-through)."""
    q, s = mx.quantize(w, **MXFP8)[:2]
    return w + mx.stop_gradient(mx.dequantize(q, s, None, dtype=w.dtype, **MXFP8) - w)


def _lin(p: dict, path: str, x: mx.array, fq) -> mx.array:
    w = p[path + ".weight"]
    if path in fq:
        w = fake_mxfp8(w)
    y = x @ w.astype(x.dtype).T
    b = p.get(path + ".bias")
    return y if b is None else y + b.astype(y.dtype)


def _norm(p: dict, path: str, x: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, 1.0 + p[path + ".weight"].astype(x.dtype), eps)


def _swiglu(x_linear, x_glu, alpha: float, limit: float):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + 1)


def _rope(s: Spec, x: mx.array, offset) -> mx.array:
    return mx.fast.rope(x, s.rope_dims, traditional=False, base=s.rope_theta, scale=1.0, offset=offset)


# ----------------------------------------------------------------------------------------------- forward
def context_kv(p: dict, s: Spec, feats: mx.array, ctx_start: int, fq=frozenset()) -> list:
    """Per draft layer (K, V) of the context rows ``feats`` (1, C, n_taps * H) at positions ctx_start.."""
    count = int(feats.shape[1])
    if count == 0:
        return [None] * s.layers
    ctx = _norm(p, "hidden_norm", _lin(p, "fc", feats, fq), s.eps)
    out = []
    for li in range(s.layers):
        pre = f"layers.{li}.self_attn."
        k = _lin(p, pre + "k_proj", ctx, fq).reshape(1, count, s.kv_heads, s.head_dim)
        k = _rope(s, _norm(p, pre + "k_norm", k, s.eps).transpose(0, 2, 1, 3), ctx_start)
        v = _lin(p, pre + "v_proj", ctx, fq).reshape(1, count, s.kv_heads, s.head_dim).transpose(0, 2, 1, 3)
        out.append((k, v))
    return out


def attention_mask(anchors: np.ndarray, span: int, ctx_start: int, ctx_len: int, window: int) -> mx.array:
    """(M * S, C + M * S) bool: block row (m, i) at position q = anchors[m] + i sees context position j iff
    j < anchors[m] and (no window or j > q - window), and every row of its own block."""
    anchors = np.asarray(anchors, np.int64)
    m = len(anchors)
    qpos = (anchors[:, None] + np.arange(span)[None]).reshape(-1)
    owner = np.repeat(np.arange(m), span)
    cpos = ctx_start + np.arange(ctx_len)
    ctx_ok = cpos[None, :] < np.repeat(anchors, span)[:, None]
    if window:
        ctx_ok &= cpos[None, :] > (qpos[:, None] - window)
    blk_ok = owner[:, None] == owner[None, :]
    return mx.array(np.concatenate([ctx_ok, blk_ok], axis=1))


def block_hidden(p: dict, s: Spec, embed, kv: list, anchors: mx.array, block_tokens: mx.array, mask: mx.array,
                 fq=frozenset()) -> mx.array:
    """Drafter output (M, S, H) (after the final norm) for M blocks ``block_tokens`` (M, S) at ``anchors`` (M,)."""
    m, span = int(block_tokens.shape[0]), int(block_tokens.shape[1])
    rows = m * span
    h = embed(block_tokens)
    H, KV, D = s.heads, s.kv_heads, s.head_dim
    for li in range(s.layers):
        pre = f"layers.{li}."
        x = _norm(p, pre + "input_layernorm", h, s.eps)
        a = pre + "self_attn."
        q = _norm(p, a + "q_norm", _lin(p, a + "q_proj", x, fq).reshape(m, span, H, D), s.eps).transpose(0, 2, 1, 3)
        k = _norm(p, a + "k_norm", _lin(p, a + "k_proj", x, fq).reshape(m, span, KV, D), s.eps).transpose(0, 2, 1, 3)
        v = _lin(p, a + "v_proj", x, fq).reshape(m, span, KV, D).transpose(0, 2, 1, 3)
        q, k = _rope(s, q, anchors), _rope(s, k, anchors)
        qf = q.transpose(1, 0, 2, 3).reshape(1, H, rows, D)
        kf = k.transpose(1, 0, 2, 3).reshape(1, KV, rows, D)
        vf = v.transpose(1, 0, 2, 3).reshape(1, KV, rows, D)
        if kv[li] is not None:
            kc, vc = kv[li]
            kf, vf = mx.concatenate([kc, kf], axis=2), mx.concatenate([vc, vf], axis=2)
        o = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=D ** -0.5, mask=mask)
        o = o.reshape(H, m, span, D).transpose(1, 2, 0, 3).reshape(m, span, H * D)
        h = h + _lin(p, a + "o_proj", o, fq)
        y = _norm(p, pre + "post_attention_layernorm", h, s.eps)
        h = h + _lin(p, pre + "mlp.down_proj",
                     _swiglu(_lin(p, pre + "mlp.up_proj", y, fq), _lin(p, pre + "mlp.gate_proj", y, fq), s.alpha,
                             s.limit), fq)
    return _norm(p, "norm", h, s.eps)


def draft_logits(p: dict, head, hidden: mx.array, prev: mx.array):
    """(base logits (M, S, V) fp32 = target head, Markov bias (M, S, V) fp32, Markov latents (M, S, r) bf16). The
    drafter's logits are base + bias."""
    base = head(hidden).astype(mx.float32)
    lat = mx.take(p["markov_head.markov_w1.weight"], prev, axis=0).astype(hidden.dtype)
    bias = (lat @ p["markov_head.markov_w2.weight"].astype(hidden.dtype).T).astype(mx.float32)
    return base, bias, lat


def default_candidates() -> int:
    """The runtime's Markov candidate count (``VMLX_DSPARK_MARKOV_CANDIDATES``, default 32; 0 = full vocabulary)."""
    from ..drafter import _markov_candidates

    return _markov_candidates()


def pick_tokens(base: mx.array, bias: mx.array, candidates: int) -> mx.array:
    """The token the serving chain drafts at each row: argmax of base + bias over the top-``candidates`` base logits
    (``DSparkDrafter._draft_candidates``), or over the whole vocabulary (``candidates`` 0 / >= V)."""
    k = int(candidates)
    if 0 < k < int(base.shape[-1]):
        cand = mx.argpartition(-base, kth=k - 1, axis=-1)[..., :k]
        score = mx.take_along_axis(base, cand, axis=-1) + mx.take_along_axis(bias, cand, axis=-1)
        return mx.take_along_axis(cand, mx.argmax(score, axis=-1)[..., None], axis=-1)[..., 0]
    return mx.argmax(base + bias, axis=-1)


def confidence_logits(p: dict, hidden: mx.array, lat: mx.array) -> mx.array:
    """Pre-sigmoid confidence (M, S) on stopped features (the head calibrates; it does not steer the drafter)."""
    feat = mx.concatenate([mx.stop_gradient(hidden).astype(mx.float32), mx.stop_gradient(lat).astype(mx.float32)], -1)
    w = p["confidence_head.proj.weight"].astype(mx.float32)
    b = p["confidence_head.proj.bias"].astype(mx.float32)
    return (feat @ w.T)[..., 0] + b[0]


# ----------------------------------------------------------------------------------------------- batches
@dataclass
class Batch:
    feats: mx.array            # (1, C, n_taps * H) bf16
    ctx_start: int
    anchors: mx.array          # (M,) int32
    block_tokens: mx.array     # (M, S)
    prev: mx.array             # (M, S)
    labels: mx.array           # (M, S), 0 where invalid
    weights: mx.array          # (M, S) fp32, 0 where invalid
    valid: mx.array            # (M, S) bool
    mask: mx.array             # (M * S, C + M * S) bool
    anchors_np: np.ndarray
    rec_id: str = ""
    soft_ids: Optional[mx.array] = None     # (M, S, K)
    soft_lp: Optional[mx.array] = None


def make_batch(ds, rec, anchors: np.ndarray, span: int, s: Spec, *, on_path: bool = True,
               pos_decay: float = 0.0, soft: bool = False) -> Batch:
    """Anchors (absolute positions, sorted) of record ``rec`` -> one batch. Row i of anchor p is trained iff positions
    p + 1 .. p + i + 1 were all produced by the target (an assistant span / the continuation) and, with ``on_path``,
    the sequence's tokens at p + 1 .. p + i equal the target's labels there (the prefix row i conditions on is the
    target's own greedy path; always true for greedy generations)."""
    anchors = np.asarray(anchors, np.int64)
    n = rec.n
    tokens, labels, gen = ds.tokens(rec), ds.labels(rec), ds.gen(rec)
    idx = anchors[:, None] + np.arange(1, span + 1)[None]           # label position of row i: p + i + 1
    inb = idx < n
    idc = np.minimum(idx, n - 1)
    lab = labels[idc].astype(np.int64)
    valid = np.logical_and.accumulate(gen[idc].astype(bool) & inb, axis=1)
    if on_path:
        agree = (tokens[idc] == labels[idc]) & inb
        before = np.ones_like(agree)
        before[:, 1:] = np.logical_and.accumulate(agree[:, :-1], axis=1)
        valid &= before
    prev = np.empty_like(lab)
    prev[:, 0] = tokens[anchors]
    prev[:, 1:] = lab[:, :-1]
    block = np.full_like(lab, s.mask_token_id)
    block[:, 0] = tokens[anchors]
    decay = np.exp(-np.arange(span) / pos_decay) if pos_decay and pos_decay > 0 else np.ones(span)
    weights = valid * decay[None, :]
    cs = rec.tap_start
    if s.window:
        cs = max(cs, int(anchors.min()) - s.window + 1)
    ce = max(cs, int(anchors.max()))
    feats = np.asarray(ds.taps(rec, cs, ce))
    feats = mx.view(mx.array(feats), mx.bfloat16)[None] if len(feats) else mx.zeros((1, 0, ds.width), mx.bfloat16)
    out = Batch(feats=feats, ctx_start=cs, anchors=mx.array(anchors.astype(np.int32)),
                block_tokens=mx.array(block.astype(np.int32)), prev=mx.array(prev.astype(np.int32)),
                labels=mx.array(np.where(valid, lab, 0).astype(np.int32)),
                weights=mx.array(weights.astype(np.float32)), valid=mx.array(valid),
                mask=attention_mask(anchors, span, cs, ce - cs, s.window), anchors_np=anchors, rec_id=rec.id)
    if soft and ds.topk:
        ids = ds.per_position(rec, "topk_ids")[np.maximum(idc - 1, 0)]          # row j - 1 predicts position j
        lp = ds.per_position(rec, "topk_logprobs")[np.maximum(idc - 1, 0)]
        out.soft_ids, out.soft_lp = mx.array(ids.astype(np.int32)), mx.array(lp.astype(np.float32))
    return out


# ----------------------------------------------------------------------------------------------- loss
def forward(p: dict, s: Spec, head, b: Batch, fq=frozenset()):
    """(hidden (M, S, H), base logits, Markov bias, Markov latents)."""
    kv = context_kv(p, s, b.feats, b.ctx_start, fq)
    hidden = block_hidden(p, s, head.embed, kv, b.anchors, b.block_tokens, b.mask, fq)
    base, bias, lat = draft_logits(p, head.head, hidden, b.prev)
    return hidden, base, bias, lat


def _ce(logits: mx.array, labels: mx.array) -> tuple:
    lse = mx.logsumexp(logits, axis=-1)
    return lse - mx.take_along_axis(logits, labels[..., None], axis=-1)[..., 0], lse


def loss_fn(train: dict, frozen: dict, s: Spec, head, b: Batch, fq=frozenset(), *, conf_weight: float = 1.0,
            soft_weight: float = 0.0, base_weight: float = 0.0, candidates: int = 32):
    """(total loss, aux dict). ``train`` holds the trainable tensors (the differentiated argument).

    ``base_weight``: an extra cross-entropy on the base logits alone (the target head, no Markov bias). The runtime
    rescores only the top-``candidates`` base logits with the Markov bias, so a label the bias alone would promote is
    unreachable at serve time; this term keeps the label high in the base logits. ``correct`` (and the confidence
    targets) use the runtime's candidate pick."""
    p = {**frozen, **train}
    hidden, base, bias, lat = forward(p, s, head, b, fq)
    logits = base + bias
    ce, lse = _ce(logits, b.labels)
    wsum = mx.maximum(b.weights.sum(), 1e-6)
    loss = (ce * b.weights).sum() / wsum
    if soft_weight and b.soft_ids is not None:
        q = mx.softmax(b.soft_lp, axis=-1)                                  # renormalized target top-K
        soft = -(q * (mx.take_along_axis(logits, b.soft_ids, axis=-1) - lse[..., None])).sum(-1)
        loss = (1 - soft_weight) * loss + soft_weight * (soft * b.weights).sum() / wsum
    if base_weight:
        loss = loss + base_weight * (_ce(base, b.labels)[0] * b.weights).sum() / wsum
    correct = mx.stop_gradient((pick_tokens(base, bias, candidates) == b.labels) & b.valid)
    z = confidence_logits(p, hidden, lat)
    y = correct.astype(mx.float32)
    vf = b.valid.astype(mx.float32)
    bce = ((nn.losses.binary_cross_entropy(z, y, with_logits=True, reduction="none")) * vf).sum() / mx.maximum(
        vf.sum(), 1.0)
    total = loss + conf_weight * bce
    return total, {"ce": loss, "bce": bce, "ce_rows": mx.stop_gradient(ce), "correct": correct}


def accepted_lengths(correct: mx.array, valid: mx.array) -> np.ndarray:
    """Per anchor: how many leading block rows are correct (= the drafts a greedy verify accepts at width >= that)."""
    ok = np.asarray(correct) & np.asarray(valid)
    return np.logical_and.accumulate(ok, axis=1).sum(1)
