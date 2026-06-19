# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 EAGLE3 speculative decode loop (verify/accept/rollback).

Greedy spec-decode: draft K tokens from the target's EAGLE3 aux taps, verify the
whole chain in ONE target forward, accept the longest greedy-matching prefix plus
the target's bonus token, roll the MSA dual cache back to the accepted length.

Correctness invariant (gated by verify_greedy_equivalence): the emitted sequence is
token-for-token identical to plain greedy decoding — every emitted token is either a
draft that matched the target's own greedy choice, or the target's greedy correction.

Created by Jinho Jang (eric@jangq.ai).
"""
from __future__ import annotations

import mlx.core as mx

try:
    from .eagle3_draft import load_eagle3_draft
    from .cache import truncate_minimax_m3_cache
except Exception:  # standalone
    from eagle3_draft import load_eagle3_draft
    from cache import truncate_minimax_m3_cache


def _aux_at(aux, idx):
    """concat(aux@2, aux@30, aux@57) at sequence position idx -> [1,1,naux*H]."""
    if idx < 0:                                   # normalize: a[:, -1:0, :] is an EMPTY slice
        idx += aux[0].shape[1]
    return mx.concatenate([a[:, idx:idx + 1, :] for a in aux], axis=-1)


def _encode(tok, prompt, use_chat):
    if use_chat and getattr(tok, "chat_template", None):
        s = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        return tok.encode(s, add_special_tokens=False)
    return tok.encode(prompt)


def generate_greedy(model, ids, max_tokens, eos_ids):
    """Plain greedy reference (one target forward per token)."""
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)
    out = [int(mx.argmax(logits[0, -1]))]
    while len(out) < max_tokens and out[-1] not in eos_ids:
        logits = model(mx.array([[out[-1]]]), cache=cache)
        out.append(int(mx.argmax(logits[0, -1])))
    return out


def generate_eagle3(model, draft, tok, eos_ids, prompt, max_tokens=128, K=4,
                    use_chat=True, ids=None):
    """EAGLE3 greedy spec-decode. Returns (tokens, n_target_forwards)."""
    if ids is None:
        ids = _encode(tok, prompt, use_chat)
    embed_fn = lambda t: model.model.embed_tokens(mx.array([[int(t)]]))

    cache = model.make_cache()
    logits, aux = model(mx.array([ids]), cache=cache, return_aux=True)
    mx.eval(logits, *aux)
    p = cache[0].offset                       # absolute cache length (= len(ids))
    t = int(mx.argmax(logits[0, -1]))         # first verified token (greedy token 0)
    aux_p = _aux_at(aux, -1)
    out = [t]
    n_forward = 1

    while len(out) < max_tokens and t not in eos_ids:
        drafts = draft.propose(aux_p, t, embed_fn, K)        # [d0..d_{K-1}]
        verify_in = [t] + drafts                              # K+1 tokens at pos p..p+K
        vlogits, vaux = model(mx.array([verify_in]), cache=cache, return_aux=True)
        mx.eval(vlogits, *vaux)
        n_forward += 1

        c = [int(mx.argmax(vlogits[0, i])) for i in range(K + 1)]  # target greedy at each fed pos
        a = 0
        for i in range(K):
            if drafts[i] == c[i]:
                a += 1
            else:
                break
        new_tokens = drafts[:a] + [c[a]]                     # accepted drafts + correction/bonus
        out.extend(new_tokens)

        truncate_minimax_m3_cache(cache, p + 1 + a)          # keep t + accepted drafts
        t = c[a]
        aux_p = _aux_at(vaux, a)                              # hidden that predicted t
        p = p + 1 + a

        for j, tok_id in enumerate(new_tokens):              # honor EOS inside the chain
            if tok_id in eos_ids:
                out = out[: len(out) - len(new_tokens) + j + 1]
                return out[:max_tokens], n_forward
    return out[:max_tokens], n_forward


def verify_greedy_equivalence(model, draft, tok, eos_ids, prompt, n=48, K=4, use_chat=True):
    """Step 8 gate: spec-decode output must equal plain greedy, token-for-token."""
    ids = _encode(tok, prompt, use_chat)
    g = generate_greedy(model, ids, n, eos_ids)
    s, nf = generate_eagle3(model, draft, tok, eos_ids, prompt, max_tokens=n, K=K,
                            use_chat=use_chat, ids=ids)
    m = min(len(g), len(s))
    ok = g[:m] == s[:m]
    first_div = next((i for i in range(m) if g[i] != s[i]), -1)
    return ok, first_div, g[:m], s[:m], nf
