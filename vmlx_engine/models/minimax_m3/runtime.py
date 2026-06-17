# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 runtime: bundle loader + autoregressive decode loop.

Loads the JANG_2L bundle (quantizing only the modules the bundle actually
packed; indexer/norms stay fp16) and runs token-by-token decoding through the
MSA dual cache, with greedy / temperature / top-p sampling, repetition penalty,
EOS stopping, and streaming.

  python -m vmlx_engine.models.minimax_m3.runtime <bundle> \
      --prompt "Write a palindrome checker in Python" --max-tokens 200 --temp 0.0

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    from . import minimax_m3 as M
except Exception:  # pragma: no cover - standalone
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import minimax_m3 as M  # type: ignore


# ── load ───────────────────────────────────────────────────────────
def load_minimax_m3(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    args = M.ModelArgs.from_dict(cfg)
    model = M.Model(args)

    weights = {}
    # Exclude EAGLE3 (or any) sidecar weights: they load standalone into the draft
    # module and must NOT merge into the main model weight_map (names collide:
    # embed_tokens/lm_head/norm). See eagle3_config.json runtime_notes.
    _sidecars = set()
    _e3 = path / "eagle3_config.json"
    if _e3.is_file():
        try:
            _wf = json.loads(_e3.read_text()).get("weights_file")
            if _wf:
                _sidecars.add(_wf)
        except Exception:
            pass
    for sf in glob.glob(str(path / "*.safetensors")):
        _name = Path(sf).name
        if _name in _sidecars or _name.startswith("eagle3"):
            continue
        weights.update(mx.load(sf))
    weights = model.sanitize(weights)

    def _remap(k):
        if k.startswith("language_model.model."):
            k = "model." + k[len("language_model.model."):]
        elif k.startswith("language_model.lm_head"):
            k = "lm_head" + k[len("language_model.lm_head"):]
        return k.replace(".block_sparse_moe.", ".mlp.").replace(
            ".self_attn.index_", ".self_attn.indexer.index_")

    qcfg = cfg.get("quantization", {})
    qover = {_remap(k): v for k, v in qcfg.items() if isinstance(v, dict)}
    qdef = {"group_size": qcfg.get("group_size", 64), "bits": qcfg.get("bits", 8)}

    def predicate(p, module):
        if f"{p}.scales" not in weights:
            return False                       # indexer / norms stay unquantized
        o = qover.get(p, qdef)
        return {"group_size": o["group_size"], "bits": o["bits"]}

    nn.quantize(model, class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    eos = cfg.get("text_config", cfg).get("eos_token_id", tok.eos_token_id)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    return model, tok, eos_ids


# ── sampling ─────────────────────────────────────────────────────────
def _sample(logits, temp, top_p, prev_ids, rep_penalty):
    logits = logits.astype(mx.float32)
    if rep_penalty and rep_penalty != 1.0 and prev_ids:
        idx = mx.array(list(set(prev_ids)))
        vals = logits[idx]
        vals = mx.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
        logits[idx] = vals
    if temp == 0.0:
        return int(mx.argmax(logits))
    logits = logits / temp
    if top_p and top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs)
        sp = probs[order]
        csum = mx.cumsum(sp)
        mask = csum - sp > top_p          # keep tokens until cumulative > top_p
        sp = mx.where(mask, 0.0, sp)
        sp = sp / sp.sum()
        choice = mx.random.categorical(mx.log(sp + 1e-12))
        return int(order[choice])
    return int(mx.random.categorical(logits))


# ── decode loop ──────────────────────────────────────────────────────
def generate(model, tok, eos_ids, prompt, max_tokens=200, temp=0.0, top_p=1.0,
             rep_penalty=1.0, use_chat=True, stream=True):
    if use_chat and getattr(tok, "chat_template", None):
        templated = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)
        ids = tok.encode(templated, add_special_tokens=False)
    else:
        ids = tok.encode(prompt)

    cache = model.make_cache()
    t0 = time.time()
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    prefill_dt = time.time() - t0

    out, text = [], ""
    detok = []
    t1 = time.time()
    pending_logits = False
    for _ in range(max_tokens):
        prev_ids = ids + out if rep_penalty and rep_penalty != 1.0 else None
        nxt = _sample(logits[0, -1], temp, top_p, prev_ids, rep_penalty)
        if nxt in eos_ids:
            break
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)
        # Decode is strictly token-dependent, but streaming/detok can overlap
        # the next GPU command once the sampled token is known.
        mx.async_eval(logits)
        pending_logits = True
        if stream:
            detok.append(nxt)
            piece = tok.decode(detok)
            if not piece.endswith("�"):   # flush only complete utf-8
                print(piece, end="", flush=True)
                text += piece
                detok = []
    if pending_logits:
        mx.eval(logits)
    decode_dt = time.time() - t1
    if not stream:
        text = tok.decode(out)
    elif detok:
        text += tok.decode(detok)
    n = len(out)
    print(flush=True)
    print(f"\n[{n} tok | prefill {len(ids)}tok/{prefill_dt:.1f}s | "
          f"decode {n/max(decode_dt,1e-9):.1f} tok/s | "
          f"cache offset {cache[3].offset}]", flush=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--prompt", default="Write a Python function that checks if a string is a palindrome.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-chat", action="store_true")
    args = ap.parse_args()
    t = time.time()
    model, tok, eos = load_minimax_m3(args.bundle)
    print(f"loaded in {time.time()-t:.0f}s  eos={eos}\n", flush=True)
    generate(model, tok, eos, args.prompt, args.max_tokens, args.temp, args.top_p,
             args.rep_penalty, use_chat=not args.no_chat)


if __name__ == "__main__":
    main()
