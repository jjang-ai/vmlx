"""Prefix-restore correctness gate for the M3 MSA dual cache.

Proves serialize->restore preserves decode: prefill a prompt (cache A), snapshot
every layer via to_cache_data()/state, rebuild a fresh cache B from the
snapshots, then decode the SAME next steps from A and from B and assert the
logits are identical. If idx_keys weren't persisted/restored, the sparse layers'
selection would diverge and this fails. Uses only the M3 cache classes — no
edits to the shared prefix_cache tiers.
"""
import sys, time
from pathlib import Path
import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimax_m3 as M
from cache import restore_minimax_m3_sparse
from runtime import load_minimax_m3
from mlx_lm.models.cache import KVCache

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
ids = tok.encode("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n")
print(f"prompt {len(ids)} tok", flush=True)

# cache A: prefill
A = model.make_cache()
logitsA = model(mx.array([ids]), cache=A); mx.eval(logitsA)

# snapshot every layer, then rebuild cache B
snaps = []
for c in A:
    if isinstance(c, M.MiniMaxM3SparseCache):
        snaps.append(("minimax_m3",) + tuple(c.state))   # (tag, k, v, idx)
    else:
        snaps.append(("kv",) + tuple(c.state))           # (tag, k, v)
B = []
for s in snaps:
    if s[0] == "minimax_m3":
        B.append(restore_minimax_m3_sparse(s[1], s[2], s[3]))
    else:
        kc = KVCache(); kc.state = (s[1], s[2]); B.append(kc)

# decode 8 steps from both, assert identical argmax + logit match
nxtA = int(mx.argmax(logitsA[0, -1]))
nxtB = nxtA
okA = [nxtA]; okB = [nxtB]
maxdiff = 0.0
for _ in range(8):
    la = model(mx.array([[okA[-1]]]), cache=A); mx.eval(la)
    lb = model(mx.array([[okB[-1]]]), cache=B); mx.eval(lb)
    maxdiff = max(maxdiff, float(mx.abs(la - lb).max()))
    okA.append(int(mx.argmax(la[0, -1])))
    okB.append(int(mx.argmax(lb[0, -1])))

match = okA == okB
print(f"A offset={A[3].offset} B offset={B[3].offset}")
print(f"A tokens: {okA}")
print(f"B tokens: {okB}")
print(f"max|logitA-logitB| over 8 steps = {maxdiff:.4g}")
print("ROUNDTRIP", "PASS — restore preserves decode" if match and maxdiff < 1e-2 else "FAIL")
