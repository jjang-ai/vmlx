"""Regression gate: confirm the additive minimax_m3 edits did NOT disturb the
existing cache tags that other models (DSV4, ZAYA, Gemma4, Mamba-hybrids,
quantized-KV) rely on. Exercises the L2 codec + the record validator on a
synthetic block mixing every tag type."""
import sys
from pathlib import Path
import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vmlx_engine.block_disk_store import _serialize_block, _deserialize_block
from vmlx_engine.cache_record_validator import validate_cache_record

B, H, S, D = 1, 4, 40, 128
mx.random.seed(3)


def k(): return mx.random.normal((B, H, S, D)).astype(mx.float16)
def qtuple():
    # (data, scales, zeros) quantized KV components
    data = mx.zeros((B, H, S, D // 32), dtype=mx.uint32)
    sc = mx.ones((B, H, S, D // 32), dtype=mx.float16)
    ze = mx.zeros((B, H, S, D // 32), dtype=mx.float16)
    return (data, sc, ze)


# one block, mixing every supported tag
cache_data = [
    ("kv", k(), k()),                                          # standard KV
    ("quantized_kv", qtuple(), qtuple(), ("0", "32", "4")),    # QuantizedKVCache
    ("rotating_kv", k(), k(), 4096, 0, S, S),                  # RotatingKVCache
    ("cumulative", [k(), k()], (str(S),), "MambaCache"),       # SSM cumulative
    ("minimax_m3", k(), k(), mx.random.normal((B, 1, S, D)).astype(mx.float16)),
    ("skip",),                                                  # placeholder
]
mx.eval(*[t for e in cache_data for t in e[1:]
          if hasattr(t, "shape")])

print("=== L2 codec round-trip on mixed tags ===")
tensors, dtype, n = _serialize_block(cache_data)
# normalize numpy/bf16 like the real writer, then save/load via safetensors
import tempfile, os
with tempfile.TemporaryDirectory() as td:
    fp = os.path.join(td, "blk.safetensors")
    mx.save_safetensors(fp, {kk: (vv if isinstance(vv, mx.array) else mx.array(vv))
                             for kk, vv in tensors.items()})
    loaded = mx.load(fp)
restored = _deserialize_block(loaded, dtype)
got_tags = [e[0] for e in restored]
print("  in :", [e[0] for e in cache_data])
print("  out:", got_tags)
# skip layers come back as ("skip",); the rest must keep their tag
expect = ["kv", "quantized_kv", "rotating_kv", "cumulative", "minimax_m3", "skip"]
assert got_tags == expect, (got_tags, expect)
print("  all tags preserved through L2 codec")

print("\n=== validator accepts every tag (incl. minimax_m3) ===")
ok, reason, nbytes = validate_cache_record(cache_data, source="regression")
print(f"  validate_cache_record -> ok={ok} bytes={nbytes} reason={reason!r}")
assert ok, f"validator rejected mixed block: {reason}"

# unknown tag must still be rejected (validator didn't go permissive)
bad = [("totally_unknown_tag", k(), k())]
ok2, reason2, _ = validate_cache_record(bad, source="regression-neg")
print(f"  unknown tag rejected? ok={ok2} reason={reason2!r}")
assert not ok2 and "unknown tag" in reason2

print("\nREGRESSION PASS — existing tags intact, minimax_m3 added, "
      "unknown tags still rejected")
