"""Fast (no-model) micro-test: drive the prefix-cache tier on synthetic M3
states so we can see exactly which tag each stage emits."""
import sys, tempfile
from pathlib import Path
import mlx.core as mx
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vmlx_engine.prefix_cache import BlockAwarePrefixCache, _is_minimax_m3_cache_class
from vmlx_engine.paged_cache import PagedCacheManager

print("_is_minimax_m3_cache_class('MiniMaxM3SparseCache') =",
      _is_minimax_m3_cache_class("MiniMaxM3SparseCache"))

B, H, S, D, Hi = 1, 4, 130, 128, 1
NLAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
mx.random.seed(1)
# fake model cache: layers 0-2 dense KVCache, rest M3 sparse
cache_data = []
for i in range(NLAYERS):
    k = mx.random.normal((B, H, S, D)).astype(mx.float16)
    v = mx.random.normal((B, H, S, D)).astype(mx.float16)
    if i < 3:
        cache_data.append({"state": (k, v), "meta_state": (str(S),),
                           "class_name": "KVCache"})
    else:
        idx = mx.random.normal((B, Hi, S, D)).astype(mx.float16)
        cache_data.append({"state": (k, v, idx), "meta_state": (str(S),),
                           "class_name": "MiniMaxM3SparseCache"})
mx.eval(*[t for d in cache_data for t in d["state"]])


class FakeModel:
    pass


pcm = PagedCacheManager(block_size=64, max_blocks=512, disk_store=None)
bac = BlockAwarePrefixCache(model=FakeModel(), paged_cache_manager=pcm)
if "--force-heads" in sys.argv:
    bac._allowed_n_kv_heads = {4}
try:
    print("computed allowed_n_kv_heads =", sorted(bac._get_allowed_n_kv_heads()))
except Exception as e:
    print("allowed_n_kv_heads raised:", e)

# directly probe _extract_block_tensor_slice for block [0:64]
slices = bac._extract_block_tensor_slice(cache_data, 0, 64, is_last_block=False,
                                         np_sources=None)
print("block[0:64] tags:", [s[0] for s in slices])

tokens = list(range(S))
bt = bac.store_cache("r", tokens, cache_data)
print("stored blocks:", len(bt.block_ids), "num_tokens:", bt.num_tokens)
# inspect the stored block cache_data tags
for bid in bt.block_ids:
    blk = pcm.allocated_blocks.get(bid)
    cd = blk.cache_data
    print(f"  block {bid}: cache_data tags =",
          [e[0] for e in cd] if cd else None)

rebuilt = bac.reconstruct_cache(bt)
print("rebuilt classes:", [type(c).__name__ for c in rebuilt] if rebuilt else None)
