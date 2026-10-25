# DeepSeek-V4.1-Flash — engine integration

Engine-side contract for DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, family
`deepseek_v4_1`). The runtime is `jang_tools.dsv41` (jangq). Its `RUNTIME_SERVING.md`
covers Engram I/O, the prefix-cache contract and measured results, and
`DECODE_DSPARK_METHODS.md` covers the DSpark decode methods.

## Loading and generator selection

- `vmlx_engine/loaders/load_jangtq_dsv41.py` loads the bundle through
  `jang_tools.dsv4_1.mlx_model.load_deepseek_v41` and attaches the bundle's DSpark draft
  as `model._dspark` (default on; `VMLX_DSV41_DSPARK=0` serves AR only).
- With a draft and `max_num_seqs == 1`, the scheduler uses `DSV41DSparkGenerator`
  (`vmlx_engine/utils/dsv41_dspark_generator.py`), a `SingleBatchGenerator` subclass.
  DSpark runs only for greedy sampling without logits processors or logprobs; other
  requests take the inherited AR path. Anything that checks the generator type must use
  `isinstance`, not a class-name comparison.
- The CLI policy for `deepseek_v4_1` serves one sequence, with no JIT, smelt or flash-MoE,
  and a 900 s request timeout.

## Prefill

- Chunked prefill loops pass the next chunk to the model as `engram_lookahead`
  (`SingleBatchGenerator._model_call(..., lookahead=)`, gated on
  `model_type == "deepseek_v41"`). The runtime reads that chunk's Engram rows from SSD
  while the current chunk computes.
- `mlx_memory.model_cache_limit_floor_bytes` gives `deepseek_v41` a 16 GB allocator-cache
  floor. The generic 4 GB bound made a 2048-token chunk re-create Metal buffers (21k-token
  cold prompt: 475 → 769 pp/s). `VMLX_MLX_CACHE_LIMIT_MB` overrides exactly.
- `DSV41DSparkGenerator` caps the prefill chunk at 1024 tokens. With the 8.3 GB draft
  resident, a 2048-token chunk's working set exceeds Metal's recommended working set and
  the GPU slows down (2048: 545 pp/s, 1024: 708). `VMLX_DSV41_DSPARK_PREFILL_STEP`
  overrides.

## Prefix cache

- Prompt-boundary snapshots are taken at N-1 prompt tokens, before the last token, and
  reconstruction rejects a restored offset that disagrees with the block table.
- Stored entries (RAM and block-disk L2) keep the compressed-KV pools in their 4-bit packed
  form (format v2): ~0.9 KB/token plus ~5 MB of fixed windows, so ~0.9 GB for a 1M-token
  entry. Older entries are rejected, never migrated. DSpark snapshots also carry the draft
  windows' last 128 positions, so drafting after a hit starts with a full window.
- VL requests salt their cache keys with a hash of the image pixels. A hit that covers every
  image leaves a text-only tail (DSpark eligible); a different image misses.
  `VMLX_DSV41_VL_PREFIX_CACHE=0` restores the old bypass.

## Tests

`tests/test_dsv41_*.py` (CPU only): prompt boundary, paged store, VL assembly, tool parser,
runtime policy, allocator floor, DSpark prefill step. `jang_tools/dsv41/tests/serve_acceptance.py`
(jangq) is the end-to-end serving acceptance against a real `vmlx serve`.
