# DeepSeek-V4.1-Flash — engine integration

Engine-side contract for DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, family
`deepseek_v4_1`). The runtime is `jang_tools.dsv41` (jangq). Its `RUNTIME_SERVING.md`
covers Engram I/O, the prefix-cache contract and measured results, and
`DECODE_DSPARK_METHODS.md` covers the DSpark decode methods.

## Speed-critical engine pieces

Each item removes a measured cause of slowness; jangq `RUNTIME_SERVING.md` opens with the full ledger
(runtime and engine).
- Allocator-cache floor 16 GB for `deepseek_v41`: the generic 4 GB cap made every 2048-token chunk free
  and re-create its Metal buffers (21k cold prompt 475 → 769 pp/s).
- 1024-token prefill chunks while the DSpark draft is resident: 2048 crossed Metal's recommended working
  set and slowed the GPU ~30% (545 → 708 pp/s).
- `engram_lookahead`: each chunk hands the runtime the next chunk, so its SSD Engram reads overlap this
  chunk's compute.
- `prompt_end`: lets the runtime's layers 21-39 skip prompt rows the prefill end no longer reads (100k
  875 → 1584 pp/s, exact at the N-1 snapshot).
- DSpark generator: one verify forward per step with in-place partial-accept commit and a governor that
  never runs below AR; the runtime's fused decode kernels make each verify row cost what a 1-row decode does.

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
- With the runtime's group-major `wo_a` (jangq 66966c2) served cold prefill with DSpark on is
  822 / 933 / 872 pp/s at 1.6k / 6.5k / 26.5k tokens.
- Long context: the runtime's indexer uses this repo's `metal/dsa_index_scores.dsa_pool_index_scores`
  for prefill (optional import from jang_tools) with `bf16_operands=True`, which runs
  `metal/dsa_index_scores_mpp.py` on the M5 matrix units (~4.3x the fp32 simdgroup kernel; falls back
  to it elsewhere), plus a block top-k for prefill rows. Cold prefill to 100k averages 884 pp/s (88–100k
  window 910). `bf16_operands` is opt-in, so GLM-5.3 and other callers are unchanged.
- Prompt-suffix pruning: `SingleBatchGenerator._prefill` passes every chunk the absolute cache position
  where that prefill call ends (`_model_call(..., prompt_end=)`, DSV4.1 only), and the runtime's layers
  21-39 run only the trailing rows the prefill's last token still reads (jangq `Model.prune_rows`). The
  cache at that position, and so the N-1 snapshot, matches a full prefill. Cold prefill without the
  draft: 8k / 32k / 100k 1452 / 1641 / 1584 pp/s (875 at 100k without). `VMLX_DSV41_SUFFIX_PRUNE=0`
  disables it in the runtime.
- Served cold prefill with DSpark on (2026-10-26): 845 / 1331 / 1554 pp/s at 1.6k / 6.5k / 26.5k.

## Decode

- DSpark (`jang_tools.dsv41.dspark.DSparkRunner`) runs one verify forward per step. Partial accepts
  are committed in place (`DSV41.commit_prefix`) rather than with a second forward, and a wall-clock
  governor keeps every step at or above AR speed. Logits come from an fp32 LM-head input, as in the
  reference; bf16 logits tied too often for greedy decoding to be stable.
- Output equals greedy AR except at near-ties (AR top-2 margin below ~0.05): a multi-row verify and a
  1-row decode differ in the last bits of the bf16 backbone.
- `vmlx_engine/metal/row_exact_qmv.py` (row-exact multi-row QMV) works on MLX 0.32.4 but is not wired
  for V4.1. Its verify cost is dominated by routed experts and attention, so it gave no measurable
  speedup and does not make verify bit-identical to decode.
- The runtime's fused decode kernels (jangq `metal_decode.py`: per-layer async eval, bf16 router GEMV,
  2-dispatch MoE, fused hyper-connection stage; bit-exact, `VMLX_DSV41_{LAYER_ASYNC,ROUTER_GEMV,FUSED_MOE,FUSED_HC}=0`
  disable) cover decode and DSpark verify rows. AR decode 29.0 -> 39.2 tok/s at 2k context, 26.0 -> 34.3 at 100k.
- Serving decode (M5 Ultra, 256 tokens, thinking off, 2026-10-26): code 90.2, code2 62.3, structured 111.9,
  prose 38.7 tok/s; TTFT 0.21-0.46 s.

## Prefix cache

- Prompt-boundary snapshots are taken at N-1 prompt tokens, before the last token, and
  reconstruction rejects a restored offset that disagrees with the block table.
- Stored entries (RAM and block-disk L2) keep the compressed-KV pools in their 4-bit packed
  form (format v2). Measured at a real 1M-token context: 905 MiB resident, 909 MiB stored
  (949 MB; 905 B/token), the same as the resident cache. Older entries are rejected, never migrated. DSpark snapshots also carry the draft
  windows' last 128 positions, so drafting after a hit starts with a full window.
- Long prompts: `deepseek_v4` state tensors are bounded by the cache offset cap (2M rows), not the generic 256K
  per-dimension cap, in both the record validator (#332) and the SSD header validator (#334). Before, every V4.1
  entry above ~262k tokens was rejected and re-prefilled. Measured on a 924k-token prompt: cold 1352 s; repeat from
  RAM 5.8 s (923,823 cached); after a server restart, from SSD 6.5 s (3,609 blocks, 849 MB on disk); same answer in
  all three. The persist at 1M takes ~3.9 s in the background.
- VL requests salt their cache keys with a hash of the image pixels. A hit that covers every
  image leaves a text-only tail (DSpark eligible); a different image misses.
  `VMLX_DSV41_VL_PREFIX_CACHE=0` restores the old bypass.

## Tests

`tests/test_dsv41_*.py` (CPU only): prompt boundary, paged store, VL assembly, tool parser,
runtime policy, allocator floor, DSpark prefill step. `jang_tools/dsv41/tests/serve_acceptance.py`
(jangq) is the end-to-end serving acceptance against a real `vmlx serve`.
