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
  (`vmlx_engine/utils/dsv41_dspark_generator.py`), a `SingleBatchGenerator` subclass. It prefills the
  request's engine cache with `DSparkRunner.prefill(session=<engine cache>)` and decodes with
  `DSparkRunner.decode_stream` on that session.
- DSpark serves text requests, greedy or sampled. Greedy means temperature 0 (the `_vmlx_greedy` /
  `_vmlx_is_greedy` tag or a recorded temperature of 0); `_vmlx_accepts_logits` is not a greedy marker,
  because the compact top-k sampler (temperature > 0, top_k > 0) carries it too. A temperature > 0 request
  runs exact sampled speculation (`SpecSampler`: the AR sampler's temperature / top_p / min_p / top_k, Gumbel
  noise keyed by the request seed and the absolute position). The adaptive governor's AR steps use the same
  position-keyed noise, so a fixed `seed` gives the same tokens whatever the governor does. Image turns,
  logits processors (repetition / frequency / presence penalty, logit_bias) and logprobs take the inherited
  AR path. `VMLX_DSV41_DSPARK_SAMPLED=0` sends sampled requests to AR.
- The scheduler identifies generators by class name (its unit tests drive it with stand-in classes), so a
  check that must cover the DSpark generator lists `DSV41DSparkGenerator` next to `SingleBatchGenerator`
  (restored-prefix context, prompt-snapshot telemetry). The telemetry in `Scheduler.get_stats` and the cache
  stats label them separately (`dsv41_dspark` vs `single_active`).
- The CLI policy for `deepseek_v4_1` serves one sequence, with no JIT, smelt or flash-MoE,
  and a 900 s request timeout.

## Chat rendering

- The bundle's `encoding/encoding.py` renders every prompt. It is loaded once per bundle
  (`models/deepseek_v4_1/chat_encoding.load_bundle_encoding`) and shared by the loader's
  `apply_chat_template` shim (text turns), `dsv41_vl_preprocess` (image turns) and the DSML tool parser.
- Reasoning effort: the encoder accepts an int 1..100 or low/high/max and asserts on anything else. Every
  endpoint (`/v1/chat/completions`, `/v1/responses` `reasoning.effort`, `/v1/messages` `output_config.effort`,
  `/api/chat`) and both render paths map the effort to an int budget with the Swift runtime's ladder
  (`DeepseekV4ChatEncoder.reasoningBudget`): minimal 25, low 50, medium 62, high 75, xhigh 88, max 100; ints
  and numeric strings clamp to 1..100. A generic `thinking_mode=reasoning` keeps the bundle default (high, 75).
  `none` (and off / false / disabled) selects the chat rail, as in the osaurus app. An unknown string is
  logged and the encoder default is used.
- Thinking-token budgets (`max_thinking_tokens`, Anthropic `thinking.budget_tokens`, Responses
  `reasoning.budget_tokens`) are accepted but do not apply to V4.1: it steers reasoning depth with the effort
  only, like DSV4 (`deepseek_v4_1` is not in `_THINKING_BUDGET_CAP_FAMILIES`). The server logs this once per
  request at info level and drops `thinking_budget` from the template kwargs.
- `add_generation_prompt` is accepted but not applied: the encoder always ends a user or tool turn with the
  assistant rail, so the measured generation-prompt length is 0. The prefix cache relies on that: the stored
  state is the exact N-1 boundary, which already drops the trailing `<think>` / `</think>` token, so the next
  turn's history extends it.
- Sampling defaults: the IMATRIX bundle's `generation_config.json` declares do_sample true, temperature 1.0,
  top_p 1.0 (DeepSeek's recommended values). A request with no sampling params resolves to those on all three
  endpoints (`_resolve_temperature` / `_resolve_top_p`; no bundle top_k, min_p or repetition penalty), so
  DSpark takes the sampled path.

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
  `VMLX_DSV41_VL_PREFIX_CACHE=0` restores the old bypass. Known limitation: the salt hashes the request's
  whole image set, so adding an image to a conversation changes the salt and that turn misses the history
  stored under the earlier image set (it re-prefills from the start).
- RAM prefix cache (memory-aware, used when neither paged RAM nor block-disk L2 is on): a `DeepseekV41Cache`
  list is stored as one copy only when it sits at the N-1 boundary (it cannot be trimmed, and every layer
  shares the sequence position and Engram history); a hit returns an independent copy and a shorter
  (reverse) match is a miss. A V4.1 cache is never rebuilt as a V4.0 `DeepseekV4Cache`.
- V4.1 shares the DSV4 composite cache store (`DeepseekV41Cache` is in the DSV4 cache-class set: no generic
  KV quantization, typed paged/SSD records, prompt-boundary snapshots). The V4.0-only gates are split off on
  `Scheduler._uses_dsv41_cache`: the 256-token snapshot store threshold (V4.1 stores short prompts too), the
  forced 256-token paged block size, the `DSV4_*` block-namespace scope (V4.1 uses
  `dsv41_cache_schema=deepseek_v41_fmt2`), the `DSV4_MAX_PREFILL_TOKENS` guard and the idle shadow re-key.

## Environment flags

Read by the server (`1` = on; `0` / `false` / `off` / `no` = off):

| Flag | Default | Effect |
|---|---|---|
| `VMLX_DSV41_DSPARK` | on | Load the bundle's dSpark draft and use `DSV41DSparkGenerator`; off serves AR only. |
| `VMLX_DSV41_DSPARK_ADAPTIVE` | on | Wall-clock governor (AR or speculative step) and acceptance-tracked draft width; off speculates every step at full width. |
| `VMLX_DSV41_DSPARK_SAMPLED` | on | Exact sampled speculation for temperature > 0; off sends sampled requests to the AR path. |
| `VMLX_DSV41_DSPARK_PREFILL_STEP` | 1024 | Prefill chunk while the draft is resident (the configured step is capped at 1024 otherwise). |
| `VMLX_DSV41_VL` | on | Image turns on a VL bundle; off forces text-only. |
| `VMLX_DSV41_VL_PREFIX_CACHE` | on | Prefix cache for image turns, salted by the image pixels; off bypasses the cache for them. |

Read by the jangq runtime in the same process (the server passes the environment through unchanged):

| Flag | Default | Effect |
|---|---|---|
| `VMLX_DSV41_SUFFIX_PRUNE` | on | Layers 21-39 run only the prompt rows the prefill's last token still reads. |
| `VMLX_DSV41_LAYER_ASYNC` | on | Per-layer async eval for decode / verify forwards of up to 16 rows. |
| `VMLX_DSV41_ROUTER_GEMV` | on | bf16 router GEMV for up to 8 rows (no f32 weight copy). |
| `VMLX_DSV41_FUSED_MOE` | on | Routed + shared experts in two Metal dispatches for up to 8 rows. |
| `VMLX_DSV41_FUSED_HC` | on | Fused hyper-connection stage for up to 8 rows. |
| `VMLX_DSV41_FAST_ROPE` | on | `mx.fast.rope` for consecutive positions. |
| `VMLX_DSV41_FUSED_INDEX` | on | vMLX's fused DSA index-score kernel. |
| `VMLX_DSV41_INDEX_BF16` | on | bf16 operands for the fused index scores (M5 matrix units). |

## Tests

`tests/test_dsv41_*.py` (CPU only): prompt boundary, paged store, RAM prompt-cache store, cache policy split,
VL assembly, tool parser, reasoning effort, sampling defaults, DSpark sampled routing and seeded governor,
runtime policy, allocator floor, DSpark prefill step. `jang_tools/dsv41/tests/serve_acceptance.py`
(jangq) is the end-to-end serving acceptance against a real `vmlx serve`.
