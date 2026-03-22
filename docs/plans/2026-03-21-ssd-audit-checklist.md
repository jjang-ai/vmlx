# SSD Disk Streaming — Full Audit Checklist

**Last updated: 2026-03-21 session g (post-compact full cross-check audit)**

## VERIFIED WORKING (Python-level tests passed)

| # | Model | Type | Layers | Cache | Temp Save | Generate | Multi-req |
|---|-------|------|--------|-------|-----------|----------|-----------|
| 1 | MiniMax-M2.5-JANG_2L-CRACK | Standard MoE | 62 | 62 (KVCache) | npz OK | PASS | PASS |
| 2 | Qwen3.5-35B-A3B-JANG_4K | Standard MoE | 40 | 40 (KVCache) | npz OK | PASS | - |
| 3 | Qwen3.5-VL-9B-8bit | VLM (text path) | 32 | 32 (KVCache) | npz OK | PASS | - |

NOTE: These passed in direct Python testing (ssd_stream_generate called directly).
The app-level test (via panel UI) has NOT been verified working yet due to earlier bugs
with BatchedEngine routing, embed_tokens detection, and prefill lifecycle.
The app was rebuilt with all fixes but user has not yet confirmed a successful app-level test.

## KNOWN NOT WORKING

| # | Model | Type | Issue | Status |
|---|-------|------|-------|--------|
| 1 | Nemotron-Cascade-2 JANG | Hybrid SSM | 52 layers vs 29 cache entries (ArraysCache) | Clean error message |
| 2 | Any VLM via MLLM path | VLM BatchedEngine | server.py SSD init only wires LLM path | Silent fallback |

## BUGS FOUND AND FIXED THIS SESSION

| # | Bug | Root Cause | Fix | File:Line |
|---|-----|-----------|-----|-----------|
| F1 | --continuous-batching overrides SSD | cli.py didnt force continuous_batching=False | Added args.continuous_batching = False | cli.py:223 |
| F2 | Cannot find embed_tokens on Nemotron | _find_model_components only checked embed_tokens | Added embeddings, embedding, wte | ssd_generate.py:76-94 |
| F3 | Cannot find norm on Nemotron | Only checked norm | Added norm_f, final_norm, ln_f | ssd_generate.py:100-108 |
| F4 | JANG temp save PEP 3118 buffer | safetensors rejects QuantizedLinear uint32 | Switched to mx.savez (npz) | weight_index.py:130 |
| F5 | ArraysCache.make_mask unexpected kwarg | create_attention_mask passes kwargs Nemotron doesnt accept | Added TypeError to except | ssd_generate.py:155 |
| F6 | Second request rms_norm weight has 1 element | Prefill used model() but weights freed after first req | Rewrote to per-layer prefill | ssd_generate.py:297-326 |
| F7 | free_layer_weights no mx_sync before gc | Old weights stay in lazy graph | Added mx_sync after update | weight_index.py:247 |
| F8 | free_all_layer_weights gc.collect N times | Wasteful for 62+ layers | Single gc at end | weight_index.py:280-298 |
| F9 | 52 layers vs 29 cache zip truncates silently | Nemotron hybrid SSM non-1:1 mapping | Added check + RuntimeError | ssd_generate.py:296-301 |

## BUGS FOUND IN AUDIT (cross-checked session g, fixed session h)

| # | Severity | Bug | File | Impact | Status |
|---|----------|-----|------|--------|--------|
| A1 | CRITICAL | test_disk_streaming.py imports 5 removed symbols (12 tests in 4 classes) | tests:401-662 | Tests crash with ImportError | **FIXED** — 4 zombie test classes deleted, 27 tests pass |
| A2 | HIGH | --ssd-memory-budget and --ssd-prefetch-layers parsed but never forwarded | cli/server/panel | UI sliders do nothing | **FIXED** — Dead sliders removed from UI, dead args removed from buildArgs + preview |
| ~~A3~~ | ~~HIGH~~ | ~~Deep sleep wake: SSD not re-configured~~ | ~~server.py~~ | ~~Weight recycling stops after sleep~~ | **FALSE POSITIVE** — admin_wake passes stream_from_disk + stream_memory_percent to load_model() |
| A4 | MEDIUM | MLLM + SSD: model class doesn't have _stream_from_disk attr | server.py | VLM via MLLM path gets no SSD benefit | **FIXED** — Warning logged when MLLM model doesn't support SSD |
| A5 | MEDIUM | mx.load reads entire shard for one layer | weight_index.py:199 | Perf: loads 5GB file per layer | KNOWN (perf) — future optimization |
| A6 | MEDIUM | make_sampler import from mlx_lm.generate — path may differ | ssd_generate.py:255 | Could crash on newer mlx_lm | KNOWN (compat risk) |
| A7 | LOW | Repetition penalty .at[].add() — MLX version dependent | ssd_generate.py:417 | Older MLX may fail | KNOWN (compat risk) |
| A8 | LOW | JANG temp dir never cleaned up on shutdown | server.py:1081 | Disk space leak | **FIXED** — atexit handler + previous temp dir cleanup on reload |
| ~~A9~~ | ~~LOW~~ | ~~bench subparser missing new SSD args~~ | ~~cli.py~~ | ~~Cant bench SSD~~ | **FALSE POSITIVE** — bench has --stream-from-disk at cli.py:1271 |
| A10 | LOW | free_all_layer_weights imported but never called | ssd_generate.py:28 | Dead import | **FIXED** — Removed unused import |
| NEW | — | Loading progress stalls at 60% during SSD setup | sessions.ts + weight_index.py | No user feedback during JANG temp save | **FIXED** — 3 SSD progress patterns added + per-10-layer logging |

## SETTINGS INTERACTION MATRIX

| Setting | With SSD ON | Verified |
|---------|------------|----------|
| --continuous-batching | Forced FALSE | YES |
| --use-paged-cache | Forced FALSE | YES |
| --enable-prefix-cache | Forced FALSE | YES |
| --kv-cache-quantization | Forced none | YES |
| --enable-disk-cache | Forced FALSE | YES |
| --enable-block-disk-cache | Forced FALSE | YES |
| --cache-memory-percent | Forced 0.0 | YES |
| --cache-memory-mb | Forced 0 | YES |
| --max-num-seqs | Forced 1 | YES |
| --speculative-model | Forced None | YES |
| --ssd-memory-budget | Parsed NOT USED | BUG A2 |
| --ssd-prefetch-layers | Parsed NOT USED | BUG A2 |
| --stream-memory-percent | Sets Metal limit | YES |

## ENGINE ROUTING MATRIX

| Model Type | SSD ON | Engine | SSD Active? | Status |
|-----------|--------|--------|-------------|--------|
| Standard LLM | YES | SimpleEngine | YES | WORKING |
| JANG LLM | YES | SimpleEngine | YES (npz) | WORKING |
| MoE LLM | YES | SimpleEngine | YES | WORKING |
| VLM text path | YES | SimpleEngine LLM | YES | WORKING |
| VLM MLLM path | YES | SimpleEngine | NO | BUG A4 |
| Hybrid SSM | YES | SimpleEngine | NO | KNOWN LIMIT |
| BatchedEngine | YES | IMPOSSIBLE | N/A | Gated by cli.py |

## LAZY LOADING CHAIN (session g cross-check)

All these consumers read server._stream_from_disk to enable lazy mmap loading:

| Consumer | File:Line | Reads Flag | Passes To | Status |
|----------|-----------|------------|-----------|--------|
| tokenizer.py (LLM load) | tokenizer.py:141 | `getattr(_server_module, '_stream_from_disk', False)` | load_jang_model(lazy=), _load_with_tokenizer_fallback(lazy=), load(lazy=) | LIVE |
| mllm.py (JANG VL load) | mllm.py:735 | `getattr(_server_module, '_stream_from_disk', False)` | load_jang_vlm_model(lazy=) | LIVE |
| mllm.py (standard VL load) | mllm.py:757 | `getattr(_server_module, '_stream_from_disk', False)` | mlx_vlm.load(lazy=) | LIVE |
| mllm_batch_generator.py | mllm_batch_gen:986 | `getattr(_server_module, '_stream_from_disk', False)` | Skips wired limit + cache limit override if True | LIVE |

## DEEP SLEEP WAKE CHAIN (session g cross-check — A3 verified FALSE POSITIVE)

| Step | File:Line | Verified |
|------|-----------|----------|
| _cli_args saves stream_from_disk | server.py:837 | YES |
| _cli_args saves stream_memory_percent | server.py:838 | YES |
| admin_wake passes stream_from_disk | server.py:1424 | YES |
| admin_wake passes stream_memory_percent | server.py:1425 | YES |
| load_model re-runs SSD init (1047-1092) | server.py:1047 | YES — rebuilds weight_index, saves JANG temp, re-sets all 4 LLM attrs |

## FRONTEND ↔ BACKEND WIRING (session g cross-check)

| Field | TS Type | Form | Default | Whitelist | buildArgs | Preview | CLI Flag | Status |
|-------|---------|------|---------|-----------|-----------|---------|----------|--------|
| streamFromDisk | server.ts:56 | CheckField | false | sessions.ts:861 | --stream-from-disk | SessionSettings:66 | cli.py:937 | LIVE |
| streamMemoryPercent | server.ts:58 | Slider 50-95 | 90 | sessions.ts:861 | --stream-memory-percent N | SessionSettings:68 | cli.py:945 | LIVE |
| ssdMemoryBudget | server.ts:59 | Slider 0-16384 | 0 | sessions.ts:861 | --ssd-memory-budget N | SessionSettings:71 | cli.py:953 | DEAD (A2) |
| ssdPrefetchLayers | server.ts:60 | Slider 0-8 | 0 | sessions.ts:861 | --ssd-prefetch-layers N | SessionSettings:74 | cli.py:961 | DEAD (A2) |

## OLD SYMBOLS REMOVAL VERIFICATION (session g cross-check)

All old streaming symbols CONFIRMED removed from production code. Only in test zombie code + docstrings.

| Symbol | Production | Tests | Docs | Status |
|--------|-----------|-------|------|--------|
| StreamingLayerWrapper | CLEAN | 4 zombie tests | docstring only | DELETE TESTS |
| apply_streaming_layers | CLEAN | 3 zombie tests | docstring only | DELETE TESTS |
| compute_streaming_wired_limit | CLEAN | 2 zombie tests | mentioned | DELETE TESTS |
| lock_wired_limit | CLEAN | 3 zombie tests | mentioned | DELETE TESTS |
| unlock_wired_limit | CLEAN | 3 zombie tests | mentioned | DELETE TESTS |

## EVERY FILE CHANGED

| File | Action | What Changed |
|------|--------|-------------|
| vmlx_engine/utils/ssd_generate.py | NEW+REWRITTEN | Custom generate loop, per-layer prefill+decode |
| vmlx_engine/utils/weight_index.py | NEW+FIXED | Weight mapping, npz save, load/free with sync |
| vmlx_engine/utils/streaming_wrapper.py | REPLACED | Removed broken code, kept _find_layers only |
| vmlx_engine/models/llm.py | MODIFIED | Added SSD attrs + routing in generate/stream_generate |
| vmlx_engine/server.py | MODIFIED | Replaced old wrapper init with weight recycling init |
| vmlx_engine/cli.py | MODIFIED | New args, force SimpleEngine, updated banner |
| panel SessionConfigForm.tsx | MODIFIED | ssdMemoryBudget + ssdPrefetchLayers sliders |
| panel sessions.ts | MODIFIED | buildArgs + config serialization for new fields |
| panel SessionSettings.tsx | MODIFIED | Command preview for new args |
| panel server.ts | MODIFIED | Type definitions for new fields |
| vmlx_engine/mllm_batch_generator.py | MODIFIED | Reads _stream_from_disk, skips wired/cache limit override if True (line 986) |
| vmlx_engine/models/mllm.py | MODIFIED | Reads _stream_from_disk, passes lazy= to load_jang_vlm_model and mlx_vlm.load (lines 735, 757) |
| vmlx_engine/utils/tokenizer.py | MODIFIED | Reads _stream_from_disk, passes lazy= to all model loaders (line 141) |
| docs/plans/ssd-streaming-design.md | UPDATED | Status to IMPLEMENTED |
| docs/plans/ssd-implementation-matrix.md | NEW | Full technical docs |
| docs/plans/ssd-audit-checklist.md | NEW | This file (updated session g) |
