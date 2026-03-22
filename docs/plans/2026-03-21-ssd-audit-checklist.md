# SSD Disk Streaming — Full Audit Checklist

**Last updated: 2026-03-21 session f (post-live-testing)**

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

## BUGS FOUND BUT NOT YET FIXED

| # | Severity | Bug | File | Impact |
|---|----------|-----|------|--------|
| A1 | CRITICAL | test_disk_streaming.py imports 5 removed symbols | tests:401-662 | 15 tests crash, breaks CI |
| A2 | HIGH | --ssd-memory-budget and --ssd-prefetch-layers never forwarded | cli/server/ssd_generate | UI sliders do nothing |
| A3 | HIGH | Deep sleep wake: SSD not re-configured | server.py:904-911 | Weight recycling stops after sleep |
| A4 | MEDIUM | MLLM + SSD: silently falls through | server.py:1054-1086 | VLM gets no SSD benefit |
| A5 | MEDIUM | mx.load reads entire shard for one layer | weight_index.py:199 | Perf: 5GB read per layer |
| A6 | MEDIUM | make_sampler import path may differ | ssd_generate.py:255 | Could crash on newer mlx_lm |
| A7 | LOW | Repetition penalty .at[].add() version dep | ssd_generate.py:385 | Older MLX may fail |
| A8 | LOW | JANG temp dir never cleaned up | server.py:1077 | Disk space leak |
| A9 | LOW | bench subparser missing new SSD args | cli.py | Cant bench SSD |

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
| docs/plans/ssd-streaming-design.md | UPDATED | Status to IMPLEMENTED |
| docs/plans/ssd-implementation-matrix.md | NEW | Full technical docs |
| docs/plans/ssd-audit-checklist.md | NEW | This file |
