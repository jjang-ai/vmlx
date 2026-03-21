# vMLX Cross-Check Matrix — Exhaustive Verification Checklist

> **Purpose:** Before ANY release, run through this matrix systematically. Each cell is a specific combination of features that MUST be verified.
>
> **How to use:** For each section, check every numbered item. Mark [x] when verified. If an item fails, create a fix and re-verify the entire section.

---

## A. MODEL LOADING x MODEL TYPE

| # | Check | LLM | Hybrid SSM | MoE | JANG Text | JANG MoE | VLM | JANG VL | Image |
|---|-------|-----|-----------|-----|-----------|----------|-----|---------|-------|
| A1 | Loads without error | | | | | | | | |
| A2 | model_type detected | | | | | | | | |
| A3 | Registry returns family | | | | | | | | |
| A4 | Tool parser selected | | | | | | | | |
| A5 | Reasoning parser selected | | | | | | | | |
| A6 | is_mllm correct | | | | | | | | |
| A7 | Tokenizer loads | | | | | | | | |
| A8 | First inference works | | | | | | | | |
| A9 | Streaming works | | | | | | | | |

- [ ] A10: JANG gate dequant for MoE (log: "Dequantized gate:")
- [ ] A11: fc1/fc2 rename ONLY Nemotron
- [ ] A12: MTP filter ONLY Nemotron
- [ ] A13: bfloat16 override ONLY >= 512 experts
- [ ] A14: VLM text_config nesting for gate dequant
- [ ] A15: mistral3 dispatches to mistral4 backbone
- [ ] A16: MLLMModelWrapper extracts .logits

## B. CACHING x MODEL TYPE x CACHE TYPE

| # | Check | Prefix | Mem-Aware | Paged | Block Disk | Disk L2 | KV q8 | KV q4 |
|---|-------|--------|----------|-------|-----------|---------|-------|-------|
| B1 | Created (not None) | | | | | | | |
| B2 | Stores on first req | | | | | | | |
| B3 | Hits on repeat prompt | | | | | | | |
| B4 | Stats in API | | | | | | | |
| B5 | Clear works | | | | | | | |
| B6 | + Standard LLM | | | | | | | |
| B7 | + Hybrid SSM | | | | | | | |
| B8 | + MoE CacheList | | | | | | | |
| B9 | + JANG | | | | | | | |
| B10 | + VLM | | | | | | | |

- [ ] B11: Hybrid auto-switch to paged
- [ ] B12: KV quant dequant correctness
- [ ] B13: KV quant skips MambaCache
- [ ] B14: Block disk writes on main thread
- [ ] B15: Disk cache model+quant hash scoping
- [ ] B16: Paged CacheList support
- [ ] B17: QuantizedKVCache round-trip
- [ ] B18: Cache TTL works
- [ ] B19: Memory-aware respects limits

## C. DISK STREAMING x EVERYTHING

- [ ] C1: ALL cache flags forced off
- [ ] C2: max_num_seqs=1
- [ ] C3: speculative_model=None
- [ ] C4: Metal limit RAISED (multiplier of RAM, not capped)
- [ ] C5: Metal cache limit = 0
- [ ] C6-C10: lazy=True in ALL 5 load paths
- [ ] C11: mx.eval skipped when lazy
- [ ] C12: Size estimated from safetensors files
- [ ] C13: Warning if model > 90% RAM
- [ ] C14: Zero cache objects in scheduler
- [ ] C15: MLLMBatchGenerator skips Metal override
- [ ] C16: /v1/cache/stats no crash
- [ ] C17-C19: Chat/completions/concurrent work
- [ ] C20-C24: Deep sleep preserves ALL stream settings
- [ ] C25: Soft sleep handles None caches
- [ ] C26-C27: bench_command gating
- [ ] C28-C31: UI/IPC consistency
- [ ] C32: Normal mode COMPLETELY unaffected

## D. SLEEP/WAKE x MODEL x CACHE

| # | Check | Soft | Deep | JIT | Manual |
|---|-------|------|------|-----|--------|
| D1 | Endpoint responds | | | | |
| D2 | _standby_state set | | | | |
| D3 | /health reports state | | | | |
| D4 | Caches cleared | | | | |
| D5 | Model unloaded | | | | |
| D6 | _cli_args preserved | | | | |
| D7 | Cache limit saved | | | | |
| D8 | Cache limit restored (is not None) | | | | |
| D9 | Model reloads correctly | | | | |
| D10 | Inference works after | | | | |

- [ ] D11: JIT wake from /v1/chat/completions
- [ ] D12: _wake_lock prevents double load
- [ ] D13: Wake timeout 300s
- [ ] D14: Soft sleep + stream mode (None caches)
- [ ] D15: Deep wake + stream mode (lazy re-applied)
- [ ] D16: Deep wake + JANG (gate dequant re-runs)
- [ ] D17: Idle timer tracks ALL request types
- [ ] D18: Image model sleep/wake

## E. API ENDPOINTS

| # | Endpoint | OpenAI | Anthropic | Stream | Non-Stream | Tools | Reasoning |
|---|---------|--------|-----------|--------|-----------|-------|-----------|
| E1 | /v1/chat/completions | | | | | | |
| E2 | /v1/completions | | | | | | |
| E3 | /v1/models | | | | | | |
| E4 | /health | | | | | | |
| E5 | /v1/cache/stats | | | | | | |
| E6 | /v1/embeddings | | | | | | |
| E7 | /v1/rerank | | | | | | |
| E8 | /v1/images/generations | | | | | | |
| E9 | /admin/soft-sleep | | | | | | |
| E10 | /admin/deep-sleep | | | | | | |
| E11 | /admin/wake | | | | | | |

- [ ] E12: Anthropic format conversion
- [ ] E13: SSE data: prefix + [DONE]
- [ ] E14: Tool call accumulation across chunks
- [ ] E15: reasoning_content separated in streaming
- [ ] E16: Stop button aborts generation
- [ ] E17: Client disconnect detected
- [ ] E18: Reranker local ref inside lock
- [ ] E19: Proper HTTP status codes

## F. TOOL CALLING x MODEL

| # | Mistral | DeepSeek | Llama | Qwen | Nemotron | MiniMax |
|---|---------|----------|-------|------|----------|---------|
| F1 | Parser loads | | | | | |
| F2 | Auto-detect | | | | | |
| F3 | Single call | | | | | |
| F4 | Parallel calls | | | | | |
| F5 | Stream accum | | | | | |
| F6 | Native format | | | | | |

## G. REASONING x MODEL

| # | Qwen3 | DeepSeek R1 | GPT-OSS | Think Tag | None |
|---|-------|-------------|---------|-----------|------|
| G1 | Parser loads | | | | |
| G2 | Auto-detect | | | | |
| G3 | Batch extract | | | | |
| G4 | Stream extract | | | | |
| G5 | Suppress mode | | | | |
| G6 | OFF = no parser | | | | |
| G7 | Multi-turn | | | | |

## H. JANG DEEP CHECKS

- [ ] H1: All config filenames detected
- [ ] H2: v2 mmap load
- [ ] H3: v1 repack still works
- [ ] H4: Gate dequant bits [8,6,4,3,2]
- [ ] H5: Gate output bfloat16
- [ ] H6: e_score_correction_bias NOT caught by gate filter
- [ ] H7: Shard flush NOT guarded by lazy
- [ ] H8: Mixed group_size handled
- [ ] H9: VL lm_head quant fix
- [ ] H10: JANG forces text path (not VLM)

## I. HYBRID SSM DEEP CHECKS

- [ ] I1: _is_hybrid correctly identifies Mamba+KV
- [ ] I2: CacheList discarded (MoE not falsely hybrid)
- [ ] I3: BatchMambaCache.extend() correct
- [ ] I4: _fix_hybrid_cache expansion
- [ ] I5: Chunked prefill no broadcast error
- [ ] I6: Auto-switch to paged for hybrid
- [ ] I7: Cache reconstruction layer count

## J. UI/IPC CONSISTENCY

- [ ] J1: Every CLI arg has TS config field
- [ ] J2: Every config field has DEFAULT_CONFIG
- [ ] J3: Every restart setting in RESTART_REQUIRED_KEYS
- [ ] J4: buildArgs mirrors buildCommandPreview
- [ ] J5: MODEL_TYPE_TO_FAMILY has all types
- [ ] J6: registerFamily has all families
- [ ] J7: ServerConfig has all fields
- [ ] J8: CachePanel handles all states
- [ ] J9: PerformancePanel handles all states

## K. NAMING CONSISTENCY

| Python | CLI | TypeScript |
|--------|-----|-----------|
| stream_from_disk | --stream-from-disk | streamFromDisk |
| stream_memory_percent | --stream-memory-percent | streamMemoryPercent |
| enable_prefix_cache | --enable-prefix-cache | enablePrefixCache |
| use_paged_cache | --use-paged-cache | usePagedCache |
| kv_cache_quantization | --kv-cache-quantization | kvCacheQuantization |
| enable_disk_cache | --enable-disk-cache | enableDiskCache |
| enable_block_disk_cache | --enable-block-disk-cache | enableBlockDiskCache |
| max_num_seqs | --max-num-seqs | maxNumSeqs |
| tool_call_parser | --tool-call-parser | toolCallParser |
| reasoning_parser | --reasoning-parser | reasoningParser |

## L. KNOWN EDGE CASES (from past bugs)

| # | Edge Case | Root Cause | Where |
|---|-----------|-----------|-------|
| L1 | Metal crash on background thread | mx.save_safetensors accesses Metal buffers | block_disk_store.py |
| L2 | MoEGate not QuantizedLinear | nn.quantize skips nn.Module, JANG must dequant | jang_loader.py |
| L3 | Cache limit 0 is valid | Truthy check fails, use `is not None` | server.py |
| L4 | Reranker race condition | Lock released before .rerank() call | server.py |
| L5 | VLM text_config nesting | n_routed_experts at wrong level | jang_loader.py |
| L6 | Hybrid + memory-aware | Can't truncate MambaCache | scheduler.py |
| L7 | Metal limit too LOW | Prevents macOS SSD paging | server.py (must RAISE) |
| L8 | KVCache list vs tuple | Paged stores list, quant expects tuple | mamba_cache.py |
| L9 | Streaming auto-scroll | Content pushes bottom away | MessageList.tsx |
| L10 | Bundled Python isolation | -s flag blocks user packages | Bundle all deps |

**This checklist grows with the codebase. Never shrink it.**
