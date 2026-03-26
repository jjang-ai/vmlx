# vMLX Engine Architecture

## Request Flow — End to End

```
                         ┌─────────────────────────────────────────┐
                         │              HTTP CLIENT                │
                         │   (Chat UI / Claude Code / curl / SDK)  │
                         └────────────┬───────────┬────────────────┘
                                      │           │
                              OpenAI format   Anthropic format
                                      │           │
                                      ▼           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           server.py (FastAPI)                              │
│                                                                           │
│  POST /v1/chat/completions ──────────────┐                                │
│  POST /v1/messages ── anthropic_adapter ─┤                                │
│  POST /v1/completions ───────────────────┤                                │
│  POST /v1/images/generations ────────────┤── image_gen.py (Flux/mflux)    │
│  GET  /v1/models ────────────────────────┤                                │
│  GET  /health ───────────────────────────┤                                │
│  POST /admin/{soft-sleep,deep-sleep,wake}┘                                │
│                                                                           │
│  Middleware: JIT wake, timeout, CORS, rate limit, API key                 │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    get_engine()       │
              │                      │
              │  --continuous-batch?  │
              │    YES → BatchedEngine│
              │    NO  → SimpleEngine │
              └───────┬───────┬───────┘
                      │       │
          ┌───────────┘       └────────────┐
          ▼                                ▼
┌──────────────────┐            ┌──────────────────────┐
│  SimpleEngine    │            │   BatchedEngine       │
│  (single request)│            │   (concurrent batch)  │
│                  │            │                       │
│  mlx_lm.generate │            │  ┌─────────────────┐ │
│  or              │            │  │ Scheduler        │ │
│  mlx_vlm.generate│            │  │ (LLM or MLLM)   │ │
│                  │            │  └────────┬─────────┘ │
│  No caching      │            │           │           │
│  No batching     │            │           ▼           │
│  No TQ compress  │            │  ┌─────────────────┐ │
└──────────────────┘            │  │ BatchGenerator   │ │
                                │  │ (prefill+decode) │ │
                                │  └─────────────────┘ │
                                └──────────────────────┘
```

## Scheduler + Cache Architecture (BatchedEngine only)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SCHEDULER (LLM or MLLM)                          │
│                                                                     │
│  add_request() ──► waiting queue ──► _schedule() ──► running batch  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              CACHE FETCH (checked in order)                    │  │
│  │                                                                │  │
│  │  1. ┌─────────────────┐   Block hash chain lookup              │  │
│  │     │  Paged Cache    │   64-tok blocks, SHA-256 hashed        │  │
│  │     │  (paged_cache)  │   + SSM Companion (hybrid models)      │  │
│  │     └────────┬────────┘                                        │  │
│  │              │ MISS                                            │  │
│  │              ▼                                                 │  │
│  │  2. ┌─────────────────┐   Token-trie prefix matching           │  │
│  │     │  Memory Cache   │   LRU, memory-limited                  │  │
│  │     │  (memory_cache) │   Skips hybrid SSM models              │  │
│  │     └────────┬────────┘                                        │  │
│  │              │ MISS                                            │  │
│  │              ▼                                                 │  │
│  │  3. ┌─────────────────┐   Safetensors on SSD                   │  │
│  │     │  Disk L2 Cache  │   Background write (no Metal on IO)    │  │
│  │     │  (disk_cache)   │   TQ→KVCache remap on fetch            │  │
│  │     └────────┬────────┘                                        │  │
│  │              │ MISS                                            │  │
│  │              ▼                                                 │  │
│  │  4. Full prefill (vision encode + attention)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              CACHE STORE (after generation)                    │  │
│  │                                                                │  │
│  │  compress_cache() ─► TQ 3-bit (if TurboQuant active)          │  │
│  │       │                                                        │  │
│  │       ▼                                                        │  │
│  │  extract .state ──► float KV arrays                            │  │
│  │       │                                                        │  │
│  │       ├──► Paged blocks (float, for multi-turn reuse)          │  │
│  │       ├──► SSM companion store (hybrid models)                 │  │
│  │       ├──► KV quantize to q8/q4 (if enabled)                   │  │
│  │       └──► Disk L2 write (background thread)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  mx.clear_cache() when batch empties (reclaim Metal memory)         │
└──────────────────────────────────────────────────────────────────────┘
```

## Token Generation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                  MLLMBatchGenerator                              │
│                                                                  │
│  PREFILL PHASE (process all prompt tokens at once)               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  1. Cache fetch (paged → memory → disk → full prefill)    │  │
│  │  2. Vision encode (if images present)                      │  │
│  │  3. model.forward(all_prompt_tokens, cache=kv_cache)       │  │
│  │  4. TurboQuant: FILL phase (float KV, zero overhead)      │  │
│  │  5. Capture SSM state (hybrid models)                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  DECODE PHASE (one token at a time, batched across requests)     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Loop:                                                     │  │
│  │    1. model.forward(last_token, cache=kv_cache)            │  │
│  │    2. sample(logits, temperature, top_p, top_k)            │  │
│  │    3. Check EOS / stop strings / max_tokens                │  │
│  │    4. Yield token to streaming pipeline                    │  │
│  │                                                            │  │
│  │  On finish:                                                │  │
│  │    5. TurboQuant: COMPRESS (float → 3-bit, 5x reduction)  │  │
│  │    6. Extract cache via .state (back to float for storage) │  │
│  │    7. Store to paged cache / disk                          │  │
│  │    8. mx.clear_cache() if batch empty                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Streaming Output Pipeline

```
Engine yields token
       │
       ▼
┌──────────────────────────────────────────────────────┐
│           stream_chat_completion()                   │
│                                                      │
│  ┌────────────────────┐                              │
│  │ Reasoning Parser   │  <think>...</think>          │
│  │ (qwen3/deepseek/   │  [THINK]...[/THINK]         │
│  │  mistral/gptoss)   │  <analysis>...</analysis>    │
│  └────────┬───────────┘                              │
│           │                                          │
│           ├── reasoning_content → SSE chunk           │
│           └── content → SSE chunk                    │
│                                                      │
│  ┌────────────────────┐                              │
│  │ Tool Call Detector  │  Buffers on marker detect   │
│  │ (qwen/llama/       │  Parses at end of stream    │
│  │  mistral/hermes..) │  Emits tool_calls chunks    │
│  └────────────────────┘                              │
│                                                      │
│  Keep-alive: SSE comments during long prefills       │
│  Usage: prompt_tokens + completion_tokens at end     │
│  [DONE] sentinel                                     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
        SSE to client
   data: {"choices":[{"delta":{"content":"Hello"}}]}
```

## Model Loading

```
CLI args / Panel UI
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                   load_model()                       │
│                                                      │
│  1. Detect model type                                │
│     ├── is_jang_model() → JANG v2 loader (instant)   │
│     └── standard → mlx_lm.load() / mlx_vlm.load()   │
│                                                      │
│  2. Apply TurboQuant                                 │
│     ├── JANG: _patch_turboquant_make_cache()         │
│     └── MLX:  _apply_turboquant_to_model()           │
│                                                      │
│  3. Detect model features                            │
│     ├── Model config registry lookup                 │
│     ├── Hybrid SSM detection (KV + Mamba layers)     │
│     ├── MLA detection (kv_lora_rank > 0)             │
│     └── VLM detection (vision_config in config.json) │
│                                                      │
│  4. Configure scheduler                              │
│     ├── Paged cache vs memory-aware vs legacy         │
│     ├── KV quant (disabled for MLA)                  │
│     ├── Auto-switch to paged for hybrid SSM          │
│     └── SSM companion cache init                     │
│                                                      │
│  5. Apply overrides                                  │
│     ├── Custom chat template (registry or --flag)     │
│     ├── Tool parser auto-detection                   │
│     ├── Reasoning parser auto-detection              │
│     └── Max prompt token limit estimation            │
└──────────────────────────────────────────────────────┘
```

## Model Type Compatibility Matrix

```
                    Paged   Memory   Disk   SSM      Turbo    KV
Model Type          Cache   Cache    L2     Companion Quant   Quant
─────────────────── ─────── ──────── ────── ──────── ──────── ──────
Standard LLM         YES     YES     YES*    N/A      YES     YES
MoE (DeepSeek,MM)    YES     YES     YES*    N/A      YES     YES
Hybrid SSM           YES     SKIP    YES*    YES      attn    YES
VLM text-only        YES     SKIP    YES*    YES      attn    YES
VLM with images      SKIP    SKIP    SKIP    N/A      N/A     N/A
MLA (Mistral 4)      YES     YES     YES*    N/A      YES     SKIP

* Disk L2 Python engine working, UI flags disabled for v1.3.x
  SKIP = intentionally skipped (not compatible)
  attn = TQ only on attention layers, SSM layers unchanged
```

## Memory Lifecycle

```
IDLE:        Model weights only (19GB for Qwen3.5-VL-35B JANG 4-bit)
                │
PREFILL:     + Float KV allocation (full precision, temp spike)
                │
COMPRESS:    TurboQuant compresses KV to 3-bit (~5x reduction)
                │
GENERATE:    Model weights + compressed KV + small float window
                │
STORE:       .state extracts float → paged cache blocks (float)
                │
BATCH EMPTY: mx.clear_cache() → Metal frees allocator buffers
                │
IDLE:        Back to model weights only
                │
SOFT SLEEP:  Clear all caches → back to absolute baseline
```

## File Map

```
vmlx_engine/
├── server.py              API endpoints, streaming, adapters
├── cli.py                 CLI argument parsing, startup
├── engine/
│   ├── base.py            Engine interface
│   ├── batched.py         BatchedEngine (continuous batching)
│   └── simple.py          SimpleEngine (single request)
├── scheduler.py           LLM request scheduling + cache mgmt
├── mllm_scheduler.py      MLLM (VLM) scheduling
├── mllm_batch_generator.py  Batch prefill + decode + SSM companion
├── paged_cache.py         Block-based paged KV cache
├── memory_cache.py        Memory-aware LRU prefix cache
├── prefix_cache.py        Legacy trie-based prefix cache
├── disk_cache.py          L2 disk persistence (safetensors)
├── block_disk_store.py    Per-block disk persistence
├── reasoning/             Reasoning parsers (qwen3, deepseek, mistral, gptoss)
├── tool_parsers/          Tool call parsers (12+ implementations)
├── api/
│   ├── models.py          Request/response Pydantic models
│   ├── utils.py           is_mllm_model(), resolve_to_local_path()
│   ├── anthropic_adapter.py  Anthropic ↔ OpenAI conversion
│   └── tool_calling.py    Tool call parsing utilities
├── models/
│   └── mllm.py            MLLMEngine (vision model wrapper)
├── utils/
│   ├── jang_loader.py     JANG v2 instant model loader
│   ├── tokenizer.py       Tokenizer loading + TQ patching
│   ├── cache_types.py     Cache type registry
│   └── mamba_cache.py     BatchMambaCache for hybrid SSM
├── model_configs.py       90+ model family configurations
├── model_config_registry.py  Model detection + lookup
└── image_gen.py           Flux/mflux image generation
```
