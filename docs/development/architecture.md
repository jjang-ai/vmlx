# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                     vLLM API Layer                        │
│  (OpenAI-compatible: chat, completions, embeddings,      │
│   audio, tools, MCP, reasoning)                          │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      MLXPlatform                          │
│       (vLLM platform plugin for Apple Silicon)           │
└──────────────────────────────────────────────────────────┘
                           │
       ┌──────────┬────────┴────────┬──────────┐
       ▼          ▼                 ▼          ▼
┌───────────┐┌───────────┐┌─────────────┐┌──────────────┐
│  mlx-lm   ││  mlx-vlm  ││  mlx-audio  ││mlx-embeddings│
│  (LLM)    ││  (Vision) ││  (STT/TTS)  ││ (Embeddings) │
└───────────┘└───────────┘└─────────────┘└──────────────┘
       │          │                 │          │
       └──────────┴────────┬────────┴──────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                         MLX                               │
│          (Apple ML Framework - Metal kernels)            │
└──────────────────────────────────────────────────────────┘
```

## Engine Architecture

### Simple Engine
- Direct mlx-lm/mlx-vlm wrapper
- Maximum throughput for single user
- Zero batching overhead

### Batched Engine
- AsyncEngineCore with continuous batching
- Multiple concurrent requests
- Scheduler with priority queue

## Paged KV Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedCacheManager                          │
├─────────────────────────────────────────────────────────────────┤
│  FreeKVCacheBlockQueue     │  BlockHashToBlockMap               │
│  (O(1) doubly linked list) │  (hash → block for prefix caching) │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │  {hash_0: block_5}                 │
│  │ 3 │↔│ 7 │↔│ 2 │↔│ 9 │  │  {hash_1: block_12}                │
│  └───┘ └───┘ └───┘ └───┘  │  {hash_2: block_5}  (shared!)      │
│   LRU ───────────▶ MRU    │                                     │
├─────────────────────────────────────────────────────────────────┤
│  CacheBlock[0..N]:                                              │
│  - block_id, ref_count, block_hash                              │
│  - prev_free_block, next_free_block (doubly linked)             │
│  - cache_data: List[(keys, values)] per layer                   │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Flow

```
Request Completion                    Cache Storage
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────┐
│ response.cache() │ ───────────▶ │ Extract .state      │
│ (KVCache objects)│              │ (keys, values)      │
└──────────────────┘              └─────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ Slice into 64-token │
                                  │ blocks + chain hash │
                                  └─────────────────────┘
                                            │
       New Request                          ▼
       │                          ┌─────────────────────┐
       ▼                          │ BlockHashToBlockMap │
┌──────────────────┐              │ deduplicate & share │
│ compute_block_   │ ◀─────────── └─────────────────────┘
│ hash(parent, tok)│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Reconstruct via  │
│ mx.concatenate() │
│ + KVCache.from_  │
│ state()          │
└──────────────────┘
```

## Key Features

| Feature | Benefit |
|---------|---------|
| **1.14x Speedup** | Faster inference by reusing cached KV computations |
| **80% Memory Savings** | Share system prompt blocks across concurrent users |
| **vLLM Architecture** | FreeKVCacheBlockQueue, BlockHashToBlockMap, chain hashing |
| **Real Tensor Storage** | Extracts actual KV data using `.state` |
| **Block Deduplication** | Hash-based detection prevents duplicate storage |
| **Copy-on-Write (COW)** | Shared blocks only copied when modified |
| **O(1) LRU Eviction** | Doubly linked list for efficient cleanup |

## Module Structure

```
vmlx_engine/
├── api/
│   ├── models.py         # Pydantic models
│   ├── utils.py          # Shared utilities
│   ├── streaming.py      # Streaming JSON encoder
│   └── tool_calling.py   # Tool call parsing
├── audio/
│   ├── processor.py      # Audio preprocessing
│   ├── stt.py            # Speech-to-Text
│   └── tts.py            # Text-to-Speech
├── engine/
│   ├── base.py           # BaseEngine ABC
│   ├── simple.py         # SimpleEngine
│   └── batched.py        # BatchedEngine
├── mcp/
│   ├── client.py         # MCP client
│   ├── config.py         # Config loading
│   ├── executor.py       # Tool execution
│   ├── security.py       # Command validation
│   ├── tools.py          # Tool sandbox
│   └── manager.py        # Server management
├── models/
│   ├── llm.py            # MLXLanguageModel
│   └── mllm.py           # MLXMultimodalLM
├── tool_parsers/         # Tool call parsers (13 formats)
├── reasoning/            # Reasoning parsers (qwen3, deepseek_r1, openai_gptoss)
├── server.py             # FastAPI server
├── engine_core.py        # AsyncEngineCore
├── scheduler.py          # LLM request scheduler
├── mllm_scheduler.py     # MLLM request scheduler
├── mllm_batch_generator.py # MLLM batch generation
├── paged_cache.py        # Paged KV cache
├── prefix_cache.py       # Prefix cache manager
├── output_collector.py   # Request output collector
├── model_registry.py     # Model detection & registry
├── model_config_registry.py # Model config registry (parsers, cache types, etc.)
├── model_configs.py      # Model family configurations
├── vision_embedding_cache.py # Vision preprocessing cache (LRU, order-sensitive hashing)
└── cli.py                # CLI commands
```

## Request Flow

1. **API Request** → FastAPI endpoint (auth, rate limit)
2. **Engine Selection** → Simple or Batched based on config
3. **Sampling Resolution** → Request params > CLI flags > model defaults (from `generation_config.json`)
4. **Template Application** → Chat template formatting (with tool definitions if enabled)
5. **Generation** → mlx-lm, mlx-vlm, mlx-audio, or mlx-embeddings
6. **Post-processing** → Tool call parsing, reasoning extraction
7. **Streaming** → SSE response chunks
8. **Caching** → KV cache storage for reuse

## Sampling Parameter Resolution

Sampling parameters (temperature, top_p, top_k, min_p, repetition_penalty) follow a priority chain:

1. **Request value** — Explicit parameter in API request body
2. **CLI flag** — Server launch argument (e.g., `--temperature 0.7`)
3. **Model default** — From `generation_config.json` in the model folder (read at chat creation in the panel app)

The panel app reads `generation_config.json` via `readGenerationDefaults()` and stores values in `chat_overrides` (SQLite). These are sent in every API request body.

## MLLM Batched Sampling

When continuous batching is enabled for multimodal (VLM) models, the `MLLMScheduler` uses per-request sampling parameters from the first waiting request. Each `MLLMBatchRequest` carries its own `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty`. The batch generator is recreated when sampling parameters change between batches.

### MLLM Stop Sequences

String stop sequences (e.g., `["###", "<|end|>"]`) flow through the full MLLM path:

1. **API** → `server.py` resolves `stop` from request
2. **Engine** → `batched.py` passes `stop=stop` to `add_request_async()`
3. **Scheduler** → `mllm_scheduler.add_request()` stores stop in `SamplingParams`
4. **Post-decode** → `_process_batch_responses()` checks decoded text against stop strings
5. **Truncation** → Output text truncated at stop match, finish_reason set to "stop"
6. **Cleanup** → Batch generator removes the request to stop further generation

### MLLM Paged Cache Storage

VLM paged cache follows the same N-1 token truncation as LLM:

1. On request completion, `_extracted_cache()` returns raw KVCache from batch generator
2. Cache is truncated to `prompt_len - 1` tokens (so last token can be re-fed on cache hit)
3. Quantized (if KV quant enabled) via `_quantize_cache_for_storage()`
4. Converted to state-dict format via `_extract_cache_states()`
5. Stored in `BlockAwarePrefixCache` with truncated token key

### Chat Template Fallback

When `apply_chat_template()` fails or silently drops tool definitions:

1. **Progressive stripping** — removes kwargs one by one (last-added first)
2. Preserves `tools` and `enable_thinking` as long as possible
3. Only strips essential kwargs as a last resort

### Tool Fallback Injection

Some models (e.g., Qwen 3.5 with `enable_thinking=False`) have chat templates that
silently drop tool schemas from the rendered prompt. `check_and_inject_fallback_tools()`
in `api/tool_calling.py` detects this and injects a standard XML `<tool_call>` instruction
set into the system message. This works for **all models**, not just Qwen:

1. After rendering the prompt, check if the first tool name appears in it
2. If missing → inject tool definitions as XML schema into system message
3. Re-apply template with modified messages (tools removed from kwargs)
4. Both `SimpleEngine` and `BatchedEngine` call this after every template application

## Settings Dependencies

Feature activation follows a dependency chain:

```
Continuous Batching → Prefix Cache → Paged Cache
                                   → KV Quantization
                                   → Disk Cache
```

- **Paged cache** requires prefix cache ON
- **KV quantization** requires continuous batching + prefix cache ON
- **Disk cache** requires continuous batching + prefix cache ON (and is mutually exclusive with paged cache)

## Stop/Cancel Architecture

Two-pronged cancellation:

1. **TCP abort** — `AbortController.abort()` kills the SSE connection immediately
2. **Engine abort** — POST to `/cancel` endpoint calls `engine.abort_request(request_id)`
   - SimpleEngine: Sets `_abort_requested = True`, checked between tokens
   - BatchedEngine: Scheduler removes request from GPU batch immediately
   - MLLM: Signals asyncio Queue with `None` for immediate unblock

## Streaming Persistence

When user navigates away from a chat:
- Generation continues in the background (useEffect cleanup does NOT abort)
- Periodic 5-second DB saves preserve in-progress content
- When user returns, messages are loaded from DB with latest content

## Hardware Detection

vmlx-engine auto-detects Apple Silicon:
- Chip name (M1, M2, M3, M4)
- Total memory
- Neural engine cores
- GPU cores

```python
from vmlx_engine.hardware import get_hardware_info

hw = get_hardware_info()
print(f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)")
```
