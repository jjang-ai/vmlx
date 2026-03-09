# Changelog

All notable changes to vMLX Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.18] - 2026-03-09

### Fixed
- **Stop token cleanup on abort**: Per-request stop tokens added to `BatchGenerator.stop_tokens` are now properly cleaned up when a request is aborted. Previously, stop tokens accumulated indefinitely, eventually causing false-positive stops for unrelated requests.
- **Ghost request time-based reaping**: Ghost requests (orphaned in the engine loop) now have a 30-second time-based fallback in addition to the existing count threshold. Prevents ghosts from stalling indefinitely at just below the count threshold.
- **reasoning_effort dead code**: Removed impossible `if _ct_kwargs is None:` guard in both Chat Completions and Responses API streaming paths. `_ct_kwargs` is always `{}` from `request.chat_template_kwargs or {}`.
- **Disk cache flag without paged cache**: `--enable-block-disk-cache` without `--use-paged-cache` now correctly disables disk cache instead of just warning.
- **Module-level `re` import in scheduler**: Moved `import re` out of per-token hot loop in string stop token matching to module-level imports.
- **CancelledError SSE hang**: Engine loop `CancelledError` handler now calls `_fail_active_requests()` to unblock waiting SSE consumers. Previously, cancellation left active streams hanging.
- **Paged cache block leak on abort**: `abort_request` now uses `delete_block_table()` (decrements ref_counts) instead of `detach_request()` (preserves for LRU). Aborted requests don't enter prefix cache, so `detach` would orphan blocks with permanently elevated ref_counts. Fixed in both `Scheduler` and `MLLMScheduler` (abort + error-recovery paths).
- **VLM disk cache key mismatch**: Disk cache store used `truncated_tokens` (N-1) but fetch used `token_list` (N), causing 100% cache miss rate. Now both use `token_list`.
- **KV dequantize None crash**: `_dequantize_cache()` can return `None` on failure but 3 callers didn't guard against it — passing `None` to `_fix_hybrid_cache` or assigning it as `req.prompt_cache` with truncated `input_ids`. All callers now check for `None` and fall back to full prefill.
- **Reasoning trailing window false positives**: Tool call marker detection searched the full `accumulated_reasoning` buffer, triggering false positives when earlier reasoning text discussed tool syntax. Now uses a 30-char trailing window.
- **DeepSeek Unicode tool markers**: Added `<\uff5ctool\u2581calls\u2581begin\uff5c>` (Unicode fullwidth/block chars) to `_TOOL_CALL_MARKERS` for DeepSeek model variants.
- **GPT-OSS fallback threshold too high**: `_FALLBACK_THRESHOLD` reduced from 10 to 3 chars. Previously, short non-Harmony responses ("Hi", "Yes", "OK") were swallowed.
- **GPT-OSS strip_partial_marker too aggressive**: Minimum partial match length raised from 3 to 5 chars. Previously, common Python keywords ("class", "pass") were falsely stripped.
- **Tool fallback first-tool-only check**: `check_and_inject_fallback_tools()` now verifies ALL tool names are in the prompt (using `all()`), not just the first one. Templates that only rendered some tools went undetected.
- **Mistral tool parser invalid JSON**: New-format tool calls now validated with `json.loads()` — malformed JSON arguments are rejected instead of passed through.

## [0.2.12] - 2026-03-07

### Fixed
- **Critical: tool_choice="none" content swallowing in streaming**: Chat Completions streaming path did not gate `tool_call_active` by `_suppress_tools`, causing content to be silently buffered/swallowed when tool markers were detected despite `tool_choice="none"`. Now correctly disables tool buffering.
- **suppress_reasoning leaks in Responses API**: `response.reasoning.done` event was emitted even when `suppress_reasoning=True`. Reasoning-only fallback in Chat Completions also leaked reasoning as content when suppressed.
- **Non-streaming tool_choice="none"**: Both non-streaming Chat Completions and Responses API paths now skip tool parsing when `tool_choice="none"`.
- **PagedCacheManager crash on invalid input**: `block_size=0` caused `ZeroDivisionError`, `max_blocks<2` caused silent failures. Now raises `ValueError` with clear messages.
- **Hybrid detection silent failures**: `_is_hybrid_model` now logs warnings when `make_cache()` raises instead of silently swallowing exceptions.
- **Memory cache 0-memory fallback**: `compute_memory_limit` now logs a warning when psutil returns 0 bytes, explaining the 8GB fallback assumption.

### Improved
- **First-launch UX**: Auto-creates initial chat for new users instead of showing empty state. Skips `detectConfig` for remote sessions (no local model path to inspect).
- **About page**: App version now reads dynamically via IPC instead of hardcoded. Correct website and GitHub links.

## [0.2.11] - 2026-03-07

### Fixed
- **Hybrid VLM paged cache OOM crash**: Hybrid models (Qwen3.5-VL, Jamba-VL) with paged cache crashed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` after a few requests. Root cause: `fetch_cache()` incremented block ref_counts, but when hybrid models couldn't use the cached KV blocks (missing companion SSM state), the refs were never decremented — blocks accumulated until Metal GPU memory was exhausted. Fix: check SSM state BEFORE `reconstruct_cache()`, call `release_cache()` to decrement block refs when SSM state is missing, then `continue` to skip reconstruction and do full prefill instead. Added 5 new tests in `test_hybrid_batching.py`.
- **Suppress reasoning drops thinking entirely**: When reasoning is toggled off for always-thinking models (MiniMax M2.5, Prism Pro), thinking text is now fully hidden instead of being redirected as visible content. Users see a brief pause then only the final answer.
- **Deprecated MLX API calls**: Replaced `mx.metal.device_info()` → `mx.device_info()` and `mx.metal.set_cache_limit()` → `mx.set_cache_limit()` with backward-compatible fallbacks for older MLX versions.

## [0.2.10] - 2026-03-06

### Fixed
- **Reasoning parser for always-thinking models**: Fixed `effective_think_in_template` being unconditionally set to `False` when user disables reasoning. For models whose templates always inject `<think>` (MiniMax M2.5, Prism Pro), the parser now stays in implicit reasoning mode so it correctly classifies reasoning vs content. The `suppress_reasoning` flag handles hiding reasoning from the user. Fixed in both Chat Completions and Responses API paths.

### Improved
- **Parser dropdown UI**: Reasoning and tool parser dropdown labels now include model names directly (e.g., "Qwen3 — Qwen / QwQ / MiniMax / StepFun"). Help panel auto-opens when a manual parser is selected. More comprehensive model compatibility lists. Auto-detect labels say "(recommended)".

## [0.2.9] - 2026-03-05

### Added

#### Speculative Decoding (Phase 3 — vllm-metal Feature Integration)
- **New module**: `speculative.py` — Speculative decoding using mlx-lm's native `speculative_generate_step()`
  - `SpeculativeConfig` dataclass with model, num_tokens, disable_by_batch_size
  - `load_draft_model()` / `unload_draft_model()` lifecycle management
  - `get_spec_stats()` for API health endpoint integration
  - `is_speculative_enabled()` global state check
- **CLI flags**:
  - `--speculative-model` — path/name of draft model (same tokenizer required)
  - `--num-draft-tokens` — tokens drafted per step (default: 3, sweet spot 2-5)
- **How it works**: Draft model proposes N tokens → target model verifies all N in a single forward pass → accepted tokens skip individual decode → 20-90% throughput improvement with zero quality loss
- **Server integration**: `/health` endpoint now reports `speculative_decoding` status (enabled, draft_model, num_draft_tokens, draft_model_loaded)
- **Startup banner**: Speculative decoding status displayed in security/feature summary
- **Test suite**: 21 new tests in `tests/test_speculative.py`:
  - Config validation (defaults, clamping, warnings)
  - Global state lifecycle (load/unload/enable check)
  - Stats reporting (not configured, partial, fully loaded)
  - Error handling (invalid model → ValueError, auto-disable)
  - CLI argument parsing
  - mlx-lm integration (speculative_generate_step importable, stream_generate accepts draft_model)
  - Server health endpoint integration

### Phase 1 Status: RotatingKVCache
- **Already implemented** — confirmed RotatingKVCache support across: `mllm_batch_generator.py`, `scheduler.py`, `prefix_cache.py`, `disk_cache.py`, `memory_cache.py`, `utils/mamba_cache.py`, `utils/cache_types.py`

## [0.2.8] - 2026-03-03

### Fixed

#### Multi-Turn VLM 0-Token Output (Critical)
- **Root cause**: `model_dump()` without `exclude_none=True` included `image_url=None` on text ContentParts. Jinja2 templates check key existence (`'image_url' in item`) which returned True even for None values, causing Qwen3VLProcessor to count 2× image_pad tokens for 1 image → IndexError → fallback to PyTorch → crash.
- **Fix**: All three Pydantic-to-dict conversion paths now use `model_dump(exclude_none=True)`:
  - `server.py` Chat Completions MLLM path
  - `server.py` Responses API `_resolve_content()`
  - `mllm.py` content extraction (defense-in-depth)
- **Fix**: `batched.py` MLLM single-turn path now passes `extra_template_kwargs` (enable_thinking, reasoning_effort)

#### Hybrid Cache Mismatch Returns Corrupt Cache (Critical)
- **Root cause**: `_fix_hybrid_cache()` returned the short (attention-only) reconstructed cache when its length didn't match expected KV positions, instead of a fresh full-length cache from `make_cache()`. This gave the model a cache with wrong layer count.
- **Fix**: Both mismatch paths now return `language_model.make_cache()` (fresh full cache) or `template` (from already-called make_cache).

#### SimpleEngine MLLM Drops Reasoning Kwargs
- **Root cause**: SimpleEngine `chat()` and `stream_generate()` MLLM paths only forwarded `enable_thinking`, silently dropping `reasoning_effort` and `chat_template_kwargs`.
- **Fix**: Both paths now forward all three kwargs to the underlying mlx-vlm model.

#### Miscellaneous
- Removed hardcoded model name from `_template_always_thinks()` — now uses only dynamic template testing
- `make_cache()` failure in MLLMBatchGenerator init now logs warning instead of bare `except: pass`

### Added

#### Comprehensive Test Suite Expansion
- **64 MLLM serialization tests** (`test_mllm_message_serialization.py`): model_dump behavior, Jinja2 key-existence simulation, multi-turn image counting, _fix_hybrid_cache correctness, SimpleEngine kwargs forwarding, mllm.py model_dump paths
- **89 model config registry tests** (`test_model_config_registry.py`): Every model family's tool parser, reasoning parser, cache type, MLLM flag, think_in_template, and priority — plus cross-family consistency checks (valid parsers, no duplicate model_types, priority ordering, MLLM completeness)
- **Total engine test suite**: 1295+ tests passing (up from 1237)

## [0.2.7] - 2026-03-02

### Fixed

#### Continuous Batching Stability (2026-03-02)

- **Continuous Batching Thread Safety**: Added `threading.RLock()` to protect queue mutations across asynchronous loops and sync `MLLM` vision tasks running over background threads (`step`, `add_request`, `abort_request`). Resolves latent data race failures under heavy loads.
- **Bounded Queues**: Fixed unbounded growth mapping of the stream generation output by explicitly setting max size values (`asyncio.Queue(maxsize=8192)`). Ensures memory safety during downstream socket unresponsiveness scaling.
- **Ghost Abort Subsystem**: Fast-tracked the `_ghost_check_counter` interval from checking every 500 loops to 50 loops on the core Engine allowing rapid recycling of broken API memory references for stability endpoints.
- **Batched Engine Rescheduling Safety**: Gracefully intercepted GPU-metal-level corruption traps within generation steps by ensuring requests accurately respool via `retryable` queue structures dropping erroring chunk pointers automatically without completely abandoning API sessions. 

#### Mamba & SSM Native Paged Routing (2026-03-02)
- **Automatic Multi-Array Cache Re-Routing**: Intercepts `model.make_cache()` structure arrays matching hybrid combinations (`MambaCache` alongside `KVCache`) natively detecting standard LLM truncations violations. Auto switches memory parameters inside the `Scheduler` to natively fall back to compatible Paged and Legacy parameters gracefully to prevent sequence faults.

### Fixed

#### Reasoning Content Leaking as Visible Text During Tool Calls (2026-03-02)
- **Reasoning leak on tool follow-ups**: When using agentic tool calling with thinking models (Qwen3, Qwen3.5-VL), reasoning text leaked into visible content on follow-up requests after tool execution. Root cause: `effective_think_in_template` was forced to `False` when tool results were present, breaking the reasoning parser's `think_in_prompt` state. Fix: keep `think_in_prompt=True` — the parser's streaming extraction handles `<think>`→`</think>`→content transitions correctly regardless of tool results.
- **Duplicate content when reasoning disabled**: When reasoning was turned off (`enable_thinking=False`) but the model still produced reasoning text, content appeared twice. Root cause: end-of-stream tool call extraction re-emitted `cleaned_text` that was already streamed (either as content or as redirected reasoning). Fix: track `accumulated_content` during streaming and subtract already-emitted content from the final emission.
- **False-positive tool call buffer flush**: When tool call markers were detected but no actual tool calls were parsed, the entire `accumulated_text` was flushed as content — including text that was already streamed. Fix: only flush the un-streamed portion.
- **Responses API `enable_thinking` guard**: Added missing `_effective_thinking is False` guard to Responses API streaming path for parity with Chat Completions.
- **Tool fallback injection for broken templates**: Some model chat templates silently drop tool schemas when `enable_thinking=False` (e.g., Qwen 3.5 family). Added `check_and_inject_fallback_tools()` that detects when tools are missing from the rendered prompt and injects a standard XML `<tool_call>` instruction set into the system message. Works for all models — not just Qwen.

#### Integrated Tool Call System — Deep Audit & Fixes (2026-03-02)

- **Responses API `tool_choice` handling**: The Responses API endpoint now fully mirrors the Chat Completions handler — `tool_choice="none"` suppresses all tools, `tool_choice={"function":{"name":"X"}}` filters to the named tool only. Previously the Responses API ignored `tool_choice` entirely.
- **Responses API `suppress_reasoning` parity**: When the client sets `enable_thinking=False` but the model forces reasoning (e.g., MiniMax), the Responses API streaming path now redirects suppressed reasoning as content (matching Chat Completions behavior). Previously it silently dropped reasoning deltas, causing the stream to appear to hang.
- **Responses API JSON schema validation**: The non-streaming Responses API path now validates output against `json_schema` with `strict=True` and returns HTTP 400 on validation failure, matching the Chat Completions behavior. Previously it only prompt-injected JSON instructions with no post-generation validation.
- **`gitCommand` shell injection prevention**: Added shell metacharacter blocking (`;|&`$(){}`) to prevent command injection via `/bin/sh -c`. Dangerous git operations (`push --force`, `reset --hard`, `clean -f`, `branch -D`) already blocked.
- **`run_command` kill reason accuracy**: Added `!killReason` guards on all three kill paths (stdout overflow, stderr overflow, timeout) so only the first reason is preserved. Previously, a second overflow event could overwrite the original reason.
- **`ask_user` always available**: Moved `ask_user` out of `UTILITY_TOOLS` category so it cannot be accidentally disabled by the `utilityToolsEnabled: false` toggle. It's a core IPC tool that should always be available.
- **`insertText` / `replaceLines` null guards**: Added parameter validation to prevent silent corruption (NaN splice index, TypeError on undefined text).
- **`fetchUrl` truncation reporting**: Fixed truncation footer to show original content length instead of the truncated length.
- **`batchEdit` conditional write**: Only writes the file when at least one edit succeeded (prevents unnecessary mtime updates on all-fail).
- **`get_diagnostics` dead code removal**: Removed dead TSC single-file branch. Fixed schema mismatch (`path` was marked required but is optional).

#### Comprehensive Test Suite Expansion
- Added `tests/test_tool_format.py` with 54 new tests covering:
  - `ResponsesToolDefinition.to_chat_completions_format()` conversion
  - `convert_tools_for_template()` with all input formats
  - `tool_choice` suppression and filtering for both APIs
  - `response_format.strict` enforcement
  - `max_tokens` fallback chain behavior
  - Model config registry flags (parser assignments, is_mllm, native tool format)
  - Audio model defaults and settings
  - `_responses_input_to_messages()` conversion (string, list, multimodal)
  - `ToolDefinition` Pydantic model edge cases

### Added

#### VLM Caching Pipeline (Pioneer MLX Feature)
- **Paged KV Cache for VLMs**: Full integration of `PagedCacheManager` + `BlockAwarePrefixCache` into the MLLM scheduler for Vision-Language Models
- **Prefix Cache for VLMs**: Token-level prefix matching and cache reuse across VLM requests — stores KV blocks after generation, retrieves on subsequent requests with shared prompt prefixes
- **KV Cache Quantization for VLMs (Q4/Q8)**: Quantized KV cache storage in prefix cache, reducing VLM cache memory by 2-4x
  - Init-time head_dim validation, group_size auto-adjustment, and round-trip testing
  - Quantize on store (`_quantize_cache_for_storage`), dequantize on fetch (`_dequantize_cache_for_use`)
- **Config Propagation**: `SchedulerConfig` cache settings (`enable_prefix_cache`, `use_paged_cache`, `kv_cache_quantization`, `kv_cache_group_size`, etc.) now properly forwarded to `MLLMSchedulerConfig` via `batched.py`
- **VLM Cache Architecture**: Per-request prefill uses standard `KVCache` (integer offsets), then converts to `BatchKVCache` (mx.array offsets) for batched autoregressive decode — bridging `mlx_lm` and `mlx_vlm` cache expectations
- **Mamba Hybrid VLM Support**: Auto-detects VLMs with mixed KVCache + MambaCache/ArraysCache layers (Jamba-VL, VLM-Mamba, MaTVLM). Uses `model.make_cache()` for correct per-layer cache types, `BatchMambaCache` for batched decode, auto-switches to paged cache for Mamba models

### Fixed

#### VLM Cache Crash: `'list' object has no attribute 'offset'` (Critical)
- **Root Cause**: `BlockAwarePrefixCache.fetch_cache()` returns `(block_table, remaining_tokens)` tuple, but code was assigning this raw tuple as the model cache — passed to every decoder layer as `c` in `zip(layers, cache)`
- **Fix**: Proper tuple unpacking + `reconstruct_cache()` on cache hits
- **Additional Fix**: Removed broken `BatchKVCache.offset` monkey-patch (was preventing `BatchKVCache.__init__` from setting its own offset attribute)

#### VLM Cache Merge: `Slice indices must be integers or None`
- **Root Cause**: Per-request VLM prefill used `BatchKVCache` (which has `mx.array` offsets), but `BatchKVCache.merge()` internally uses `.offset` as a Python slice index
- **Fix**: Changed prefill to use standard `KVCache` (integer offsets), then convert to `BatchKVCache` after prefill for batched decode

#### GLM-4.7 / GPT-OSS Harmony Protocol Support
- **GLM-4.7 Flash and GLM-4.7** now use `openai_gptoss` reasoning parser (Harmony protocol: `<|channel|>analysis/final`)
- Previously mapped to `deepseek_r1` which caused leaked `<|start|>assistant<|channel|>analysis<|message|>` tokens in chat
- `think_in_template=False` for GLM Flash — uses channel markers instead of `<think>` prefix injection
- Reasoning effort selector (Low/Med/High) only appears when GPT-OSS/Harmony parser is active

#### Expanded Model Registry
- **Devstral** and **Codestral** added to both TS and Python registries (don't match `/mistral/i` by name)
- Unified GPT-OSS dropdown label: "GPT-OSS / Harmony — GLM-4.7, GLM-4.7 Flash, GLM-Z1, GPT-OSS-20B/120B"

#### Client-Side Content Cleanup
- **Harmony protocol tokens** (`<|start|>`, `<|channel|>`, `<|message|>`) added to TEMPLATE_STOP_TOKENS fallback
- **Hallucinated tool calls** from Anthropic-trained models (`<read_file>`, `<write_file>`, `<run_command>`, etc.) stripped from content
- Both streaming buffering (line-start pattern) and final cleanup (regex) catch these patterns
- Abort path applies identical cleanup to prevent partial tool XML in saved messages

#### Bundled Python Distribution
- `panel/scripts/bundle-python.sh` creates relocatable Python 3.12 + all deps for standalone distribution
- App checks bundled Python first, falls back to system vmlx-engine binary
- Bundled spawn uses `python3 -m vmlx_engine.cli serve` (avoids shebang issues)
- Engine auto-update on startup: compares installed vs source `pyproject.toml` version
- Setup screen skipped entirely when bundled Python detected

### Fixed

#### GLM-4.7 Flash Reasoning Leak (Critical)
- GLM Flash was configured with `deepseek_r1` parser and `think_in_template=true`
- Model actually uses Harmony/GPT-OSS protocol (`<|channel|>analysis/final`), NOT `<think>` tags
- All reasoning content and raw protocol tokens leaked into visible chat output
- Fixed by switching to `openai_gptoss` parser with `think_in_template=false`

#### Reasoning Effort Visibility
- Low/Med/High reasoning effort buttons appeared for ALL models when thinking was enabled
- Only GPT-OSS/Harmony models support `reasoning_effort` parameter
- Now conditionally rendered only when `reasoningParser === 'openai_gptoss'`

### Previously Added

#### Universal Thinking/Reasoning Toggle
- **Per-chat toggle** in Chat Settings (💡 Enable Thinking checkbox) to turn reasoning on/off
- **Default: ON** — matches current behavior, models produce `<think>` blocks
- **When OFF**: `enable_thinking=False` passed to chat template; models skip reasoning for faster, direct responses
- **Pipeline**: UI toggle → `ChatOverrides` DB → request body → `ChatCompletionRequest` → server → engine `apply_chat_template`
- **Compatible models**: Qwen3, DeepSeek-R1, MiniMax M2/M2.5, GLM-4.7, StepFun, and any model with `enable_thinking` template support
- **Server override**: Streaming handler respects the toggle — when OFF, `think_in_template` is forced false (no `<think>` prefix injection)

#### MiniMax M2/M2.5 Model Support
- **New Tool Parser**: `minimax` parser for MiniMax's unique XML tool calling format (`<minimax:tool_call><invoke><parameter>`)
- **Model Config**: Registered `minimax-m2.5` (priority 5) and `minimax-m2` (priority 10) families
  - EOS token: `[e~[]` (ID 200020)
  - Reasoning: `qwen3` parser (standard `<think>` tags)
  - Native tool format: Enabled (chat template handles `role="tool"` natively)
- **Auto-Detection**: MiniMax models auto-detected by model name pattern
- **Streaming**: Added `<minimax:tool_call>` to streaming tool call markers for proper buffer-then-parse behavior
- **16 new tests**: Comprehensive parser tests covering single/multi tool calls, streaming, type conversion, think tags
- **UI**: Added `minimax` to tool parser dropdown in Session Config

#### Chat Scroll Behavior Fix
- **Issue**: Auto-scroll always yanked user to bottom during streaming, preventing scroll-up to read earlier content
- **Fix**: Added `isNearBottom` detection in `MessageList.tsx` — only auto-scrolls when user is within 100px of bottom
- **UX**: Users can now scroll up freely during streaming; scroll resumes when they return to bottom

#### Full Pipeline Audit — Verified Working
- **Qwen hybrid Mamba+KV cache**: Auto-detected in scheduler, auto-switches to paged cache; cache hits work correctly
- **Chat template application**: `apply_chat_template` passes `tools` + `enable_thinking` kwargs with `TypeError` fallback for unsupported templates
- **Tool parser auto-detect**: Model name pattern → `ModelConfigRegistry.get_tool_parser()` → correct parser (all 14 parsers verified)
- **Reasoning parser auto-detect**: Model name pattern → `ModelConfigRegistry.get_reasoning_parser()` → correct parser (`qwen3`, `deepseek_r1`)
- **API completions (non-streaming)**: `/v1/chat/completions` returns correct OpenAI-spec response format
- **API completions (streaming SSE)**: `choices[0].delta.content` + `usage` fields parsed correctly by Electron panel
- **API responses wire format**: `/v1/responses` SSE events (`response.output_text.delta`, `response.completed`) parsed correctly
- **Stop button**: `chat:abort` handler aborts SSE stream + sends `POST /v1/chat/completions/{id}/cancel` for server-side GPU release
- **Reasoning box**: `max-h-[300px] overflow-y-auto` independently scrollable; auto-expands on stream, auto-collapses 1s after done
- **Tool call display**: `ToolCallStatus` component — collapsible, grouped by tool, args+result shown on expand, not spammed
- **Loop prevention**: `MAX_TOOL_ITERATIONS` defaults to 10 (configurable via ChatSettings slider); auto-continue capped at 2 rounds
- **Agentic tool flow**: Full cycle verified — model → `tool_calls` → execute (MCP or builtin) → push results → follow-up request → stream
- **Streaming stats**: TTFT, TGS, PPS computed from SSE stream in `chat.ts:emitDelta()` — generation-only time (gaps >2s excluded)
- **EOS token handling**: Model-specific EOS tokens flow from `ModelConfig` → `MLXLanguageModel.load()` → tokenizer
- **Native tool format**: `SUPPORTS_NATIVE_TOOL_FORMAT` flag preserves `role="tool"` messages through pipeline for models that support it
- **Model selection**: Electron panel scans local filesystem for MLX models; `/v1/models` returns the loaded model
- **Session config UI**: All parser dropdowns (tool + reasoning) list all available parsers with auto-detect default

#### Request Cancellation (OpenAI-Compatible)
- **Feature**: Stop ongoing inference requests to save GPU compute
- **Endpoints**:
  - `POST /v1/chat/completions/{request_id}/cancel`
  - `POST /v1/completions/{request_id}/cancel`
- **Auto-Detection**: Automatically abort when client closes stream connection
- **Unified Request ID**: Response ID (chatcmpl-xxx) is the request ID
- **Compatibility**: Works seamlessly with exploit.bot cancel button (no frontend changes needed)
- **Documentation**: Complete API docs at `docs/api/cancellation.md`
- **Benefits**:
  - Immediate GPU compute savings when user clicks stop
  - Partial responses preserved (no data loss)
  - Works with `reader.cancel()` pattern (auto-detect)
  - Optional explicit API call for programmatic control
  - < 10ms cancel latency

### Fixed

#### Streaming Unicode Character Corruption (Critical)
- **Issue**: Emoji, CJK (Chinese/Japanese/Korean), Arabic, and other multi-byte UTF-8 characters displayed as replacement characters (`�`) during streaming responses
- **Root Cause**: Single-token decoding split multi-byte characters across tokens, producing incomplete byte sequences
- **Fix**: Integrated `StreamingDetokenizer` from mlx-lm into both `scheduler.py` and `mllm_scheduler.py`
  - Per-request detokenizer pool buffers partial characters
  - Only emits text when complete UTF-8 codepoints are assembled
  - Automatically uses optimized BPE detokenizer when available
  - Falls back to `NaiveStreamingDetokenizer` for compatibility
- **Impact**: All streaming clients (vMLX panel, OpenAI SDK, curl) now correctly display multi-byte characters
- **Verification**: 827 tests passing, extensive live server testing with emoji/CJK/Arabic confirmed clean output

#### Hybrid Model Cache Reconstruction (Qwen3-Coder-Next, Nemotron)
- **Issue**: Models with mixed cache types (MambaCache + KVCache) produced null/empty content on cache hits
- **Root Cause**: Two issues:
  1. **KV duplication**: Storing N tokens but re-feeding last token created duplicate with wrong positional encoding
  2. **MambaCache state mismatch**: Cumulative state from post-generation included output tokens
- **Fix**:
  - **N-1 truncation**: Cache stores N-1 tokens so last prompt token can be re-fed for generation kickoff
  - **Prefill-only forward pass**: For hybrid models, runs `model(prompt[:-1])` separately to get clean cache state
  - **Auto-detection**: Hybrid models automatically switch to paged cache (MambaCache can't be truncated)
  - **Cache-hit skip optimization**: Skips redundant prefill on repeated prompts
- **Files Modified**:
  - `scheduler.py`: Added `_is_hybrid`, `_prefill_for_prompt_only_cache()`, modified cache extraction
  - `prefix_cache.py`: Block hashing uses FULL prompt (N tokens), cache DATA has N-1 tokens
  - `paged_cache.py`: Partial block matching for short prompts
- **Impact**: Cache reuse works correctly for all model architectures (pure KVCache, RotatingKVCache, hybrid MambaCache+KVCache)

#### Memory-Aware Cache System
- **Issue**: Cache eviction needed better memory management for large contexts (100k+ tokens)
- **Fixes**:
  - `cache_memory_percent` set to 30% of available RAM (was hardcoded limits)
  - Per-entry size limit is 95% of max_memory (prevents single-entry domination)
  - `_evict_lru()` no longer calls `gc.collect()`/`mx.clear_memory_cache()` during eviction loop
  - Store path removed `mx.clear_memory_cache()` to avoid GPU operation interference
  - Scheduler guards `mx.clear_memory_cache()` with `not self.running` check
  - Memory-aware cache stores raw KVCache object references (not extracted dicts)
- **Impact**: Stable memory usage for long-running servers with large context windows

#### Metal GPU Timeout Prevention
- **Issue**: macOS kills processes when GPU operations exceed ~20-30s
- **Fix**:
  - `mx.eval()` after KV concatenation in `reconstruct_cache()` materializes lazy ops
  - `BatchGenerator.prefill_step_size=2048` controls chunking (safe for Metal timeout)
  - Scheduler memory multiplier reduced to 1.5x (was 2.5x which was too conservative)
- **Impact**: Stable inference for 50K+ token contexts without GPU timeout crashes

### Added

#### Production Readiness Features
- **Streaming detokenizer pool**: Per-request UTF-8-aware token decoding
- **Comprehensive emoji support**: All emoji types verified working:
  - ✅ Basic emoji (🌟 🎯 🔥 🚀 🐍)
  - ✅ Skin tone modifiers (👋🏻 👋🏼 👋🏽 👋🏾 👋🏿)
  - ✅ Family/relationship (👨‍👩‍👧‍👦 👨‍👨‍👦)
  - ✅ Flag emoji (🇺🇸 🇬🇧 🇯🇵 🇧🇷 🇮🇳)
  - ✅ ZWJ sequences (🏳️‍🌈 👩‍💻 👨‍🚀 🧑‍⚕️)
  - ✅ High codepoints (🦀 🦐 🦒 🧀 🧑 🧠)
  - ✅ Ultra-high codepoints (🪐 🪑 🪒 🫀 🫁 🫂)
- **Hybrid model auto-detection**: Automatically switches to paged cache for MambaCache+KVCache models
- **Cache type detection**: Robust detection supporting all cache types:
  - Fully supported: KVCache, RotatingKVCache, MambaCache, ArraysCache, CacheList
  - Partially supported: QuantizedKVCache (detected but no BatchQuantizedKVCache)
- **Memory-aware caching**: Intelligent eviction based on RAM availability
- **Extensive test coverage**: 827 tests covering all cache types, streaming, and emoji scenarios

#### Documentation
- **MEMORY.md**: Comprehensive project memory with all cache system details
  - KV Cache tensor dimensionality handling (3D vs 4D)
  - mlx-lm BatchGenerator cache flow
  - Model config registry patterns
  - Cache system design decisions
  - Hybrid model cache architecture
  - Metal GPU timeout prevention strategies

### Changed

#### Cache Storage Strategy
- **Block hashing**: Uses FULL prompt tokens (N) for matching
- **Cache data**: Stores N-1 tokens to prevent duplication
- **Paged cache**: Default for hybrid models (auto-enabled)
- **Memory-aware cache**: Default for pure KVCache models
- **Forward prefix matching**: Works for multi-turn chat (each turn extends previous)

#### Scheduler Improvements
- **Detokenizer lifecycle**: Created on first request, cleaned up on finish, cleared on reset
- **Cache extraction**: Per-layer error handling prevents one bad layer from killing all
- **Prefill optimization**: Skips re-extraction on cache-hit requests
- **Chunked prefill**: 2048 token chunks prevent Metal GPU timeout

### Technical Details

#### Cache Key and Value Truncation
- **Store key**: Prompt tokens only (not prompt+output) for exact matching
- **Cache value**: Must be truncated to N-1 tokens before storing
  - N-1 because on cache hit, last prompt token is re-fed for generation kickoff
  - If stored at N tokens, last token's KV is duplicated with wrong positional encoding
- **Hybrid models**: `_prefill_for_prompt_only_cache(prompt[:-1])` runs separate forward pass
- **Cache-hit skip**: Blocks already exist from cold store, no redundant prefill needed

#### KV Cache Tensor Dimensionality
- **3D tensors**: `(n_kv_heads, seq, dim)` - Qwen3-Coder-Next and others
- **4D tensors**: `(batch, n_kv_heads, seq, dim)` - BatchGenerator always produces 4D
- **Detection**: Uses `ndim` check before slicing: `seq_dim = 1 if ndim == 3 else 2`
- **Concatenation**: Axis adapts: `axis=1` for 3D, `axis=2` for 4D
- **Fallback check**: `_is_positional_cache` uses `len(shape) in (3, 4)` not just `== 4`

#### Streaming Detokenizer Implementation
```python
# Per-request detokenizer pool
self._detokenizer_pool: Dict[str, Any] = {}

def _get_detokenizer(self, request_id: str) -> Any:
    if request_id not in self._detokenizer_pool:
        # Prefer tokenizer's optimized detokenizer
        if hasattr(self._actual_tokenizer, "detokenizer"):
            detok = self._actual_tokenizer.detokenizer
        else:
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
        detok.reset()
        self._detokenizer_pool[request_id] = detok
    return self._detokenizer_pool[request_id]

# In _process_batch_responses():
detok = self._get_detokenizer(request_id)
detok.add_token(response.token)
new_text = detok.last_segment  # Only emits complete UTF-8 codepoints

# On finish:
detok.finalize()
output.output_text = detok.text
```

### Compatibility

#### Model Architecture Support
- **Pure KVCache**: Llama, Mistral, Qwen (non-Next) - uses memory-aware cache
- **RotatingKVCache**: Models with sliding window attention
- **Hybrid (MambaCache + KVCache)**: Qwen3-Coder-Next (36 Mamba + 12 KV layers), Nemotron
- **ArraysCache**: Alternative cache implementations
- **CacheList**: Composite cache structures

#### Chat Template Compatibility
- All models: Native format support
- Mistral: Fixed tool calling template error with native format
- Qwen, DeepSeek, Granite, Nemotron: Added tool call parsers
- MedGemma: MLLM detection patterns updated

### Testing

#### Test Coverage
- **827 tests passing** across all modules
  - 14 comprehensive emoji tests (all categories verified)
- **Streaming detokenizer tests**: 13 tests covering emoji, CJK, Arabic, cache hits
- **Cache system tests**: All cache types (KV, RotatingKV, Mamba, Arrays, CacheList)
- **Live server tests**: Extensive emoji/unicode streaming verification
- **Integration tests**: Multi-turn conversations, system prompts, cache reuse

#### Verified Scenarios
- ✅ Emoji streaming (no replacement characters)
- ✅ CJK (Chinese/Japanese/Korean) streaming
- ✅ Arabic and RTL text streaming
- ✅ Cache hits producing correct content
- ✅ Hybrid model cache reconstruction
- ✅ Multi-turn conversations with cache
- ✅ 100k+ token contexts without GPU timeout
- ✅ Memory-aware cache eviction under pressure

## [0.2.5] - Previous Release

(Previous changelog entries would go here)
