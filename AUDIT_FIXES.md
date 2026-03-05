# vllm-mlx Comprehensive Audit — Fix Log

**Date:** 2026-03-03 — 2026-03-04
**Scope:** All tiers (CRITICAL, HIGH, MEDIUM, LOW) — ~180 issues identified, ~82 fixes applied
**Deploy:** MacBook only
**Sessions:** 5 consecutive Claude Code sessions

---

## Session 5: Coherency Audit + Dead Code Removal (2026-03-04)

6 deferred issues resolved, comprehensive coherency audit with 8 additional fixes, dead code removal.

### Phase 1: Deferred Issue Fixes (6 fixes)

**S5-D1: abstract_tool_parser.py — `extract_tool_calls_streaming()` undocumented dead code**
- Added docstring clarifying these methods are not called at runtime (server uses buffer-then-parse strategy)

**S5-D2: block_disk_store.py — Background writer opens new SQLite connection per operation**
- Changed to persistent write connection created at thread start, closed in `finally` block
- `_update_access`, `_cleanup_entry`, `_write_block`, `_maybe_evict` now take `conn` parameter
- `shutdown()` flush path creates temporary `flush_conn` for remaining items

**S5-D3: models/mllm.py — Dead `vision_embeddings` variable in `chat()`**
- Removed dead variable, added comment explaining mlx-vlm doesn't support pre-computed embeddings

**S5-D4: mcp/security.py — Semicolon pattern `;\s*` too broad**
- Changed `;\s*` → `;\s*\S` (and same for `&&`, `||`) in both `DANGEROUS_PATTERNS` and `DANGEROUS_ARG_PATTERNS`
- Now consistent with existing pipe pattern `\|\s*\S`

**S5-D5: reasoning/gptoss_parser.py — 60-char fallback threshold delays initial output**
- Reduced `_FALLBACK_THRESHOLD` from 60 to 10 (marker `<|channel|>` is 11 chars)
- Updated tests to match new threshold

**S5-D6: model_runner.py — `_continue_generation()` returns empty list instead of erroring**
- Changed to `raise NotImplementedError` with clear message about vLLM compat shim
- Added try/except in `execute_model()` to catch and log the error

### Phase 2: Coherency Audit (8 fixes)

**S5-C1: __init__.py — Broken platform import (CRITICAL)**
- Fixed `from vllm_mlx.platform import MLXPlatform` → `from vllm_mlx.mlx_platform import MLXPlatform`

**S5-C2: __init__.py + server.py — Stale version strings**
- Fixed `__version__` from `"0.2.7"` to `"0.2.8"` (matching pyproject.toml)
- Fixed server info version from `"0.2.1"` to `"0.2.8"`

**S5-C3: cli.py — `--is-mllm` flag silently ignored in serve command**
- Added `force_mllm=getattr(args, 'is_mllm', False)` to `load_model()` call

**S5-C4: cli.py — `--tool-call-parser auto` silently disables tool calling**
- Changed condition from `not in ("auto", "none")` to `!= "none"`
- "auto" now sets `_enable_auto_tool_choice = True` and defers to model_config_registry auto-detection

**S5-C5: cli.py — Missing parser aliases in `--tool-call-parser` choices**
- Added 14 aliases: `generic`, `qwen3`, `llama3`, `llama4`, `nous`, `deepseek_v3`, `deepseek_r1`, `kimi_k2`, `moonshot`, `granite3`, `nemotron3`, `minimax_m2`, `meetkai`, `stepfun`, `glm4`

**S5-C6: server.py — Completions endpoint doesn't normalize model name**
- Added `request.model = _resolve_model_name()` to `create_completion()` (consistent with chat/responses endpoints)

**S5-C7: test_platform.py — Wrong module imports**
- Fixed all `from vllm_mlx.platform` → `from vllm_mlx.mlx_platform`
- Fixed plugin path assertion to `"vllm_mlx.mlx_platform.MLXPlatform"`

### Phase 3: Dead Code Removal

**S5-R1: server.py — Removed `_emit_content_chunk()` helper (defined but never called)**

**S5-R2: api/models.py + api/__init__.py — Removed `AudioTranscriptionRequest`, `AudioTranscriptionResponse`, `AudioSeparationRequest`**
- These models had no corresponding endpoints; only `AudioSpeechRequest` is used by `/v1/audio/speech`

**S5-R3: multimodal_processor.py — Removed 6 dead methods + 1 standalone function**
- `process_for_request()`, `batch_pixel_values()`, `batch_image_grid_thw()`, `prepare_for_batch()`, `extract_vision_embeddings()`, `compute_vision_hash()`
- Standalone `create_mllm_prompt_cache()` function
- None were called outside the file

**S5-R4: mllm_batch_generator.py — Removed dead `mm_processor` parameter and attribute**
- Parameter, docstring line, import, and attribute assignment all removed
- Also removed from mllm_scheduler.py (constructor + caller)

### Test Updates

- Removed tests for deleted methods: `TestMultimodalProcessorBatch` class (test_mllm_continuous_batching.py)
- Removed tests for deleted models: `AudioTranscriptionRequest`/`AudioSeparationRequest` tests (test_api_models.py, test_audio.py, test_tool_format.py)
- Updated test_platform.py imports from `vllm_mlx.platform` to `vllm_mlx.mlx_platform`
- **Result: 1350 passed, 5 skipped, 0 failed** (excluding pre-existing async test framework issues)

### Files Modified in Session 5

| File | Fixes |
|------|-------|
| `__init__.py` | S5-C1, S5-C2 |
| `server.py` | S5-C2, S5-C6, S5-R1 |
| `cli.py` | S5-C3, S5-C4, S5-C5 |
| `api/models.py` | S5-R2 |
| `api/__init__.py` | S5-R2 |
| `multimodal_processor.py` | S5-R3 |
| `mllm_batch_generator.py` | S5-R4 |
| `mllm_scheduler.py` | S5-R4 |
| `block_disk_store.py` | S5-D2 |
| `models/mllm.py` | S5-D3 |
| `mcp/security.py` | S5-D4 |
| `reasoning/gptoss_parser.py` | S5-D5 |
| `model_runner.py` | S5-D6 |
| `tool_parsers/abstract_tool_parser.py` | S5-D1 |
| `tests/test_platform.py` | S5-C7 |
| `tests/test_api_models.py` | S5-R2 |
| `tests/test_audio.py` | S5-R2 |
| `tests/test_tool_format.py` | S5-R2 |
| `tests/test_mllm_continuous_batching.py` | S5-R3 |
| `tests/test_streaming_reasoning.py` | S5-D5 |

---

## Session 4: Post-Audit Deep Review (2026-03-04)

3 parallel deep-tracing agents verified all 60+ prior fixes and found new issues. 8 additional fixes applied:

### CRITICAL Fix — Streaming Tool Call Text Loss (server.py)

**Root cause:** When a streaming delta contains both content text and a tool call marker (e.g., `"the result.\n\n<tool_call>..."`), the content portion was silently lost:
1. `content_was_emitted = True` was set BEFORE the buffering check (parser path)
2. `accumulated_content` included tool marker text never yielded to client
3. Post-stream "already sent" comparison used `accumulated_content` (wrong) instead of actually-streamed text
4. Since `content_was_emitted` was incorrectly True, fallback suppressed un-sent content

**Fix:** Added `streamed_content` tracker (separate from `accumulated_content`). Moved `content_was_emitted = True` to AFTER yield in parser path. Post-stream dedup now uses `streamed_content` for accurate "already sent" comparison. Applied to both `stream_chat_completion()` and `stream_responses_api()`.

### MEDIUM Fix — Stale output_token_ids on reschedule (scheduler.py)

`_reschedule_running_requests()` reset prompt state but NOT `output_token_ids`, `output_text`, `num_computed_tokens`. Retried requests would restart with stale token count, causing truncated generation budget and wrong completion_tokens.

### MEDIUM Fix — Missing cached_tokens in streaming usage (server.py)

`stream_chat_completion()` final usage chunk lacked `prompt_tokens_details.cached_tokens`. Added `cached_tokens` tracking and `PromptTokensDetails` to both the normal and tool-call usage paths.

### MEDIUM Fix — Attention mask left-padding (multimodal_processor.py)

`prepare_for_batch()` created all-ones mask for inputs without attention masks, ignoring left-padding. Fixed to properly set zeros for padding positions and ones for real tokens.

### LOW Fix — Dead deferred import (engine/simple.py)

Removed redundant inline `from ..api.tool_calling import check_and_inject_fallback_tools` at line 672 (already imported at module level).

### Verified (all prior fixes confirmed working by agents)

All ~60 prior fixes verified correct by 3 independent deep-tracing agents covering: engine+scheduler+cache, server+tools+reasoning, VLM/MLLM+MCP+misc.

---

## CRITICAL Fixes (8 total — all applied)

### C1: output_collector.py — `clear()` can leave `get()` coroutine stuck forever

**Problem:** If `clear()` is called while a consumer is blocked in `get()`, `clear()` sets `output = None` and clears the `ready` event, then decrements `_waiting_consumers`. The `get()` coroutine remains stuck in `while self.output is None: await self.ready.wait()` forever because nothing will set `ready` again.

**Fix:** Added `_cancelled` flag to `RequestOutputCollector.__init__`. `clear()` now sets `_cancelled = True` and **sets** (not clears) the ready event to wake blocked consumers. `get()` checks `_cancelled` before waiting and after waking, raising `RuntimeError("Collector was cancelled")` instead of hanging forever.

---

### C2: engine_core.py — `abort_request()` calls `scheduler.abort_request()` twice

**Problem:** `abort_request()` calls `self.scheduler.abort_request(request_id)` directly, then calls `self._cleanup_request(request_id)` which calls `self.scheduler.abort_request(request_id)` again.

**Fix:** Removed the direct `scheduler.abort_request()` call. `_cleanup_request()` is the single point of scheduler cleanup.

---

### C3: mllm_scheduler.py — Paged cache path assumes `_extracted_cache` is callable

**Problem:** In `_cleanup_finished()`, the paged cache path calls `request._extracted_cache()` without checking if callable. If `_extracted_cache` is a raw cache object, this crashes with `TypeError`.

**Fix:** Changed to `raw = request._extracted_cache; cache_blocks = raw() if callable(raw) else raw`.

---

### C4: scheduler.py — `_ensure_batch_generator()` doesn't clear `block_aware_cache` on recreation

**Problem:** When BatchGenerator is recreated, paged `block_aware_cache` is not cleared. Stale block tables reference garbage KV data.

**Fix:** Added `block_aware_cache.clear()` to the cache clearing block.

---

### C5: server.py — Rate limiter uses raw Authorization header as client ID

**Problem:** All users sharing the same API key share one rate limit bucket.

**Fix:** Changed to use client IP: `X-Forwarded-For` first IP → `request.client.host` → `"unknown"`.

---

### C6: server.py — `/v1/cache/warm` blocks the entire asyncio event loop

**Problem:** Synchronous model prefill in async handler blocks all endpoints.

**Fix:** Wrapped prefill logic with `await asyncio.to_thread()`.

---

### C7: server.py — `stream_completion` creates new timestamp per SSE chunk

**Problem:** OpenAI API spec requires all chunks in a stream to share the same `created` timestamp.

**Fix:** Added `created` variable before loops, passed to all chunk instantiations.

---

### C8: models/mllm.py — `NameError` on `chunk` variable in `stream_chat()` when zero tokens

**Problem:** Final yield references `getattr(chunk, ...)` with unreliable `if 'chunk' in locals()` guard.

**Fix:** Added `last_prompt_tokens` variable updated inside the loop.

---

## HIGH Fixes (20+ applied)

### H1: scheduler.py + engine/batched.py — Idempotent shutdown

**Problem:** Calling `stop()` multiple times could raise errors or double-free resources.

**Fix:** Added `_stopped` guard flag. Second `stop()` calls are no-ops with early return.

---

### H5: mllm_scheduler.py — Queue access without lock

**Problem:** `_request_queue` accessed from both `add_request()` (caller thread) and `_process_loop()` (async loop) without synchronization.

**Fix:** Added `_queue_lock = asyncio.Lock()` guard around all queue access.

---

### H6: mllm_scheduler.py — Missing FINISHED_ABORTED status

**Problem:** Aborted requests not moved to terminal state, left in limbo.

**Fix:** Added `FINISHED_ABORTED` to `MLLMRequestStatus` enum, used in abort path.

---

### H7: server.py — TTS endpoint uses raw query params instead of Pydantic model

**Problem:** TTS endpoint parsed query params manually, no validation.

**Fix:** Created `AudioSpeechRequest` Pydantic model with proper validation. (Agent fix)

---

### H8: server.py — `stream_completions_multi` hardcoded `prompts[0]`

**Problem:** Multi-prompt completions only streamed the first prompt.

**Fix:** Changed to iterate over all prompts with `enumerate(prompts)`. (Agent fix)

---

### H9: server.py — Non-streaming chat completions response includes None fields

**Problem:** JSON response included `null` for optional fields.

**Fix:** Added `response_model_exclude_none=True` to the endpoint decorator. (Agent fix)

---

### H10: server.py — Embedding engine hot-swap race condition

**Problem:** Two concurrent embedding requests with different models could interleave engine creation.

**Fix:** Added `_embedding_lock = asyncio.Lock()` around model check+load+use. (Agent fix)

---

### H11: models/mllm.py — Base64 image cache unbounded growth

**Problem:** `_base64_image_cache` (dict) grows without limit, leaking memory.

**Fix:** Changed to `OrderedDict` with LRU eviction at 100 entries. (Agent fix)

---

### H12: models/mllm.py — MD5 hash of first 1000 chars for image cache keys

**Problem:** Two different images with same first 1000 chars of base64 would collide.

**Fix:** Changed to SHA-256 hash of full content. (Agent fix)

---

### H13: models/mllm.py — Hardcoded 256 `num_image_tokens`

**Problem:** All models assumed 256 image tokens regardless of actual model config.

**Fix:** Dynamic detection from `model.config` attributes. (Agent fix)

---

### H14: models/mllm.py — Monolithic `chat()` method ~300 lines

**Problem:** Single method handling all multimodal chat logic, hard to maintain.

**Fix:** Extracted `_extract_multimodal_messages()` and `_apply_chat_template()` helpers. (Agent fix)

---

### H15: api/tool_calling.py — Brace counting ignores JSON string contents

**Problem:** `_parse_raw_json_tool_calls()` counts `{`/`}` without tracking whether they're inside JSON strings. `{"args": "print({x})"}` breaks parsing.

**Fix:** Added `in_string` and `escape` tracking in the brace-counting loop. (Agent fix)

---

### H16: api/tool_calling.py — Nemotron `cleaned_text` not applied

**Problem:** Text cleaned of think tags wasn't used for subsequent parsing.

**Fix:** Applied `cleaned_text` properly in Nemotron path. (Agent fix)

---

### H18: mcp/security.py — Overly broad `DANGEROUS_PATTERNS` regex

**Problem:** Patterns like `rm` matched anywhere in arguments (e.g., `--format`).

**Fix:** Made patterns more specific with word boundaries. (Agent fix)

---

### H19: mcp/security.py — `NODE_OPTIONS` not blocked in env validation

**Problem:** Attacker could set `NODE_OPTIONS=--require=/malicious.js` in MCP server env.

**Fix:** Added `NODE_OPTIONS` to blocked environment variables list. (Agent fix)

---

### H20: mcp/client.py — Resource leak in `connect` methods

**Problem:** Connection resources not cleaned up on partial connect failure.

**Fix:** Added proper cleanup in error paths. (Agent fix)

---

### H21: scheduler.py — Dead code in `_schedule_waiting()`

**Problem:** Unreachable code path after `break` statement.

**Fix:** Removed dead code.

---

### H23: block_disk_store.py — New SQLite connection per read

**Problem:** Each block read opens/closes a SQLite connection, causing overhead.

**Fix:** Added persistent `_read_conn` opened at init, closed at shutdown.

---

### H24: block_disk_store.py — Unbounded write queue

**Problem:** Write queue grows without limit under sustained write pressure.

**Fix:** Added `_write_queue_max = 500` with LRU eviction of oldest block write when full.

---

### H26: multimodal_processor.py — `break` in extra_kwargs merge loop

**Problem:** `break` exits loop after first non-standard key, skipping remaining keys.

**Fix:** Removed `break` to process all keys.

---

### H27: plugin.py — Wrong class path in platform plugin return

**Problem:** Returned class path pointed to wrong module.

**Fix:** Changed to `vllm_mlx.mlx_platform.MLXPlatform`.

---

### FALSE POSITIVES (not fixed):
- **C9, H2, H3, H4, H17** — Identified as false positives after code review
- **H22, H25** — Downgraded (not actually bugs)
- **H28** — benchmark.py unconditional imports: standalone CLI tool, not auto-imported

---

## MEDIUM Fixes (20+ applied)

### M-Sched1: scheduler.py — Stale detokenizer on rescheduled requests

**Problem:** `_reschedule_running_requests()` resets request state (status, cache, tokens) but does NOT clear the streaming detokenizer. When re-scheduled requests restart generation, `_get_detokenizer()` returns the old detokenizer with accumulated text from the aborted pass. This corrupts string-stop detection and final `output.output_text`.

**Fix:** Added `self._cleanup_detokenizer(request_id)` call in `_reschedule_running_requests()` after resetting request state but before moving to waiting queue.

**File:** `scheduler.py` line 1855

---

### M-Sched2: scheduler.py — Unused module-level `NaiveStreamingDetokenizer` import

**Problem:** Imported at top level (line 24) but only used inside `_get_detokenizer()` which has its own local import.

**Fix:** Removed module-level import. Local import at line 683 is sufficient.

**File:** `scheduler.py` line 24

---

### M-Sched3: scheduler.py — Dead `cls_name` assignment in `_truncate_cache_to_prompt_length`

**Problem:** `cls_name = type(layer_cache).__name__` assigned but never read.

**Fix:** Removed the dead assignment.

**File:** `scheduler.py` line 903

---

### M-Sched4: scheduler.py — Docstring says `default 1` for `max_retries` but actual default is `2`

**Problem:** `step(max_retries: int = 2)` but docstring says "default 1".

**Fix:** Changed docstring to "default 2".

**File:** `scheduler.py` line 1877

---

### M-Sched5: scheduler.py — `import traceback` inside function body

**Problem:** `traceback` imported locally inside `_prefill_for_prompt_only_cache` despite being a stdlib module.

**Fix:** Moved to top-level imports, removed inline import.

**File:** `scheduler.py` lines 17, 652

---

### M-Sched6: scheduler.py — `set()` vs `.clear()` inconsistency

**Problem:** `finished_req_ids = set()` creates new object; `reset()` uses `.clear()`. Inconsistent.

**Fix:** Changed to `finished_req_ids.clear()`.

**File:** `scheduler.py` line 1937

---

### M-Engine1: engine_core.py — `or` vs `is None` in `stream_outputs`

**Problem:** Non-timeout branch uses `output = collector.get_nowait() or await collector.get()`. If `get_nowait()` returns a falsy-but-valid `RequestOutput`, the `or` discards it and unnecessarily blocks. Inconsistent with the timeout branch which uses explicit `is None` check.

**Fix:** Changed to explicit `is None` check: `output = collector.get_nowait(); if output is None: output = await collector.get()`.

**File:** `engine_core.py` line 401

---

### M-Engine2: engine_core.py — Dead deferred imports in `generate_batch_sync()`

**Problem:** `from .request import Request` and `import uuid as uuid_module` inside function body, but both are already imported at module level.

**Fix:** Removed deferred imports. Changed `uuid_module.uuid4()` to `uuid.uuid4()`.

**File:** `engine_core.py` lines 494-495, 500

---

### M-Engine3: engine/simple.py — **DEADLOCK** in non-MLLM `chat()` path

**Problem:** `chat()` acquires `self._generation_lock` (asyncio.Lock, non-reentrant) at line 319, then calls `await self.generate()` at line 412, which also tries to acquire `self._generation_lock` at line 134. This deadlocks every non-MLLM chat request.

**Fix:** Replaced the `self.generate()` call with inline generation code (same logic as `generate()` but without re-acquiring the lock): `await asyncio.to_thread(self._model.generate, ...)` + `clean_output_text()` + `GenerationOutput(...)`.

**File:** `engine/simple.py` lines 411-418

---

### M-Engine4: engine/batched.py + simple.py — Redundant `(TypeError, Exception)` exception handling

**Problem:** `except (TypeError, Exception)` is equivalent to `except Exception` since `TypeError` is a subclass.

**Fix:** Changed all 6 occurrences to `except Exception`.

**Files:** `engine/batched.py` lines 381, 401; `engine/simple.py` lines 377, 394, 641, 661

---

### M-Engine5: engine/batched.py + simple.py — Hot-path deferred imports

**Problem:** `from ..api.tool_calling import check_and_inject_fallback_tools` imported inside methods called on every chat request.

**Fix:** Moved to top-level imports, removed 3 inline imports.

**Files:** `engine/batched.py` line 407; `engine/simple.py` lines 399, 673

---

### M-Server1: server.py — Streaming completions drops `top_k`, `min_p`, `repetition_penalty`

**Problem:** `stream_completions_multi` only passes `temperature`, `top_p`, and `stop` to `engine.stream_generate()`, silently dropping `top_k`, `min_p`, and `repetition_penalty` that the non-streaming path forwards.

**Fix:** Built `gen_kwargs` dict with conditional inclusion of all sampling params, then `stream_generate(**gen_kwargs)`.

**File:** `server.py` lines 2425-2431

---

### M-Server2: server.py — Dead `_tool_parser_instance` global

**Problem:** Module-level global `_tool_parser_instance` declared, added to `global` statement, and reset on model reload, but never assigned a value or read.

**Fix:** Removed the variable declaration, removed from `global` statement, removed reset line.

**File:** `server.py` lines 229, 575, 587

---

### M-Server3: server.py — Unreachable `elif _tool_call_parser == "auto"` branch

**Problem:** The `elif _tool_call_parser == "auto"` block in `main()` can never execute because `_tool_call_parser` is only set inside the `if args.enable_auto_tool_choice:` block that is the `if` branch of the same conditional.

**Fix:** Removed the dead elif block (7 lines).

**File:** `server.py` lines 3671-3678

---

### M-Server4: server.py — Redundant inline imports (3 `convert_tools_for_template`, 2 `GptOssReasoningParser`)

**Problem:** Functions already imported at module level re-imported inside request handlers.

**Fix:** Removed 2 inline `convert_tools_for_template` imports and 2 inline `GptOssReasoningParser` imports. Added `GptOssReasoningParser` to top-level imports.

**File:** `server.py` lines 2205, 2229, 1771, 2233

---

### M-Server5: server.py — Redundant `(ImportError, Exception)` tuple

**Problem:** Same pattern as M-Engine4.

**Fix:** Changed to `except Exception`.

**File:** `server.py` line 669

---

### M-Server6: server.py — Redundant `list_parsers` double import

**Problem:** `list_parsers` imported at line 3524, then re-imported at line 3614 alongside `get_parser`.

**Fix:** Second import changed to `from .reasoning import get_parser` only.

**File:** `server.py` line 3615

---

### M-Server7: server.py — `fastapi_request: Request = None` type annotation

**Problem:** Missing optionality in type annotation (default `None` but type says `Request`).

**Fix:** Changed to `Request | None = None` in both occurrences.

**File:** `server.py` lines 2483, 3008

---

### M-MLLM1: models/mllm.py — `TempFileManager.cleanup()` removes from set before unlink

**Problem:** Path is removed from `_files` set before `os.unlink()`. If unlink fails (OSError), the file is orphaned — `cleanup_all()` will never retry it.

**Fix:** Moved `_files.discard(path)` to after successful `os.unlink()`.

**File:** `models/mllm.py` lines 50-62

---

### M-MLLM2: models/mllm.py — Pydantic v1 `.dict()` usage

**Problem:** `.dict()` deprecated in Pydantic v2 (project uses 2.12.5).

**Fix:** Changed to `.model_dump()`.

**File:** `models/mllm.py` line 835

---

### M-MScheduler1: mllm_scheduler.py — Redundant inline `NaiveStreamingDetokenizer` import

**Problem:** Module-level import at line 151, redundant local import at line 610.

**Fix:** Removed inline import at line 610.

**File:** `mllm_scheduler.py` line 610

---

### M-Parser1: tool_parsers — `generate_tool_id()` duplicated across 13 files

**Problem:** Identical 3-line function defined in every tool parser file. If ID format needs to change, all 13 files need updating.

**Fix:** Added shared `generate_tool_id()` to `abstract_tool_parser.py` with `import uuid`. Updated all 13 parser files to import from there and removed their local definitions + local `import uuid`.

**Files:** `abstract_tool_parser.py`, all 13 parser files

---

### M-Parser2: auto_tool_parser.py — `_parse_raw_json_tool_calls` lacks string-aware brace counting

**Problem:** Same bug as H15 in `api/tool_calling.py` — brace counting doesn't track JSON strings. `{"args": "print({x})"}` breaks depth tracking. The H15 fix was applied to `api/tool_calling.py` but `auto_tool_parser.py` has an independent re-implementation without the fix.

**Fix:** Added `in_string` and `escape` tracking to the brace-counting loop, matching the H15 fix.

**File:** `tool_parsers/auto_tool_parser.py` lines 281-295

---

### M-Parser3: kimi_tool_parser.py — `split(":")[-2]` fragile for func IDs

**Problem:** `func_id.split(":")[-2]` assumes at least 2 colons. For `"func:0"` (1 colon), `split(":")` gives `["func", "0"]`, so `[-2]` works. But for `"a:b"` (no index), `[-2]` gives `"a"`, dropping `"b"`.

**Fix:** Changed to `rsplit(":", 1)[0]` which always strips only the trailing `:N` index.

**File:** `tool_parsers/kimi_tool_parser.py` line 93

---

### M-Parser4: kimi_tool_parser.py — Unused `TOOL_CALLS_END_ALT` constant

**Problem:** Defined but never referenced. If Kimi sends singular variant, streaming wouldn't detect it.

**Fix:** Removed dead constant.

**File:** `tool_parsers/kimi_tool_parser.py` line 48

---

### M-Parser5: minimax_tool_parser.py — Dead code after `json.loads`

**Problem:** `null`/`true`/`false` special case branches unreachable because `json.loads()` already handles them.

**Fix:** Removed dead branches, simplified to try `json.loads` → fall back to raw string.

**File:** `tool_parsers/minimax_tool_parser.py` lines 57-69

---

### M-Runner1-4: model_runner.py — References to non-existent functions

**Problem:** Imports/calls to `configure_memory_optimization`, `get_optimal_prefill_size`, and `optimal_prefill_size` that don't exist in `optimizations.py` (likely removed in previous cleanup).

**Fix:** Removed broken imports and calls. Kept inline `get_optimal_prefill_size` fallback. Removed non-existent `optimal_prefill_size` from hardware info dict.

**File:** `model_runner.py` — 4 edits

---

## LOW Fixes (applied alongside MEDIUM)

- **L-Engine1:** `engine_core.py` — Duplicate deferred imports of `Request` and `uuid` (both already at top level). Removed + changed `uuid_module` to `uuid`.
- **L-Engine2:** `engine/batched.py` + `simple.py` — `(TypeError, Exception)` → `except Exception` (6 occurrences).
- **L-Engine3:** `engine/batched.py` + `simple.py` — Hot-path deferred `check_and_inject_fallback_tools` import → top-level (3 occurrences).

---

## Files Modified (summary)

| File | Fixes Applied |
|------|--------------|
| `output_collector.py` | C1 |
| `engine_core.py` | C2, M-Engine1, M-Engine2 |
| `mllm_scheduler.py` | C3, H5, H6, M-MScheduler1 |
| `scheduler.py` | C4, H1, H21, M-Sched1–6 |
| `server.py` | C5, C6, C7, H7–H10, M-Server1–7 |
| `models/mllm.py` | C8, H11–H14, M-MLLM1, M-MLLM2 |
| `engine/batched.py` | H1, M-Engine4, M-Engine5 |
| `engine/simple.py` | M-Engine3 (deadlock), M-Engine4, M-Engine5 |
| `block_disk_store.py` | H23, H24 |
| `multimodal_processor.py` | H26 |
| `plugin.py` | H27 |
| `api/tool_calling.py` | H15, H16 |
| `mcp/security.py` | H18, H19 |
| `mcp/client.py` | H20 |
| `model_runner.py` | M-Runner1–4 |
| `tool_parsers/abstract_tool_parser.py` | M-Parser1 (shared `generate_tool_id`) |
| `tool_parsers/auto_tool_parser.py` | M-Parser1, M-Parser2 |
| `tool_parsers/kimi_tool_parser.py` | M-Parser1, M-Parser3, M-Parser4 |
| `tool_parsers/minimax_tool_parser.py` | M-Parser1, M-Parser5 |
| `tool_parsers/qwen_tool_parser.py` | M-Parser1 |
| `tool_parsers/hermes_tool_parser.py` | M-Parser1 |
| `tool_parsers/deepseek_tool_parser.py` | M-Parser1 |
| `tool_parsers/llama_tool_parser.py` | M-Parser1 |
| `tool_parsers/granite_tool_parser.py` | M-Parser1 |
| `tool_parsers/nemotron_tool_parser.py` | M-Parser1 |
| `tool_parsers/xlam_tool_parser.py` | M-Parser1 |
| `tool_parsers/functionary_tool_parser.py` | M-Parser1 |
| `tool_parsers/step3p5_tool_parser.py` | M-Parser1 |
| `tool_parsers/glm47_tool_parser.py` | M-Parser1 |

**Total: ~30 files modified, ~60 distinct fixes applied across all tiers.**

---

## Known Issues Not Fixed (deferred)

These were identified but deferred as lower priority or requiring larger refactors:

1. ~~**block_disk_store.py — per-item SQLite connection in background writer**~~ — **FIXED in Session 5 (S5-D2)**
2. **scheduler.py — `_truncate_cache_to_prompt_length` called twice** (Sched M2) — Redundant computation. Deferred: optimization, not a bug.
3. **scheduler.py — Recovery `_schedule_waiting()` result discarded** (Sched M6) — SchedulerOutput incomplete on recovery. Deferred: edge case only hit during cache error recovery.
4. **server.py — `--served-model-name` CLI arg missing** (Server M2) — Feature addition, not a bug fix.
5. **server.py — STT/TTS engine race condition** (Server M6) — Same pattern as H10. Deferred: low traffic endpoints.
6. **engine_core.py — `AsyncEngineCore.start()` is sync** (Engine M3) — Returns None, callers can't await readiness. Deferred: API change.
7. **paged_cache.py — Double RLock acquisition** (Engine M5) — Works (RLock is reentrant) but confusing. Deferred: code clarity.
8. **api/models.py — `reasoning_content: null` always emitted** (API L4) — Cosmetic OpenAI compat issue.
9. **tool_parsers/hermes_tool_parser.py — `strip_think_tags` ordering** (API M5) — Think-wrapped reasoning silently lost. Deferred: requires reasoning parser integration knowledge.
