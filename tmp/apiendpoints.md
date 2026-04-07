# API Endpoints

This document summarizes the implemented endpoint families in `vmlx_engine/server.py`. The machine-readable route inventory is in [openapi.yml](/Users/owenmatsumoto/vmlx/tmp/openapi.yml).

## Local-Only Runtime Guarantee

- Inference runs locally on MLX (`inference_backend: "mlx"` in `/health`).
- No OpenAI/Anthropic cloud inference forwarding is used by server inference paths.
- Cache systems remain active and unchanged (prefix/paged/disk/multimodal caches).
- Cloud-only OpenAI management surfaces return structured `501 not_implemented_local`.

## Base URL

- Default local server: `http://127.0.0.1:8000`

## Authentication

- If server is started with `--api-key`, endpoints require:
  - `Authorization: Bearer <key>`
  - `Authorization: Token <key>` (also accepted)
  - `x-api-key: <key>` (also accepted)
  - `api-key: <key>` (also accepted)

## Health / Admin

- `GET /health`
- `POST /admin/soft-sleep`
- `POST /admin/deep-sleep`
- `POST /admin/wake`

## Cache

- `GET /v1/cache/stats`
- `GET /v1/cache/entries`
- `POST /v1/cache/warm`

## OpenAI-Compatible APIs

### Core text/chat/responses

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /responses/{response_id}`
- `GET /v1/responses/{response_id}`
- `GET /responses/{response_id}/input_items`
- `GET /v1/responses/{response_id}/input_items`
- `DELETE /responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `POST /chat/completions` (spec alias)
- `POST /responses` (spec alias)
- `POST /completions` (spec alias)
- `POST /embeddings` (spec alias)
- `GET /models` (spec alias)
- `GET /models/{model}` (spec alias)

### Cancellation

- `POST /v1/chat/completions/{request_id}/cancel`
- `POST /v1/completions/{request_id}/cancel`
- `POST /v1/responses/{response_id}/cancel`

### Realtime (WebSocket + session creation)

- `POST /v1/realtime/sessions`
- `POST /v1/realtime/client_secrets`
- `POST /v1/realtime/transcription_sessions`
- `WS /v1/realtime`
- `POST /realtime/sessions` (spec alias)
- `POST /realtime/client_secrets` (spec alias)
- `POST /realtime/transcription_sessions` (spec alias)
- `WS /realtime` (spec alias)

Supported inbound websocket event types:
- `session.update`
- `conversation.item.create`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `response.create`

Primary outbound websocket event types:
- `session.created`
- `session.updated`
- `conversation.item.created`
- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.output_item.done`
- `response.done`
- `response.error`

### Embeddings / rerank

- `POST /v1/embeddings`
- `POST /v1/rerank`

### Images

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /images/generations` (spec alias)
- `POST /images/edits` (spec alias)
- `POST /images/variations` (spec compatibility stub)

### Audio

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `GET /v1/audio/voices`
- `POST /audio/transcriptions` (spec alias)
- `POST /audio/translations` (spec compatibility alias to transcription)
- `POST /audio/speech` (spec alias)
- `GET /audio/voices` (spec alias)

### Moderation

- `POST /moderations`

### Files

- `GET /files`
- `GET /v1/files`
- `POST /files`
- `POST /v1/files`
- `GET /files/{file_id}`
- `GET /v1/files/{file_id}`
- `GET /files/{file_id}/content`
- `GET /v1/files/{file_id}/content`
- `DELETE /files/{file_id}`
- `DELETE /v1/files/{file_id}`

## Anthropic-Compatible API

- `POST /v1/messages`

## Ollama-Compatible API

- `GET /api/tags`
- `GET /api/ps`
- `GET /api/version`
- `POST /api/show`
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embeddings`
- `POST /api/embed`
- `POST /api/pull`
- `POST /api/delete`
- `POST /api/copy`
- `POST /api/create`

## LM Studio Native v1 API

- `GET /lmstudio/v1/models`
- `POST /lmstudio/v1/models/load`
- `POST /lmstudio/v1/models/unload`
- `POST /lmstudio/v1/models/download`
- `GET /lmstudio/v1/models/download/status`
- `POST /lmstudio/v1/chat`

## Deepgram-Compatible API (Local MLX)

Implemented for local-only operation with MLX-backed engines:
- `POST /deepgram/vl/listen`
- `POST /deepgram/vl/speak`
- `POST /deepgram/vl/read`
- `GET /deepgram/vl/models`
- `GET /deepgram/vl/models/{model_id}`

Notes:
- Deepgram's `/v1/models` and `/v1/models/{model_id}` paths overlap with OpenAI compatibility paths in this unified server.
- To avoid route collisions, Deepgram model listing/detail is exposed at `/deepgram/vl/models*`.
- `/deepgram/vl/listen` and `/deepgram/vl/read` in local-only mode accept JSON `url` sources only when using `file://` URLs.

## Localized Cloud-Style APIs

These OpenAI-style APIs now have local implementations instead of defaulting to `501`:
- Realtime client secrets and transcription sessions
- Files list/upload/retrieve/content/delete
- Responses retrieval, input item listing, and deletion using local in-memory persistence

## MCP APIs

- `GET /v1/mcp/tools`
- `GET /v1/mcp/servers`
- `POST /v1/mcp/execute`

## Notes on Model Compatibility

The server supports OpenAI/Anthropic/Ollama formats plus Realtime-compatible flows. Model-family detection includes:
- Qwen3 Omni variants (`qwen3_omni*` and name fallback containing `qwen3` + `omni`)
- Voxtral Realtime variants (`voxtral*` and name fallback containing `voxtral`)

For local single-model serving, requests naming a different model ID are normalized to the currently loaded model.

## Cloud-only Spec Endpoints

OpenAI cloud/resource-management families (assistants, threads, uploads, files, vector stores, organization/project admin, etc.) are surfaced with OpenAI-style `501 not_implemented_local` responses in local mode.
