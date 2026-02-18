# vMLX v1 — Changelog from Recent Sessions

## Features Added

### Date Injection (chat.ts)
- Every chat request now includes `Current date: [weekday], [month] [day], [year], [time]` in the system prompt
- Placed at END of system prompt to maximize prefix cache hits (stable prefix before the date)
- If no system prompt is set, injects a minimal `You are a helpful assistant.` + date

### Parser Registry Updates
- **Gemma 3**: Added `toolParser: 'hermes'`, `reasoningParser: 'deepseek_r1'`, `enableAutoToolChoice: true`
- **Phi 4 / Phi 4 Reasoning**: Added `toolParser: 'hermes'`, `enableAutoToolChoice: true`
- Python-side `model_configs.py` mirrors TypeScript registry
- Comprehensive model lists in all parser dropdown tooltips

### Session Config Form
- Updated TOOL_PARSER_OPTIONS with 15 parsers and model lists
- Updated REASONING_PARSER_OPTIONS with model families
- Tooltips show all supported models for each parser

## Bug Fixes

### Remote Session DNS Resolution (CRITICAL)
- **Root cause**: Node.js `fetch` resolves `.local` (mDNS/Bonjour) hostnames to IPv6 link-local addresses (`fe80::...`) which are unreachable
- **Fix**: `resolveUrl()` function pre-resolves `.local` hostnames to IPv4 using `dns.lookup({ family: 4 })`
- Applied to: session connect, health monitor, and chat message sending
- Trailing slash stripped from resolved URLs to prevent double-slash in paths

### Remote Session Endpoint Resolution in Chat (CRITICAL)
- **Root cause**: When renderer passes `endpoint` to `chat:sendMessage`, the `resolved` object had NO session attached, making `isRemote` always false
- **Consequences**: Wrong health check path (`/health` vs `/v1/models`), no auth headers, misleading error messages
- **Fix**: Session lookup from line 313 (already existed for timeout/reasoning parser) now saved as `chatSession` and attached to the resolved endpoint

### False Heartbeat Disconnect During Inference (CRITICAL)
- **Root cause**: `incrementFailAndCheck()` called `isProcessAlive()` which always returns false for remote sessions (no PID). This triggered `handleSessionDown()` on the FIRST health check failure
- **Fix**: Skip the "process dead → immediate disconnect" fast-path for remote sessions; use normal fail-count threshold (60 failures = 5+ min)
- Remote health check timeout: 5s → 10s (matching local)
- Remote health check failures now use dampened counting (every 3rd) like local sessions
- Emits `busy: true` health event for remote failures

### Double-Connect Crash Prevention
- **Root cause**: `_connectRemoteSession()` had no guard against being called on already-running sessions, and `handleSessionDown()` never aborted in-flight inference for remote sessions
- **Fix 1**: `startSession()` guards remote sessions — returns immediately if already `running` or `loading`
- **Fix 2**: `handleSessionDown()` emits `session:abortInference` for remote sessions, consumed by `ipc/sessions.ts` to call `abortByEndpoint()` before marking stopped
- Prevents orphaned SSE streams that would cause double inference on reconnect

### Crash Recovery for Remote Sessions
- **Root cause**: `detectAndAdoptAll()` only handled local processes via `ps aux` — remote sessions left as "running" in SQLite were never reset on app restart
- **Fix**: Added code to reset stale remote sessions to "stopped" on startup

### Remote Session Reconnect Retry
- `_connectRemoteSession()` now retries 3 times with increasing delay for transient DNS/network issues

## Files Modified

### TypeScript (Panel)
- `panel/src/main/sessions.ts` — resolveUrl, health monitor dampening, remote session guards, abort inference event
- `panel/src/main/ipc/chat.ts` — Date injection, session attachment to endpoint, DNS resolution, logging
- `panel/src/main/ipc/sessions.ts` — session:abortInference listener
- `panel/src/main/model-config-registry.ts` — Gemma 3, Phi 4 parser configs
- `panel/src/renderer/src/components/sessions/SessionConfigForm.tsx` — Parser dropdowns

### Python (Engine)
- `vllm_mlx/model_configs.py` — Gemma 3, Phi 4 parser configs
