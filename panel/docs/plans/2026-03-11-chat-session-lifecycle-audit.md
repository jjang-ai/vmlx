# Chat-Session Lifecycle Audit

**Date:** 2026-03-11
**Status:** Completed — all issues fixed

---

## Architecture Overview

### Chat-Session Binding
- Chats store `model_path` in DB (the binding key)
- Sessions store `model_path` as unique identifier
- No FK constraint between chats and sessions (intentional — chats survive session deletion)
- Resolution: `App.tsx handleChatSelect` → prefer running session with matching modelPath → any matching → any running → first available

### State Flow
```
SessionsContext (sessions list + live status)
  → App.tsx (derives sessionEndpoint from activeSession.status)
    → ChatModeToolbar (displays session status, settings access)
    → ChatInterface (sends messages via endpoint, shows "Model not running" banner)
```

### Session Events (IPC → SessionsContext → UI re-render)
- `session:created` → refresh list
- `session:starting` → set status='loading'
- `session:ready` → set status='running', update port
- `session:stopped` → set status='stopped'
- `session:error` → set status='error'
- `session:health` → set status='running', update modelName
- `session:deleted` → refresh list

---

## Verified Working Behaviors

### Background Streaming
- Stream continues when user switches to another chat (cleanup doesn't abort)
- Periodic 5s DB saves persist partial content during streaming
- Final `db.addMessage()` on completion writes complete content
- `isStreaming` IPC check on chat return re-syncs loading state
- Multiple chats can stream concurrently (per-chat `activeRequests` Map keying)
- `chat:complete` event fires regardless of which chat is active; DB has final content

### Session State Propagation
- All status changes propagate via IPC events to SessionsContext
- UI re-renders immediately when session status changes
- `sessionEndpoint` becomes `undefined` when session stops → banner appears
- Toolbar shows live status dot (green=running, yellow=loading, red=error)

### Settings Sync
- Session config (server-level): requires restart, stored as JSON blob in sessions table
- Chat overrides (per-chat): no restart needed, sent per-request
- Both settings panels (ServerSettingsDrawer, ChatSettings) load from active session
- When user switches session via toolbar dropdown, both panels reload

### Abort on Session Stop
- `abortByEndpoint(host, port)` cancels all active SSE streams targeting stopped endpoint
- Server-side cancel request sent (fire-and-forget) before abort
- Prevents orphaned network requests

---

## Issues Found & Fixed

### 1. Session Deletion Leaves Stale Active State (Fixed)
**Problem:** Deleting the active session left `activeSessionId` pointing to a nonexistent session. UI showed stale data.
**Fix:** `App.tsx` useEffect detects when `activeSessionId` is no longer in sessions list, falls back to next available session.

### 2. Send Allowed When Model Not Running (Fixed)
**Problem:** InputBox wasn't disabled when model was stopped. User could type and send, causing `resolveServerEndpoint` to fall back to default port (127.0.0.1:8093) — potentially wrong/dead.
**Fix:** InputBox `disabled` when `!sessionEndpoint && sessionId`. `handleSend` guards with toast message before any IPC call.

### 3. Background Stream Completion (Verified OK)
**Not an issue.** Final content is saved to DB via `db.addMessage()` on completion. `getMessages()` on return gets final content. The `isStreaming` check handles mid-stream returns.

---

## Concurrency Model

| Scenario | Behavior |
|----------|----------|
| Send in Chat A, switch to Chat B | Chat A streams in background, Chat B usable |
| Send in Chat A, send in Chat B | Both stream concurrently (different `activeRequests` keys) |
| Send in Chat A, send in Chat A again | Blocked by concurrency guard (stale lock recovery after 30min cap) |
| Session stops during stream | `abortByEndpoint` cancels stream, UI shows error |
| Delete session during stream | Stream aborted (session stop happens first), UI falls back to next session |

---

## Test Coverage (Phase 6)

Added 18 tests in `comprehensive-audit.test.ts`:
- Session deletion fallback resolution (4 tests)
- Send guard when model not running (5 tests)
- Chat selection session resolution (5 tests)
- Session endpoint derivation (4 tests)
