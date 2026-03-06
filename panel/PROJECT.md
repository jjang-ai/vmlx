# vMLX — Project Design Document

**A native macOS app for managing multiple vMLX Engine inference servers simultaneously**

---

## Project Goals

1. **Session-centric** — Each model loaded = one session with its own port, config, and chat history
2. **Multi-instance** — Run multiple vMLX Engine servers simultaneously on different ports
3. **Zero-config** — Auto-detect models, auto-assign ports, adopt running processes, auto-install vMLX Engine
4. **Persistent** — Chat history tied to model path, survives across load/unload cycles
5. **Full control** — Every vMLX Engine parameter exposed in the UI

---

## Architecture

### Tech Stack

- **Electron 28** — Native macOS app with three-layer IPC architecture
- **React + TypeScript** — Renderer UI with component-based views
- **Tailwind CSS** — Styling with custom component classes
- **SQLite WAL** (better-sqlite3) — Chat history, sessions, config persistence
- **Node.js** — Main process: process management, database, IPC handlers

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│  Renderer (React + TypeScript + Tailwind)        │
│  SetupScreen / SessionDashboard / CreateSession  │
│  SessionView / ChatInterface / UpdateManager     │
└────────────────────┬────────────────────────────┘
                     │  IPC (contextBridge)
┌────────────────────┴────────────────────────────┐
│  Preload (preload/index.ts)                      │
│  window.api.sessions / chat / models / vllm      │
└────────────────────┬────────────────────────────┘
                     │  ipcMain.handle
┌────────────────────┴────────────────────────────┐
│  Main Process (Node.js)                          │
│  SessionManager  → spawn/kill vMLX Engine processes │
│  DatabaseManager → SQLite WAL (chats, sessions)  │
│  VllmManager     → install/update/detect vMLX Engine│
│  IPC Handlers    → sessions, chat, models, vllm  │
└─────────────────────────────────────────────────┘
```

### View Routing

```
App.tsx (view routing, no sidebar)
├── SetupScreen         → First-run vMLX Engine installer gate
├── SessionDashboard    → Grid of session cards (home screen)
├── CreateSession       → Two-step wizard (model picker → config → launch)
├── SessionView         → Header + ChatInterface + Settings drawers (per-session)
│   ├── ChatSettings    → Per-chat inference params drawer
│   └── ServerSettings  → Inline server config drawer
├── SessionSettings     → Full-page vMLX Engine server config editor
└── About               → UpdateManager + app info
```

View state: `setup | dashboard | create | session:${id} | sessionSettings:${id} | about`

Navigation: title bar with Home button + About button. No sidebar tabs.

---

## User Flow

### 0. First Launch (Setup)

App checks for vMLX Engine installation. If not found:
- Auto-detects available installers (uv preferred, pip3 fallback with Python >=3.10 check)
- One-click install with streaming terminal output
- On success, proceeds to dashboard

### 1. Dashboard (Home)

App opens to a dashboard showing all sessions as cards. Sessions grouped into "Active" (running/loading) and "Inactive" (stopped/error).

Each card shows: model name, truncated path, host:port, PID, status indicator.

Actions:
- **Open** — Navigate to session interior (running sessions only)
- **Start** — Spawn the vMLX Engine process
- **Stop** — SIGTERM with 3s SIGKILL fallback
- **Delete** — Stop + remove from database
- **Configure** (gear icon) — Open full-page settings editor

Buttons:
- **New Session** — Launch creation wizard
- **Detect Processes** — Scan for running `vmlx-engine serve` processes

### 2. Create Session (Two-Step Wizard)

**Step 1: Select Model**
- Scans configured directories for MLX-format models
- Add/remove model directories via built-in directory manager
- Filterable search by model name
- Manual path entry for models not in scan directories

**Step 2: Configure Server**
Every vMLX Engine parameter in collapsible sections:

| Section | Parameters |
|---------|-----------|
| Server Settings | host, port (auto-assigned), API key, rate limit, timeout |
| Concurrent Processing | max sequences, prefill batch size, completion batch size, continuous batching |
| Prefix Cache | enable/disable, memory-aware vs entry-count, memory limit MB, memory limit percent |
| Paged KV Cache | enable/disable, block size, max blocks |
| Performance | stream interval, max tokens |
| Tool Integration | MCP config path, auto tool choice, parser selection |
| Additional Arguments | raw CLI flags textbox |

**Launch**: Creates session + spawns `vmlx-engine serve`. Loading screen shows real-time server logs.

### 3. Session Interior

**Header bar**: Model name, `host:port`, PID, status indicator, Stop button.

**Chat interface**: Streaming responses, markdown + code highlighting, metrics (t/s, pp/s, TTFT).

**Chat isolation**: Only shows chats for THIS model. New chats auto-tagged with `modelPath`.

**Settings drawers**:
- Chat Settings (per-chat): temperature, top_p, max_tokens, system prompt, stop sequences
- Server Settings (per-session): full vMLX Engine config form

### 4. About

vMLX Engine update manager: check for updates, one-click upgrade with streaming output, release notes.

---

## Database Schema

### Sessions Table
```sql
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  model_path TEXT NOT NULL UNIQUE,
  model_name TEXT,
  host TEXT NOT NULL DEFAULT '127.0.0.1',
  port INTEGER NOT NULL,
  pid INTEGER,
  status TEXT NOT NULL DEFAULT 'stopped'
    CHECK(status IN ('running','stopped','error','loading')),
  config TEXT NOT NULL,              -- JSON blob of ServerConfig
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_started_at INTEGER,
  last_stopped_at INTEGER
);
```

### Chats Table
```sql
CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  folder_id TEXT,
  model_id TEXT,
  model_path TEXT,                   -- ties chat to specific model
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
```

### Other Tables
- `messages` — Chat messages with metrics (tokens_per_sec, time_to_first_token)
- `folders` — Hierarchical chat organization
- `chat_overrides` — Per-chat inference parameter overrides (temperature, top_p, max_tokens, system_prompt, stop_sequences)
- `settings` — Key-value store (model directories, preferences)

---

## SessionManager (Multi-Instance Process Management)

**File**: `src/main/sessions.ts`

### Key Methods

| Method | Description |
|--------|-----------|
| `createSession(modelPath, config)` | Create DB record, auto-assign port if needed |
| `startSession(sessionId)` | Spawn `vmlx-engine serve`, wait for `/health` (120s timeout) |
| `stopSession(sessionId)` | SIGTERM → 3s → SIGKILL, with 15s hard timeout |
| `deleteSession(sessionId)` | Stop + remove from DB |
| `detectAndAdoptAll()` | Scan all running vLLM processes, create/update session records |
| `startGlobalMonitor()` | 5s interval: health-check all running sessions (3-strike retry) |
| `stopAll()` | Kill all managed processes (SIGTERM → SIGKILL) |

### Events (EventEmitter)

| Event | When |
|-------|------|
| `session:created/starting/ready/stopped/error/health/log/deleted` | Lifecycle events forwarded to renderer |

### Health Monitor
- Single `setInterval(5000)` iterates all sessions with `status === 'running'`
- 3-strike failure threshold before marking session as stopped
- Counter resets on any successful health check

---

## VllmManager (Install/Update)

**File**: `src/main/vllm-manager.ts`

### Key Functions

| Function | Description |
|----------|-----------|
| `checkVllmInstallation()` | Find binary, detect version and install method |
| `detectAvailableInstallers()` | Find uv/pip3, validate Python version |
| `installVllmStreaming()` | Streaming install/upgrade via spawn with real-time log output |
| `cancelInstall()` | Kill active install process |
| `checkForUpdates()` | Compare installed version against PyPI latest |
| `detectInstallMethod()` | Determine if installed via uv, pip, brew, conda (resolves symlinks) |

### Install Flow
1. `detectAvailableInstallers()` — checks uv paths, then pip3 with Python >=3.10 validation
2. User picks method (or auto-selects uv if available)
3. `installVllmStreaming()` — spawns `uv tool install vmlx-engine` or `pip3 install vmlx-engine`
4. Real-time stdout/stderr forwarded as `vllm:install-log` IPC events
5. On completion, Promise resolves with success/failure

---

## IPC Channels

### Session Management
| Channel | Direction | Description |
|---------|-----------|-------------|
| `sessions:list/get/create/start/stop/delete/detect/update` | invoke | Full CRUD + lifecycle |
| `session:starting/ready/stopped/error/health/log/created/deleted` | event | Real-time events |

### Chat
| Channel | Direction | Description |
|---------|-----------|-------------|
| `chat:create/get/getAll/getByModel/getMessages/addMessage/sendMessage/update/delete/search` | invoke | Chat CRUD |
| `chat:abort` | invoke | Cancel active generation |
| `chat:setOverrides/getOverrides/clearOverrides` | invoke | Per-chat settings |
| `chat:stream/complete` | event | SSE streaming |

### vMLX Engine
| Channel | Direction | Description |
|---------|-----------|-------------|
| `vllm:check-installation/detect-installers/check-updates` | invoke | Detection |
| `vllm:install-streaming/cancel-install/update/install` | invoke | Install/update |
| `vllm:install-log/install-complete` | event | Streaming output |

### Models
| Channel | Direction | Description |
|---------|-----------|-------------|
| `models:scan/info/getDirectories/addDirectory/removeDirectory/browseDirectory` | invoke | Model management |

### Settings
| Channel | Direction | Description |
|---------|-----------|-------------|
| `settings:get/set/delete` | invoke | App-level key-value settings (API keys, preferences) |

---

## Process Lifecycle

### Startup
1. Open SQLite database (WAL mode), run migrations
2. `sessionManager.detectAndAdoptAll()` — scan for running vLLM processes
3. `sessionManager.startGlobalMonitor()` — start 5s health-check interval
4. Check vMLX Engine installation → SetupScreen or Dashboard

### Session Start
1. `buildArgs()` converts SessionConfig → CLI flags
2. `findVllmMlx()` locates binary
3. `spawn('vmlx-engine', ['serve', modelPath, ...args])`
4. Stream stdout/stderr via `session:log` events
5. `waitForReady()` polls `/health` every 2s (120s timeout)
6. On success: emit `session:ready`, update DB status to `running`

### Session Stop
1. `SIGTERM` to process
2. Wait 3 seconds
3. If still alive: `SIGKILL`
4. Hard timeout: 15s (resolves even if process never exits)
5. Update DB status to `stopped`, emit `session:stopped`

### App Quit
1. `sessionManager.stopGlobalMonitor()`
2. `Promise.race([sessionManager.stopAll(), 8s timeout])`
3. Close database
4. `app.exit(0)`

---

## Key Files

| File | Purpose |
|------|---------|
| `src/main/sessions.ts` | SessionManager — multi-instance lifecycle, health monitoring (3-strike), process detection |
| `src/main/database.ts` | SQLite WAL schema + CRUD for all tables |
| `src/main/vllm-manager.ts` | vMLX Engine detection, streaming install (uv/pip), update, version checking |
| `src/main/index.ts` | App lifecycle: startup adoption, global monitor, graceful quit |
| `src/main/ipc/sessions.ts` | Session IPC handlers with window getter pattern |
| `src/main/ipc/chat.ts` | Chat IPC + SSE streaming + AbortController + per-chat concurrency guard |
| `src/main/ipc/models.ts` | Model scanning with configurable directories |
| `src/main/ipc/vllm.ts` | vLLM install/update IPC with streaming log forwarding |
| `src/preload/index.ts` | IPC bridge with unsubscribe-pattern event listeners |
| `src/preload/index.d.ts` | Complete TypeScript declarations |
| `src/renderer/src/App.tsx` | View routing: setup / dashboard / create / session / settings / about |
| `src/renderer/src/components/setup/SetupScreen.tsx` | First-run installer gate with streaming output |
| `src/renderer/src/components/sessions/SessionDashboard.tsx` | Home screen with session card grid |
| `src/renderer/src/components/sessions/CreateSession.tsx` | Two-step wizard with directory manager |
| `src/renderer/src/components/sessions/SessionView.tsx` | Session interior: header + chat + settings drawers |
| `src/renderer/src/components/sessions/SessionSettings.tsx` | Full-page server config editor |
| `src/renderer/src/components/sessions/SessionConfigForm.tsx` | Shared config form component |
| `src/renderer/src/components/sessions/ServerSettingsDrawer.tsx` | Inline server settings panel |
| `src/renderer/src/components/chat/ChatInterface.tsx` | Chat with streaming, markdown, metrics |
| `src/renderer/src/components/chat/ChatSettings.tsx` | Per-chat inference settings drawer |
| `src/renderer/src/components/chat/ChatList.tsx` | Chat history sidebar with folders, search, rename |
| `src/renderer/src/components/update/UpdateManager.tsx` | vMLX Engine update checker and installer |

---

**Created:** 2026-02-04
**Updated:** 2026-02-15
**Current Version:** v0.3.5
