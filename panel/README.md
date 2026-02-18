# vMLX — vLLM-MLX Instance Manager

**A native macOS app for managing multiple vLLM-MLX inference servers simultaneously**

Run multiple models on different ports, each with full configuration control, persistent chat history per model, and real-time health monitoring.

---

## Quick Start

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build + package for production
npm run build
npx electron-builder --mac --dir
```

---

## How It Works

### 1. First Launch — Setup

On first launch, vMLX checks for a vLLM-MLX installation. If not found, it offers **one-click install** via `uv` (preferred) or `pip3` (Python 3.10+ required). You can also install manually and click "Check Again".

### 2. Dashboard — See All Sessions

The dashboard shows all your vLLM-MLX sessions as cards. Each session represents one model loaded on one port. Sessions can be running, stopped, loading, or in an error state.

If vLLM-MLX is already running (started from terminal), click **Detect Processes** — the app scans for running `vllm-mlx serve` processes, health-checks each one, and automatically creates session records.

### 3. Create a Session — Pick Model + Configure

Click **New Session** to launch the two-step wizard:

1. **Select Model** — Scans configured directories for MLX-format models. Add custom model directories in the directory manager.
2. **Configure Server** — Every vLLM-MLX parameter is exposed:
   - **Server**: host, port (auto-assigned), API key, rate limit, timeout
   - **Concurrent Processing**: max sequences, prefill/completion batch sizes, continuous batching
   - **Prefix Cache**: enable/disable, memory-aware vs entry-count, memory limits
   - **Paged KV Cache**: block size, max blocks
   - **Performance**: stream interval, max tokens
   - **Tools**: MCP config, auto tool choice, parser
   - **Additional**: raw CLI arguments

Click **Launch** — the app spawns `vllm-mlx serve` and shows live server logs. When the health endpoint responds OK, you're taken into the session.

### 4. Inside a Session — Chat + API Info

Each session shows:
- **Header**: model name, `host:port`, PID, health status, Stop button
- **Chat**: full conversational interface with streaming, markdown, code highlighting, metrics (tokens/sec, prompt processing speed, TTFT)
- **Chat History**: persisted per model path — unload a model today, reload it tomorrow, your chats are still there
- **Chat Settings** (gear icon): side drawer with per-chat inference parameters — temperature, top_p, max_tokens, system prompt, stop sequences
- **Server Settings** (gear icon): inline drawer or full-page server configuration editor

Multiple sessions can run simultaneously on different ports.

### 5. About — vLLM-MLX Management

Access via the **About** button in the title bar. Check for vLLM-MLX updates, install/upgrade with streaming terminal output, and view release notes.

---

## Architecture

```
App.tsx (view routing)
├── SetupScreen         → First-run vLLM-MLX installer gate
├── SessionDashboard    → Grid of session cards (home screen)
├── CreateSession       → Two-step wizard (model picker → config → launch)
├── SessionView         → Header + ChatInterface + Settings drawers (per-session)
│   ├── ChatSettings    → Per-chat inference params drawer
│   └── ServerSettings  → Inline server config drawer
├── SessionSettings     → Full-page vLLM-MLX server config editor
└── About               → UpdateManager + app info
```

### Three-Layer Electron Architecture

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
│  SessionManager  → spawn/kill vLLM-MLX processes │
│  DatabaseManager → SQLite WAL (chats, sessions)  │
│  VllmManager     → install/update/detect vLLM-MLX│
│  IPC Handlers    → sessions, chat, models, vllm  │
└─────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/main/sessions.ts` | `SessionManager` — multi-instance lifecycle, process detection, health monitoring (3-strike retry) |
| `src/main/database.ts` | SQLite WAL schema + CRUD for sessions, chats, messages, folders, overrides, settings |
| `src/main/vllm-manager.ts` | vLLM-MLX detection, install (uv/pip streaming), update, version checking |
| `src/main/ipc/sessions.ts` | IPC handlers: list, get, create, start, stop, delete, detect, update |
| `src/main/ipc/chat.ts` | Chat handlers: create, sendMessage (SSE streaming), abort, getByModel |
| `src/main/ipc/models.ts` | Model scanning, directory management |
| `src/main/ipc/vllm.ts` | vLLM install/update IPC handlers with streaming log output |
| `src/main/index.ts` | App lifecycle: startup adoption, global monitor, graceful quit (8s timeout) |
| `src/preload/index.ts` | IPC bridge exposing `window.api` to renderer |
| `src/renderer/src/App.tsx` | View routing: setup / dashboard / create / session / sessionSettings / about |
| `src/renderer/src/components/setup/SetupScreen.tsx` | First-run vLLM-MLX installer gate |
| `src/renderer/src/components/sessions/` | Dashboard, Card, Create, View, Settings, ConfigForm, ServerSettingsDrawer |
| `src/renderer/src/components/chat/` | ChatInterface, ChatList, ChatSettings, MessageList, MessageBubble |
| `src/renderer/src/components/update/UpdateManager.tsx` | vLLM-MLX update checker and installer |

### Database Schema

```sql
-- Sessions: one per model path
sessions (id, model_path UNIQUE, model_name, host, port, pid, status, config JSON, timestamps)

-- Chats: tied to model_path for per-model history
chats (id, title, folder_id, model_id, model_path, timestamps)

-- Messages, folders, chat_overrides, settings: supporting tables
```

### IPC Channels

**Session Management:**
| Channel | Description |
|---------|-------------|
| `sessions:list/get/create/start/stop/delete/detect/update` | Full session CRUD + lifecycle |
| `session:starting/ready/stopped/error/health/log/created/deleted` | Real-time events |

**Chat:**
| Channel | Description |
|---------|-------------|
| `chat:create/get/getAll/getByModel/getMessages/sendMessage/delete/search` | Chat CRUD |
| `chat:abort` | Cancel active generation |
| `chat:stream/complete` | SSE streaming events |

**vLLM-MLX Management:**
| Channel | Description |
|---------|-------------|
| `vllm:check-installation/detect-installers/check-updates` | Detection |
| `vllm:install-streaming/cancel-install/update` | Install/update with streaming output |
| `vllm:install-log/install-complete` | Streaming events |

**Models:**
| Channel | Description |
|---------|-------------|
| `models:scan/info/getDirectories/addDirectory/removeDirectory/browseDirectory` | Model management |

---

## Requirements

- macOS 12+ (Apple Silicon recommended)
- Node.js 18+
- vLLM-MLX installed (auto-installed on first launch, or manually via `uv tool install vllm-mlx`)
- MLX-format models (configurable scan directories, defaults: `~/.lmstudio/models/`, `~/.cache/huggingface/hub/`)

---

## Development

```bash
npm run dev          # Development mode (hot reload)
npm run build        # Build for production
npm run typecheck    # TypeScript validation
npm run lint         # ESLint
```

### Project Structure

```
src/
├── main/                           # Electron main process
│   ├── index.ts                    # App lifecycle, startup
│   ├── sessions.ts                 # SessionManager (multi-instance)
│   ├── database.ts                 # SQLite WAL schema + queries
│   ├── vllm-manager.ts            # vLLM-MLX install/update/detect
│   ├── server.ts                   # Legacy ServerManager (reference only)
│   └── ipc/
│       ├── sessions.ts             # Session IPC handlers
│       ├── chat.ts                 # Chat IPC + SSE streaming + abort
│       ├── models.ts               # Model scanning + directories
│       └── vllm.ts                 # vLLM-MLX install/update handlers
├── renderer/                       # React UI
│   └── src/
│       ├── App.tsx                 # View routing
│       ├── components/
│       │   ├── setup/              # SetupScreen (first-run installer)
│       │   ├── sessions/           # Dashboard, Card, Create, View, Settings, ConfigForm, ServerSettingsDrawer
│       │   ├── chat/               # ChatInterface, ChatList, ChatSettings, Messages
│       │   └── update/             # UpdateManager
│       └── index.css               # Tailwind + custom classes
└── preload/
    ├── index.ts                    # IPC bridge (contextBridge)
    └── index.d.ts                  # TypeScript declarations
```

---

## Distribution

```bash
npm run build                        # Build app
npx electron-builder --mac --dir     # Package as .app
```

Output: `release/mac-arm64/vMLX.app`

Deploy: `cp -R release/mac-arm64/vMLX.app /Applications/`

---

## Troubleshooting

### vLLM-MLX not found
On first launch, the app offers one-click install. If you prefer manual install:
```bash
uv tool install vllm-mlx          # Recommended (fastest)
pip3 install vllm-mlx             # Alternative (needs Python 3.10+)
```

### Models not detected
Add custom model directories via the directory manager in the Create Session wizard. Default scan locations:
- `~/.lmstudio/models/`
- `~/.cache/huggingface/hub/`

### Session won't start
- Check the loading screen logs for errors
- Verify the model path exists and contains valid MLX weights
- Ensure the port isn't already in use (the app auto-assigns ports)
- Check system memory — large models need significant RAM

### Chat history missing
- Chat history is tied to exact `modelPath`. If you moved the model, chats won't appear.
- Database location: `~/Library/Application Support/vllm-mlx-panel/chats.db`

### Process detected but not adopted
- Click "Detect Processes" on the dashboard
- The process must respond to `/health` endpoint
- Only processes running `vllm-mlx serve` are detected

---

## Credits

- **vLLM-MLX** by [ml-explore](https://github.com/ml-explore/vllm-mlx)
- **Electron** by OpenJS Foundation

---

## License

MIT License
