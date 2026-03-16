# Setup Guide

## Prerequisites

- macOS 26+ (Tahoe) — Apple Silicon required (MLX Metal shaders require Metal 4.0)
- Node.js 18+
- vMLX Engine (auto-installed on first launch, or install manually)

## Quick Start

```bash
cd ~/mlx/vmlx-engine
npm install
npm run dev
```

## Installing vMLX Engine

vMLX auto-detects and offers to install vMLX Engine on first launch. You can also install manually:

```bash
# Recommended — uv manages its own Python venv
uv tool install vmlx-engine

# Alternative — needs Python 3.10+
pip3 install vmlx-engine
```

The app searches for `vmlx-engine` in these locations:
1. `~/.local/bin/vmlx-engine` (uv/pip user install)
2. `/opt/homebrew/bin/vmlx-engine` (Homebrew)
3. `/usr/local/bin/vmlx-engine`
4. Falls back to `which vmlx-engine`

## Development

```bash
npm run dev          # Electron + Vite hot reload
npm run build        # Production build
npm run typecheck    # TypeScript checking
npm run lint         # ESLint
```

## Building for Distribution

```bash
# Build the app
npm run build

# Package as .app bundle
npx electron-builder --mac --dir

# Output: release/mac-arm64/vMLX.app
```

## Deploying

```bash
# Kill existing, copy new, launch
pkill -9 -f "vMLX" 2>/dev/null
sleep 1
rm -rf /Applications/vMLX.app
cp -R release/mac-arm64/vMLX.app /Applications/
open /Applications/vMLX.app
```

## Cross-Machine Deployment (Mac Studio)

```bash
# Sync source files
rsync -avz --delete \
  --exclude node_modules --exclude dist --exclude out \
  --exclude release --exclude .git --exclude '*.db*' \
  ~/mlx/vmlx-engine/ macstudio:~/mlx/vmlx-engine/

# On Mac Studio: install, build, package, deploy
ssh macstudio 'echo "#!/bin/bash
cd /path/to/vmlx
npm ci
npm run build
npx electron-builder --mac --dir
pkill -9 -f vMLX 2>/dev/null; sleep 1
rm -rf /Applications/vMLX.app
cp -R release/mac-arm64/vMLX.app /Applications/
open /Applications/vMLX.app" > /tmp/deploy.sh && chmod +x /tmp/deploy.sh && /tmp/deploy.sh'
```

## Project Structure

```
src/
├── main/                       # Electron main process (Node.js)
│   ├── index.ts                # App lifecycle, startup adoption, quit
│   ├── sessions.ts             # SessionManager (multi-instance, health monitor)
│   ├── database.ts             # SQLite WAL schema + CRUD
│   ├── vllm-manager.ts         # vMLX Engine detect/install/update
│   └── ipc/                    # IPC handlers
│       ├── sessions.ts         # Session CRUD + lifecycle
│       ├── chat.ts             # Chat + SSE streaming + abort
│       ├── models.ts           # Model scanning + directories
│       └── vllm.ts             # Install/update streaming
├── renderer/                   # React UI
│   └── src/
│       ├── App.tsx             # View routing
│       └── components/
│           ├── setup/          # First-run installer
│           ├── sessions/       # Dashboard, Card, Create, View, Settings
│           ├── chat/           # Chat interface, list, settings, messages
│           └── update/         # vMLX Engine update manager
└── preload/                    # IPC bridge
    ├── index.ts                # contextBridge API
    └── index.d.ts              # TypeScript declarations
```

## Database

SQLite database at `~/Library/Application Support/vmlx-engine/chats.db`

Tables: `sessions`, `chats`, `messages`, `folders`, `chat_overrides`, `settings`

Uses WAL mode for concurrent read performance.

## Configuration

All session configuration is stored as JSON blobs in the `sessions` table. No external config files needed.

Model scan directories are stored in the `settings` table and configurable via the UI.
