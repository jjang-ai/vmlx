#!/bin/bash
set -e

echo "=== Step 1: Sync Python changes to macstudio ==="
rsync -avz --compress /Users/eric/mlx/vllm-mlx/vllm_mlx/ macstudio:/Users/eric/mlx/vllm-mlx/vllm_mlx/ 2>&1 | tail -3

echo "=== Step 2: Build Electron app ==="
cd /Users/eric/mlx/vllm-mlx/panel
npm run build 2>&1 | tail -5

echo "=== Step 3: Package app ==="
npx electron-builder --mac --dir 2>&1 | tail -5

echo "=== Step 4: Create tarball ==="
APP_PATH=$(find /Users/eric/mlx/vllm-mlx/panel/release -name "vMLX.app" -type d -maxdepth 3 | head -1)
if [ -z "$APP_PATH" ]; then
  echo "ERROR: Could not find vMLX.app in release/"
  exit 1
fi
echo "Found: $APP_PATH"
cd "$(dirname "$APP_PATH")"
tar czf /tmp/vMLX.tar.gz vMLX.app
echo "Tarball: $(du -h /tmp/vMLX.tar.gz | cut -f1)"

echo "=== Step 5: Close vMLX on macstudio (NOT exploitbot) ==="
ssh macstudio "osascript -e 'tell application \"vMLX\" to quit' 2>/dev/null; sleep 2; killall vMLX 2>/dev/null || true"

echo "=== Step 6: Transfer ==="
scp /tmp/vMLX.tar.gz macstudio:/tmp/vMLX.tar.gz

echo "=== Step 7: Install ==="
ssh macstudio "rm -rf /Applications/vMLX.app && cd /tmp && tar xzf vMLX.tar.gz && mv vMLX.app /Applications/vMLX.app && rm vMLX.tar.gz && echo 'Installed'"

echo "=== Step 8: Launch ==="
ssh macstudio "open /Applications/vMLX.app"

echo "=== Step 9: Verify ==="
sleep 5
ssh macstudio "ps aux | grep 'MacOS/vMLX' | grep -v grep | head -3"

echo "=== DONE ==="
