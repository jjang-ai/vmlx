---
description: Build the vMLX Electron app and install to /Applications
---

# Build & Install vMLX

## Steps

1. Build the frontend (compiles TypeScript + React + Tailwind):
// turbo
```bash
cd /Users/eric/mlx/vllm-mlx/panel && npm run build
```

2. Package the Mac app (creates DMG + .app in release/):
```bash
cd /Users/eric/mlx/vllm-mlx/panel && npm run dist
```

3. Copy to /Applications (replaces existing):
```bash
rm -rf /Applications/vMLX.app && cp -R /Users/eric/mlx/vllm-mlx/panel/release/mac-arm64/vMLX.app /Applications/vMLX.app
```

## Notes
- Output: `panel/release/vMLX-0.1.0-arm64.dmg` and `panel/release/mac-arm64/vMLX.app`
- Code signing is disabled (`identity: null` in package.json)
- macOS minimum: 26.0.0 (Tahoe — MLX requires Metal 4.0)
- Architecture: arm64 only
