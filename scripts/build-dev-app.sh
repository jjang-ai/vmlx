#!/usr/bin/env bash
set -euo pipefail

SWIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SWIFT_DIR/.." && pwd)"
APP="${1:-$REPO_ROOT/build/vMLX-Swift-Dev.app}"

cd "$SWIFT_DIR"

swift build -c release --product vMLX

BIN="$SWIFT_DIR/.build/arm64-apple-macosx/release/vMLX"
BUILD_DIR="$SWIFT_DIR/.build/arm64-apple-macosx/release"
RESOURCES="$APP/Contents/Resources"

case "$APP" in
  ""|"/"|"$HOME"|"$SWIFT_DIR"|"$REPO_ROOT")
    echo "build-dev-app: refusing unsafe app path: $APP" >&2
    exit 1
    ;;
esac

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RESOURCES"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>vMLX</string>
  <key>CFBundleIdentifier</key><string>ai.jang.vmlx.swift.dev</string>
  <key>CFBundleName</key><string>vMLX Swift Dev</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict>
</plist>
PLIST

cp "$BIN" "$APP/Contents/MacOS/vMLX"
chmod +x "$APP/Contents/MacOS/vMLX"

if [[ -f "$BUILD_DIR/vmlx_Cmlx.bundle/default.metallib" ]]; then
  cp "$BUILD_DIR/vmlx_Cmlx.bundle/default.metallib" "$RESOURCES/default.metallib"
  cp "$BUILD_DIR/vmlx_Cmlx.bundle/default.metallib" "$RESOURCES/mlx.metallib"
  cp "$BUILD_DIR/vmlx_Cmlx.bundle/default.metallib" "$APP/Contents/MacOS/mlx.metallib"
elif [[ -f "$SWIFT_DIR/Sources/Cmlx/default.metallib" ]]; then
  cp "$SWIFT_DIR/Sources/Cmlx/default.metallib" "$RESOURCES/default.metallib"
  cp "$SWIFT_DIR/Sources/Cmlx/default.metallib" "$RESOURCES/mlx.metallib"
  cp "$SWIFT_DIR/Sources/Cmlx/default.metallib" "$APP/Contents/MacOS/mlx.metallib"
else
  echo "build-dev-app: missing default.metallib" >&2
  exit 1
fi

find "$BUILD_DIR" -maxdepth 1 -name '*.bundle' -type d -exec cp -R {} "$RESOURCES/" \;

codesign --force --deep --sign - "$APP" >/dev/null
echo "$APP"
