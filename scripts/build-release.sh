#!/usr/bin/env bash
# build-release.sh — vMLX Swift: regenerate Xcode project, archive, export,
# notarize, staple, and package into a notarized DMG.
#
# Usage:
#   cd swift
#   source .env.signing
#   scripts/build-release.sh [VERSION]
#
# Env vars required (sourced from .env.signing):
#   NOTARIZE_APPLE_ID        Apple ID used for notarytool
#   NOTARIZE_TEAM_ID         Developer Team ID (e.g. ABCDE12345)
#   NOTARIZE_PASSWORD        App-specific password for notarytool
#   CODESIGN_IDENTITY        Developer ID Application cert common name
#
# Optional:
#   VERSION                  Marketing version (defaults to project.yml value)
#   SKIP_NOTARIZE=1          Skip notarization (local smoke-test builds)

set -euo pipefail

SWIFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SWIFT_DIR"

VERSION="${1:-${VERSION:-}}"
if [[ -z "${VERSION}" ]]; then
    VERSION=$(grep -A1 'MARKETING_VERSION' project.yml | tail -n1 | sed -E 's/.*"([^"]+)".*/\1/' || echo "2.0.0")
fi

ARCHIVE_PATH="$SWIFT_DIR/release/vMLX-$VERSION.xcarchive"
EXPORT_PATH="$SWIFT_DIR/release/export-$VERSION"
DMG_PATH="$SWIFT_DIR/release/vMLX-$VERSION-arm64.dmg"
EXPORT_PLIST="$SWIFT_DIR/release/export-options.plist"

mkdir -p "$SWIFT_DIR/release"

echo "==> [1/6] Regenerating Xcode project"
if ! command -v xcodegen >/dev/null 2>&1; then
    echo "ERROR: xcodegen not installed. Run: brew install xcodegen" >&2
    exit 1
fi
xcodegen generate --spec project.yml --project .

echo "==> [2/6] Writing export options plist"
cat > "$EXPORT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>developer-id</string>
    <key>teamID</key>
    <string>${NOTARIZE_TEAM_ID}</string>
    <key>signingStyle</key>
    <string>manual</string>
    <key>signingCertificate</key>
    <string>${CODESIGN_IDENTITY:-Developer ID Application}</string>
    <key>destination</key>
    <string>export</string>
</dict>
</plist>
EOF

echo "==> [3/6] Archiving (xcodebuild archive)"
xcodebuild archive \
    -project vMLX.xcodeproj \
    -scheme vMLX \
    -configuration Release \
    -destination "generic/platform=macOS" \
    -archivePath "$ARCHIVE_PATH" \
    MARKETING_VERSION="$VERSION" \
    ONLY_ACTIVE_ARCH=NO \
    | xcbeautify 2>/dev/null || true

echo "==> [4/6] Exporting signed .app"
xcodebuild -exportArchive \
    -archivePath "$ARCHIVE_PATH" \
    -exportPath "$EXPORT_PATH" \
    -exportOptionsPlist "$EXPORT_PLIST"

APP_PATH="$EXPORT_PATH/vMLX.app"
if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: vMLX.app not found at $APP_PATH" >&2
    exit 1
fi

if [[ "${SKIP_NOTARIZE:-0}" != "1" ]]; then
    echo "==> [5/6] Notarizing (xcrun notarytool submit)"
    if [[ -z "${NOTARIZE_APPLE_ID:-}" || -z "${NOTARIZE_PASSWORD:-}" || -z "${NOTARIZE_TEAM_ID:-}" ]]; then
        echo "ERROR: notarization creds missing. Source .env.signing first." >&2
        exit 1
    fi

    # Zip the app for notarytool.
    ZIP_PATH="$EXPORT_PATH/vMLX-$VERSION.zip"
    /usr/bin/ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

    xcrun notarytool submit "$ZIP_PATH" \
        --apple-id "$NOTARIZE_APPLE_ID" \
        --team-id "$NOTARIZE_TEAM_ID" \
        --password "$NOTARIZE_PASSWORD" \
        --wait

    echo "==> Stapling ticket"
    xcrun stapler staple "$APP_PATH"
else
    echo "==> [5/6] SKIP_NOTARIZE=1 — skipping notarization"
fi

echo "==> [6/6] Building DMG"
rm -f "$DMG_PATH"
if command -v create-dmg >/dev/null 2>&1; then
    create-dmg \
        --volname "vMLX $VERSION" \
        --window-size 540 380 \
        --icon-size 96 \
        --app-drop-link 400 180 \
        --icon "vMLX.app" 140 180 \
        "$DMG_PATH" \
        "$APP_PATH" || true
fi
if [[ ! -f "$DMG_PATH" ]]; then
    # Fallback: hdiutil
    /usr/bin/hdiutil create \
        -volname "vMLX $VERSION" \
        -srcfolder "$APP_PATH" \
        -ov -format UDZO \
        "$DMG_PATH"
fi

if [[ "${SKIP_NOTARIZE:-0}" != "1" ]]; then
    echo "==> Stapling DMG"
    xcrun stapler staple "$DMG_PATH" || true
fi

echo ""
echo "==> Done: $DMG_PATH"
ls -lh "$DMG_PATH"
