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
#   NOTARIZE_TEAM_ID         Developer Team ID (e.g. 55KGF2S5AY)
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

# Keep only one DMG on disk at a time. Any prior vMLX-*.dmg / stale
# xcarchive / export-* dirs are pruned so `release/` never accumulates
# old betas. 2026-04-18 preference: ship-or-delete.
find "$SWIFT_DIR/release" -maxdepth 1 -type f -name 'vMLX-*-arm64.dmg' \
     ! -name "$(basename "$DMG_PATH")" -delete 2>/dev/null || true
find "$SWIFT_DIR/release" -maxdepth 1 -type f \( -name 'vMLX-*.zip' -o -name 'vMLX-*-rev*.zip' \) \
     ! -name "vMLX-$VERSION.zip" -delete 2>/dev/null || true
find "$SWIFT_DIR/release" -maxdepth 1 -type d \( -name '*.xcarchive' -o -name 'export-*' -o -name 'beta*-export' \) \
     ! -name "$(basename "$ARCHIVE_PATH")" \
     ! -name "$(basename "$EXPORT_PATH")" \
     -prune -exec rm -rf {} + 2>/dev/null || true

# 2026-04-18: Switched from `xcodebuild archive` + `-exportArchive` to
# a direct SwiftPM build + manual bundle. Rationale:
#   1. Xcode 26 changed the `exportOptionsPlist` method enumeration;
#      every documented value (`developer-id`, `developer-id-with-profile`,
#      `mac-application`, etc) returns the cryptic
#      `error: exportArchive exportOptionsPlist error for key "method"
#      expected one {} but found <value>`. Apple's documentation has
#      not caught up.
#   2. `xcodebuild archive` on our SwiftPM-packaged scheme produces
#      an empty Products/Applications subdirectory, so even if the
#      export plist was correct, there's nothing to export.
#   3. The SwiftPM build we already run for `swift test` + CI produces
#      a working release executable at
#      `.build/arm64-apple-macosx/release/vMLX`. Copying that into a
#      minimal .app bundle with Info.plist + metallib + bundled
#      resource bundles + Developer ID signature yields the exact
#      same shippable artifact Xcode archive would, with zero
#      dependency on Xcode's undocumented plist format.
# The xcodegen step still runs because it wires up the Info.plist
# template, entitlements, and asset catalog that we copy into the
# bundle below.

echo "==> [1/5] Regenerating Xcode project (for Info.plist + entitlements)"
if ! command -v xcodegen >/dev/null 2>&1; then
    echo "ERROR: xcodegen not installed. Run: brew install xcodegen" >&2
    exit 1
fi
xcodegen generate --spec project.yml --project .

echo "==> [2/5] SwiftPM release build (vMLX executable)"
swift build -c release --product vMLX
SWIFT_BIN="$SWIFT_DIR/.build/arm64-apple-macosx/release/vMLX"
if [[ ! -f "$SWIFT_BIN" ]]; then
    echo "ERROR: SwiftPM build did not produce $SWIFT_BIN" >&2
    exit 1
fi

echo "==> [3/5] Staging .app bundle"
APP_PATH="$EXPORT_PATH/vMLX.app"
rm -rf "$EXPORT_PATH"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
cp "$SWIFT_BIN" "$APP_PATH/Contents/MacOS/vMLX"
cp vMLX/Info.plist "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" \
    "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $VERSION" \
    "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion ${VERSION##*.}" \
    "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string ${VERSION##*.}" \
    "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleExecutable vMLX" \
    "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleExecutable string vMLX" \
    "$APP_PATH/Contents/Info.plist"
# Pre-built Metal library — required by MLX's `load_swiftpm_library`.
cp Sources/Cmlx/default.metallib "$APP_PATH/Contents/Resources/"
# SwiftPM produces per-target resource bundles at .build/.../release.
# They must sit alongside the executable for the runtime to find them.
find .build/arm64-apple-macosx/release -maxdepth 1 -name '*.bundle' \
     -type d -exec cp -R {} "$APP_PATH/Contents/Resources/" \;

echo "==> [4/5] Code signing with Developer ID + hardened runtime"
codesign --force --deep --options runtime \
    --entitlements vMLX/vMLX.entitlements \
    --timestamp \
    --sign "${CODESIGN_IDENTITY:-Developer ID Application: ShieldStack LLC (55KGF2S5AY)}" \
    "$APP_PATH"
codesign --verify --verbose "$APP_PATH"

if [[ "${SKIP_NOTARIZE:-0}" != "1" ]]; then
    echo "==> [5/5] Notarizing (xcrun notarytool submit)"
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
    # DMGs must be notarized SEPARATELY from the .app they contain —
    # Apple's notary service tickets each artifact individually, and
    # `stapler staple` on a fresh-built DMG fails with
    # `Could not find base64 encoded ticket in response` unless the
    # DMG itself has been submitted. Discovered shipping beta.3 + .4
    # (2026-04-18) — the stapled .app inside is still Gatekeeper-OK,
    # but a stapled DMG is strictly better because Gatekeeper can
    # verify the ticket offline while the user is still dragging.
    echo "==> Notarizing DMG (separate submission)"
    xcrun notarytool submit "$DMG_PATH" \
        --apple-id "$NOTARIZE_APPLE_ID" \
        --team-id "$NOTARIZE_TEAM_ID" \
        --password "$NOTARIZE_PASSWORD" \
        --wait || true
    echo "==> Stapling DMG"
    xcrun stapler staple "$DMG_PATH" || true
    echo "==> Validating DMG"
    xcrun stapler validate "$DMG_PATH" || true
fi

# Final housekeeping: the export-$VERSION dir contains the raw .app
# bundle, already captured inside the DMG. Nuke it so `release/`
# has exactly one shippable artifact.
rm -rf "$EXPORT_PATH" "$ARCHIVE_PATH" 2>/dev/null || true

echo ""
echo "==> Done: $DMG_PATH"
ls -lh "$DMG_PATH"
echo ""
echo "SHA256: $(shasum -a 256 "$DMG_PATH" | awk '{print $1}')"
