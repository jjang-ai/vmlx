#!/usr/bin/env bash
# notarize-only.sh — Submit an already-built vMLX.app (or DMG) to notarytool,
# wait for the verdict, and staple the ticket. Use this when `build-release.sh`
# already produced an archive but notarization failed or was skipped.
#
# Usage:
#   source .env.signing
#   scripts/notarize-only.sh path/to/vMLX.app
#   scripts/notarize-only.sh path/to/vMLX-2.0.0-arm64.dmg

set -euo pipefail

ARTIFACT="${1:-}"
if [[ -z "$ARTIFACT" || ! -e "$ARTIFACT" ]]; then
    echo "Usage: $0 <path-to-.app-or-.dmg>" >&2
    exit 1
fi

: "${NOTARIZE_APPLE_ID:?missing NOTARIZE_APPLE_ID — source .env.signing}"
: "${NOTARIZE_TEAM_ID:?missing NOTARIZE_TEAM_ID — source .env.signing}"
: "${NOTARIZE_PASSWORD:?missing NOTARIZE_PASSWORD — source .env.signing}"

SUBMIT_PATH="$ARTIFACT"
CLEANUP=""
if [[ "$ARTIFACT" == *.app ]]; then
    ZIP_PATH="$(dirname "$ARTIFACT")/$(basename "$ARTIFACT" .app).zip"
    /usr/bin/ditto -c -k --keepParent "$ARTIFACT" "$ZIP_PATH"
    SUBMIT_PATH="$ZIP_PATH"
    CLEANUP="$ZIP_PATH"
fi

echo "==> Submitting $SUBMIT_PATH to notarytool"
xcrun notarytool submit "$SUBMIT_PATH" \
    --apple-id "$NOTARIZE_APPLE_ID" \
    --team-id "$NOTARIZE_TEAM_ID" \
    --password "$NOTARIZE_PASSWORD" \
    --wait

echo "==> Stapling $ARTIFACT"
xcrun stapler staple "$ARTIFACT"

if [[ -n "$CLEANUP" ]]; then
    rm -f "$CLEANUP"
fi

echo "Done."
