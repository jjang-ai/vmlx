# Shared build inputs for staleness checks. Source from other panel scripts.
: "${PANEL_DIR:?PANEL_DIR must be set}"
: "${BUNDLE_DIR:?BUNDLE_DIR must be set}"

BUILD_INPUT_PATHS=(
  "$PANEL_DIR/src"
  "$PANEL_DIR/dist"
  "$PANEL_DIR/package.json"
  "$PANEL_DIR/package-lock.json"
  "$PANEL_DIR/electron.vite.config.ts"
  "$PANEL_DIR/tsconfig.json"
  "$PANEL_DIR/tsconfig.node.json"
  "$PANEL_DIR/build"
  "$PANEL_DIR/scripts/electron-builder-before-pack.cjs"
  "$PANEL_DIR/scripts/electron-builder-after-pack.cjs"
  "$BUNDLE_DIR/.sync-stamp"
  "$BUNDLE_DIR/.bundle-stamp"
)
