import { useState, useEffect } from 'react'

const DISMISS_KEY = 'vmlx-swift-migration-banner-dismissed'

// User audit 2026-05-01: clicking "Download Swift v2" used to land on
// https://github.com/jjang-ai/vmlx/releases/latest which now resolves
// to the LATEST RELEASE — that's the Python wheel (vmlx 1.5.0) not the
// Swift app DMG. Pin to the Swift releases tag list (filtered by tag
// prefix) so the user lands on a page where every visible asset is a
// Swift DMG. As Swift v2 promotes from beta → stable, only this URL
// changes.
const SWIFT_RELEASE_URL =
  'https://github.com/jjang-ai/vmlx/releases?q=tag%3Av2&expanded=true'
// Direct fallback if user wants the canonical current beta DMG.
const SWIFT_BETA_DMG_URL =
  'https://github.com/jjang-ai/vmlx/releases/download/v2.0.0-beta.15/vMLX-2.0.0-beta.15-arm64.dmg'

export function SwiftMigrationBanner() {
  const [dismissed, setDismissed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(DISMISS_KEY) === '1'
    } catch {
      return false
    }
  })

  useEffect(() => {
    if (dismissed) {
      try {
        localStorage.setItem(DISMISS_KEY, '1')
      } catch {
        // ignore
      }
    }
  }, [dismissed])

  if (dismissed) return null

  const onOpen = () => {
    // Resolve the actual Swift download URL from the runtime update
    // manifest when available. Latest manifest (mlxstudio/latest.json
    // ships `swiftBetaUrl`) wins over the hardcoded beta DMG link so
    // a user who hasn't updated the panel still gets the freshest
    // Swift DMG. Falls through to the tag list if the manifest hasn't
    // been fetched yet.
    const win = window as unknown as {
      api?: {
        openExternal?: (url: string) => void
        update?: { manifest?: () => Promise<{ swiftBetaUrl?: string } | null> }
      }
    }
    const open = (url: string) => {
      try {
        if (win.api?.openExternal) win.api.openExternal(url)
        else window.open(url, '_blank', 'noopener,noreferrer')
      } catch {
        window.open(url, '_blank', 'noopener,noreferrer')
      }
    }
    ;(async () => {
      try {
        const m = await win.api?.update?.manifest?.()
        if (m?.swiftBetaUrl) {
          open(m.swiftBetaUrl)
          return
        }
      } catch {
        // fall through to static URL
      }
      // Default: tag list filtered to Swift v2 releases. User picks the
      // latest .dmg asset from there. The hardcoded direct DMG URL
      // (SWIFT_BETA_DMG_URL) is also viable but a tag list is more
      // forward-compatible across beta cuts.
      open(SWIFT_RELEASE_URL)
    })()
  }
  // Reference to silence unused-import warnings in environments where
  // the static beta-DMG fallback is preferred over the tag list.
  void SWIFT_BETA_DMG_URL

  return (
    <div className="w-full bg-gradient-to-r from-orange-600 via-amber-600 to-orange-600 text-white shadow-md border-b border-amber-700/50">
      <div className="px-4 py-2.5 flex items-center justify-between gap-3 text-sm">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-base shrink-0">⚡</span>
          <span className="truncate">
            <strong className="font-semibold">vMLX v2 (Swift) is now the main app.</strong>{' '}
            Native Swift + Metal, 50–95 t/s on M-series (vs 11–60 t/s on this Python build), zero PyTorch in the hot path. This Python panel stays around for legacy support.
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onOpen}
            className="px-3 py-1 rounded-md bg-white/20 hover:bg-white/30 transition-colors font-medium"
          >
            Get vMLX v2 (Swift)
          </button>
          <button
            onClick={() => setDismissed(true)}
            className="px-2 py-1 rounded-md hover:bg-white/15 transition-colors"
            aria-label="Dismiss"
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  )
}
