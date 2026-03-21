import { BrowserWindow } from 'electron'
import { net } from 'electron'

const LATEST_URLS = [
  'https://mlx.studio/update/latest.json',
  'https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json',
]
const CHECK_DELAY_MS = 5000 // Wait 5s after startup before checking

interface LatestRelease {
  version: string
  url: string
  notes?: string
}

function compareVersions(current: string, latest: string): boolean {
  // Strip pre-release suffixes (e.g., "1.2.0-beta.1" → "1.2.0")
  const clean = (v: string) => v.replace(/-.*$/, '')
  const a = clean(current).split('.').map(Number)
  const b = clean(latest).split('.').map(Number)
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] ?? 0
    const bv = b[i] ?? 0
    if (isNaN(av) || isNaN(bv)) return false
    if (bv > av) return true
    if (bv < av) return false
  }
  return false
}

export function checkForUpdates(getWindow: () => BrowserWindow | null, currentVersion: string): void {
  setTimeout(async () => {
    let data: LatestRelease | null = null

    // Try each update URL in order (mlx.studio first, GitHub fallback)
    for (const url of LATEST_URLS) {
      try {
        const response = await net.fetch(url, { method: 'GET' })
        if (!response.ok) {
          console.log(`[UPDATE] ${url}: HTTP ${response.status}`)
          continue
        }
        const parsed = await response.json()
        if (parsed.version && parsed.url) {
          data = parsed as LatestRelease
          console.log(`[UPDATE] Fetched manifest from ${url}`)
          break
        }
      } catch (err) {
        console.log(`[UPDATE] ${url}: ${(err as Error).message}`)
      }
    }

    if (!data) {
      console.log('[UPDATE] All update sources failed')
      return
    }

    // Only accept HTTPS URLs from trusted domains
    try {
      const parsed = new URL(data.url)
      const trusted = ['github.com', 'mlx.studio']
      if (parsed.protocol !== 'https:' || !trusted.some(d => parsed.hostname === d || parsed.hostname.endsWith(`.${d}`))) {
        console.log(`[UPDATE] Rejected untrusted URL: ${data.url}`)
        return
      }
    } catch {
      console.log(`[UPDATE] Invalid URL in manifest: ${data.url}`)
      return
    }

    if (compareVersions(currentVersion, data.version)) {
      console.log(`[UPDATE] New version available: ${currentVersion} → ${data.version}`)
      const win = getWindow()
      if (win && !win.isDestroyed()) {
        win.webContents.send('app:updateAvailable', {
          currentVersion,
          latestVersion: data.version,
          url: data.url,
          notes: data.notes
        })
      }
    } else {
      console.log(`[UPDATE] Up to date (${currentVersion})`)
    }
  }, CHECK_DELAY_MS)
}
