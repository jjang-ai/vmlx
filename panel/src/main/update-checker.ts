import { BrowserWindow } from 'electron'
import { net } from 'electron'

const LATEST_URL = 'https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json'
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
    try {
      const response = await net.fetch(LATEST_URL, { method: 'GET' })
      if (!response.ok) {
        console.log(`[UPDATE] Check failed: HTTP ${response.status}`)
        return
      }
      const data: LatestRelease = await response.json()
      if (!data.version || !data.url) {
        console.log('[UPDATE] Invalid manifest: missing version or url')
        return
      }
      // Only accept HTTPS GitHub URLs to prevent redirect attacks if manifest is compromised
      try {
        const parsed = new URL(data.url)
        if (parsed.protocol !== 'https:' || !(parsed.hostname === 'github.com' || parsed.hostname.endsWith('.github.com'))) {
          console.log(`[UPDATE] Rejected non-GitHub URL: ${data.url}`)
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
    } catch (err) {
      console.log('[UPDATE] Check failed:', (err as Error).message)
    }
  }, CHECK_DELAY_MS)
}
