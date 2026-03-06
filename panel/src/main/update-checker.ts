import { BrowserWindow } from 'electron'
import { net } from 'electron'

const LATEST_URL = 'https://raw.githubusercontent.com/vmlxllm/vmlx/main/latest.json'
const CHECK_DELAY_MS = 5000 // Wait 5s after startup before checking

interface LatestRelease {
  version: string
  url: string
  notes?: string
}

function compareVersions(current: string, latest: string): boolean {
  const a = current.split('.').map(Number)
  const b = latest.split('.').map(Number)
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] || 0
    const bv = b[i] || 0
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
      if (data.version && compareVersions(currentVersion, data.version)) {
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
