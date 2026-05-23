import { BrowserWindow } from 'electron'
import { net } from 'electron'
import {
  compareVersions,
  isValidUpdateUrl,
  LatestRelease,
  selectDownloadForMacOS,
  selectHighestRelease,
} from './update-manifest'

// Query BOTH sources and take the highest version — mlx.studio drifts stale
// when CI release workflow only updates GitHub, so relying on first-hit order
// silently blocks updates for anyone past the stale source's version.
const LATEST_URLS = [
  'https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json',
  'https://mlx.studio/update/latest.json',
]
const CHECK_DELAY_MS = 5000 // Wait 5s after startup before checking

export function checkForUpdates(getWindow: () => BrowserWindow | null, currentVersion: string): void {
  if (process.env.VMLX_SKIP_UPDATE_CHECK === '1') {
    console.log('[UPDATE] Skipping update check because VMLX_SKIP_UPDATE_CHECK=1')
    return
  }

  setTimeout(async () => {
    // Fetch ALL sources in parallel and pick the highest version — prevents a
    // stale mirror from silently suppressing updates on newer clients.
    const fetches = LATEST_URLS.map(async (url) => {
      try {
        const response = await net.fetch(url, { method: 'GET' })
        if (!response.ok) {
          console.log(`[UPDATE] ${url}: HTTP ${response.status}`)
          return null
        }
        const parsed = await response.json()
        if (parsed.version && parsed.url) {
          console.log(`[UPDATE] Fetched manifest from ${url}: v${parsed.version}`)
          return parsed as LatestRelease
        }
      } catch (err) {
        console.log(`[UPDATE] ${url}: ${(err as Error).message}`)
      }
      return null
    })

    const results = (await Promise.all(fetches)).filter((r): r is LatestRelease => r != null)
    let data = selectHighestRelease(results)

    if (!data) {
      console.log('[UPDATE] All update sources failed')
      return
    }
    console.log(`[UPDATE] Picked highest version across sources: v${data.version}`)
    const selected = selectDownloadForMacOS(data)
    if (selected.url !== data.url) {
      console.log(`[UPDATE] Selected native macOS download: ${selected.url}`)
      data = selected
    }

    // Only accept HTTPS URLs from trusted domains
    if (!isValidUpdateUrl(data.url)) {
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
