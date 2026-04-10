import { BrowserWindow } from 'electron'
import { net } from 'electron'

// Query BOTH sources and take the highest version — mlx.studio drifts stale
// when CI release workflow only updates GitHub, so relying on first-hit order
// silently blocks updates for anyone past the stale source's version.
const LATEST_URLS = [
  'https://raw.githubusercontent.com/jjang-ai/mlxstudio/main/latest.json',
  'https://mlx.studio/update/latest.json',
]
const CHECK_DELAY_MS = 5000 // Wait 5s after startup before checking

interface LatestRelease {
  version: string
  url: string
  notes?: string
}

function parseVersion(v: string): number[] {
  const clean = v.replace(/-.*$/, '') // Strip pre-release suffix
  return clean.split('.').map(Number)
}

// Returns >0 if a > b, <0 if a < b, 0 if equal. NaN components → 0.
function versionCompare(a: string, b: string): number {
  const av = parseVersion(a)
  const bv = parseVersion(b)
  for (let i = 0; i < Math.max(av.length, bv.length); i++) {
    const ai = av[i] ?? 0
    const bi = bv[i] ?? 0
    if (isNaN(ai) || isNaN(bi)) return 0
    if (ai > bi) return 1
    if (ai < bi) return -1
  }
  return 0
}

function compareVersions(current: string, latest: string): boolean {
  return versionCompare(latest, current) > 0
}

export function checkForUpdates(getWindow: () => BrowserWindow | null, currentVersion: string): void {
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
    let data: LatestRelease | null = null
    for (const r of results) {
      if (data === null || versionCompare(r.version, data.version) > 0) {
        data = r
      }
    }

    if (!data) {
      console.log('[UPDATE] All update sources failed')
      return
    }
    console.log(`[UPDATE] Picked highest version across sources: v${data.version}`)

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
