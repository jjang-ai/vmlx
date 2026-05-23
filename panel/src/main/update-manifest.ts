export interface ReleaseAsset {
  url: string
  sha256?: string
}

export interface LatestRelease {
  version: string
  url: string
  sha256?: string
  notes?: string
  tahoeUrl?: string
  tahoeSha256?: string
  downloads?: {
    sequoia?: ReleaseAsset
    tahoe?: ReleaseAsset
  }
}

function parseVersion(v: string): number[] {
  const clean = v.replace(/-.*$/, '')
  return clean.split('.').map(Number)
}

export function versionCompare(a: string, b: string): number {
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

export function compareVersions(current: string, latest: string): boolean {
  return versionCompare(latest, current) > 0
}

export function isValidUpdateUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    const trusted = ['github.com', 'mlx.studio']
    return parsed.protocol === 'https:' && trusted.some(d => parsed.hostname === d || parsed.hostname.endsWith(`.${d}`))
  } catch {
    return false
  }
}

function macOSMajorVersion(systemVersion?: string): number | null {
  const major = Number(String(systemVersion || '').split('.')[0])
  return Number.isFinite(major) ? major : null
}

function currentMacOSVersion(): string | undefined {
  const getter = (process as unknown as { getSystemVersion?: () => string }).getSystemVersion
  return typeof getter === 'function' ? getter() : undefined
}

export function selectDownloadForMacOS(
  release: LatestRelease,
  systemVersion = currentMacOSVersion(),
): LatestRelease {
  const major = macOSMajorVersion(systemVersion)
  if (major === null || major < 26) {
    const sequoia = release.downloads?.sequoia
    return sequoia?.url ? { ...release, url: sequoia.url, sha256: sequoia.sha256 ?? release.sha256 } : release
  }

  const tahoe = release.downloads?.tahoe
  if (tahoe?.url) {
    return { ...release, url: tahoe.url, sha256: tahoe.sha256 }
  }
  if (release.tahoeUrl) {
    return { ...release, url: release.tahoeUrl, sha256: release.tahoeSha256 }
  }
  if (release.url.includes('-sequoia-arm64.dmg')) {
    return {
      ...release,
      url: release.url.replace('-sequoia-arm64.dmg', '-tahoe-arm64.dmg'),
      sha256: undefined,
    }
  }
  return release
}

export function selectHighestRelease(results: LatestRelease[]): LatestRelease | null {
  let data: LatestRelease | null = null
  for (const r of results) {
    if (data === null || versionCompare(r.version, data.version) > 0) {
      data = r
    }
  }
  return data
}
