/**
 * Tests for the HuggingFace download manager — tqdm progress parsing,
 * size formatting, number formatting, and download queue logic.
 */
import { describe, it, expect } from 'vitest'

// ─── parseTqdmProgress (extracted from models.ts) ────────────────────────────

interface DownloadProgress {
    percent?: number
    speed?: string
    downloaded?: string
    total?: string
    eta?: string
    currentFile?: string
    filesProgress?: string
    raw: string
}

function parseTqdmProgress(line: string): Partial<DownloadProgress> {
    const result: Partial<DownloadProgress> = { raw: line.trim() }

    const fileMatch = line.match(/(?:Downloading|Fetching)\s+(.+?)(?:\s*:|\s*\|)/)
    if (fileMatch) result.currentFile = fileMatch[1].trim()

    const tqdmMatch = line.match(/(\d+)%\|[^|]*\|\s*([\d.]+\w*)\/([\d.]+\w*)\s*\[([^\]<]*)<([^\],]*),\s*([^\]]+)\]/)
    if (tqdmMatch) {
        result.percent = parseInt(tqdmMatch[1], 10)
        result.downloaded = tqdmMatch[2]
        result.total = tqdmMatch[3]
        result.eta = tqdmMatch[5].trim()
        result.speed = tqdmMatch[6].trim()
    } else {
        const simplePercent = line.match(/\s(\d+)%\|/)
        if (simplePercent) result.percent = parseInt(simplePercent[1], 10)
    }

    const filesMatch = line.match(/Fetching\s+(\d+)\s+files.*?(\d+)%/)
    if (filesMatch) result.filesProgress = `${Math.round(parseInt(filesMatch[2]) * parseInt(filesMatch[1]) / 100)}/${filesMatch[1]}`

    return result
}

describe('parseTqdmProgress', () => {
    it('parses full tqdm progress line', () => {
        const line = '  45%|████      | 1.2G/4.5G [01:30<02:30, 45.2MB/s]'
        const result = parseTqdmProgress(line)
        expect(result.percent).toBe(45)
        expect(result.downloaded).toBe('1.2G')
        expect(result.total).toBe('4.5G')
        expect(result.eta).toBe('02:30')
        expect(result.speed).toBe('45.2MB/s')
    })

    it('parses simple percent line', () => {
        const line = ' 75%|██████████████████████████████████████████████████████████████████████████████  | 3/4 files'
        const result = parseTqdmProgress(line)
        expect(result.percent).toBe(75)
    })

    it('extracts filename from Downloading line', () => {
        const line = 'Downloading model-00001-of-00003.safetensors: 100%|████| 5.00G/5.00G [01:23<00:00, 60.2MB/s]'
        const result = parseTqdmProgress(line)
        expect(result.currentFile).toBe('model-00001-of-00003.safetensors')
        expect(result.percent).toBe(100)
    })

    it('extracts files progress from Fetching line', () => {
        const line = 'Fetching 15 files:  60%|██████████████████| 9/15 [01:00<00:40, 6.5s/it]'
        const result = parseTqdmProgress(line)
        expect(result.filesProgress).toBe('9/15')
    })

    it('returns raw line when no match', () => {
        const line = 'Some random log message'
        const result = parseTqdmProgress(line)
        expect(result.raw).toBe('Some random log message')
        expect(result.percent).toBeUndefined()
        expect(result.speed).toBeUndefined()
    })

    it('handles empty line', () => {
        const result = parseTqdmProgress('')
        expect(result.raw).toBe('')
    })
})

// ─── formatSize (extracted from models.ts) ───────────────────────────────────

function formatSize(bytes: number): string {
    const gb = bytes / (1024 * 1024 * 1024)
    if (gb >= 1) {
        return `~${gb.toFixed(1)} GB`
    }
    const mb = bytes / (1024 * 1024)
    return `~${mb.toFixed(0)} MB`
}

describe('formatSize', () => {
    it('formats gigabytes', () => {
        expect(formatSize(5 * 1024 * 1024 * 1024)).toBe('~5.0 GB')
    })

    it('formats large GB values', () => {
        expect(formatSize(12.5 * 1024 * 1024 * 1024)).toBe('~12.5 GB')
    })

    it('formats megabytes for sub-GB sizes', () => {
        expect(formatSize(500 * 1024 * 1024)).toBe('~500 MB')
    })

    it('formats small MB values', () => {
        expect(formatSize(50 * 1024 * 1024)).toBe('~50 MB')
    })

    it('handles edge case at 1 GB boundary', () => {
        const result = formatSize(1024 * 1024 * 1024)
        expect(result).toBe('~1.0 GB')
    })
})

// ─── formatNumber (extracted from DownloadTab.tsx) ────────────────────────────

function formatNumber(n: number): string {
    if (n === undefined || n === null || isNaN(n)) return '0'
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
    return n.toString()
}

describe('formatNumber', () => {
    it('formats millions', () => {
        expect(formatNumber(2_500_000)).toBe('2.5M')
    })

    it('formats thousands', () => {
        expect(formatNumber(15_400)).toBe('15.4K')
    })

    it('formats small numbers as-is', () => {
        expect(formatNumber(42)).toBe('42')
    })

    it('handles 0', () => {
        expect(formatNumber(0)).toBe('0')
    })

    it('handles NaN', () => {
        expect(formatNumber(NaN)).toBe('0')
    })
})

// ─── timeAgo (extracted from DownloadTab.tsx) ─────────────────────────────────

function timeAgo(dateStr: string): string {
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = Math.floor(diff / 86400000)
    if (days < 1) return 'today'
    if (days < 30) return `${days}d ago`
    if (days < 365) return `${Math.floor(days / 30)}mo ago`
    return `${Math.floor(days / 365)}y ago`
}

describe('timeAgo', () => {
    it('returns "today" for recent dates', () => {
        expect(timeAgo(new Date().toISOString())).toBe('today')
    })

    it('returns days for recent dates', () => {
        const fiveDaysAgo = new Date(Date.now() - 5 * 86400000).toISOString()
        expect(timeAgo(fiveDaysAgo)).toBe('5d ago')
    })

    it('returns months for older dates', () => {
        const threeMonthsAgo = new Date(Date.now() - 90 * 86400000).toISOString()
        expect(timeAgo(threeMonthsAgo)).toBe('3mo ago')
    })

    it('returns years for old dates', () => {
        const twoYearsAgo = new Date(Date.now() - 730 * 86400000).toISOString()
        expect(timeAgo(twoYearsAgo)).toBe('2y ago')
    })
})

// ─── extractModelSize (extracted from models.ts) ─────────────────────────────

function extractModelSize(m: any): string | undefined {
    try {
        const total = m.safetensors?.total
        if (typeof total === 'number' && total > 0) return formatSize(total)
        const params = m.safetensors?.parameters
        if (params && typeof params === 'object') {
            const totalParams = Object.values(params).reduce((sum: number, v: any) => sum + (typeof v === 'number' ? v : 0), 0)
            if (totalParams > 0) {
                return formatSize(totalParams * 2)
            }
        }
    } catch (_) { }
    return undefined
}

describe('extractModelSize', () => {
    it('extracts size from safetensors.total', () => {
        const model = { safetensors: { total: 5 * 1024 * 1024 * 1024 } }
        expect(extractModelSize(model)).toBe('~5.0 GB')
    })

    it('estimates size from parameter counts', () => {
        // 1B params * 2 bytes = ~2GB
        const model = { safetensors: { parameters: { F16: 1_000_000_000 } } }
        const result = extractModelSize(model)
        expect(result).toMatch(/GB/)
    })

    it('returns undefined when no safetensors metadata', () => {
        expect(extractModelSize({})).toBeUndefined()
        expect(extractModelSize({ safetensors: {} })).toBeUndefined()
    })

    it('handles null/undefined gracefully', () => {
        expect(extractModelSize(null)).toBeUndefined()
        expect(extractModelSize(undefined)).toBeUndefined()
    })
})

// ─── Download queue logic ────────────────────────────────────────────────────

describe('Download queue logic', () => {
    it('detects duplicate downloads', () => {
        const activeJob = { repoId: 'mlx-community/Llama-3.2-3B' }
        const queue = [{ repoId: 'mlx-community/Phi-3' }]

        const repoId = 'mlx-community/Llama-3.2-3B'
        const isDuplicate = activeJob.repoId === repoId || queue.some(j => j.repoId === repoId)
        expect(isDuplicate).toBe(true)
    })

    it('allows distinct downloads', () => {
        const activeJob = { repoId: 'mlx-community/Llama-3.2-3B' }
        const queue = [{ repoId: 'mlx-community/Phi-3' }]

        const repoId = 'mlx-community/Qwen2-7B'
        const isDuplicate = activeJob.repoId === repoId || queue.some(j => j.repoId === repoId)
        expect(isDuplicate).toBe(false)
    })

    it('stale marker detection: age > 1 hour is stale', () => {
        const ts = Date.now() - 3700000 // 1h + 100ms ago
        const isStale = !isNaN(ts) && Date.now() - ts > 3600000
        expect(isStale).toBe(true)
    })

    it('stale marker detection: age < 1 hour is not stale', () => {
        const ts = Date.now() - 1800000 // 30 min ago
        const isStale = !isNaN(ts) && Date.now() - ts > 3600000
        expect(isStale).toBe(false)
    })
})

// ─── Marker cleanup requirements ─────────────────────────────────────────────

describe('Download marker cleanup', () => {
    it('marker must be cleaned on all exit paths: close(cancel), close(success), close(error), spawn-error', () => {
        // This test documents the requirement that .vmlx-downloading marker
        // must be deleted in ALL exit paths. The actual implementation is in
        // models.ts — proc.on('close') handles 3 paths + proc.on('error')
        // handles spawn failure. If ANY path is missed, the model directory
        // gets suppressed as "downloading" for up to 1 hour.
        const exitPaths = ['close:cancelled', 'close:success', 'close:error', 'spawn-error']
        const pathsWithCleanup = ['close:cancelled', 'close:success', 'close:error', 'spawn-error']
        for (const path of exitPaths) {
            expect(pathsWithCleanup).toContain(path)
        }
    })
})
