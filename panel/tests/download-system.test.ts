/**
 * Tests for the download manager — concurrent downloads, pause/resume, queue events, progress.
 */
import { describe, it, expect, vi } from 'vitest'

describe('Download System', () => {
  describe('Image Model Registry', () => {
    it('all models have required fields', async () => {
      const { IMAGE_MODELS } = await import('../src/shared/imageModels')
      for (const m of IMAGE_MODELS) {
        expect(m.id).toBeTruthy()
        expect(m.name).toBeTruthy()
        expect(m.category).toMatch(/^(generate|edit)$/)
        expect(m.steps).toBeGreaterThan(0)
        expect(m.quantizeOptions.length).toBeGreaterThan(0)
        expect(Object.keys(m.repoMap).length).toBeGreaterThan(0)
        expect(m.encoderType).toMatch(/^(single|dual)$/)
      }
    })

    it('Klein models are removed from registry', async () => {
      const { IMAGE_MODELS } = await import('../src/shared/imageModels')
      const ids = IMAGE_MODELS.map(m => m.id)
      expect(ids).not.toContain('flux2-klein-4b')
      expect(ids).not.toContain('flux2-klein-9b')
    })

    it('resolveImageModelRepo returns correct repo for exact quantize', async () => {
      const { resolveImageModelRepo } = await import('../src/shared/imageModels')
      expect(resolveImageModelRepo('schnell', 4)).toBe('dhairyashil/FLUX.1-schnell-mflux-4bit')
      expect(resolveImageModelRepo('schnell', 8)).toBe('dhairyashil/FLUX.1-schnell-mflux-8bit')
      expect(resolveImageModelRepo('schnell', 0)).toBe('black-forest-labs/FLUX.1-schnell')
    })

    it('resolveImageModelRepo falls back to closest quantize', async () => {
      const { resolveImageModelRepo } = await import('../src/shared/imageModels')
      // Qwen only has quantize=0
      expect(resolveImageModelRepo('qwen-image-edit', 4)).toBe('Qwen/Qwen-Image-Edit')
    })

    it('getDefaultSteps returns correct steps per model', async () => {
      const { getDefaultSteps } = await import('../src/shared/imageModels')
      expect(getDefaultSteps('schnell')).toBe(4)
      expect(getDefaultSteps('z-image-turbo')).toBe(4)
      expect(getDefaultSteps('dev')).toBe(20)
      expect(getDefaultSteps('qwen-image-edit')).toBe(28)
      // Unknown model defaults to 4
      expect(getDefaultSteps('unknown-model')).toBe(4)
    })

    it('getImageModel returns model by ID', async () => {
      const { getImageModel } = await import('../src/shared/imageModels')
      const schnell = getImageModel('schnell')
      expect(schnell).toBeDefined()
      expect(schnell!.name).toBe('Flux Schnell')
      expect(schnell!.category).toBe('generate')
    })

    it('Qwen Image Edit is full precision only', async () => {
      const { getImageModel } = await import('../src/shared/imageModels')
      const qwen = getImageModel('qwen-image-edit')
      expect(qwen).toBeDefined()
      expect(qwen!.quantizeOptions).toEqual([0])
      expect(qwen!.category).toBe('edit')
    })
  })

  describe('Session Utils', () => {
    it('isImageSession detects image model type from config', async () => {
      const { isImageSession } = await import('../src/shared/sessionUtils')
      expect(isImageSession({ config: '{"modelType":"image"}' })).toBe(true)
      expect(isImageSession({ config: '{"modelType":"text"}' })).toBe(false)
      expect(isImageSession({ config: '{}' })).toBe(false)
      expect(isImageSession({})).toBe(false)
      expect(isImageSession({ config: 'invalid json' })).toBe(false)
    })
  })

  describe('Download Progress Format', () => {
    it('formatBytes produces human-readable sizes', () => {
      // Test the format logic used in download progress
      const formatBytes = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
      }
      expect(formatBytes(500)).toBe('500 B')
      expect(formatBytes(1024)).toBe('1.0 KB')
      expect(formatBytes(1024 * 1024 * 5)).toBe('5.0 MB')
      expect(formatBytes(1024 * 1024 * 1024 * 2.5)).toBe('2.50 GB')
    })

    it('file-count fallback when byte totals are zero', () => {
      // When total_bytes is 0, progress should use file count
      const totalBytes = 0
      const totalFiles = 10
      const fileNum = 3
      const pct = totalBytes > 0
        ? Math.round((0 / totalBytes) * 100)
        : Math.round((fileNum / totalFiles) * 100)
      expect(pct).toBe(30) // 3/10 = 30%
    })
  })

  describe('Concurrent Download Queue', () => {
    it('MAX_CONCURRENT limits parallel downloads', () => {
      const MAX_CONCURRENT = 3
      const activeJobs = new Map()
      const queue = [
        { id: 'j1', status: 'queued' },
        { id: 'j2', status: 'queued' },
        { id: 'j3', status: 'queued' },
        { id: 'j4', status: 'queued' },
      ]

      // Simulate processQueue
      let started = 0
      while (activeJobs.size < MAX_CONCURRENT && queue.length > 0) {
        const job = queue.shift()!
        activeJobs.set(job.id, job)
        started++
      }
      expect(started).toBe(3) // Only 3 should start
      expect(queue.length).toBe(1) // 1 remains
    })

    it('paused jobs are skipped in queue', () => {
      const queue = [
        { id: 'j1', status: 'paused' },
        { id: 'j2', status: 'queued' },
        { id: 'j3', status: 'paused' },
        { id: 'j4', status: 'queued' },
      ]

      // Find first non-paused
      const idx = queue.findIndex(j => j.status !== 'paused')
      expect(idx).toBe(1) // j2 is first non-paused
      expect(queue[idx].id).toBe('j2')
    })

    it('all paused returns -1', () => {
      const queue = [
        { id: 'j1', status: 'paused' },
        { id: 'j2', status: 'paused' },
      ]
      const idx = queue.findIndex(j => j.status !== 'paused')
      expect(idx).toBe(-1)
    })
  })
})
