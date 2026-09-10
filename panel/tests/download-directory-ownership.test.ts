import { mkdtemp, mkdir, writeFile, readFile, access, rm } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { readFileSync } from 'fs'
import { describe, it, expect } from 'vitest'
import { prepareDownloadDirectory, removeOwnedDownloadDirectory, type DownloadDirectoryOwner } from '../src/main/downloadDirectoryOwnership'

describe('download cancellation directory ownership', () => {
  it.each(['active', 'paused', 'restored'])('preserves pre-existing files for %s cancellation', async (mode) => {
    const root = await mkdtemp(join(tmpdir(), 'vmlx-cancel-'))
    try {
      const modelDir = join(root, 'model')
      await mkdir(modelDir)
      await writeFile(join(modelDir, 'weights.safetensors'), 'existing-user-bytes')
      const job: DownloadDirectoryOwner = { modelDir }
      if (mode !== 'restored') await prepareDownloadDirectory(job)
      await writeFile(join(modelDir, 'partial.incomplete'), 'new-partial-bytes')
      await removeOwnedDownloadDirectory(job)
      expect(await readFile(join(modelDir, 'weights.safetensors'), 'utf8')).toBe('existing-user-bytes')
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  it('removes a newly owned destination and keeps ownership across pause/resume', async () => {
    const root = await mkdtemp(join(tmpdir(), 'vmlx-cancel-'))
    try {
      const job: DownloadDirectoryOwner = { modelDir: join(root, 'new-model') }
      await prepareDownloadDirectory(job)
      expect(job.ownsModelDir).toBe(true)
      await writeFile(join(job.modelDir, 'partial.incomplete'), 'partial')
      await prepareDownloadDirectory(job)
      expect(job.ownsModelDir).toBe(true)
      await removeOwnedDownloadDirectory(job)
      await expect(access(job.modelDir)).rejects.toMatchObject({ code: 'ENOENT' })
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  it('routes both active and paused cancellation through the ownership guard', () => {
    const source = readFileSync(join(process.cwd(), 'src/main/ipc/models.ts'), 'utf8')
    expect(source).toContain('await prepareDownloadDirectory(job)')
    expect(source).toContain('await removeOwnedDownloadDirectory(job)')
    expect(source).toContain('await removeOwnedDownloadDirectory(removed)')
    expect(source).not.toMatch(/await rm\((job|removed)\.modelDir/)
  })
})
