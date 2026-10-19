/**
 * Built-in tool working-directory validation.
 *
 * A missing working directory used to fail deep inside resolvePath with a raw
 * `ENOENT: ... lstat '<path>'`. The chat renders only "failed" per tool row, so
 * the model could not distinguish that from a genuine tool error: it spent its
 * whole tool-iteration budget on mkdir/pwd/ls recovery and ended the turn with
 * no visible answer. Observed live on DSV4 against a deleted release-gate run
 * directory.
 */
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect, vi } from 'vitest'

vi.mock('electron', () => ({ clipboard: { readText: () => '', writeText: () => {} } }))
vi.mock('../src/main/database', () => ({ db: {} }))

describe('built-in tool working directory validation', () => {
  it('names a missing working directory instead of leaking ENOENT', async () => {
    const { executeBuiltinTool } = await import('../src/main/tools/executor')
    const missing = join(tmpdir(), `vmlx-missing-working-directory-${process.pid}`, 'r20')

    const result = await executeBuiltinTool(
      'write_file',
      { path: 'a.txt', content: 'x' },
      missing,
    )

    expect(result.is_error).toBe(true)
    expect(result.content).toContain('does not exist')
    expect(result.content).toContain(missing)
    expect(result.content).toContain('Working Directory')
    expect(result.content).not.toContain('ENOENT')
  })

  it('names an unset working directory', async () => {
    const { executeBuiltinTool } = await import('../src/main/tools/executor')

    const result = await executeBuiltinTool('list_directory', { path: '.' }, '')

    expect(result.is_error).toBe(true)
    expect(result.content).toContain('No working directory is configured')
  })

  it('still runs normally when the working directory exists', async () => {
    const { executeBuiltinTool } = await import('../src/main/tools/executor')

    const result = await executeBuiltinTool('list_directory', { path: '.' }, process.cwd())

    expect(result.is_error).toBe(false)
  })

  it('distinguishes empty files from nonempty files smaller than a kilobyte', async () => {
    const { executeBuiltinTool } = await import('../src/main/tools/executor')
    const directory = mkdtempSync(join(tmpdir(), 'vmlx-tool-sizes-'))
    try {
      writeFileSync(join(directory, 'empty.txt'), '')
      writeFileSync(join(directory, 'small.txt'), 'x')
      writeFileSync(join(directory, 'kilobyte.txt'), Buffer.alloc(1024))

      const result = await executeBuiltinTool('list_directory', { path: '.' }, directory)

      expect(result.is_error).toBe(false)
      expect(result.content).toContain('empty.txt  (0 B)')
      expect(result.content).toContain('small.txt  (1 B)')
      expect(result.content).toContain('kilobyte.txt  (1.0 KB)')
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  })

})
