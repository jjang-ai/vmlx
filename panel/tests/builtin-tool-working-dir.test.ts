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
})
