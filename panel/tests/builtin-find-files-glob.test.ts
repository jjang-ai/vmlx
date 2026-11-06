import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
vi.mock('electron', () => ({ clipboard: {} }))
vi.mock('../src/main/database', () => ({ db: { getSetting: vi.fn(() => undefined) } }))
import { executeBuiltinTool } from '../src/main/tools/executor'
const roots: string[] = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
function tree() {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-find-'))
  roots.push(dir)
  mkdirSync(join(dir, 'src', 'lib'), { recursive: true })
  writeFileSync(join(dir, 'LICENSE'), 'x')
  writeFileSync(join(dir, 'src', 'lib', 'a.ts'), 'x')
  writeFileSync(join(dir, 'src', 'b.ts'), 'x')
  return dir
}
// Models commonly send path-style globs; both backends match a bare -name / --glob against the basename only, so these
// returned "Found 0 file(s)" and the model truthfully told the user that an existing file does not exist.
describe('find_files globs', () => {
  it('a leading **/ finds files at the root and below', async () => {
    const dir = tree()
    const license = await executeBuiltinTool('find_files', { pattern: '**/LICENSE*', path: '.' }, dir)
    expect(license.is_error).toBe(false)
    expect(license.content).toContain('Found 1 file(s)')
    expect(license.content).toContain('LICENSE')
    const ts = await executeBuiltinTool('find_files', { pattern: '**/*.ts' }, dir)
    expect(ts.content).toContain('Found 2 file(s)')
  })
  it('a pattern naming directories matches the path, and a bare name still works', async () => {
    const dir = tree()
    const nested = await executeBuiltinTool('find_files', { pattern: 'src/lib/*.ts' }, dir)
    expect(nested.content).toContain('Found 1 file(s)')
    expect(nested.content).toContain('a.ts')
    const bare = await executeBuiltinTool('find_files', { pattern: '*.ts' }, dir)
    expect(bare.content).toContain('Found 2 file(s)')
  })
})
