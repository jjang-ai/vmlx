import { mkdtempSync, mkdirSync, symlinkSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { sameLocalBundlePath } from '../src/main/local-bundle-identity'

const roots: string[] = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
it('does not reuse a same-name bundle on another path', () => {
  const root = mkdtempSync(join(tmpdir(), 'vmlx-path-identity-')); roots.push(root)
  const original = join(root, 'original', 'Model')
  const copy = join(root, 'copy', 'Model')
  mkdirSync(original, { recursive: true }); mkdirSync(copy, { recursive: true })
  expect(sameLocalBundlePath(original, copy)).toBe(false)
  expect(sameLocalBundlePath(original, original)).toBe(true)
  const alias = join(root, 'alias')
  symlinkSync(original, alias)
  expect(sameLocalBundlePath(original, alias)).toBe(true)
})
it('does not guess unresolved or differently cased paths', () => {
  expect(sameLocalBundlePath('/missing/a/Model', '/missing/b/Model')).toBe(false)
  expect(sameLocalBundlePath('org/Model', '/missing/Model')).toBe(false)
  expect(sameLocalBundlePath('', '')).toBe(false)
})
