import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { publishImageOutputs } from '../src/main/imageOutputPublication'

const roots: string[] = []
const root = () => { const p = mkdtempSync(join(tmpdir(), 'vmlx-image-publication-')); roots.push(p); return p }
afterEach(() => { for (const p of roots.splice(0)) rmSync(p, { recursive: true, force: true }) })
describe('image output publication before history', () => {
  it('publishes every complete file before calling history', () => {
    const p = root()
    const files = ['source.png', 'one.png', 'two.png'].map(name => ({ path: join(p, name), data: Buffer.from(name) }))
    let called = false
    publishImageOutputs(files, () => {
      called = true
      for (const file of files) expect(readFileSync(file.path)).toEqual(file.data)
      expect(readdirSync(p).sort()).toEqual(['one.png', 'source.png', 'two.png'])
    })
    expect(called).toBe(true)
  })
  it('removes its whole new set after history rollback, preserving existing files', () => {
    const p = root()
    writeFileSync(join(p, 'keep.png'), 'existing')
    expect(() => publishImageOutputs([{path:join(p,'new.png'),data:Buffer.from('new')}], () => { throw Error('rollback') })).toThrow('rollback')
    expect(readdirSync(p)).toEqual(['keep.png'])
    expect(readFileSync(join(p, 'keep.png'), 'utf8')).toBe('existing')
  })
  it('never overwrites a colliding existing output or calls history', () => {
    const p = root()
    writeFileSync(join(p, 'existing.png'), 'original')
    let called = false
    expect(() => publishImageOutputs([
      {path:join(p,'first.png'),data:Buffer.from('first')},
      {path:join(p,'existing.png'),data:Buffer.from('replacement')},
    ], () => { called = true })).toThrow()
    expect(called).toBe(false)
    expect(readdirSync(p)).toEqual(['existing.png'])
    expect(readFileSync(join(p,'existing.png'),'utf8')).toBe('original')
  })
  it('rolls back earlier files if a later destination cannot be written', () => {
    const p = root()
    let called = false
    expect(() => publishImageOutputs([
      {path:join(p,'first.png'),data:Buffer.from('first')},
      {path:join(p,'missing','next.png'),data:Buffer.from('next')},
    ], () => { called = true })).toThrow()
    expect(called).toBe(false)
    expect(readdirSync(p)).toEqual([])
  })
})
