import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { rows, db } = vi.hoisted(() => {
  const rows: any[] = []
  const db = {
    getSessions: () => rows,
    getSession: (id: string) => rows.find(r => r.id === id),
    getSessionByModelPath: (path: string) => rows.find(r => r.modelPath === path),
    getSetting: () => undefined,
    createSession: (row: any) => rows.push(row),
    updateSession: (id: string, patch: any) => Object.assign(rows.find(r => r.id === id), patch),
  }
  return { rows, db }
})
vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({ app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined } }))
import { SessionManager } from '../src/main/sessions'
const dirs: string[] = []
function bundle(modelType: string, maxNew = 4096): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-cap-intent-'))
  dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: modelType }))
  writeFileSync(join(dir, 'generation_config.json'), JSON.stringify({ max_new_tokens: maxNew }))
  return dir
}
function args(manager: SessionManager, row: any): string[] {
  return (manager as any).buildArgs({ ...JSON.parse(row.config), modelPath: row.modelPath })
}
beforeEach(() => { rows.length = 0 })
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

describe('explicit server output cap survives creation and legacy baseline migration', () => {
  it.each(['nanbeige', 'qwen3_5', 'qwen4_exp', 'minimax', 'minimax_m3', 'openpangu_v2', 'gemma4', 'lfm2', 'muse_glimmer', 'laguna'])(
    'preserves a fresh explicit4096 on %s even when it equals bundle defaults', async family => {
      const manager = new SessionManager()
      const row = await manager.createSession(bundle(family), { port: 8123, maxTokens: 4096 })
      expect(JSON.parse(row.config).maxTokens).toBe(4096)
      const command = args(manager, row)
      expect(command[command.indexOf('--max-tokens') + 1]).toBe('4096')
    },
  )
  it.each([12000, 12068, 32768, 777])('preserves explicit%d rather than classifying intent by numeric value', async cap => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle('nanbeige', cap), { port: 8123, maxTokens: cap })
    expect(JSON.parse(row.config).maxTokens).toBe(cap)
  })
  it('leaves a new default session without a CLI override', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle('nanbeige'), { port: 8123, maxTokens: 0 })
    expect(args(manager, row)).not.toContain('--max-tokens')
  })
  it('migrates an old stopped baseline before merging a fresh explicit cap', async () => {
    const manager = new SessionManager()
    const path = bundle('nanbeige')
    rows.push({ id: 'old', type: 'local', status: 'stopped', modelPath: path, port: 8123, host: '127.0.0.1',
      config: JSON.stringify({ maxTokens: 12000, cacheStackStartupDefaultsVersion: 17 }) })
    const row = await manager.createSession(path, { port: 8123, maxTokens: 4096 })
    expect(JSON.parse(row.config).maxTokens).toBe(4096)
  })
  it('migrates an old saved baseline before a settings edit, then preserves the edit', async () => {
    const manager = new SessionManager()
    const path = bundle('nanbeige')
    rows.push({ id: 'old', type: 'local', status: 'stopped', modelPath: path, port: 8123, host: '127.0.0.1',
      config: JSON.stringify({ maxTokens: 12000, cacheStackStartupDefaultsVersion: 17 }) })
    await manager.updateSessionConfig('old', { maxTokens: 4096 })
    expect(JSON.parse(rows[0].config)).toMatchObject({ maxTokens: 4096, generationStartupDefaultsVersion: 4 })
  })
})
