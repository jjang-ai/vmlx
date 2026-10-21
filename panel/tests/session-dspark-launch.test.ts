import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
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
import { addDraftAdmissionWeights } from './helpers/draftAdmissionFixture'
const dirs: string[] = []
function bundle(stampDefault: boolean | null, drafter = true): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-dspark-launch-'))
  dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'minicpmv4_7', vision_config: {} }))
  if (stampDefault !== null) {
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify({ speculative: { method: 'dspark', drafter_path: 'dspark', default: stampDefault } }))
  }
  if (drafter) {
    mkdirSync(join(dir, 'dspark'))
    writeFileSync(join(dir, 'dspark', 'config.json'), JSON.stringify({ architectures: ['Qwen3DSparkModel'], dflash_config: { projector_type: 'dspark' } }))
    addDraftAdmissionWeights(dir, 'dspark')
  }
  return dir
}
function args(manager: SessionManager, row: any): string[] {
  return (manager as any).buildArgs({ ...JSON.parse(row.config), modelPath: row.modelPath })
}
const flag = (command: string[], name: string) => (command.includes(name) ? command[command.indexOf(name) + 1] : undefined)
beforeEach(() => { rows.length = 0 })
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

describe('bundled DSpark launch wiring', () => {
  it('a bundle stamped on runs DSpark on the single-request lane, flag auto', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true), { port: 8124 })
    const command = args(manager, row)
    expect(flag(command, '--dspark')).toBe('auto')
    expect(command).toContain('--no-continuous-batching')
  })

  it('a shipped drafter is on by default even when the stamp says false (the files decide)', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(false), { port: 8125 })
    const command = args(manager, row)
    expect(flag(command, '--dspark')).toBe('auto')
    expect(command).toContain('--no-continuous-batching')
  })

  it('a bundle shipped without the drafter keeps the batching lane and gets no DSpark flag', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true, false), { port: 8130 })
    const command = args(manager, row)
    expect(command).not.toContain('--dspark')
    expect(command).toContain('--continuous-batching')
  })

  it('the server-settings choice overrides the stamp both ways', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(false), { port: 8126, useBundledDspark: true })
    let command = args(manager, row)
    expect(flag(command, '--dspark')).toBe('on')
    expect(command).toContain('--no-continuous-batching')
    await manager.updateSessionConfig(row.id, { useBundledDspark: false })
    command = args(manager, row)
    expect(flag(command, '--dspark')).toBe('off')
    expect(command).toContain('--continuous-batching')
  })

  it('an explicit draft model wins and bundles without a drafter get no flag', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true), { port: 8127, speculativeModel: '/models/other-draft' })
    expect(flag(args(manager, row), '--dspark')).toBe('off')
    const plain = await manager.createSession(bundle(null, false), { port: 8128 })
    expect(args(manager, plain)).not.toContain('--dspark')
  })

  it('Multimodal Force Off keeps DSpark: the bundle has only a multimodal runtime, --text-only restricts media', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true), { port: 8131, isMultimodal: false })
    const command = args(manager, row)
    expect(command).toContain('--text-only')
    expect(flag(command, '--dspark')).toBe('auto')
    expect(command).toContain('--no-continuous-batching')
  })

  it('a draft model shows DSpark off: the launch passes --dspark off', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true), { port: 8132, speculativeModel: '/models/other-draft' })
    expect(flag(args(manager, row), '--dspark')).toBe('off')
  })

  it("passes the session's SSD tier settings to the DSpark lane (its multi-turn session snapshots)", async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(true), { port: 8129,
      enableBlockDiskCache: true, blockDiskCacheDir: '/private/test-dspark-ssd', blockDiskCacheMaxPercent: 7 })
    const command = args(manager, row)
    expect(command).toContain('--no-continuous-batching')
    expect(command).toContain('--enable-block-disk-cache')
    expect(flag(command, '--block-disk-cache-dir')).toBe('/private/test-dspark-ssd')
    expect(flag(command, '--block-disk-cache-max-percent')).toBe('7')
    await manager.updateSessionConfig(row.id, { enableBlockDiskCache: false })
    expect(args(manager, row)).toContain('--disable-block-disk-cache')
  })
})
