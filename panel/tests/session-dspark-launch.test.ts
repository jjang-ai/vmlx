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
import { addDraftAdmissionWeights, writeDsv41Bundle, writeMinimaxDsparkBundle, type Dsv41Layout } from './helpers/draftAdmissionFixture'
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
function m3Bundle(drafter = true): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-dspark-m31-launch-'))
  dirs.push(dir)
  writeMinimaxDsparkBundle(dir)
  if (!drafter) rmSync(join(dir, 'dspark'), { recursive: true, force: true })
  return dir
}
function dsv41Bundle(layout: Dsv41Layout): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-dspark-dsv41-launch-'))
  dirs.push(dir)
  writeDsv41Bundle(dir, layout)
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

describe('bundled DSpark launch wiring, MiniMax-M3 / M3.1 (text lane)', () => {
  // Everything but the model path, the port and --dspark must be the argv of the same bundle without a drafter.
  const shape = (command: string[], bundleDir: string) => {
    const out = [...command]
    const i = out.indexOf('--dspark')
    if (i >= 0) out.splice(i, 2)
    return out.map(a => (a === bundleDir ? '<bundle>' : a))
  }
  const configs: Array<[Record<string, unknown>, string]> = [
    [{}, 'auto'], [{ useBundledDspark: true }, 'on'], [{ useBundledDspark: false }, 'off'],
  ]

  it('default / on / off only choose --dspark; batching and the cache stack match the no-drafter launch', async () => {
    for (const [extra, mode] of configs) {
      for (const cache of [{}, { enableBlockDiskCache: true, blockDiskCacheDir: '/private/test-m31-ssd', blockDiskCacheMaxPercent: 7 },
        { enableDiskCache: true }, { continuousBatching: false }]) {
        rows.length = 0
        const manager = new SessionManager()
        const withDrafter = m3Bundle()
        const without = m3Bundle(false)
        const row = await manager.createSession(withDrafter, { port: 8140, ...cache, ...extra })
        const plain = await manager.createSession(without, { port: 8140, ...cache, ...extra })
        const command = args(manager, row)
        expect(flag(command, '--dspark')).toBe(mode)
        expect(args(manager, plain)).not.toContain('--dspark')
        expect(shape(command, withDrafter)).toEqual(shape(args(manager, plain), without))
        if ((cache as any).continuousBatching === false) expect(command).toContain('--no-continuous-batching')
        else {
          expect(command).toContain('--continuous-batching')
          expect(command).not.toContain('--no-continuous-batching')
        }
      }
    }
  })

  it('toggling the saved choice relaunches with --dspark on / off and the same batching flags', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(m3Bundle(), { port: 8141 })
    expect(flag(args(manager, row), '--dspark')).toBe('auto')
    await manager.updateSessionConfig(row.id, { useBundledDspark: false })
    const off = args(manager, row)
    await manager.updateSessionConfig(row.id, { useBundledDspark: true })
    const on = args(manager, row)
    expect(flag(off, '--dspark')).toBe('off')
    expect(flag(on, '--dspark')).toBe('on')
    expect(off).toContain('--continuous-batching')
    expect(on).toContain('--continuous-batching')
    expect(shape(off, row.modelPath)).toEqual(shape(on, row.modelPath))
  })
})

describe('bundled DSpark launch wiring, DeepSeek-V4.1 (dspark/dspark.safetensors, DSV4 single-sequence lane)', () => {
  const shape = (command: string[], bundleDir: string) => {
    const out = [...command]
    const i = out.indexOf('--dspark')
    if (i >= 0) out.splice(i, 2)
    return out.map(a => (a === bundleDir ? '<bundle>' : a))
  }

  it('JANGHT1.4 and JANG_2K layouts: auto / on / off pass --dspark; the DSV4 lane (--max-num-seqs 1) is unchanged', async () => {
    for (const layout of ['janght', 'jang2k'] as const) {
      for (const [extra, mode] of [[{}, 'auto'], [{ useBundledDspark: true }, 'on'], [{ useBundledDspark: false }, 'off']] as const) {
        rows.length = 0
        const manager = new SessionManager()
        const withDraft = dsv41Bundle(layout)
        const without = dsv41Bundle('none')
        const row = await manager.createSession(withDraft, { port: 8150, ...extra })
        const plain = await manager.createSession(without, { port: 8150, ...extra })
        const command = args(manager, row)
        expect(flag(command, '--dspark')).toBe(mode)
        // The engine's DSV41DSparkGenerator engages only at max_num_seqs == 1; the DSV4 lane always passes it.
        expect(flag(command, '--max-num-seqs')).toBe('1')
        expect(command).toContain('--continuous-batching')
        expect(command).not.toContain('--no-continuous-batching')
        expect(args(manager, plain)).not.toContain('--dspark')
        expect(shape(command, withDraft)).toEqual(shape(args(manager, plain), without))
      }
    }
  })

  it('toggling the saved choice relaunches with --dspark off / on and the same flags otherwise', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(dsv41Bundle('janght'), { port: 8151 })
    expect(flag(args(manager, row), '--dspark')).toBe('auto')
    await manager.updateSessionConfig(row.id, { useBundledDspark: false })
    const off = args(manager, row)
    await manager.updateSessionConfig(row.id, { useBundledDspark: true })
    const on = args(manager, row)
    expect(flag(off, '--dspark')).toBe('off')
    expect(flag(on, '--dspark')).toBe('on')
    expect(shape(off, row.modelPath)).toEqual(shape(on, row.modelPath))
  })

  it('a stale draft model does not switch V4.1 DSpark off (the DSV4 engine ignores --speculative-model)', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(dsv41Bundle('janght'), { port: 8152, speculativeModel: '/models/other-draft' })
    const command = args(manager, row)
    expect(flag(command, '--dspark')).toBe('auto')
    expect(command).not.toContain('--speculative-model')
  })
})
