import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
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
import { detectModelConfigFromDir } from '../src/main/model-config-registry'
import { addDraftAdmissionWeights } from './helpers/draftAdmissionFixture'

const dirs: string[] = []
/** A Ripple-shaped bundle: model_type ricursive, a text-only target, and a DFlash2 drafter with a NoPE full layer. */
function rippleBundle(drafter = true): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-ripple-launch-'))
  dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'ricursive', architectures: ['RicursiveForCausalLM'],
    sliding_window: 1024, max_position_embeddings: 2097152 }))
  writeFileSync(join(dir, 'jang_config.json'), JSON.stringify({ capabilities: { family: 'ricursive', reasoning_parser: 'qwen3',
    tool_parser: 'hermes', think_in_template: false, supports_tools: true, supports_thinking: true, modality: 'text' } }))
  writeFileSync(join(dir, 'chat_template.jinja'), "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}")
  if (drafter) {
    mkdirSync(join(dir, 'dflash2'))
    writeFileSync(join(dir, 'dflash2', 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'],
      layer_types: ['sliding_attention', 'full_attention'], no_rope_layers: [1], sliding_window: 4096,
      dflash_config: { block_size: 16 } }))
    addDraftAdmissionWeights(dir, 'dflash2')
  }
  return dir
}
function args(manager: SessionManager, row: any): string[] {
  return (manager as any).buildArgs({ ...JSON.parse(row.config), modelPath: row.modelPath })
}
const flag = (command: string[], name: string) => (command.includes(name) ? command[command.indexOf(name) + 1] : undefined)
beforeEach(() => { rows.length = 0 })
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

describe('Ripple (ricursive) detection and launch', () => {
  it('detects the ricursive family with hermes tools and qwen3 reasoning', () => {
    const detected = detectModelConfigFromDir(rippleBundle())
    expect(detected.family).toBe('ricursive')
    expect(detected.toolParser).toBe('hermes')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBeFalsy()
    // The template never reads enable_thinking: the model decides per turn, so no thinking switch is offered.
    expect(detected.honorsEnableThinking).toBe(false)
    expect(detected.supportsInstructMode).toBe(false)
    // No effort buttons: the template renders no reasoning_effort and the bundle declares no levels.
    expect(detected.supportedReasoningEfforts ?? []).toEqual([])
  })

  it('the bundled DFlash2 drafter runs by default on the non-batched lane', async () => {
    const manager = new SessionManager()
    const row = await manager.createSession(rippleBundle(), { port: 8231 })
    const command = args(manager, row)
    expect(command).toContain('--no-continuous-batching')
    expect(command).not.toContain('--no-bundled-dflash2')
    expect(flag(command, '--tool-call-parser') ?? 'hermes').toBe('hermes')
    expect(flag(command, '--reasoning-parser') ?? 'qwen3').toBe('qwen3')
    if (process.env.RIPPLE_PRINT_ARGV) console.log('RIPPLE_ARGV ' + JSON.stringify(command))
  })

  it('the real bundle (RIPPLE_BUNDLE) launches the same way', async () => {
    const real = process.env.RIPPLE_BUNDLE
    if (!real || !existsSync(join(real, 'config.json'))) return
    const manager = new SessionManager()
    const row = await manager.createSession(real, { port: 8232 })
    const command = args(manager, row)
    expect(command).toContain('--no-continuous-batching')
    const realDetected = detectModelConfigFromDir(real)
    expect(realDetected.family).toBe('ricursive')
    expect(realDetected.honorsEnableThinking).toBe(false)
    expect(realDetected.supportedReasoningEfforts ?? []).toEqual([])
    console.log('RIPPLE_REAL_ARGV ' + JSON.stringify(command))
  })
})
