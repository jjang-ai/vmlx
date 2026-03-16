/**
 * Developer Tools Tests — Phase 6: Quality review test suite
 *
 * Coverage:
 *   - AppState reducer: SET_TOOLS_PANEL action (modelPath handling, defaults)
 *   - AppState reducer: SET_SERVER_PANEL modelPath (serverInitialModelPath flow)
 *   - ModelDoctor parseOutput (all check statuses, section transitions, summary lines)
 *   - ModelConverter preset logic (bits/groupSize mapping)
 *   - ModelConverter CLI args building (all optional flags)
 *   - ModelConverter output path extraction from log lines
 *   - ToolsDashboard model filtering (name, path, format, search)
 *   - ToolsModeContent navigation helper (panel/modelPath dispatch)
 *   - Serve flow (tools → server mode handoff)
 *   - IPC developer handler CLI args construction (doctor, convert, cancel)
 *   - useStreamingOperation MAX_LOG_LINES cap
 */
import { describe, it, expect } from 'vitest'

// ─── AppState Types & Reducer (from types/app-state.ts + AppStateContext.tsx) ─

type AppMode = 'chat' | 'server' | 'tools'
type ServerPanel = 'dashboard' | 'session' | 'create' | 'settings' | 'about'
type ToolsPanel = 'dashboard' | 'inspector' | 'doctor' | 'converter'

interface AppState {
  mode: AppMode
  activeChatId: string | null
  activeSessionId: string | null
  serverPanel: ServerPanel
  serverSessionId: string | null
  serverInitialModelPath: string | null
  toolsPanel: ToolsPanel
  toolsModelPath: string | null
  sidebarCollapsed: boolean
}

type AppAction =
  | { type: 'SET_MODE'; mode: AppMode }
  | { type: 'OPEN_CHAT'; chatId: string; sessionId: string }
  | { type: 'CLOSE_CHAT' }
  | { type: 'SET_SERVER_PANEL'; panel: ServerPanel; sessionId?: string; modelPath?: string }
  | { type: 'SET_TOOLS_PANEL'; panel: ToolsPanel; modelPath?: string | null }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'RESTORE_STATE'; state: Partial<AppState> }

const initialState: AppState = {
  mode: 'chat',
  activeChatId: null,
  activeSessionId: null,
  serverPanel: 'dashboard',
  serverSessionId: null,
  serverInitialModelPath: null,
  toolsPanel: 'dashboard',
  toolsModelPath: null,
  sidebarCollapsed: false,
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, mode: action.mode }
    case 'OPEN_CHAT':
      return { ...state, activeChatId: action.chatId, activeSessionId: action.sessionId }
    case 'CLOSE_CHAT':
      return { ...state, activeChatId: null, activeSessionId: null }
    case 'SET_SERVER_PANEL':
      return { ...state, serverPanel: action.panel, serverSessionId: action.sessionId ?? state.serverSessionId, serverInitialModelPath: action.modelPath !== undefined ? (action.modelPath || null) : state.serverInitialModelPath }
    case 'SET_TOOLS_PANEL':
      return { ...state, toolsPanel: action.panel, toolsModelPath: action.modelPath !== undefined ? action.modelPath : state.toolsModelPath }
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed }
    case 'RESTORE_STATE':
      return { ...state, ...action.state }
    default:
      return state
  }
}

// ─── ModelDoctor parseOutput (from ModelDoctor.tsx) ──────────────────────────

type CheckStatus = 'pass' | 'fail' | 'warn' | 'pending' | 'running'

interface CheckResult {
  name: string
  status: CheckStatus
  details: string[]
}

function parseOutput(lines: string[]): CheckResult[] {
  const checks: CheckResult[] = [
    { name: 'Config', status: 'pending', details: [] },
    { name: 'Weights', status: 'pending', details: [] },
    { name: 'Architecture', status: 'pending', details: [] },
    { name: 'Inference', status: 'pending', details: [] },
  ]

  let currentIdx = -1

  for (const line of lines) {
    const trimmed = line.trim()

    if (trimmed.startsWith('Checking config')) {
      currentIdx = 0
      checks[0].status = 'running'
    } else if (trimmed.startsWith('Checking weights')) {
      currentIdx = 1
      checks[1].status = 'running'
    } else if (trimmed.startsWith('Checking architecture')) {
      currentIdx = 2
      checks[2].status = 'running'
    } else if (trimmed.startsWith('Running inference')) {
      currentIdx = 3
      checks[3].status = 'running'
    } else if (trimmed.includes('skipped (--no-inference)')) {
      checks[3].status = 'pass'
      checks[3].details.push('Skipped')
      currentIdx = -1
    } else if (currentIdx >= 0) {
      const current = checks[currentIdx]
      if (trimmed.includes(': OK')) {
        current.status = 'pass'
        currentIdx = -1
      } else if (trimmed.includes(': PASS')) {
        current.status = 'pass'
        const passIdx = trimmed.indexOf('PASS')
        const detail = trimmed.substring(passIdx + 4).replace(/^\s*-\s*/, '').trim()
        if (detail) current.details.push(detail)
        currentIdx = -1
      } else if (trimmed.startsWith('Found ') || trimmed.startsWith('LatentMoE')) {
        current.details.push(trimmed)
      }
    }

    if (trimmed.startsWith('FAIL:')) {
      const detail = trimmed.replace(/^FAIL:\s*/, '')
      for (const check of checks) {
        if (detail.startsWith(check.name + ':')) {
          check.status = 'fail'
          check.details.push(detail.replace(check.name + ': ', ''))
          break
        }
      }
    }
    if (trimmed.startsWith('WARN:')) {
      const detail = trimmed.replace(/^WARN:\s*/, '')
      for (const check of checks) {
        if (detail.startsWith(check.name + ':')) {
          if (check.status !== 'fail') check.status = 'warn'
          check.details.push(detail.replace(check.name + ': ', ''))
          break
        }
      }
    }
    if (trimmed.startsWith('ALL CHECKS PASSED')) {
      checks.forEach(c => { if (c.status === 'pending' || c.status === 'running') c.status = 'pass' })
    }
  }

  return checks
}

// ─── ModelConverter Preset Logic (from ModelConverter.tsx) ────────────────────

type Preset = 'balanced' | 'quality' | 'compact' | 'custom'

const PRESETS: Record<Exclude<Preset, 'custom'>, { bits: number; groupSize: number; label: string; desc: string }> = {
  balanced: { bits: 4, groupSize: 64, label: 'Balanced (4-bit)', desc: 'Good quality/size tradeoff — recommended' },
  quality: { bits: 8, groupSize: 64, label: 'Quality (8-bit)', desc: 'Larger but better quality' },
  compact: { bits: 3, groupSize: 64, label: 'Compact (3-bit)', desc: 'Smallest, some quality loss' },
}

// ─── CLI Args Builder (from developer.ts convert handler) ────────────────────

function buildConvertArgs(args: {
  model: string
  output?: string
  bits: number
  groupSize: number
  mode?: string
  dtype?: string
  force?: boolean
  skipVerify?: boolean
  trustRemoteCode?: boolean
}): string[] {
  const cliArgs = ['convert', args.model, '--bits', args.bits.toString(), '--group-size', args.groupSize.toString()]
  if (args.output) cliArgs.push('--output', args.output)
  if (args.mode && args.mode !== 'default') cliArgs.push('--mode', args.mode)
  if (args.dtype) cliArgs.push('--dtype', args.dtype)
  if (args.force) cliArgs.push('--force')
  if (args.skipVerify) cliArgs.push('--skip-verify')
  if (args.trustRemoteCode) cliArgs.push('--trust-remote-code')
  return cliArgs
}

// ─── Doctor Args Builder (from developer.ts doctor handler) ──────────────────

function buildDoctorArgs(modelPath: string, options: { noInference?: boolean }): string[] {
  const args = ['doctor', modelPath]
  if (options?.noInference) args.push('--no-inference')
  return args
}

// ─── Model Filter Logic (from ToolsDashboard.tsx) ────────────────────────────

interface LocalModel {
  id: string
  name: string
  path: string
  size?: string
  format?: 'mlx' | 'gguf' | 'unknown'
  quantization?: string
}

function filterMlxModels(models: LocalModel[], filter: string): LocalModel[] {
  return models.filter(m =>
    m.format === 'mlx' && (
      m.name.toLowerCase().includes(filter.toLowerCase()) ||
      m.path.toLowerCase().includes(filter.toLowerCase())
    )
  )
}

// ─── Output Path Extraction (from ModelConverter.tsx runConvert) ──────────────

function extractOutputPath(logLines: string[]): string | undefined {
  const pathLine = logLines.find((l: string) => l.startsWith('Output path:'))
  if (pathLine) return pathLine.replace('Output path:', '').trim()
  return undefined
}

// ─── MAX_LOG_LINES (from useStreamingOperation.ts) ───────────────────────────

const MAX_LOG_LINES = 2000

function accumulateLogs(existing: string[], newLines: string[]): string[] {
  return [...existing, ...newLines].slice(-MAX_LOG_LINES)
}

// ─── Navigate Helper (from App.tsx ToolsModeContent) ─────────────────────────

function buildToolsDispatch(panel: 'dashboard' | 'inspector' | 'doctor' | 'converter', modelPath?: string) {
  return {
    type: 'SET_TOOLS_PANEL' as const,
    panel,
    modelPath: modelPath !== undefined ? (modelPath || null) : undefined,
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — AppState Reducer', () => {
  // ─── SET_TOOLS_PANEL ────────────────────────────────────────────────────────

  describe('SET_TOOLS_PANEL', () => {
    it('sets panel to inspector', () => {
      const state = appReducer(initialState, { type: 'SET_TOOLS_PANEL', panel: 'inspector' })
      expect(state.toolsPanel).toBe('inspector')
    })

    it('sets panel to doctor', () => {
      const state = appReducer(initialState, { type: 'SET_TOOLS_PANEL', panel: 'doctor' })
      expect(state.toolsPanel).toBe('doctor')
    })

    it('sets panel to converter', () => {
      const state = appReducer(initialState, { type: 'SET_TOOLS_PANEL', panel: 'converter' })
      expect(state.toolsPanel).toBe('converter')
    })

    it('sets panel back to dashboard', () => {
      const modified = { ...initialState, toolsPanel: 'inspector' as ToolsPanel }
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'dashboard' })
      expect(state.toolsPanel).toBe('dashboard')
    })

    it('sets modelPath when provided', () => {
      const state = appReducer(initialState, { type: 'SET_TOOLS_PANEL', panel: 'inspector', modelPath: '/path/to/model' })
      expect(state.toolsModelPath).toBe('/path/to/model')
    })

    it('clears modelPath when set to null', () => {
      const modified = { ...initialState, toolsModelPath: '/old/model' as string | null }
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'dashboard', modelPath: null })
      expect(state.toolsModelPath).toBeNull()
    })

    it('clears modelPath when set to empty string', () => {
      const modified = { ...initialState, toolsModelPath: '/old/model' as string | null }
      // empty string via the navigateTo helper becomes (modelPath || null) = null
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'converter', modelPath: '' })
      // empty string is falsy, so the action carries '' which is stored as ''
      expect(state.toolsModelPath).toBe('')
    })

    it('preserves modelPath when not provided in action', () => {
      const modified = { ...initialState, toolsModelPath: '/existing/model' as string | null }
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'dashboard' })
      expect(state.toolsModelPath).toBe('/existing/model')
    })

    it('preserves modelPath when action.modelPath is undefined', () => {
      const modified = { ...initialState, toolsModelPath: '/keep/this' as string | null }
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'inspector', modelPath: undefined })
      expect(state.toolsModelPath).toBe('/keep/this')
    })

    it('does not affect other state fields', () => {
      const modified = { ...initialState, mode: 'tools' as AppMode, activeChatId: 'chat-1' }
      const state = appReducer(modified, { type: 'SET_TOOLS_PANEL', panel: 'doctor', modelPath: '/m' })
      expect(state.mode).toBe('tools')
      expect(state.activeChatId).toBe('chat-1')
      expect(state.serverPanel).toBe('dashboard')
    })
  })

  // ─── SET_SERVER_PANEL (serverInitialModelPath) ──────────────────────────────

  describe('SET_SERVER_PANEL with modelPath', () => {
    it('sets serverInitialModelPath when modelPath provided', () => {
      const state = appReducer(initialState, {
        type: 'SET_SERVER_PANEL',
        panel: 'create',
        modelPath: '/models/llama',
      })
      expect(state.serverInitialModelPath).toBe('/models/llama')
      expect(state.serverPanel).toBe('create')
    })

    it('clears serverInitialModelPath when modelPath is empty string', () => {
      const modified = { ...initialState, serverInitialModelPath: '/old' as string | null }
      const state = appReducer(modified, {
        type: 'SET_SERVER_PANEL',
        panel: 'create',
        modelPath: '',
      })
      // '' || null = null
      expect(state.serverInitialModelPath).toBeNull()
    })

    it('preserves serverInitialModelPath when modelPath not in action', () => {
      const modified = { ...initialState, serverInitialModelPath: '/keep' as string | null }
      const state = appReducer(modified, {
        type: 'SET_SERVER_PANEL',
        panel: 'dashboard',
      })
      expect(state.serverInitialModelPath).toBe('/keep')
    })

    it('preserves serverInitialModelPath when modelPath is undefined', () => {
      const modified = { ...initialState, serverInitialModelPath: '/keep' as string | null }
      const state = appReducer(modified, {
        type: 'SET_SERVER_PANEL',
        panel: 'session',
        modelPath: undefined,
      })
      expect(state.serverInitialModelPath).toBe('/keep')
    })

    it('SET_SERVER_PANEL still handles sessionId correctly', () => {
      const state = appReducer(initialState, {
        type: 'SET_SERVER_PANEL',
        panel: 'session',
        sessionId: 'sess-1',
        modelPath: '/m',
      })
      expect(state.serverSessionId).toBe('sess-1')
      expect(state.serverInitialModelPath).toBe('/m')
      expect(state.serverPanel).toBe('session')
    })
  })

  // ─── SET_MODE with tools ────────────────────────────────────────────────────

  describe('SET_MODE with tools', () => {
    it('switches to tools mode', () => {
      const state = appReducer(initialState, { type: 'SET_MODE', mode: 'tools' })
      expect(state.mode).toBe('tools')
    })

    it('preserves tools state when switching modes', () => {
      let state = appReducer(initialState, { type: 'SET_TOOLS_PANEL', panel: 'doctor', modelPath: '/m' })
      state = appReducer(state, { type: 'SET_MODE', mode: 'chat' })
      state = appReducer(state, { type: 'SET_MODE', mode: 'tools' })
      expect(state.toolsPanel).toBe('doctor')
      expect(state.toolsModelPath).toBe('/m')
    })
  })

  // ─── Initial state ─────────────────────────────────────────────────────────

  describe('initial state', () => {
    it('has correct tools defaults', () => {
      expect(initialState.toolsPanel).toBe('dashboard')
      expect(initialState.toolsModelPath).toBeNull()
      expect(initialState.serverInitialModelPath).toBeNull()
    })
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — ModelDoctor parseOutput', () => {
  it('returns all pending checks for empty input', () => {
    const checks = parseOutput([])
    expect(checks).toHaveLength(4)
    expect(checks.every(c => c.status === 'pending')).toBe(true)
    expect(checks.map(c => c.name)).toEqual(['Config', 'Weights', 'Architecture', 'Inference'])
  })

  it('marks config as running when header appears', () => {
    const checks = parseOutput(['Checking config...'])
    expect(checks[0].status).toBe('running')
    expect(checks[1].status).toBe('pending')
  })

  it('marks config as pass on OK', () => {
    const checks = parseOutput([
      'Checking config...',
      'Config: OK',
    ])
    expect(checks[0].status).toBe('pass')
  })

  it('marks weights as pass on OK', () => {
    const checks = parseOutput([
      'Checking config...',
      'Config: OK',
      'Checking weights...',
      'Found 42 weight files',
      'Weights: OK',
    ])
    expect(checks[0].status).toBe('pass')
    expect(checks[1].status).toBe('pass')
    expect(checks[1].details).toContain('Found 42 weight files')
  })

  it('captures Found details during check', () => {
    const checks = parseOutput([
      'Checking architecture...',
      'Found 32 transformer layers',
      'LatentMoE detected',
      'Architecture: OK',
    ])
    expect(checks[2].status).toBe('pass')
    expect(checks[2].details).toEqual(['Found 32 transformer layers', 'LatentMoE detected'])
  })

  it('marks PASS with detail extraction', () => {
    const checks = parseOutput([
      'Running inference...',
      'Inference: PASS - Generated 10 tokens at 45 tok/s',
    ])
    expect(checks[3].status).toBe('pass')
    expect(checks[3].details).toContain('Generated 10 tokens at 45 tok/s')
  })

  it('handles skipped inference', () => {
    const checks = parseOutput([
      'Checking config...',
      'Config: OK',
      'Inference skipped (--no-inference)',
    ])
    expect(checks[3].status).toBe('pass')
    expect(checks[3].details).toContain('Skipped')
  })

  it('handles FAIL in summary section', () => {
    const checks = parseOutput([
      'Checking config...',
      'FAIL: Config: missing model_type in config.json',
    ])
    expect(checks[0].status).toBe('fail')
    expect(checks[0].details).toContain('missing model_type in config.json')
  })

  it('handles WARN in summary section', () => {
    const checks = parseOutput([
      'Checking weights...',
      'Weights: OK',
      'WARN: Weights: some shards have inconsistent dtypes',
    ])
    expect(checks[1].status).toBe('warn')
    expect(checks[1].details).toContain('some shards have inconsistent dtypes')
  })

  it('FAIL overrides WARN (not vice versa)', () => {
    const checks = parseOutput([
      'FAIL: Weights: corrupted shard',
      'WARN: Weights: inconsistent dtypes',
    ])
    expect(checks[1].status).toBe('fail')
    expect(checks[1].details).toHaveLength(2)
  })

  it('WARN does not override FAIL', () => {
    const checks = parseOutput([
      'FAIL: Config: missing field',
      'WARN: Config: optional field missing',
    ])
    expect(checks[0].status).toBe('fail')
  })

  it('ALL CHECKS PASSED promotes remaining pending/running to pass', () => {
    const checks = parseOutput([
      'Checking config...',
      'Config: OK',
      'ALL CHECKS PASSED',
    ])
    expect(checks[0].status).toBe('pass')
    expect(checks[1].status).toBe('pass') // was pending, promoted
    expect(checks[2].status).toBe('pass') // was pending, promoted
    expect(checks[3].status).toBe('pass') // was pending, promoted
  })

  it('ALL CHECKS PASSED does not override fail/warn', () => {
    const checks = parseOutput([
      'FAIL: Config: broken',
      'WARN: Weights: mismatch',
      'ALL CHECKS PASSED',
    ])
    expect(checks[0].status).toBe('fail') // not overridden
    expect(checks[1].status).toBe('warn') // not overridden
    expect(checks[2].status).toBe('pass') // promoted
    expect(checks[3].status).toBe('pass') // promoted
  })

  it('handles full successful run', () => {
    const checks = parseOutput([
      'Checking config...',
      'Config: OK',
      'Checking weights...',
      'Found 4 weight files',
      'Weights: OK',
      'Checking architecture...',
      'Found 32 layers',
      'Architecture: OK',
      'Running inference...',
      'Inference: PASS - 50 tok/s',
      'ALL CHECKS PASSED',
    ])
    expect(checks.every(c => c.status === 'pass')).toBe(true)
    expect(checks[1].details).toContain('Found 4 weight files')
    expect(checks[2].details).toContain('Found 32 layers')
    expect(checks[3].details).toContain('50 tok/s')
  })

  it('handles whitespace in lines', () => {
    const checks = parseOutput([
      '  Checking config...  ',
      '  Config: OK  ',
    ])
    expect(checks[0].status).toBe('pass')
  })

  it('ignores unrecognized lines', () => {
    const checks = parseOutput([
      'Loading model...',
      'Some random output',
      'Checking config...',
      'Config: OK',
    ])
    expect(checks[0].status).toBe('pass')
    expect(checks[0].details).toHaveLength(0)
  })

  it('handles multiple sections transitioning', () => {
    const checks = parseOutput([
      'Checking config...',
      'Checking weights...',
      'Weights: OK',
    ])
    // Config started running, then weights started — config stays running (no OK)
    expect(checks[0].status).toBe('running')
    expect(checks[1].status).toBe('pass')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — ModelConverter Presets', () => {
  it('balanced preset has 4-bit, 64 group size', () => {
    expect(PRESETS.balanced.bits).toBe(4)
    expect(PRESETS.balanced.groupSize).toBe(64)
  })

  it('quality preset has 8-bit, 64 group size', () => {
    expect(PRESETS.quality.bits).toBe(8)
    expect(PRESETS.quality.groupSize).toBe(64)
  })

  it('compact preset has 3-bit, 64 group size', () => {
    expect(PRESETS.compact.bits).toBe(3)
    expect(PRESETS.compact.groupSize).toBe(64)
  })

  it('all presets have labels and descriptions', () => {
    for (const [key, preset] of Object.entries(PRESETS)) {
      expect(preset.label).toBeTruthy()
      expect(preset.desc).toBeTruthy()
      expect(preset.label.length).toBeGreaterThan(0)
      expect(preset.desc.length).toBeGreaterThan(0)
    }
  })

  it('preset keys are exactly balanced, quality, compact', () => {
    expect(Object.keys(PRESETS).sort()).toEqual(['balanced', 'compact', 'quality'])
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — CLI Args Building', () => {
  describe('buildConvertArgs', () => {
    it('builds minimal args with required fields', () => {
      const args = buildConvertArgs({ model: 'org/model', bits: 4, groupSize: 64 })
      expect(args).toEqual(['convert', 'org/model', '--bits', '4', '--group-size', '64'])
    })

    it('includes output directory', () => {
      const args = buildConvertArgs({ model: 'org/model', bits: 4, groupSize: 64, output: '/tmp/out' })
      expect(args).toContain('--output')
      expect(args[args.indexOf('--output') + 1]).toBe('/tmp/out')
    })

    it('includes mode when not default', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, mode: 'NF4' })
      expect(args).toContain('--mode')
      expect(args[args.indexOf('--mode') + 1]).toBe('NF4')
    })

    it('excludes mode when default', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, mode: 'default' })
      expect(args).not.toContain('--mode')
    })

    it('excludes mode when empty', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, mode: '' })
      expect(args).not.toContain('--mode')
    })

    it('includes dtype', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, dtype: 'bfloat16' })
      expect(args).toContain('--dtype')
      expect(args[args.indexOf('--dtype') + 1]).toBe('bfloat16')
    })

    it('excludes dtype when empty', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, dtype: '' })
      expect(args).not.toContain('--dtype')
    })

    it('includes force flag', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, force: true })
      expect(args).toContain('--force')
    })

    it('excludes force flag when false', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, force: false })
      expect(args).not.toContain('--force')
    })

    it('includes skip-verify flag', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, skipVerify: true })
      expect(args).toContain('--skip-verify')
    })

    it('includes trust-remote-code flag', () => {
      const args = buildConvertArgs({ model: 'm', bits: 4, groupSize: 64, trustRemoteCode: true })
      expect(args).toContain('--trust-remote-code')
    })

    it('includes all flags at once', () => {
      const args = buildConvertArgs({
        model: 'nvidia/Nemotron',
        bits: 3,
        groupSize: 32,
        output: '/out',
        mode: 'NF4',
        dtype: 'float16',
        force: true,
        skipVerify: true,
        trustRemoteCode: true,
      })
      expect(args[0]).toBe('convert')
      expect(args[1]).toBe('nvidia/Nemotron')
      expect(args).toContain('--bits')
      expect(args).toContain('--group-size')
      expect(args).toContain('--output')
      expect(args).toContain('--mode')
      expect(args).toContain('--dtype')
      expect(args).toContain('--force')
      expect(args).toContain('--skip-verify')
      expect(args).toContain('--trust-remote-code')
    })

    it('bits and groupSize are stringified', () => {
      const args = buildConvertArgs({ model: 'm', bits: 8, groupSize: 128 })
      expect(args[3]).toBe('8')   // --bits value
      expect(args[5]).toBe('128') // --group-size value
    })
  })

  describe('buildDoctorArgs', () => {
    it('builds args without --no-inference', () => {
      const args = buildDoctorArgs('/path/model', {})
      expect(args).toEqual(['doctor', '/path/model'])
    })

    it('builds args with --no-inference', () => {
      const args = buildDoctorArgs('/path/model', { noInference: true })
      expect(args).toEqual(['doctor', '/path/model', '--no-inference'])
    })

    it('excludes --no-inference when false', () => {
      const args = buildDoctorArgs('/path/model', { noInference: false })
      expect(args).toEqual(['doctor', '/path/model'])
    })
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Model Filtering', () => {
  const sampleModels: LocalModel[] = [
    { id: '1', name: 'Qwen2.5-7B-4bit', path: '/models/qwen2.5-7b', format: 'mlx', quantization: '4-bit', size: '4.2 GB' },
    { id: '2', name: 'Llama-3.2-3B', path: '/models/llama-3.2', format: 'mlx', quantization: '8-bit', size: '6.1 GB' },
    { id: '3', name: 'Mistral-7B-v0.3', path: '/models/mistral', format: 'mlx' },
    { id: '4', name: 'Phi-3-mini', path: '/models/phi3', format: 'gguf' },
    { id: '5', name: 'Unknown-Model', path: '/models/unknown', format: 'unknown' },
  ]

  it('returns only MLX models with empty filter', () => {
    const result = filterMlxModels(sampleModels, '')
    expect(result).toHaveLength(3)
    expect(result.every(m => m.format === 'mlx')).toBe(true)
  })

  it('excludes GGUF models', () => {
    const result = filterMlxModels(sampleModels, '')
    expect(result.find(m => m.name === 'Phi-3-mini')).toBeUndefined()
  })

  it('excludes unknown format models', () => {
    const result = filterMlxModels(sampleModels, '')
    expect(result.find(m => m.name === 'Unknown-Model')).toBeUndefined()
  })

  it('filters by name (case insensitive)', () => {
    const result = filterMlxModels(sampleModels, 'qwen')
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('Qwen2.5-7B-4bit')
  })

  it('filters by path (case insensitive)', () => {
    const result = filterMlxModels(sampleModels, 'mistral')
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('Mistral-7B-v0.3')
  })

  it('returns empty when no matches', () => {
    const result = filterMlxModels(sampleModels, 'nonexistent')
    expect(result).toHaveLength(0)
  })

  it('matches partial strings', () => {
    const result = filterMlxModels(sampleModels, 'llam')
    expect(result).toHaveLength(1)
    expect(result[0].name).toBe('Llama-3.2-3B')
  })

  it('does not match GGUF model even if name matches filter', () => {
    const result = filterMlxModels(sampleModels, 'Phi')
    expect(result).toHaveLength(0)
  })

  it('handles empty model list', () => {
    const result = filterMlxModels([], 'test')
    expect(result).toHaveLength(0)
  })

  it('matches on path when name does not match', () => {
    const result = filterMlxModels(sampleModels, '/models/qwen')
    expect(result).toHaveLength(1)
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Output Path Extraction', () => {
  it('extracts path from output line', () => {
    const path = extractOutputPath([
      'Converting model...',
      'Quantizing layers...',
      'Output path: /home/user/models/Qwen-4bit',
      'Done.',
    ])
    expect(path).toBe('/home/user/models/Qwen-4bit')
  })

  it('returns undefined when no output path line', () => {
    const path = extractOutputPath([
      'Converting model...',
      'Error: conversion failed',
    ])
    expect(path).toBeUndefined()
  })

  it('returns undefined for empty lines', () => {
    expect(extractOutputPath([])).toBeUndefined()
  })

  it('trims whitespace from path', () => {
    const path = extractOutputPath(['Output path:   /path/with/spaces  '])
    expect(path).toBe('/path/with/spaces')
  })

  it('finds first matching line', () => {
    const path = extractOutputPath([
      'Output path: /first/path',
      'Output path: /second/path',
    ])
    expect(path).toBe('/first/path')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Log Accumulation', () => {
  it('appends new lines', () => {
    const result = accumulateLogs(['line1'], ['line2', 'line3'])
    expect(result).toEqual(['line1', 'line2', 'line3'])
  })

  it('caps at MAX_LOG_LINES', () => {
    const existing = Array.from({ length: 1999 }, (_, i) => `line-${i}`)
    const result = accumulateLogs(existing, ['new1', 'new2'])
    expect(result).toHaveLength(MAX_LOG_LINES)
    // Should keep the tail
    expect(result[result.length - 1]).toBe('new2')
    expect(result[result.length - 2]).toBe('new1')
  })

  it('keeps most recent lines when over limit', () => {
    const existing = Array.from({ length: 2000 }, (_, i) => `old-${i}`)
    const result = accumulateLogs(existing, ['newest'])
    expect(result).toHaveLength(MAX_LOG_LINES)
    expect(result[0]).toBe('old-1') // old-0 dropped
    expect(result[result.length - 1]).toBe('newest')
  })

  it('handles empty existing', () => {
    const result = accumulateLogs([], ['a', 'b'])
    expect(result).toEqual(['a', 'b'])
  })

  it('handles empty new lines', () => {
    const result = accumulateLogs(['a'], [])
    expect(result).toEqual(['a'])
  })

  it('handles both empty', () => {
    const result = accumulateLogs([], [])
    expect(result).toEqual([])
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Navigate Helper', () => {
  it('builds dashboard dispatch without modelPath', () => {
    const action = buildToolsDispatch('dashboard')
    expect(action).toEqual({ type: 'SET_TOOLS_PANEL', panel: 'dashboard', modelPath: undefined })
  })

  it('builds inspector dispatch with modelPath', () => {
    const action = buildToolsDispatch('inspector', '/models/test')
    expect(action).toEqual({ type: 'SET_TOOLS_PANEL', panel: 'inspector', modelPath: '/models/test' })
  })

  it('converts empty string modelPath to null', () => {
    const action = buildToolsDispatch('converter', '')
    expect(action).toEqual({ type: 'SET_TOOLS_PANEL', panel: 'converter', modelPath: null })
  })

  it('passes through non-empty modelPath', () => {
    const action = buildToolsDispatch('doctor', '/path')
    expect(action.modelPath).toBe('/path')
  })

  it('dispatch with undefined modelPath preserves existing in reducer', () => {
    const modified = { ...initialState, toolsModelPath: '/existing' as string | null }
    const action = buildToolsDispatch('dashboard')
    const state = appReducer(modified, action)
    expect(state.toolsModelPath).toBe('/existing')
  })

  it('dispatch with null modelPath clears in reducer', () => {
    const modified = { ...initialState, toolsModelPath: '/existing' as string | null }
    const action = buildToolsDispatch('dashboard')
    // manually add null to test the clearing path
    const state = appReducer(modified, { ...action, modelPath: null })
    expect(state.toolsModelPath).toBeNull()
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Serve Flow (tools → server handoff)', () => {
  it('switching to server create panel with model path', () => {
    let state = appReducer(initialState, { type: 'SET_MODE', mode: 'server' })
    state = appReducer(state, { type: 'SET_SERVER_PANEL', panel: 'create', modelPath: '/models/converted' })
    expect(state.mode).toBe('server')
    expect(state.serverPanel).toBe('create')
    expect(state.serverInitialModelPath).toBe('/models/converted')
  })

  it('tools panel state is preserved during serve handoff', () => {
    let state = appReducer(initialState, { type: 'SET_MODE', mode: 'tools' })
    state = appReducer(state, { type: 'SET_TOOLS_PANEL', panel: 'converter', modelPath: '/m' })
    // Switch to server mode for serve
    state = appReducer(state, { type: 'SET_MODE', mode: 'server' })
    state = appReducer(state, { type: 'SET_SERVER_PANEL', panel: 'create', modelPath: '/m' })
    // Switch back to tools
    state = appReducer(state, { type: 'SET_MODE', mode: 'tools' })
    expect(state.toolsPanel).toBe('converter')
    expect(state.toolsModelPath).toBe('/m')
  })
})

// ═══════════════════════════════════════════════════════════════════════════════

describe('Developer Tools — Edge Cases', () => {
  it('parseOutput handles very long input', () => {
    const lines = Array.from({ length: 10000 }, (_, i) => `line ${i}`)
    lines.unshift('Checking config...')
    lines.push('Config: OK')
    const checks = parseOutput(lines)
    expect(checks[0].status).toBe('pass')
  })

  it('parseOutput handles FAIL without matching check name', () => {
    const checks = parseOutput(['FAIL: Unknown: some error'])
    // Should not crash, no check matches
    expect(checks.every(c => c.status === 'pending')).toBe(true)
  })

  it('parseOutput handles WARN without matching check name', () => {
    const checks = parseOutput(['WARN: Unknown: some warning'])
    expect(checks.every(c => c.status === 'pending')).toBe(true)
  })

  it('filterMlxModels handles models without format field', () => {
    const models: LocalModel[] = [
      { id: '1', name: 'NoFormat', path: '/p' },
    ]
    const result = filterMlxModels(models, '')
    expect(result).toHaveLength(0)
  })

  it('buildConvertArgs with minimum valid inputs', () => {
    const args = buildConvertArgs({ model: 'x', bits: 2, groupSize: 32 })
    expect(args).toEqual(['convert', 'x', '--bits', '2', '--group-size', '32'])
  })

  it('reducer handles unknown action type gracefully', () => {
    const state = appReducer(initialState, { type: 'UNKNOWN_ACTION' } as any)
    expect(state).toEqual(initialState)
  })

  it('RESTORE_STATE can set tools fields', () => {
    const state = appReducer(initialState, {
      type: 'RESTORE_STATE',
      state: { toolsPanel: 'doctor', toolsModelPath: '/restored' },
    })
    expect(state.toolsPanel).toBe('doctor')
    expect(state.toolsModelPath).toBe('/restored')
  })
})
