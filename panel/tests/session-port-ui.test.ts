/**
 * Session, Port, Status & UI Logic Tests
 *
 * Coverage:
 *   - Port assignment logic (findAvailablePort)
 *   - SessionView status dot colors (local + remote)
 *   - SessionCard status colors and labels
 *   - ImageTopBar status colors (starting vs loading naming)
 *   - Quantize label formatting
 *   - ImagePromptBar canSubmit logic (disabled, generating, prompt, edit mode)
 *   - isImageEdit detection from session config (imageMode field)
 *   - connectHost address rewriting (0.0.0.0 -> 127.0.0.1)
 *   - Menu bar button visibility (text/image, local/remote, per status)
 */
import { describe, it, expect } from 'vitest'

// ─── 1. Port Assignment Logic ────────────────────────────────────────────────
// Re-implemented from sessions.ts findAvailablePort (line 1472)
// The real version also checks isPortFree via net.createServer; we test the
// pure port-selection logic only (Set-based filtering of active sessions).

interface PortSession {
  port: number
  status: string
}

function findAvailablePort(sessions: PortSession[]): number {
  const usedPorts = new Set(
    sessions
      .filter(s => s.status === 'running' || s.status === 'loading')
      .map(s => s.port)
  )
  let port = 8000
  while (usedPorts.has(port)) {
    port++
    if (port > 65535) throw new Error('No available ports')
  }
  return port
}

// ─── 2. SessionView Status Dot Colors ────────────────────────────────────────
// Re-implemented from SessionView.tsx (line 204-208)

function getStatusColor(status: string, isRemote: boolean): string {
  if (status === 'running') return isRemote ? 'bg-success' : 'bg-primary'
  if (status === 'loading') return 'bg-warning'
  if (status === 'error') return 'bg-destructive'
  return 'bg-muted-foreground' // stopped
}

// ─── 3. SessionCard Status Colors ────────────────────────────────────────────
// Re-implemented from SessionCard.tsx (line 31-36)

const sessionCardStatusColors: Record<string, string> = {
  running: 'bg-primary',
  stopped: 'bg-muted-foreground',
  error: 'bg-destructive',
  loading: 'bg-warning'
}

const sessionCardStatusLabels: Record<string, string> = {
  running: 'Running',
  stopped: 'Stopped',
  error: 'Error',
  loading: 'Loading...'
}

// ─── 4. ImageTopBar Status Colors ────────────────────────────────────────────
// Re-implemented from ImageTopBar.tsx (line 73-78)
// NOTE: Image tab uses 'starting' not 'loading'

function getImageStatusColor(status: string): string {
  if (status === 'running') return 'bg-green-500'
  if (status === 'starting') return 'bg-yellow-500 animate-pulse'
  if (status === 'error') return 'bg-red-500'
  return 'bg-gray-400' // stopped
}

function getImageStatusLabel(status: string, port: number | null): string {
  if (status === 'running' && port) return `Running on :${port}`
  if (status === 'starting') return 'Starting...'
  if (status === 'error') return 'Error'
  return 'Stopped'
}

// ─── 5. Quantize Label ──────────────────────────────────────────────────────
// Re-implemented from ImageTopBar.tsx (line 30)

function getQuantizeLabel(quantize: number): string {
  return quantize === 0 ? 'Full' : `${quantize}-bit`
}

// ─── 6. ImagePromptBar canSubmit ─────────────────────────────────────────────
// Re-implemented from ImagePromptBar.tsx (line 43)

function canSubmit(
  disabled: boolean,
  generating: boolean,
  prompt: string,
  isEdit: boolean,
  hasSourceImage: boolean
): boolean {
  return !disabled && !generating && prompt.trim().length > 0 && (!isEdit || hasSourceImage)
}

// ─── 7. isImageEdit Detection ────────────────────────────────────────────────
// Re-implemented from SessionView.tsx (line 209-213)
// Edit vs gen detection uses config.imageMode — no regex on model names.

function isImageEdit(modelType: string | undefined, imageMode: string | undefined): boolean {
  if (modelType !== 'image') return false
  return imageMode === 'edit'
}

// ─── 8. connectHost ──────────────────────────────────────────────────────────
// Re-implemented from sessions.ts (line 36-38)

function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

// ─── 9. Menu Bar Button Visibility ───────────────────────────────────────────
// Re-implemented from SessionView.tsx (line 255-374)

function getVisibleButtons(isImage: boolean, isRemote: boolean, status: string): string[] {
  const buttons: string[] = []
  if (!isImage) {
    buttons.push('Chat', 'ChatSettings')
  }
  if (isImage) {
    buttons.push('ImageTab')
  }
  buttons.push('ServerSettings', 'Logs')
  if (!isRemote && !isImage && status === 'running') {
    buttons.push('Cache', 'Bench', 'Embed', 'Perf')
  }
  if (status === 'running') buttons.push('Stop')
  if (status === 'stopped' || status === 'error') buttons.push('Start')
  if (status === 'loading') buttons.push('Cancel')
  return buttons
}

// ─── Helper: shortName extraction (SessionView line 203, SessionCard line 48) ─
function shortName(modelName: string | undefined, modelPath: string): string {
  return modelName || modelPath.split('/').pop() || modelPath
}

// =============================================================================
// TESTS
// =============================================================================

describe('Port Assignment Logic', () => {
  it('returns 8000 for empty sessions', () => {
    expect(findAvailablePort([])).toBe(8000)
  })

  it('returns 8000 when all sessions are stopped', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'stopped' },
      { port: 8001, status: 'stopped' },
    ])).toBe(8000)
  })

  it('skips port used by running session', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
    ])).toBe(8001)
  })

  it('skips port used by loading session', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'loading' },
    ])).toBe(8001)
  })

  it('does not skip port used by stopped session', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'stopped' },
    ])).toBe(8000)
  })

  it('does not skip port used by error session', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'error' },
    ])).toBe(8000)
  })

  it('skips multiple running sessions', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 8001, status: 'running' },
      { port: 8002, status: 'running' },
    ])).toBe(8003)
  })

  it('finds gap between used ports (8000 used, 8001 free)', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 8002, status: 'running' },
    ])).toBe(8001)
  })

  it('skips mix of running and loading', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 8001, status: 'loading' },
    ])).toBe(8002)
  })

  it('ignores stopped among active sessions', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 8001, status: 'stopped' },
      { port: 8002, status: 'running' },
    ])).toBe(8001)
  })

  it('handles non-contiguous ports', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 9000, status: 'running' },
    ])).toBe(8001)
  })

  it('handles large port numbers', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 60000, status: 'running' },
    ])).toBe(8001)
  })

  it('handles duplicate ports in sessions', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'running' },
      { port: 8000, status: 'stopped' },
    ])).toBe(8001)
  })

  it('throws when all ports are used', () => {
    // Simulate all ports from 8000 to 65535 being used
    const sessions: PortSession[] = []
    for (let p = 8000; p <= 65535; p++) {
      sessions.push({ port: p, status: 'running' })
    }
    expect(() => findAvailablePort(sessions)).toThrow('No available ports')
  })

  it('returns 8000 with only high-port sessions', () => {
    expect(findAvailablePort([
      { port: 50000, status: 'running' },
      { port: 60000, status: 'running' },
    ])).toBe(8000)
  })

  it('handles single stopped session on 8000', () => {
    expect(findAvailablePort([
      { port: 8000, status: 'stopped' },
    ])).toBe(8000)
  })

  it('handles 100 consecutive running sessions', () => {
    const sessions: PortSession[] = []
    for (let i = 0; i < 100; i++) {
      sessions.push({ port: 8000 + i, status: 'running' })
    }
    expect(findAvailablePort(sessions)).toBe(8100)
  })
})

describe('SessionView Status Dot Colors', () => {
  describe('local sessions (isRemote=false)', () => {
    it('running → bg-primary', () => {
      expect(getStatusColor('running', false)).toBe('bg-primary')
    })

    it('loading → bg-warning', () => {
      expect(getStatusColor('loading', false)).toBe('bg-warning')
    })

    it('error → bg-destructive', () => {
      expect(getStatusColor('error', false)).toBe('bg-destructive')
    })

    it('stopped → bg-muted-foreground', () => {
      expect(getStatusColor('stopped', false)).toBe('bg-muted-foreground')
    })
  })

  describe('remote sessions (isRemote=true)', () => {
    it('running → bg-success (different from local)', () => {
      expect(getStatusColor('running', true)).toBe('bg-success')
    })

    it('loading → bg-warning (same as local)', () => {
      expect(getStatusColor('loading', true)).toBe('bg-warning')
    })

    it('error → bg-destructive (same as local)', () => {
      expect(getStatusColor('error', true)).toBe('bg-destructive')
    })

    it('stopped → bg-muted-foreground (same as local)', () => {
      expect(getStatusColor('stopped', true)).toBe('bg-muted-foreground')
    })
  })

  describe('cross-status distinctions', () => {
    it('stopped and error have different colors', () => {
      expect(getStatusColor('stopped', false)).not.toBe(getStatusColor('error', false))
    })

    it('running local and running remote have different colors', () => {
      expect(getStatusColor('running', false)).not.toBe(getStatusColor('running', true))
    })

    it('all four local statuses produce unique colors', () => {
      const colors = ['running', 'loading', 'error', 'stopped'].map(s => getStatusColor(s, false))
      expect(new Set(colors).size).toBe(4)
    })

    it('unknown status falls through to bg-muted-foreground', () => {
      expect(getStatusColor('unknown', false)).toBe('bg-muted-foreground')
    })

    it('empty string falls through to bg-muted-foreground', () => {
      expect(getStatusColor('', false)).toBe('bg-muted-foreground')
    })
  })
})

describe('SessionCard Status Colors', () => {
  it('running → bg-primary', () => {
    expect(sessionCardStatusColors['running']).toBe('bg-primary')
  })

  it('stopped → bg-muted-foreground', () => {
    expect(sessionCardStatusColors['stopped']).toBe('bg-muted-foreground')
  })

  it('error → bg-destructive', () => {
    expect(sessionCardStatusColors['error']).toBe('bg-destructive')
  })

  it('loading → bg-warning', () => {
    expect(sessionCardStatusColors['loading']).toBe('bg-warning')
  })

  it('has all 4 statuses', () => {
    expect(Object.keys(sessionCardStatusColors)).toEqual(['running', 'stopped', 'error', 'loading'])
  })

  describe('labels', () => {
    it('running → Running', () => {
      expect(sessionCardStatusLabels['running']).toBe('Running')
    })

    it('stopped → Stopped', () => {
      expect(sessionCardStatusLabels['stopped']).toBe('Stopped')
    })

    it('error → Error', () => {
      expect(sessionCardStatusLabels['error']).toBe('Error')
    })

    it('loading → Loading...', () => {
      expect(sessionCardStatusLabels['loading']).toBe('Loading...')
    })
  })

  describe('consistency with SessionView', () => {
    it('stopped uses bg-muted-foreground in both SessionCard and SessionView', () => {
      expect(sessionCardStatusColors['stopped']).toBe('bg-muted-foreground')
      expect(getStatusColor('stopped', false)).toBe('bg-muted-foreground')
    })

    it('error uses bg-destructive in both', () => {
      expect(sessionCardStatusColors['error']).toBe('bg-destructive')
      expect(getStatusColor('error', false)).toBe('bg-destructive')
    })

    it('loading uses bg-warning in both', () => {
      expect(sessionCardStatusColors['loading']).toBe('bg-warning')
      expect(getStatusColor('loading', false)).toBe('bg-warning')
    })

    it('running uses bg-primary in both (for local)', () => {
      expect(sessionCardStatusColors['running']).toBe('bg-primary')
      expect(getStatusColor('running', false)).toBe('bg-primary')
    })
  })
})

describe('ImageTopBar Status Colors', () => {
  it('running → bg-green-500', () => {
    expect(getImageStatusColor('running')).toBe('bg-green-500')
  })

  it('starting → bg-yellow-500 animate-pulse', () => {
    expect(getImageStatusColor('starting')).toBe('bg-yellow-500 animate-pulse')
  })

  it('error → bg-red-500', () => {
    expect(getImageStatusColor('error')).toBe('bg-red-500')
  })

  it('stopped → bg-gray-400', () => {
    expect(getImageStatusColor('stopped')).toBe('bg-gray-400')
  })

  it('image uses starting not loading (different from sessions)', () => {
    // 'loading' is unknown to image tab — falls to default
    expect(getImageStatusColor('loading')).toBe('bg-gray-400')
    // 'starting' is the image-tab equivalent
    expect(getImageStatusColor('starting')).toBe('bg-yellow-500 animate-pulse')
  })

  it('unknown status → gray', () => {
    expect(getImageStatusColor('whatever')).toBe('bg-gray-400')
  })

  describe('status labels', () => {
    it('running with port shows port', () => {
      expect(getImageStatusLabel('running', 8000)).toBe('Running on :8000')
    })

    it('running without port shows Stopped', () => {
      expect(getImageStatusLabel('running', null)).toBe('Stopped')
    })

    it('starting shows Starting...', () => {
      expect(getImageStatusLabel('starting', null)).toBe('Starting...')
    })

    it('error shows Error', () => {
      expect(getImageStatusLabel('error', null)).toBe('Error')
    })

    it('stopped shows Stopped', () => {
      expect(getImageStatusLabel('stopped', null)).toBe('Stopped')
    })
  })
})

describe('Quantize Label', () => {
  it('0 → Full', () => {
    expect(getQuantizeLabel(0)).toBe('Full')
  })

  it('4 → 4-bit', () => {
    expect(getQuantizeLabel(4)).toBe('4-bit')
  })

  it('8 → 8-bit', () => {
    expect(getQuantizeLabel(8)).toBe('8-bit')
  })

  it('2 → 2-bit', () => {
    expect(getQuantizeLabel(2)).toBe('2-bit')
  })

  it('16 → 16-bit', () => {
    expect(getQuantizeLabel(16)).toBe('16-bit')
  })

  it('3 → 3-bit', () => {
    expect(getQuantizeLabel(3)).toBe('3-bit')
  })
})

describe('ImagePromptBar canSubmit', () => {
  describe('generate mode (isEdit=false)', () => {
    it('enabled + not generating + prompt → true', () => {
      expect(canSubmit(false, false, 'a cat', false, false)).toBe(true)
    })

    it('disabled → false', () => {
      expect(canSubmit(true, false, 'a cat', false, false)).toBe(false)
    })

    it('generating → false', () => {
      expect(canSubmit(false, true, 'a cat', false, false)).toBe(false)
    })

    it('empty prompt → false', () => {
      expect(canSubmit(false, false, '', false, false)).toBe(false)
    })

    it('whitespace-only prompt → false', () => {
      expect(canSubmit(false, false, '   ', false, false)).toBe(false)
    })

    it('tab-only prompt → false', () => {
      expect(canSubmit(false, false, '\t\t', false, false)).toBe(false)
    })

    it('newline-only prompt → false', () => {
      expect(canSubmit(false, false, '\n\n', false, false)).toBe(false)
    })

    it('generate mode does not require source image', () => {
      expect(canSubmit(false, false, 'test', false, false)).toBe(true)
    })

    it('generate mode with source image is fine', () => {
      expect(canSubmit(false, false, 'test', false, true)).toBe(true)
    })

    it('disabled + generating → false', () => {
      expect(canSubmit(true, true, 'test', false, false)).toBe(false)
    })

    it('disabled + empty prompt → false', () => {
      expect(canSubmit(true, false, '', false, false)).toBe(false)
    })
  })

  describe('edit mode (isEdit=true)', () => {
    it('prompt + source image → true', () => {
      expect(canSubmit(false, false, 'make it red', true, true)).toBe(true)
    })

    it('prompt but no source image → false', () => {
      expect(canSubmit(false, false, 'make it red', true, false)).toBe(false)
    })

    it('source image but empty prompt → false', () => {
      expect(canSubmit(false, false, '', true, true)).toBe(false)
    })

    it('no prompt + no source image → false', () => {
      expect(canSubmit(false, false, '', true, false)).toBe(false)
    })

    it('disabled with prompt + source image → false', () => {
      expect(canSubmit(true, false, 'edit', true, true)).toBe(false)
    })

    it('generating with prompt + source image → false', () => {
      expect(canSubmit(false, true, 'edit', true, true)).toBe(false)
    })

    it('whitespace prompt in edit mode → false even with image', () => {
      expect(canSubmit(false, false, '  ', true, true)).toBe(false)
    })
  })

  describe('edge cases', () => {
    it('single character prompt works', () => {
      expect(canSubmit(false, false, 'x', false, false)).toBe(true)
    })

    it('very long prompt works', () => {
      expect(canSubmit(false, false, 'a'.repeat(10000), false, false)).toBe(true)
    })

    it('prompt with only spaces around text works', () => {
      expect(canSubmit(false, false, '  hello  ', false, false)).toBe(true)
    })
  })
})

describe('isImageEdit Detection', () => {
  describe('non-image models return false regardless of imageMode', () => {
    it('text model type → false', () => {
      expect(isImageEdit('text', 'edit')).toBe(false)
    })

    it('undefined model type → false', () => {
      expect(isImageEdit(undefined, 'edit')).toBe(false)
    })

    it('empty model type → false', () => {
      expect(isImageEdit('', 'edit')).toBe(false)
    })

    it('llm model type → false', () => {
      expect(isImageEdit('llm', 'edit')).toBe(false)
    })
  })

  describe('image models with generate mode return false', () => {
    it('imageMode=generate → false', () => {
      expect(isImageEdit('image', 'generate')).toBe(false)
    })

    it('imageMode=undefined → false', () => {
      expect(isImageEdit('image', undefined)).toBe(false)
    })

    it('imageMode="" → false', () => {
      expect(isImageEdit('image', '')).toBe(false)
    })
  })

  describe('image models with edit mode return true', () => {
    it('imageMode=edit → true', () => {
      expect(isImageEdit('image', 'edit')).toBe(true)
    })
  })

  describe('edge cases', () => {
    it('imageMode is case-sensitive (Edit != edit)', () => {
      expect(isImageEdit('image', 'Edit')).toBe(false)
    })

    it('imageMode with extra whitespace → false', () => {
      expect(isImageEdit('image', ' edit ')).toBe(false)
    })
  })
})

describe('connectHost', () => {
  it('rewrites 0.0.0.0 to 127.0.0.1', () => {
    expect(connectHost('0.0.0.0')).toBe('127.0.0.1')
  })

  it('passes through 127.0.0.1 unchanged', () => {
    expect(connectHost('127.0.0.1')).toBe('127.0.0.1')
  })

  it('passes through localhost unchanged', () => {
    expect(connectHost('localhost')).toBe('localhost')
  })

  it('passes through specific IP unchanged', () => {
    expect(connectHost('192.168.1.100')).toBe('192.168.1.100')
  })

  it('passes through empty string unchanged', () => {
    expect(connectHost('')).toBe('')
  })

  it('does not rewrite 0.0.0.1', () => {
    expect(connectHost('0.0.0.1')).toBe('0.0.0.1')
  })

  it('passes through IPv6 localhost unchanged', () => {
    expect(connectHost('::1')).toBe('::1')
  })

  it('passes through hostname unchanged', () => {
    expect(connectHost('myserver.local')).toBe('myserver.local')
  })
})

describe('Menu Bar Button Visibility', () => {
  describe('text model, local, running', () => {
    const buttons = getVisibleButtons(false, false, 'running')

    it('shows Chat', () => {
      expect(buttons).toContain('Chat')
    })

    it('shows ChatSettings', () => {
      expect(buttons).toContain('ChatSettings')
    })

    it('shows ServerSettings', () => {
      expect(buttons).toContain('ServerSettings')
    })

    it('shows Logs', () => {
      expect(buttons).toContain('Logs')
    })

    it('shows Cache', () => {
      expect(buttons).toContain('Cache')
    })

    it('shows Bench', () => {
      expect(buttons).toContain('Bench')
    })

    it('shows Embed', () => {
      expect(buttons).toContain('Embed')
    })

    it('shows Perf', () => {
      expect(buttons).toContain('Perf')
    })

    it('shows Stop', () => {
      expect(buttons).toContain('Stop')
    })

    it('does not show ImageTab', () => {
      expect(buttons).not.toContain('ImageTab')
    })

    it('does not show Start', () => {
      expect(buttons).not.toContain('Start')
    })

    it('does not show Cancel', () => {
      expect(buttons).not.toContain('Cancel')
    })

    it('has 9 buttons total', () => {
      expect(buttons).toEqual(['Chat', 'ChatSettings', 'ServerSettings', 'Logs', 'Cache', 'Bench', 'Embed', 'Perf', 'Stop'])
    })
  })

  describe('text model, local, stopped', () => {
    const buttons = getVisibleButtons(false, false, 'stopped')

    it('shows Chat and ChatSettings', () => {
      expect(buttons).toContain('Chat')
      expect(buttons).toContain('ChatSettings')
    })

    it('shows ServerSettings and Logs', () => {
      expect(buttons).toContain('ServerSettings')
      expect(buttons).toContain('Logs')
    })

    it('shows Start', () => {
      expect(buttons).toContain('Start')
    })

    it('does not show Cache/Bench/Embed/Perf (only shown when running)', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
      expect(buttons).not.toContain('Embed')
      expect(buttons).not.toContain('Perf')
    })

    it('does not show Stop', () => {
      expect(buttons).not.toContain('Stop')
    })
  })

  describe('text model, local, error', () => {
    const buttons = getVisibleButtons(false, false, 'error')

    it('shows Start (can restart after error)', () => {
      expect(buttons).toContain('Start')
    })

    it('does not show Stop', () => {
      expect(buttons).not.toContain('Stop')
    })

    it('does not show developer tools', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
    })
  })

  describe('text model, local, loading', () => {
    const buttons = getVisibleButtons(false, false, 'loading')

    it('shows Cancel', () => {
      expect(buttons).toContain('Cancel')
    })

    it('does not show Start or Stop', () => {
      expect(buttons).not.toContain('Start')
      expect(buttons).not.toContain('Stop')
    })

    it('does not show developer tools', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
      expect(buttons).not.toContain('Embed')
      expect(buttons).not.toContain('Perf')
    })
  })

  describe('image model, local, running', () => {
    const buttons = getVisibleButtons(true, false, 'running')

    it('shows ImageTab', () => {
      expect(buttons).toContain('ImageTab')
    })

    it('shows ServerSettings', () => {
      expect(buttons).toContain('ServerSettings')
    })

    it('shows Logs', () => {
      expect(buttons).toContain('Logs')
    })

    it('shows Stop', () => {
      expect(buttons).toContain('Stop')
    })

    it('does not show Chat or ChatSettings', () => {
      expect(buttons).not.toContain('Chat')
      expect(buttons).not.toContain('ChatSettings')
    })

    it('does not show Cache/Bench/Embed/Perf (image models skip these)', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
      expect(buttons).not.toContain('Embed')
      expect(buttons).not.toContain('Perf')
    })

    it('has 4 buttons total', () => {
      expect(buttons).toEqual(['ImageTab', 'ServerSettings', 'Logs', 'Stop'])
    })
  })

  describe('image model, local, loading', () => {
    const buttons = getVisibleButtons(true, false, 'loading')

    it('shows ImageTab, ServerSettings, Logs, Cancel', () => {
      expect(buttons).toEqual(['ImageTab', 'ServerSettings', 'Logs', 'Cancel'])
    })

    it('does not show Stop', () => {
      expect(buttons).not.toContain('Stop')
    })
  })

  describe('image model, local, stopped', () => {
    const buttons = getVisibleButtons(true, false, 'stopped')

    it('shows Start', () => {
      expect(buttons).toContain('Start')
    })

    it('does not show Stop or Cancel', () => {
      expect(buttons).not.toContain('Stop')
      expect(buttons).not.toContain('Cancel')
    })
  })

  describe('image model, local, error', () => {
    const buttons = getVisibleButtons(true, false, 'error')

    it('shows Start', () => {
      expect(buttons).toContain('Start')
    })

    it('has ImageTab, ServerSettings, Logs, Start', () => {
      expect(buttons).toEqual(['ImageTab', 'ServerSettings', 'Logs', 'Start'])
    })
  })

  describe('remote text model, running', () => {
    const buttons = getVisibleButtons(false, true, 'running')

    it('shows Chat and ChatSettings', () => {
      expect(buttons).toContain('Chat')
      expect(buttons).toContain('ChatSettings')
    })

    it('shows ServerSettings and Logs', () => {
      expect(buttons).toContain('ServerSettings')
      expect(buttons).toContain('Logs')
    })

    it('shows Stop', () => {
      expect(buttons).toContain('Stop')
    })

    it('does not show Cache/Bench/Embed/Perf (remote sessions skip these)', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
      expect(buttons).not.toContain('Embed')
      expect(buttons).not.toContain('Perf')
    })

    it('has 5 buttons', () => {
      expect(buttons).toEqual(['Chat', 'ChatSettings', 'ServerSettings', 'Logs', 'Stop'])
    })
  })

  describe('remote text model, stopped', () => {
    const buttons = getVisibleButtons(false, true, 'stopped')

    it('shows Start (Connect for remote)', () => {
      expect(buttons).toContain('Start')
    })

    it('does not show developer tools', () => {
      expect(buttons).not.toContain('Cache')
      expect(buttons).not.toContain('Bench')
    })
  })

  describe('remote text model, loading', () => {
    const buttons = getVisibleButtons(false, true, 'loading')

    it('shows Cancel', () => {
      expect(buttons).toContain('Cancel')
    })

    it('does not show developer tools', () => {
      expect(buttons).not.toContain('Cache')
    })
  })
})

describe('shortName extraction', () => {
  it('uses modelName when available', () => {
    expect(shortName('Llama-3', '/models/llama-3-8b')).toBe('Llama-3')
  })

  it('extracts last path segment when modelName is undefined', () => {
    expect(shortName(undefined, '/tmp/models/Qwen2.5-7B')).toBe('Qwen2.5-7B')
  })

  it('extracts last path segment when modelName is empty', () => {
    expect(shortName('', '/models/mistral-7b')).toBe('mistral-7b')
  })

  it('handles path with no slashes', () => {
    expect(shortName(undefined, 'gemma-2b')).toBe('gemma-2b')
  })

  it('handles path ending with slash — pop returns empty, falls through to full path', () => {
    // split('/').pop() returns '' which is falsy, so || chains to modelPath
    expect(shortName(undefined, '/models/test/')).toBe('/models/test/')
  })

  it('handles remote:// paths — extracts after last slash', () => {
    // split('/').pop() returns 'gpt-4@api.openai.com'
    expect(shortName(undefined, 'remote://gpt-4@api.openai.com')).toBe('gpt-4@api.openai.com')
  })

  it('prefers modelName over path', () => {
    expect(shortName('Custom Name', '/deep/nested/model-path-v2')).toBe('Custom Name')
  })
})

describe('Cross-component color consistency', () => {
  it('SessionView and SessionCard agree on stopped color', () => {
    expect(getStatusColor('stopped', false)).toBe(sessionCardStatusColors['stopped'])
  })

  it('SessionView and SessionCard agree on error color', () => {
    expect(getStatusColor('error', false)).toBe(sessionCardStatusColors['error'])
  })

  it('SessionView and SessionCard agree on loading color', () => {
    expect(getStatusColor('loading', false)).toBe(sessionCardStatusColors['loading'])
  })

  it('SessionView and SessionCard agree on running color (local)', () => {
    expect(getStatusColor('running', false)).toBe(sessionCardStatusColors['running'])
  })

  it('ImageTopBar uses different color system than sessions', () => {
    // Image tab uses Tailwind raw colors (green-500, red-500, etc.)
    // Session components use theme tokens (bg-primary, bg-destructive, etc.)
    expect(getImageStatusColor('running')).toBe('bg-green-500')
    expect(getStatusColor('running', false)).toBe('bg-primary')
    // These are intentionally different — image tab is a distinct visual context
  })

  it('image tab uses starting while sessions use loading', () => {
    expect(getImageStatusColor('starting')).toContain('bg-yellow-500')
    expect(getStatusColor('loading', false)).toBe('bg-warning')
    // Both represent "in progress" but with different status names and color tokens
  })
})

describe('Port + Session integration scenarios', () => {
  it('new session gets port 8000 by default', () => {
    const port = findAvailablePort([])
    expect(port).toBe(8000)
  })

  it('second concurrent session gets 8001', () => {
    const port = findAvailablePort([{ port: 8000, status: 'running' }])
    expect(port).toBe(8001)
  })

  it('restarted session can reuse its own port (stopped status)', () => {
    const port = findAvailablePort([
      { port: 8000, status: 'stopped' },
      { port: 8001, status: 'running' },
    ])
    // 8000 is available because the session holding it is stopped
    expect(port).toBe(8000)
  })

  it('error session port is available for reuse', () => {
    const port = findAvailablePort([
      { port: 8000, status: 'error' },
    ])
    expect(port).toBe(8000)
  })

  it('five concurrent running sessions use sequential ports', () => {
    const sessions = Array.from({ length: 5 }, (_, i) => ({
      port: 8000 + i,
      status: 'running'
    }))
    expect(findAvailablePort(sessions)).toBe(8005)
  })
})

describe('Menu bar button state transitions', () => {
  it('stopped → has Start, no Stop, no Cancel', () => {
    const b = getVisibleButtons(false, false, 'stopped')
    expect(b).toContain('Start')
    expect(b).not.toContain('Stop')
    expect(b).not.toContain('Cancel')
  })

  it('loading → has Cancel, no Start, no Stop', () => {
    const b = getVisibleButtons(false, false, 'loading')
    expect(b).toContain('Cancel')
    expect(b).not.toContain('Start')
    expect(b).not.toContain('Stop')
  })

  it('running → has Stop, no Start, no Cancel', () => {
    const b = getVisibleButtons(false, false, 'running')
    expect(b).toContain('Stop')
    expect(b).not.toContain('Start')
    expect(b).not.toContain('Cancel')
  })

  it('error → has Start, no Stop, no Cancel', () => {
    const b = getVisibleButtons(false, false, 'error')
    expect(b).toContain('Start')
    expect(b).not.toContain('Stop')
    expect(b).not.toContain('Cancel')
  })

  it('ServerSettings and Logs always present regardless of status', () => {
    for (const status of ['running', 'stopped', 'error', 'loading']) {
      const b = getVisibleButtons(false, false, status)
      expect(b).toContain('ServerSettings')
      expect(b).toContain('Logs')
    }
  })

  it('developer tools only appear for local text running', () => {
    const devTools = ['Cache', 'Bench', 'Embed', 'Perf']
    // Only this combo shows dev tools
    const running = getVisibleButtons(false, false, 'running')
    for (const t of devTools) expect(running).toContain(t)

    // All other combos should NOT show dev tools
    const combos: [boolean, boolean, string][] = [
      [false, false, 'stopped'],
      [false, false, 'error'],
      [false, false, 'loading'],
      [false, true, 'running'],  // remote
      [true, false, 'running'],  // image
      [true, true, 'running'],   // remote image
    ]
    for (const [isImage, isRemote, status] of combos) {
      const b = getVisibleButtons(isImage, isRemote, status)
      for (const t of devTools) {
        expect(b).not.toContain(t)
      }
    }
  })
})
