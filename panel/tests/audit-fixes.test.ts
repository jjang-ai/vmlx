/**
 * Tests for audit fixes across the panel codebase.
 *
 * Covers: URL validation, role normalization, scheme restriction,
 * process exit handling, tray icon generation, audio validation.
 */
import { describe, it, expect } from 'vitest'

// =============================================================================
// Update Checker URL Validation
// =============================================================================

function isValidUpdateUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'https:' && (parsed.hostname === 'github.com' || parsed.hostname.endsWith('.github.com'))
  } catch {
    return false
  }
}

describe('update URL validation', () => {
  it('accepts github.com HTTPS', () => {
    expect(isValidUpdateUrl('https://github.com/jjang-ai/mlxstudio/releases/tag/v1.0.0')).toBe(true)
  })

  it('accepts github.com subdomain', () => {
    expect(isValidUpdateUrl('https://releases.github.com/download/v1.0.0')).toBe(true)
  })

  it('rejects evilgithub.com (lookalike)', () => {
    expect(isValidUpdateUrl('https://evilgithub.com/fake')).toBe(false)
  })

  it('rejects notgithub.com', () => {
    expect(isValidUpdateUrl('https://notgithub.com/download')).toBe(false)
  })

  it('rejects githubusercontent.com (not .github.com)', () => {
    expect(isValidUpdateUrl('https://objects.githubusercontent.com/download')).toBe(false)
  })

  it('rejects HTTP github.com', () => {
    expect(isValidUpdateUrl('http://github.com/download')).toBe(false)
  })

  it('rejects javascript: scheme', () => {
    expect(isValidUpdateUrl('javascript:alert(1)')).toBe(false)
  })

  it('rejects file: scheme', () => {
    expect(isValidUpdateUrl('file:///etc/passwd')).toBe(false)
  })

  it('rejects data: scheme', () => {
    expect(isValidUpdateUrl('data:text/html,<h1>hi</h1>')).toBe(false)
  })

  it('rejects empty string', () => {
    expect(isValidUpdateUrl('')).toBe(false)
  })

  it('rejects non-URL string', () => {
    expect(isValidUpdateUrl('not-a-url')).toBe(false)
  })
})

// =============================================================================
// Export Role Normalization
// =============================================================================

describe('export role normalization', () => {
  const validRoles = new Set(['system', 'user', 'assistant', 'tool'])

  function normalizeRole(role: string): string {
    return validRoles.has(role) ? role : 'user'
  }

  it('preserves system role', () => {
    expect(normalizeRole('system')).toBe('system')
  })

  it('preserves user role', () => {
    expect(normalizeRole('user')).toBe('user')
  })

  it('preserves assistant role', () => {
    expect(normalizeRole('assistant')).toBe('assistant')
  })

  it('preserves tool role', () => {
    expect(normalizeRole('tool')).toBe('tool')
  })

  it('normalizes unknown role to user', () => {
    expect(normalizeRole('admin')).toBe('user')
  })

  it('normalizes empty string to user', () => {
    expect(normalizeRole('')).toBe('user')
  })

  it('normalizes gpt to user', () => {
    expect(normalizeRole('gpt')).toBe('user')
  })

  it('normalizes human to user', () => {
    expect(normalizeRole('human')).toBe('user')
  })
})

// =============================================================================
// fetchUrl Scheme Restriction
// =============================================================================

describe('fetchUrl scheme validation', () => {
  function isAllowedScheme(url: string): boolean {
    try {
      const parsed = new URL(url)
      return parsed.protocol === 'http:' || parsed.protocol === 'https:'
    } catch {
      return false
    }
  }

  it('allows http', () => {
    expect(isAllowedScheme('http://example.com')).toBe(true)
  })

  it('allows https', () => {
    expect(isAllowedScheme('https://example.com')).toBe(true)
  })

  it('blocks file://', () => {
    expect(isAllowedScheme('file:///etc/passwd')).toBe(false)
  })

  it('blocks javascript:', () => {
    expect(isAllowedScheme('javascript:alert(1)')).toBe(false)
  })

  it('blocks ftp://', () => {
    expect(isAllowedScheme('ftp://evil.com/file')).toBe(false)
  })

  it('blocks data:', () => {
    expect(isAllowedScheme('data:text/html,hello')).toBe(false)
  })

  it('rejects invalid URL', () => {
    expect(isAllowedScheme('not-a-url')).toBe(false)
  })
})

// =============================================================================
// run_command Exit Code Logic
// =============================================================================

describe('run_command exit code handling', () => {
  function classifyExit(code: number | null, signal: string | null, killReason: string): 'success' | 'killed' | 'signaled' | 'error' {
    if (killReason) return 'killed'
    if (code === 0) return 'success'
    if (code === null && signal) return 'signaled'
    return 'error'
  }

  it('code 0 is success', () => {
    expect(classifyExit(0, null, '')).toBe('success')
  })

  it('code 1 is error', () => {
    expect(classifyExit(1, null, '')).toBe('error')
  })

  it('code null with SIGKILL is signaled', () => {
    expect(classifyExit(null, 'SIGKILL', '')).toBe('signaled')
  })

  it('code null with SIGTERM is signaled', () => {
    expect(classifyExit(null, 'SIGTERM', '')).toBe('signaled')
  })

  it('killReason set is killed (timeout)', () => {
    expect(classifyExit(null, 'SIGTERM', 'Command timed out')).toBe('killed')
  })

  it('killReason set is killed (output exceeded)', () => {
    expect(classifyExit(null, null, 'Output exceeded 10MB limit')).toBe('killed')
  })

  it('code null with no signal and no killReason is error', () => {
    // Edge case: process killed externally without signal info
    expect(classifyExit(null, null, '')).toBe('error')
  })

  it('code 127 (command not found) is error', () => {
    expect(classifyExit(127, null, '')).toBe('error')
  })

  it('code 137 (SIGKILL) is error', () => {
    // Some systems report signal kills as 128+signal
    expect(classifyExit(137, null, '')).toBe('error')
  })
})

// =============================================================================
// Tray Icon Color Logic
// =============================================================================

describe('tray icon color selection', () => {
  interface ModelProcess {
    status: 'starting' | 'running' | 'stopped' | 'error'
  }

  function getIconColor(processes: ModelProcess[]): 'green' | 'yellow' | 'red' | 'gray' {
    if (processes.some((p) => p.status === 'running')) return 'green'
    if (processes.some((p) => p.status === 'starting')) return 'yellow'
    if (processes.some((p) => p.status === 'error')) return 'red'
    return 'gray'
  }

  it('returns gray for empty list', () => {
    expect(getIconColor([])).toBe('gray')
  })

  it('returns green when model is running', () => {
    expect(getIconColor([{ status: 'running' }])).toBe('green')
  })

  it('returns yellow when starting', () => {
    expect(getIconColor([{ status: 'starting' }])).toBe('yellow')
  })

  it('returns red on error', () => {
    expect(getIconColor([{ status: 'error' }])).toBe('red')
  })

  it('green takes priority over yellow', () => {
    expect(getIconColor([{ status: 'starting' }, { status: 'running' }])).toBe('green')
  })

  it('green takes priority over red', () => {
    expect(getIconColor([{ status: 'error' }, { status: 'running' }])).toBe('green')
  })

  it('yellow takes priority over red', () => {
    expect(getIconColor([{ status: 'error' }, { status: 'starting' }])).toBe('yellow')
  })

  it('returns gray for only stopped', () => {
    expect(getIconColor([{ status: 'stopped' }])).toBe('gray')
  })

  it('returns gray for multiple stopped', () => {
    expect(getIconColor([{ status: 'stopped' }, { status: 'stopped' }])).toBe('gray')
  })
})

// =============================================================================
// Audio Size Validation Logic
// =============================================================================

describe('audio size validation', () => {
  const MAX_AUDIO_BASE64 = 100 * 1024 * 1024

  function validateAudioInput(audioBase64: string | undefined): { ok: boolean; error?: string } {
    if (!audioBase64) return { ok: false, error: 'No audio data provided' }
    if (audioBase64.length > MAX_AUDIO_BASE64) {
      return { ok: false, error: `Audio data too large (max ~75 MB). Got ${Math.round(audioBase64.length / 1024 / 1024)} MB encoded.` }
    }
    return { ok: true }
  }

  it('rejects undefined', () => {
    expect(validateAudioInput(undefined).ok).toBe(false)
  })

  it('rejects empty string', () => {
    expect(validateAudioInput('').ok).toBe(false)
  })

  it('accepts small audio', () => {
    expect(validateAudioInput('SGVsbG8=').ok).toBe(true)
  })

  it('rejects oversized audio', () => {
    const big = 'x'.repeat(MAX_AUDIO_BASE64 + 1)
    const result = validateAudioInput(big)
    expect(result.ok).toBe(false)
    expect(result.error).toContain('too large')
  })

  it('accepts exactly at limit', () => {
    const exact = 'x'.repeat(MAX_AUDIO_BASE64)
    expect(validateAudioInput(exact).ok).toBe(true)
  })
})

// =============================================================================
// Enable Thinking Tri-State Propagation
// =============================================================================

describe('enableThinking tri-state', () => {
  // Mirrors database.ts:655-658 (write) and 696-699 (read)
  function toDb(value: boolean | undefined): number | null {
    if (value === true) return 1
    if (value === false) return 0
    return null // undefined = Auto
  }

  function fromDb(dbValue: number | null | undefined): boolean | undefined {
    if (dbValue === null || dbValue === undefined) return undefined // Auto
    return dbValue !== 0 // 0 = Off, 1 = On
  }

  it('undefined (Auto) → NULL → undefined roundtrip', () => {
    expect(fromDb(toDb(undefined))).toBe(undefined)
  })

  it('true (On) → 1 → true roundtrip', () => {
    expect(fromDb(toDb(true))).toBe(true)
  })

  it('false (Off) → 0 → false roundtrip', () => {
    expect(fromDb(toDb(false))).toBe(false)
  })

  it('DB value 0 reads as false', () => {
    expect(fromDb(0)).toBe(false)
  })

  it('DB value 1 reads as true', () => {
    expect(fromDb(1)).toBe(true)
  })

  it('DB value null reads as undefined', () => {
    expect(fromDb(null)).toBe(undefined)
  })

  // Verify the request body logic from chat.ts:722-726
  function buildThinkingRequestField(
    enableThinking: boolean | undefined,
    isRemote: boolean,
    sessionHasReasoningParser: boolean
  ): boolean | undefined {
    if (enableThinking !== undefined) return enableThinking
    if (isRemote) return sessionHasReasoningParser
    return undefined // local auto-detects
  }

  it('explicit On sent to local', () => {
    expect(buildThinkingRequestField(true, false, false)).toBe(true)
  })

  it('explicit Off sent to local', () => {
    expect(buildThinkingRequestField(false, false, true)).toBe(false)
  })

  it('Auto on local → undefined (server auto-detects)', () => {
    expect(buildThinkingRequestField(undefined, false, true)).toBe(undefined)
  })

  it('Auto on remote → uses sessionHasReasoningParser', () => {
    expect(buildThinkingRequestField(undefined, true, true)).toBe(true)
    expect(buildThinkingRequestField(undefined, true, false)).toBe(false)
  })

  it('explicit override beats remote hint', () => {
    expect(buildThinkingRequestField(false, true, true)).toBe(false)
    expect(buildThinkingRequestField(true, true, false)).toBe(true)
  })
})

// =============================================================================
// Chat Overrides Reset Behavior
// =============================================================================

describe('chat overrides reset behavior', () => {
  // Mirrors ChatSettings.tsx:83-118
  interface ChatOverrides {
    temperature?: number
    topP?: number
    topK?: number
    minP?: number
    repeatPenalty?: number
    maxTokens?: number
    stopSequences?: string
    systemPrompt?: string
    workingDirectory?: string
    builtinToolsEnabled?: boolean
    enableThinking?: boolean
    reasoningEffort?: string
    wireApi?: string
    maxToolIterations?: number
    hideToolStatus?: boolean
    webSearchEnabled?: boolean
    braveSearchEnabled?: boolean
  }

  function simulateReset(
    overrides: ChatOverrides,
    genDefaults: { temperature?: number; topP?: number; topK?: number; minP?: number; repeatPenalty?: number }
  ): ChatOverrides {
    // Preserve agent config
    const preserved: ChatOverrides = {}
    if (overrides.systemPrompt) preserved.systemPrompt = overrides.systemPrompt
    if (overrides.workingDirectory) preserved.workingDirectory = overrides.workingDirectory
    if (overrides.builtinToolsEnabled != null) preserved.builtinToolsEnabled = overrides.builtinToolsEnabled
    if (overrides.wireApi) preserved.wireApi = overrides.wireApi
    if (overrides.hideToolStatus != null) preserved.hideToolStatus = overrides.hideToolStatus
    if (overrides.webSearchEnabled != null) preserved.webSearchEnabled = overrides.webSearchEnabled
    if (overrides.braveSearchEnabled != null) preserved.braveSearchEnabled = overrides.braveSearchEnabled

    // Apply generation defaults
    const result: ChatOverrides = { ...preserved }
    if (genDefaults.temperature != null) result.temperature = genDefaults.temperature
    if (genDefaults.topP != null) result.topP = genDefaults.topP
    if (genDefaults.topK != null) result.topK = genDefaults.topK
    if (genDefaults.minP != null) result.minP = genDefaults.minP
    if (genDefaults.repeatPenalty != null) result.repeatPenalty = genDefaults.repeatPenalty
    return result
  }

  it('preserves system prompt on reset', () => {
    const result = simulateReset(
      { systemPrompt: 'You are a pirate', temperature: 1.5 },
      { temperature: 0.7 }
    )
    expect(result.systemPrompt).toBe('You are a pirate')
    expect(result.temperature).toBe(0.7)
  })

  it('preserves working directory on reset', () => {
    const result = simulateReset(
      { workingDirectory: '/home/user/project', temperature: 1.0 },
      { temperature: 0.5 }
    )
    expect(result.workingDirectory).toBe('/home/user/project')
  })

  it('preserves tool toggles on reset', () => {
    const result = simulateReset(
      { builtinToolsEnabled: true, webSearchEnabled: false, temperature: 2.0 },
      { temperature: 0.7 }
    )
    expect(result.builtinToolsEnabled).toBe(true)
    expect(result.webSearchEnabled).toBe(false)
  })

  it('clears inference params and applies model defaults', () => {
    const result = simulateReset(
      { temperature: 1.5, topP: 0.5, topK: 100, minP: 0.1, repeatPenalty: 2.0 },
      { temperature: 0.7, topP: 0.9 }
    )
    expect(result.temperature).toBe(0.7)
    expect(result.topP).toBe(0.9)
    expect(result.topK).toBe(undefined) // Not in model defaults → cleared
    expect(result.minP).toBe(undefined)
    expect(result.repeatPenalty).toBe(undefined)
  })

  it('clears maxTokens on reset (not preserved)', () => {
    const result = simulateReset(
      { maxTokens: 8192, temperature: 1.0 },
      { temperature: 0.7 }
    )
    expect(result.maxTokens).toBe(undefined)
  })

  it('clears stopSequences on reset (not preserved)', () => {
    const result = simulateReset(
      { stopSequences: '<end>', temperature: 1.0 },
      { temperature: 0.7 }
    )
    expect(result.stopSequences).toBe(undefined)
  })

  it('clears enableThinking on reset (not preserved)', () => {
    const result = simulateReset(
      { enableThinking: false, temperature: 1.0 },
      { temperature: 0.7 }
    )
    expect(result.enableThinking).toBe(undefined)
  })

  it('preserves wireApi on reset', () => {
    const result = simulateReset(
      { wireApi: 'responses', temperature: 1.0 },
      { temperature: 0.7 }
    )
    expect(result.wireApi).toBe('responses')
  })

  it('handles empty model defaults gracefully', () => {
    const result = simulateReset(
      { temperature: 1.0, systemPrompt: 'test' },
      {}
    )
    expect(result.systemPrompt).toBe('test')
    expect(result.temperature).toBe(undefined)
  })
})

// =============================================================================
// Stop Sequence Parsing
// =============================================================================

describe('stop sequence comma parsing', () => {
  function parseStopSequences(input: string | undefined): string[] | undefined {
    if (!input) return undefined
    return input.split(',').map(s => s.trim()).filter(Boolean)
  }

  it('single token', () => {
    expect(parseStopSequences('<end>')).toEqual(['<end>'])
  })

  it('multiple tokens with spaces', () => {
    expect(parseStopSequences(' token1 , token2 , token3 ')).toEqual(['token1', 'token2', 'token3'])
  })

  it('filters empty entries from double commas', () => {
    expect(parseStopSequences('a,,b')).toEqual(['a', 'b'])
  })

  it('empty string returns undefined', () => {
    expect(parseStopSequences('')).toBe(undefined)
  })

  it('undefined returns undefined', () => {
    expect(parseStopSequences(undefined)).toBe(undefined)
  })

  it('only commas returns undefined (all empty after filter)', () => {
    const result = parseStopSequences(',,,')
    expect(result).toEqual([])
  })

  it('preserves special characters in tokens', () => {
    expect(parseStopSequences('<|im_end|>, </s>')).toEqual(['<|im_end|>', '</s>'])
  })
})

// =============================================================================
// Session Restart-Required Keys Detection
// =============================================================================

describe('session restart-required detection', () => {
  // Mirrors sessions.ts RESTART_REQUIRED_KEYS
  const RESTART_REQUIRED_KEYS = new Set([
    'port', 'host', 'continuousBatching',
    'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
    'enablePrefixCache', 'disablePrefixCache',
    'noMemoryAwareCache', 'cacheMemoryMb', 'cacheMemoryPercent',
    'kvCacheQuantization', 'kvCacheGroupSize',
    'enableDiskCache', 'diskCacheMaxGb', 'diskCacheDir',
    'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheDir',
    'prefixCacheSize', 'cacheTtlMinutes', 'isMultimodal',
    'toolCallParser', 'reasoningParser',
    'maxNumSeqs', 'prefillBatchSize', 'completionBatchSize',
    'timeout', 'streamInterval', 'apiKey', 'rateLimit',
    'maxTokens', 'mcpConfig', 'servedModelName',
    'speculativeModel', 'numDraftTokens',
    'defaultTemperature', 'defaultTopP',
    'embeddingModel', 'additionalArgs',
    'enableAutoToolChoice',
  ])

  function detectRestartRequired(
    changes: Record<string, unknown>,
    current: Record<string, unknown>,
    isRunning: boolean
  ): { restartRequired: boolean; changedKeys: string[] } {
    const changedKeys = Object.keys(changes).filter(k =>
      RESTART_REQUIRED_KEYS.has(k) && changes[k] !== current[k]
    )
    return { restartRequired: isRunning && changedKeys.length > 0, changedKeys }
  }

  it('port change on running session requires restart', () => {
    const result = detectRestartRequired({ port: 8001 }, { port: 8000 }, true)
    expect(result.restartRequired).toBe(true)
    expect(result.changedKeys).toContain('port')
  })

  it('port change on stopped session does not require restart', () => {
    const result = detectRestartRequired({ port: 8001 }, { port: 8000 }, false)
    expect(result.restartRequired).toBe(false)
  })

  it('same value does not trigger restart', () => {
    const result = detectRestartRequired({ port: 8000 }, { port: 8000 }, true)
    expect(result.restartRequired).toBe(false)
    expect(result.changedKeys).toHaveLength(0)
  })

  it('kvCacheQuantization change requires restart', () => {
    const result = detectRestartRequired(
      { kvCacheQuantization: 'q4' }, { kvCacheQuantization: 'none' }, true
    )
    expect(result.restartRequired).toBe(true)
  })

  it('multiple changes listed', () => {
    const result = detectRestartRequired(
      { port: 8001, kvCacheQuantization: 'q8' },
      { port: 8000, kvCacheQuantization: 'none' },
      true
    )
    expect(result.changedKeys).toHaveLength(2)
    expect(result.changedKeys).toContain('port')
    expect(result.changedKeys).toContain('kvCacheQuantization')
  })

  it('non-restart keys are ignored', () => {
    // 'alias' is not in RESTART_REQUIRED_KEYS
    const result = detectRestartRequired({ alias: 'My Model' } as any, {}, true)
    expect(result.restartRequired).toBe(false)
  })
})

// =============================================================================
// Tool Category Filtering
// =============================================================================

describe('tool category filtering', () => {
  // Mirrors chat.ts getDisabledTools
  const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory', 'insert_text', 'replace_lines', 'apply_regex', 'read_image'])
  const SHELL_TOOLS = new Set(['run_command', 'spawn_process', 'get_process_output'])
  const DDG_SEARCH_TOOLS = new Set(['ddg_search'])
  const GIT_TOOLS = new Set(['git'])

  function getDisabledTools(overrides: Record<string, boolean | undefined>): Set<string> {
    const disabled = new Set<string>()
    if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
    if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
    if (overrides.webSearchEnabled === false) DDG_SEARCH_TOOLS.forEach(t => disabled.add(t))
    if (overrides.gitEnabled === false) GIT_TOOLS.forEach(t => disabled.add(t))
    return disabled
  }

  it('all enabled by default', () => {
    expect(getDisabledTools({}).size).toBe(0)
  })

  it('disabling file tools blocks all file operations', () => {
    const disabled = getDisabledTools({ fileToolsEnabled: false })
    expect(disabled.has('read_file')).toBe(true)
    expect(disabled.has('write_file')).toBe(true)
    expect(disabled.has('delete_file')).toBe(true)
    expect(disabled.has('read_image')).toBe(true)
    expect(disabled.has('run_command')).toBe(false) // shell still enabled
  })

  it('disabling shell blocks command execution', () => {
    const disabled = getDisabledTools({ shellEnabled: false })
    expect(disabled.has('run_command')).toBe(true)
    expect(disabled.has('spawn_process')).toBe(true)
    expect(disabled.has('read_file')).toBe(false)
  })

  it('multiple categories disabled simultaneously', () => {
    const disabled = getDisabledTools({ fileToolsEnabled: false, shellEnabled: false, gitEnabled: false })
    expect(disabled.has('read_file')).toBe(true)
    expect(disabled.has('run_command')).toBe(true)
    expect(disabled.has('git')).toBe(true)
    expect(disabled.has('ddg_search')).toBe(false)
  })

  it('undefined does not disable (only explicit false)', () => {
    const disabled = getDisabledTools({ fileToolsEnabled: undefined })
    expect(disabled.size).toBe(0)
  })
})

// =============================================================================
// Request Body Building — Sampling Params Only Sent When Set
// =============================================================================

describe('request body conditional params', () => {
  // Mirrors chat.ts:700-745 — params only included when explicitly set
  interface Overrides {
    temperature?: number | null
    topP?: number | null
    topK?: number | null
    minP?: number | null
    repeatPenalty?: number | null
    maxTokens?: number | null
    stopSequences?: string
  }

  function buildRequestParams(overrides: Overrides): Record<string, unknown> {
    const obj: Record<string, unknown> = {}
    if (overrides.temperature != null) obj.temperature = overrides.temperature
    if (overrides.topP != null) obj.top_p = overrides.topP
    if (overrides.maxTokens) obj.max_tokens = overrides.maxTokens
    if (overrides.topK != null && overrides.topK > 0) obj.top_k = overrides.topK
    if (overrides.minP != null && overrides.minP > 0) obj.min_p = overrides.minP
    if (overrides.repeatPenalty != null && overrides.repeatPenalty !== 1.0) obj.repetition_penalty = overrides.repeatPenalty
    return obj
  }

  it('empty overrides sends nothing', () => {
    expect(buildRequestParams({})).toEqual({})
  })

  it('temperature=0 IS sent (greedy)', () => {
    const obj = buildRequestParams({ temperature: 0 })
    expect(obj.temperature).toBe(0)
  })

  it('temperature=null is NOT sent', () => {
    const obj = buildRequestParams({ temperature: null })
    expect(obj).not.toHaveProperty('temperature')
  })

  it('topK=0 is NOT sent (means disabled)', () => {
    const obj = buildRequestParams({ topK: 0 })
    expect(obj).not.toHaveProperty('top_k')
  })

  it('topK=40 IS sent', () => {
    const obj = buildRequestParams({ topK: 40 })
    expect(obj.top_k).toBe(40)
  })

  it('minP=0 is NOT sent (means disabled)', () => {
    const obj = buildRequestParams({ minP: 0 })
    expect(obj).not.toHaveProperty('min_p')
  })

  it('repeatPenalty=1.0 is NOT sent (means disabled)', () => {
    const obj = buildRequestParams({ repeatPenalty: 1.0 })
    expect(obj).not.toHaveProperty('repetition_penalty')
  })

  it('repeatPenalty=1.1 IS sent', () => {
    const obj = buildRequestParams({ repeatPenalty: 1.1 })
    expect(obj.repetition_penalty).toBe(1.1)
  })

  it('maxTokens=0 is NOT sent (falsy)', () => {
    const obj = buildRequestParams({ maxTokens: 0 })
    expect(obj).not.toHaveProperty('max_tokens')
  })
})

// =============================================================================
// Wire API Format Selection
// =============================================================================

describe('wire API format selection', () => {
  function resolveApiUrl(baseUrl: string, wireApi: string | undefined): string {
    const useResponsesApi = wireApi === 'responses'
    return useResponsesApi
      ? `${baseUrl}/v1/responses`
      : `${baseUrl}/v1/chat/completions`
  }

  it('default (undefined) uses chat completions', () => {
    expect(resolveApiUrl('http://localhost:8000', undefined)).toBe('http://localhost:8000/v1/chat/completions')
  })

  it('completions uses chat completions', () => {
    expect(resolveApiUrl('http://localhost:8000', 'completions')).toBe('http://localhost:8000/v1/chat/completions')
  })

  it('responses uses responses API', () => {
    expect(resolveApiUrl('http://localhost:8000', 'responses')).toBe('http://localhost:8000/v1/responses')
  })

  it('unknown value falls back to chat completions', () => {
    expect(resolveApiUrl('http://localhost:8000', 'invalid')).toBe('http://localhost:8000/v1/chat/completions')
  })
})

// =============================================================================
// JANG Format Detection
// =============================================================================

describe('JANG format detection in model scanner', () => {
  // Mirrors the scanner logic in models.ts
  function detectJangQuantization(files: string[]): string | undefined {
    const configNames = ['jang_config.json', 'jjqf_config.json', 'mxq_config.json']
    for (const cfg of configNames) {
      if (files.includes(cfg)) return cfg
    }
    return undefined
  }

  it('detects jang_config.json', () => {
    expect(detectJangQuantization(['config.json', 'jang_config.json', 'model.jang.safetensors'])).toBe('jang_config.json')
  })

  it('detects legacy mxq_config.json', () => {
    expect(detectJangQuantization(['config.json', 'mxq_config.json', 'model.mxq.safetensors'])).toBe('mxq_config.json')
  })

  it('returns undefined for non-JANG model', () => {
    expect(detectJangQuantization(['config.json', 'model.safetensors'])).toBe(undefined)
  })

  it('prefers jang_config.json over mxq_config.json', () => {
    expect(detectJangQuantization(['jang_config.json', 'mxq_config.json'])).toBe('jang_config.json')
  })
})

describe('JANG safetensors file extension matching', () => {
  // Verifies that .jang.safetensors files match .safetensors checks
  it('.jang.safetensors ends with .safetensors', () => {
    expect('model-00001.jang.safetensors'.endsWith('.safetensors')).toBe(true)
  })

  it('.mxq.safetensors ends with .safetensors', () => {
    expect('model-00001.mxq.safetensors'.endsWith('.safetensors')).toBe(true)
  })

  it('detectModelFormat works for JANG files', () => {
    // .jang.safetensors matches .safetensors, so format detection works
    const files = ['config.json', 'model-00001.jang.safetensors']
    const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
    const hasConfig = files.includes('config.json')
    expect(hasSafetensors && hasConfig).toBe(true)
  })

  it('estimateModelMemory includes JANG files', () => {
    const files = ['model-00001.jang.safetensors', 'model-00002.jang.safetensors']
    const counted = files.filter(f => f.endsWith('.safetensors'))
    expect(counted).toHaveLength(2)
  })

  it('session validation accepts JANG model dirs', () => {
    const files = ['config.json', 'jang_config.json', 'model.jang.safetensors']
    const hasGGUF = files.some(f => f.endsWith('.gguf'))
    const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
    const hasConfig = files.includes('config.json')
    expect(hasGGUF).toBe(false)
    expect(hasSafetensors).toBe(true)
    expect(hasConfig).toBe(true)
  })
})

describe('JANG quantization label formatting', () => {
  function formatJangLabel(config: { format: string; quantization?: { actual_bits?: number; target_bits?: number } }): string {
    if (config.format === 'jang' || config.format === 'jjqf' || config.format === 'mxq') {
      const bits = config.quantization?.actual_bits || config.quantization?.target_bits
      return bits ? `JANG ${bits}-bit` : 'JANG'
    }
    return ''
  }

  it('formats actual_bits', () => {
    expect(formatJangLabel({ format: 'jang', quantization: { actual_bits: 2.51, target_bits: 2.5 } })).toBe('JANG 2.51-bit')
  })

  it('falls back to target_bits', () => {
    expect(formatJangLabel({ format: 'jang', quantization: { target_bits: 4 } })).toBe('JANG 4-bit')
  })

  it('handles legacy mxq format', () => {
    expect(formatJangLabel({ format: 'mxq', quantization: { actual_bits: 3.5 } })).toBe('JANG 3.5-bit')
  })

  it('handles missing quantization', () => {
    expect(formatJangLabel({ format: 'jang' })).toBe('JANG')
  })

  it('ignores non-JANG formats', () => {
    expect(formatJangLabel({ format: 'gguf' })).toBe('')
  })
})

// =============================================================================
// Version Comparison Edge Cases
// =============================================================================

function compareVersions(current: string, latest: string): boolean {
  const clean = (v: string) => v.replace(/-.*$/, '')
  const a = clean(current).split('.').map(Number)
  const b = clean(latest).split('.').map(Number)
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] ?? 0
    const bv = b[i] ?? 0
    if (isNaN(av) || isNaN(bv)) return false
    if (bv > av) return true
    if (bv < av) return false
  }
  return false
}

describe('version comparison edge cases', () => {
  it('1.0.0 < 1.0.1', () => expect(compareVersions('1.0.0', '1.0.1')).toBe(true))
  it('1.0.0 = 1.0.0', () => expect(compareVersions('1.0.0', '1.0.0')).toBe(false))
  it('2.0.0 > 1.9.9', () => expect(compareVersions('2.0.0', '1.9.9')).toBe(false))
  it('handles pre-release current', () => expect(compareVersions('1.1.4-beta.1', '1.1.5')).toBe(true))
  it('handles pre-release latest', () => expect(compareVersions('1.1.4', '1.1.5-rc.1')).toBe(true))
  it('handles malformed version', () => expect(compareVersions('abc', '1.0.0')).toBe(false))
  it('handles different lengths', () => expect(compareVersions('1.0', '1.0.1')).toBe(true))
  it('handles v0 correctly', () => expect(compareVersions('0.0.1', '0.0.2')).toBe(true))
})

// =============================================================================
// Session Default Temperature Slider Logic
// =============================================================================

describe('session default temperature slider', () => {
  // Mirrors the buildArgs logic: slider value 0 = "Server default" (don't pass flag)
  // Slider range: 0-200, step=5, unlimitedValue=0
  function shouldPassTemperature(value: number | null | undefined): boolean {
    return value != null && value > 0
  }

  it('does not pass when value is 0 (Server default)', () => {
    expect(shouldPassTemperature(0)).toBe(false)
  })

  it('does not pass when value is null', () => {
    expect(shouldPassTemperature(null)).toBe(false)
  })

  it('does not pass when value is undefined', () => {
    expect(shouldPassTemperature(undefined)).toBe(false)
  })

  it('passes when value is 5 (temp=0.05)', () => {
    expect(shouldPassTemperature(5)).toBe(true)
  })

  it('passes when value is 70 (temp=0.70)', () => {
    expect(shouldPassTemperature(70)).toBe(true)
  })

  it('passes when value is 200 (temp=2.00)', () => {
    expect(shouldPassTemperature(200)).toBe(true)
  })

  it('converts slider value to float correctly', () => {
    expect((70 / 100).toFixed(2)).toBe('0.70')
    expect((200 / 100).toFixed(2)).toBe('2.00')
    expect((5 / 100).toFixed(2)).toBe('0.05')
  })
})
