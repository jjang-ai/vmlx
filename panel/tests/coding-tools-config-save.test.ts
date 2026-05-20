import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const electronMock = vi.hoisted(() => ({
  handlers: new Map<string, (...args: any[]) => Promise<any>>(),
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: vi.fn((channel: string, handler: (...args: any[]) => Promise<any>) => {
      electronMock.handlers.set(channel, handler)
    }),
  },
}))

describe('coding tool config saving', () => {
  let home: string
  const oldHome = process.env.HOME
  const oldPath = process.env.PATH
  const oldFetch = global.fetch

  beforeEach(async () => {
    home = mkTempHome()
    process.env.HOME = home
    process.env.PATH = `${join(home, '.local', 'bin')}:${oldPath || ''}`
    global.fetch = vi.fn(async (url: any) => {
      const target = String(url)
      if (target.endsWith('/health')) {
        return {
          ok: true,
          json: async () => ({ max_prompt_tokens: 24576 }),
        } as any
      }
      if (target.includes('/v1/models/') && target.endsWith('/capabilities')) {
        return {
          ok: true,
          json: async () => ({
            sampling_defaults: { max_new_tokens: 6144 },
          }),
        } as any
      }
      return { ok: false, json: async () => ({}) } as any
    }) as any
    for (const cmd of ['claude', 'codex', 'opencode', 'openclaw']) {
      const bin = join(home, '.local', 'bin', cmd)
      mkdirSync(join(home, '.local', 'bin'), { recursive: true })
      writeFileSync(bin, '#!/bin/sh\nexit 0\n', { mode: 0o755 })
    }
    electronMock.handlers.clear()
    vi.resetModules()
    const mod = await import('../src/main/ipc/coding-tools')
    mod.registerCodingToolHandlers()
  })

  afterEach(() => {
    process.env.HOME = oldHome
    process.env.PATH = oldPath
    global.fetch = oldFetch
    rmSync(home, { recursive: true, force: true })
  })

  it('writes Claude, Codex, OpenCode, and OpenClaw configs with backups and model-derived names', async () => {
    const add = electronMock.handlers.get('tools:addCodingToolConfig')
    const snippets = electronMock.handlers.get('tools:getConfigSnippets')
    expect(add).toBeTruthy()
    expect(snippets).toBeTruthy()

    const baseUrl = 'http://127.0.0.1:8080'
    const model = 'JANGQ/Qwen3.6-27B-MXFP8-MTP'

    mkdirSync(join(home, '.codex'), { recursive: true })
    writeFileSync(join(home, '.codex', 'config.toml'), 'profile = "existing"\n')

    for (const tool of ['claude-code', 'codex', 'opencode', 'openclaw']) {
      const result = await add!({}, tool, baseUrl, model, 8080)
      expect(result).toEqual({ success: true })
    }

    const claude = JSON.parse(readFileSync(join(home, '.claude', 'settings.json'), 'utf8'))
    expect(claude.env.ANTHROPIC_BASE_URL).toBe(baseUrl)
    expect(claude.env.ANTHROPIC_MODEL).toBe(model)
    expect(claude.env._mlxstudio).toBe('true')
    expect(statMode(join(home, '.claude', 'settings.json'))).toBe(0o600)

    const codex = readFileSync(join(home, '.codex', 'config.toml'), 'utf8')
    expect(codex).toContain('[model_providers.MLXSTUDIO_JANGQ_QWEN3_6_27B_MXFP8_MTP]')
    expect(codex).toContain(`base_url = "${baseUrl}/v1"`)
    expect(codex).toContain('wire_api = "responses"')
    expect(codex).toContain('max_context = 24576')
    expect(existsSync(join(home, '.codex', 'config.toml.bak'))).toBe(true)
    expect(statMode(join(home, '.codex', 'config.toml'))).toBe(0o600)

    const opencode = JSON.parse(readFileSync(join(home, '.config', 'opencode', 'opencode.json'), 'utf8'))
    const opencodeKey = 'mlxstudio-JANGQ-Qwen3-6-27B-MXFP8-MTP'
    expect(opencode.provider[opencodeKey].options.baseURL).toBe(`${baseUrl}/v1`)
    expect(opencode.provider[opencodeKey].models[model].name).toBe(model)
    expect(opencode.provider[opencodeKey].models[model].limit).toEqual({ context: 24576, output: 6144 })
    expect(opencode.provider[opencodeKey]._mlxstudio).toBe(true)

    const openclaw = JSON.parse(readFileSync(join(home, '.openclaw', 'openclaw.json'), 'utf8'))
    expect(openclaw.models.providers.mlxstudio.baseUrl).toBe(`${baseUrl}/v1`)
    expect(openclaw.models.providers.mlxstudio.api).toBe('openai-completions')
    expect(openclaw.models.providers.mlxstudio.models[0].id).toBe(model)
    expect(openclaw.models.providers.mlxstudio.models[0].contextWindow).toBe(24576)
    expect(openclaw.models.providers.mlxstudio.models[0].maxTokens).toBe(6144)
    expect(openclaw.agents.defaults.models[`mlxstudio/${model}`].alias).toBe(model)

    const snippetResult = await snippets!({}, baseUrl, model, 8080)
    expect(snippetResult.codex.snippet).toContain('wire_api = "responses"')
    expect(snippetResult.codex.snippet).toContain('max_context = 24576')
    expect(snippetResult['claude-code'].snippet).toContain(model)
    expect(snippetResult.opencode.snippet).toContain('"output": 6144')
    expect(snippetResult.openclaw.snippet).toContain(`mlxstudio/${model}`)
    expect(snippetResult.openclaw.snippet).toContain('"maxTokens": 6144')
  })
})

function mkTempHome(): string {
  const dir = join(tmpdir(), `vmlx-coding-tools-${Date.now()}-${Math.random().toString(16).slice(2)}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function statMode(path: string): number {
  return statSync(path).mode & 0o777
}
