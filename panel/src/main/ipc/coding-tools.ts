// MLX Studio — Coding Tool Integration IPC
// Non-destructive config management for Claude Code, Codex CLI, OpenCode
import { ipcMain } from 'electron'
import { execFileSync } from 'child_process'
import { homedir } from 'os'
import { join } from 'path'
import { existsSync, readFileSync, writeFileSync, copyFileSync, mkdirSync } from 'fs'

const MLXSTUDIO_TAG = '_mlxstudio'  // Tag to identify our entries

interface ToolConfig {
  detect: () => boolean
  installCmd: string
  installArgs: string[]
  configPath: string
  getEntries: () => Array<{ label: string; baseUrl: string }>
  addEntry: (baseUrl: string, modelName: string, port: number | null) => void
  removeEntry: (label: string) => void
}

function safeReadJSON(path: string): any {
  try {
    if (!existsSync(path)) return null
    return JSON.parse(readFileSync(path, 'utf-8'))
  } catch { return null }
}

function safeReadTOML(path: string): string | null {
  try {
    if (!existsSync(path)) return null
    return readFileSync(path, 'utf-8')
  } catch { return null }
}

function safeWriteTOML(path: string, content: string): void {
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, content, { encoding: 'utf-8', mode: 0o600 })
}

function safeWriteJSON(path: string, data: any): void {
  // Backup before writing
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, JSON.stringify(data, null, 2) + '\n', { encoding: 'utf-8', mode: 0o600 })
}

function commandExists(cmd: string): boolean {
  // Check common install locations (Electron strips user PATH)
  const paths = [
    join(homedir(), '.local', 'bin', cmd),
    join(homedir(), '.npm-global', 'bin', cmd),
    '/usr/local/bin/' + cmd,
    '/usr/bin/' + cmd,
    '/opt/homebrew/bin/' + cmd,
    join(homedir(), '.cargo', 'bin', cmd),
    join(homedir(), '.bun', 'bin', cmd),
    join(homedir(), '.volta', 'bin', cmd),
  ]
  if (paths.some(p => existsSync(p))) return true
  // Fallback: try which with augmented PATH (includes nvm, pnpm, yarn)
  try {
    const extraPaths = [
      `${homedir()}/.local/bin`,
      '/opt/homebrew/bin',
      '/usr/local/bin',
      `${homedir()}/.cargo/bin`,
      `${homedir()}/.bun/bin`,
      `${homedir()}/.volta/bin`,
      `${homedir()}/.yarn/bin`,
    ].join(':')
    const env = { ...process.env, PATH: `${process.env.PATH}:${extraPaths}` }
    execFileSync('which', [cmd], { stdio: 'pipe', env })
    return true
  } catch { return false }
}

// ═══ Claude Code ═══
// Config: ~/.claude/settings.json — env vars (ANTHROPIC_BASE_URL, ANTHROPIC_MODEL)
// Claude Code requires Anthropic Messages API format (/v1/messages) — vMLX supports this
const CLAUDE_SETTINGS = join(homedir(), '.claude', 'settings.json')
const claudeCode: ToolConfig = {
  detect: () => commandExists('claude'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@anthropic-ai/claude-code'],
  configPath: CLAUDE_SETTINGS,
  getEntries: () => {
    const cfg = safeReadJSON(CLAUDE_SETTINGS)
    if (!cfg?.env?.ANTHROPIC_BASE_URL || !cfg?.env?.[MLXSTUDIO_TAG]) return []
    return [{ label: cfg.env.ANTHROPIC_MODEL || 'default', baseUrl: cfg.env.ANTHROPIC_BASE_URL }]
  },
  addEntry: (baseUrl, modelName) => {
    const cfg = safeReadJSON(CLAUDE_SETTINGS) || {}
    if (!cfg.env) cfg.env = {}
    // Claude Code appends /v1/messages itself — base URL should NOT include /v1
    cfg.env.ANTHROPIC_BASE_URL = baseUrl
    cfg.env.ANTHROPIC_MODEL = modelName
    cfg.env.ANTHROPIC_API_KEY = 'mlxstudio'  // Required non-empty value
    cfg.env[MLXSTUDIO_TAG] = 'true'
    safeWriteJSON(CLAUDE_SETTINGS, cfg)
  },
  removeEntry: () => {
    const cfg = safeReadJSON(CLAUDE_SETTINGS)
    if (!cfg?.env) return
    delete cfg.env.ANTHROPIC_BASE_URL
    delete cfg.env.ANTHROPIC_MODEL
    delete cfg.env.ANTHROPIC_API_KEY
    delete cfg.env[MLXSTUDIO_TAG]
    // Clean up empty env object
    if (Object.keys(cfg.env).length === 0) delete cfg.env
    safeWriteJSON(CLAUDE_SETTINGS, cfg)
  },
}

// ═══ Codex CLI ═══
// Config: ~/.codex/config.toml — TOML format with [model_providers.NAME] sections
const CODEX_TOML = join(homedir(), '.codex', 'config.toml')
const codexCli: ToolConfig = {
  detect: () => commandExists('codex'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@openai/codex'],
  configPath: CODEX_TOML,
  getEntries: () => {
    const toml = safeReadTOML(CODEX_TOML)
    if (!toml) return []
    // Parse TOML: find [model_providers.MLXSTUDIO_*] sections with _mlxstudio marker
    const entries: Array<{ label: string; baseUrl: string }> = []
    const sectionRegex = /\[model_providers\.([^\]]+)\]/g
    let match
    while ((match = sectionRegex.exec(toml)) !== null) {
      const name = match[1]
      if (!name.startsWith('MLXSTUDIO_')) continue
      // Extract base_url from this section
      const sectionStart = match.index + match[0].length
      const nextSection = toml.indexOf('\n[', sectionStart)
      const sectionBody = nextSection >= 0 ? toml.slice(sectionStart, nextSection) : toml.slice(sectionStart)
      const urlMatch = sectionBody.match(/base_url\s*=\s*"([^"]+)"/)
      entries.push({ label: name, baseUrl: urlMatch ? urlMatch[1] : '' })
    }
    return entries
  },
  addEntry: (baseUrl, modelName) => {
    let toml = safeReadTOML(CODEX_TOML) || ''
    const providerKey = `MLXSTUDIO_${modelName.replace(/[^a-zA-Z0-9_]/g, '_').toUpperCase()}`
    // Remove existing section if present
    const sectionPattern = new RegExp(`\\[model_providers\\.${providerKey}\\][\\s\\S]*?(?=\\n\\[|$)`, 'g')
    toml = toml.replace(sectionPattern, '').replace(/\n{3,}/g, '\n\n').trim()
    // Append new section
    const section = `\n\n[model_providers.${providerKey}]\nname = "MLX Studio (${modelName})"\nbase_url = "${baseUrl}/v1"\nwire_api = "responses"\nmax_context = 32768\n`
    toml += section
    safeWriteTOML(CODEX_TOML, toml)
  },
  removeEntry: (label) => {
    let toml = safeReadTOML(CODEX_TOML)
    if (!toml) return
    // Remove the [model_providers.LABEL] section
    const sectionPattern = new RegExp(`\\[model_providers\\.${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\][\\s\\S]*?(?=\\n\\[|$)`, 'g')
    toml = toml.replace(sectionPattern, '').replace(/\n{3,}/g, '\n\n').trim()
    safeWriteTOML(CODEX_TOML, toml + '\n')
  },
}

// ═══ OpenCode ═══
// Config: ~/.config/opencode/opencode.json — we add provider entries tagged with MLXSTUDIO_TAG
const openCode: ToolConfig = {
  detect: () => commandExists('opencode'),
  installCmd: 'npm',
  installArgs: ['install', '-g', 'opencode'],
  configPath: join(homedir(), '.config', 'opencode', 'opencode.json'),
  getEntries: () => {
    const cfg = safeReadJSON(join(homedir(), '.config', 'opencode', 'opencode.json'))
    if (!cfg?.provider) return []
    return Object.entries(cfg.provider)
      .filter(([_, v]: any) => v?.[MLXSTUDIO_TAG])
      .map(([k, v]: any) => ({ label: k, baseUrl: (v as any)?.options?.baseURL || '' }))
  },
  addEntry: (baseUrl, modelName) => {
    const path = join(homedir(), '.config', 'opencode', 'opencode.json')
    const cfg = safeReadJSON(path) || { '$schema': 'https://opencode.ai/config.json' }
    if (!cfg.provider) cfg.provider = {}
    const key = `mlxstudio-${modelName.replace(/[^a-zA-Z0-9-]/g, '-')}`
    cfg.provider[key] = {
      npm: '@ai-sdk/openai-compatible',
      name: `MLX Studio (${modelName})`,
      options: { baseURL: `${baseUrl}/v1` },
      models: {
        [modelName]: {
          name: modelName,
          limit: { context: 32768, output: 4096 },
          modalities: { input: ['text'], output: ['text'] },
        },
      },
      [MLXSTUDIO_TAG]: true,
    }
    safeWriteJSON(path, cfg)
  },
  removeEntry: (label) => {
    const path = join(homedir(), '.config', 'opencode', 'opencode.json')
    const cfg = safeReadJSON(path)
    if (!cfg?.provider?.[label]) return
    delete cfg.provider[label]
    safeWriteJSON(path, cfg)
  },
}

const TOOLS: Record<string, ToolConfig> = {
  'claude-code': claudeCode,
  'codex': codexCli,
  'opencode': openCode,
}

let registered = false

export function registerCodingToolHandlers(): void {
  if (registered) return
  registered = true

  ipcMain.handle('tools:getCodingToolStatus', async () => {
    const result: Record<string, any> = {}
    for (const [id, tool] of Object.entries(TOOLS)) {
      const installed = tool.detect()
      const entries = installed ? tool.getEntries() : []
      result[id] = {
        installed,
        configured: entries.length > 0,
        configPath: tool.configPath,
        entries,
      }
    }
    return result
  })

  ipcMain.handle('tools:installCodingTool', async (_, toolId: string) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    try {
      execFileSync(tool.installCmd, tool.installArgs, { stdio: 'pipe', timeout: 120000 })
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })

  ipcMain.handle('tools:addCodingToolConfig', async (_, toolId: string, baseUrl: string, modelName: string, port: number | null) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    if (!tool.detect()) return { success: false, error: 'Tool not installed' }
    try {
      tool.addEntry(baseUrl, modelName, port)
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })

  ipcMain.handle('tools:removeCodingToolConfig', async (_, toolId: string, label: string) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    try {
      tool.removeEntry(label)
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })
}
