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

function safeWriteJSON(path: string, data: any): void {
  // Backup before writing
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, JSON.stringify(data, null, 2) + '\n', 'utf-8')
}

function commandExists(cmd: string): boolean {
  // Check common install locations (Electron strips user PATH)
  const paths = [
    join(homedir(), '.local', 'bin', cmd),
    join(homedir(), '.npm-global', 'bin', cmd),
    '/usr/local/bin/' + cmd,
    '/opt/homebrew/bin/' + cmd,
    join(homedir(), '.nvm', 'versions', 'node'),  // nvm — just check the dir
  ]
  if (paths.some(p => existsSync(p))) return true
  // Fallback: try which with full PATH
  try {
    const env = { ...process.env, PATH: `${process.env.PATH}:${homedir()}/.local/bin:/opt/homebrew/bin:/usr/local/bin` }
    execFileSync('which', [cmd], { stdio: 'pipe', env })
    return true
  } catch { return false }
}

// ═══ Claude Code ═══
// Config: ~/.claude.json — we add/remove "apiUrl" and "apiModelId" fields
const claudeCode: ToolConfig = {
  detect: () => commandExists('claude'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@anthropic-ai/claude-code'],
  configPath: join(homedir(), '.claude.json'),
  getEntries: () => {
    const cfg = safeReadJSON(join(homedir(), '.claude.json'))
    if (!cfg) return []
    const entries: Array<{ label: string; baseUrl: string }> = []
    if (cfg.apiUrl && cfg[MLXSTUDIO_TAG]) {
      entries.push({ label: cfg.apiModelId || 'default', baseUrl: cfg.apiUrl })
    }
    return entries
  },
  addEntry: (baseUrl, modelName) => {
    const path = join(homedir(), '.claude.json')
    const cfg = safeReadJSON(path) || {}
    cfg.apiUrl = baseUrl
    cfg.apiModelId = modelName
    cfg[MLXSTUDIO_TAG] = true
    safeWriteJSON(path, cfg)
  },
  removeEntry: () => {
    const path = join(homedir(), '.claude.json')
    const cfg = safeReadJSON(path)
    if (!cfg) return
    delete cfg.apiUrl
    delete cfg.apiModelId
    delete cfg[MLXSTUDIO_TAG]
    safeWriteJSON(path, cfg)
  },
}

// ═══ Codex CLI ═══
// Config: ~/.codex/config.json — we add provider entries tagged with MLXSTUDIO_TAG
const codexCli: ToolConfig = {
  detect: () => commandExists('codex'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@openai/codex'],
  configPath: join(homedir(), '.codex', 'config.json'),
  getEntries: () => {
    const cfg = safeReadJSON(join(homedir(), '.codex', 'config.json'))
    if (!cfg?.providers) return []
    return Object.entries(cfg.providers)
      .filter(([_, v]: any) => v?.[MLXSTUDIO_TAG])
      .map(([k, v]: any) => ({ label: k, baseUrl: v?.baseUrl || '' }))
  },
  addEntry: (baseUrl, modelName) => {
    const path = join(homedir(), '.codex', 'config.json')
    const cfg = safeReadJSON(path) || {}
    if (!cfg.providers) cfg.providers = {}
    const key = `mlxstudio-${modelName.replace(/[^a-zA-Z0-9-]/g, '-')}`
    cfg.providers[key] = {
      name: `MLX Studio (${modelName})`,
      baseUrl: `${baseUrl}/v1`,
      model: modelName,
      [MLXSTUDIO_TAG]: true,
    }
    safeWriteJSON(path, cfg)
  },
  removeEntry: (label) => {
    const path = join(homedir(), '.codex', 'config.json')
    const cfg = safeReadJSON(path)
    if (!cfg?.providers?.[label]) return
    delete cfg.providers[label]
    safeWriteJSON(path, cfg)
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
