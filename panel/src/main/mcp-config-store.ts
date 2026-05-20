import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { basename, extname, join } from 'node:path'
import { validateMcpConfigText } from '../shared/mcpConfigValidation'

export interface ManagedMcpImportOptions {
  storeDir: string
}

export interface ManagedMcpImportResult {
  success: true
  managed: true
  sourcePath: string
  importedPath: string
  serverCount?: number
  servers: any[]
  redactedConfig?: any
}

export function defaultManagedMcpConfigDir(userDataPath: string): string {
  return join(userDataPath, 'mcp-configs')
}

function safeBaseName(filePath: string): string {
  const rawExt = extname(filePath)
  const rawBase = basename(filePath, rawExt) || 'mcp'
  const base = rawBase
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'mcp'
  const ext = rawExt && /^[.][A-Za-z0-9]+$/.test(rawExt) ? rawExt : '.json'
  return `${base}${ext}`
}

export function importMcpConfigFile(filePath: string, options: ManagedMcpImportOptions): ManagedMcpImportResult {
  const sourcePath = String(filePath || '').trim()
  if (!sourcePath) throw new Error('MCP config path is empty')
  if (!options?.storeDir?.trim()) throw new Error('Managed MCP config store path is empty')
  if (!existsSync(sourcePath)) throw new Error(`MCP config file does not exist: ${sourcePath}`)
  if (!statSync(sourcePath).isFile()) throw new Error(`MCP config path is not a file: ${sourcePath}`)

  const raw = readFileSync(sourcePath, 'utf8')
  const validated = validateMcpConfigText(raw, sourcePath)
  mkdirSync(options.storeDir, { recursive: true, mode: 0o700 })

  const safeName = safeBaseName(sourcePath)
  const ext = extname(safeName)
  const base = basename(safeName, ext)
  const hash = createHash('sha256').update(raw).digest('hex').slice(0, 12)
  const importedPath = join(options.storeDir, `${base}-${hash}${ext || '.json'}`)

  writeFileSync(importedPath, raw, { mode: 0o600 })
  return {
    ...validated,
    success: true,
    managed: true,
    sourcePath,
    importedPath,
  }
}
