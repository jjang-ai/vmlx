import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { importMcpConfigFile } from '../src/main/mcp-config-store'

const roots: string[] = []

function tempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), 'vmlx-mcp-import-'))
  roots.push(root)
  return root
}

afterEach(() => {
  while (roots.length) {
    const root = roots.pop()
    if (root) rmSync(root, { recursive: true, force: true })
  }
})

describe('managed MCP config import store', () => {
  it('copies a validated mcp.json/jsonc into a managed store without leaking secrets in metadata', () => {
    const root = tempRoot()
    const source = join(root, 'work mcp.jsonc')
    const storeDir = join(root, 'managed')
    writeFileSync(source, `{
      // user-owned file should not be modified
      "mcpServers": {
        "github_remote": {
          "transport": "http",
          "url": "https://example.test/mcp?api_key=real-api-key-value&safe=ok",
          "headers": { "Authorization": "Bearer real-secret-value" },
          "env": { "GITHUB_TOKEN": "ghp_real_secret_value" }
        }
      }
    }`)

    const imported = importMcpConfigFile(source, { storeDir })

    expect(imported.success).toBe(true)
    expect(imported.importedPath).toContain(storeDir)
    expect(imported.importedPath).toMatch(/work-mcp-[a-f0-9]{12}\.jsonc$/)
    expect(existsSync(imported.importedPath)).toBe(true)
    expect(readFileSync(imported.importedPath, 'utf8')).toContain('real-secret-value')
    expect(readFileSync(source, 'utf8')).toContain('user-owned file should not be modified')
    expect(imported.servers[0].name).toBe('github_remote')
    expect(imported.servers[0].url_redacted).toContain('api_key=<redacted>')
    expect(JSON.stringify(imported.redactedConfig)).not.toContain('real-secret-value')
    expect(JSON.stringify(imported.redactedConfig)).not.toContain('ghp_real_secret_value')
    expect(JSON.stringify(imported.redactedConfig)).not.toContain('real-api-key-value')
  })
})
