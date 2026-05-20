import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  buildMcpPolicyArgs,
  normalizeMcpPolicyList,
  redactMcpConfigForDisplay,
} from '../src/shared/mcpPolicy'
import { validateMcpConfigText } from '../src/shared/mcpConfigValidation'

describe('MCP policy shared helpers', () => {
  it('normalizes comma/newline policy lists without empty values', () => {
    expect(normalizeMcpPolicyList('fs, web\nfs__read_file\n')).toEqual([
      'fs',
      'web',
      'fs__read_file',
    ])
  })

  it('builds CLI args for session-level MCP policy overlay', () => {
    expect(buildMcpPolicyArgs({
      mcpEnabledServers: 'fs,web',
      mcpDisabledServers: 'browser',
      mcpEnabledTools: 'fs__read_file',
      mcpDisabledTools: 'web__dangerous',
    })).toEqual([
      '--mcp-enabled-servers', 'fs,web',
      '--mcp-disabled-servers', 'browser',
      '--mcp-enabled-tools', 'fs__read_file',
      '--mcp-disabled-tools', 'web__dangerous',
    ])
  })

  it('redacts MCP env and header values for renderer display', () => {
    const redacted = redactMcpConfigForDisplay({
      mcpServers: {
        github: {
          transport: 'http',
          url: 'https://example.test/mcp',
          headers: { Authorization: 'Bearer secret-token' },
          env: { GITHUB_TOKEN: 'ghp_secret' },
        },
      },
    })

    expect(redacted.mcpServers.github.headers.Authorization).toBe('<redacted>')
    expect(redacted.mcpServers.github.env.GITHUB_TOKEN).toBe('<redacted>')
    expect(JSON.stringify(redacted)).not.toContain('secret-token')
    expect(JSON.stringify(redacted)).not.toContain('ghp_secret')
  })

  it('validates JSONC MCP config and returns only redacted server summaries', () => {
    const validated = validateMcpConfigText(`{
      // local stdio server
      "mcpServers": {
        "github": {
          "transport": "http",
          "url": "https://example.test/mcp?token=real-token-value",
          "headers": { "Authorization": "Bearer real-secret-value" },
          "env": { "GITHUB_TOKEN": "ghp_real_secret_value" }
        }
      }
    }`)

    expect(validated.serverCount).toBe(1)
    expect(validated.servers[0].name).toBe('github')
    expect(validated.servers[0].url_redacted).toContain('token=<redacted>')
    expect(validated.servers[0].header_keys).toEqual(['Authorization'])
    expect(validated.servers[0].env_keys).toEqual(['GITHUB_TOKEN'])
    expect(JSON.stringify(validated)).not.toContain('real-secret-value')
    expect(JSON.stringify(validated)).not.toContain('ghp_real_secret_value')
    expect(JSON.stringify(validated)).not.toContain('real-token-value')
  })

  it('wires MCP policy flags through session launch and command preview', () => {
    const sessionsSource = readFileSync('src/main/sessions.ts', 'utf8')
    const settingsSource = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
    const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')

    for (const flag of ['--mcp-enabled-servers', '--mcp-disabled-servers', '--mcp-enabled-tools', '--mcp-disabled-tools']) {
      expect(sessionsSource).toContain(flag)
      expect(settingsSource).toContain(flag)
    }

    for (const key of ['mcpEnabledServers', 'mcpDisabledServers', 'mcpEnabledTools', 'mcpDisabledTools']) {
      expect(sessionsSource).toContain(key)
      expect(formSource).toContain(key)
    }
    expect(settingsSource).toContain('buildMcpPolicyArgs(config)')
  })

  it('exposes redacted live MCP status through session IPC', () => {
    const ipcSource = readFileSync('src/main/ipc/sessions.ts', 'utf8')
    const preloadSource = readFileSync('src/preload/index.ts', 'utf8')
    const envSource = readFileSync('src/env.d.ts', 'utf8')
    const validatorSource = readFileSync('src/shared/mcpConfigValidation.ts', 'utf8')

    expect(ipcSource).toContain("sessions:mcpStatus")
    expect(ipcSource).toContain("/v1/mcp/tools")
    expect(ipcSource).toContain("/v1/mcp/servers")
    expect(ipcSource).toContain("Authorization")
    expect(preloadSource).toContain("mcpStatus")
    expect(envSource).toContain("mcpStatus")
    expect(ipcSource).toContain("sessions:browseMcpConfig")
    expect(ipcSource).toContain("sessions:validateMcpConfig")
    expect(ipcSource).toContain("validateMcpConfigText")
    expect(validatorSource).toContain("redactMcpConfigForDisplay")
    expect(validatorSource).toContain("MCP config must contain")
    expect(preloadSource).toContain("browseMcpConfig")
    expect(preloadSource).toContain("validateMcpConfig")
    expect(envSource).toContain("browseMcpConfig")
    expect(envSource).toContain("validateMcpConfig")
  })

  it('renders live MCP servers and tools in the session config form', () => {
    const formSource = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')

    expect(formSource).toContain('window.api.sessions.mcpStatus')
    expect(formSource).toContain('window.api.sessions.browseMcpConfig')
    expect(formSource).toContain('window.api.sessions.validateMcpConfig')
    expect(formSource).toContain('mcpValidation')
    expect(formSource).toContain('mcpStatus?.servers')
    expect(formSource).toContain('mcpStatus?.tools')
    expect(formSource).toContain('toggleMcpServer')
    expect(formSource).toContain('toggleMcpTool')
  })

  it('gates ambiguous MCP gateway requests on explicit model routing', () => {
    const gatewaySource = readFileSync('src/main/api-gateway.ts', 'utf8')

    expect(gatewaySource).toContain('isMcpGatewayRoute')
    expect(gatewaySource).toContain('MCP gateway requests require a model')
    expect(gatewaySource).toContain('model_required')
  })
})
