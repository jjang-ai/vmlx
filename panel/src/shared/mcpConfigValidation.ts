import { redactMcpConfigForDisplay, redactUrlSecrets } from './mcpPolicy'

export function stripJsonComments(input: string): string {
  let out = ''
  let inString = false
  let quote = ''
  let escaped = false
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    const next = input[i + 1]
    if (inString) {
      out += ch
      if (escaped) escaped = false
      else if (ch === '\\') escaped = true
      else if (ch === quote) inString = false
      continue
    }
    if (ch === '"' || ch === "'") {
      inString = true
      quote = ch
      out += ch
      continue
    }
    if (ch === '/' && next === '/') {
      while (i < input.length && input[i] !== '\n') i++
      out += '\n'
      continue
    }
    if (ch === '/' && next === '*') {
      i += 2
      while (i < input.length && !(input[i] === '*' && input[i + 1] === '/')) i++
      i++
      continue
    }
    out += ch
  }
  return out
}

export function validateMcpConfigText(raw: string, filePath = 'mcp.json'): any {
  if (/\.(ya?ml)$/i.test(filePath)) {
    throw new Error('YAML MCP validation is not available in the panel yet; use JSON/JSONC or validate through the engine at session start')
  }
  const parsed = JSON.parse(stripJsonComments(raw))
  const servers = parsed?.mcpServers || parsed?.servers
  if (!servers || typeof servers !== 'object' || Array.isArray(servers)) {
    throw new Error('MCP config must contain an object named "mcpServers" or "servers"')
  }
  const serverEntries = Object.entries(servers as Record<string, any>).map(([name, cfg]) => {
    const config = cfg && typeof cfg === 'object' ? cfg : {}
    return {
      name,
      transport: String(config.transport || (config.url ? 'http' : 'stdio')),
      enabled: config.enabled !== false,
      command_redacted: config.command ? String(config.command) : null,
      url_redacted: config.url ? redactUrlSecrets(String(config.url)) : null,
      env_keys: config.env && typeof config.env === 'object' ? Object.keys(config.env).sort() : [],
      header_keys: config.headers && typeof config.headers === 'object' ? Object.keys(config.headers).sort() : [],
      error: !config.command && !config.url ? 'Missing command or url' : null,
    }
  })
  return {
    success: true,
    serverCount: serverEntries.length,
    servers: serverEntries,
    redactedConfig: redactMcpConfigForDisplay(parsed),
  }
}
