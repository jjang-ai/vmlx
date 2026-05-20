export interface McpPolicyConfig {
  mcpEnabledServers?: string
  mcpDisabledServers?: string
  mcpEnabledTools?: string
  mcpDisabledTools?: string
}

export function normalizeMcpPolicyList(value?: string | string[] | null): string[] {
  if (value == null) return []
  const parts = Array.isArray(value)
    ? value
    : value.split(/[\n,]/g)
  return parts.map(v => String(v).trim()).filter(Boolean)
}

function joined(value?: string | string[] | null): string | undefined {
  const normalized = normalizeMcpPolicyList(value)
  return normalized.length ? normalized.join(',') : undefined
}

export function redactUrlSecrets(value: string): string {
  return value.replace(/([?&][^=&#]*(?:key|token|secret|password)[^=&#]*=)[^&#]*/gi, '$1<redacted>')
}

export function buildMcpPolicyArgs(config: McpPolicyConfig): string[] {
  const args: string[] = []
  const enabledServers = joined(config.mcpEnabledServers)
  const disabledServers = joined(config.mcpDisabledServers)
  const enabledTools = joined(config.mcpEnabledTools)
  const disabledTools = joined(config.mcpDisabledTools)
  if (enabledServers) args.push('--mcp-enabled-servers', enabledServers)
  if (disabledServers) args.push('--mcp-disabled-servers', disabledServers)
  if (enabledTools) args.push('--mcp-enabled-tools', enabledTools)
  if (disabledTools) args.push('--mcp-disabled-tools', disabledTools)
  return args
}

export function redactMcpConfigForDisplay<T = any>(config: T): T {
  const clone = JSON.parse(JSON.stringify(config ?? {}))
  const sections = [clone?.mcpServers, clone?.servers]
  for (const servers of sections) {
    if (!servers || typeof servers !== 'object') continue
    for (const server of Object.values(servers) as any[]) {
      if (!server || typeof server !== 'object') continue
      if (server.env && typeof server.env === 'object') {
        for (const key of Object.keys(server.env)) server.env[key] = '<redacted>'
      }
      if (server.headers && typeof server.headers === 'object') {
        for (const key of Object.keys(server.headers)) server.headers[key] = '<redacted>'
      }
      if (typeof server.url === 'string') {
        server.url = redactUrlSecrets(server.url)
      }
    }
  }
  return clone
}
