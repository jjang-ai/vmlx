const TOOL_PARSER_CANONICAL_ALIASES: Record<string, string> = {
  deepseek_v4: 'dsml',
  hy_v3: 'hunyuan',
}

export function canonicalizeToolParserId(value: string | undefined): string | undefined {
  if (value === undefined) return undefined
  return TOOL_PARSER_CANONICAL_ALIASES[value] || value
}
