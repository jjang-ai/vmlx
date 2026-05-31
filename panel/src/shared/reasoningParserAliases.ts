export const REASONING_PARSERS_FOR_CLI = new Set([
  'qwen3',
  'deepseek_r1',
  'minimax_m2',
  'openai_gptoss',
  'mistral',
  'gemma4',
  'think_xml',
])

export function canonicalizeReasoningParserForCli(parser?: string): string | undefined {
  if (!parser || parser === 'auto' || parser === '') return undefined
  if (parser === 'none') return 'none'
  if (parser === 'minimax' || parser === 'minimax_m2_5') return 'minimax_m2'
  return REASONING_PARSERS_FOR_CLI.has(parser) ? parser : undefined
}
