export const REASONING_PARSERS_FOR_CLI = new Set([
  'qwen3',
  'deepseek_r1',
  'minimax_m2',
  'minimax_m3',
  'openai_gptoss',
  'mistral',
  'gemma4',
  'think_xml',
  // The engine accepts `muse_glimmer` (and the `muse` alias). Without it here
  // canonicalizeReasoningParserForCli returned undefined, so selecting it in
  // the session form emitted NO --reasoning-parser flag at all and the choice
  // silently fell back to Auto.
  'muse_glimmer',
  'inkling',
  // dots3_note plain <think> rail; the engine registers 'dots3' on the qwen3
  // think-tag contract.
  'dots3',
  // ERNIE 5.5 Flash <ernie:think>...</ernie:think> reasoning rail.
  'ernie',
  // Hy4-flash <think:opensource>...</think:opensource> reasoning rail.
  'hy4',
])

export function canonicalizeReasoningParserForCli(parser?: string): string | undefined {
  if (!parser || parser === 'auto' || parser === '') return undefined
  if (parser === 'none') return 'none'
  if (parser === 'minimax' || parser === 'minimax_m2_5') return 'minimax_m2'
  if (parser === 'ernie5_5' || parser === 'ernie_5_5' || parser === 'ernie5_5_moe') return 'ernie'
  if (parser === 'hy4_flash' || parser === 'hy_v4_flash') return 'hy4'
  // Poolside/Laguna publishes `poolside_v1` in generation_config.json, while
  // the engine registers it as an exact alias of deepseek_r1. Canonicalize at
  // the panel/argv boundary so the vendor spelling cannot be silently dropped
  // to Auto or replaced by the incompatible think_xml parser.
  if (parser === 'poolside_v1') return 'deepseek_r1'
  // GLM-5.3-Flash bundles stamp glm_think_block — the same <think> rail
  // contract; the engine registers the literal name as a deepseek_r1 alias.
  // Canonicalize so released engines without the alias still resolve it.
  if (parser === 'glm_think_block') return 'deepseek_r1'
  // The engine registers `muse` as an alias of `muse_glimmer`; canonicalize so
  // either spelling survives the argv boundary instead of dropping to Auto.
  if (parser === 'muse') return 'muse_glimmer'
  return REASONING_PARSERS_FOR_CLI.has(parser) ? parser : undefined
}

export interface ReasoningParserResolution {
  configuredParser?: string | null
  detectedParser?: string | null
  supportsThinking?: boolean
}

/**
 * Resolve the one parser identity used by launch argv, command preview,
 * gateway capabilities, chat request policy, and renderer settings.
 *
 * `""` and `"none"` are explicit opt-outs. `auto` and an omitted setting use
 * current bundle detection. A detector that explicitly says the model does
 * not support thinking emits the engine's literal opt-out rather than leaving
 * an absent CLI flag that the backend could reinterpret as Auto.
 */
export function resolveEffectiveReasoningParser(
  input: ReasoningParserResolution,
): string | undefined {
  const configured = typeof input.configuredParser === 'string'
    ? input.configuredParser.trim()
    : undefined
  const detected = typeof input.detectedParser === 'string'
    ? input.detectedParser.trim()
    : undefined

  if (configured === '' || configured === 'none') return 'none'
  if (input.supportsThinking === false) return 'none'

  return canonicalizeReasoningParserForCli(
    configured && configured !== 'auto' ? configured : detected,
  )
}

export function reasoningParserIsEnabled(parser?: string): boolean {
  const canonical = canonicalizeReasoningParserForCli(parser)
  return canonical !== undefined && canonical !== 'none'
}
