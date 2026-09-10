const TOOL_PARSER_CANONICAL_ALIASES: Record<string, string> = {
  deepseek_v4: 'dsml',
  hy_v3: 'hunyuan',
  // Muse Glimmer's parser registers under all three names in the engine
  // (atem_tool_parser.py: register_module(["atem", "muse_glimmer", "muse"])).
  muse: 'atem',
  muse_glimmer: 'atem',
  minicpm5_xml_function: 'minicpm5',
  // Zaya likewise registers "zaya"/"zyphra" alongside "zaya_xml".
  zaya: 'zaya_xml',
  zyphra: 'zaya_xml',
  // Qwen3.6-27B D-series (and the Qwen 3.8 line) stamp `qwen3_coder`, whose
  // template emits the XML-function shape (<function=NAME><parameter=KEY>),
  // NOT the plain qwen `<tool_call>{json}`. The engine registers the literal
  // name too, but the alias maps to a name every RELEASED engine also knows,
  // so app sessions work against older bundled runtimes as well.
  qwen3_coder: 'xml_function',
  // dots3_note's dots XML dialect registers as dots/dots3/dots3_note in the
  // engine (dots_tool_parser.py); canonical serving name is 'dots'.
  dots3: 'dots',
  dots3_note: 'dots',
  // GLM-5.3-Flash (glm5_next) bundles stamp glm_xml_args; the dialect is the
  // glm47 arg_key/arg_value XML and the engine registers the literal name as
  // a glm47 alias. Canonicalize to a name every released engine also knows.
  glm_xml_args: 'glm47',
}

export const TOOL_PARSERS_FOR_CLI = new Set([
  'mistral',
  'qwen',
  'llama',
  'hermes',
  'deepseek',
  'kimi',
  'lfm2',
  'minicpm5',
  'granite',
  'nemotron',
  'minimax',
  'xlam',
  'functionary',
  'glm47',
  'spark25',
  'step3p5',
  'gemma3',
  'gemma3n',
  'xml_function',
  'dsml',
  'zaya_xml',
  'hunyuan',
  'openpangu',
  'generic',
  'qwen3',
  'llama3',
  'llama4',
  'nous',
  'deepseek_v3',
  'deepseek_r1',
  'kimi_k2',
  'moonshot',
  'liquid',
  'granite3',
  'nemotron3',
  'minimax_m2',
  'minimax_m3',
  'meetkai',
  'stepfun',
  'glm4',
  'gemma4',
  'tencent',
  'openpangu_v2',
  // A name missing here is not a no-op: canonicalizeToolParserId returns
  // undefined and toolLaunchArgs then emits NEITHER --tool-call-parser NOR
  // --enable-auto-tool-choice, so the family loses tool calling entirely when
  // launched from the app while `vmlx serve --tool-call-parser <name>` works.
  // That is how Muse Glimmer shipped with dead tool calling in the app: the
  // panel's own registry stamps toolParser:'atem' and the dropdown offers it,
  // but 'atem' was absent here so the choice could not even stick.
  'atem',
  'poolside_v1',
  'mimo_xml_function',
  'dots',
])

export function canonicalizeToolParserId(
  value: string | null | undefined,
): string | undefined {
  if (value == null) return undefined
  const parser = value.trim()
  if (!parser || parser === 'auto' || parser === 'none') return parser
  const canonical = TOOL_PARSER_CANONICAL_ALIASES[parser] || parser
  return TOOL_PARSERS_FOR_CLI.has(canonical) ? canonical : undefined
}

/**
 * Human-readable form of a DETECTED parser name for "Auto (detected: …)"
 * labels. A bundle-stamped alias (qwen3_coder, muse, zaya…) is shown together
 * with the canonical parser it launches with, because the dropdown only
 * offers the canonical name — without the arrow the detected value looks
 * unrelated to every selectable option.
 */
export function describeDetectedToolParser(
  detected: string | null | undefined,
): string | undefined {
  const raw = typeof detected === 'string' ? detected.trim() : ''
  if (!raw) return undefined
  const canonical = TOOL_PARSER_CANONICAL_ALIASES[raw]
  return canonical && canonical !== raw ? `${raw} → ${canonical}` : raw
}

export interface ToolParserResolution {
  configuredParser?: string | null
  detectedParser?: string | null
}

/**
 * Resolve the one parser identity used by launch argv, command preview, and
 * protocol capability reporting.
 *
 * Empty string and "none" are explicit opt-outs. Auto/missing settings use
 * current bundle detection. A stale unsupported saved value falls back to the
 * current detector instead of reaching argparse as an invalid choice.
 */
export function resolveEffectiveToolParser(
  input: ToolParserResolution,
): string | undefined {
  const configured = typeof input.configuredParser === 'string'
    ? input.configuredParser.trim()
    : undefined
  const detected = typeof input.detectedParser === 'string'
    ? input.detectedParser.trim()
    : undefined

  if (configured === '' || configured === 'none') return 'none'

  const explicit = configured && configured !== 'auto'
    ? canonicalizeToolParserId(configured)
    : undefined
  if (explicit) return explicit
  return canonicalizeToolParserId(detected)
}

export function toolParserIsEnabled(parser?: string): boolean {
  const canonical = canonicalizeToolParserId(parser)
  return canonical !== undefined && canonical !== '' && canonical !== 'auto' && canonical !== 'none'
}
