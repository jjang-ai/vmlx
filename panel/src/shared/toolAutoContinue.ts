export interface ToolAutoContinueInput {
  content: string
  iterationTokenCount: number
  finishReason?: string | null
  thresholdTokens: number
}

/** Describe only observed loop/recovery state; do not imply a tool succeeded. */
export function noVisibleToolAnswerWarning(input: {
  toolIteration: number
  maxToolIterations: number
  recoveryAttempted: boolean
}): string {
  if (input.toolIteration >= input.maxToolIterations) {
    const recovery = input.recoveryAttempted
      ? 'A direct-answer recovery was attempted.'
      : 'No direct-answer recovery was attempted.'
    return `The tool loop reached its iteration limit (${input.maxToolIterations}) without a visible answer. ${recovery}`
  }
  if (input.recoveryAttempted) {
    return 'The tool loop ended without a visible answer after a direct-answer recovery was attempted.'
  }
  return 'The tool loop ended without a visible answer before reaching its iteration limit. No direct-answer recovery was attempted.'
}

interface NamedToolDefinition {
  function: {
    name: string
  }
}

export type CurrentTurnToolChoice =
  | 'none'
  | { type: 'function'; name: string }
  | { type: 'function'; function: { name: string } }
  | undefined

export interface ToolRequestFields {
  hasTools: boolean
  tools?: unknown
  hasToolChoice: boolean
  toolChoice?: unknown
}

export interface CompletedToolAnswerPassInput {
  directAnswerAfterSingleTool: boolean
  exactFinalToolCount: number
  exactFinalToolsComplete: boolean
  exactlyOnceToolCount: number
  exactlyOnceToolsComplete: boolean
}

/**
 * Decide when a fulfilled local tool request must keep its original render.
 *
 * An authorized `exactly once` contract is complete after its last requested
 * tool runs regardless of how the user phrases the following answer. Making
 * schema preservation depend on a second "after the result" wording heuristic
 * removes the schema at the front of native templates such as DSV4 and turns
 * the whole tool-result continuation into a cold prefix.
 */
export function shouldPlanCompletedToolAnswerPass(
  input: CompletedToolAnswerPassInput,
): boolean {
  return (
    ((input.directAnswerAfterSingleTool || input.exactFinalToolCount > 1) &&
      input.exactFinalToolsComplete) ||
    (input.exactlyOnceToolCount > 0 && input.exactlyOnceToolsComplete)
  )
}

export function captureToolRequestFields(
  request: Record<string, unknown>,
): ToolRequestFields {
  return {
    hasTools: Object.prototype.hasOwnProperty.call(request, 'tools'),
    tools: request.tools,
    hasToolChoice: Object.prototype.hasOwnProperty.call(request, 'tool_choice'),
    toolChoice: request.tool_choice,
  }
}

export function applyPostToolRequestFields(
  request: Record<string, unknown>,
  options: {
    finalAnswerRecovery: boolean
    plannedDirectAnswerPass: boolean
    isRemote: boolean
    previous?: ToolRequestFields
  },
): void {
  if (options.finalAnswerRecovery) {
    delete request.tools
    request.tool_choice = 'none'
    return
  }
  if (!options.plannedDirectAnswerPass || !options.previous) return

  // A local exact-tool continuation must render the identical scoped schema
  // prefix that selected the tool. The app has already validated and executed
  // the call; an out-of-band fulfilled marker tells vmlx-engine that the
  // repeated choice is for stable prompt identity, not a second-call demand.
  if (options.previous.hasTools) request.tools = options.previous.tools
  else delete request.tools
  if (!options.isRemote && options.previous.hasToolChoice) {
    request.tool_choice = options.previous.toolChoice
  } else {
    // Remote providers do not understand vMLX's fulfilled marker. Keep their
    // schema prefix stable but return the standard choice policy to auto.
    delete request.tool_choice
  }
}

export function requiredToolChoiceNamesForCurrentTurn(
  exactFinalToolNames: string[],
  exactlyOnceToolNames: string[],
): string[] {
  if (exactFinalToolNames.length > 0) return exactFinalToolNames
  // "Call X exactly once" is itself a current-turn execution contract. It
  // must not silently degrade to auto choice merely because the requested
  // post-tool answer is phrased differently (or is open ended).
  return exactlyOnceToolNames
}

export function unavailableRequestedToolNames(
  requestedNames: string[],
  availableNames: string[],
): string[] {
  const available = new Set(availableNames.map(name => name.toLowerCase()))
  return requestedNames.filter(name => !available.has(name.toLowerCase()))
}

export function toolChoiceForCurrentTurn(
  suppressAllTools: boolean,
  exactFinalToolNames: string[],
  wireApi: 'responses' | 'chat',
  isRemote = false,
): CurrentTurnToolChoice {
  // Remote providers already receive no schemas for an explicit no-tool turn;
  // keep omitting `tool_choice="none"` for strict third-party compatibility.
  // A singular explicit tool contract is different: the scoped schema alone
  // still means auto choice, so forward the standard endpoint-native specific
  // choice instead of silently weakening the user's instruction.
  if (suppressAllTools) return isRemote ? undefined : 'none'
  if (exactFinalToolNames.length === 0) return undefined

  // Specific tool_choice is singular in both wire APIs. For an exact ordered
  // multi-tool contract, authorize the first remaining function; the next
  // request advances after that function is recorded as completed.
  const name = exactFinalToolNames[0]
  return wireApi === 'responses'
    ? { type: 'function', name }
    : { type: 'function', function: { name } }
}

export function scopeToolDefinitionsByName<T extends NamedToolDefinition>(
  tools: T[],
  names: string[],
): T[] {
  if (names.length === 0) return tools
  const allowed = new Set(names.map(name => name.toLowerCase()))
  return tools.filter(tool => allowed.has(tool.function.name.toLowerCase()))
}

export function isToolNameProvidedForCurrentTurn(
  toolName: string,
  providedNames: string[],
): boolean {
  const normalized = toolName.toLowerCase()
  return providedNames.some(name => name.toLowerCase() === normalized)
}

export function isToolAuthorizedForCurrentTurn(
  toolName: string,
  providedNames: string[],
  suppressAllTools: boolean,
  restrictedToolNames: string[],
): boolean {
  const restrictToProvidedNames =
    suppressAllTools || restrictedToolNames.length > 0
  return (
    !restrictToProvidedNames ||
    isToolNameProvidedForCurrentTurn(toolName, providedNames)
  )
}

export function shouldAutoContinueAfterToolUse({
  content,
  iterationTokenCount,
  finishReason,
  thresholdTokens,
}: ToolAutoContinueInput): boolean {
  if (!content.trim()) return true
  return finishReason === 'length' && iterationTokenCount < thresholdTokens
}

export function repeatsOnlyCompletedExactlyOnceTools(
  toolNames: string[],
  exactlyOnceToolNames: string[],
  completedExactlyOnceTools: ReadonlySet<string>,
): boolean {
  if (toolNames.length === 0 || exactlyOnceToolNames.length === 0) return false
  const requested = new Set(exactlyOnceToolNames.map(name => name.toLowerCase()))
  return toolNames.every(name => {
    const normalized = name.toLowerCase()
    return requested.has(normalized) && completedExactlyOnceTools.has(normalized)
  })
}

export function shouldFinishZayaAppleScriptToolRound(
  isAppleScriptToolBundle: boolean,
  toolNames: string[],
): boolean {
  return (
    isAppleScriptToolBundle &&
    toolNames.length > 0 &&
    toolNames.every((name) => name === 'run_applescript')
  )
}

const CONSECUTIVE_REPLAY_SAFE_BUILTIN_TOOLS = new Set(['read_file'])

function stableToolArgumentValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableToolArgumentValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, stableToolArgumentValue(item)]),
    )
  }
  return value
}

export function replaySafeToolCallKey(
  toolName: string,
  args: Record<string, unknown>,
): string | undefined {
  const normalizedName = toolName.toLowerCase()
  // Keep this intentionally narrow. Replaying run_command, write/edit/delete,
  // MCP, network, clipboard, process, or clock calls can be semantically
  // required or side-effecting. An immediately repeated successful read_file
  // with byte-identical arguments is the one live-proven redundant case.
  if (!CONSECUTIVE_REPLAY_SAFE_BUILTIN_TOOLS.has(normalizedName)) return undefined
  return `${normalizedName}:${JSON.stringify(stableToolArgumentValue(args))}`
}

export function requestedOnceToolNames(text: string): string[] {
  // An explicit per-tool `exactly once` directive is an execution invariant,
  // independent of how the user phrases the final-answer request. Retire each
  // named tool after its first call while leaving other tools available for
  // genuinely agentic multi-tool work.
  const matches = [
    ...text.matchAll(
      /\b(?:call|use|invoke|run|execute)\s+(?:the\s+)?(?:built[- ]in\s+)?`?([a-z][\w-]*)`?(?:\s+(?:tool|function))?\s+exactly\s+once\b/gi,
    ),
    // Natural prompts often put the concrete argument before the cardinality:
    // "Call read_file with path README.md exactly once." Keep the argument
    // span inside one clause and reject another tool verb so a later
    // "then call Y exactly once" cannot accidentally constrain X as well.
    ...text.matchAll(
      /\b(?:call|use|invoke|run|execute)\s+(?:the\s+)?(?:built[- ]in\s+)?`?([a-z][\w-]*)`?(?:\s+(?:tool|function))?\s+(?:with|using|on)\b(?:(?!\b(?:call|use|invoke|run|execute)\b|[!?;\n]|\.(?:\s|$)).){0,96}\bexactly\s+once\b/gi,
    ),
    ...text.matchAll(
      /\b(?:call|use|invoke|run|execute)\s+exactly\s+(?:one|1)\s+(?:the\s+)?(?:built[- ]in\s+)?`?([a-z][\w-]*)`?(?:\s+(?:tool|function))?(?:\s+call)?\b/gi,
    ),
  ].sort((a, b) => (a.index || 0) - (b.index || 0))
  const names = matches
    .filter(match => !isNegatedDirectiveAt(text, match.index || 0))
    .map(match => match[1].toLowerCase())
  if (names.length === 0 || new Set(names).size !== names.length) return []
  return names
}

function escapedToolName(name: string): string {
  return name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function isNegatedDirectiveAt(text: string, directiveIndex: number): boolean {
  const prefix = text.slice(Math.max(0, directiveIndex - 96), directiveIndex)
  // Only inspect the current clause. A prior "Do not guess." must not negate
  // a later positive "Use file_info" directive, while "do not use ..." and
  // "never call ..." must not be reinterpreted as authorization.
  return /\b(?:do\s+not|don['’]t|never|without)\b[^.!?;\n]{0,80}$/i.test(
    prefix,
  )
}

function explicitlyRequestedCatalogToolNames(
  text: string,
  catalogToolNames: string[],
): string[] {
  const matches: Array<{ name: string; index: number }> = []
  const seen = new Set<string>()

  for (const rawName of catalogToolNames) {
    const name = rawName.toLowerCase()
    if (!name || seen.has(name)) continue
    const requestPattern = new RegExp(
      `\\b(?:call|use|invoke|run|execute)\\s+(?:the\\s+)?(?:built[- ]in\\s+)?` +
        `(?:\`${escapedToolName(name)}\`|${escapedToolName(name)})` +
        `(?:\\s+(?:tool|function))?\\b`,
      'i',
    )
    const match = requestPattern.exec(text)
    if (!match || isNegatedDirectiveAt(text, match.index)) continue
    seen.add(name)
    matches.push({ name, index: match.index })
  }

  return matches.sort((a, b) => a.index - b.index).map(match => match.name)
}

function permitsAdditionalToolUse(text: string): boolean {
  const pattern = /\buse\s+tools?\s+as\s+needed\b|\b(?:use|call|invoke|run|execute)\s+(?:any\s+)?(?:other|additional|more|further)\s+tools?\b/gi
  return [...text.matchAll(pattern)].some(
    match => !isNegatedDirectiveAt(text, match.index || 0),
  )
}

export function requestedScopedToolNames(
  text: string,
  catalogToolNames: string[] = [],
): string[] {
  const exactlyOnceNames = requestedOnceToolNames(text)
  if (exactlyOnceNames.length === 0 || permitsAdditionalToolUse(text)) return []

  const explicitlyRequestedNames = explicitlyRequestedCatalogToolNames(
    text,
    catalogToolNames,
  )
  if (explicitlyRequestedNames.length === 0) return exactlyOnceNames

  const requested = new Set(explicitlyRequestedNames)
  for (const name of exactlyOnceNames) requested.add(name)
  return [...requested]
}

export function requestedExactFinalToolNames(
  text: string,
  catalogToolNames: string[] = [],
): string[] {
  // This optimization removes the whole tool catalog from the final-answer
  // follow-up, so keep it narrower than the per-tool exactly-once invariant.
  // A multi-tool contract remains agentic until every explicitly named tool
  // has completed.
  const names = requestedOnceToolNames(text)
  if (names.length === 0) return []

  // Do not narrow an explicitly open-ended agentic request merely because it
  // also constrains one named tool. In that case the remaining catalog is part
  // of the user's stated contract.
  if (permitsAdditionalToolUse(text)) return []

  // Keep this bounded to the final-result clause, but allow ordinary
  // modifiers from the exact contract ("the real tool result", "both tool
  // results"). The previous literal-only match left tools enabled and a live
  // Bonsai turn executed the same file_info call five times.
  const exactFinalAfterResults =
    /\bafter\b[^.!?\n]{0,96}\b(?:the\s+(?:real\s+)?tool|its|that|both\s+tool)\s+results?\b[^.!?\n]{0,64}\breply exactly\b/i.test(
      text,
    )
  // A natural equivalent is a bounded two-step contract:
  // "Call file_info exactly once. Use the real tool. Then reply exactly ...".
  // Treating only the "After the tool result" spelling as scoped exposed the
  // whole built-in catalog; an LFM2 turn then called write_file even though
  // the current request named only file_info. This changes the schemas sent to
  // the model rather than hiding an emitted call.
  const exactFinalThen =
    /\bthen\s*,?\s*(?:reply\s+exactly\b|(?:(?:your|the)\s+)?(?:only\s+)?visible\s+answer\s+(?:must\s+be\s+|is\s+|exactly\s+))/i.test(
      text,
    )
  if (!(exactFinalAfterResults || exactFinalThen)) return []

  // "Exactly once" may constrain only one step in a bounded multi-tool
  // contract. Preserve every other explicitly requested current-catalog tool
  // instead of accidentally scoping it away:
  // "Call file_info exactly once. Call read_file ... Then reply exactly ...".
  const explicitlyRequestedNames = explicitlyRequestedCatalogToolNames(
    text,
    catalogToolNames,
  )
  if (explicitlyRequestedNames.length === 0) return names

  const requested = new Set(explicitlyRequestedNames)
  for (const name of names) requested.add(name)
  return [...requested]
}

export function requestsBoundedFinalAnswerAfterToolResult(
  text: string,
  catalogToolNames: string[] = [],
): boolean {
  const exactlyOnceNames = requestedOnceToolNames(text)
  if (exactlyOnceNames.length === 0) return false
  if (permitsAdditionalToolUse(text)) return false
  const exactlyOnceSet = new Set(exactlyOnceNames)
  if (
    explicitlyRequestedCatalogToolNames(text, catalogToolNames).some(
      (name) => !exactlyOnceSet.has(name),
    )
  ) {
    return false
  }

  // This is deliberately narrower than a generic mention of a tool result:
  // the same clause must explicitly say that the user wants the assistant's
  // answer after that result. It covers normal bounded wording such as
  // "After the tool result is returned, reply briefly" without requiring the
  // answer itself to have exact literal wording.
  return /\bafter\b[^.!?\n]{0,96}\b(?:(?:the|both)\s+(?:real\s+)?(?:tool|function)\s+results?|(?:its|that)\s+results?)\b[^.!?\n]{0,96}\b(?:reply|respond|answer)\b/i.test(
    text,
  )
}

function containsExplicitToolRequest(text: string): boolean {
  // Resolve an explicit prohibition before looking for positive tool verbs.
  // Otherwise phrases such as "do not call a tool" are misread as a request
  // for a tool literally named "a" by the broad explicit-tool grammar below.
  if (requestsNoToolCalls(text)) return false

  const explicitToolRequest =
    /\b(?:call|use|invoke|run|execute)\s+(?:the\s+)?(?:built[- ]in\s+)?`?[a-z][\w-]*`?(?:\s+(?:tool|function))?\b/i
  const toolResultContract =
    /\bafter\b[^.!?\n]{0,120}\b(?:tool|function)\s+results?\b/i
  const mustUseTool =
    /\bmust\s+(?:call|use|invoke|run|execute)\s+(?:the\s+)?(?:built[- ]in\s+)?(?:`?[a-z][\w-]*`?\s+)?(?:tool|function)\b/i

  return (
    explicitToolRequest.test(text) ||
    toolResultContract.test(text) ||
    mustUseTool.test(text) ||
    requestedExactFinalToolNames(text).length > 0
  )
}

export function requestsExactTextOnlyWithoutToolUse(text: string): boolean {
  const strictTextAnswer =
    /\breply exactly\b/i.test(text) ||
    /\banswer exactly\b/i.test(text) ||
    /\bvisible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text) ||
    /\bonly visible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text) ||
    /\byour only visible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text)
  if (!strictTextAnswer) return false
  if (requestedExactFinalToolNames(text).length > 0) return false

  // A previous chat/profile may leave builtin tools enabled. For strict
  // exact-answer probes that do not ask for tool use, sending the whole tool
  // catalog changes the prompt and lets small/native models answer from schema
  // text instead of the current user turn. Keep this directive-shaped so normal
  // agentic coding chats still receive tools.
  return !containsExplicitToolRequest(text)
}

export function requestsPrivateReasoningWithoutToolUse(text: string): boolean {
  if (containsExplicitToolRequest(text)) return false

  const privateReasoningDirective =
    /\b(?:privately|private(?:ly)?\s+reason|private\s+(?:calculation|check|solution)|in\s+private|do\s+not\s+expose\s+(?:the\s+)?reasoning|without\s+showing\s+(?:the\s+)?reasoning|silent(?:ly)?\s+(?:reason|think)|think\s+privately)\b/i
  if (!privateReasoningDirective.test(text)) return false

  const textOnlyReasoningTask =
    /\b(?:calculate|calculation|compute|solve|solution|double[- ]?check|check\s+(?:whether|if)|verify|reason\s+through|derive|evaluate)\b/i
  return textOnlyReasoningTask.test(text)
}

export function requestsNoToolCalls(text: string): boolean {
  // Keep these directive-shaped so quoted discussion of tool policy does not
  // silently disable the catalog. The UI omits tool schemas entirely when
  // this returns true; that is the stable no-tool request contract for both
  // Responses and Chat Completions.
  const explicitProhibition =
    /(?:^|[.!?;:\]\n])\s*(?:please\s+)?(?:do not|don['’]?t|dont|never)\s+(?:call|use)\s+(?:(?:a|the|any|another|additional|more)\s+)?tools?\b(?!\s+unless)/i
  const explicitWithoutTools =
    /(?:^|[.!?;:\]\n])\s*(?:please\s+)?without\s+(?:(?:using|calling)\s+)?(?:any\s+)?tools?\b/i
  // Imperative phrasing can put the restriction after the verb, as in
  // "Now solve this without tools:". Do not match quotations, explanations
  // about the phrase, or conditional permission to use tools.
  const imperativeWithoutTools =
    /(?:^|[.!?;:\]\n])\s*(?:please\s+)?(?:now\s+)?(?:answer|respond|solve|calculate|compute|work\s+(?:this|it)\s+out)(?:\s+(?:this|that|it|directly|the\s+(?:question|problem)))?\s+without\s+(?:(?:using|calling)\s+)?(?:any\s+)?tools?\b(?!\s+(?:unless|if|when))/i
  return explicitProhibition.test(text) || explicitWithoutTools.test(text) || imperativeWithoutTools.test(text)
}
