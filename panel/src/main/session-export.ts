import type { Chat, Message } from './database'

const REQUEST_SETTINGS = [
  'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'max_output_tokens',
  'max_completion_tokens', 'repetition_penalty', 'frequency_penalty', 'presence_penalty',
  'seed', 'stop', 'enable_thinking', 'thinking_mode', 'reasoning_effort', 'reasoning',
  'max_thinking_tokens', 'chat_template_kwargs', 'tool_choice', 'parallel_tool_calls',
  'response_format', 'text', 'image_token_budget', 'video_fps', 'video_max_frames',
  'video_max_pixels', 'video_token_budget', 'stream',
  'skip_prefix_cache', 'cache_salt',
] as const
const SERVER_SETTINGS = [
  'maxContextLength', 'maxTokens', 'defaultMaxNewTokens', 'defaultTemperature',
  'defaultTopP', 'defaultTopK', 'defaultMinP', 'defaultRepetitionPenalty',
  'reasoningParser', 'toolCallParser', 'enableAutoToolChoice', 'kvCacheQuantization',
  'kvCacheGroupSize', 'pagedCacheBlockSize', 'maxCacheBlocks', 'enablePrefixCache',
  'enableBlockDiskCache', 'blockDiskCacheMaxPercent', 'blockDiskCacheMaxGb',
  'usePagedCache', 'prefillBatchSize', 'prefillStepSize', 'maxNumSeqs', 'streamInterval',
  'enableThinking', 'thinkingMode', 'reasoningEffort', 'imageTokenBudget', 'videoFps', 'videoMaxFrames',
  'videoMaxPixels', 'videoTokenBudget',
  // These are requested execution controls, not a claim that an accelerator ran.
  'nativeMtpMode', 'nativeMtpDepth', 'nativeMtpDepthOverride',
  'speculativeModel', 'numDraftTokens', 'omniBackend', 'enableJit',
] as const

function pick(value: Record<string, unknown>, keys: readonly string[]): Record<string, unknown> {
  return Object.fromEntries(keys.filter(k => value[k] !== undefined).map(k => [k, value[k]]))
}

export interface GenerationRecord {
  version: 1
  status: 'in_progress' | 'completed' | 'interrupted'
  finishReason?: string
  toolExchange?: unknown[]
  passes: ReturnType<typeof captureGenerationPass>[]
}

/** Snapshot only the request/config fields needed for reproducibility, never headers or credentials. */
export function captureGenerationPass(args: {
  body: Record<string, any>
  modelPath?: string
  family?: string
  wireApi: string
  serverConfig: Record<string, unknown>
  health?: Record<string, any>
  toolExchange?: unknown[]
  now?: number
}) {
  const { body, health } = args
  // Serialize now: later settings edits and tool-loop mutations must not alter history.
  return JSON.parse(JSON.stringify({
    submittedAt: args.now ?? Date.now(),
    modelId: body.model,
    modelPath: args.modelPath ?? null,
    family: args.family ?? null,
    wireApi: args.wireApi,
    requestSettings: pick(body, REQUEST_SETTINGS),
    serverSettings: pick(args.serverConfig, SERVER_SETTINGS),
    serverDefaults: health?.effective_defaults ?? null,
    bundleDefaults: health?.sampling_defaults ?? null,
    maxPromptTokens: health?.max_prompt_tokens ?? args.serverConfig.maxContextLength ?? null,
    activeParsers: health?.active_parsers ?? null,
    // Distinguish the submitted instructions from a later chat-settings value.
    instructions: body.instructions ?? null,
    systemMessages: (body.messages ?? body.input ?? []).filter((m: any) =>
      m?.role === 'system' || m?.role === 'developer'),
    tools: body.tools ?? [],
    toolExchange: args.toolExchange ?? [],
  })) as {
    submittedAt: number; modelId: string; modelPath: string | null; family: string | null;
    wireApi: string; requestSettings: Record<string, unknown>; serverSettings: Record<string, unknown>;
    serverDefaults: Record<string, unknown> | null; bundleDefaults: Record<string, unknown> | null;
    maxPromptTokens: number | null; activeParsers: unknown; instructions: unknown;
    systemMessages: unknown[]; tools: unknown[]; toolExchange: unknown[];
  }
}

export function literalBlock(text: string, language = 'text'): string {
  const longest = Math.max(2, ...Array.from(text.matchAll(/`+/g), m => m[0].length))
  const fence = '`'.repeat(longest + 1)
  return `${fence}${language}\n${text}\n${fence}`
}

function jsonBlock(raw: string): string {
  try { return literalBlock(JSON.stringify(JSON.parse(raw), null, 2), 'json') }
  catch { return literalBlock(raw) }
}

/** Uses saved records only: exporting must never reinterpret history using current settings. */
export function renderSessionMarkdown(chat: Chat, messages: Message[], now = new Date()): string {
  const lines = [
    '# Session export', '', literalBlock(chat.title), '',
    `Exported: ${now.toISOString()}`, `Created: ${new Date(chat.createdAt).toISOString()}`, '',
    '## Conversation identity', '',
    literalBlock(JSON.stringify({ chatId: chat.id, modelId: chat.modelId, modelPath: chat.modelPath ?? null }, null, 2), 'json'), '',
    'Settings units: requestSettings and serverDefaults use API units. serverSettings preserves the app configuration; defaultTemperature, defaultTopP, defaultMinP and defaultRepetitionPenalty are stored as integer percentages. Omitted request fields inherit the recorded server defaults when available.', '',
    'The conversation identity is the saved chat association. Per-generation records below identify the model used for each recorded request.', '',
    'Reasoning is the full trace recorded by this app, when supplied by the model/provider. Unavailable historical settings are not reconstructed from current defaults. Request overrides and server-reported defaults are recorded separately; provider-internal choices are not inferred.', '',
  ]
  messages.forEach((m, index) => {
    lines.push(`## ${index + 1}. ${m.role}`, '', `Timestamp: ${new Date(m.timestamp).toISOString()}`, '')
    if (m.generationRecordJson) {
      lines.push('### Generation record', '', jsonBlock(m.generationRecordJson), '')
    } else if (m.role === 'assistant') {
      lines.push('Historical generation settings, effective system prompt and per-request model identity: unavailable (not recorded for this message).', '')
    }
    if (m.reasoningSegmentsJson) {
      lines.push('### Ordered reasoning segments', '', jsonBlock(m.reasoningSegmentsJson), '')
    }
    if (m.reasoningContent) lines.push('### Recorded reasoning', '', literalBlock(m.reasoningContent), '')
    lines.push('### Content', '', literalBlock(m.content), '')
    for (const [title, value] of [
      ['Tool calls', m.toolCallsOaiJson], ['Tool results', m.toolResultsOaiJson],
      ['Tool activity', m.toolCallsJson], ['Warnings', m.warningsJson], ['Metrics', m.metricsJson],
    ]) {
      if (value) lines.push(`### ${title}`, '', jsonBlock(value), '')
    }
    if (m.toolCallId) lines.push('Tool call ID:', '', literalBlock(m.toolCallId), '')
    if (m.toolCapabilityFingerprint) lines.push('### Tool schema fingerprint', '', literalBlock(m.toolCapabilityFingerprint), '')
  })
  return lines.join('\n')
}

/** Parse this export's literal blocks without treating fenced model text as headings. */
export function parseSessionMarkdown(raw: string): { title: string; modelId?: string; modelPath?: string; createdAt?: number; messages: Partial<Message>[] } | null {
  if (!raw.startsWith('# Session export\n')) return null
  const messages: Partial<Message>[] = []
  let current: Partial<Message> | undefined
  let section = ''
  let title: string | undefined
  const identity: { modelId?: string; modelPath?: string; createdAt?: number } = {}
  let fence: string | undefined
  let block: string[] = []
  for (const line of raw.split('\n')) {
    if (fence) {
      if (line === fence) {
        const value = block.join('\n')
        if (title === undefined) title = value
        else if (!current && section === 'Conversation identity') {
          try {
            const saved = JSON.parse(value)
            if (typeof saved?.modelId === 'string') identity.modelId = saved.modelId
            if (typeof saved?.modelPath === 'string') identity.modelPath = saved.modelPath
          } catch { /* Edited/legacy exports may not contain structured identity. */ }
        }
        else if (current) {
          const field = ({
            'Content': 'content', 'Recorded reasoning': 'reasoningContent',
            'Ordered reasoning segments': 'reasoningSegmentsJson',
            'Generation record': 'generationRecordJson', 'Tool calls': 'toolCallsOaiJson',
            'Tool results': 'toolResultsOaiJson', 'Tool activity': 'toolCallsJson',
            'Warnings': 'warningsJson', 'Metrics': 'metricsJson',
            'Tool call ID': 'toolCallId', 'Tool schema fingerprint': 'toolCapabilityFingerprint',
          } as const)[section as 'Content']
          if (field) current[field] = value
        }
        fence = undefined
        block = []
      } else block.push(line)
      continue
    }
    const opening = line.match(/^(`{3,})(?:text|json)$/)
    if (opening) { fence = opening[1]; continue }
    const turn = line.match(/^## \d+\. (user|assistant|system)$/)
    if (turn) {
      current = { role: turn[1] as Message['role'], content: '' }
      messages.push(current)
      section = ''
    } else if (line === '## Conversation identity') section = 'Conversation identity'
    else if (line === 'Tool call ID:') section = 'Tool call ID'
    else if (line.startsWith('### ')) section = line.slice(4)
    else if (line.startsWith('Created: ') && !current) {
      const time = Date.parse(line.slice(9))
      if (Number.isFinite(time)) identity.createdAt = time
    }
    else if (line.startsWith('Timestamp: ') && current) {
      const time = Date.parse(line.slice(11))
      if (Number.isFinite(time)) current.timestamp = time
    }
  }
  return { title: title ?? 'Imported session', ...identity, messages }
}
