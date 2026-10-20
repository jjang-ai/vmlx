import { REASONING_WITHOUT_ANSWER_NOTICE } from './responsesStreamRecovery'

export interface PersistedAssistantToolHistory {
  generationRecordJson?: unknown
  content?: unknown
  warningsJson?: unknown
  reasoningContent?: unknown
  reasoningSegmentsJson?: unknown
  toolCallsJson?: unknown
  toolCallsOaiJson?: unknown
  toolResultsOaiJson?: unknown
}

export interface AssistantHistoryReplayOptions {
  /**
   * Private reasoning from an older turn is not user-visible conversation
   * state.  A current explicit no-tool directive must not inherit an older
   * assistant's internal debate about tool requirements, because small native
   * reasoning models can mistake that stale text for current authority.
   * Calls, results, visible answers, and assistant boundaries are still
   * replayed when this is false.
   */
  includeReasoning?: boolean
}

interface OaiToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OaiToolResult {
  tool_call_id: string
  content: string
}

function parseArray(value: unknown): any[] {
  if (typeof value !== 'string' || value.length === 0) return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function validCalls(value: unknown): OaiToolCall[] {
  return parseArray(value).filter(
    (call: any) =>
      call &&
      typeof call.id === 'string' &&
      call.id.length > 0 &&
      call.function &&
      typeof call.function.name === 'string' &&
      typeof call.function.arguments === 'string',
  )
}

function validResults(value: unknown): OaiToolResult[] {
  return parseArray(value)
    .filter(
      (result: any) =>
        result &&
        typeof result.tool_call_id === 'string' &&
        typeof result.content === 'string',
    )
    .map((result: any) => ({
      tool_call_id: result.tool_call_id,
      content: result.content,
    }))
}

function persistedReasoningSegments(message: PersistedAssistantToolHistory): string[] {
  const segments = parseArray(message.reasoningSegmentsJson).map((segment) =>
    typeof segment === 'string' ? segment : '',
  )
  if (segments.length > 0) return segments
  return typeof message.reasoningContent === 'string' && message.reasoningContent
    ? [message.reasoningContent]
    : []
}

function reasoningItem(text: string): any {
  return {
    type: 'reasoning',
    content: [{ type: 'reasoning', text }],
  }
}

/** A completed request can have no displayable text (for example rejected
 * native tool markup). Its assistant boundary still separates user turns. */
export function hasCompletedAssistantGeneration(message: PersistedAssistantToolHistory): boolean {
  try {
    const record = JSON.parse(typeof message.generationRecordJson === 'string' ? message.generationRecordJson : '{}')
    return record.status === 'completed' && Array.isArray(record.passes) && record.passes.length > 0
  } catch {
    return false
  }
}

/**
 * Rebuild one persisted assistant row into model-visible history.
 *
 * Tool-loop rows collapse several model generations into one SQLite row.  The
 * UI already stores the native calls/results, per-generation reasoning
 * segments, and call iteration numbers.  Re-expand those fields in their real
 * order so Responses and Chat Completions templates see:
 * reasoning -> call -> result -> reasoning -> answer.
 */
export function replayPersistedAssistantHistory(
  message: PersistedAssistantToolHistory,
  useResponsesApi: boolean,
  options: AssistantHistoryReplayOptions = {},
): any[] {
  const calls = validCalls(message.toolCallsOaiJson)
  const results = validResults(message.toolResultsOaiJson)
  let content = typeof message.content === 'string' ? message.content : ''
  const nativeToolContent = new Map<string, string>()
  try {
    const record = JSON.parse(typeof message.generationRecordJson === 'string' ? message.generationRecordJson : '{}')
    if (record.passes?.[0]?.family === 'naive_n05_flash') {
      if (typeof record.nativeFinalContent === 'string') content = record.nativeFinalContent
      let pendingContent = ''
      for (const item of record.toolExchange || []) {
        if (item.type === 'output_text' && typeof item.text === 'string') pendingContent += item.text
        if (item.type === 'function_call') nativeToolContent.set(item.call_id, pendingContent)
        if (item.type === 'function_call_output') pendingContent = ''
        if (item.role === 'assistant' && Array.isArray(item.tool_calls)) {
          for (const call of item.tool_calls) nativeToolContent.set(call.id, typeof item.content === 'string' ? item.content : '')
        }
      }
    }
  } catch { /* Legacy rows use the existing typed fields. */ }
  const persistedSegments = persistedReasoningSegments(message)
  const segments = options.includeReasoning === false ? [] : persistedSegments

  // This exact app-generated notice is display metadata, not an assistant
  // answer. Require the persisted diagnostic as well, so literal model text
  // quoting the notice is preserved. Keep the real reasoning and turn boundary.
  if (
    calls.length === 0 &&
    persistedSegments.some((segment) => segment.trim()) &&
    content === REASONING_WITHOUT_ANSWER_NOTICE &&
    parseArray(message.warningsJson).some((warning) =>
      typeof warning === 'string' && warning.startsWith('This response produced reasoning only ('),
    )
  ) content = ''

  const callIterations = new Map<string, number>()
  for (const status of parseArray(message.toolCallsJson)) {
    if (
      status?.phase === 'calling' &&
      typeof status.toolCallId === 'string' &&
      Number.isInteger(status.iteration)
    ) {
      callIterations.set(status.toolCallId, Math.max(0, status.iteration))
    }
  }

  if (calls.length === 0) {
    if (!content.trim() && !segments[0]?.trim() && hasCompletedAssistantGeneration(message)) {
      return [useResponsesApi
        ? { type: 'message', role: 'assistant', content: '' }
        : { role: 'assistant', content: '' }]
    }
    if (useResponsesApi) {
      // A persisted reasoning-only assistant row usually means the previous
      // turn hit an answer-recovery/output-budget edge and produced no visible
      // assistant text. Replaying that hidden reasoning as a standalone
      // Responses reasoning item makes the next request treat stale private
      // thoughts as live assistant context. Preserve only the assistant turn
      // boundary; tool-loop rows above still replay their ordered reasoning.
      if (!content.trim() && persistedSegments[0]?.trim()) {
        return [{ type: 'message', role: 'assistant', content: '' }]
      }
      const items: any[] = []
      if (segments[0]?.trim()) items.push(reasoningItem(segments[0]))
      if (content.trim()) items.push({ type: 'output_text', text: content })
      return items
    }
    if (!content.trim() && !segments[0]?.trim()) return []
    return [
      {
        role: 'assistant',
        content,
        ...(segments[0]?.trim()
          ? { reasoning_content: segments[0] }
          : {}),
      },
    ]
  }

  const groups = new Map<number, OaiToolCall[]>()
  calls.forEach((call, index) => {
    const iteration = callIterations.get(call.id) ?? index
    const group = groups.get(iteration) || []
    group.push(call)
    groups.set(iteration, group)
  })
  const iterations = [...groups.keys()].sort((a, b) => a - b)
  const resultByCall = new Map(results.map((result) => [result.tool_call_id, result]))
  const replay: any[] = []

  // Reasoning segments are stored in step order (one per tool step, then the
  // final answer's). The live loop's status records number tool steps from 1
  // ("generating" is 0) while older rows and fixtures number them from 0, so
  // map each distinct step to its RANK, not its raw iteration value; indexing
  // by the raw value gave the first tool step the final answer's reasoning
  // and dropped the final reasoning, which re-rendered the turn differently
  // from the prompt the model actually saw and broke prefix reuse.
  iterations.forEach((iteration, rank) => {
    const group = groups.get(iteration) || []
    const reasoning = segments[rank]
    const toolContent = nativeToolContent.get(group[0]?.id) || ''
    if (useResponsesApi) {
      if (reasoning?.trim()) replay.push(reasoningItem(reasoning))
      if (toolContent) replay.push({ type: 'output_text', text: toolContent })
      for (const call of group) {
        replay.push({
          type: 'function_call',
          call_id: call.id,
          name: call.function.name,
          arguments: call.function.arguments,
        })
      }
      for (const call of group) {
        const result = resultByCall.get(call.id)
        if (result) {
          replay.push({
            type: 'function_call_output',
            call_id: result.tool_call_id,
            output: result.content,
          })
        }
      }
    } else {
      replay.push({
        role: 'assistant',
        content: toolContent || null,
        tool_calls: group,
        ...(reasoning?.trim() ? { reasoning_content: reasoning } : {}),
      })
      for (const call of group) {
        const result = resultByCall.get(call.id)
        if (result) {
          replay.push({
            role: 'tool',
            tool_call_id: result.tool_call_id,
            content: result.content,
          })
        }
      }
    }
  })

  const finalReasoning = segments[iterations.length]
  if (useResponsesApi) {
    if (finalReasoning?.trim()) replay.push(reasoningItem(finalReasoning))
    if (content.trim()) replay.push({ type: 'output_text', text: content })
  } else if (content.trim() || finalReasoning?.trim()) {
    replay.push({
      role: 'assistant',
      content,
      ...(finalReasoning?.trim()
        ? { reasoning_content: finalReasoning }
        : {}),
    })
  }
  return replay
}
