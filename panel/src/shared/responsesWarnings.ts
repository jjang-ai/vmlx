/**
 * Responses API `warnings` field extractor.
 *
 * vMLX extends the OpenAI Responses API schema with a non-fatal
 * `warnings: string[] | null` field on `ResponsesObject`. It surfaces when:
 *   - `previous_response_id` chained a response that produced reasoning only
 *     (no visible message, no tool calls). Chat continuity may be impaired
 *     and prefix-cache reuse may be lower than expected.
 *   - (future) other coherence/observability signals
 *
 * This helper is pure and shared by the main process and renderer so warning
 * parsing stays identical across IPC, persistence, and UI rendering.
 */

export interface ResponsesPayloadLike {
  type?: unknown
  warnings?: unknown
  message?: unknown
  // Other fields are present but not required for warning extraction.
}

/**
 * Extract a clean `string[]` of warnings from a Responses API payload.
 *
 * Returns `null` (NOT `[]`) when no warnings are present — caller can use
 * truthy check to decide whether to render any UI. Returns deduplicated,
 * trimmed, non-empty strings only.
 */
export function extractResponsesWarnings(
  payload: ResponsesPayloadLike | null | undefined,
): string[] | null {
  if (!payload) return null
  const raw = payload.warnings
  if (payload.type === 'response.warning' && typeof payload.message === 'string') {
    const text = payload.message.trim()
    return text ? [text] : null
  }
  if (!Array.isArray(raw)) return null
  const seen = new Set<string>()
  const cleaned: string[] = []
  for (const item of raw) {
    if (typeof item !== 'string') continue
    const text = item.trim()
    if (!text) continue
    if (seen.has(text)) continue
    seen.add(text)
    cleaned.push(text)
  }
  return cleaned.length > 0 ? cleaned : null
}

/**
 * Map a Responses warning to a stable category key, useful for UI styling
 * or analytics. Falls back to "other" for unrecognized warnings so the UI
 * always has something to render.
 */
export function categorizeResponsesWarning(warning: string): string {
  const lower = warning.toLowerCase()
  if (lower.includes('reasoning only') || lower.includes('previous_response_id')) {
    return 'chain_reasoning_only'
  }
  if (lower.includes('cache') || lower.includes('prefix')) {
    return 'cache_alignment'
  }
  return 'other'
}
