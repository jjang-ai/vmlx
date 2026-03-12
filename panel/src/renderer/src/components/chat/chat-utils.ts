/** Pure utility functions for chat UI — extracted for testability */

/** Format a timestamp into a human-readable relative string.
 *  - Within 1 minute: "Just now"
 *  - Today: "2:30 PM"
 *  - Yesterday: "Yesterday 2:30 PM"
 *  - This week: "Mon 2:30 PM"
 *  - Older: "Mar 5, 2:30 PM"
 */
export function formatTimestamp(ts: number): string {
  const now = new Date()
  const date = new Date(ts)
  const diffMs = now.getTime() - date.getTime()

  const timeStr = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })

  if (diffMs < 60_000) return 'Just now'

  const today = new Date(now); today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1)
  const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)

  if (ts >= today.getTime()) return timeStr
  if (ts >= yesterday.getTime()) return `Yesterday ${timeStr}`
  if (ts >= weekAgo.getTime()) {
    const day = date.toLocaleDateString([], { weekday: 'short' })
    return `${day} ${timeStr}`
  }
  const monthDay = date.toLocaleDateString([], { month: 'short', day: 'numeric' })
  return `${monthDay}, ${timeStr}`
}

/** Calculate auto-resize height for a textarea based on content.
 *  Returns a pixel height between minRows and maxRows line heights. */
export function calcTextareaHeight(
  text: string,
  lineHeight: number = 24,
  minRows: number = 1,
  maxRows: number = 8,
  paddingY: number = 24
): number {
  const lineCount = (text.match(/\n/g)?.length ?? 0) + 1
  const clampedLines = Math.min(Math.max(lineCount, minRows), maxRows)
  return clampedLines * lineHeight + paddingY
}

/** Try to parse a JSON content array (multimodal message).
 *  Returns null if not a valid content array. */
export function parseContentArray(
  content: string
): Array<{ type: string; text?: string; image_url?: { url: string } }> | null {
  if (!content.startsWith('[')) return null
  try {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed) && parsed.length > 0 && parsed.every(p => p && typeof p.type === 'string')) return parsed
  } catch { /* not JSON */ }
  return null
}

/** Extract text-only content from a message, handling both plain text and multimodal JSON. */
export function extractTextContent(content: string): string {
  const parts = parseContentArray(content)
  if (parts) {
    return parts
      .filter(p => p.type === 'text' && p.text)
      .map(p => p.text ?? '')
      .join('\n')
  }
  return content
}

/** Decide which metrics fields to display.
 *  Returns an array of { label, value, title } for rendering. */
export interface MetricItem {
  label: string
  value: string
  title: string
  dimmed?: boolean
}

export interface MessageMetrics {
  tokenCount: number
  promptTokens?: number
  cachedTokens?: number
  tokensPerSecond: string
  ppSpeed?: string
  ttft: string
  totalTime?: string
  elapsed?: string
}

export function getMetricsItems(metrics: MessageMetrics, isStreaming: boolean): MetricItem[] {
  const items: MetricItem[] = []

  items.push({
    label: `${metrics.tokenCount} tokens`,
    value: `${metrics.tokenCount}`,
    title: isStreaming ? 'Tokens generated so far' : 'Completion tokens',
  })

  items.push({
    label: `${metrics.tokensPerSecond} t/s`,
    value: metrics.tokensPerSecond,
    title: 'Generation speed',
  })

  if (metrics.ppSpeed) {
    items.push({
      label: `${metrics.ppSpeed} pp/s`,
      value: metrics.ppSpeed,
      title: 'Prompt processing speed',
    })
  }

  if (metrics.promptTokens && metrics.promptTokens > 0) {
    const cached = metrics.cachedTokens ? ` (${metrics.cachedTokens} cached)` : ''
    items.push({
      label: `${metrics.promptTokens} prompt${cached}`,
      value: `${metrics.promptTokens}`,
      title: 'Prompt tokens processed',
      dimmed: true,
    })
  }

  if (metrics.ttft && parseFloat(metrics.ttft) > 0) {
    items.push({
      label: `${metrics.ttft}s TTFT`,
      value: metrics.ttft,
      title: 'Time to first token',
      dimmed: !isStreaming,
    })
  }

  if (isStreaming && metrics.elapsed) {
    items.push({
      label: `${metrics.elapsed}s`,
      value: metrics.elapsed,
      title: 'Elapsed time',
    })
  }

  if (!isStreaming && metrics.totalTime) {
    items.push({
      label: `${metrics.totalTime}s total`,
      value: metrics.totalTime,
      title: 'Total request time',
    })
  }

  return items
}

/** Determine whether to show the scroll-to-bottom button.
 *  Visible when user has scrolled up more than `threshold` pixels from bottom. */
export function shouldShowScrollButton(
  scrollHeight: number,
  scrollTop: number,
  clientHeight: number,
  threshold: number = 200
): boolean {
  return scrollHeight - scrollTop - clientHeight > threshold
}

/** Classify drag event files to determine which types are valid image drops.
 *  Returns { hasImages, imageCount, totalCount } for UI feedback. */
export function classifyDropFiles(
  types: readonly string[],
  items?: DataTransferItemList
): { hasImages: boolean; imageCount: number; totalCount: number } {
  if (!items) {
    // DataTransfer types only (no item access during dragover)
    const hasImages = types.some(t => t.startsWith('image/')) || types.includes('Files')
    return { hasImages, imageCount: 0, totalCount: 0 }
  }
  const all = Array.from(items)
  const images = all.filter(i => i.type.startsWith('image/'))
  return { hasImages: images.length > 0, imageCount: images.length, totalCount: all.length }
}

// ─── Tool Call Utilities ────────────────────────────────────────────────────

/** Truncate a string to max length, appending ellipsis. */
export function truncateStr(s: string, max: number): string {
  if (!s) return ''
  return s.length > max ? s.slice(0, max) + '\u2026' : s
}

/** Parse JSON args string into object, returns null on failure. */
export function parseToolArgs(detail?: string): Record<string, any> | null {
  if (!detail) return null
  try { return JSON.parse(detail) } catch { return null }
}

/** Pretty-print a JSON string. Returns original on parse failure. */
export function formatJson(s: string): string {
  try { return JSON.stringify(JSON.parse(s), null, 2) } catch { return s }
}

/** Get a human-readable summary for a tool call header. */
export function getToolSummary(
  name: string,
  args: Record<string, any> | null
): { label: string; context: string } {
  if (!args) return { label: name, context: '' }

  switch (name) {
    case 'edit_file':
      return { label: 'Edit', context: args.path || '' }
    case 'read_file':
      return { label: 'Read', context: args.path ? `${args.path}${args.offset ? `:${args.offset}` : ''}` : '' }
    case 'write_file':
      return { label: 'Write', context: args.path || '' }
    case 'patch_file':
      return { label: 'Patch', context: args.path || '' }
    case 'batch_edit':
      return { label: 'Batch Edit', context: args.path || '' }
    case 'insert_text':
      return { label: 'Insert', context: args.path ? `${args.path}:${args.line || ''}` : '' }
    case 'replace_lines':
      return { label: 'Replace Lines', context: args.path ? `${args.path}:${args.start_line}-${args.end_line}` : '' }
    case 'run_command':
      return { label: '$', context: truncateStr(args.command, 80) }
    case 'git':
      return { label: '$ git', context: truncateStr(args.command, 80) }
    case 'search_files':
      return { label: 'Search', context: `"${truncateStr(args.pattern, 40)}"${args.path && args.path !== '.' ? ` in ${args.path}` : ''}` }
    case 'find_files':
      return { label: 'Find', context: `"${truncateStr(args.pattern, 40)}"${args.path && args.path !== '.' ? ` in ${args.path}` : ''}` }
    case 'list_directory':
      return { label: 'List', context: args.path || '.' }
    case 'get_tree':
      return { label: 'Tree', context: args.path || '.' }
    case 'delete_file':
      return { label: 'Delete', context: args.path || '' }
    case 'move_file':
      return { label: 'Move', context: args.source && args.destination ? `${args.source} \u2192 ${args.destination}` : args.source || '' }
    case 'copy_file':
      return { label: 'Copy', context: args.source && args.destination ? `${args.source} \u2192 ${args.destination}` : args.source || '' }
    case 'create_directory':
      return { label: 'Mkdir', context: args.path || '' }
    case 'file_info':
      return { label: 'Info', context: args.path || '' }
    case 'diff_files':
      return { label: 'Diff', context: args.path_a && args.path_b ? `${args.path_a} \u2194 ${args.path_b}` : args.path_a || args.path_b || '' }
    case 'apply_regex':
      return { label: 'Regex', context: args.path || args.glob || '' }
    case 'web_search':
    case 'ddg_search':
      return { label: 'Search Web', context: `"${truncateStr(args.query, 50)}"` }
    case 'fetch_url':
      return { label: 'Fetch', context: truncateStr(args.url, 60) }
    case 'spawn_process':
      return { label: 'Spawn', context: truncateStr(args.command, 60) }
    case 'get_process_output':
      return { label: 'Process Output', context: `PID ${args.pid}` }
    case 'ask_user':
      return { label: 'Ask User', context: truncateStr(args.question, 50) }
    case 'read_image':
      return { label: 'Read Image', context: args.path || '' }
    case 'count_tokens':
      return { label: 'Count Tokens', context: '' }
    case 'clipboard_read':
      return { label: 'Clipboard Read', context: '' }
    case 'clipboard_write':
      return { label: 'Clipboard Write', context: '' }
    case 'get_current_datetime':
      return { label: 'Datetime', context: '' }
    case 'get_diagnostics':
      return { label: 'Diagnostics', context: args.path || '' }
    default:
      return { label: name, context: '' }
  }
}

// ─── Empty State Utilities ──────────────────────────────────────────────────

/** Determine which empty state to show based on chat/session context. */
export type EmptyStateType =
  | 'no-session'      // No sessions configured at all
  | 'model-stopped'   // Session exists but model not running
  | 'no-chat'         // No chat selected
  | 'no-messages'     // Chat selected but empty
  | 'ready'           // Everything is set up, ready to chat

export function getEmptyStateType(opts: {
  chatId: string | null
  messageCount: number
  sessionId?: string
  sessionEndpoint?: { host: string; port: number }
  sessionCount: number
}): EmptyStateType {
  if (opts.sessionCount === 0) return 'no-session'
  if (!opts.chatId) return 'no-chat'
  if (opts.messageCount === 0) {
    if (!opts.sessionEndpoint && opts.sessionId) return 'model-stopped'
    return 'no-messages'
  }
  return 'ready'
}

/** Group chat list items by relative date period. */
export function groupChatsByDate<T extends { updatedAt: number }>(
  items: T[],
  now?: number
): Array<{ label: string; items: T[] }> {
  const d = now ? new Date(now) : new Date()
  const today = new Date(d); today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1)
  const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)
  const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)

  const groups: Record<string, T[]> = {
    'Today': [],
    'Yesterday': [],
    'This Week': [],
    'This Month': [],
    'Older': [],
  }

  for (const item of items) {
    const ts = item.updatedAt
    if (ts >= today.getTime()) groups['Today'].push(item)
    else if (ts >= yesterday.getTime()) groups['Yesterday'].push(item)
    else if (ts >= weekAgo.getTime()) groups['This Week'].push(item)
    else if (ts >= monthAgo.getTime()) groups['This Month'].push(item)
    else groups['Older'].push(item)
  }

  return Object.entries(groups)
    .filter(([_, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }))
}
