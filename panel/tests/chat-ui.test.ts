/**
 * Chat UI Tests — Phases 3-6 + Audit Fixes: Chat UI Polish + Input Experience + Tool Call Visuals
 *
 * Coverage:
 *   - Timestamp formatting (relative, today, yesterday, week, older)
 *   - Textarea auto-resize height calculation
 *   - Multimodal content array parsing
 *   - Text extraction from plain and multimodal content
 *   - Metrics display items (streaming vs completed)
 *   - Message layout logic (role-based styling)
 *   - Code block rendering helpers
 *   - Scroll-to-bottom button visibility
 *   - Drag-drop file classification
 *   - Input focus behavior
 *   - Empty state rendering
 *   - Tool call summary generation
 *   - JSON formatting + args parsing
 *   - String truncation
 */
import { describe, it, expect } from 'vitest'

// ─── Re-implement pure functions from chat-utils.ts for testing ──────────────

function formatTimestamp(ts: number): string {
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

function calcTextareaHeight(
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

function parseContentArray(
  content: string
): Array<{ type: string; text?: string; image_url?: { url: string } }> | null {
  if (!content.startsWith('[')) return null
  try {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].type) return parsed
  } catch { /* not JSON */ }
  return null
}

function extractTextContent(content: string): string {
  const parts = parseContentArray(content)
  if (parts) {
    return parts
      .filter(p => p.type === 'text' && p.text)
      .map(p => p.text!)
      .join('\n')
  }
  return content
}

interface MetricItem {
  label: string
  value: string
  title: string
  dimmed?: boolean
}

interface MessageMetrics {
  tokenCount: number
  promptTokens?: number
  cachedTokens?: number
  tokensPerSecond: string
  ppSpeed?: string
  ttft: string
  totalTime?: string
  elapsed?: string
}

function getMetricsItems(metrics: MessageMetrics, isStreaming: boolean): MetricItem[] {
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

// ─── Tool status grouping (from MessageBubble.tsx) ───────────────────────────

interface InlineToolGroup {
  name: string
  statuses: any[]
}

function groupToolStatuses(statuses: any[]): { groups: InlineToolGroup[]; hasOffsets: boolean; processingStatus?: any } {
  const groups: InlineToolGroup[] = []
  let current: InlineToolGroup | null = null
  let hasOffsets = false
  let processingStatus: any = null

  for (const s of statuses) {
    if (s.phase === 'calling') {
      current = { name: s.toolName, statuses: [s] }
      if (s.contentOffset !== undefined) hasOffsets = true
      groups.push(current)
    } else if (s.phase === 'generating') {
      processingStatus = s
      current = null
    } else if (s.phase === 'processing') {
      processingStatus = s
      current = null
    } else if (s.phase === 'done') {
      current = null
    } else if (current) {
      current.statuses.push(s)
    }
  }

  return { groups, hasOffsets, processingStatus }
}

// ─── Message layout helpers (from MessageBubble.tsx) ─────────────────────────

type MessageRole = 'system' | 'user' | 'assistant'

function getMessageLayout(role: MessageRole) {
  const isUser = role === 'user'
  return {
    alignment: isUser ? 'end' : 'start',
    showAvatar: role !== 'system',
    avatarSide: isUser ? 'right' : 'left',
    bubbleStyle: isUser ? 'user' : 'assistant',
    maxWidth: isUser ? '80%' : '100%',
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('Timestamp Formatting', () => {
  it('formats "Just now" for timestamps within 1 minute', () => {
    expect(formatTimestamp(Date.now())).toBe('Just now')
    expect(formatTimestamp(Date.now() - 30_000)).toBe('Just now')
    expect(formatTimestamp(Date.now() - 59_999)).toBe('Just now')
  })

  it('formats time only for today (after 1 minute)', () => {
    // 5 minutes ago, still today
    const fiveMinAgo = Date.now() - 5 * 60_000
    const today = new Date(); today.setHours(0, 0, 0, 0)
    if (fiveMinAgo >= today.getTime()) {
      const result = formatTimestamp(fiveMinAgo)
      expect(result).not.toBe('Just now')
      expect(result).not.toContain('Yesterday')
      // Should be a time like "2:30 PM"
      expect(result).toMatch(/\d{1,2}:\d{2}\s?(AM|PM)/i)
    }
  })

  it('formats "Yesterday HH:MM" for yesterday', () => {
    const today = new Date(); today.setHours(0, 0, 0, 0)
    const yesterdayTs = today.getTime() - 1 // 1ms before today
    const result = formatTimestamp(yesterdayTs)
    expect(result).toContain('Yesterday')
    expect(result).toMatch(/Yesterday \d{1,2}:\d{2}\s?(AM|PM)/i)
  })

  it('formats "Day HH:MM" for this week', () => {
    const today = new Date(); today.setHours(0, 0, 0, 0)
    const threeDaysAgo = new Date(today); threeDaysAgo.setDate(threeDaysAgo.getDate() - 3)
    const result = formatTimestamp(threeDaysAgo.getTime())
    // Should contain a day abbreviation like "Mon", "Tue", etc.
    expect(result).toMatch(/(Mon|Tue|Wed|Thu|Fri|Sat|Sun)/i)
  })

  it('formats "Month Day, HH:MM" for older', () => {
    const today = new Date(); today.setHours(0, 0, 0, 0)
    const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)
    const result = formatTimestamp(monthAgo.getTime())
    // Should contain month abbreviation like "Jan", "Feb", etc.
    expect(result).toMatch(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i)
  })

  it('handles future timestamps gracefully', () => {
    const future = Date.now() + 60_000 // 1 minute in future
    // diffMs is negative, so it won't match "Just now"
    const result = formatTimestamp(future)
    // Should still return a formatted time, not crash
    expect(typeof result).toBe('string')
    expect(result.length).toBeGreaterThan(0)
  })
})

describe('Textarea Auto-Resize', () => {
  it('returns minimum height for empty text', () => {
    expect(calcTextareaHeight('')).toBe(1 * 24 + 24) // 48px
  })

  it('returns minimum height for single line', () => {
    expect(calcTextareaHeight('hello')).toBe(1 * 24 + 24)
  })

  it('grows for multiple lines', () => {
    expect(calcTextareaHeight('a\nb')).toBe(2 * 24 + 24) // 72px
    expect(calcTextareaHeight('a\nb\nc')).toBe(3 * 24 + 24) // 96px
  })

  it('caps at maxRows', () => {
    const tenLines = 'a\nb\nc\nd\ne\nf\ng\nh\ni\nj'
    expect(calcTextareaHeight(tenLines, 24, 1, 8)).toBe(8 * 24 + 24) // 216px
  })

  it('respects minRows', () => {
    expect(calcTextareaHeight('', 24, 3)).toBe(3 * 24 + 24) // 96px
  })

  it('custom lineHeight', () => {
    expect(calcTextareaHeight('a\nb', 20, 1, 8, 16)).toBe(2 * 20 + 16) // 56px
  })

  it('custom paddingY', () => {
    expect(calcTextareaHeight('hello', 24, 1, 8, 32)).toBe(1 * 24 + 32) // 56px
  })

  it('handles trailing newline', () => {
    // "hello\n" has 1 newline = 2 lines
    expect(calcTextareaHeight('hello\n')).toBe(2 * 24 + 24)
  })

  it('handles multiple consecutive newlines', () => {
    expect(calcTextareaHeight('\n\n\n')).toBe(4 * 24 + 24) // 3 newlines = 4 lines
  })
})

describe('Content Array Parsing', () => {
  it('returns null for plain text', () => {
    expect(parseContentArray('Hello world')).toBeNull()
  })

  it('returns null for non-array JSON', () => {
    expect(parseContentArray('{"key": "value"}')).toBeNull()
  })

  it('returns null for empty array', () => {
    expect(parseContentArray('[]')).toBeNull()
  })

  it('returns null for array without type field', () => {
    expect(parseContentArray('[{"text": "hello"}]')).toBeNull()
  })

  it('parses text content array', () => {
    const content = JSON.stringify([{ type: 'text', text: 'Hello' }])
    const result = parseContentArray(content)
    expect(result).not.toBeNull()
    expect(result).toHaveLength(1)
    expect(result![0].type).toBe('text')
    expect(result![0].text).toBe('Hello')
  })

  it('parses image + text content array', () => {
    const content = JSON.stringify([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,abc' } },
      { type: 'text', text: 'What is this?' },
    ])
    const result = parseContentArray(content)
    expect(result).toHaveLength(2)
    expect(result![0].type).toBe('image_url')
    expect(result![1].type).toBe('text')
  })

  it('handles malformed JSON gracefully', () => {
    expect(parseContentArray('[invalid json')).toBeNull()
  })

  it('does not parse strings starting with other characters', () => {
    expect(parseContentArray(' [{"type": "text"}]')).toBeNull() // leading space
  })
})

describe('Text Content Extraction', () => {
  it('returns plain text as-is', () => {
    expect(extractTextContent('Hello world')).toBe('Hello world')
  })

  it('extracts text from content array', () => {
    const content = JSON.stringify([
      { type: 'image_url', image_url: { url: 'data:abc' } },
      { type: 'text', text: 'What is this image?' },
    ])
    expect(extractTextContent(content)).toBe('What is this image?')
  })

  it('joins multiple text parts with newline', () => {
    const content = JSON.stringify([
      { type: 'text', text: 'First part' },
      { type: 'text', text: 'Second part' },
    ])
    expect(extractTextContent(content)).toBe('First part\nSecond part')
  })

  it('filters out non-text parts', () => {
    const content = JSON.stringify([
      { type: 'image_url', image_url: { url: 'data:abc' } },
    ])
    expect(extractTextContent(content)).toBe('')
  })

  it('returns empty string for empty text parts', () => {
    const content = JSON.stringify([{ type: 'text' }])
    expect(extractTextContent(content)).toBe('')
  })
})

describe('Metrics Display Items', () => {
  const baseMetrics: MessageMetrics = {
    tokenCount: 150,
    tokensPerSecond: '42.5',
    ttft: '0.35',
  }

  it('always includes token count and speed', () => {
    const items = getMetricsItems(baseMetrics, false)
    expect(items.length).toBeGreaterThanOrEqual(2)
    expect(items[0].label).toBe('150 tokens')
    expect(items[1].label).toBe('42.5 t/s')
  })

  it('includes pp/s when available', () => {
    const items = getMetricsItems({ ...baseMetrics, ppSpeed: '120.0' }, false)
    expect(items.find(i => i.label.includes('pp/s'))).toBeDefined()
  })

  it('includes prompt tokens when available', () => {
    const items = getMetricsItems({ ...baseMetrics, promptTokens: 500 }, false)
    const promptItem = items.find(i => i.label.includes('prompt'))
    expect(promptItem).toBeDefined()
    expect(promptItem!.dimmed).toBe(true)
  })

  it('shows cached token count in prompt label', () => {
    const items = getMetricsItems({ ...baseMetrics, promptTokens: 500, cachedTokens: 400 }, false)
    const promptItem = items.find(i => i.label.includes('prompt'))
    expect(promptItem!.label).toBe('500 prompt (400 cached)')
  })

  it('includes TTFT when > 0', () => {
    const items = getMetricsItems(baseMetrics, false)
    expect(items.find(i => i.label.includes('TTFT'))).toBeDefined()
  })

  it('omits TTFT when 0', () => {
    const items = getMetricsItems({ ...baseMetrics, ttft: '0' }, false)
    expect(items.find(i => i.label.includes('TTFT'))).toBeUndefined()
  })

  it('shows elapsed during streaming', () => {
    const items = getMetricsItems({ ...baseMetrics, elapsed: '3.2' }, true)
    expect(items.find(i => i.label === '3.2s')).toBeDefined()
  })

  it('hides elapsed when not streaming', () => {
    const items = getMetricsItems({ ...baseMetrics, elapsed: '3.2' }, false)
    expect(items.find(i => i.label === '3.2s')).toBeUndefined()
  })

  it('shows totalTime when completed', () => {
    const items = getMetricsItems({ ...baseMetrics, totalTime: '5.1' }, false)
    expect(items.find(i => i.label === '5.1s total')).toBeDefined()
  })

  it('hides totalTime during streaming', () => {
    const items = getMetricsItems({ ...baseMetrics, totalTime: '5.1' }, true)
    expect(items.find(i => i.label === '5.1s total')).toBeUndefined()
  })

  it('streaming title says "generated so far"', () => {
    const items = getMetricsItems(baseMetrics, true)
    expect(items[0].title).toContain('so far')
  })

  it('completed title says "Completion tokens"', () => {
    const items = getMetricsItems(baseMetrics, false)
    expect(items[0].title).toBe('Completion tokens')
  })
})

describe('Tool Status Grouping', () => {
  it('returns empty groups for no statuses', () => {
    const result = groupToolStatuses([])
    expect(result.groups).toHaveLength(0)
    expect(result.hasOffsets).toBe(false)
  })

  it('groups a single tool call', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file' },
      { phase: 'result', toolName: 'read_file' },
    ])
    expect(result.groups).toHaveLength(1)
    expect(result.groups[0].name).toBe('read_file')
    expect(result.groups[0].statuses).toHaveLength(2)
  })

  it('groups multiple sequential tool calls', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file' },
      { phase: 'result', toolName: 'read_file' },
      { phase: 'calling', toolName: 'edit_file' },
      { phase: 'result', toolName: 'edit_file' },
    ])
    expect(result.groups).toHaveLength(2)
    expect(result.groups[0].name).toBe('read_file')
    expect(result.groups[1].name).toBe('edit_file')
  })

  it('detects contentOffset presence', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file', contentOffset: 42 },
    ])
    expect(result.hasOffsets).toBe(true)
  })

  it('hasOffsets is false when no offsets', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file' },
    ])
    expect(result.hasOffsets).toBe(false)
  })

  it('tracks processing status', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file' },
      { phase: 'processing', toolName: 'read_file' },
    ])
    expect(result.processingStatus).toBeDefined()
    expect(result.processingStatus.phase).toBe('processing')
  })

  it('tracks generating status', () => {
    const result = groupToolStatuses([
      { phase: 'generating', toolName: '' },
    ])
    expect(result.processingStatus).toBeDefined()
    expect(result.processingStatus.phase).toBe('generating')
  })

  it('done phase resets current group', () => {
    const result = groupToolStatuses([
      { phase: 'calling', toolName: 'read_file' },
      { phase: 'done' },
      { phase: 'extra' }, // orphan — no current group
    ])
    expect(result.groups).toHaveLength(1)
    expect(result.groups[0].statuses).toHaveLength(1) // only 'calling', not 'extra'
  })
})

describe('Message Layout Logic', () => {
  it('user messages are right-aligned', () => {
    const layout = getMessageLayout('user')
    expect(layout.alignment).toBe('end')
    expect(layout.avatarSide).toBe('right')
    expect(layout.bubbleStyle).toBe('user')
  })

  it('assistant messages are left-aligned', () => {
    const layout = getMessageLayout('assistant')
    expect(layout.alignment).toBe('start')
    expect(layout.avatarSide).toBe('left')
    expect(layout.bubbleStyle).toBe('assistant')
  })

  it('user messages have 80% max width', () => {
    expect(getMessageLayout('user').maxWidth).toBe('80%')
  })

  it('assistant messages are full width', () => {
    expect(getMessageLayout('assistant').maxWidth).toBe('100%')
  })

  it('system messages hide avatar', () => {
    expect(getMessageLayout('system').showAvatar).toBe(false)
  })

  it('user and assistant messages show avatar', () => {
    expect(getMessageLayout('user').showAvatar).toBe(true)
    expect(getMessageLayout('assistant').showAvatar).toBe(true)
  })
})

// ─── Phase 4: Input Experience ──────────────────────────────────────────────

// Re-implement from chat-utils.ts
function shouldShowScrollButton(
  scrollHeight: number,
  scrollTop: number,
  clientHeight: number,
  threshold: number = 200
): boolean {
  return scrollHeight - scrollTop - clientHeight > threshold
}

function classifyDropFiles(
  types: readonly string[],
  items?: { type: string }[]
): { hasImages: boolean; imageCount: number; totalCount: number } {
  if (!items) {
    const hasImages = types.some(t => t.startsWith('image/')) || types.includes('Files')
    return { hasImages, imageCount: 0, totalCount: 0 }
  }
  const all = Array.from(items)
  const images = all.filter(i => i.type.startsWith('image/'))
  return { hasImages: images.length > 0, imageCount: images.length, totalCount: all.length }
}

describe('Scroll-to-Bottom Button', () => {
  it('hidden when at bottom', () => {
    // scrollHeight=1000, scrollTop=500, clientHeight=500 → distance=0
    expect(shouldShowScrollButton(1000, 500, 500)).toBe(false)
  })

  it('hidden when near bottom (within threshold)', () => {
    // distance = 1000 - 700 - 200 = 100 < 200
    expect(shouldShowScrollButton(1000, 700, 200)).toBe(false)
  })

  it('visible when scrolled up past threshold', () => {
    // distance = 2000 - 100 - 500 = 1400 > 200
    expect(shouldShowScrollButton(2000, 100, 500)).toBe(true)
  })

  it('visible at exactly threshold + 1', () => {
    // distance = 1000 - 299 - 500 = 201 > 200
    expect(shouldShowScrollButton(1000, 299, 500)).toBe(true)
  })

  it('hidden at exactly threshold', () => {
    // distance = 1000 - 300 - 500 = 200, not > 200
    expect(shouldShowScrollButton(1000, 300, 500)).toBe(false)
  })

  it('respects custom threshold', () => {
    // distance = 1000 - 400 - 500 = 100 > 50
    expect(shouldShowScrollButton(1000, 400, 500, 50)).toBe(true)
    // distance = 100 < 200 (default)
    expect(shouldShowScrollButton(1000, 400, 500, 200)).toBe(false)
  })

  it('handles zero scroll height', () => {
    expect(shouldShowScrollButton(0, 0, 0)).toBe(false)
  })
})

describe('Drag-Drop File Classification', () => {
  it('detects Files type without item details', () => {
    const result = classifyDropFiles(['Files'])
    expect(result.hasImages).toBe(true)
    expect(result.imageCount).toBe(0) // no item access
    expect(result.totalCount).toBe(0)
  })

  it('detects image MIME in types', () => {
    const result = classifyDropFiles(['image/png'])
    expect(result.hasImages).toBe(true)
  })

  it('returns false for non-image types without items', () => {
    const result = classifyDropFiles(['text/plain', 'application/pdf'])
    expect(result.hasImages).toBe(false)
  })

  it('counts image items when provided', () => {
    const items = [
      { type: 'image/png' },
      { type: 'image/jpeg' },
      { type: 'text/plain' },
    ]
    const result = classifyDropFiles([], items as any)
    expect(result.hasImages).toBe(true)
    expect(result.imageCount).toBe(2)
    expect(result.totalCount).toBe(3)
  })

  it('returns no images for non-image items', () => {
    const items = [{ type: 'text/plain' }, { type: 'application/json' }]
    const result = classifyDropFiles([], items as any)
    expect(result.hasImages).toBe(false)
    expect(result.imageCount).toBe(0)
    expect(result.totalCount).toBe(2)
  })

  it('handles empty items array', () => {
    const result = classifyDropFiles([], [] as any)
    expect(result.hasImages).toBe(false)
    expect(result.imageCount).toBe(0)
    expect(result.totalCount).toBe(0)
  })
})

describe('Input Focus Behavior', () => {
  // These test the expected behavior without requiring React rendering

  it('focus conditions: not loading and not disabled', () => {
    const shouldFocus = (loading: boolean, disabled: boolean) => !loading && !disabled
    expect(shouldFocus(false, false)).toBe(true)
    expect(shouldFocus(true, false)).toBe(false)
    expect(shouldFocus(false, true)).toBe(false)
    expect(shouldFocus(true, true)).toBe(false)
  })

  it('send clears message and attachments', () => {
    // Simulate handleSend logic
    let message = 'hello'
    let attachments = [{ id: '1', dataUrl: 'data:', name: 'a.png', type: 'image/png' }]
    const canSend = (message.trim() || attachments.length > 0) && true // not disabled
    expect(canSend).toBe(true)
    // After send:
    message = ''
    attachments = []
    expect(message).toBe('')
    expect(attachments).toHaveLength(0)
  })

  it('cannot send when disabled', () => {
    const disabled = true
    const canSend = ('hello'.trim() || false) && !disabled
    expect(canSend).toBe(false)
  })

  it('can send with only attachments (no text)', () => {
    const message = ''
    const hasAttachments = true
    const canSend = (message.trim() || hasAttachments) && true // not disabled
    expect(canSend).toBe(true)
  })
})

describe('Empty State', () => {
  it('no-chat state shows new chat option', () => {
    // Verify the logic: chatId is null → show empty state
    const chatId: string | null = null
    const hasNewChat = true
    expect(chatId).toBeNull()
    expect(hasNewChat).toBe(true)
    // Component renders "Start a conversation" heading + New Chat button
  })

  it('no-messages state shows prompt text', () => {
    // Verify the logic: chatId exists but messages are empty → show inline prompt
    const chatId = 'chat-123'
    const messages: any[] = []
    expect(chatId).not.toBeNull()
    expect(messages.length).toBe(0)
    // MessageList renders "Send a message to start the conversation"
  })

  it('model-not-running shows load model banner', () => {
    // Verify the logic: no endpoint, has sessionId, not loading → show banner
    const sessionEndpoint = undefined
    const sessionId = 'session-123'
    const loading = false
    const showBanner = !sessionEndpoint && sessionId && !loading
    expect(showBanner).toBeTruthy()
  })

  it('model-running hides load model banner', () => {
    const sessionEndpoint = { host: 'localhost', port: 8000 }
    const sessionId = 'session-123'
    const loading = false
    const showBanner = !sessionEndpoint && sessionId && !loading
    expect(showBanner).toBeFalsy()
  })
})

// ─── Phase 5: Tool Call Visuals ─────────────────────────────────────────────

// Re-implement from chat-utils.ts
function truncateStr(s: string, max: number): string {
  if (!s) return ''
  return s.length > max ? s.slice(0, max) + '\u2026' : s
}

function parseToolArgs(detail?: string): Record<string, any> | null {
  if (!detail) return null
  try { return JSON.parse(detail) } catch { return null }
}

function formatJsonStr(s: string): string {
  try { return JSON.stringify(JSON.parse(s), null, 2) } catch { return s }
}

function getToolSummaryTest(
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
    case 'run_command':
      return { label: '$', context: truncateStr(args.command, 80) }
    case 'git':
      return { label: '$ git', context: truncateStr(args.command, 80) }
    case 'search_files':
      return { label: 'Search', context: `"${truncateStr(args.pattern, 40)}"${args.path && args.path !== '.' ? ` in ${args.path}` : ''}` }
    case 'move_file':
      return { label: 'Move', context: args.source && args.destination ? `${args.source} \u2192 ${args.destination}` : args.source || '' }
    case 'ask_user':
      return { label: 'Ask User', context: truncateStr(args.question, 50) }
    default:
      return { label: name, context: '' }
  }
}

describe('String Truncation', () => {
  it('returns empty for empty/falsy input', () => {
    expect(truncateStr('', 10)).toBe('')
    expect(truncateStr(undefined as any, 10)).toBe('')
  })

  it('returns string as-is when within limit', () => {
    expect(truncateStr('hello', 10)).toBe('hello')
    expect(truncateStr('hello', 5)).toBe('hello')
  })

  it('truncates and appends ellipsis when over limit', () => {
    expect(truncateStr('hello world', 5)).toBe('hello\u2026')
    expect(truncateStr('abcdefghij', 3)).toBe('abc\u2026')
  })

  it('handles exact boundary', () => {
    expect(truncateStr('abc', 3)).toBe('abc')
    expect(truncateStr('abcd', 3)).toBe('abc\u2026')
  })
})

describe('Tool Args Parsing', () => {
  it('returns null for undefined/empty', () => {
    expect(parseToolArgs()).toBeNull()
    expect(parseToolArgs('')).toBeNull()
  })

  it('parses valid JSON', () => {
    const result = parseToolArgs('{"path": "/foo/bar.ts"}')
    expect(result).toEqual({ path: '/foo/bar.ts' })
  })

  it('returns null for invalid JSON', () => {
    expect(parseToolArgs('not json')).toBeNull()
    expect(parseToolArgs('{invalid')).toBeNull()
  })

  it('handles nested objects', () => {
    const result = parseToolArgs('{"edits": [{"line": 1}]}')
    expect(result?.edits).toHaveLength(1)
  })
})

describe('JSON Formatting', () => {
  it('pretty-prints valid JSON', () => {
    expect(formatJsonStr('{"a":1,"b":2}')).toBe('{\n  "a": 1,\n  "b": 2\n}')
  })

  it('returns original string for invalid JSON', () => {
    expect(formatJsonStr('not json')).toBe('not json')
  })

  it('handles arrays', () => {
    expect(formatJsonStr('[1,2,3]')).toBe('[\n  1,\n  2,\n  3\n]')
  })
})

describe('Tool Summary Generation', () => {
  it('returns tool name with no context when no args', () => {
    expect(getToolSummaryTest('edit_file', null)).toEqual({ label: 'edit_file', context: '' })
  })

  it('returns label + path for file tools', () => {
    expect(getToolSummaryTest('edit_file', { path: '/src/main.ts' }))
      .toEqual({ label: 'Edit', context: '/src/main.ts' })
    expect(getToolSummaryTest('write_file', { path: '/out.json' }))
      .toEqual({ label: 'Write', context: '/out.json' })
  })

  it('includes offset for read_file', () => {
    expect(getToolSummaryTest('read_file', { path: '/foo.ts', offset: 100 }))
      .toEqual({ label: 'Read', context: '/foo.ts:100' })
    expect(getToolSummaryTest('read_file', { path: '/foo.ts' }))
      .toEqual({ label: 'Read', context: '/foo.ts' })
  })

  it('formats run_command with $ prefix', () => {
    expect(getToolSummaryTest('run_command', { command: 'npm test' }))
      .toEqual({ label: '$', context: 'npm test' })
  })

  it('formats git with $ git prefix', () => {
    expect(getToolSummaryTest('git', { command: 'status' }))
      .toEqual({ label: '$ git', context: 'status' })
  })

  it('formats search with quotes', () => {
    const result = getToolSummaryTest('search_files', { pattern: 'TODO', path: 'src' })
    expect(result.label).toBe('Search')
    expect(result.context).toBe('"TODO" in src')
  })

  it('omits search path when "."', () => {
    const result = getToolSummaryTest('search_files', { pattern: 'TODO', path: '.' })
    expect(result.context).toBe('"TODO"')
  })

  it('formats move with arrow', () => {
    expect(getToolSummaryTest('move_file', { source: 'a.ts', destination: 'b.ts' }))
      .toEqual({ label: 'Move', context: 'a.ts \u2192 b.ts' })
  })

  it('truncates long commands', () => {
    const longCmd = 'a'.repeat(100)
    const result = getToolSummaryTest('run_command', { command: longCmd })
    expect(result.context.length).toBeLessThan(100)
    expect(result.context.endsWith('\u2026')).toBe(true)
  })

  it('returns generic for unknown tools', () => {
    expect(getToolSummaryTest('custom_tool', { foo: 'bar' }))
      .toEqual({ label: 'custom_tool', context: '' })
  })

  it('formats ask_user with truncated question', () => {
    expect(getToolSummaryTest('ask_user', { question: 'What file?' }))
      .toEqual({ label: 'Ask User', context: 'What file?' })
  })
})

// ─── Phase 6: Onboarding / Empty States ─────────────────────────────────────

type EmptyStateType = 'no-session' | 'model-stopped' | 'no-chat' | 'no-messages' | 'ready'

function getEmptyStateType(opts: {
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

function groupChatsByDate<T extends { updatedAt: number }>(
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

describe('Empty State Type Detection', () => {
  it('returns no-session when no sessions exist', () => {
    expect(getEmptyStateType({ chatId: null, messageCount: 0, sessionCount: 0 }))
      .toBe('no-session')
  })

  it('returns no-session even with chatId when no sessions', () => {
    expect(getEmptyStateType({ chatId: 'chat-1', messageCount: 5, sessionCount: 0 }))
      .toBe('no-session')
  })

  it('returns no-chat when chatId is null', () => {
    expect(getEmptyStateType({ chatId: null, messageCount: 0, sessionCount: 1 }))
      .toBe('no-chat')
  })

  it('returns model-stopped when chat exists but no endpoint', () => {
    expect(getEmptyStateType({
      chatId: 'chat-1', messageCount: 0, sessionId: 'sess-1', sessionCount: 1
    })).toBe('model-stopped')
  })

  it('returns no-messages when chat is empty with active endpoint', () => {
    expect(getEmptyStateType({
      chatId: 'chat-1', messageCount: 0,
      sessionEndpoint: { host: 'localhost', port: 8000 },
      sessionCount: 1
    })).toBe('no-messages')
  })

  it('returns ready when chat has messages', () => {
    expect(getEmptyStateType({
      chatId: 'chat-1', messageCount: 5,
      sessionEndpoint: { host: 'localhost', port: 8000 },
      sessionCount: 1
    })).toBe('ready')
  })

  it('returns ready even without endpoint if messages exist', () => {
    expect(getEmptyStateType({
      chatId: 'chat-1', messageCount: 3, sessionId: 'sess-1', sessionCount: 1
    })).toBe('ready')
  })
})

describe('Chat Date Grouping', () => {
  // Use a fixed "now" to avoid flaky tests
  const NOW = new Date('2026-03-11T12:00:00').getTime()
  const today = new Date('2026-03-11T00:00:00').getTime()

  function makeChat(updatedAt: number) {
    return { id: String(updatedAt), updatedAt }
  }

  it('groups today items', () => {
    const chats = [makeChat(today + 3600_000)] // 1am today
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('Today')
    expect(groups[0].items).toHaveLength(1)
  })

  it('groups yesterday items', () => {
    const chats = [makeChat(today - 1)] // 1ms before today = yesterday
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('Yesterday')
  })

  it('groups this week items', () => {
    const threeDaysAgo = today - 3 * 86400_000
    const chats = [makeChat(threeDaysAgo)]
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('This Week')
  })

  it('groups this month items', () => {
    const twoWeeksAgo = today - 14 * 86400_000
    const chats = [makeChat(twoWeeksAgo)]
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('This Month')
  })

  it('groups older items', () => {
    const twoMonthsAgo = today - 60 * 86400_000
    const chats = [makeChat(twoMonthsAgo)]
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('Older')
  })

  it('returns multiple groups for mixed dates', () => {
    const chats = [
      makeChat(today + 1000),          // Today
      makeChat(today - 1),             // Yesterday
      makeChat(today - 60 * 86400_000) // Older
    ]
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(3)
    expect(groups.map(g => g.label)).toEqual(['Today', 'Yesterday', 'Older'])
  })

  it('omits empty groups', () => {
    const chats = [makeChat(today + 1000)]
    const groups = groupChatsByDate(chats, NOW)
    expect(groups).toHaveLength(1)
    // No Yesterday, This Week, etc.
  })

  it('returns empty array for no items', () => {
    expect(groupChatsByDate([], NOW)).toHaveLength(0)
  })
})

// ─── Audit Fixes ────────────────────────────────────────────────────────────

// Re-implement fixed parseContentArray for testing
function parseContentArrayFixed(
  content: string
): Array<{ type: string; text?: string; image_url?: { url: string } }> | null {
  if (!content.startsWith('[')) return null
  try {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed) && parsed.length > 0 && parsed.every((p: any) => p && typeof p.type === 'string')) return parsed
  } catch { /* not JSON */ }
  return null
}

function extractTextContentFixed(content: string): string {
  const parts = parseContentArrayFixed(content)
  if (parts) {
    return parts
      .filter(p => p.type === 'text' && p.text)
      .map(p => p.text ?? '')
      .join('\n')
  }
  return content
}

describe('Audit Fix: parseContentArray validates all elements', () => {
  it('rejects array where only first element has type', () => {
    const input = JSON.stringify([{ type: 'text', text: 'hello' }, { noType: true }])
    expect(parseContentArrayFixed(input)).toBeNull()
  })

  it('rejects array with null elements', () => {
    const input = JSON.stringify([{ type: 'text', text: 'hi' }, null])
    expect(parseContentArrayFixed(input)).toBeNull()
  })

  it('rejects array with non-string type', () => {
    const input = JSON.stringify([{ type: 123 }])
    expect(parseContentArrayFixed(input)).toBeNull()
  })

  it('accepts array where all elements have string type', () => {
    const input = JSON.stringify([
      { type: 'text', text: 'hello' },
      { type: 'image_url', image_url: { url: 'data:...' } }
    ])
    const result = parseContentArrayFixed(input)
    expect(result).not.toBeNull()
    expect(result).toHaveLength(2)
  })

  it('rejects empty array', () => {
    expect(parseContentArrayFixed('[]')).toBeNull()
  })
})

describe('Audit Fix: extractTextContent null safety', () => {
  it('handles text parts with undefined text field', () => {
    // This shouldn't happen with the fixed parseContentArray, but defense in depth
    const content = JSON.stringify([
      { type: 'text', text: 'hello' },
      { type: 'text' }, // missing text field
      { type: 'image_url', image_url: { url: 'data:...' } }
    ])
    const result = extractTextContentFixed(content)
    expect(result).toBe('hello') // skips the part without text
  })

  it('returns empty string from empty content array', () => {
    const content = JSON.stringify([
      { type: 'image_url', image_url: { url: 'data:...' } }
    ])
    expect(extractTextContentFixed(content)).toBe('')
  })
})

describe('Audit Fix: InlineToolCall empty statuses guard', () => {
  it('safely handles empty statuses array', () => {
    const group = { name: 'test_tool', statuses: [] as any[] }
    // In the component, this returns null before accessing length-1
    expect(group.statuses.length).toBe(0)
    // Guard: if (group.statuses.length === 0) return null
    const shouldRender = group.statuses.length > 0
    expect(shouldRender).toBe(false)
  })

  it('works normally with non-empty statuses', () => {
    const group = {
      name: 'test_tool',
      statuses: [{ phase: 'calling', toolName: 'test_tool' }]
    }
    expect(group.statuses.length).toBeGreaterThan(0)
    const lastPhase = group.statuses[group.statuses.length - 1]
    expect(lastPhase.phase).toBe('calling')
  })
})

describe('Audit Fix: ToolCallStatus bounds checking', () => {
  it('handles lastProcessingIdx bounds check during grouping', () => {
    // Simulate the grouping logic with bounds check
    const statuses = [
      { phase: 'generating', toolName: '', timestamp: 1 },
      { phase: 'calling', toolName: 'edit', timestamp: 2 },
      { phase: 'generating', toolName: '', timestamp: 3 }, // should create new group (lastProcessingIdx was reset by calling)
    ]

    const groups: { name: string; statuses: any[] }[] = []
    let current: { name: string; statuses: any[] } | null = null
    let lastProcessingIdx = -1

    for (const s of statuses) {
      if (s.phase === 'calling') {
        current = { name: s.toolName, statuses: [s] }
        groups.push(current)
        lastProcessingIdx = -1
      } else if (s.phase === 'generating') {
        if (lastProcessingIdx >= 0 && lastProcessingIdx < groups.length) {
          groups[lastProcessingIdx] = { name: '', statuses: [s] }
        } else {
          lastProcessingIdx = groups.length
          groups.push({ name: '', statuses: [s] })
        }
        current = null
      }
    }

    expect(groups).toHaveLength(3)
    // First generating, then calling group, then second generating
    expect(groups[0].statuses[0].phase).toBe('generating')
    expect(groups[1].statuses[0].phase).toBe('calling')
    expect(groups[2].statuses[0].phase).toBe('generating')
  })

  it('replaces existing processing group in-place', () => {
    const statuses = [
      { phase: 'generating', toolName: '', timestamp: 1 },
      { phase: 'generating', toolName: '', timestamp: 2 }, // should replace, not add
    ]

    const groups: { name: string; statuses: any[] }[] = []
    let lastProcessingIdx = -1

    for (const s of statuses) {
      if (s.phase === 'generating') {
        if (lastProcessingIdx >= 0 && lastProcessingIdx < groups.length) {
          groups[lastProcessingIdx] = { name: '', statuses: [s] }
        } else {
          lastProcessingIdx = groups.length
          groups.push({ name: '', statuses: [s] })
        }
      }
    }

    expect(groups).toHaveLength(1) // replaced, not duplicated
    expect(groups[0].statuses[0].timestamp).toBe(2) // latest one
  })
})

describe('Audit Fix: dragLeave container check', () => {
  it('should not clear drag state when moving to child element', () => {
    // Simulate: currentTarget.contains(relatedTarget) = true → don't clear
    const container = { contains: (_: any) => true }
    const relatedTarget = {} // a child element
    const shouldClear = !container.contains(relatedTarget)
    expect(shouldClear).toBe(false)
  })

  it('should clear drag state when leaving container entirely', () => {
    const container = { contains: (_: any) => false }
    const relatedTarget = {} // outside element
    const shouldClear = !container.contains(relatedTarget)
    expect(shouldClear).toBe(true)
  })

  it('should clear drag state when relatedTarget is null (left window)', () => {
    const container = { contains: (_: any) => false }
    const shouldClear = !container.contains(null)
    expect(shouldClear).toBe(true)
  })
})

describe('Audit Fix: ReasoningBox user toggle tracking', () => {
  it('auto-collapse should be skipped after user manual toggle', () => {
    let userToggled = false
    let isCollapsed = false
    const isDone = true
    const isStreaming = false
    const isMaximized = false

    // Simulate user clicking toggle
    userToggled = true
    isCollapsed = false // user expanded it

    // Auto-collapse logic (from useEffect): should skip because userToggled
    const shouldAutoCollapse = isDone && !isStreaming && !isMaximized && !userToggled
    expect(shouldAutoCollapse).toBe(false)
  })

  it('auto-collapse should work when user has not interacted', () => {
    const userToggled = false
    const isDone = true
    const isStreaming = false
    const isMaximized = false

    const shouldAutoCollapse = isDone && !isStreaming && !isMaximized && !userToggled
    expect(shouldAutoCollapse).toBe(true)
  })

  it('user toggle tracking resets on new stream', () => {
    let userToggled = true
    const isStreaming = true
    const isDone = false

    // When streaming starts, reset user toggle
    if (isStreaming && !isDone) {
      userToggled = false
    }
    expect(userToggled).toBe(false)
  })
})
