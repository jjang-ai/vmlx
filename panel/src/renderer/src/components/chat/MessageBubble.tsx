import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import DOMPurify from 'dompurify'
import { useState, useMemo, useCallback, memo } from 'react'
import { ReasoningBox } from './ReasoningBox'
import { ToolCallStatus } from './ToolCallStatus'
import { InlineToolCall, InlineToolGroup } from './InlineToolCall'
import { TTSPlayer } from './VoiceChat'

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

interface Message {
  id: string
  role: 'system' | 'user' | 'assistant'
  content: string
  timestamp: number
  tokens?: number
}

interface MessageBubbleProps {
  message: Message
  isStreaming?: boolean
  metrics?: MessageMetrics | null
  reasoningContent?: string
  reasoningDone?: boolean
  toolStatuses?: any[]
}

// Custom renderer: wraps code blocks with a copy button
const renderer = new marked.Renderer()
renderer.code = (code, lang) => {
  let highlighted: string
  if (lang && hljs.getLanguage(lang)) {
    highlighted = hljs.highlight(code, { language: lang }).value
  } else {
    highlighted = hljs.highlightAuto(code).value
  }
  const langLabel = lang ? `<span style="position:absolute;top:6px;left:12px;font-size:11px;color:#8b90a0;user-select:none">${lang}</span>` : ''
  return `<pre>${langLabel}<button class="code-copy-btn">Copy</button><code class="hljs language-${lang || 'plaintext'}">${highlighted}</code></pre>`
}

// Configure marked with code highlighting and custom renderer
marked.setOptions({
  renderer,
  breaks: true,
  gfm: true
})

/** Sanitize HTML using DOMPurify — allows safe markdown output, blocks XSS */
function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_TAGS: ['pre', 'code'],
    ADD_ATTR: ['class']
  })
}

/** Try to parse a JSON content array (multimodal message). Returns null if not a content array. */
function parseContentArray(content: string): Array<{ type: string; text?: string; image_url?: { url: string } }> | null {
  if (!content.startsWith('[')) return null
  try {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].type) return parsed
  } catch { /* not JSON */ }
  return null
}

/** Group tool statuses into tool call groups. Each 'calling' phase starts a new group.
 *  Also extracts contentOffset for inline positioning. */
function groupToolStatuses(statuses: any[]): { groups: InlineToolGroup[]; hasOffsets: boolean; processingStatus?: any; doneStatus?: any } {
  const groups: InlineToolGroup[] = []
  let current: InlineToolGroup | null = null
  let hasOffsets = false
  let processingStatus: any = null
  let doneStatus: any = null

  for (const s of statuses) {
    if (s.phase === 'calling') {
      current = { name: s.toolName, statuses: [s], contentOffset: s.contentOffset }
      if (s.contentOffset !== undefined) hasOffsets = true
      groups.push(current)
    } else if (s.phase === 'processing') {
      processingStatus = s
      current = null
    } else if (s.phase === 'done') {
      doneStatus = s
      current = null
    } else if (current) {
      current.statuses.push(s)
    }
  }

  return { groups, hasOffsets, processingStatus, doneStatus }
}

export const MessageBubble = memo(function MessageBubble({ message, isStreaming, metrics, reasoningContent, reasoningDone, toolStatuses }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const [zoomedImage, setZoomedImage] = useState<string | null>(null)

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Event delegation for code-copy buttons (DOMPurify strips onclick attributes)
  const handleProseClick = useCallback((e: React.MouseEvent) => {
    const btn = (e.target as HTMLElement).closest('.code-copy-btn') as HTMLElement | null
    if (!btn) return
    const code = btn.closest('pre')?.querySelector('code')
    if (code) {
      navigator.clipboard.writeText(code.textContent || '')
      btn.textContent = 'Copied!'
      setTimeout(() => { btn.textContent = 'Copy' }, 1500)
    }
  }, [])

  // Group tool statuses for inline rendering
  const toolGroups = useMemo(() => {
    if (!toolStatuses || toolStatuses.length === 0) return null
    return groupToolStatuses(toolStatuses)
  }, [toolStatuses])

  // Render markdown segment from a content substring
  const renderMarkdownSegment = useCallback((text: string, key: string) => {
    if (!text) return null
    const html = sanitizeHtml(marked.parse(text) as string)
    return (
      <div
        key={key}
        className="prose prose-invert max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all"
        dangerouslySetInnerHTML={{ __html: html }}
        onClick={handleProseClick}
      />
    )
  }, [handleProseClick])

  const renderUserContent = () => {
    // Check for multimodal content (images + text stored as JSON content array)
    const contentParts = parseContentArray(message.content)
    if (contentParts) {
      const images = contentParts.filter(p => p.type === 'image_url' && p.image_url?.url)
      const textParts = contentParts.filter(p => p.type === 'text' && p.text)
      return (
        <div>
          {images.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {images.map((img, i) => (
                <img
                  key={i}
                  src={img.image_url!.url}
                  alt={`Attached image ${i + 1}`}
                  className="max-w-[300px] max-h-[200px] rounded border border-primary-foreground/20 cursor-pointer hover:opacity-90 transition-opacity object-contain"
                  onClick={() => setZoomedImage(img.image_url!.url)}
                />
              ))}
            </div>
          )}
          {textParts.map((p, i) => (
            <p key={i} className="whitespace-pre-wrap">{p.text}</p>
          ))}
        </div>
      )
    }
    return <p className="whitespace-pre-wrap">{message.content}</p>
  }

  /** Render assistant content with inline tool calls interleaved at their content offsets */
  const renderInlineContent = () => {
    if (!message.content && (!toolGroups || toolGroups.groups.length === 0)) return null

    const content = message.content || ''

    // If we have tool groups with content offsets, split content and interleave
    if (toolGroups && toolGroups.hasOffsets && toolGroups.groups.length > 0) {
      // Sort groups by contentOffset ascending
      const sorted = [...toolGroups.groups]
        .filter(g => (g as any).contentOffset !== undefined)
        .sort((a, b) => ((a as any).contentOffset || 0) - ((b as any).contentOffset || 0))

      if (sorted.length > 0) {
        const elements: JSX.Element[] = []
        let lastOffset = 0

        for (let i = 0; i < sorted.length; i++) {
          const group = sorted[i]
          const offset = (group as any).contentOffset || 0

          // Render text segment before this tool call
          if (offset > lastOffset) {
            const segment = content.slice(lastOffset, offset).trim()
            if (segment) {
              elements.push(renderMarkdownSegment(segment, `seg-${i}`) as JSX.Element)
            }
          }

          // Render inline tool call
          elements.push(
            <InlineToolCall
              key={`tool-${i}`}
              group={group}
              isStreaming={!!isStreaming}
            />
          )

          lastOffset = Math.max(lastOffset, offset)
        }

        // Render remaining text after last tool call
        if (lastOffset < content.length) {
          const remaining = content.slice(lastOffset).trim()
          if (remaining) {
            elements.push(renderMarkdownSegment(remaining, 'seg-last') as JSX.Element)
          }
        }

        // Show processing/done status
        if (toolGroups.processingStatus && isStreaming) {
          elements.push(
            <div key="processing" className="flex items-center gap-2 text-muted-foreground text-xs py-1">
              <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse" />
              <span>Processing tool results...</span>
            </div>
          )
        }

        return <>{elements}</>
      }
    }

    // Fallback: render all content then tool calls at bottom (legacy behavior)
    if (!content) return null
    const html = sanitizeHtml(marked.parse(content) as string)
    return (
      <div
        className="prose prose-invert max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all"
        dangerouslySetInnerHTML={{ __html: html }}
        onClick={handleProseClick}
      />
    )
  }

  const renderMetrics = () => {
    if (message.role !== 'assistant') return null

    // Show live metrics during streaming
    if (isStreaming && metrics) {
      return (
        <div className="mt-3 pt-2 border-t border-border/50 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
            {metrics.tokenCount} tokens
          </span>
          <span title="Generation speed">{metrics.tokensPerSecond} t/s</span>
          {metrics.ppSpeed && <span title="Prompt processing speed">{metrics.ppSpeed} pp/s</span>}
          {metrics.ttft && parseFloat(metrics.ttft) > 0 && (
            <span title="Time to first token" className="opacity-70">{metrics.ttft}s TTFT</span>
          )}
          {metrics.promptTokens && metrics.promptTokens > 0 && (
            <span title="Prompt tokens processed" className="opacity-70">{metrics.promptTokens} prompt</span>
          )}
          {metrics.elapsed && <span>{metrics.elapsed}s</span>}
        </div>
      )
    }

    // Show final metrics for completed messages
    if (metrics && !isStreaming) {
      return (
        <div className="mt-3 pt-2 border-t border-border/50 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span title="Completion tokens">{metrics.tokenCount} tokens</span>
          <span title="Generation speed (tokens/second)">{metrics.tokensPerSecond} t/s</span>
          {metrics.ppSpeed && (
            <span title="Prompt processing speed (prompt tokens/second)">{metrics.ppSpeed} pp/s</span>
          )}
          {metrics.promptTokens && metrics.promptTokens > 0 && (
            <span title="Prompt tokens processed by the model" className="opacity-70">
              {metrics.promptTokens} prompt{metrics.cachedTokens ? ` (${metrics.cachedTokens} cached)` : ''}
            </span>
          )}
          {metrics.ttft && parseFloat(metrics.ttft) > 0 && (
            <span title="Time to first token">{metrics.ttft}s TTFT</span>
          )}
          {metrics.totalTime && <span title="Total request time">{metrics.totalTime}s total</span>}
        </div>
      )
    }

    // Fallback to just token count if no metrics
    if (message.tokens) {
      return (
        <div className="mt-2 text-xs text-muted-foreground">
          {message.tokens} tokens
        </div>
      )
    }

    return null
  }

  return (
    <div
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-[80%] rounded-lg p-4 ${
          message.role === 'user'
            ? 'bg-primary text-primary-foreground'
            : 'bg-card border border-border'
        }`}
      >
        <div className="flex items-start justify-between gap-4 mb-2">
          <span className="text-sm font-medium">
            {message.role === 'user' ? '$ you' : '> assistant'}
          </span>

          {message.role === 'assistant' && !isStreaming && (
            <div className="flex items-center gap-1">
              <TTSPlayer text={message.content} />
              <button
                onClick={() => copyToClipboard(message.content)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                {copied ? 'copied' : 'copy'}
              </button>
            </div>
          )}
        </div>

        {/* Collapsible reasoning box — hide when content matches reasoning
            (server fallback copies reasoning→content when model has no </think>) */}
        {message.role === 'assistant' && reasoningContent &&
         !(message.content && reasoningContent.trim() === message.content.trim()) && (
          <ReasoningBox
            content={reasoningContent}
            isStreaming={!!isStreaming}
            isDone={reasoningDone ?? false}
          />
        )}

        {message.role === 'user' && renderUserContent()}

        {/* Inline tool calls: interleave content segments with tool call widgets */}
        {message.role === 'assistant' && renderInlineContent()}

        {/* Fallback: legacy tool call display at bottom (no contentOffset data) */}
        {message.role === 'assistant' && toolStatuses && toolStatuses.length > 0 &&
         !(toolGroups && toolGroups.hasOffsets) && (
          <ToolCallStatus statuses={toolStatuses} isStreaming={!!isStreaming} />
        )}

        {isStreaming && !message.content && !reasoningContent && !(toolStatuses && toolStatuses.length > 0) && (
          <div className="flex items-center gap-2 text-muted-foreground text-sm py-1">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
            <span>Thinking...</span>
          </div>
        )}

        {renderMetrics()}
      </div>

      {/* Image zoom overlay */}
      {zoomedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center cursor-pointer"
          onClick={() => setZoomedImage(null)}
        >
          <img
            src={zoomedImage}
            alt="Zoomed"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"
            onClick={e => e.stopPropagation()}
          />
          <button
            onClick={() => setZoomedImage(null)}
            className="absolute top-4 right-4 text-white/80 hover:text-white text-2xl font-bold"
          >
            x
          </button>
        </div>
      )}
    </div>
  )
})
