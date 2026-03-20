import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import DOMPurify from 'dompurify'
import { useState, useMemo, useCallback, memo } from 'react'
import { Copy, Check, User, Sparkles } from 'lucide-react'
import { ReasoningBox } from './ReasoningBox'
import { ToolCallStatus } from './ToolCallStatus'
import { InlineToolCall, InlineToolGroup } from './InlineToolCall'
import { TTSPlayer } from './VoiceChat'
import { formatTimestamp, parseContentArray, getMetricsItems, type MessageMetrics } from './chat-utils'

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
  sessionId?: string
  sessionEndpoint?: { host: string; port: number }
}

// Custom renderer: wraps code blocks with a header bar (language label + copy button)
const renderer = new marked.Renderer()
renderer.code = (code, lang) => {
  let highlighted: string
  if (lang && hljs.getLanguage(lang)) {
    highlighted = hljs.highlight(code, { language: lang }).value
  } else {
    highlighted = hljs.highlightAuto(code).value
  }
  const headerHtml = `<div class="code-header"><span class="code-lang">${lang || 'code'}</span><button class="code-copy-btn">Copy</button></div>`
  return `<div class="code-block-wrapper">${headerHtml}<pre><code class="hljs language-${lang || 'plaintext'}">${highlighted}</code></pre></div>`
}

// Configure marked with code highlighting and custom renderer
marked.setOptions({
  renderer,
  breaks: true,
  gfm: true
})

/**
 * Sanitize HTML using DOMPurify — allows safe markdown output, blocks XSS.
 * All user/model content passes through this before being rendered via innerHTML.
 */
function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_TAGS: ['pre', 'code'],
    ADD_ATTR: ['class']
  })
}

/** Group tool statuses into tool call groups. Each 'calling' phase starts a new group.
 *  Also extracts contentOffset for inline positioning. */
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

/** Prose classes for rendered markdown */
const proseClasses = 'prose prose-invert max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5 [&_h1]:text-lg [&_h2]:text-base [&_h3]:text-sm [&_pre]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-muted-foreground [&_table]:text-sm [&_th]:text-left [&_th]:font-medium [&_th]:px-3 [&_th]:py-1.5 [&_td]:px-3 [&_td]:py-1.5'

export const MessageBubble = memo(function MessageBubble({ message, isStreaming, metrics, reasoningContent, reasoningDone, toolStatuses, sessionId, sessionEndpoint }: MessageBubbleProps) {
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
    const code = btn.closest('.code-block-wrapper')?.querySelector('code')
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

  // Render a DOMPurify-sanitized markdown segment
  const renderMarkdownSegment = useCallback((text: string, key: string) => {
    if (!text) return null
    const safeHtml = sanitizeHtml(marked.parse(text) as string)
    return (
      <div
        key={key}
        className={proseClasses}
        dangerouslySetInnerHTML={{ __html: safeHtml }}
        onClick={handleProseClick}
      />
    )
  }, [handleProseClick])

  const renderUserContent = () => {
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
                  className="max-w-[300px] max-h-[200px] rounded-md border border-white/10 cursor-pointer hover:opacity-90 transition-opacity object-contain"
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

  const renderInlineContent = () => {
    if (!message.content && (!toolGroups || toolGroups.groups.length === 0)) return null

    const content = message.content || ''

    if (toolGroups && toolGroups.hasOffsets && toolGroups.groups.length > 0) {
      const elements: JSX.Element[] = []

      for (let i = 0; i < toolGroups.groups.length; i++) {
        elements.push(
          <InlineToolCall
            key={`tool-${i}`}
            group={toolGroups.groups[i]}
            isStreaming={!!isStreaming}
          />
        )
      }

      if (toolGroups.processingStatus && isStreaming) {
        const isGenerating = toolGroups.processingStatus.phase === 'generating'
        elements.push(
          <div key="processing" className="flex items-center gap-2 text-muted-foreground text-xs py-1">
            <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${isGenerating ? 'bg-primary' : 'bg-warning'}`} />
            <span>{isGenerating ? 'Generating tool call...' : 'Processing tool results...'}</span>
          </div>
        )
      }

      if (content.trim()) {
        elements.push(renderMarkdownSegment(content, 'seg-content') as JSX.Element)
      }

      return <>{elements}</>
    }

    if (!content.trim()) return null
    const safeHtml = sanitizeHtml(marked.parse(content) as string)
    return (
      <div
        className={proseClasses}
        dangerouslySetInnerHTML={{ __html: safeHtml }}
        onClick={handleProseClick}
      />
    )
  }

  const renderMetrics = () => {
    if (message.role !== 'assistant') return null

    if (metrics) {
      const items = getMetricsItems(metrics, !!isStreaming)
      return (
        <div className="mt-3 pt-2 border-t border-border/30 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground/70">
          {isStreaming && (
            <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
          )}
          {items.map((item, i) => (
            <span key={i} title={item.title} className={item.dimmed ? 'opacity-60' : ''}>
              {item.label}
            </span>
          ))}
        </div>
      )
    }

    if (message.tokens) {
      return (
        <div className="mt-2 text-[11px] text-muted-foreground/60">
          {message.tokens} tokens
        </div>
      )
    }

    return null
  }

  const isUser = message.role === 'user'

  // ─── User message: compact right-aligned bubble ───────────────────
  if (isUser) {
    return (
      <div className="flex justify-end gap-2.5 ml-[5%] md:ml-[10%] lg:ml-[15%]">
        <div className="flex flex-col items-end max-w-full">
          <div className="bg-primary text-primary-foreground rounded-2xl rounded-br-md px-4 py-2.5 text-sm">
            {renderUserContent()}
          </div>
          <span className="text-[10px] text-muted-foreground/50 mt-1 mr-1">
            {formatTimestamp(message.timestamp)}
          </span>
        </div>
        <div className="w-7 h-7 rounded-full bg-primary/15 flex items-center justify-center flex-shrink-0 mt-0.5">
          <User className="h-3.5 w-3.5 text-primary" />
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
  }

  // ─── Assistant message: left-aligned with avatar ───────────────
  return (
    <div className="flex gap-2.5 group mr-[5%] md:mr-[10%] lg:mr-[15%]">
      <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
      </div>
      <div className="flex-1 min-w-0">
        {/* Header: timestamp + actions */}
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] text-muted-foreground/50">
            {formatTimestamp(message.timestamp)}
          </span>
          {!isStreaming && message.content && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <TTSPlayer text={message.content} endpoint={sessionEndpoint} sessionId={sessionId} />
              <button
                onClick={() => copyToClipboard(message.content)}
                className="text-muted-foreground/50 hover:text-foreground transition-colors"
                title="Copy response"
              >
                {copied
                  ? <Check className="h-3 w-3 text-success" />
                  : <Copy className="h-3 w-3" />
                }
              </button>
            </div>
          )}
        </div>

        {/* Reasoning box */}
        {reasoningContent &&
         !(message.content && reasoningContent.trim() === message.content.trim()) && (
          <ReasoningBox
            content={reasoningContent}
            isStreaming={!!isStreaming}
            isDone={reasoningDone ?? false}
          />
        )}

        {/* Main content */}
        <div className="text-sm">
          {renderInlineContent()}
        </div>

        {/* Legacy tool call display */}
        {toolStatuses && toolStatuses.length > 0 &&
         !(toolGroups && toolGroups.hasOffsets) && (
          <ToolCallStatus statuses={toolStatuses} isStreaming={!!isStreaming} />
        )}

        {/* Typing indicator */}
        {isStreaming && !message.content && !reasoningContent && !(toolStatuses && toolStatuses.length > 0) && (
          <div className="flex items-center gap-2 text-muted-foreground text-sm py-1">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          </div>
        )}

        {renderMetrics()}
      </div>
    </div>
  )
})
