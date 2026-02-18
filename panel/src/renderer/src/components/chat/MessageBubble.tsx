import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import DOMPurify from 'dompurify'
import { useState, useMemo, useRef, useCallback, memo } from 'react'
import { ReasoningBox } from './ReasoningBox'
import { ToolCallStatus } from './ToolCallStatus'

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

export const MessageBubble = memo(function MessageBubble({ message, isStreaming, metrics, reasoningContent, reasoningDone, toolStatuses }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const proseRef = useRef<HTMLDivElement>(null)

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Always render markdown — including during streaming so formatting stays
  // consistent and doesn't "break" when generation completes. marked.parse()
  // is fast enough (<5ms) for typical streaming rates.
  const renderedHtml = useMemo(() => {
    if (message.role !== 'assistant' || !message.content) return null
    return sanitizeHtml(marked.parse(message.content) as string)
  }, [message.role, message.content])

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

  const renderContent = () => {
    if (message.role === 'user') {
      return <p className="whitespace-pre-wrap">{message.content}</p>
    }

    if (!message.content) return null

    // Render markdown for all assistant messages (streaming and completed)
    if (renderedHtml) {
      return (
        <div
          ref={proseRef}
          className="prose prose-invert max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all"
          dangerouslySetInnerHTML={{ __html: renderedHtml }}
          onClick={handleProseClick}
        />
      )
    }

    return null
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
            <button
              onClick={() => copyToClipboard(message.content)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              {copied ? 'copied' : 'copy'}
            </button>
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

        {renderContent()}

        {/* Tool call status indicators */}
        {message.role === 'assistant' && toolStatuses && toolStatuses.length > 0 && (
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
    </div>
  )
})
