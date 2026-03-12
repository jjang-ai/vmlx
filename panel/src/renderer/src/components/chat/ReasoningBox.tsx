import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { ChevronRight, Maximize2, Minimize2 } from 'lucide-react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

interface ReasoningBoxProps {
  content: string
  isStreaming: boolean
  isDone: boolean
}

/** Sanitize HTML via DOMPurify — same config as MessageBubble for XSS safety */
function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_TAGS: ['pre', 'code'],
    ADD_ATTR: ['class']
  })
}

export function ReasoningBox({ content, isStreaming, isDone }: ReasoningBoxProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const userScrolledUp = useRef(false)
  const userToggledRef = useRef(false)

  // Auto-expand when streaming starts (reset user toggle tracking for new stream)
  useEffect(() => {
    if (isStreaming && !isDone) {
      userToggledRef.current = false
      setIsCollapsed(false)
    }
  }, [isStreaming, isDone])

  // Auto-collapse when reasoning ends — skip if user manually toggled or maximized
  useEffect(() => {
    if (isDone && !isStreaming && !isMaximized && !userToggledRef.current) {
      const timer = setTimeout(() => setIsCollapsed(true), 1000)
      return () => clearTimeout(timer)
    }
    return undefined
  }, [isDone, isStreaming, isMaximized])

  // Auto-scroll to bottom when new content arrives (unless user scrolled up)
  useEffect(() => {
    if (!userScrolledUp.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [content])

  // Reset scroll tracking when streaming starts
  useEffect(() => {
    if (isStreaming && !isDone) {
      userScrolledUp.current = false
    }
  }, [isStreaming, isDone])

  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    const atBottom = scrollHeight - scrollTop - clientHeight < 30
    userScrolledUp.current = !atBottom
  }

  // Render markdown with code highlighting (reuses global marked config from MessageBubble)
  const renderedHtml = useMemo(() => {
    if (!content) return ''
    return sanitizeHtml(marked.parse(content) as string)
  }, [content])

  // Handle copy button clicks inside code blocks (same as MessageBubble)
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

  if (!content) return null

  const label = isStreaming && !isDone ? 'Thinking' : 'Reasoning'

  return (
    <div className={`mb-3 rounded border overflow-hidden transition-all duration-200 ${
      isStreaming && !isDone
        ? 'border-primary/40 border-l-primary border-l-2'
        : 'border-border'
    } bg-popover`}
    >
      <button
        onClick={() => { userToggledRef.current = true; setIsCollapsed(!isCollapsed) }}
        className="w-full px-3 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <ChevronRight className={`h-3.5 w-3.5 transition-transform duration-150 ${isCollapsed ? '' : 'rotate-90'}`} />
        <span className="font-medium">
          {label}
          {isStreaming && !isDone && (
            <span className="inline-flex ml-1">
              <span className="animate-pulse">...</span>
            </span>
          )}
        </span>
        <span className="ml-auto flex items-center gap-2">
          <span className="text-[10px] opacity-60">{content.length} chars</span>
          <span
            role="button"
            onClick={(e) => { e.stopPropagation(); setIsMaximized(!isMaximized) }}
            className="text-[10px] opacity-40 hover:opacity-80 transition-opacity cursor-pointer"
            title={isMaximized ? 'Restore size' : 'Maximize'}
          >
            {isMaximized ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
          </span>
        </span>
      </button>

      {!isCollapsed && (
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className={`px-3 py-2 border-t border-border text-xs text-muted-foreground overflow-y-auto ${isMaximized ? '' : 'max-h-[300px]'}`}
          style={{ lineHeight: '1.6' }}
        >
          <div
            className="prose prose-invert prose-xs max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all [&_pre]:text-[11px] [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_pre]:my-1.5 [&_pre]:rounded [&_blockquote]:my-1"
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
            onClick={handleProseClick}
          />
          {isStreaming && !isDone && (
            <span className="inline-block w-1.5 h-3.5 bg-primary/60 animate-pulse ml-0.5 align-text-bottom" />
          )}
        </div>
      )}
    </div>
  )
}
