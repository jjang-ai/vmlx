import { useState, useEffect } from 'react'

interface ReasoningBoxProps {
  content: string
  isStreaming: boolean
  isDone: boolean
}

export function ReasoningBox({ content, isStreaming, isDone }: ReasoningBoxProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Auto-expand when streaming starts
  useEffect(() => {
    if (isStreaming && !isDone) {
      setIsCollapsed(false)
    }
  }, [isStreaming, isDone])

  // Auto-collapse when reasoning ends and content starts
  useEffect(() => {
    if (isDone && !isStreaming) {
      const timer = setTimeout(() => setIsCollapsed(true), 1000)
      return () => clearTimeout(timer)
    }
    return undefined
  }, [isDone, isStreaming])

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
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full px-3 py-2 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <span className="text-[10px] transition-transform duration-150" style={{ transform: isCollapsed ? 'rotate(0deg)' : 'rotate(90deg)' }}>
          &#9654;
        </span>
        <span className="font-medium">
          {label}
          {isStreaming && !isDone && (
            <span className="inline-flex ml-1">
              <span className="animate-pulse">...</span>
            </span>
          )}
        </span>
        <span className="ml-auto text-[10px] opacity-60">{content.length} chars</span>
      </button>

      {!isCollapsed && (
        <div
          className="px-3 py-2 border-t border-border text-xs text-muted-foreground whitespace-pre-wrap max-h-[300px] overflow-y-auto"
          style={{ lineHeight: '1.6' }}
        >
          {content}
          {isStreaming && !isDone && (
            <span className="inline-block w-1.5 h-3.5 bg-primary/60 animate-pulse ml-0.5 align-text-bottom" />
          )}
        </div>
      )}
    </div>
  )
}
