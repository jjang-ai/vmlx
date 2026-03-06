import { useEffect, useRef } from 'react'
import { MessageBubble } from './MessageBubble'

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
  metrics?: MessageMetrics
}

interface MessageListProps {
  messages: Message[]
  streamingMessageId: string | null
  currentMetrics?: MessageMetrics | null
  reasoningMap?: Record<string, string>
  reasoningDoneMap?: Record<string, boolean>
  toolStatusMap?: Record<string, any[]>
  hideToolStatus?: boolean
  sessionId?: string
  sessionEndpoint?: { host: string; port: number }
}

export function MessageList({ messages, streamingMessageId, currentMetrics, reasoningMap, reasoningDoneMap, toolStatusMap, hideToolStatus, sessionId, sessionEndpoint }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const isNearBottomRef = useRef(true)

  // Track whether user is near the bottom of the chat
  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 100
  }

  // Auto-scroll to bottom when new messages arrive, BUT only if user is near bottom.
  // This lets users scroll up to read earlier content without being yanked back.
  // Use 'auto' (instant) during streaming to avoid smooth-scroll stutter,
  // 'smooth' only when a new message appears.
  const prevMsgCountRef = useRef(messages.length)
  // Derive a cheap change signal from reasoning/tool maps without deep-comparing objects
  const reasoningVersion = reasoningMap ? Object.values(reasoningMap).reduce((n, s) => n + s.length, 0) : 0
  const toolStatusVersion = toolStatusMap ? Object.values(toolStatusMap).reduce((n, arr) => n + arr.length, 0) : 0
  useEffect(() => {
    const isNewMessage = messages.length !== prevMsgCountRef.current
    prevMsgCountRef.current = messages.length
    // Always scroll for new messages (user just sent); only scroll during streaming if near bottom
    if (isNewMessage || isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: streamingMessageId && !isNewMessage ? 'auto' : 'smooth' })
    }
  }, [messages, streamingMessageId, reasoningVersion, toolStatusVersion])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-muted-foreground">
          Start a conversation by typing a message below
        </p>
      </div>
    )
  }

  return (
    <div ref={containerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4">
      {messages.map(message => (
        <MessageBubble
          key={message.id}
          message={message}
          isStreaming={message.id === streamingMessageId}
          metrics={message.id === streamingMessageId ? currentMetrics : message.metrics}
          reasoningContent={reasoningMap?.[message.id]}
          reasoningDone={reasoningDoneMap?.[message.id] ?? false}
          toolStatuses={hideToolStatus ? undefined : toolStatusMap?.[message.id]}
          sessionId={sessionId}
          sessionEndpoint={sessionEndpoint}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
