import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { ArrowDown, MessageCircle } from 'lucide-react'
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
  const [showScrollBtn, setShowScrollBtn] = useState(false)

  // Track whether user is near the bottom of the chat.
  // Threshold pair (auto-pause @ 80, button-show @ 60) gives a small hysteresis
  // so the scroll-to-bottom button doesn't flicker on/off when content reflows
  // by a few pixels during streaming. The earlier (100, 200) pair left a 100-px
  // dead zone where auto-scroll was paused but the button was still hidden,
  // which made the chat look frozen.
  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    isNearBottomRef.current = distFromBottom < 80
    setShowScrollBtn(distFromBottom > 60)
  }, [])

  // Wheel/touch up = explicit user intent to scroll-break. Setting
  // isNearBottomRef=false BEFORE the scroll event fires prevents an in-flight
  // streaming chunk from re-pinning the user back to bottom on the next eval.
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    if (e.deltaY < 0) {
      isNearBottomRef.current = false
      setShowScrollBtn(true)
    }
  }, [])

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    isNearBottomRef.current = true
    setShowScrollBtn(false)
  }, [])

  // Auto-scroll to bottom when new messages arrive, BUT only if user is near
  // bottom OR the message count increased (user just sent a new turn).
  // This lets users scroll up to read earlier content without being yanked
  // back during streaming. 'auto' (instant) during streaming avoids
  // smooth-scroll stutter; 'smooth' only when a brand-new message appears.
  const prevMsgCountRef = useRef(messages.length)
  // Derive a cheap change signal from reasoning/tool maps without deep-comparing objects
  const reasoningVersion = useMemo(() => reasoningMap ? Object.values(reasoningMap).reduce((n, s) => n + s.length, 0) : 0, [reasoningMap])
  const toolStatusVersion = useMemo(() => toolStatusMap ? Object.values(toolStatusMap).reduce((n, arr) => n + arr.length, 0) : 0, [toolStatusMap])
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
        <div className="text-center">
          <MessageCircle className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
          <p className="text-sm text-muted-foreground/60">
            Send a message to start the conversation
          </p>
          <p className="text-xs text-muted-foreground/40 mt-1">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative flex-1 overflow-hidden">
      <div ref={containerRef} onScroll={handleScroll} onWheel={handleWheel} className="h-full overflow-y-auto overflow-x-hidden px-6 py-6 space-y-5 w-full">
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

      {/* Scroll-to-bottom button — appears when user scrolls up */}
      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 p-2 rounded-full bg-card border border-border shadow-lg hover:bg-accent transition-all text-muted-foreground hover:text-foreground"
          title="Scroll to bottom"
        >
          <ArrowDown className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}
