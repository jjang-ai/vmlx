import { useState, useEffect } from 'react'
import { MessageList } from './MessageList'
import { InputBox } from './InputBox'
import { useToast } from '../Toast'

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
  chatId: string
  role: 'system' | 'user' | 'assistant'
  content: string
  timestamp: number
  tokens?: number
  metrics?: MessageMetrics
  metricsJson?: string
  reasoningContent?: string
  reasoningDone?: boolean
}

/** Hydrate metrics from DB metricsJson field */
function hydrateMessages(msgs: Message[]): Message[] {
  return msgs.map(m => {
    if (m.metricsJson && !m.metrics) {
      try {
        return { ...m, metrics: JSON.parse(m.metricsJson) }
      } catch { /* ignore bad json */ }
    }
    return m
  })
}

interface ChatInterfaceProps {
  chatId: string | null
  onNewChat?: () => void
  sessionEndpoint?: { host: string; port: number }
}

export function ChatInterface({ chatId, onNewChat, sessionEndpoint }: ChatInterfaceProps) {
  const { showToast } = useToast()
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [currentMetrics, setCurrentMetrics] = useState<MessageMetrics | null>(null)
  // Reasoning state: track per-message reasoning content and done status
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({})
  const [reasoningDoneMap, setReasoningDoneMap] = useState<Record<string, boolean>>({})
  // Tool call status: track per-message tool call phases
  const [toolStatusMap, setToolStatusMap] = useState<Record<string, Array<{ phase: string; toolName: string; detail?: string; iteration?: number; timestamp: number }>>>({})
  // Per-chat setting: hide tool status display
  const [hideToolStatus, setHideToolStatus] = useState(false)

  // Load messages and set up stream listeners when chat changes
  useEffect(() => {
    if (!chatId) {
      setMessages([])
      return
    }

    // Load existing messages (hydrate persisted metrics)
    window.api.chat.getMessages(chatId).then(msgs => setMessages(hydrateMessages(msgs)))

    // Load hideToolStatus from chat overrides
    window.api.chat.getOverrides(chatId).then((o: any) => {
      setHideToolStatus(o?.hideToolStatus ?? false)
    })

    // Typing indicator: model is processing, waiting for first token
    const handleTyping = (data: any) => {
      if (data.chatId !== chatId) return
      setStreamingMessageId(data.messageId)
      // Add placeholder assistant message so the typing indicator renders
      setMessages(prev => {
        if (prev.find(m => m.id === data.messageId)) return prev
        return [...prev, {
          id: data.messageId,
          chatId: data.chatId,
          role: 'assistant' as const,
          content: '',
          timestamp: Date.now()
        }]
      })
    }

    const handleStream = (data: any) => {
      if (data.chatId !== chatId) return
      setStreamingMessageId(data.messageId)
      if (data.metrics) setCurrentMetrics(data.metrics)

      if (data.isReasoning) {
        // Track reasoning content separately
        setReasoningMap(prev => ({
          ...prev,
          [data.messageId]: data.fullContent
        }))
        // Ensure the message exists in the list (for rendering reasoning box)
        setMessages(prev => {
          const existing = prev.find(m => m.id === data.messageId)
          if (!existing) {
            return [...prev, {
              id: data.messageId,
              chatId: data.chatId,
              role: 'assistant' as const,
              content: '',
              timestamp: Date.now(),
              metrics: data.metrics
            }]
          }
          return prev.map(m =>
            m.id === data.messageId ? { ...m, metrics: data.metrics } : m
          )
        })
        return
      }

      // Regular content update
      setMessages(prev => {
        const existing = prev.find(m => m.id === data.messageId)
        if (existing) {
          return prev.map(m =>
            m.id === data.messageId
              ? { ...m, content: data.fullContent, metrics: data.metrics }
              : m
          )
        }
        return [...prev, {
          id: data.messageId,
          chatId: data.chatId,
          role: 'assistant' as const,
          content: data.fullContent,
          timestamp: Date.now(),
          metrics: data.metrics
        }]
      })
    }

    const handleComplete = (data: any) => {
      if (data.chatId !== chatId) return
      setMessages(prev => prev.map(m =>
        m.id === data.messageId
          ? {
              ...m,
              // Use provided content if available (abort saves final content with [Generation interrupted])
              content: data.content || m.content,
              tokens: data.metrics?.tokenCount,
              metrics: data.metrics
            }
          : m
      ))
      setStreamingMessageId(null)
      setCurrentMetrics(null)
    }

    const handleReasoningDone = (data: any) => {
      if (data.chatId !== chatId) return
      setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: true }))
      // Also store the final reasoning content
      if (data.reasoningContent) {
        setReasoningMap(prev => ({ ...prev, [data.messageId]: data.reasoningContent }))
      }
    }

    const handleToolStatus = (data: any) => {
      if (data.chatId !== chatId) return
      setToolStatusMap(prev => ({
        ...prev,
        [data.messageId]: [
          ...(prev[data.messageId] || []),
          {
            phase: data.phase,
            toolName: data.toolName || '',
            detail: data.detail,
            iteration: data.iteration,
            timestamp: Date.now()
          }
        ]
      }))
    }

    // Store individual cleanup functions (avoids removeAllListeners race conditions)
    const cleanupTyping = window.api.chat.onTyping(handleTyping)
    const cleanupStream = window.api.chat.onStream(handleStream)
    const cleanupComplete = window.api.chat.onComplete(handleComplete)
    const cleanupReasoningDone = window.api.chat.onReasoningDone(handleReasoningDone)
    const cleanupToolStatus = window.api.chat.onToolStatus(handleToolStatus)

    return () => {
      // Abort any active generation when switching chats to prevent stuck locks
      if (chatId) {
        window.api.chat.abort(chatId).catch(() => {})
      }
      cleanupTyping()
      cleanupStream()
      cleanupComplete()
      cleanupReasoningDone()
      cleanupToolStatus()
      setReasoningMap({})
      setReasoningDoneMap({})
      setToolStatusMap({})
    }
  }, [chatId])

  const handleAbort = async () => {
    if (!chatId) return
    try {
      await window.api.chat.abort(chatId)
    } catch (err) {
      console.error('Failed to abort:', err)
    }
  }

  const handleSend = async (content: string) => {
    if (!chatId || !content.trim()) return

    setLoading(true)
    setStreamingMessageId(null)
    setCurrentMetrics(null)

    // Add temp user message for instant UI feedback
    const tempId = `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`
    const tempUserMessage: Message = {
      id: tempId,
      chatId,
      role: 'user',
      content,
      timestamp: Date.now()
    }
    setMessages(prev => [...prev, tempUserMessage])

    try {
      // sendMessage persists user msg to DB and streams assistant response.
      // Returns: assistant message object (success or abort with content), or null (abort before content).
      // Only throws on real errors (timeout, connection lost, API errors).
      const result = await window.api.chat.sendMessage(chatId, content, sessionEndpoint)
      const assistantId = result?.id

      // Replace the temp user message with the real one from DB, but keep
      // the streamed assistant message in place to avoid a full re-render
      // that causes stutter at end of generation.
      const freshMessages = await window.api.chat.getMessages(chatId)
      setMessages(prev => {
        const streamedAssistant = assistantId
          ? prev.find(m => m.id === assistantId && m.role === 'assistant')
          : null
        if (streamedAssistant) {
          const hydrated = hydrateMessages(freshMessages)
          return hydrated.map(m => {
            if (m.id === streamedAssistant.id) {
              return { ...streamedAssistant, tokens: m.tokens, metrics: m.metrics || streamedAssistant.metrics, metricsJson: m.metricsJson }
            }
            return m
          })
        }
        return hydrateMessages(freshMessages)
      })
    } catch (error: any) {
      console.error('Failed to send message:', error)
      const msg = error?.message || 'Unknown error'
      showToast('error', 'Message failed', msg)
      // Reload messages from DB to restore consistent state
      const freshMessages = await window.api.chat.getMessages(chatId)
      if (freshMessages.length > 0) {
        setMessages(hydrateMessages(freshMessages))
      } else {
        setMessages(prev => prev.filter(m => m.id !== tempId))
      }
    } finally {
      setLoading(false)
      setStreamingMessageId(null)
    }
  }

  if (!chatId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="text-4xl mb-6 text-primary font-bold">$_</div>
          <h2 className="text-2xl font-bold mb-2">vMLX Chat</h2>
          <p className="text-muted-foreground mb-6">
            Start a conversation with your local LLM. Make sure you have a model loaded first.
          </p>
          {onNewChat && (
            <button
              onClick={onNewChat}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 font-medium"
            >
              + New Chat
            </button>
          )}
          <p className="text-xs text-muted-foreground mt-4">
            Or click the menu in the sidebar to see existing chats
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <MessageList
        messages={messages}
        streamingMessageId={streamingMessageId}
        currentMetrics={currentMetrics}
        reasoningMap={reasoningMap}
        reasoningDoneMap={reasoningDoneMap}
        toolStatusMap={toolStatusMap}
        hideToolStatus={hideToolStatus}
      />
      <InputBox
        onSend={handleSend}
        onAbort={handleAbort}
        disabled={loading}
        loading={loading}
      />
    </div>
  )
}
