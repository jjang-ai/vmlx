import { useState, useEffect } from 'react'
import { MessageList } from './MessageList'
import { InputBox, ImageAttachment } from './InputBox'
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
  toolCallsJson?: string
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
  sessionId?: string
}

export function ChatInterface({ chatId, onNewChat, sessionEndpoint, sessionId }: ChatInterfaceProps) {
  const { showToast } = useToast()
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [currentMetrics, setCurrentMetrics] = useState<MessageMetrics | null>(null)
  // Reasoning state: track per-message reasoning content and done status
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({})
  const [reasoningDoneMap, setReasoningDoneMap] = useState<Record<string, boolean>>({})
  // Tool call status: track per-message tool call phases
  const [toolStatusMap, setToolStatusMap] = useState<Record<string, Array<{ phase: string; toolName: string; detail?: string; iteration?: number; contentOffset?: number; timestamp: number }>>>({})
  // Per-chat setting: hide tool status display
  const [hideToolStatus, setHideToolStatus] = useState(false)
  // ask_user tool: question from model and input state
  const [askUserQuestion, setAskUserQuestion] = useState<string | null>(null)
  const [askUserInput, setAskUserInput] = useState('')

  // Load messages and set up stream listeners when chat changes
  useEffect(() => {
    if (!chatId) {
      setMessages([])
      return
    }

    // Load existing messages (hydrate persisted metrics, tool calls, reasoning)
    window.api.chat.getMessages(chatId).then(msgs => {
      setMessages(hydrateMessages(msgs))
      // Hydrate tool status map from persisted tool_calls_json
      const restoredTools: Record<string, any[]> = {}
      const restoredReasoning: Record<string, string> = {}
      const restoredReasoningDone: Record<string, boolean> = {}
      for (const m of msgs) {
        if (m.toolCallsJson) {
          try {
            const parsed = JSON.parse(m.toolCallsJson)
            if (Array.isArray(parsed) && parsed.length > 0) {
              restoredTools[m.id] = parsed.map((s: any) => ({
                ...s,
                timestamp: s.timestamp || m.timestamp
              }))
            }
          } catch { /* ignore bad json */ }
        }
        if (m.reasoningContent) {
          restoredReasoning[m.id] = m.reasoningContent
          restoredReasoningDone[m.id] = true
        }
      }
      if (Object.keys(restoredTools).length > 0) {
        setToolStatusMap(restoredTools)
      }
      if (Object.keys(restoredReasoning).length > 0) {
        setReasoningMap(restoredReasoning)
        setReasoningDoneMap(restoredReasoningDone)
      }
    })

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
      // Append truncation warning if server indicated max_tokens was hit
      let finalContent = data.content || ''
      if (data.finishReason === 'length' && finalContent) {
        finalContent += '\n\n---\n*[Output truncated — max tokens reached. Increase "Default Max Tokens" in session settings or send a follow-up message to continue.]*'
      }
      setMessages(prev => prev.map(m =>
        m.id === data.messageId
          ? {
            ...m,
            content: finalContent || m.content,
            tokens: data.metrics?.tokenCount,
            metrics: data.metrics
          }
          : m
      ))
      // Finalize reasoning state from completion event (ensures reasoning box persists
      // even if chat:reasoningDone was missed due to event ordering)
      if (data.reasoningContent) {
        setReasoningMap(prev => ({ ...prev, [data.messageId]: data.reasoningContent }))
        setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: true }))
      }
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
            contentOffset: data.contentOffset,
            timestamp: Date.now()
          }
        ]
      }))
    }

    // ask_user tool: model asks user a question mid-tool-loop
    const handleAskUser = (data: any) => {
      if (data.chatId !== chatId) return
      setAskUserQuestion(data.question)
      setAskUserInput('')
    }

    // Store individual cleanup functions (avoids removeAllListeners race conditions)
    const cleanupTyping = window.api.chat.onTyping(handleTyping)
    const cleanupStream = window.api.chat.onStream(handleStream)
    const cleanupComplete = window.api.chat.onComplete(handleComplete)
    const cleanupReasoningDone = window.api.chat.onReasoningDone(handleReasoningDone)
    const cleanupToolStatus = window.api.chat.onToolStatus(handleToolStatus)
    const cleanupAskUser = window.api.chat.onAskUser(handleAskUser)

    return () => {
      // Do NOT abort active generation when navigating away — the user explicitly
      // wants generation to continue in the background. Only clean up event listeners.
      // The abort button in InputBox handles explicit user cancellation.
      cleanupTyping()
      cleanupStream()
      cleanupComplete()
      cleanupReasoningDone()
      cleanupToolStatus()
      cleanupAskUser()
      setReasoningMap({})
      setReasoningDoneMap({})
      setToolStatusMap({})
      setAskUserQuestion(null)
    }
  }, [chatId])

  const handleAbort = async () => {
    if (!chatId) return
    try {
      await window.api.chat.abort(chatId)
    } catch (err) {
      console.error('Failed to abort:', err)
    }
    // Immediately clear UI state — don't wait for sendMessage IPC to complete.
    // The background handler will finish cleanup (DB save, etc.) independently.
    setLoading(false)
    setStreamingMessageId(null)
    setCurrentMetrics(null)
    setAskUserQuestion(null)
  }

  const handleSend = async (content: string, attachments?: ImageAttachment[]) => {
    if (!chatId || (!content.trim() && (!attachments || attachments.length === 0))) return

    setLoading(true)
    setStreamingMessageId(null)
    setCurrentMetrics(null)

    // Build display content for user message: if images attached, store as JSON content array
    const displayContent = attachments && attachments.length > 0
      ? JSON.stringify([
        ...attachments.map(a => ({ type: 'image_url', image_url: { url: a.dataUrl } })),
        ...(content.trim() ? [{ type: 'text', text: content }] : [])
      ])
      : content

    // Add temp user message for instant UI feedback
    const tempId = `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`
    const tempUserMessage: Message = {
      id: tempId,
      chatId,
      role: 'user',
      content: displayContent,
      timestamp: Date.now()
    }
    setMessages(prev => [...prev, tempUserMessage])

    try {
      // sendMessage persists user msg to DB and streams assistant response.
      // Returns: assistant message object (success or abort with content), or null (abort before content).
      // Only throws on real errors (timeout, connection lost, API errors).
      const result = await window.api.chat.sendMessage(chatId, content, sessionEndpoint, attachments)
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
              return { ...streamedAssistant, tokens: m.tokens, metrics: m.metrics || streamedAssistant.metrics, metricsJson: m.metricsJson, toolCallsJson: m.toolCallsJson, reasoningContent: m.reasoningContent }
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
        sessionId={sessionId}
        sessionEndpoint={sessionEndpoint}
      />
      {/* ask_user tool: inline question from model */}
      {askUserQuestion && chatId && (
        <div className="border-t border-border bg-card px-4 py-3">
          <div className="max-w-2xl mx-auto">
            <div className="text-xs font-medium text-primary mb-1.5">Model is asking:</div>
            <div className="text-sm mb-2 whitespace-pre-wrap">{askUserQuestion}</div>
            <form onSubmit={e => {
              e.preventDefault()
              if (!askUserInput.trim()) return
              window.api.chat.answerUser(chatId, askUserInput.trim())
              setAskUserQuestion(null)
              setAskUserInput('')
            }} className="flex gap-2">
              <input
                type="text"
                value={askUserInput}
                onChange={e => setAskUserInput(e.target.value)}
                placeholder="Type your answer..."
                autoFocus
                className="flex-1 px-3 py-1.5 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <button
                type="submit"
                disabled={!askUserInput.trim()}
                className="px-4 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
              >
                Reply
              </button>
              <button
                type="button"
                onClick={() => {
                  window.api.chat.answerUser(chatId, '(User skipped this question)')
                  setAskUserQuestion(null)
                  setAskUserInput('')
                }}
                className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent"
              >
                Skip
              </button>
            </form>
          </div>
        </div>
      )}
      <InputBox
        onSend={handleSend}
        onAbort={handleAbort}
        disabled={loading}
        loading={loading}
        sessionEndpoint={sessionEndpoint}
        sessionId={sessionId}
      />
    </div>
  )
}
