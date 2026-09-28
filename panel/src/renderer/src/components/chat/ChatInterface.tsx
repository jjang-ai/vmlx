import { useState, useEffect, useRef } from 'react'
import { Sparkles, Loader2 } from 'lucide-react'
import { MessageList } from './MessageList'
import { InputBox, MediaAttachment } from './InputBox'
import { useToast } from '../Toast'
import { useTranslation } from '../../i18n'
import { useSessionsContext } from '../../contexts/SessionsContext'
import { formatResidentLoad } from '../sessions/loadProgressFormat'
import { extractResponsesWarnings } from '../../lib/responsesWarnings'
import { restoreUserMessageContent } from './messageReplay'

interface MessageMetrics {
  tokenCount: number
  promptTokens?: number
  cachedTokens?: number
  cacheDetail?: string  // e.g. "paged", "paged+ssm(23)+tq", "disk"
  tokensPerSecond: string
  ppMetricSource?: string
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
  warnings?: string[]
  warningsJson?: string
  toolCallsJson?: string
  reasoningContent?: string
  reasoningSegmentsJson?: string
  reasoningDone?: boolean
}

interface ToolStatusEntry {
  phase: string
  toolName: string
  toolCallId?: string
  detail?: string
  iteration?: number
  contentOffset?: number
  timestamp: number
}

function mergeToolStatusHistory(saved: ToolStatusEntry[], live: ToolStatusEntry[]): ToolStatusEntry[] {
  // The final DB transcript may overlap the suffix observed after a remount.
  // Compare producer identity, not renderer timestamps or truncated DB detail.
  const key = (s: ToolStatusEntry) => JSON.stringify([s.phase, s.toolName, s.toolCallId, s.iteration, s.contentOffset])
  for (let overlap = Math.min(saved.length, live.length); overlap > 0; overlap--) {
    const suffix = saved.slice(-overlap)
    if (suffix.some(s => !!s.toolCallId) && suffix.every((s, i) => key(s) === key(live[i]))) {
      return [...saved.slice(0, -overlap), ...live]
    }
  }
  // Unanchored progress markers are not unique identities. Keep them rather
  // than collapsing repeated events or distinct calls sharing a tool name.
  return [...saved, ...live]
}

/** Hydrate metrics from DB metricsJson field */
function hydrateMessages(msgs: Message[]): Message[] {
  return msgs.map(m => {
    let hydrated: Message = m
    if (m.metricsJson && !m.metrics) {
      try {
        hydrated = { ...hydrated, metrics: JSON.parse(m.metricsJson) }
      } catch { /* ignore bad json */ }
    }
    if (m.warningsJson && !m.warnings) {
      try {
        const warnings = extractResponsesWarnings({ warnings: JSON.parse(m.warningsJson) })
        if (warnings) hydrated = { ...hydrated, warnings }
      } catch { /* ignore bad json */ }
    }
    return hydrated
  })
}

function latestPendingAssistantId(msgs: Message[]): string | null {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    if (
      m.role === 'assistant' &&
      !String(m.content || '').trim() &&
      !String(m.reasoningContent || '').trim() &&
      !m.tokens &&
      !m.metrics
    ) {
      return m.id
    }
  }
  return null
}

function audioFormatFromDataUrl(dataUrl: string): string {
  const mime = dataUrl.match(/^data:([^;,]+)[;,]/)?.[1]?.toLowerCase() || ''
  if (mime === 'audio/mpeg' || mime === 'audio/mp3') return 'mp3'
  if (mime === 'audio/wave' || mime === 'audio/x-wav' || mime === 'audio/wav') return 'wav'
  if (mime === 'audio/mp4' || mime === 'audio/x-m4a') return 'm4a'
  if (mime.startsWith('audio/')) return mime.slice('audio/'.length)
  return 'wav'
}

function audioDataFromDataUrl(dataUrl: string): string {
  return dataUrl.includes(',') ? dataUrl.split(',', 2)[1] : dataUrl
}

function attachmentContentPart(a: MediaAttachment): any {
  if (a.kind === 'audio') {
    return {
      type: 'input_audio',
      input_audio: {
        data: audioDataFromDataUrl(a.dataUrl),
        format: audioFormatFromDataUrl(a.dataUrl),
      },
    }
  }
  if (a.kind === 'text') {
    return {
      type: 'text',
      text: `[Attached file: ${a.name}]\n${a.text ?? ''}`.trim(),
    }
  }
  if (a.kind === 'video') return { type: 'video_url', video_url: { url: a.dataUrl } }
  return { type: 'image_url', image_url: { url: a.dataUrl } }
}

function isExpectedChatDisconnectError(error: any): boolean {
  const code = String(error?.code || '')
  const message = String(error?.message || error || '')
  const cause = error?.cause
  const wrappedDisconnects = [
    cause,
    error?.reason,
    error?.error,
    error?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray(error?.errors) ? error.errors : []
  return (
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedChatDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedChatDisconnectError(nested))
  )
}

function formatChatSendErrorMessage(
  error: any,
  t: (key: string, params?: Record<string, string | number>) => string,
): string {
  if (isExpectedChatDisconnectError(error)) {
    return t('chat.interface.toast.connectionLost')
  }
  // ipcRenderer.invoke rejections arrive wrapped as
  // "Error invoking remote method 'chat:sendMessage': Error: <real message>" —
  // strip the plumbing so the toast reads like a sentence (GH #253).
  return (
    String(error?.message || t('chat.interface.toast.unknownError'))
      .replace(/^Error invoking remote method '[^']+':\s*/i, '')
      .replace(/^Error:\s*/, '')
      .trim() || t('chat.interface.toast.unknownError')
  )
}

interface ChatInterfaceProps {
  chatId: string | null
  onNewChat?: () => void
  sessionEndpoint?: { host: string; port: number }
  sessionId?: string
  sessionStatus?: string
  overridesVersion?: number
}

export function ChatInterface({ chatId, onNewChat, sessionEndpoint, sessionId, sessionStatus, overridesVersion }: ChatInterfaceProps) {
  const { showToast } = useToast()
  const { t } = useTranslation()
  // Live model-load/wake progress for the bound session: a sleeping model
  // woken by a chat message (JIT wake) surfaces here as status 'loading'
  // with resident-RAM progress, and stays visible until the weights are
  // actually in RAM (the settle phase after the server is already serving).
  const { loadProgress } = useSessionsContext()
  const sessionLoadProgress = sessionId ? loadProgress.get(sessionId) : undefined
  const [messages, setMessages] = useState<Message[]>([])
  // Track current chatId via ref so async handleSend can detect stale closures
  const chatIdRef = useRef(chatId)
  chatIdRef.current = chatId

  const [loading, setLoading] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [currentMetrics, setCurrentMetrics] = useState<MessageMetrics | null>(null)
  // Reasoning state: track per-message reasoning content and done status
  const [reasoningMap, setReasoningMap] = useState<Record<string, string>>({})
  const [reasoningSegmentMap, setReasoningSegmentMap] = useState<Record<string, string[]>>({})
  const [reasoningDoneMap, setReasoningDoneMap] = useState<Record<string, boolean>>({})
  const [answerPassMap, setAnswerPassMap] = useState<Record<string, boolean>>({})
  // Tool call status: track per-message tool call phases
  const [toolStatusMap, setToolStatusMap] = useState<Record<string, ToolStatusEntry[]>>({})
  // Per-chat setting: hide tool status display
  const [hideToolStatus, setHideToolStatus] = useState(false)
  // ask_user tool: question from model and input state
  const [askUserQuestion, setAskUserQuestion] = useState<string | null>(null)
  const [askUserInput, setAskUserInput] = useState('')

  // Load messages and set up stream listeners when chat changes
  useEffect(() => {
    if (!chatId) {
      setMessages([])
      setReasoningMap({})
      setReasoningSegmentMap({})
      setReasoningDoneMap({})
      setAnswerPassMap({})
      return
    }

    // Reset streaming state for the new chat — prevents stale loading
    // from the previous chat leaking into this one
    setLoading(false)
    setStreamingMessageId(null)
    setCurrentMetrics(null)

    let disposed = false
    let terminalObserved = false
    const liveMessageIds = new Set<string>()
    const isCurrentChat = () => !disposed && chatIdRef.current === chatId

    // Load existing messages (hydrate persisted metrics, tool calls, reasoning)
    window.api.chat.getMessages(chatId).then(msgs => {
      if (!isCurrentChat()) return
      const hydrated = hydrateMessages(msgs)
      // IPC hydration can arrive after live deltas or completion. Restore the
      // older history without replacing the newer message/event state.
      setMessages(prev => {
        const live = new Map(prev.filter(m => liveMessageIds.has(m.id)).map(m => [m.id, m]))
        const ids = new Set(hydrated.map(m => m.id))
        return [
          ...hydrated.map(m => live.has(m.id) ? { ...m, ...live.get(m.id)! } : m),
          ...prev.filter(m => liveMessageIds.has(m.id) && !ids.has(m.id))
        ]
      })
      // Hydrate tool status map from persisted tool_calls_json
      const restoredTools: Record<string, any[]> = {}
      const restoredReasoning: Record<string, string> = {}
      const restoredReasoningSegments: Record<string, string[]> = {}
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
        if (m.reasoningSegmentsJson) {
          try {
            const parsed = JSON.parse(m.reasoningSegmentsJson)
            if (Array.isArray(parsed) && parsed.some((s: any) => typeof s === 'string' && s.trim())) {
              restoredReasoningSegments[m.id] = parsed.filter((s: any) => typeof s === 'string')
            }
          } catch { /* ignore bad json */ }
        }
        if (m.reasoningContent) {
          restoredReasoning[m.id] = m.reasoningContent
          restoredReasoningDone[m.id] = true
          if (!restoredReasoningSegments[m.id]) {
            restoredReasoningSegments[m.id] = [m.reasoningContent]
          }
        }
      }
      if (Object.keys(restoredTools).length > 0) {
        setToolStatusMap(prev => {
          const merged = { ...restoredTools, ...prev }
          for (const id of Object.keys(restoredTools)) {
            if (prev[id]) merged[id] = mergeToolStatusHistory(restoredTools[id], prev[id])
          }
          return merged
        })
      }
      if (Object.keys(restoredReasoning).length > 0) {
        setReasoningMap(prev => ({ ...restoredReasoning, ...prev }))
        setReasoningDoneMap(prev => ({ ...restoredReasoningDone, ...prev }))
      }
      if (Object.keys(restoredReasoningSegments).length > 0) {
        setReasoningSegmentMap(prev => ({ ...restoredReasoningSegments, ...prev }))
      }

      // If the user navigates/reloads during TTFT, the DB already has the
      // assistant placeholder but no stream delta may arrive until the first
      // token. Keep it visually bound to the active request instead of
      // rendering it as a completed blank message.
      window.api.chat.isStreaming(chatId).then((isActive: boolean) => {
        if (!isActive || !isCurrentChat() || terminalObserved) return
        setLoading(true)
        setStreamingMessageId(prev => prev || latestPendingAssistantId(hydrated))
      })
    })

    // Check if generation is still active for this chat (handles switch-away-and-back)
    window.api.chat.isStreaming(chatId).then((isActive: boolean) => {
      if (isActive && isCurrentChat() && !terminalObserved) {
        setLoading(true)
        // streamingMessageId will be set by the next stream event
      }
    })

    // Typing indicator: model is processing, waiting for first token
    const handleTyping = (data: any) => {
      if (data.chatId !== chatId || !isCurrentChat()) return
      liveMessageIds.add(data.messageId)
      setLoading(true)
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
      if (data.chatId !== chatId || !isCurrentChat()) return
      liveMessageIds.add(data.messageId)
      setLoading(true)
      setStreamingMessageId(data.messageId)
      if (data.metrics) setCurrentMetrics(data.metrics)

      if (data.isReasoning) {
        // Track reasoning content separately
        setReasoningMap(prev => ({
          ...prev,
          [data.messageId]: data.fullContent
        }))
        setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: false }))
        if (Array.isArray(data.reasoningSegments)) {
          setReasoningSegmentMap(prev => ({
            ...prev,
            [data.messageId]: data.reasoningSegments
          }))
        }
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

      setAnswerPassMap(prev => ({ ...prev, [data.messageId]: false }))

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
      if (data.chatId !== chatId || !isCurrentChat()) return
      terminalObserved = true
      liveMessageIds.add(data.messageId)
      // A remounted chat has no handleSend promise whose finally can settle it.
      setLoading(false)
      const responseWarnings = extractResponsesWarnings({ warnings: data.warnings }) ?? undefined
      setMessages(prev => {
        const existing = prev.find(m => m.id === data.messageId)
        const completed: Message = {
          ...(existing ?? { id: data.messageId, chatId, role: 'assistant', timestamp: Date.now() }),
          content: data.content || existing?.content || '',
          tokens: data.metrics?.tokenCount,
          metrics: data.metrics,
          warnings: responseWarnings ?? existing?.warnings
        }
        return existing ? prev.map(m => m.id === data.messageId ? completed : m) : [...prev, completed]
      })
      // Finalize reasoning state from completion event (ensures reasoning box persists
      // even if chat:reasoningDone was missed due to event ordering)
      if (data.reasoningContent) {
        setReasoningMap(prev => ({ ...prev, [data.messageId]: data.reasoningContent }))
        setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: true }))
      }
      if (Array.isArray(data.reasoningSegments)) {
        setReasoningSegmentMap(prev => ({ ...prev, [data.messageId]: data.reasoningSegments }))
      }
      setStreamingMessageId(null)
      setCurrentMetrics(null)
      setAnswerPassMap(prev => ({ ...prev, [data.messageId]: false }))
    }

    const handleReasoningDone = (data: any) => {
      if (data.chatId !== chatId || !isCurrentChat()) return
      setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: true }))
      // Also store the final reasoning content
      if (data.reasoningContent) {
        setReasoningMap(prev => ({ ...prev, [data.messageId]: data.reasoningContent }))
      }
      if (Array.isArray(data.reasoningSegments)) {
        setReasoningSegmentMap(prev => ({ ...prev, [data.messageId]: data.reasoningSegments }))
      }
    }

    const handleAnswerPass = (data: any) => {
      if (data.chatId !== chatId || !isCurrentChat()) return
      setAnswerPassMap(prev => ({ ...prev, [data.messageId]: true }))
    }

    const handleToolStatus = (data: any) => {
      if (data.chatId !== chatId || !isCurrentChat()) return
      setToolStatusMap(prev => ({
        ...prev,
        [data.messageId]: [
          ...(prev[data.messageId] || []),
          {
            phase: data.phase,
            toolName: data.toolName || '',
            toolCallId: data.toolCallId,
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
      if (data.chatId !== chatId || !isCurrentChat()) return
      setAskUserQuestion(data.question)
      setAskUserInput('')
    }

    // Store individual cleanup functions (avoids removeAllListeners race conditions)
    const cleanupTyping = window.api.chat.onTyping(handleTyping)
    const cleanupStream = window.api.chat.onStream(handleStream)
    const cleanupComplete = window.api.chat.onComplete(handleComplete)
    const cleanupReasoningDone = window.api.chat.onReasoningDone(handleReasoningDone)
    const cleanupAnswerPass = window.api.chat.onAnswerPass(handleAnswerPass)
    const cleanupToolStatus = window.api.chat.onToolStatus(handleToolStatus)
    const cleanupAskUser = window.api.chat.onAskUser(handleAskUser)

    return () => {
      disposed = true
      // Do NOT abort active generation when navigating away — the user explicitly
      // wants generation to continue in the background. Only clean up event listeners.
      // The abort button in InputBox handles explicit user cancellation.
      cleanupTyping()
      cleanupStream()
      cleanupComplete()
      cleanupReasoningDone()
      cleanupAnswerPass()
      cleanupToolStatus()
      cleanupAskUser()
      setReasoningMap({})
      setReasoningSegmentMap({})
      setReasoningDoneMap({})
      setAnswerPassMap({})
      setToolStatusMap({})
      setAskUserQuestion(null)
    }
  }, [chatId])

  // Sync hideToolStatus from chat overrides (re-reads when settings are saved)
  useEffect(() => {
    if (!chatId) return
    window.api.chat.getOverrides(chatId).then((o: any) => {
      setHideToolStatus(o?.hideToolStatus ?? false)
    })
  }, [chatId, overridesVersion])

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

  const handleSend = async (content: string, attachments?: MediaAttachment[]) => {
    if (!chatId || (!content.trim() && (!attachments || attachments.length === 0))) return

    // Guard: don't send if model isn't running (prevents fallback to wrong endpoint)
    if (!sessionEndpoint && sessionId) {
      showToast('error', t('chat.interface.toast.modelNotRunningTitle'), t('chat.interface.toast.modelNotRunningBody'))
      return
    }

    setLoading(true)
    setStreamingMessageId(null)
    setCurrentMetrics(null)

    // Build display content for user message: if attachments present, store as JSON content array.
    // Images use image_url; videos use video_url; audio uses input_audio so
    // Nemotron-Omni/Parakeet receives actual media instead of transcribed text.
    const displayContent = attachments && attachments.length > 0
      ? JSON.stringify([
        ...(content.trim() ? [{ type: 'text', text: content }] : []),
        ...attachments.map(attachmentContentPart),
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

      // Guard: if user switched chats while we were awaiting, don't touch state
      if (chatIdRef.current !== chatId) return

      const assistantId = result?.id

      // Replace the temp user message with the real one from DB, but keep
      // the streamed assistant message in place to avoid a full re-render
      // that causes stutter at end of generation.
      const freshMessages = await window.api.chat.getMessages(chatId)
      if (chatIdRef.current !== chatId) return
      setMessages(prev => {
        const streamedAssistant = assistantId
          ? prev.find(m => m.id === assistantId && m.role === 'assistant')
          : null
        if (streamedAssistant) {
          const hydrated = hydrateMessages(freshMessages)
          return hydrated.map(m => {
            if (m.id === streamedAssistant.id) {
              return {
                ...streamedAssistant,
                content: m.content ?? streamedAssistant.content,
                tokens: m.tokens,
                metrics: m.metrics || streamedAssistant.metrics,
                metricsJson: m.metricsJson,
                warnings: m.warnings || streamedAssistant.warnings,
                warningsJson: m.warningsJson,
                toolCallsJson: m.toolCallsJson,
                reasoningContent: m.reasoningContent
              }
            }
            return m
          })
        }
        return hydrateMessages(freshMessages)
      })
    } catch (error: any) {
      // Guard: if user switched chats, don't show error or touch state
      if (chatIdRef.current !== chatId) return
      if (!isExpectedChatDisconnectError(error)) {
        console.error('Failed to send message:', error)
      }
      const msg = formatChatSendErrorMessage(error, t)
      showToast('error', t('chat.interface.toast.messageFailedTitle'), msg)
      // Reload messages from DB to restore consistent state
      try {
        const freshMessages = await window.api.chat.getMessages(chatId)
        if (chatIdRef.current !== chatId) return
        // An empty persisted history is authoritative too: a rejected first
        // request can leave a typing-only assistant in the optimistic UI.
        setMessages(hydrateMessages(freshMessages))
      } catch {
        // If reload also fails, at least remove the temp message
        if (chatIdRef.current === chatId) {
          setMessages(prev => prev.filter(m => m.id !== tempId))
        }
      }
    } finally {
      // Only reset loading state if still on the same chat
      if (chatIdRef.current === chatId) {
        setLoading(false)
        setStreamingMessageId(null)
      }
    }
  }

  // Regenerate: truncate the original user turn and its response, then re-send
  // that user content once. Calling handleSend after deleting only the assistant
  // persisted a duplicate consecutive user message, so the regenerated prompt
  // was not the prompt being regenerated (live DSV4 Max repro: duplicated user
  // turn followed by a reasoning repetition loop).
  const handleRegenerate = async () => {
    if (!chatId || loading) return
    const lastUser = [...messages].reverse().find(m => m.role === 'user')
    if (!lastUser) return
    const { content, attachments } = restoreUserMessageContent(lastUser.content)
    // Delete the original user turn and everything after it in one DB operation.
    // handleSend() will persist exactly one replacement user turn. This also
    // removes any tool/assistant continuation rows belonging to the old turn.
    try {
      await window.api.chat.deleteMessagesFrom(chatId, lastUser.timestamp)
    } catch (error) {
      console.error('Failed to truncate chat for regeneration:', error)
      showToast('error', t('chat.interface.toast.regenerateFailedTitle'), t('chat.interface.toast.regenerateFailedBody'))
      return
    }
    if (chatIdRef.current !== chatId) return
    setMessages(prev => prev.filter(m => m.timestamp < lastUser.timestamp))
    await handleSend(content, attachments)
  }

  // Edit & resend: truncate conversation at the edited message, resend with new content
  const handleEdit = async (messageId: string, newContent: string) => {
    if (!chatId || loading) return
    const idx = messages.findIndex(m => m.id === messageId)
    if (idx < 0 || messages[idx].role !== 'user') return
    const { attachments } = restoreUserMessageContent(messages[idx].content)
    if (!newContent.trim() && !attachments?.length) return
    // Batch-delete all messages from this point forward (single SQL query)
    const fromTs = messages[idx].timestamp
    try {
      await window.api.chat.deleteMessagesFrom(chatId, fromTs)
    } catch (error) {
      console.error('Failed to truncate chat for editing:', error)
      showToast('error', t('chat.interface.toast.messageFailedTitle'), formatChatSendErrorMessage(error, t))
      return
    }
    if (chatIdRef.current !== chatId) return
    setMessages(prev => prev.slice(0, idx))
    await handleSend(newContent, attachments)
  }

  if (!chatId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
            <Sparkles className="h-6 w-6 text-primary" />
          </div>
          <h2 className="text-xl font-semibold mb-2">{t('chat.interface.emptyStateTitle')}</h2>
          <p className="text-sm text-muted-foreground mb-6">
            {t('chat.interface.emptyStateBody')}
          </p>
          {onNewChat && (
            <button
              onClick={onNewChat}
              className="px-5 py-2.5 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 font-medium text-sm transition-colors"
            >
              {t('chat.interface.newChat')}
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex justify-end px-4 py-1 border-b border-border/40">
        <button
          type="button"
          className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded"
          onClick={async () => {
            try {
              const result = await window.api.chat.export(chatId, 'markdown')
              if (result.success) showToast('success', 'Session exported', result.path)
            } catch (error) {
              showToast('error', 'Export failed', (error as Error).message)
            }
          }}
        >Export session</button>
      </div>
      <MessageList
        messages={messages}
        streamingMessageId={streamingMessageId}
        currentMetrics={currentMetrics}
        reasoningMap={reasoningMap}
        reasoningSegmentMap={reasoningSegmentMap}
        reasoningDoneMap={reasoningDoneMap}
        answerPassMap={answerPassMap}
        toolStatusMap={toolStatusMap}
        hideToolStatus={hideToolStatus}
        sessionId={sessionId}
        sessionEndpoint={sessionEndpoint}
        onRegenerate={handleRegenerate}
        onEdit={handleEdit}
      />
      {/* ask_user tool: inline question from model */}
      {askUserQuestion && chatId && (
        <div className="border-t border-border bg-card px-4 py-3">
          <div className="max-w-2xl mx-auto">
            <div className="text-xs font-medium text-primary mb-1.5">{t('chat.interface.askUserLabel')}</div>
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
                placeholder={t('chat.interface.askUserPlaceholder')}
                autoFocus
                className="flex-1 px-3 py-1.5 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <button
                type="submit"
                disabled={!askUserInput.trim()}
                className="px-4 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
              >
                {t('chat.interface.askUserReply')}
              </button>
              <button
                type="button"
                onClick={() => {
                  window.api.chat.answerUser(chatId, t('chat.interface.askUserSkipResponse'))
                  setAskUserQuestion(null)
                  setAskUserInput('')
                }}
                className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent"
              >
                {t('chat.interface.askUserSkip')}
              </button>
            </form>
          </div>
        </div>
      )}
      {/* Model sleeping banner */}
      {sessionEndpoint && sessionId && !loading && sessionStatus === 'standby' && (
        <div className="flex items-center justify-center gap-2 px-4 py-2 border-t border-border bg-blue-500/5">
          <span className="text-xs text-blue-400">{t('chat.interface.standbyBanner')}</span>
        </div>
      )}
      {/* Model loading / waking progress banner. Covers the cold start, a
          wake triggered by this chat's message or an API request (status
          'loading'), and the settle phase where the server already answers
          but the weights are still being copied into RAM (status 'running'
          with progress < 100). Hidden once the main process sends the
          terminal 100%. */}
      {sessionId && (
        sessionLoadProgress?.preflightActive === true || sessionStatus === 'loading' ||
        (sessionStatus === 'running' && sessionLoadProgress && sessionLoadProgress.progress < 100)
      ) && (
        <div data-vmlx-section="chat-model-load-progress" role="status" className="px-4 py-2 border-t border-border bg-yellow-500/5">
          <div className="flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 text-yellow-500 animate-spin flex-shrink-0" />
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full bg-yellow-500 rounded-full transition-all duration-500 ease-out ${sessionLoadProgress?.indeterminate !== false ? 'animate-pulse' : ''}`}
                style={{ width: sessionLoadProgress?.indeterminate === false ? `${sessionLoadProgress.progress}%` : '100%' }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground flex-shrink-0">
              {sessionLoadProgress
                ? `${sessionLoadProgress.labelKey
                    ? t(sessionLoadProgress.labelKey, {
                        defaultValue: sessionLoadProgress.label,
                        ...(sessionLoadProgress.labelParams || {}),
                      })
                    : sessionLoadProgress.label}${sessionLoadProgress.indeterminate === false ? ` (${sessionLoadProgress.progress}%)` : ''}`
                : t('chat.interface.loadingBanner')}
            </span>
          </div>
          {formatResidentLoad(sessionLoadProgress) && (
            <p className="text-[10px] text-muted-foreground/80 mt-1 text-center">
              {t('sessions.card.residentRam')} {formatResidentLoad(sessionLoadProgress)}
            </p>
          )}
        </div>
      )}
      {/* TTFT / Waking up banner */}
      {loading && !streamingMessageId && (
        <div className="flex items-center justify-center gap-2 px-4 py-2 border-t border-border bg-primary/5">
          <Loader2 className="h-3.5 w-3.5 text-primary animate-spin" />
          <span className="text-xs text-primary/80">
            {sessionStatus === 'standby' ? t('chat.interface.wakingBanner') : t('chat.interface.evaluatingBanner')}
          </span>
        </div>
      )}
      {/* Model not running banner */}
      {!sessionEndpoint && sessionId && !loading && sessionStatus !== 'loading' && !sessionLoadProgress?.preflightActive && (
        <div className="flex items-center justify-center gap-3 px-4 py-2 border-t border-border bg-warning/5">
          <span className="text-xs text-muted-foreground">{t('chat.interface.notRunningBanner')}</span>
          <button
            data-vmlx-control="chat-load-model"
            onClick={async () => {
              try {
                const result = await window.api.sessions.start(sessionId)
                if (!result?.success) {
                  throw new Error(result?.error || t('chat.interface.toast.failedToStart'))
                }
              } catch (e) {
                showToast('error', t('chat.interface.toast.failedToStart'), (e as Error).message)
              }
            }}
            className="text-xs px-3 py-1 bg-success text-success-foreground rounded hover:bg-success/90 transition-colors font-medium"
          >
            {t('chat.interface.loadModelButton')}
          </button>
        </div>
      )}
      <InputBox
        onSend={handleSend}
        onAbort={handleAbort}
        disabled={loading || (!sessionEndpoint && !!sessionId)}
        loading={loading}
        sessionEndpoint={sessionEndpoint}
        sessionId={sessionId}
      />
    </div>
  )
}
