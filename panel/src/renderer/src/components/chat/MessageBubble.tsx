import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import { useState, useMemo, useCallback, useRef, useEffect, memo } from 'react'
import { AlertTriangle, Copy, Check, User, Sparkles, RefreshCw, Pencil, Loader2 } from 'lucide-react'
import { VideoAttachmentPreview } from './VideoAttachmentPreview'
import { ReasoningBox } from './ReasoningBox'
import { ToolCallStatus } from './ToolCallStatus'
import { InlineToolCall, InlineToolGroup } from './InlineToolCall'
import { useElapsedSeconds } from './useElapsedSeconds'
import { isTypewriterAppend } from './typewriter'
import { TTSPlayer } from './VoiceChat'
import { formatTimestamp, parseContentArray, getMetricsItems, type MessageMetrics } from './chat-utils'
import { useRelativeTime } from '../ui/use-relative-time'
import { renderChatMarkdownHtml } from './mathMarkdown'
import { reasoningSegmentsForDisplay as getReasoningSegmentsForDisplay } from '../../../../shared/interleavedReasoning'
import { useTranslation } from '../../i18n'
import { sanitizeChatHtml } from './sanitizeChatHtml'
import { restoreUserMessageContent } from './messageReplay'
import { validateAssistantDisplayTimeline, type AssistantDisplayTimeline } from '../../../../shared/assistantDisplayTimeline'

interface Message {
  id: string
  role: 'system' | 'user' | 'assistant'
  content: string
  timestamp: number
  tokens?: number
  displayTimeline?: AssistantDisplayTimeline
}

interface MessageBubbleProps {
  message: Message
  isStreaming?: boolean
  metrics?: MessageMetrics | null
  reasoningContent?: string
  reasoningSegments?: string[]
  reasoningDone?: boolean
  answerPassPending?: boolean
  toolStatuses?: any[]
  warnings?: string[]
  sessionId?: string
  sessionEndpoint?: { host: string; port: number }
  isLastAssistant?: boolean
  onRegenerate?: () => void
  onEdit?: (messageId: string, newContent: string) => void
}

// Language auto-detection (hljs.highlightAuto tries every grammar) is cosmetic, and on large untagged blocks it is the
// whole render cost: a 370k-character pasted document (130 untagged blocks) took 2.3 s of blocking main thread, 99 % of
// it highlightAuto (measured 2026-10-17). Untagged blocks above this size render as escaped plain text; tagged blocks
// and small untagged ones still highlight. In USER messages untagged blocks are never auto-detected (pasted text; the
// cap alone left 1.2 s on the same document, spread over its many small blocks).
export const AUTO_HIGHLIGHT_MAX_CHARS = 4000

export function escapeCodeHtml(code: string): string {
  return code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

export function highlightCodeBlock(code: string, lang?: string, userContent = false): string {
  if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value
  if (userContent || code.length > AUTO_HIGHLIGHT_MAX_CHARS) return escapeCodeHtml(code)
  return hljs.highlightAuto(code).value
}

// Custom renderer: wraps code blocks with a localized header bar.
function parseMarkdown(
  markdown: string,
  defaultLanguage: string,
  copyLabel: string,
  userContent = false,
): string {
  const renderer = new marked.Renderer()
  renderer.code = (code, lang) => {
  const highlighted = highlightCodeBlock(code, lang, userContent)
  const headerHtml = `<div class="code-header"><span class="code-lang">${lang || defaultLanguage}</span><button class="code-copy-btn">${copyLabel}</button></div>`
  return `<div class="code-block-wrapper">${headerHtml}<pre><code class="hljs language-${lang || 'plaintext'}">${highlighted}</code></pre></div>`
  }
  // renderChatMarkdownHtml owns prepare -> marked -> KaTeX-splice ordering so
  // renderer-owned math HTML never flows through marked. Assistant-only
  // presentation guards must not rewrite user content.
  return renderChatMarkdownHtml(markdown, {
    renderer,
    assistantContent: !userContent,
  })
}

/**
 * Sanitize HTML using DOMPurify — allows safe markdown output, blocks XSS.
 * All user/model content passes through this before being rendered via innerHTML.
 */

/** Group tool statuses into tool call groups. Each 'calling' phase starts a new group.
 *  Also extracts contentOffset for inline positioning. */
function groupToolStatuses(statuses: any[]): { groups: InlineToolGroup[]; hasOffsets: boolean; processingStatus?: any } {
  const groups: InlineToolGroup[] = []
  let current: InlineToolGroup | null = null
  let hasOffsets = false
  let processingStatus: any = null
  const groupIds = new Map<string, InlineToolGroup>()
  const isOpen = (group: InlineToolGroup) => {
    const last = group.statuses[group.statuses.length - 1]
    return last && last.phase !== 'result' && last.phase !== 'error' && last.phase !== 'done'
  }
  const findOpenGroup = (s: any) => {
    if (s.toolCallId && groupIds.has(s.toolCallId)) return groupIds.get(s.toolCallId)!
    return [...groups].reverse().find(g => g.name === s.toolName && isOpen(g)) || current
  }

  for (const s of statuses) {
    if (s.phase === 'calling') {
      current = { name: s.toolName, statuses: [s] }
      if (s.toolCallId) groupIds.set(s.toolCallId, current)
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
    } else {
      const target = findOpenGroup(s)
      if (target) target.statuses.push(s)
    }
  }

  return { groups, hasOffsets, processingStatus }
}

function stripToolMarkupSegment(segment: string): string {
  return segment
    .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/gi, '')
    .replace(/<zyphra_tool_call\b[^>]*>[\s\S]*?<\/zyphra_tool_call>/gi, '')
    .replace(/<minimax:tool_call\b[^>]*>[\s\S]*?<\/minimax:tool_call>/gi, '')
    .replace(/<function(?:=[^>\s]+|\b[^>]*)>[\s\S]*?<\/function>/gi, '')
    .replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/gi, '')
    .replace(/<(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)\b[^>]*>[\s\S]*?<\/(?:read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash)>/gi, '')
    .replace(/(?:^|\n)\s*\[Calling tool:[^\n]*(?=\n|$)/gi, '\n')
}

function stripStructuredToolMarkup(content: string): string {
  // Code fences / inline code may legitimately CONTAIN tool-markup examples
  // (e.g. the model documenting DSML/XML tool syntax) — never strip inside
  // them. Unterminated fences (mid-stream) protect through end of content.
  const parts = content.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/)
  return parts.map((part, i) => (i % 2 === 1 ? part : stripToolMarkupSegment(part))).join('')
}

/** Prose classes for rendered markdown */
const proseClasses = 'prose prose-invert max-w-none break-words overflow-x-auto [&_pre]:overflow-x-auto [&_code]:break-all [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5 [&_h1]:text-lg [&_h2]:text-base [&_h3]:text-sm [&_pre]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-muted-foreground [&_table]:text-sm [&_th]:text-left [&_th]:font-medium [&_th]:px-3 [&_th]:py-1.5 [&_td]:px-3 [&_td]:py-1.5'

/** rAF-based typewriter for smooth streaming — reveals characters gradually instead of
 *  showing 3-5 tokens at once (caused by TCP batching + React 18 state coalescing). */
function useTypewriter(fullContent: string, isStreaming: boolean): string {
  const [displayed, setDisplayed] = useState(fullContent)
  const rafRef = useRef<number>(0)
  const lenRef = useRef(fullContent.length)
  const shownRef = useRef(fullContent)
  const fullRef = useRef(fullContent)
  const wasStreamingRef = useRef(isStreaming)
  fullRef.current = fullContent

  useEffect(() => {
    const wasStreaming = wasStreamingRef.current
    wasStreamingRef.current = isStreaming
    cancelAnimationFrame(rafRef.current)

    // Not an append of what is on screen (shrink, or a same/longer rewrite such
    // as the completion trimming the stream's leading whitespace) — snap.
    // Tracking only the LENGTH froze "\n\n4" on screen when "482" completed.
    if (!isTypewriterAppend(shownRef.current, fullContent)) {
      lenRef.current = fullContent.length
      shownRef.current = fullContent
      setDisplayed(fullContent)
      return
    }

    // Ordinary non-stream updates (history hydration, edit, regeneration
    // replacement) should remain immediate. A just-finished stream is
    // different: React can receive the final content and completion IPC in one
    // batch. Draining that existing backlog preserves visible streaming instead
    // of snapping the entire answer into view when isStreaming flips false.
    if (!isStreaming && !wasStreaming) {
      setDisplayed(fullContent)
      shownRef.current = fullContent
      lenRef.current = fullContent.length
      return
    }
    if (lenRef.current >= fullContent.length) return

    const tick = () => {
      const full = fullRef.current
      const cur = lenRef.current
      if (cur >= full.length) return
      const remaining = full.length - cur
      // Catch up within ~200ms (12 frames at 60fps), min 1 char/frame
      const chars = Math.max(1, Math.ceil(remaining / 12))
      const newLen = Math.min(cur + chars, full.length)
      lenRef.current = newLen
      shownRef.current = full.slice(0, newLen)
      setDisplayed(shownRef.current)
      if (newLen < full.length) {
        rafRef.current = requestAnimationFrame(tick)
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [fullContent, isStreaming])

  return displayed
}

export const MessageBubble = memo(function MessageBubble({ message, isStreaming, metrics, reasoningContent, reasoningSegments, reasoningDone, answerPassPending, toolStatuses, warnings, sessionId, sessionEndpoint, isLastAssistant, onRegenerate, onEdit }: MessageBubbleProps) {
  const { t, locale } = useTranslation()
  const timeLabel = useRelativeTime(message.timestamp, (ts) => formatTimestamp(ts, t, locale))
  const [copied, setCopied] = useState(false)
  const [zoomedImage, setZoomedImage] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState('')
  const [editHasAttachments, setEditHasAttachments] = useState(false)

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
      btn.textContent = t('chat.bubble.codeCopied')
      setTimeout(() => { btn.textContent = t('chat.bubble.codeCopy') }, 1500)
    }
  }, [t])

  // Group tool statuses for inline rendering
  const toolGroups = useMemo(() => {
    if (!toolStatuses || toolStatuses.length === 0) return null
    return groupToolStatuses(toolStatuses)
  }, [toolStatuses])

  // Live processing/generating row: real elapsed seconds since the producer
  // emitted that status (no synthetic percentage).
  const processingElapsed = useElapsedSeconds(
    toolGroups?.processingStatus?.timestamp,
    !!isStreaming && !!toolGroups?.processingStatus,
  )

  // Typewriter: smooth character-by-character reveal during streaming
  const displayedContent = useTypewriter(message.content, !!isStreaming)
  const sourceReasoningSegments = Array.isArray(reasoningSegments) && reasoningSegments.length > 0
    ? reasoningSegments : reasoningContent ? [reasoningContent] : []
  const orderedTimeline = validateAssistantDisplayTimeline(message.displayTimeline, message.content, sourceReasoningSegments)
  const reasoningSegmentsForDisplay = useMemo(() => {
    const segments = Array.isArray(reasoningSegments) && reasoningSegments.length > 0
      ? reasoningSegments
      : reasoningContent
        ? [reasoningContent]
        : []
    if (orderedTimeline) return segments
    return getReasoningSegmentsForDisplay(segments, {
      liveReplace: !!isStreaming && !(reasoningDone ?? false),
    })
  }, [isStreaming, reasoningContent, reasoningDone, reasoningSegments, orderedTimeline])
  const latestReasoningSegment = reasoningSegmentsForDisplay[reasoningSegmentsForDisplay.length - 1] || ''
  // Same for the active reasoning segment (separate stream, same TCP batching problem)
  const displayedLatestReasoning = useTypewriter(latestReasoningSegment, !!isStreaming && !(reasoningDone ?? false))
  const displayedReasoningSegments = useMemo(() => {
    if (reasoningSegmentsForDisplay.length === 0) return []
    return reasoningSegmentsForDisplay.map((segment, index) =>
      index === reasoningSegmentsForDisplay.length - 1 ? displayedLatestReasoning : segment
    )
  }, [displayedLatestReasoning, reasoningSegmentsForDisplay])
  const hasToolStatus = !!(toolStatuses && toolStatuses.length > 0)
  const isEmptyAssistant =
    message.role === 'assistant' &&
    !String(message.content || '').trim() &&
    !displayedReasoningSegments.some(segment => segment.trim()) &&
    !hasToolStatus

  // Render a DOMPurify-sanitized markdown segment
  const renderMarkdownSegment = useCallback((text: string, key: string, userContent = false) => {
    if (!text) return null
    const safeHtml = sanitizeChatHtml(parseMarkdown(
      text,
      t('chat.bubble.codeDefaultLang'),
      t('chat.bubble.codeCopy'),
      userContent,
    ))
    return (
      <div
        key={key}
        className={proseClasses}
        dangerouslySetInnerHTML={{ __html: safeHtml }}
        onClick={handleProseClick}
      />
    )
  }, [handleProseClick, t])

  const renderUserContent = () => {
    const contentParts = parseContentArray(message.content)
    if (contentParts) {
      const images = contentParts.filter(p => p.type === 'image_url' && p.image_url?.url)
      const videos = contentParts.filter(p => p.type === 'video_url' && p.video_url?.url)
      const audios = contentParts.filter(p =>
        (p.type === 'input_audio' && p.input_audio?.data) ||
        (p.type === 'audio_url' && p.audio_url?.url)
      )
      const textParts = contentParts.filter(p => p.type === 'text' && p.text)
      const audioSrc = (p: any) => {
        if (p.type === 'audio_url') return p.audio_url.url
        const format = p.input_audio?.format || 'wav'
        const mime = format === 'mp3' ? 'audio/mpeg' : `audio/${format}`
        const data = p.input_audio?.data || ''
        return data.startsWith('data:') ? data : `data:${mime};base64,${data}`
      }
      return (
        <div>
          {images.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {images.map((img, i) => (
                <img
                  key={i}
                  src={img.image_url!.url}
                  alt={t('chat.bubble.imageAlt', { n: i + 1 })}
                  className="max-w-[300px] max-h-[200px] rounded-md border border-white/10 cursor-pointer hover:opacity-90 transition-opacity object-contain"
                  onClick={() => setZoomedImage(img.image_url!.url)}
                />
              ))}
            </div>
          )}
          {videos.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {videos.map((vid, i) => (
                <VideoAttachmentPreview key={`v-${i}`} src={vid.video_url!.url} />
              ))}
            </div>
          )}
          {audios.length > 0 && (
            <div className="flex flex-col gap-2 mb-2">
              {audios.map((aud, i) => (
                <audio
                  key={`a-${i}`}
                  src={audioSrc(aud)}
                  controls
                  preload="metadata"
                  className="max-w-[360px]"
                />
              ))}
            </div>
          )}
          {textParts.map((p, i) =>
            renderMarkdownSegment(p.text!, `user-text-${i}`, true)
          )}
        </div>
      )
    }
    return renderMarkdownSegment(message.content, 'user-text', true)
  }

  const renderInlineContent = () => {
    if (orderedTimeline) {
      const last = orderedTimeline.items[orderedTimeline.items.length - 1]
      return <>{orderedTimeline.items.map((item, index) => {
        if (item.kind === 'reasoning') {
          const content = (displayedReasoningSegments[item.segment] || '').slice(item.start, item.end)
          if (!content.trim()) return null
          return <ReasoningBox key={`${message.id}-reasoning-${index}`} content={content}
            isStreaming={!!isStreaming && item === last && !(reasoningDone ?? false)}
            isDone={item !== last || (reasoningDone ?? false)} />
        }
        if (item.kind === 'tool') {
          const group = toolGroups?.groups.find(group => group.statuses[0]?.toolCallId === item.callId)
          return group ? <InlineToolCall key={`tool-${item.callId}`} group={group} isStreaming={!!isStreaming} /> : null
        }
        const content = displayedContent.slice(item.start, item.end)
        return renderMarkdownSegment(toolGroups ? stripStructuredToolMarkup(content) : content, `content-${index}`)
      })}
      {/* Same live status row as the legacy inline path: while a tool pass is
          still processing/generating and nothing newer has been emitted yet,
          show it after the last recorded item instead of dropping it. */}
      {toolGroups?.processingStatus && isStreaming && (
        <div key="processing" className="flex items-center gap-2 text-muted-foreground text-xs py-1"
          data-vmlx-proof-tool-progress={toolGroups.processingStatus.phase === 'generating' ? 'generating' : 'processing'}
          data-vmlx-proof-tool-progress-elapsed-s={processingElapsed ?? undefined}>
          <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${toolGroups.processingStatus.phase === 'generating' ? 'bg-primary' : 'bg-warning'}`} />
          <span>{toolGroups.processingStatus.phase === 'generating' ? t('chat.bubble.generatingTool') : t('chat.bubble.processingTool')}{processingElapsed != null && processingElapsed >= 1 ? ` ${processingElapsed}s` : ''}</span>
        </div>
      )}</>
    }
    if (!message.content && (!toolGroups || toolGroups.groups.length === 0)) return null

    const content = toolGroups
      ? stripStructuredToolMarkup(displayedContent || '')
      : displayedContent || ''

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
          <div key="processing" className="flex items-center gap-2 text-muted-foreground text-xs py-1"
            data-vmlx-proof-tool-progress={isGenerating ? 'generating' : 'processing'}
            data-vmlx-proof-tool-progress-elapsed-s={processingElapsed ?? undefined}>
            <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${isGenerating ? 'bg-primary' : 'bg-warning'}`} />
            <span>{isGenerating ? t('chat.bubble.generatingTool') : t('chat.bubble.processingTool')}{processingElapsed != null && processingElapsed >= 1 ? ` ${processingElapsed}s` : ''}</span>
          </div>
        )
      }

      if (content.trim()) {
        elements.push(renderMarkdownSegment(content, 'seg-content') as JSX.Element)
      }

      return <>{elements}</>
    }

    if (!content.trim()) return null
    const safeHtml = sanitizeChatHtml(parseMarkdown(
      content,
      t('chat.bubble.codeDefaultLang'),
      t('chat.bubble.codeCopy')
    ))
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
      const items = getMetricsItems(metrics, !!isStreaming, t)
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
          {t('chat.message.tokenCount', { n: message.tokens })}
        </div>
      )
    }

    return null
  }

  const isUser = message.role === 'user'

  // ─── User message: compact right-aligned bubble ───────────────────
  if (isUser) {
    // Edit mode for user messages
    if (editing && onEdit) {
      const canResend = !!editText.trim() || editHasAttachments
      return (
        <div
          className="flex justify-end gap-2.5 ml-[5%] md:ml-[10%] lg:ml-[15%]"
          data-vmlx-proof-message-id={message.id}
          data-vmlx-proof-message-role="user"
        >
          <div className="flex flex-col items-end max-w-full w-full">
            <textarea
              data-vmlx-control="chat-edit-input"
              value={editText}
              onChange={e => setEditText(e.target.value)}
              autoFocus
              rows={3}
              className="w-full max-w-lg px-3 py-2 bg-background border border-primary rounded-xl text-sm resize-none focus:outline-none focus:ring-1 focus:ring-primary"
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  if (canResend) {
                    onEdit(message.id, editText)
                    setEditing(false)
                  }
                }
                if (e.key === 'Escape') setEditing(false)
              }}
            />
            <div className="flex gap-2 mt-1.5">
              <button data-vmlx-control="chat-edit-cancel" onClick={() => setEditing(false)} className="text-xs text-muted-foreground hover:text-foreground px-2 py-1">{t('chat.bubble.cancel')}</button>
              <button data-vmlx-control="chat-edit-send" disabled={!canResend} onClick={() => { if (canResend) { onEdit(message.id, editText); setEditing(false) } }} className="text-xs bg-primary text-primary-foreground px-3 py-1 rounded hover:bg-primary/90 disabled:opacity-50">{t('chat.bubble.send')}</button>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div
        className="flex justify-end gap-2.5 ml-[5%] md:ml-[10%] lg:ml-[15%] group"
        data-vmlx-proof-message-id={message.id}
        data-vmlx-proof-message-role="user"
      >
        {/* Edit button on hover */}
        {onEdit && !isStreaming && (
          <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              data-vmlx-control="chat-edit"
              onClick={() => {
                // Media data URLs can be large. Reconstruct once on open, not
                // on every keystroke while checking whether Send is enabled.
                const replay = restoreUserMessageContent(message.content)
                setEditText(replay.content)
                setEditHasAttachments(!!replay.attachments?.length)
                setEditing(true)
              }}
              className="text-muted-foreground/40 hover:text-foreground transition-colors p-1"
              title={t('chat.bubble.editTitle')}
            >
              <Pencil className="h-3 w-3" />
            </button>
          </div>
        )}
        <div className="flex flex-col items-end max-w-full">
          <div className="prose-on-accent bg-primary text-primary-foreground rounded-2xl rounded-br-md px-4 py-2.5 text-sm">
            {renderUserContent()}
          </div>
          <span className="text-[10px] text-muted-foreground/50 mt-1 mr-1">
            {timeLabel}
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
              alt={t('chat.bubble.zoomedAlt')}
              className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"
              onClick={e => e.stopPropagation()}
            />
            <button
              onClick={() => setZoomedImage(null)}
              className="absolute top-4 right-4 text-white/80 hover:text-white text-2xl font-bold"
              aria-label={t('common.close')}
              title={t('common.close')}
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
    <div
      className="flex gap-2.5 group mr-[5%] md:mr-[10%] lg:mr-[15%]"
      data-vmlx-proof-message-id={message.id}
      data-vmlx-proof-message-role="assistant"
    >
      <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
      </div>
      <div className="flex-1 min-w-0">
        {/* Header: timestamp + actions */}
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] text-muted-foreground/50">
            {timeLabel}
          </span>
          {!isStreaming && message.content && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <TTSPlayer text={message.content} endpoint={sessionEndpoint} sessionId={sessionId} />
              <button
                onClick={() => copyToClipboard(message.content)}
                className="text-muted-foreground/50 hover:text-foreground transition-colors"
                title={t('chat.bubble.copyTitle')}
              >
                {copied
                  ? <Check className="h-3 w-3 text-success" />
                  : <Copy className="h-3 w-3" />
                }
              </button>
              {isLastAssistant && onRegenerate && (
                <button
                  data-vmlx-control="chat-regenerate"
                  onClick={onRegenerate}
                  className="text-muted-foreground/50 hover:text-foreground transition-colors"
                  title={t('chat.bubble.regenerateTitle')}
                >
                  <RefreshCw className="h-3 w-3" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Reasoning boxes */}
        {!orderedTimeline && displayedReasoningSegments.length > 0 &&
         !(displayedReasoningSegments.length === 1 && message.content && displayedReasoningSegments[0].trim() === message.content.trim()) && (
          <div className="mb-3 space-y-2">
            {displayedReasoningSegments.map((segment, index) => {
              const isLast = index === displayedReasoningSegments.length - 1
              return (
                <ReasoningBox
                  key={`${message.id}-reasoning-${index}`}
                  content={segment}
                  isStreaming={!!isStreaming && isLast}
                  isDone={isLast ? (reasoningDone ?? false) : true}
                />
              )
            })}
          </div>
        )}

        {/* Main content */}
        <div
          className="text-sm"
          data-vmlx-proof-answer="true"
          data-vmlx-proof-answer-state={
            !isStreaming && displayedContent.length === message.content.length
              ? 'complete'
              : 'streaming'
          }
          data-vmlx-proof-answer-full-length={message.content.length}
          data-vmlx-proof-answer-rendered-length={displayedContent.length}
        >
          {renderInlineContent()}
          {answerPassPending && isStreaming && !message.content && (
            <div className="flex items-center gap-2 text-muted-foreground py-1">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              <span>{t('chat.bubble.finalizingAnswer')}</span>
            </div>
          )}
          {isEmptyAssistant && isStreaming && !answerPassPending && (
            <div className="flex items-center gap-2 text-muted-foreground py-1">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              <span>{t('chat.bubble.waitingForResponse')}</span>
            </div>
          )}
          {isEmptyAssistant && !isStreaming && (
            <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-foreground">
              <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 text-warning" />
              <span>{t('chat.bubble.noVisibleResponse')}</span>
            </div>
          )}
        </div>

        {/* Non-content response warnings from the engine/API */}
        {warnings && warnings.length > 0 && (
          <div className="mt-3 space-y-1.5">
            {warnings.map((warning, index) => (
              <div
                key={`${index}-${warning}`}
                className="flex gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-foreground"
              >
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-warning" />
                <span className="whitespace-pre-wrap break-words">{warning}</span>
              </div>
            ))}
          </div>
        )}

        {/* Legacy tool call display */}
        {toolStatuses && toolStatuses.length > 0 &&
         !orderedTimeline && !(toolGroups && toolGroups.hasOffsets) && (
          <ToolCallStatus statuses={toolStatuses} isStreaming={!!isStreaming} />
        )}

        {/* Typing indicator */}
        {isStreaming && !message.content && displayedReasoningSegments.length === 0 && !hasToolStatus && !isEmptyAssistant && (
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
