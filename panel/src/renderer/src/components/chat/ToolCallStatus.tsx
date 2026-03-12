import { useState, useEffect } from 'react'
import { Check, X, Square, ChevronRight, AlertTriangle, Loader2, StopCircle } from 'lucide-react'
import { formatJson } from './chat-utils'

export interface ToolStatus {
  phase: 'calling' | 'executing' | 'result' | 'error' | 'processing' | 'done' | 'generating' | 'asking'
  toolName: string
  detail?: string
  iteration?: number
  timestamp: number
}

interface ToolCallStatusProps {
  statuses: ToolStatus[]
  isStreaming: boolean
}

export function ToolCallStatus({ statuses, isStreaming }: ToolCallStatusProps) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({})
  const [sectionOpen, setSectionOpen] = useState(false)

  // Auto-open when actively streaming tool calls (stays open after for review)
  useEffect(() => {
    const lastStatus = statuses[statuses.length - 1]
    if (!lastStatus) return
    if (isStreaming && lastStatus.phase !== 'done') {
      setSectionOpen(true)
    }
  }, [statuses, isStreaming])

  if (statuses.length === 0) return null

  // Group by tool call (each 'calling' starts a new group)
  // Skip intermediate 'processing' events — only keep the last one
  const groups: { name: string; statuses: ToolStatus[] }[] = []
  let current: { name: string; statuses: ToolStatus[] } | null = null
  let lastProcessingIdx = -1

  for (const s of statuses) {
    if (s.phase === 'calling') {
      current = { name: s.toolName, statuses: [s] }
      groups.push(current)
      lastProcessingIdx = -1
    } else if (s.phase === 'generating') {
      if (lastProcessingIdx >= 0 && lastProcessingIdx < groups.length) {
        groups[lastProcessingIdx] = { name: '', statuses: [s] }
      } else {
        lastProcessingIdx = groups.length
        groups.push({ name: '', statuses: [s] })
      }
      current = null
    } else if (s.phase === 'processing') {
      if (lastProcessingIdx >= 0 && lastProcessingIdx < groups.length) {
        groups[lastProcessingIdx] = { name: '', statuses: [s] }
      } else {
        lastProcessingIdx = groups.length
        groups.push({ name: '', statuses: [s] })
      }
      current = null
    } else if (s.phase === 'done') {
      groups.push({ name: '', statuses: [s] })
      current = null
      lastProcessingIdx = -1
    } else if (current) {
      current.statuses.push(s)
    } else {
      groups.push({ name: s.toolName, statuses: [s] })
    }
  }

  // Count actual tool calls (not processing/done events)
  const toolCallGroups = groups.filter(g => g.statuses[0].phase === 'calling')
  const toolCount = toolCallGroups.length
  const errorCount = toolCallGroups.filter(g =>
    g.statuses.some(s => s.phase === 'error')
  ).length
  const lastStatus = statuses[statuses.length - 1]
  const isActive = isStreaming && lastStatus.phase !== 'done'

  // Check if any tool calls were interrupted (not completed)
  const hasInterrupted = !isActive && toolCallGroups.some(g => {
    const last = g.statuses[g.statuses.length - 1]
    return last.phase !== 'result' && last.phase !== 'error'
  })

  // Build summary text
  const isGenerating = lastStatus.phase === 'generating'
  const summaryParts: string[] = []
  if (isGenerating) {
    summaryParts.push('Generating tool call\u2026')
  } else if (isActive) {
    summaryParts.push(`Using ${toolCount} tool${toolCount !== 1 ? 's' : ''}\u2026`)
  } else if (hasInterrupted) {
    summaryParts.push(`${toolCount} tool${toolCount !== 1 ? 's' : ''} interrupted`)
  } else {
    summaryParts.push(`Used ${toolCount} tool${toolCount !== 1 ? 's' : ''}`)
  }
  if (errorCount > 0) {
    summaryParts.push(`(${errorCount} failed)`)
  }
  const summary = summaryParts.join(' ')

  // Unique tool names for compact display
  const toolNames = [...new Set(toolCallGroups.map(g => g.name))].filter(Boolean)

  return (
    <div className={`my-2 rounded-lg border overflow-hidden transition-all duration-200 ${
      isActive ? 'border-warning/40 border-l-warning border-l-2' : 'border-border'
    } bg-popover`}>
      {/* Compact header — always visible */}
      <button
        onClick={() => setSectionOpen(!sectionOpen)}
        className="w-full px-3 py-1.5 flex items-center gap-2 text-xs hover:bg-accent/30 transition-colors"
      >
        {isGenerating ? (
          <Loader2 className="h-3.5 w-3.5 text-primary animate-spin flex-shrink-0" />
        ) : isActive ? (
          <Loader2 className="h-3.5 w-3.5 text-warning animate-spin flex-shrink-0" />
        ) : errorCount > 0 ? (
          <AlertTriangle className="h-3.5 w-3.5 text-warning flex-shrink-0" />
        ) : hasInterrupted ? (
          <StopCircle className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
        ) : (
          <Check className="h-3.5 w-3.5 text-primary flex-shrink-0" />
        )}

        <span className="text-muted-foreground">{summary}</span>

        {/* Show tool names in compact mode */}
        {!sectionOpen && toolNames.length > 0 && (
          <span className="font-mono text-foreground/70 truncate">
            {toolNames.join(', ')}
          </span>
        )}

        <ChevronRight className={`ml-auto h-3 w-3 text-muted-foreground/50 flex-shrink-0 transition-transform duration-150 ${sectionOpen ? 'rotate-90' : ''}`} />
      </button>

      {/* Expanded detail view */}
      {sectionOpen && (
        <div className="px-3 pb-2 text-xs border-t border-border/50">
          {groups.map((group, gi) => {
            if (group.statuses[0].phase === 'generating') {
              return (
                <div key={gi} className="flex items-center gap-2 text-muted-foreground py-1">
                  {isActive && lastStatus.phase === 'generating' ? (
                    <>
                      <Loader2 className="h-3 w-3 text-primary animate-spin" />
                      <span>Generating tool call\u2026</span>
                    </>
                  ) : (
                    <>
                      <Check className="h-3 w-3 text-primary" />
                      <span>Tool call generated</span>
                    </>
                  )}
                </div>
              )
            }

            if (group.statuses[0].phase === 'processing') {
              return (
                <div key={gi} className="flex items-center gap-2 text-muted-foreground py-1">
                  {isActive && lastStatus.phase === 'processing' ? (
                    <>
                      <Loader2 className="h-3 w-3 text-warning animate-spin" />
                      <span>Processing tool results\u2026</span>
                    </>
                  ) : (
                    <>
                      <Check className="h-3 w-3 text-primary" />
                      <span>Tool results processed</span>
                    </>
                  )}
                </div>
              )
            }

            if (group.statuses[0].phase === 'done') return null

            const lastPhase = group.statuses[group.statuses.length - 1]
            const isDone = lastPhase.phase === 'result' || lastPhase.phase === 'error'
            const isItemExpanded = expanded[gi] ?? false

            const callingStatus = group.statuses.find(s => s.phase === 'calling')
            const resultStatus = group.statuses.find(s => s.phase === 'result' || s.phase === 'error')

            return (
              <div key={gi} className="py-1">
                <button
                  onClick={() => setExpanded(prev => ({ ...prev, [gi]: !prev[gi] }))}
                  className="flex items-center gap-2 w-full text-left hover:text-foreground transition-colors"
                >
                  {isDone ? (
                    resultStatus?.phase === 'error'
                      ? <X className="h-3 w-3 text-destructive flex-shrink-0" />
                      : <Check className="h-3 w-3 text-primary flex-shrink-0" />
                  ) : isActive ? (
                    <Loader2 className="h-3 w-3 text-warning animate-spin flex-shrink-0" />
                  ) : (
                    <StopCircle className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                  )}

                  <span className="font-mono text-foreground">{group.name}</span>

                  <span className="text-muted-foreground">
                    {!isDone && !isActive ? 'interrupted' :
                      lastPhase.phase === 'calling' ? 'detected' :
                      lastPhase.phase === 'executing' ? 'running\u2026' :
                      lastPhase.phase === 'result' ? 'done' :
                      lastPhase.phase === 'error' ? 'failed' : ''}
                  </span>

                  <ChevronRight className={`ml-auto h-3 w-3 text-muted-foreground/50 flex-shrink-0 transition-transform duration-150 ${isItemExpanded ? 'rotate-90' : ''}`} />
                </button>

                {isItemExpanded && (
                  <div className="ml-5 mt-1 space-y-1">
                    {callingStatus?.detail && (
                      <div>
                        <span className="text-muted-foreground text-[10px] uppercase">args</span>
                        <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[100px] overflow-y-auto">
                          {formatJson(callingStatus.detail)}
                        </pre>
                      </div>
                    )}
                    {resultStatus?.detail && (
                      <div>
                        <span className={`text-[10px] uppercase ${resultStatus.phase === 'error' ? 'text-destructive' : 'text-muted-foreground'}`}>
                          {resultStatus.phase === 'error' ? 'error' : 'result'}
                        </span>
                        <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap break-words">
                          {resultStatus.detail}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
