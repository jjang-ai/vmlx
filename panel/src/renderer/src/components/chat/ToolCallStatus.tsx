import { useState, useEffect } from 'react'
import { formatJson } from './InlineToolCall'

export interface ToolStatus {
  phase: 'calling' | 'executing' | 'result' | 'error' | 'processing' | 'done'
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
    } else if (s.phase === 'processing') {
      // Collapse multiple processing events into one
      if (lastProcessingIdx >= 0) {
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

  // Build summary text
  const summaryParts: string[] = []
  if (isActive) {
    summaryParts.push(`Using ${toolCount} tool${toolCount !== 1 ? 's' : ''}...`)
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
    <div className={`my-2 rounded border overflow-hidden transition-all duration-200 ${
      isActive ? 'border-warning/40 border-l-warning border-l-2' : 'border-border'
    } bg-popover`}>
      {/* Compact header — always visible */}
      <button
        onClick={() => setSectionOpen(!sectionOpen)}
        className="w-full px-3 py-1.5 flex items-center gap-2 text-xs hover:bg-accent/30 transition-colors"
      >
        {isActive ? (
          <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse flex-shrink-0" />
        ) : errorCount > 0 ? (
          <span className="text-warning flex-shrink-0">&#9888;</span>
        ) : (
          <span className="text-primary flex-shrink-0">&#10003;</span>
        )}

        <span className="text-muted-foreground">{summary}</span>

        {/* Show tool names in compact mode */}
        {!sectionOpen && toolNames.length > 0 && (
          <span className="font-mono text-foreground/70 truncate">
            {toolNames.join(', ')}
          </span>
        )}

        <span className="ml-auto text-[10px] text-muted-foreground opacity-60 flex-shrink-0">
          {sectionOpen ? '[-]' : '[+]'}
        </span>
      </button>

      {/* Expanded detail view */}
      {sectionOpen && (
        <div className="px-3 pb-2 text-xs border-t border-border/50">
          {groups.map((group, gi) => {
            if (group.statuses[0].phase === 'processing') {
              return (
                <div key={gi} className="flex items-center gap-2 text-muted-foreground py-1">
                  {isActive && lastStatus.phase === 'processing' ? (
                    <>
                      <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse" />
                      <span>Processing tool results...</span>
                    </>
                  ) : (
                    <>
                      <span className="text-primary">&#10003;</span>
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

            // Get args from 'calling' phase
            const callingStatus = group.statuses.find(s => s.phase === 'calling')
            const resultStatus = group.statuses.find(s => s.phase === 'result' || s.phase === 'error')

            return (
              <div key={gi} className="py-1">
                <button
                  onClick={() => setExpanded(prev => ({ ...prev, [gi]: !prev[gi] }))}
                  className="flex items-center gap-2 w-full text-left hover:text-foreground transition-colors"
                >
                  {/* Phase indicator */}
                  {isDone ? (
                    resultStatus?.phase === 'error'
                      ? <span className="text-destructive text-[10px]">&#10007;</span>
                      : <span className="text-primary text-[10px]">&#10003;</span>
                  ) : (
                    <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse" />
                  )}

                  <span className="font-mono text-foreground">{group.name}</span>

                  {/* Current phase label */}
                  <span className="text-muted-foreground">
                    {lastPhase.phase === 'calling' && 'detected'}
                    {lastPhase.phase === 'executing' && 'running...'}
                    {lastPhase.phase === 'result' && 'done'}
                    {lastPhase.phase === 'error' && 'failed'}
                  </span>

                  <span className="ml-auto text-[10px] text-muted-foreground opacity-60">
                    {isItemExpanded ? '[-]' : '[+]'}
                  </span>
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

