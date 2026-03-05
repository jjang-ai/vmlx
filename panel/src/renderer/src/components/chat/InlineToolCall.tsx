import { useState } from 'react'

export interface InlineToolGroup {
  name: string
  statuses: Array<{ phase: string; toolName: string; detail?: string; iteration?: number }>
}

interface InlineToolCallProps {
  group: InlineToolGroup
  isStreaming: boolean
}

export function InlineToolCall({ group, isStreaming }: InlineToolCallProps) {
  const [expanded, setExpanded] = useState(false)

  const lastPhase = group.statuses[group.statuses.length - 1]
  const isDone = lastPhase.phase === 'result' || lastPhase.phase === 'error' || lastPhase.phase === 'done'
  const isError = group.statuses.some(s => s.phase === 'error')
  const callingStatus = group.statuses.find(s => s.phase === 'calling')
  const resultStatus = group.statuses.find(s => s.phase === 'result' || s.phase === 'error')

  return (
    <div className={`my-1.5 rounded border overflow-hidden transition-all duration-150 ${
      !isDone && isStreaming ? 'border-warning/40 border-l-warning border-l-2' : 'border-border/60'
    } bg-popover/80`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-2.5 py-1 flex items-center gap-1.5 text-xs hover:bg-accent/30 transition-colors"
      >
        {isDone ? (
          isError
            ? <span className="text-destructive text-[10px]">&#10007;</span>
            : <span className="text-primary text-[10px]">&#10003;</span>
        ) : (
          <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse" />
        )}

        <span className="font-mono text-foreground/80">{group.name}</span>

        <span className="text-muted-foreground text-[11px]">
          {lastPhase.phase === 'calling' && 'detected'}
          {lastPhase.phase === 'asking' && 'waiting for input...'}
          {lastPhase.phase === 'executing' && 'running...'}
          {lastPhase.phase === 'result' && ''}
          {lastPhase.phase === 'error' && 'failed'}
          {lastPhase.phase === 'done' && ''}
        </span>

        <span className="ml-auto text-[10px] text-muted-foreground opacity-50">
          {expanded ? '[-]' : '[+]'}
        </span>
      </button>

      {expanded && (
        <div className="px-2.5 pb-1.5 text-xs border-t border-border/40">
          {callingStatus?.detail && (
            <div className="mt-1">
              <span className="text-muted-foreground text-[10px] uppercase">args</span>
              <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[80px] overflow-y-auto">
                {formatJson(callingStatus.detail)}
              </pre>
            </div>
          )}
          {resultStatus?.detail && (
            <div className="mt-1">
              <span className={`text-[10px] uppercase ${resultStatus.phase === 'error' ? 'text-destructive' : 'text-muted-foreground'}`}>
                {resultStatus.phase === 'error' ? 'error' : 'result'}
              </span>
              <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[200px] overflow-y-auto whitespace-pre-wrap break-words">
                {resultStatus.detail}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function formatJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2)
  } catch {
    return s
  }
}
