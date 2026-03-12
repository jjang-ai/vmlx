import { useState, useMemo } from 'react'
import { Check, X, Square, ChevronRight, Loader2 } from 'lucide-react'
import { parseToolArgs, getToolSummary, formatJson } from './chat-utils'

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

  if (group.statuses.length === 0) return null

  const lastPhase = group.statuses[group.statuses.length - 1]
  const isDone = lastPhase.phase === 'result' || lastPhase.phase === 'error' || lastPhase.phase === 'done'
  const isError = group.statuses.some(s => s.phase === 'error')
  const callingStatus = group.statuses.find(s => s.phase === 'calling')
  const resultStatus = group.statuses.find(s => s.phase === 'result' || s.phase === 'error')
  const args = useMemo(() => parseToolArgs(callingStatus?.detail), [callingStatus?.detail])
  const summary = useMemo(() => getToolSummary(group.name, args), [group.name, args])
  const iteration = callingStatus?.iteration

  return (
    <div className={`my-1.5 rounded-lg border overflow-hidden transition-all duration-150 ${
      !isDone && isStreaming ? 'border-warning/40 border-l-warning border-l-2' : 'border-border/60'
    } bg-popover/80`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-2.5 py-1.5 flex items-center gap-1.5 text-xs hover:bg-accent/30 transition-colors"
      >
        {/* Status indicator */}
        {isDone ? (
          isError
            ? <X className="h-3.5 w-3.5 text-destructive flex-shrink-0" />
            : <Check className="h-3.5 w-3.5 text-primary flex-shrink-0" />
        ) : isStreaming ? (
          <Loader2 className="h-3.5 w-3.5 text-warning animate-spin flex-shrink-0" />
        ) : (
          <Square className="h-3 w-3 text-muted-foreground flex-shrink-0" />
        )}

        {/* Tool label */}
        <span className="font-medium text-foreground/90">{summary.label}</span>

        {/* Context (file path, command, etc.) */}
        {summary.context && (
          <span className="font-mono text-muted-foreground truncate" title={summary.context}>
            {summary.context}
          </span>
        )}

        {/* Iteration badge */}
        {iteration != null && iteration > 1 && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-accent text-muted-foreground flex-shrink-0">
            #{iteration}
          </span>
        )}

        {/* Phase label (only when not done) */}
        {!isDone && (
          <span className="text-muted-foreground text-[11px] flex-shrink-0">
            {!isStreaming ? 'interrupted' :
              lastPhase.phase === 'calling' ? 'detected' :
              lastPhase.phase === 'asking' ? 'waiting\u2026' :
              lastPhase.phase === 'executing' ? 'running\u2026' : ''}
          </span>
        )}

        {/* Error label */}
        {isError && <span className="text-destructive text-[11px] flex-shrink-0">failed</span>}

        <ChevronRight className={`ml-auto h-3 w-3 text-muted-foreground/50 flex-shrink-0 transition-transform duration-150 ${expanded ? 'rotate-90' : ''}`} />
      </button>

      {expanded && (
        <div className="border-t border-border/40">
          <ToolCallBody
            toolName={group.name}
            args={args}
            callingDetail={callingStatus?.detail}
            resultDetail={resultStatus?.detail}
            isError={isError}
          />
        </div>
      )}
    </div>
  )
}

/** Renders the expanded body content — dispatches to tool-specific renderers */
function ToolCallBody({ toolName, args, callingDetail, resultDetail, isError }: {
  toolName: string
  args: Record<string, any> | null
  callingDetail?: string
  resultDetail?: string
  isError: boolean
}) {
  // Edit tools: show diff
  if (toolName === 'edit_file' && args?.search_text != null && args?.replacement_text != null) {
    return (
      <div className="px-2.5 py-2">
        {args.path && <div className="text-[10px] text-muted-foreground/70 mb-1.5 font-mono">{args.path}</div>}
        <DiffView oldText={args.search_text} newText={args.replacement_text} />
        {resultDetail && (
          <div className="mt-1.5 text-[11px] text-muted-foreground/70">{resultDetail}</div>
        )}
      </div>
    )
  }

  // Insert text: show what's being inserted
  if (toolName === 'insert_text' && args?.text) {
    return (
      <div className="px-2.5 py-2">
        {args.path && <div className="text-[10px] text-muted-foreground/70 mb-1.5 font-mono">{args.path}:{args.line}</div>}
        <DiffView oldText="" newText={args.text} />
        {resultDetail && <div className="mt-1.5 text-[11px] text-muted-foreground/70">{resultDetail}</div>}
      </div>
    )
  }

  // Replace lines: show what's being replaced
  if (toolName === 'replace_lines' && args?.text) {
    return (
      <div className="px-2.5 py-2">
        {args.path && <div className="text-[10px] text-muted-foreground/70 mb-1.5 font-mono">{args.path}:{args.start_line}-{args.end_line}</div>}
        <DiffView oldText="" newText={args.text} />
        {resultDetail && <div className="mt-1.5 text-[11px] text-muted-foreground/70">{resultDetail}</div>}
      </div>
    )
  }

  // Batch edit: show each edit as a diff
  if (toolName === 'batch_edit' && args?.edits && Array.isArray(args.edits)) {
    return (
      <div className="px-2.5 py-2 space-y-2">
        {args.path && <div className="text-[10px] text-muted-foreground/70 font-mono">{args.path}</div>}
        {args.edits.map((edit: any, i: number) => (
          <DiffView key={i} oldText={edit.search_text || ''} newText={edit.replacement_text || ''} />
        ))}
        {resultDetail && <div className="mt-1.5 text-[11px] text-muted-foreground/70">{resultDetail}</div>}
      </div>
    )
  }

  // Command tools: show command + output
  if ((toolName === 'run_command' || toolName === 'git' || toolName === 'spawn_process') && args?.command) {
    return (
      <div className="px-2.5 py-2">
        <pre className="text-[11px] font-mono text-foreground/80 bg-background/50 rounded px-2 py-1 mb-1.5 overflow-x-auto">
          <span className="text-muted-foreground">$ </span>{toolName === 'git' ? `git ${args.command}` : args.command}
        </pre>
        {resultDetail && (
          <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap break-words">
            {resultDetail}
          </pre>
        )}
      </div>
    )
  }

  // Write file: show content preview
  if (toolName === 'write_file' && args?.content) {
    return (
      <div className="px-2.5 py-2">
        {args.path && <div className="text-[10px] text-muted-foreground/70 mb-1.5 font-mono">{args.path}</div>}
        <DiffView oldText="" newText={args.content} />
        {resultDetail && <div className="mt-1.5 text-[11px] text-muted-foreground/70">{resultDetail}</div>}
      </div>
    )
  }

  // Read file: show file content from result
  if (toolName === 'read_file' && resultDetail) {
    return (
      <div className="px-2.5 py-2">
        <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap break-words">
          {resultDetail}
        </pre>
      </div>
    )
  }

  // Default: show args + result
  return (
    <div className="px-2.5 pb-1.5 text-xs">
      {callingDetail && (
        <div className="mt-1">
          <span className="text-muted-foreground text-[10px] uppercase">args</span>
          <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[120px] overflow-y-auto whitespace-pre-wrap break-words">
            {formatJson(callingDetail)}
          </pre>
        </div>
      )}
      {resultDetail && (
        <div className="mt-1">
          <span className={`text-[10px] uppercase ${isError ? 'text-destructive' : 'text-muted-foreground'}`}>
            {isError ? 'error' : 'result'}
          </span>
          <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap break-words">
            {resultDetail}
          </pre>
        </div>
      )}
    </div>
  )
}

/** Side-by-side diff view: red for removed, green for added */
function DiffView({ oldText, newText }: { oldText: string; newText: string }) {
  const oldLines = oldText ? oldText.split('\n') : []
  const newLines = newText ? newText.split('\n') : []
  const hasOld = oldLines.length > 0 && oldText.length > 0
  const hasNew = newLines.length > 0 && newText.length > 0

  return (
    <div className="rounded-lg border border-border/40 overflow-hidden text-[11px] font-mono">
      <div className="flex">
        {/* Old (removed) — left side */}
        {hasOld && (
          <div className={`${hasNew ? 'w-1/2 border-r border-border/30' : 'w-full'} overflow-x-auto`}>
            <div className="px-1.5 py-0.5 bg-destructive/10 text-destructive/60 text-[10px] border-b border-border/30">removed</div>
            <div className="max-h-[400px] overflow-y-auto">
              {oldLines.map((line, i) => (
                <div key={i} className="px-2 py-px bg-destructive/5 text-destructive/80 whitespace-pre-wrap break-all">
                  <span className="text-destructive/40 select-none mr-1">-</span>{line}
                </div>
              ))}
            </div>
          </div>
        )}
        {/* New (added) — right side */}
        {hasNew && (
          <div className={`${hasOld ? 'w-1/2' : 'w-full'} overflow-x-auto`}>
            <div className="px-1.5 py-0.5 bg-primary/10 text-primary/60 text-[10px] border-b border-border/30">added</div>
            <div className="max-h-[400px] overflow-y-auto">
              {newLines.map((line, i) => (
                <div key={i} className="px-2 py-px bg-primary/5 text-primary/80 whitespace-pre-wrap break-all">
                  <span className="text-primary/40 select-none mr-1">+</span>{line}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
