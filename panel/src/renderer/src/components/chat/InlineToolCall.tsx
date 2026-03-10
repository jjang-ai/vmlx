import { useState, useMemo } from 'react'

export interface InlineToolGroup {
  name: string
  statuses: Array<{ phase: string; toolName: string; detail?: string; iteration?: number }>
}

interface InlineToolCallProps {
  group: InlineToolGroup
  isStreaming: boolean
}

/** Parse JSON args string into object, returns null on failure */
function parseArgs(detail?: string): Record<string, any> | null {
  if (!detail) return null
  try { return JSON.parse(detail) } catch { return null }
}

/** Get a human-readable summary for the tool call header */
function getToolSummary(name: string, args: Record<string, any> | null): { label: string; context: string } {
  if (!args) return { label: name, context: '' }

  switch (name) {
    case 'edit_file':
      return { label: 'Edit', context: args.path || '' }
    case 'read_file':
      return { label: 'Read', context: args.path ? `${args.path}${args.offset ? `:${args.offset}` : ''}` : '' }
    case 'write_file':
      return { label: 'Write', context: args.path || '' }
    case 'patch_file':
      return { label: 'Patch', context: args.path || '' }
    case 'batch_edit':
      return { label: 'Batch Edit', context: args.path || '' }
    case 'insert_text':
      return { label: 'Insert', context: args.path ? `${args.path}:${args.line || ''}` : '' }
    case 'replace_lines':
      return { label: 'Replace Lines', context: args.path ? `${args.path}:${args.start_line}-${args.end_line}` : '' }
    case 'run_command':
      return { label: '$', context: truncate(args.command, 80) }
    case 'git':
      return { label: '$ git', context: truncate(args.command, 80) }
    case 'search_files':
      return { label: 'Search', context: `"${truncate(args.pattern, 40)}"${args.path && args.path !== '.' ? ` in ${args.path}` : ''}` }
    case 'find_files':
      return { label: 'Find', context: `"${truncate(args.pattern, 40)}"${args.path && args.path !== '.' ? ` in ${args.path}` : ''}` }
    case 'list_directory':
      return { label: 'List', context: args.path || '.' }
    case 'get_tree':
      return { label: 'Tree', context: args.path || '.' }
    case 'delete_file':
      return { label: 'Delete', context: args.path || '' }
    case 'move_file':
      return { label: 'Move', context: args.source && args.destination ? `${args.source} → ${args.destination}` : args.source || '' }
    case 'copy_file':
      return { label: 'Copy', context: args.source && args.destination ? `${args.source} → ${args.destination}` : args.source || '' }
    case 'create_directory':
      return { label: 'Mkdir', context: args.path || '' }
    case 'file_info':
      return { label: 'Info', context: args.path || '' }
    case 'diff_files':
      return { label: 'Diff', context: args.path_a && args.path_b ? `${args.path_a} ↔ ${args.path_b}` : args.path_a || args.path_b || '' }
    case 'apply_regex':
      return { label: 'Regex', context: args.path || args.glob || '' }
    case 'web_search':
    case 'ddg_search':
      return { label: 'Search Web', context: `"${truncate(args.query, 50)}"` }
    case 'fetch_url':
      return { label: 'Fetch', context: truncate(args.url, 60) }
    case 'spawn_process':
      return { label: 'Spawn', context: truncate(args.command, 60) }
    case 'get_process_output':
      return { label: 'Process Output', context: `PID ${args.pid}` }
    case 'ask_user':
      return { label: 'Ask User', context: truncate(args.question, 50) }
    case 'read_image':
      return { label: 'Read Image', context: args.path || '' }
    case 'count_tokens':
      return { label: 'Count Tokens', context: '' }
    case 'clipboard_read':
      return { label: 'Clipboard Read', context: '' }
    case 'clipboard_write':
      return { label: 'Clipboard Write', context: '' }
    case 'get_current_datetime':
      return { label: 'Datetime', context: '' }
    case 'get_diagnostics':
      return { label: 'Diagnostics', context: args.path || '' }
    default:
      return { label: name, context: '' }
  }
}

function truncate(s: string, max: number): string {
  if (!s) return ''
  return s.length > max ? s.slice(0, max) + '…' : s
}

export function InlineToolCall({ group, isStreaming }: InlineToolCallProps) {
  const [expanded, setExpanded] = useState(false)

  const lastPhase = group.statuses[group.statuses.length - 1]
  const isDone = lastPhase.phase === 'result' || lastPhase.phase === 'error' || lastPhase.phase === 'done'
  const isError = group.statuses.some(s => s.phase === 'error')
  const callingStatus = group.statuses.find(s => s.phase === 'calling')
  const resultStatus = group.statuses.find(s => s.phase === 'result' || s.phase === 'error')
  const args = useMemo(() => parseArgs(callingStatus?.detail), [callingStatus?.detail])
  const summary = useMemo(() => getToolSummary(group.name, args), [group.name, args])

  return (
    <div className={`my-1.5 rounded border overflow-hidden transition-all duration-150 ${
      !isDone && isStreaming ? 'border-warning/40 border-l-warning border-l-2' : 'border-border/60'
    } bg-popover/80`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-2.5 py-1.5 flex items-center gap-1.5 text-xs hover:bg-accent/30 transition-colors"
      >
        {/* Status indicator */}
        {isDone ? (
          isError
            ? <span className="text-destructive text-[11px]">&#10007;</span>
            : <span className="text-primary text-[11px]">&#10003;</span>
        ) : (
          <span className="w-1.5 h-1.5 bg-warning rounded-full animate-pulse flex-shrink-0" />
        )}

        {/* Tool label */}
        <span className="font-medium text-foreground/90">{summary.label}</span>

        {/* Context (file path, command, etc.) */}
        {summary.context && (
          <span className="font-mono text-muted-foreground truncate" title={summary.context}>
            {summary.context}
          </span>
        )}

        {/* Phase label (only when not done) */}
        {!isDone && (
          <span className="text-muted-foreground text-[11px] flex-shrink-0">
            {lastPhase.phase === 'calling' && 'detected'}
            {lastPhase.phase === 'asking' && 'waiting…'}
            {lastPhase.phase === 'executing' && 'running…'}
          </span>
        )}

        {/* Error label */}
        {isError && <span className="text-destructive text-[11px] flex-shrink-0">failed</span>}

        <span className="ml-auto text-[10px] text-muted-foreground opacity-50 flex-shrink-0">
          {expanded ? '▾' : '▸'}
        </span>
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
          <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words">
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
        <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words">
          {resultDetail}
        </pre>
      </div>
    )
  }

  // Default: show args + result (current behavior, cleaned up)
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
          <pre className="text-[11px] text-muted-foreground bg-background/50 rounded px-2 py-1 overflow-x-auto max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words">
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
    <div className="rounded border border-border/40 overflow-hidden text-[11px] font-mono">
      <div className="flex">
        {/* Old (removed) — left side */}
        {hasOld && (
          <div className={`${hasNew ? 'w-1/2 border-r border-border/30' : 'w-full'} overflow-x-auto`}>
            <div className="px-1 py-0.5 bg-destructive/10 text-destructive/60 text-[10px] border-b border-border/30">removed</div>
            <div className="max-h-[250px] overflow-y-auto">
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
            <div className="px-1 py-0.5 bg-primary/10 text-primary/60 text-[10px] border-b border-border/30">added</div>
            <div className="max-h-[250px] overflow-y-auto">
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

export function formatJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s), null, 2)
  } catch {
    return s
  }
}
