import { useState, useEffect, useMemo } from 'react'
import { Server, Copy, Check, ExternalLink } from 'lucide-react'
import { useSessionsContext } from '../../contexts/SessionsContext'
import { EndpointList } from './EndpointList'
import { CodeSnippets } from './CodeSnippets'

interface SessionSummary {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  type?: 'local' | 'remote'
  config?: string
}

function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

export function ApiDashboard() {
  const { sessions } = useSessionsContext()
  const runningSessions = useMemo(
    () => (sessions as SessionSummary[]).filter(s => s.status === 'running'),
    [sessions]
  )
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Auto-select first running session
  useEffect(() => {
    if (!selectedId && runningSessions.length > 0) {
      setSelectedId(runningSessions[0].id)
    } else if (selectedId && !runningSessions.find(s => s.id === selectedId)) {
      setSelectedId(runningSessions[0]?.id ?? null)
    }
  }, [runningSessions, selectedId])

  const selected = runningSessions.find(s => s.id === selectedId) ?? null
  const baseUrl = selected ? `http://${connectHost(selected.host)}:${selected.port}` : null
  const modelName = selected?.modelName || selected?.modelPath?.split('/').pop() || null

  // Detect if selected session is an image server
  const isImageServer = useMemo(() => {
    if (!selected?.config) return false
    try {
      const cfg = JSON.parse(selected.config as string)
      return cfg.modelType === 'image'
    } catch { return false }
  }, [selected])

  // Extract API key from config JSON
  const apiKey = useMemo(() => {
    if (!selected?.config) return null
    try {
      const cfg = JSON.parse(selected.config as string)
      return cfg.apiKey || null
    } catch { return null }
  }, [selected])

  if (runningSessions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-8">
        <Server className="h-12 w-12 text-muted-foreground/30 mb-4" />
        <h2 className="text-lg font-medium mb-2">No Running Models</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Start a model session in <strong>Server</strong> mode to see your local API endpoints.
          The API is OpenAI-compatible and also supports the Anthropic Messages format.
        </p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold">API Reference</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Your local inference server exposes an OpenAI-compatible API. Connect any app, SDK, or tool.
          </p>
        </div>

        {/* Session selector pills */}
        <div className="flex gap-2 flex-wrap">
          {runningSessions.map(s => (
            <button
              key={s.id}
              onClick={() => setSelectedId(s.id)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors flex items-center gap-1.5 ${
                selectedId === s.id
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:bg-accent text-muted-foreground'
              }`}
            >
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500" />
              <span className="font-medium">{s.modelName || s.modelPath?.split('/').pop()}</span>
              <span className="opacity-60">:{s.port}</span>
            </button>
          ))}
        </div>

        {/* Connection info */}
        {baseUrl && (
          <div className="p-4 rounded-lg border border-border bg-card space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Connection</h3>
              {apiKey && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/15 text-green-500">Key set</span>
              )}
            </div>
            <div className="space-y-1.5">
              <CopyRow label="Base URL" value={baseUrl} />
              <CopyRow label="OpenAI Base" value={`${baseUrl}/v1`} />
              {modelName && <CopyRow label="Model" value={modelName} />}
              {apiKey && <CopyRow label="API Key" value={apiKey} masked />}
            </div>
          </div>
        )}

        {/* Code snippets */}
        {baseUrl && (
          <CodeSnippets baseUrl={baseUrl} apiKey={apiKey} modelId={modelName} isImage={isImageServer} />
        )}

        {/* Endpoint reference */}
        {baseUrl && (
          <EndpointList baseUrl={baseUrl} isImage={isImageServer} />
        )}
      </div>
    </div>
  )
}

function CopyRow({ label, value, masked }: { label: string; value: string; masked?: boolean }) {
  const [copied, setCopied] = useState(false)
  const display = masked ? `${value.slice(0, 4)}${'*'.repeat(Math.max(0, value.length - 8))}${value.slice(-4)}` : value

  const handleCopy = () => {
    navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-muted-foreground w-20 flex-shrink-0">{label}</span>
      <code className="flex-1 px-2 py-1 bg-background rounded border border-border font-mono truncate">
        {display}
      </code>
      <button
        onClick={handleCopy}
        className="p-1 text-muted-foreground hover:text-foreground rounded transition-colors flex-shrink-0"
        title="Copy"
      >
        {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
      </button>
    </div>
  )
}
