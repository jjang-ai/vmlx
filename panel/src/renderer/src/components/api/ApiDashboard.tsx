import { useState, useEffect, useMemo, useRef } from 'react'
import { Server, Copy, Check, Wifi, WifiOff } from 'lucide-react'
import { useSessionsContext } from '../../contexts/SessionsContext'
import { EndpointList } from './EndpointList'
import { CodeSnippets } from './CodeSnippets'
import { CodingToolIntegration } from './CodingToolIntegration'

export type ApiFormat = 'openai' | 'anthropic' | 'ollama'

interface SessionSummary {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  standbyDepth?: 'soft' | 'deep' | null
  type?: 'local' | 'remote'
  config?: string
  remoteUrl?: string
}

function getModelDisplayName(s: SessionSummary): string {
  try {
    const cfg = JSON.parse(s.config || '{}')
    if (cfg.servedModelName) return cfg.servedModelName
  } catch {}
  return s.modelName || s.modelPath?.split('/').pop() || 'unknown'
}

function getModelType(s: SessionSummary): string {
  try {
    const cfg = JSON.parse(s.config || '{}')
    if (cfg.modelType === 'image') return cfg.imageMode === 'edit' ? 'image-edit' : 'image-gen'
    if (cfg.isMllm) return 'vision'
  } catch {}
  return 'text'
}

export function ApiDashboard() {
  const { sessions } = useSessionsContext()
  const runningSessions = useMemo(
    () => (sessions as SessionSummary[]).filter(s => s.status === 'running' || s.status === 'standby'),
    [sessions]
  )

  const [gwPort, setGwPort] = useState('8080')
  const [portInput, setPortInput] = useState('8080')
  const [portError, setPortError] = useState<string | null>(null)
  const [format, setFormat] = useState<ApiFormat>('openai')
  const portRef = useRef<HTMLInputElement>(null)

  // Load gateway port on mount
  useEffect(() => {
    window.api.gateway?.getStatus?.().then((status: any) => {
      if (status?.port) {
        setGwPort(String(status.port))
        setPortInput(String(status.port))
      }
    }).catch(() => {})
  }, [])

  const gatewayUrl = `http://localhost:${gwPort}`
  const hasModels = runningSessions.length > 0

  // First model name for code snippets
  const firstModelName = hasModels ? getModelDisplayName(runningSessions[0]) : 'your-model-name'

  const handlePortSubmit = async () => {
    const port = parseInt(portInput, 10)
    if (isNaN(port) || port < 1024 || port > 65535) {
      setPortError('Port must be 1024-65535')
      return
    }
    setPortError(null)
    try {
      await window.api.gateway?.setPort?.(port)
      setGwPort(String(port))
    } catch (err: any) {
      setPortError(err?.message || 'Failed to change port')
    }
  }

  const handlePortKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handlePortSubmit()
  }

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold">API Gateway</h1>
          <p className="text-sm text-muted-foreground mt-1">
            All models are accessible through a single endpoint. Route by model name in your request.
          </p>
          <p className="text-[9px] text-muted-foreground/40 mt-0.5">MLX Studio by Jinho Jang &middot; mlx.studio</p>
        </div>

        {/* Gateway connection card */}
        <div className="p-4 rounded-lg border border-border bg-card space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {hasModels
                ? <Wifi className="h-4 w-4 text-green-500" />
                : <WifiOff className="h-4 w-4 text-muted-foreground/50" />
              }
              <h3 className="text-sm font-medium">Gateway</h3>
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${hasModels ? 'bg-green-500/15 text-green-500' : 'bg-muted text-muted-foreground'}`}>
                {hasModels ? `${runningSessions.length} model${runningSessions.length !== 1 ? 's' : ''}` : 'No models'}
              </span>
            </div>
          </div>

          <div className="space-y-1.5">
            {/* Gateway URL */}
            <CopyRow label="URL" value={gatewayUrl} />

            {/* Port editor */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground w-20 flex-shrink-0">Port</span>
              <input
                ref={portRef}
                type="number"
                min={1024}
                max={65535}
                value={portInput}
                onChange={e => { setPortInput(e.target.value); setPortError(null) }}
                onBlur={handlePortSubmit}
                onKeyDown={handlePortKeyDown}
                className="w-24 px-2 py-1 bg-background rounded border border-border font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              />
              {portError && <span className="text-[10px] text-red-400">{portError}</span>}
            </div>

            {/* OpenAI base URL */}
            <CopyRow label="OpenAI" value={`${gatewayUrl}/v1`} />

            {/* Anthropic base */}
            <CopyRow label="Anthropic" value={gatewayUrl} />

            {/* Ollama host */}
            <CopyRow label="Ollama" value={`OLLAMA_HOST=${gatewayUrl}`} />
          </div>
        </div>

        {/* Live model list */}
        {hasModels && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Running Models</h3>
            <div className="border border-border rounded-lg divide-y divide-border overflow-hidden">
              {runningSessions.map(s => {
                const name = getModelDisplayName(s)
                const type = getModelType(s)
                const isSleeping = s.status === 'standby'
                return (
                  <div key={s.id} className="px-3 py-2 flex items-center gap-2 text-xs">
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isSleeping ? 'bg-blue-400' : 'bg-green-500'}`} />
                    <span className="font-medium flex-1 truncate">{name}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{type}</span>
                    {isSleeping && <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400">sleeping</span>}
                    <span className="text-muted-foreground/60 font-mono">:{s.port}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Format toggle */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground mr-2">Format:</span>
          {(['openai', 'anthropic', 'ollama'] as ApiFormat[]).map(f => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                format === f
                  ? 'border-primary bg-primary/10 text-primary font-medium'
                  : 'border-border hover:bg-accent text-muted-foreground'
              }`}
            >
              {f === 'openai' ? 'OpenAI' : f === 'anthropic' ? 'Anthropic' : 'Ollama'}
            </button>
          ))}
        </div>

        {/* Code snippets */}
        <CodeSnippets
          baseUrl={gatewayUrl}
          apiKey={null}
          modelId={firstModelName}
          format={format}
        />

        {/* Coding tool integration */}
        {format === 'openai' && (
          <CodingToolIntegration baseUrl={gatewayUrl} modelName={firstModelName} port={parseInt(gwPort, 10)} />
        )}

        {/* Endpoint reference */}
        <EndpointList format={format} />
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
