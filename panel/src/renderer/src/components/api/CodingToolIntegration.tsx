// MLX Studio — Coding Tool Integration
import { useState, useEffect, useCallback } from 'react'
import { Download, Check, X, Plus, Trash2, RefreshCw, ExternalLink } from 'lucide-react'

interface ToolStatus {
  installed: boolean
  configured: boolean
  configPath: string
  entries: Array<{ label: string; baseUrl: string }>
}

interface CodingToolIntegrationProps {
  baseUrl: string | null
  modelName: string | null
  port: number | null
}

export function CodingToolIntegration({ baseUrl, modelName, port }: CodingToolIntegrationProps) {
  const [tools, setTools] = useState<Record<string, ToolStatus>>({})
  const [loading, setLoading] = useState<string | null>(null)
  const [message, setMessage] = useState<{ tool: string; text: string; type: 'success' | 'error' } | null>(null)

  const refresh = useCallback(async () => {
    try {
      const status = await window.api.tools?.getCodingToolStatus?.()
      if (status) setTools(status)
    } catch {}
  }, [])

  useEffect(() => { refresh() }, [refresh, baseUrl])

  const handleInstall = async (tool: string) => {
    setLoading(tool)
    setMessage(null)
    try {
      const result = await window.api.tools?.installCodingTool?.(tool)
      if (result?.success) {
        setMessage({ tool, text: 'Installed successfully', type: 'success' })
      } else {
        setMessage({ tool, text: result?.error || 'Install failed', type: 'error' })
      }
      await refresh()
    } catch (e) {
      setMessage({ tool, text: (e as Error).message, type: 'error' })
    }
    setLoading(null)
  }

  const handleAdd = async (tool: string) => {
    if (!baseUrl || !modelName) return
    setLoading(tool)
    setMessage(null)
    try {
      const result = await window.api.tools?.addCodingToolConfig?.(tool, baseUrl, modelName, port)
      if (result?.success) {
        setMessage({ tool, text: `Added ${modelName}`, type: 'success' })
      } else {
        setMessage({ tool, text: result?.error || 'Config failed', type: 'error' })
      }
      await refresh()
    } catch (e) {
      setMessage({ tool, text: (e as Error).message, type: 'error' })
    }
    setLoading(null)
  }

  const handleRemove = async (tool: string, label: string) => {
    setLoading(tool)
    setMessage(null)
    try {
      const result = await window.api.tools?.removeCodingToolConfig?.(tool, label)
      if (result?.success) {
        setMessage({ tool, text: `Removed ${label}`, type: 'success' })
      } else {
        setMessage({ tool, text: result?.error || 'Remove failed', type: 'error' })
      }
      await refresh()
    } catch (e) {
      setMessage({ tool, text: (e as Error).message, type: 'error' })
    }
    setLoading(null)
  }

  const TOOLS = [
    { id: 'claude-code', name: 'Claude Code', desc: 'Anthropic CLI for coding', install: 'npm install -g @anthropic-ai/claude-code', link: 'https://claude.ai/claude-code' },
    { id: 'codex', name: 'Codex CLI', desc: 'OpenAI coding agent', install: 'npm install -g @openai/codex', link: 'https://github.com/openai/codex' },
    { id: 'opencode', name: 'OpenCode', desc: 'Terminal coding assistant', install: 'npm install -g opencode', link: 'https://opencode.ai' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium">Coding Tool Integration</h3>
          <p className="text-[10px] text-muted-foreground">Connect your local model to coding agents with one click</p>
        </div>
        <button onClick={refresh} className="p-1 text-muted-foreground hover:text-foreground" title="Refresh status">
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      {TOOLS.map(tool => {
        const status = tools[tool.id]
        const isLoading = loading === tool.id
        const msg = message?.tool === tool.id ? message : null

        return (
          <div key={tool.id} className="p-3 rounded-lg border border-border bg-card">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${
                  status?.configured ? 'bg-green-500' :
                  status?.installed ? 'bg-yellow-500' :
                  'bg-gray-400'
                }`} />
                <span className="text-sm font-medium">{tool.name}</span>
                <span className="text-[10px] text-muted-foreground">{tool.desc}</span>
              </div>
              <a href={tool.link} target="_blank" rel="noopener noreferrer"
                className="text-muted-foreground hover:text-foreground">
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>

            {/* Status + Actions */}
            <div className="flex items-center gap-2 flex-wrap">
              {!status?.installed ? (
                <button
                  onClick={() => handleInstall(tool.id)}
                  disabled={isLoading}
                  className="px-2.5 py-1 text-xs bg-primary text-primary-foreground rounded flex items-center gap-1 hover:bg-primary/90 disabled:opacity-50"
                >
                  {isLoading ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                  Install
                </button>
              ) : (
                <>
                  <span className="text-[10px] text-green-500 flex items-center gap-0.5">
                    <Check className="h-3 w-3" /> Installed
                  </span>
                  {baseUrl && modelName && (
                    <button
                      onClick={() => handleAdd(tool.id)}
                      disabled={isLoading}
                      className="px-2.5 py-1 text-xs bg-blue-600 text-white rounded flex items-center gap-1 hover:bg-blue-700 disabled:opacity-50"
                      title={`Add ${modelName} to ${tool.name} config`}
                    >
                      {isLoading ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
                      Add Current Model
                    </button>
                  )}
                </>
              )}
            </div>

            {/* Configured entries */}
            {status?.entries && status.entries.length > 0 && (
              <div className="mt-2 space-y-1">
                {status.entries.map((entry, i) => (
                  <div key={i} className="flex items-center justify-between text-[11px] px-2 py-1 bg-muted/50 rounded">
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      <span className="font-medium">{entry.label}</span>
                      <span className="text-muted-foreground">{entry.baseUrl}</span>
                    </div>
                    <button
                      onClick={() => handleRemove(tool.id, entry.label)}
                      disabled={isLoading}
                      className="p-0.5 text-muted-foreground hover:text-destructive"
                      title="Remove this entry"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Message */}
            {msg && (
              <p className={`text-[10px] mt-1.5 ${msg.type === 'error' ? 'text-destructive' : 'text-green-500'}`}>
                {msg.text}
              </p>
            )}
          </div>
        )
      })}

      {!baseUrl && (
        <p className="text-[10px] text-muted-foreground italic">
          Start a model server to enable "Add Current Model" buttons
        </p>
      )}
    </div>
  )
}
