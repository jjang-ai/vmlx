// MLX Studio — Coding Tool Integration
import { useState, useEffect, useCallback } from 'react'
import { Download, Check, Plus, Trash2, RefreshCw, ExternalLink, ChevronDown, ChevronRight, Copy, CheckCheck, Terminal } from 'lucide-react'

interface ToolStatus {
  installed: boolean
  configured: boolean
  configPath: string
  entries: Array<{ label: string; baseUrl: string }>
}

interface ConfigSnippet {
  filePath: string
  language: string
  snippet: string
  notes: string
}

interface CodingToolIntegrationProps {
  baseUrl: string | null
  modelName: string | null
  port: number | null
}

export function CodingToolIntegration({ baseUrl, modelName, port }: CodingToolIntegrationProps) {
  const [tools, setTools] = useState<Record<string, ToolStatus>>({})
  const [snippets, setSnippets] = useState<Record<string, ConfigSnippet>>({})
  const [loading, setLoading] = useState<string | null>(null)
  const [message, setMessage] = useState<{ tool: string; text: string; type: 'success' | 'error' } | null>(null)
  const [expandedSnippets, setExpandedSnippets] = useState<Record<string, boolean>>({})
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const status = await window.api.tools?.getCodingToolStatus?.()
      if (status) setTools(status)
    } catch {}
  }, [])

  // Fetch config snippets whenever baseUrl/modelName changes
  useEffect(() => {
    if (!baseUrl || !modelName) { setSnippets({}); return }
    window.api.tools?.getConfigSnippets?.(baseUrl, modelName)
      .then((s: Record<string, ConfigSnippet>) => { if (s) setSnippets(s) })
      .catch(() => {})
  }, [baseUrl, modelName])

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

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const toggleSnippet = (id: string) => {
    setExpandedSnippets(prev => ({ ...prev, [id]: !prev[id] }))
  }

  const TOOLS = [
    { id: 'claude-code', name: 'Claude Code', desc: 'Anthropic CLI for coding', install: 'curl -fsSL https://claude.ai/install.sh | bash\n# Or: brew install --cask claude-code\n# Or: npm install -g @anthropic-ai/claude-code', verify: 'claude --version', link: 'https://claude.ai/claude-code' },
    { id: 'codex', name: 'Codex CLI', desc: 'OpenAI coding agent', install: 'npm install -g @openai/codex\n# Or: brew install --cask codex', verify: 'codex --version', link: 'https://github.com/openai/codex' },
    { id: 'opencode', name: 'OpenCode', desc: 'Terminal coding assistant', install: 'npm install -g opencode', verify: 'opencode --version', link: 'https://opencode.ai' },
    { id: 'openclaw', name: 'OpenClaw', desc: 'AI agent framework', install: 'npm install -g openclaw@latest\nopenclaw onboard --install-daemon', verify: 'openclaw --version\nopenclaw doctor  # Validate config', link: 'https://github.com/openclaw/openclaw' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium">Coding Tool Integration</h3>
          <p className="text-[10px] text-muted-foreground">Connect your local model to coding agents — one-click or manual config</p>
        </div>
        <button onClick={refresh} className="p-1 text-muted-foreground hover:text-foreground" title="Refresh status">
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      {TOOLS.map(tool => {
        const status = tools[tool.id]
        const isLoading = loading === tool.id
        const msg = message?.tool === tool.id ? message : null
        const snippet = snippets[tool.id]
        const isExpanded = expandedSnippets[tool.id]

        return (
          <div key={tool.id} className="rounded-lg border border-border bg-card overflow-hidden">
            {/* Header */}
            <div className="p-3">
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
                        title={`Auto-configure ${tool.name} for ${modelName}`}
                      >
                        {isLoading ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
                        Auto-Configure
                      </button>
                    )}
                  </>
                )}

                {/* Manual setup toggle */}
                {baseUrl && modelName && snippet && (
                  <button
                    onClick={() => toggleSnippet(tool.id)}
                    className="px-2.5 py-1 text-xs border border-border rounded flex items-center gap-1 hover:bg-muted/50 text-muted-foreground hover:text-foreground"
                  >
                    <Terminal className="h-3 w-3" />
                    Manual Setup
                    {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  </button>
                )}
              </div>

              {/* Configured entries */}
              {status?.entries && status.entries.length > 0 && (
                <div className="mt-2 space-y-1">
                  {status.entries.map((entry, i) => (
                    <div key={i} className="flex items-center justify-between text-[11px] px-2 py-1 bg-muted/50 rounded">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                        <span className="font-medium truncate">{entry.label}</span>
                        <span className="text-muted-foreground truncate">{entry.baseUrl}</span>
                      </div>
                      <button
                        onClick={() => handleRemove(tool.id, entry.label)}
                        disabled={isLoading}
                        className="p-0.5 text-muted-foreground hover:text-destructive flex-shrink-0"
                        title="Remove this entry from config"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                  {status.entries.length > 1 && (
                    <button
                      onClick={async () => {
                        for (const entry of status.entries) {
                          await handleRemove(tool.id, entry.label)
                        }
                      }}
                      disabled={isLoading}
                      className="text-[10px] text-muted-foreground hover:text-destructive mt-1"
                    >
                      Remove all MLX Studio entries
                    </button>
                  )}
                </div>
              )}

              {/* Message */}
              {msg && (
                <p className={`text-[10px] mt-1.5 ${msg.type === 'error' ? 'text-destructive' : 'text-green-500'}`}>
                  {msg.text}
                </p>
              )}
            </div>

            {/* Manual Setup Code Block */}
            {isExpanded && snippet && (
              <div className="border-t border-border bg-muted/30 p-3 space-y-2">
                {/* Install command */}
                {!status?.installed && (
                  <div>
                    <p className="text-[10px] font-medium text-muted-foreground mb-1">1. Install {tool.name}</p>
                    <div className="relative group">
                      <pre className="text-[11px] bg-black/80 text-green-400 rounded p-2 pr-8 overflow-x-auto font-mono">
                        {tool.install}
                      </pre>
                      <button
                        onClick={() => handleCopy(`${tool.id}-install`, tool.install)}
                        className="absolute top-1.5 right-1.5 p-1 rounded text-gray-400 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Copy"
                      >
                        {copiedId === `${tool.id}-install` ? <CheckCheck className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
                      </button>
                    </div>
                  </div>
                )}

                {/* Config file path */}
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground mb-1">
                    {!status?.installed ? '2.' : ''} Edit config file
                  </p>
                  <div className="relative group">
                    <pre className="text-[11px] bg-black/80 text-blue-400 rounded p-2 pr-8 overflow-x-auto font-mono">
                      {snippet.filePath}
                    </pre>
                    <button
                      onClick={() => handleCopy(`${tool.id}-path`, snippet.filePath)}
                      className="absolute top-1.5 right-1.5 p-1 rounded text-gray-400 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Copy file path"
                    >
                      {copiedId === `${tool.id}-path` ? <CheckCheck className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
                    </button>
                  </div>
                </div>

                {/* Config snippet */}
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground mb-1">
                    {tool.id === 'codex' ? 'Append this to the file:' : 'Merge this into the file:'}
                  </p>
                  <div className="relative group">
                    <pre className="text-[11px] bg-black/80 text-gray-200 rounded p-2 pr-8 overflow-x-auto font-mono whitespace-pre leading-relaxed max-h-64 overflow-y-auto">
                      {snippet.snippet}
                    </pre>
                    <button
                      onClick={() => handleCopy(`${tool.id}-snippet`, snippet.snippet)}
                      className="absolute top-1.5 right-1.5 p-1 rounded text-gray-400 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Copy config"
                    >
                      {copiedId === `${tool.id}-snippet` ? <CheckCheck className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
                    </button>
                  </div>
                </div>

                {/* Notes */}
                <p className="text-[10px] text-muted-foreground italic leading-snug">
                  {snippet.notes}
                </p>
              </div>
            )}
          </div>
        )
      })}

      {!baseUrl && (
        <p className="text-[10px] text-muted-foreground italic">
          Start a model server to enable configuration buttons and manual setup instructions
        </p>
      )}
    </div>
  )
}
