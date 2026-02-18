import { useState, useEffect } from 'react'
import { useToast } from '../Toast'

interface ChatOverrides {
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
  repeatPenalty?: number
  systemPrompt?: string
  stopSequences?: string
  wireApi?: 'completions' | 'responses'
  maxToolIterations?: number
  builtinToolsEnabled?: boolean
  workingDirectory?: string
  enableThinking?: boolean
  reasoningEffort?: 'low' | 'medium' | 'high'
  hideToolStatus?: boolean
  webSearchEnabled?: boolean
  braveSearchEnabled?: boolean
  fetchUrlEnabled?: boolean
  fileToolsEnabled?: boolean
  searchToolsEnabled?: boolean
  shellEnabled?: boolean
}

interface SessionInfo {
  modelName?: string
  modelPath: string
  host: string
  port: number
  status: string
  pid?: number
  type?: 'local' | 'remote'
  remoteUrl?: string
}

interface ChatSettingsProps {
  chatId: string
  session: SessionInfo
  reasoningParser?: string
  onClose: () => void
}

export function ChatSettings({ chatId, session, reasoningParser, onClose }: ChatSettingsProps) {
  const { showToast } = useToast()
  const [overrides, setOverrides] = useState<ChatOverrides>({})
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    window.api.chat.getOverrides(chatId).then((o: ChatOverrides | null) => {
      if (o) setOverrides(o)
      else setOverrides({})
      setDirty(false)
    })
  }, [chatId])

  const update = <K extends keyof ChatOverrides>(key: K, value: ChatOverrides[K]) => {
    setOverrides(prev => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await window.api.chat.setOverrides(chatId, overrides)
      setDirty(false)
    } catch (e) {
      showToast('error', 'Failed to save settings', (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    // Reset only inference parameters — preserve agent config (working dir, system prompt, tool toggles)
    const preserved: ChatOverrides = {}
    if (overrides.systemPrompt) preserved.systemPrompt = overrides.systemPrompt
    if (overrides.workingDirectory) preserved.workingDirectory = overrides.workingDirectory
    if (overrides.builtinToolsEnabled != null) preserved.builtinToolsEnabled = overrides.builtinToolsEnabled
    if (overrides.webSearchEnabled != null) preserved.webSearchEnabled = overrides.webSearchEnabled
    if (overrides.braveSearchEnabled != null) preserved.braveSearchEnabled = overrides.braveSearchEnabled
    if (overrides.fetchUrlEnabled != null) preserved.fetchUrlEnabled = overrides.fetchUrlEnabled
    if (overrides.fileToolsEnabled != null) preserved.fileToolsEnabled = overrides.fileToolsEnabled
    if (overrides.searchToolsEnabled != null) preserved.searchToolsEnabled = overrides.searchToolsEnabled
    if (overrides.shellEnabled != null) preserved.shellEnabled = overrides.shellEnabled
    if (overrides.wireApi) preserved.wireApi = overrides.wireApi
    if (overrides.hideToolStatus != null) preserved.hideToolStatus = overrides.hideToolStatus

    // Re-read model's generation_config.json for recommended inference defaults
    // Use atomic upsert (INSERT OR REPLACE) instead of clear-then-set
    const defaults: ChatOverrides = { ...preserved }
    try {
      const gen = await window.api.models.getGenerationDefaults(session.modelPath)
      if (gen) {
        if (gen.temperature != null) defaults.temperature = gen.temperature
        if (gen.topP != null) defaults.topP = gen.topP
        if (gen.topK != null) defaults.topK = gen.topK
        if (gen.repeatPenalty != null) defaults.repeatPenalty = gen.repeatPenalty
      }
    } catch (_) {}
    await window.api.chat.setOverrides(chatId, defaults)
    setOverrides(defaults)
    setDirty(false)
  }

  const shortModel = session.modelName || session.modelPath.split('/').pop() || session.modelPath

  return (
    <div className="w-80 h-full border-l border-border bg-card flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border flex-shrink-0">
        <span className="font-medium text-sm">Chat Settings</span>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-sm px-1">
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-5">
        {/* Server Info */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Server Info</h3>
          {(() => {
            const isRemote = session.type === 'remote'
            const baseUrl = isRemote && session.remoteUrl
              ? session.remoteUrl.replace(/\/+$/, '')
              : `http://${session.host}:${session.port}`
            const apiUrl = `${baseUrl}/v1/chat/completions`
            return (
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model</span>
                  <span className="text-right truncate ml-2 max-w-[180px]" title={session.modelPath}>{shortModel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Endpoint</span>
                  <span className="font-mono text-xs truncate ml-2 max-w-[180px]" title={baseUrl}>
                    {isRemote ? baseUrl : `http://${session.host}:${session.port}`}
                  </span>
                </div>
                {isRemote && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Type</span>
                    <span className="text-xs text-muted-foreground">Remote</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className={session.status === 'running' ? 'text-primary' : 'text-destructive'}>
                    {session.status}
                  </span>
                </div>
                {session.pid && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">PID</span>
                    <span>{session.pid}</span>
                  </div>
                )}
                <div className="mt-2">
                  <span className="text-xs text-muted-foreground">API URL (click to copy)</span>
                  <button
                    onClick={() => navigator.clipboard.writeText(apiUrl)}
                    className="block w-full text-left font-mono text-xs bg-background px-2 py-1.5 rounded border border-border mt-1 hover:bg-accent truncate"
                    title="Click to copy"
                  >
                    {apiUrl}
                  </button>
                </div>
              </div>
            )
          })()}
        </div>

        <div className="border-t border-border" />

        {/* Inference Settings */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Inference</h3>
          <div className="space-y-4">
            <div className="border-b border-border pb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Enable Thinking</span>
              </div>
              <div className="flex gap-1 bg-background rounded border border-border p-0.5">
                <button
                  onClick={() => update('enableThinking', undefined)}
                  className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                    overrides.enableThinking == null
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-accent text-muted-foreground'
                  }`}
                >
                  Auto
                </button>
                <button
                  onClick={() => update('enableThinking', true)}
                  className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                    overrides.enableThinking === true
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-accent text-muted-foreground'
                  }`}
                >
                  On
                </button>
                <button
                  onClick={() => update('enableThinking', false)}
                  className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                    overrides.enableThinking === false
                      ? 'bg-primary text-primary-foreground'
                      : 'hover:bg-accent text-muted-foreground'
                  }`}
                >
                  Off
                </button>
              </div>
              <p className="text-xs text-muted-foreground mt-1.5">
                Auto: reasoning models think, others don't. On: force thinking. Off: direct responses only.
              </p>
              {overrides.enableThinking !== false && reasoningParser === 'openai_gptoss' && (
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-muted-foreground">Reasoning Effort</span>
                  </div>
                  <div className="flex gap-1 bg-background rounded border border-border p-0.5">
                    <button
                      onClick={() => update('reasoningEffort', undefined)}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        overrides.reasoningEffort == null
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent text-muted-foreground'
                      }`}
                    >
                      Auto
                    </button>
                    <button
                      onClick={() => update('reasoningEffort', 'low')}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        overrides.reasoningEffort === 'low'
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent text-muted-foreground'
                      }`}
                    >
                      Low
                    </button>
                    <button
                      onClick={() => update('reasoningEffort', 'medium')}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        overrides.reasoningEffort === 'medium'
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent text-muted-foreground'
                      }`}
                    >
                      Medium
                    </button>
                    <button
                      onClick={() => update('reasoningEffort', 'high')}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        overrides.reasoningEffort === 'high'
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent text-muted-foreground'
                      }`}
                    >
                      High
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Controls thinking depth for models that support it (GPT-OSS, etc.). Auto lets the model decide.
                  </p>
                </div>
              )}
            </div>
            <SliderField
              label="Temperature"
              value={overrides.temperature ?? 0.7}
              onChange={v => update('temperature', v)}
              min={0} max={2} step={0.05}
              help="Controls randomness. Lower = more focused, higher = more creative."
            />
            <SliderField
              label="Top P"
              value={overrides.topP ?? 0.9}
              onChange={v => update('topP', v)}
              min={0} max={1} step={0.05}
              help="Nucleus sampling. Only tokens with cumulative probability above this threshold."
            />
            <NumberField
              label="Max Tokens"
              value={overrides.maxTokens}
              onChange={v => update('maxTokens', v)}
              placeholder="4096 (default)"
              help="Maximum tokens to generate per response."
            />
            <SliderField
              label="Top K"
              value={overrides.topK ?? 0}
              onChange={v => update('topK', v === 0 ? undefined : v)}
              min={0} max={200} step={1}
              help="Limits tokens to top K candidates. 0 = disabled (uses Top P only)."
              format={v => Math.round(v).toString()}
            />
            <SliderField
              label="Min P"
              value={overrides.minP ?? 0}
              onChange={v => update('minP', v === 0 ? undefined : v)}
              min={0} max={1} step={0.01}
              help="Minimum probability threshold relative to top token. 0 = disabled."
            />
            <SliderField
              label="Repetition Penalty"
              value={overrides.repeatPenalty ?? 1.0}
              onChange={v => update('repeatPenalty', v === 1.0 ? undefined : v)}
              min={1.0} max={2.0} step={0.05}
              help="Penalizes repeated tokens. 1.0 = no penalty, higher = less repetition."
            />
          </div>
        </div>

        <div className="border-t border-border" />

        {/* System Prompt */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">System Prompt</h3>
          <textarea
            value={overrides.systemPrompt ?? ''}
            onChange={e => update('systemPrompt', e.target.value || undefined)}
            placeholder="You are a helpful assistant..."
            className="w-full h-24 resize-none px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <p className="text-xs text-muted-foreground mt-1">Injected as the first message in each request.</p>
        </div>

        {/* Stop Sequences */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Stop Sequences</h3>
          <input
            type="text"
            value={overrides.stopSequences ?? ''}
            onChange={e => update('stopSequences', e.target.value || undefined)}
            placeholder="Comma-separated (uses defaults if empty)"
            className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <p className="text-xs text-muted-foreground mt-1">Custom stop tokens override the built-in template tokens.</p>
        </div>

        <div className="border-t border-border" />

        {/* Wire Format */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">API Wire Format</h3>
          <select
            value={overrides.wireApi ?? 'completions'}
            onChange={e => update('wireApi', e.target.value as 'completions' | 'responses')}
            className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="completions">Chat Completions (/v1/chat/completions)</option>
            <option value="responses">Responses (/v1/responses)</option>
          </select>
          <p className="text-xs text-muted-foreground mt-1">Wire format for API requests. Use Responses for OpenAI Responses API compatibility.</p>
        </div>

        <div className="border-t border-border" />

        {/* Agentic Settings */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Agentic</h3>
          <div className="space-y-4">
            <NumberField
              label="Max Tool Iterations"
              value={overrides.maxToolIterations}
              onChange={v => update('maxToolIterations', v)}
              placeholder="10 (default)"
              help="Maximum MCP tool call rounds per message. Higher = more autonomous."
            />

            <div className="border-t border-border pt-4">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={overrides.builtinToolsEnabled ?? false}
                  onChange={e => update('builtinToolsEnabled', e.target.checked || undefined)}
                  className="w-4 h-4 accent-primary"
                />
                <span className="font-medium">Enable Built-in Coding Tools</span>
              </label>
              <p className="text-xs text-muted-foreground mt-1 ml-6">
                File I/O, code search, shell commands, web search, and URL fetching. Full agentic coding environment.
              </p>

              {overrides.builtinToolsEnabled && (
                <div className="mt-3 ml-6 space-y-2">
                  <label className="text-sm">Working Directory</label>
                  {!overrides.workingDirectory && (
                    <div className="px-2 py-1.5 rounded text-[11px] bg-warning/10 border border-warning/30 text-warning leading-tight">
                      Working directory required. Tools will fail without a directory set.
                    </div>
                  )}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={overrides.workingDirectory ?? ''}
                      readOnly
                      placeholder="Select a project directory..."
                      className="flex-1 px-3 py-1.5 bg-background border border-input rounded text-sm font-mono truncate focus:outline-none"
                    />
                    <button
                      onClick={async () => {
                        const result = await (window as any).electron.ipcRenderer.invoke('dialog:openDirectory')
                        if (result && !result.canceled && result.filePaths?.[0]) {
                          update('workingDirectory', result.filePaths[0])
                        }
                      }}
                      className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent whitespace-nowrap"
                    >
                      Browse
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    All file operations and commands execute within this directory.
                  </p>

                  <div className="mt-3 pt-3 border-t border-border space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Tool Categories</p>
                    <ToolToggle
                      label="File I/O"
                      checked={overrides.fileToolsEnabled !== false}
                      onChange={v => update('fileToolsEnabled', v)}
                      help="read, write, edit, copy, move, delete files and directories"
                    />
                    <ToolToggle
                      label="Search"
                      checked={overrides.searchToolsEnabled !== false}
                      onChange={v => update('searchToolsEnabled', v)}
                      help="search file contents, find files, file info"
                    />
                    <ToolToggle
                      label="Shell"
                      checked={overrides.shellEnabled !== false}
                      onChange={v => update('shellEnabled', v)}
                      help="execute shell commands in working directory"
                    />
                    <ToolToggle
                      label="Web Search (DuckDuckGo)"
                      checked={overrides.webSearchEnabled !== false}
                      onChange={v => update('webSearchEnabled', v)}
                      help="free web search — no API key needed"
                    />
                    <BraveSearchToggle
                      checked={overrides.braveSearchEnabled === true}
                      onChange={v => update('braveSearchEnabled', v)}
                    />
                    <ToolToggle
                      label="URL Fetch"
                      checked={overrides.fetchUrlEnabled !== false}
                      onChange={v => update('fetchUrlEnabled', v)}
                      help="fetch and read web page content"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-border pt-4">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={overrides.hideToolStatus === true}
                  onChange={e => update('hideToolStatus', e.target.checked)}
                  className="w-4 h-4 accent-primary"
                />
                <span className="font-medium">Hide Tool Status</span>
              </label>
              <p className="text-xs text-muted-foreground mt-1 ml-6">
                Hide tool execution details from the chat. Tools still execute normally.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex items-center gap-2 px-4 py-3 border-t border-border flex-shrink-0">
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          onClick={handleReset}
          className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent"
        >
          Reset
        </button>
      </div>
    </div>
  )
}

// ─── Helper Components ────────────────────────────────────────────────────────

function SliderField({ label, value, onChange, min, max, step, help, format }: {
  label: string; value: number; onChange: (v: number) => void
  min: number; max: number; step: number; help: string
  format?: (v: number) => string
}) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="text-muted-foreground font-mono text-xs">{format ? format(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        min={min} max={max} step={step}
        className="w-full h-1.5 accent-primary"
      />
      <p className="text-xs text-muted-foreground mt-0.5">{help}</p>
    </div>
  )
}

function NumberField({ label, value, onChange, placeholder, help }: {
  label: string; value: number | undefined; onChange: (v: number | undefined) => void
  placeholder: string; help: string
}) {
  return (
    <div>
      <label className="text-sm">{label}</label>
      <input
        type="number"
        value={value ?? ''}
        onChange={e => onChange(e.target.value ? parseInt(e.target.value) : undefined)}
        placeholder={placeholder}
        className="w-full mt-1 px-3 py-1.5 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
      />
      <p className="text-xs text-muted-foreground mt-0.5">{help}</p>
    </div>
  )
}

function BraveSearchToggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  const [key, setKey] = useState('')
  const [hasKey, setHasKey] = useState(false)

  useEffect(() => {
    window.api.settings.get('braveApiKey').then((val: string | null) => {
      if (val) { setKey(val); setHasKey(true) }
      else if (checked) onChange(false) // correct stale enabled state when key was deleted
    })
  }, [])

  const saveKey = async (val: string) => {
    const trimmed = val.trim()
    setKey(trimmed)
    if (trimmed) {
      await window.api.settings.set('braveApiKey', trimmed)
      setHasKey(true)
    } else {
      await window.api.settings.delete('braveApiKey')
      setHasKey(false)
      onChange(false) // auto-disable when key removed
    }
  }

  return (
    <div>
      <label className="flex items-start gap-2 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => {
            if (e.target.checked && !hasKey) return // can't enable without key
            onChange(e.target.checked)
          }}
          disabled={!hasKey}
          className="w-3.5 h-3.5 accent-primary mt-0.5"
        />
        <span>
          <span>Brave Search</span>
          <span className="block text-xs text-muted-foreground">
            {hasKey ? 'premium web search with API key' : 'requires API key — enter below'}
          </span>
        </span>
      </label>
      <div className="ml-6 mt-1.5">
        <input
          type="password"
          value={key}
          onChange={e => setKey(e.target.value)}
          onBlur={e => saveKey(e.target.value)}
          placeholder="Brave Search API key..."
          className="w-full px-2 py-1 bg-background border border-input rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-ring"
        />
        <a
          href="https://brave.com/search/api/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[10px] text-primary hover:underline mt-0.5 inline-block"
        >
          Get a free API key →
        </a>
      </div>
    </div>
  )
}

function ToolToggle({ label, checked, onChange, help }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; help: string
}) {
  return (
    <label className="flex items-start gap-2 text-sm cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        className="w-3.5 h-3.5 accent-primary mt-0.5"
      />
      <span>
        <span>{label}</span>
        <span className="block text-xs text-muted-foreground">{help}</span>
      </span>
    </label>
  )
}
