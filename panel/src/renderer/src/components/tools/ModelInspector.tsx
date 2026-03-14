import { useState, useEffect, useRef, useCallback } from 'react'
import { ArrowLeft, Search, Loader2 } from 'lucide-react'

interface ModelInspectorProps {
  initialModelPath?: string | null
  onBack: () => void
  models?: Array<{ name: string; path: string }>
}

export function ModelInspector({ initialModelPath, onBack, models = [] }: ModelInspectorProps) {
  const [modelPath, setModelPath] = useState(initialModelPath || '')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)
  const callIdRef = useRef(0)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const runInfo = useCallback(async (path: string) => {
    if (!path.trim()) return
    const thisCallId = ++callIdRef.current
    setLoading(true)
    setError(null)
    setOutput('')
    try {
      const result = await window.api.developer.info(path.trim())
      // Stale response guard: only apply if this is still the latest call
      if (!mountedRef.current || thisCallId !== callIdRef.current) return
      if (result.success) {
        setOutput(result.output)
      } else {
        setError(result.error || 'Failed to inspect model')
      }
    } catch (err: any) {
      if (!mountedRef.current || thisCallId !== callIdRef.current) return
      setError(err.message || 'Failed to inspect model')
    } finally {
      if (mountedRef.current && thisCallId === callIdRef.current) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    if (initialModelPath) {
      setModelPath(initialModelPath)
      runInfo(initialModelPath)
    }
  }, [initialModelPath, runInfo])

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-3xl mx-auto space-y-6">
        <button
          onClick={onBack}
          className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
        >
          <ArrowLeft className="h-3 w-3" />
          Back
        </button>

        <h2 className="text-2xl font-bold">Model Inspector</h2>
        <p className="text-sm text-muted-foreground">
          View detailed model metadata from config.json
        </p>

        {/* Model input */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Model Path or HuggingFace ID</label>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={modelPath}
                onChange={e => setModelPath(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && runInfo(modelPath)}
                placeholder="/path/to/model or org/model-name"
                className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                list="model-paths"
                disabled={loading}
              />
              <datalist id="model-paths">
                {models.map(m => (
                  <option key={m.path} value={m.path}>{m.name}</option>
                ))}
              </datalist>
            </div>
            <button
              onClick={() => runInfo(modelPath)}
              disabled={loading || !modelPath.trim()}
              className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
              Inspect
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Output */}
        {output && (
          <div className="border border-border rounded-lg overflow-hidden">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Model Information</p>
            </div>
            <pre className="p-4 text-sm font-mono whitespace-pre-wrap overflow-auto max-h-[60vh] bg-background">
              {output}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
