import { useState } from 'react'

interface EmbeddingsPanelProps {
  endpoint: { host: string; port: number }
  sessionStatus: string
  sessionId?: string
}

const MODELS = [
  'mlx-community/all-MiniLM-L6-v2-4bit',
  'mlx-community/embeddinggemma-300m-6bit',
  'mlx-community/bge-large-en-v1.5-4bit'
]

export function EmbeddingsPanel({ endpoint, sessionStatus, sessionId }: EmbeddingsPanelProps) {
  const [model, setModel] = useState(MODELS[0])
  const [textA, setTextA] = useState('')
  const [textB, setTextB] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{
    dimensions: number
    tokens: number
    preview: string
    similarity?: number
  } | null>(null)

  if (sessionStatus !== 'running') {
    return (
      <div className="text-sm text-muted-foreground p-4">
        Session must be running to generate embeddings.
      </div>
    )
  }

  const handleEmbed = async () => {
    if (!textA.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const texts = textB.trim() ? [textA.trim(), textB.trim()] : [textA.trim()]
      const response = await window.api.embeddings.embed(texts, endpoint, model, sessionId)

      const embeddings = response.data.map((d: any) => d.embedding as number[])
      const dims = embeddings[0].length
      const tokens = response.usage?.total_tokens || 0
      const preview = `[${embeddings[0].slice(0, 6).map((v: number) => v.toFixed(4)).join(', ')}, ...]`

      let similarity: number | undefined
      if (embeddings.length === 2) {
        similarity = cosineSimilarity(embeddings[0], embeddings[1])
      }

      setResult({ dimensions: dims, tokens, preview, similarity })
    } catch (err: any) {
      setError(err.message || 'Embedding failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Model Selector */}
      <div>
        <label className="text-xs text-muted-foreground block mb-1">Embedding Model</label>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          className="w-full text-xs px-2 py-1.5 bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
        >
          {MODELS.map(m => (
            <option key={m} value={m}>{m.split('/').pop()}</option>
          ))}
        </select>
      </div>

      {/* Text A */}
      <div>
        <label className="text-xs text-muted-foreground block mb-1">Text A</label>
        <textarea
          value={textA}
          onChange={e => setTextA(e.target.value)}
          placeholder="Enter text to embed..."
          rows={3}
          className="w-full text-xs px-2 py-1.5 bg-background border border-input rounded resize-none focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>

      {/* Text B (similarity) */}
      <div>
        <label className="text-xs text-muted-foreground block mb-1">Text B (optional — for similarity)</label>
        <textarea
          value={textB}
          onChange={e => setTextB(e.target.value)}
          placeholder="Enter second text to compare..."
          rows={3}
          className="w-full text-xs px-2 py-1.5 bg-background border border-input rounded resize-none focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>

      {/* Embed Button */}
      <button
        onClick={handleEmbed}
        disabled={loading || !textA.trim()}
        className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 w-full"
      >
        {loading ? 'Embedding...' : textB.trim() ? 'Embed & Compare' : 'Generate Embedding'}
      </button>

      {/* Error */}
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">{error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-background px-3 py-2 rounded border border-border">
              <div className="text-xs text-muted-foreground">Dimensions</div>
              <div className="font-mono text-sm">{result.dimensions}</div>
            </div>
            <div className="bg-background px-3 py-2 rounded border border-border">
              <div className="text-xs text-muted-foreground">Tokens</div>
              <div className="font-mono text-sm">{result.tokens}</div>
            </div>
          </div>

          {result.similarity != null && (
            <div className="bg-background px-3 py-2 rounded border border-border">
              <div className="text-xs text-muted-foreground">Cosine Similarity</div>
              <div className="font-mono text-lg font-bold">
                {result.similarity.toFixed(4)}
                <span className="text-xs text-muted-foreground ml-2">
                  {result.similarity > 0.8 ? 'Very Similar' : result.similarity > 0.5 ? 'Somewhat Similar' : 'Different'}
                </span>
              </div>
              {/* Visual bar */}
              <div className="mt-1 w-full bg-muted rounded-full h-2">
                <div
                  className="h-2 rounded-full transition-all"
                  style={{
                    width: `${Math.max(0, result.similarity) * 100}%`,
                    backgroundColor: result.similarity > 0.8 ? 'hsl(var(--primary))' : result.similarity > 0.5 ? 'hsl(var(--warning, 45 93% 47%))' : 'hsl(var(--destructive))'
                  }}
                />
              </div>
            </div>
          )}

          <div className="bg-background px-3 py-2 rounded border border-border">
            <div className="text-xs text-muted-foreground">Embedding Preview</div>
            <div className="font-mono text-xs break-all">{result.preview}</div>
          </div>
        </div>
      )}
    </div>
  )
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom > 0 ? dot / denom : 0
}
