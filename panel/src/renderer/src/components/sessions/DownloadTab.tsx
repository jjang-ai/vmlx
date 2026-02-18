import { useState, useEffect, useRef, useCallback } from 'react'

interface HFModel {
  id: string
  author: string
  downloads: number
  likes: number
  lastModified: string
  tags: string[]
  pipelineTag?: string
}

interface DownloadTabProps {
  onDownloadComplete: () => void
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const days = Math.floor(diff / 86400000)
  if (days < 1) return 'today'
  if (days < 30) return `${days}d ago`
  if (days < 365) return `${Math.floor(days / 30)}mo ago`
  return `${Math.floor(days / 365)}y ago`
}

export function DownloadTab({ onDownloadComplete }: DownloadTabProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<HFModel[]>([])
  const [recommended, setRecommended] = useState<HFModel[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingRecommended, setLoadingRecommended] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Download state
  const [downloading, setDownloading] = useState<string | null>(null) // repo ID
  const [downloadProgress, setDownloadProgress] = useState<string>('')
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Download directory
  const [downloadDir, setDownloadDir] = useState('')

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const progressRef = useRef<HTMLDivElement>(null)

  // Load recommended models and download dir on mount
  useEffect(() => {
    window.api.models.getDownloadDir().then(setDownloadDir)
    window.api.models.getRecommendedModels()
      .then(setRecommended)
      .catch(err => console.error('Failed to load recommended models:', err))
      .finally(() => setLoadingRecommended(false))
  }, [])

  // Listen for download progress
  useEffect(() => {
    const unsub = window.api.models.onDownloadProgress((data) => {
      setDownloadProgress(data.progress)
    })
    return unsub
  }, [])

  // Auto-scroll progress
  useEffect(() => {
    progressRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [downloadProgress])

  // Debounced search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)

    if (!query.trim()) {
      setSearchResults([])
      setError(null)
      return
    }

    searchTimerRef.current = setTimeout(async () => {
      setLoading(true)
      setError(null)
      try {
        const results = await window.api.models.searchHF(query.trim())
        setSearchResults(results)
      } catch (err) {
        setError((err as Error).message)
        setSearchResults([])
      } finally {
        setLoading(false)
      }
    }, 400)
  }, [])

  const handleDownload = async (repoId: string) => {
    setDownloading(repoId)
    setDownloadProgress('')
    setDownloadError(null)

    try {
      const result = await window.api.models.downloadModel(repoId)
      if (result.status === 'complete') {
        setDownloading(null)
        setDownloadProgress('')
        onDownloadComplete()
      } else if (result.status === 'cancelled') {
        setDownloading(null)
        setDownloadProgress('')
      }
    } catch (err) {
      setDownloadError((err as Error).message)
      setDownloading(null)
    }
  }

  const handleCancelDownload = async () => {
    await window.api.models.cancelDownload()
    setDownloading(null)
    setDownloadProgress('')
  }

  const handleBrowseDownloadDir = async () => {
    const result = await window.api.models.browseDownloadDir()
    if (!result.canceled && result.path) {
      await window.api.models.setDownloadDir(result.path)
      setDownloadDir(result.path)
    }
  }

  const displayModels = searchQuery.trim() ? searchResults : recommended
  const showSection = searchQuery.trim() ? 'Search Results' : 'Recommended (ShieldStack LLC)'

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Search and download MLX models from HuggingFace.
      </p>

      {/* Download Directory */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground whitespace-nowrap">Download to:</span>
        <span className="text-xs font-mono truncate flex-1 text-foreground" title={downloadDir}>
          {downloadDir}
        </span>
        <button
          onClick={handleBrowseDownloadDir}
          className="px-2 py-1 text-xs border border-border rounded hover:bg-accent whitespace-nowrap"
        >
          Change
        </button>
      </div>

      {/* Search */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          placeholder="Search MLX models..."
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
          className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm"
        />
        {loading && <span className="text-xs text-muted-foreground">Searching...</span>}
      </div>

      {error && (
        <div className="p-2 bg-destructive/10 border border-destructive/30 rounded text-xs text-destructive">
          {error}
        </div>
      )}

      {/* Active Download */}
      {downloading && (
        <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Downloading: {downloading.split('/').pop()}</span>
            <button
              onClick={handleCancelDownload}
              className="px-2 py-0.5 text-xs text-destructive border border-destructive/30 rounded hover:bg-destructive/10"
            >
              Cancel
            </button>
          </div>
          <div className="bg-background/50 rounded p-2 max-h-20 overflow-auto font-mono text-[11px] text-muted-foreground">
            {downloadProgress || 'Starting download...'}
            <div ref={progressRef} />
          </div>
        </div>
      )}

      {downloadError && (
        <div className="p-2 bg-destructive/10 border border-destructive/30 rounded text-xs text-destructive">
          Download failed: {downloadError}
        </div>
      )}

      {/* Model List */}
      <div>
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{showSection}</span>
        <div className="mt-2 space-y-1">
          {(loadingRecommended && !searchQuery.trim()) ? (
            <p className="text-sm text-muted-foreground py-4 text-center">Loading recommendations...</p>
          ) : displayModels.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              {searchQuery.trim() ? 'No MLX models found' : 'No recommended models available'}
            </p>
          ) : (
            displayModels.map(model => (
              <ModelCard
                key={model.id}
                model={model}
                downloading={downloading === model.id}
                anyDownloading={!!downloading}
                onDownload={() => handleDownload(model.id)}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function ModelCard({ model, downloading, anyDownloading, onDownload }: {
  model: HFModel
  downloading: boolean
  anyDownloading: boolean
  onDownload: () => void
}) {
  const shortName = model.id.includes('/') ? model.id.split('/').slice(1).join('/') : model.id

  return (
    <div className="p-3 rounded border border-border hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm truncate" title={model.id}>
            {shortName}
          </div>
          <div className="text-xs text-muted-foreground">{model.author}</div>
          <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
            <span title="Downloads">{formatNumber(model.downloads)} downloads</span>
            <span title="Likes">{model.likes} likes</span>
            <span>{timeAgo(model.lastModified)}</span>
          </div>
          {model.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {model.tags.slice(0, 5).map(tag => (
                <span key={tag} className="px-1.5 py-0.5 bg-muted rounded text-[10px] text-muted-foreground">
                  {tag}
                </span>
              ))}
              {model.tags.length > 5 && (
                <span className="text-[10px] text-muted-foreground">+{model.tags.length - 5}</span>
              )}
            </div>
          )}
        </div>
        <button
          onClick={onDownload}
          disabled={anyDownloading}
          className="px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40 whitespace-nowrap flex-shrink-0"
        >
          {downloading ? 'Downloading...' : 'Download'}
        </button>
      </div>
    </div>
  )
}
