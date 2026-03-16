import { useState, useEffect, useRef, useCallback } from 'react'
import { useToast } from '../Toast'

interface HFModel {
  id: string
  author: string
  downloads: number
  likes: number
  lastModified: string
  tags: string[]
  pipelineTag?: string
  size?: string
}

interface DownloadTabProps {
  onDownloadComplete: () => void
}

function formatNumber(n: number): string {
  if (n === undefined || n === null || isNaN(n)) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

function timeAgo(dateStr: string | null | undefined): string {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  if (isNaN(diff)) return ''
  const days = Math.floor(diff / 86400000)
  if (days < 1) return 'today'
  if (days < 30) return `${days}d ago`
  if (days < 365) return `${Math.floor(days / 30)}mo ago`
  return `${Math.floor(days / 365)}y ago`
}

export function DownloadTab({ onDownloadComplete }: DownloadTabProps) {
  const { showToast } = useToast()
  const [searchQuery, setSearchQuery] = useState('')
  const [modelType, setModelType] = useState<'text' | 'image'>('text')
  const [sortBy, setSortBy] = useState<string>('downloads')
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc')
  const [searchResults, setSearchResults] = useState<HFModel[]>([])
  const [recommended, setRecommended] = useState<HFModel[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingRecommended, setLoadingRecommended] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Download state: track which repos are downloading/queued
  const [downloadingRepos, setDownloadingRepos] = useState<Set<string>>(new Set())
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Download directory
  const [downloadDir, setDownloadDir] = useState('')

  // Track locally available models for "already downloaded" detection
  const [localModelIds, setLocalModelIds] = useState<Set<string>>(new Set())

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onDownloadCompleteRef = useRef(onDownloadComplete)
  onDownloadCompleteRef.current = onDownloadComplete

  // Load recommended models and download dir on mount
  useEffect(() => {
    window.api.models.getDownloadDir().then(setDownloadDir)
    window.api.models.scan().then((models: any[]) => {
      // Build a set of local model identifiers for matching against HF repo IDs
      const ids = new Set<string>()
      for (const m of models) {
        if (m.id) ids.add(m.id)
        if (m.path) {
          const parts = m.path.replace(/\\/g, '/').split('/')
          if (parts.length >= 2) {
            ids.add(`${parts[parts.length - 2]}/${parts[parts.length - 1]}`)
          }
        }
      }
      setLocalModelIds(ids)
    }).catch((err) => console.error('Failed to scan models:', err))
    window.api.models.getRecommendedModels()
      .then(setRecommended)
      .catch(err => console.error('Failed to load recommended models:', err))
      .finally(() => setLoadingRecommended(false))

    // Check for any in-progress downloads
    window.api.models.getDownloadStatus().then((status: any) => {
      const repos = new Set<string>()
      if (status.active) repos.add(status.active.repoId)
      for (const q of status.queue || []) repos.add(q.repoId)
      setDownloadingRepos(repos)
    }).catch((err) => console.error('Failed to get download status:', err))

    // Cleanup search debounce timer on unmount
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    }
  }, [])

  // Listen for download events
  useEffect(() => {
    const unsubStarted = window.api.models.onDownloadStarted((data: any) => {
      setDownloadingRepos(prev => new Set(prev).add(data.repoId))
    })

    const unsubComplete = window.api.models.onDownloadComplete((data: any) => {
      setDownloadingRepos(prev => {
        const next = new Set(prev)
        next.delete(data.repoId)
        return next
      })
      if (data.status === 'complete') {
        showToast('success', `Download complete: ${data.repoId}`)
        onDownloadCompleteRef.current()
        // Refresh local model list so the "Downloaded" badge appears immediately
        window.api.models.scan().then((models: any[]) => {
          const ids = new Set<string>()
          for (const m of models) {
            if (m.id) ids.add(m.id)
            if (m.path) {
              const parts = m.path.replace(/\\/g, '/').split('/')
              if (parts.length >= 2) {
                ids.add(`${parts[parts.length - 2]}/${parts[parts.length - 1]}`)
              }
            }
          }
          setLocalModelIds(ids)
        }).catch((err) => console.error('Failed to refresh models after download:', err))
      }
    })

    const unsubError = window.api.models.onDownloadError((data: any) => {
      setDownloadingRepos(prev => {
        const next = new Set(prev)
        next.delete(data.repoId)
        return next
      })
      const errMsg = `${data.repoId.split('/').pop()}: ${data.error}`
      setDownloadError(errMsg)
      showToast('error', 'Download failed', errMsg)
    })

    return () => {
      unsubStarted()
      unsubComplete()
      unsubError()
    }
  }, [])

  // Debounced search
  const doSearch = useCallback(async (query: string, sort: string, dir: 'desc' | 'asc', type: 'text' | 'image' = 'text') => {
    if (!query.trim()) {
      setSearchResults([])
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const results = await window.api.models.searchHF(query.trim(), sort, dir, type === 'image' ? 'image' : undefined)
      setSearchResults(results)
    } catch (err) {
      setError((err as Error).message)
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    if (!query.trim()) {
      setSearchResults([])
      setError(null)
      return
    }
    searchTimerRef.current = setTimeout(() => doSearch(query, sortBy, sortDir, modelType), 400)
  }, [doSearch, sortBy, sortDir, modelType])

  const handleSortChange = useCallback((newSort: string) => {
    setSortBy(newSort)
    if (searchQuery.trim()) {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
      doSearch(searchQuery, newSort, sortDir, modelType)
    }
  }, [doSearch, searchQuery, sortDir, modelType])

  const handleModelTypeChange = useCallback((type: 'text' | 'image') => {
    setModelType(type)
    if (searchQuery.trim()) {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
      doSearch(searchQuery, sortBy, sortDir, type)
    }
  }, [doSearch, searchQuery, sortBy, sortDir])

  const handleDirToggle = useCallback(() => {
    const newDir = sortDir === 'desc' ? 'asc' : 'desc'
    setSortDir(newDir)
    if (searchQuery.trim()) {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
      doSearch(searchQuery, sortBy, newDir)
    }
  }, [doSearch, searchQuery, sortBy, sortDir])

  const handleDownload = async (repoId: string) => {
    setDownloadError(null)
    setDownloadingRepos(prev => new Set(prev).add(repoId))

    try {
      await window.api.models.startDownload(repoId)
    } catch (err) {
      setDownloadError((err as Error).message)
      setDownloadingRepos(prev => {
        const next = new Set(prev)
        next.delete(repoId)
        return next
      })
    }
  }

  const handleBrowseDownloadDir = async () => {
    const result = await window.api.models.browseDownloadDir()
    if (!result.canceled && result.path) {
      await window.api.models.setDownloadDir(result.path)
      setDownloadDir(result.path)
    }
  }

  const displayModels = searchQuery.trim() ? searchResults : recommended
  const showSection = searchQuery.trim() ? 'Search Results' : 'Recommended (JANGQ AI)'

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Search and download MLX models from HuggingFace. Downloads run in the background.
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

      {/* Model Type Filter + Search + Sort */}
      <div className="flex items-center gap-2">
        <div className="flex rounded border border-border overflow-hidden flex-shrink-0">
          <button
            onClick={() => handleModelTypeChange('text')}
            className={`px-2.5 py-2 text-xs transition-colors ${modelType === 'text' ? 'bg-primary/15 text-primary font-medium' : 'text-muted-foreground hover:bg-accent'}`}
          >
            Text
          </button>
          <button
            onClick={() => handleModelTypeChange('image')}
            className={`px-2.5 py-2 text-xs transition-colors ${modelType === 'image' ? 'bg-violet-500/15 text-violet-400 font-medium' : 'text-muted-foreground hover:bg-accent'}`}
          >
            Image
          </button>
        </div>
        <input
          type="text"
          placeholder={modelType === 'image' ? 'Search image models (flux, sdxl, z-image...)' : 'Search MLX models...'}
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
          className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm"
        />
        {searchQuery.trim() && (
          <>
            <select
              value={sortBy}
              onChange={(e) => handleSortChange(e.target.value)}
              className="px-2 py-2 bg-background border border-input rounded text-xs text-foreground"
              title="Sort results by"
            >
              <option value="downloads">Downloads</option>
              <option value="relevance">Relevance</option>
              <option value="lastModified">Recently Updated</option>
              <option value="trending">Trending</option>
              <option value="likes">Likes</option>
              <option value="size">Model Size</option>
            </select>
            {sortBy !== 'relevance' && (
              <button
                onClick={handleDirToggle}
                className="px-1.5 py-2 bg-background border border-input rounded text-xs text-foreground hover:bg-accent"
                title={sortDir === 'desc' ? 'Highest first' : 'Lowest first'}
              >
                {sortDir === 'desc' ? '\u2193' : '\u2191'}
              </button>
            )}
          </>
        )}
        {loading && <span className="text-xs text-muted-foreground">Searching...</span>}
      </div>

      {error && (
        <div className="p-2 bg-destructive/10 border border-destructive/30 rounded text-xs text-destructive">
          {error}
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
                isDownloading={downloadingRepos.has(model.id)}
                isDownloaded={localModelIds.has(model.id)}
                onDownload={() => handleDownload(model.id)}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function ModelCard({ model, isDownloading, isDownloaded, onDownload }: {
  model: HFModel
  isDownloading: boolean
  isDownloaded: boolean
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
            {model.size && <span title="Model size">{model.size}</span>}
            {timeAgo(model.lastModified) && <span>{timeAgo(model.lastModified)}</span>}
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
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={(e) => { e.stopPropagation(); window.open(`https://huggingface.co/${model.id}`, '_blank') }}
            className="px-1.5 py-1.5 text-xs text-muted-foreground hover:text-foreground border border-border rounded"
            title="View on HuggingFace"
          >
            ↗
          </button>
          {isDownloaded && !isDownloading ? (
            <span className="px-3 py-1.5 text-xs text-primary border border-primary/30 rounded whitespace-nowrap">
              Downloaded
            </span>
          ) : (
            <button
              onClick={onDownload}
              disabled={isDownloading}
              className="px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40 whitespace-nowrap"
            >
              {isDownloading ? 'Downloading...' : 'Download'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
