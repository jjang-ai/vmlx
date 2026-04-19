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
  note?: string
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

const COLLECTION_SLUGS = {
  jang: 'jangq/jang-quantized-gguf-for-mlx',
  uncensored: 'dealignai/crack-xtreme-quality-uncensored-gguf-on-mlx-69ba7ed343004d49cf8ca53f',
} as const
type CollectionTab = keyof typeof COLLECTION_SLUGS

export function DownloadTab({ onDownloadComplete }: DownloadTabProps) {
  const { showToast } = useToast()
  const [searchQuery, setSearchQuery] = useState('')
  const [modelType, setModelType] = useState<'text' | 'image'>('text')
  const [sortBy, setSortBy] = useState<string>('downloads')
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc')
  const [searchResults, setSearchResults] = useState<HFModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Collection tabs (JANG / Uncensored)
  const [collectionTab, setCollectionTab] = useState<CollectionTab>('jang')
  const [collectionModels, setCollectionModels] = useState<Record<string, HFModel[]>>({})
  const [loadingCollectionTabs, setLoadingCollectionTabs] = useState<Record<string, boolean>>({})
  // ms#68: distinguish a genuinely empty HF collection from a fetch
  // failure (no internet, HF down, HF rate-limited, slug outdated).
  // Both previously showed the same "No models in this collection"
  // text — the user had no way to tell whether to retry.
  const [collectionErrors, setCollectionErrors] = useState<Record<string, string>>({})

  // Download state: track which repos are downloading/queued
  const [downloadingRepos, setDownloadingRepos] = useState<Set<string>>(new Set())
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Download directory
  const [downloadDir, setDownloadDir] = useState('')

  // HuggingFace token
  const [hfToken, setHfToken] = useState('')
  const [showHfToken, setShowHfToken] = useState(false)
  const [hfTokenSaving, setHfTokenSaving] = useState(false)

  // Track locally available models for "already downloaded" detection
  const [localModelIds, setLocalModelIds] = useState<Set<string>>(new Set())

  // Split layout: selected model for right-side README panel
  const [selectedModel, setSelectedModel] = useState<HFModel | null>(null)
  const [selectedReadme, setSelectedReadme] = useState<string | null>(null)
  const [loadingReadme, setLoadingReadme] = useState(false)

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onDownloadCompleteRef = useRef(onDownloadComplete)
  onDownloadCompleteRef.current = onDownloadComplete

  // Load recommended models, download dir, and HF token on mount
  useEffect(() => {
    window.api.models.getDownloadDir().then(setDownloadDir)
    window.api.settings.get('hf_api_key').then((val: string | null) => {
      if (val) setHfToken(val)
    })
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
    // Fetch default collection (JANG)
    setLoadingCollectionTabs(prev => ({ ...prev, jang: true }))
    setCollectionErrors(prev => { const next = { ...prev }; delete next.jang; return next })
    window.api.models.getCollectionModels(COLLECTION_SLUGS.jang)
      .then(models => setCollectionModels(prev => ({ ...prev, jang: models })))
      .catch(err => {
        // ms#68: record the error so the UI can show "Failed to load —
        // click to retry" instead of the ambiguous empty-state text.
        console.error('Failed to load JANG collection:', err)
        setCollectionErrors(prev => ({
          ...prev,
          jang: (err instanceof Error ? err.message : String(err)) || 'Fetch failed',
        }))
      })
      .finally(() => setLoadingCollectionTabs(prev => ({ ...prev, jang: false })))

    // Check for any in-progress downloads (activeAll covers concurrent downloads)
    window.api.models.getDownloadStatus().then((status: any) => {
      const repos = new Set<string>()
      for (const aj of status.activeAll || []) repos.add(aj.repoId)
      if (!repos.size && status.active) repos.add(status.active.repoId) // fallback for older IPC
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
      if (data.gated) {
        showToast('error', 'Gated model — HuggingFace token required',
          'This model requires authentication. Add your HF token in the download settings below.')
      } else {
        showToast('error', 'Download failed', errMsg)
      }
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
      doSearch(searchQuery, sortBy, newDir, modelType)
    }
  }, [doSearch, searchQuery, sortBy, sortDir, modelType])

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

  const handleSaveHfToken = async (token: string) => {
    setHfTokenSaving(true)
    try {
      if (token.trim()) {
        await window.api.settings.set('hf_api_key', token.trim())
      } else {
        await window.api.settings.delete('hf_api_key')
      }
      setHfToken(token.trim())
      showToast('success', token.trim() ? 'HuggingFace token saved' : 'HuggingFace token removed')
    } catch (err) {
      showToast('error', 'Failed to save token', (err as Error).message)
    } finally {
      setHfTokenSaving(false)
    }
  }

  const handleCollectionTabChange = useCallback(async (tab: CollectionTab) => {
    setCollectionTab(tab)
    // ms#68: refetch on tab click when we have no cached result AND no
    // prior error. If there's an error, let the user click the retry
    // button (below) to opt-in to another network roundtrip.
    if (!collectionModels[tab] && !collectionErrors[tab]) {
      setLoadingCollectionTabs(prev => ({ ...prev, [tab]: true }))
      setCollectionErrors(prev => { const next = { ...prev }; delete next[tab]; return next })
      try {
        const models = await window.api.models.getCollectionModels(COLLECTION_SLUGS[tab])
        setCollectionModels(prev => ({ ...prev, [tab]: models }))
      } catch (err) {
        console.error(`Failed to load ${tab} collection:`, err)
        setCollectionErrors(prev => ({
          ...prev,
          [tab]: (err instanceof Error ? err.message : String(err)) || 'Fetch failed',
        }))
      } finally {
        setLoadingCollectionTabs(prev => ({ ...prev, [tab]: false }))
      }
    }
  }, [collectionModels, collectionErrors])

  // ms#68: explicit retry — clears the error and refetches. Wired to the
  // "click to retry" button shown when a collection fetch has failed.
  const retryCollectionFetch = useCallback(async (tab: CollectionTab) => {
    setLoadingCollectionTabs(prev => ({ ...prev, [tab]: true }))
    setCollectionErrors(prev => { const next = { ...prev }; delete next[tab]; return next })
    try {
      const models = await window.api.models.getCollectionModels(COLLECTION_SLUGS[tab])
      setCollectionModels(prev => ({ ...prev, [tab]: models }))
    } catch (err) {
      console.error(`Retry failed for ${tab} collection:`, err)
      setCollectionErrors(prev => ({
        ...prev,
        [tab]: (err instanceof Error ? err.message : String(err)) || 'Fetch failed',
      }))
    } finally {
      setLoadingCollectionTabs(prev => ({ ...prev, [tab]: false }))
    }
  }, [])

  const handleBrowseDownloadDir = async () => {
    const result = await window.api.models.browseDownloadDir()
    if (!result.canceled && result.path) {
      await window.api.models.setDownloadDir(result.path)
      setDownloadDir(result.path)
    }
  }

  const activeCollection = collectionModels[collectionTab] || []
  const displayModels = searchQuery.trim() ? searchResults : activeCollection
  const isCollectionLoading = !searchQuery.trim() && !!loadingCollectionTabs[collectionTab]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Search and download MLX models from HuggingFace. Downloads run in the background.
        </p>
        <button
          onClick={() => window.dispatchEvent(new Event('open-download-popup'))}
          className="text-xs px-2 py-1 border border-border rounded hover:bg-accent text-muted-foreground hover:text-foreground flex-shrink-0"
        >
          View Downloads
        </button>
      </div>

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

      {/* HuggingFace Token */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">HF Token:</span>
          <div className="flex-1 relative">
            <input
              type={showHfToken ? 'text' : 'password'}
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="w-full px-2 py-1 pr-16 bg-background border border-input rounded text-xs font-mono"
            />
            <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <button
                onClick={() => setShowHfToken(!showHfToken)}
                className="px-1 py-0.5 text-[10px] text-muted-foreground hover:text-foreground"
                title={showHfToken ? 'Hide token' : 'Show token'}
              >
                {showHfToken ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
          <button
            onClick={() => handleSaveHfToken(hfToken)}
            disabled={hfTokenSaving}
            className="px-2 py-1 text-xs border border-border rounded hover:bg-accent whitespace-nowrap disabled:opacity-40"
          >
            {hfTokenSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
        <p className="text-[10px] text-muted-foreground ml-14">
          Required for gated models (Flux, Llama, etc).{' '}
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Get your token
          </a>
        </p>
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

      {/* Split Layout: Model List (left) + README (right) */}
      <div className="flex gap-4" style={{ height: 'calc(100vh - 280px)', minHeight: '400px' }}>
        {/* Left: Model List */}
        <div className="w-1/2 flex flex-col min-w-0">
          {searchQuery.trim() ? (
            <span className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Search Results</span>
          ) : (
            <div className="flex items-center gap-1 mb-2">
              <button
                onClick={() => handleCollectionTabChange('jang')}
                className={`px-2.5 py-1 text-xs rounded transition-colors ${collectionTab === 'jang' ? 'bg-primary/15 text-primary font-medium' : 'text-muted-foreground hover:bg-accent'}`}
              >
                JANG Models
              </button>
              <button
                onClick={() => handleCollectionTabChange('uncensored')}
                className={`px-2.5 py-1 text-xs rounded transition-colors ${collectionTab === 'uncensored' ? 'bg-red-500/15 text-red-400 font-medium' : 'text-muted-foreground hover:bg-accent'}`}
              >
                Uncensored
              </button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto space-y-1 pr-1">
            {isCollectionLoading ? (
              <p className="text-sm text-muted-foreground py-4 text-center">Loading models...</p>
            ) : !searchQuery.trim() && collectionErrors[collectionTab] ? (
              // ms#68: fetch failure — distinct from empty collection. Show
              // the actual error + a retry button so the user isn't stuck
              // staring at "No models" wondering if the network died.
              <div className="text-sm py-4 text-center space-y-2">
                <p className="text-muted-foreground">
                  Failed to load {collectionTab === 'jang' ? 'JANG' : 'Uncensored'} collection from HuggingFace.
                </p>
                <p className="text-xs text-muted-foreground/70 max-w-md mx-auto break-words">
                  {collectionErrors[collectionTab]}
                </p>
                <button
                  onClick={() => retryCollectionFetch(collectionTab)}
                  className="px-3 py-1 text-xs rounded border border-border hover:bg-accent"
                >
                  Retry
                </button>
              </div>
            ) : displayModels.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">
                {searchQuery.trim() ? (modelType === 'image' ? 'No image models found' : 'No MLX models found') : 'No models in this collection'}
              </p>
            ) : (
              displayModels.map(model => (
                <div key={model.id} onClick={() => {
                  setSelectedModel(model)
                  setSelectedReadme(null)
                  setLoadingReadme(true)
                  window.api.models.fetchReadme(model.id).then(text => {
                    setSelectedReadme(text || 'No README available.')
                    setLoadingReadme(false)
                  }).catch(() => {
                    setSelectedReadme('Failed to load README.')
                    setLoadingReadme(false)
                  })
                }} className={`cursor-pointer ${selectedModel?.id === model.id ? 'ring-1 ring-primary rounded' : ''}`}>
                  <ModelCard
                    model={model}
                    isDownloading={downloadingRepos.has(model.id)}
                    isDownloaded={localModelIds.has(model.id)}
                    onDownload={() => handleDownload(model.id)}
                  />
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right: README Panel */}
        <div className="w-1/2 flex flex-col min-w-0 border-l border-border pl-4">
          {selectedModel ? (
            <>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold truncate">{selectedModel.id}</h3>
                {selectedModel.size && <span className="text-xs font-bold text-foreground ml-2 flex-shrink-0">{selectedModel.size}</span>}
              </div>
              <div className="flex-1 overflow-y-auto">
                {loadingReadme ? (
                  <p className="text-xs text-muted-foreground py-4">Loading README...</p>
                ) : (
                  <ReadmeContent markdown={selectedReadme || ''} />
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              Click a model to view its README
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/** Simple markdown-to-HTML converter for HuggingFace READMEs.
 * Sanitized: only allows safe tags (no script, no event handlers).
 * Content comes from HF README files which are trusted markdown. */
function simpleMarkdownToHtml(md: string): string {
  let html = md
    // Code blocks (``` ... ```)
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Images: ![alt](url)
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" loading="lazy" />')
    // Links: [text](url)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold and italic
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // Horizontal rule
    .replace(/^---+$/gm, '<hr />')
    // Lists
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // Paragraphs (double newlines)
    .replace(/\n\n/g, '</p><p>')
    // Single newlines to <br> (within paragraphs)
    .replace(/\n/g, '<br />')
  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li>.*?<\/li>(?:<br \/>)?)+/g, (match) =>
    '<ul>' + match.replace(/<br \/>/g, '') + '</ul>'
  )
  // Sanitize: strip any script tags, event handlers, or dangerous attributes
  html = html.replace(/<script[\s\S]*?<\/script>/gi, '')
  html = html.replace(/\bon\w+\s*=\s*"[^"]*"/gi, '')
  html = html.replace(/\bon\w+\s*=\s*'[^']*'/gi, '')
  html = html.replace(/javascript:/gi, '')
  return '<p>' + html + '</p>'
}

function ReadmeContent({ markdown }: { markdown: string }) {
  const html = simpleMarkdownToHtml(markdown)
  return (
    <div
      className="prose prose-xs prose-invert max-w-none text-xs text-muted-foreground [&_h1]:text-sm [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1 [&_h2]:text-xs [&_h2]:font-bold [&_h2]:mt-2 [&_h2]:mb-1 [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:mt-2 [&_p]:my-1 [&_img]:max-w-full [&_img]:rounded [&_img]:my-2 [&_table]:text-[10px] [&_table]:border-collapse [&_td]:border [&_td]:border-border [&_td]:px-1.5 [&_td]:py-0.5 [&_th]:border [&_th]:border-border [&_th]:px-1.5 [&_th]:py-0.5 [&_th]:bg-muted [&_code]:bg-muted [&_code]:px-1 [&_code]:rounded [&_code]:text-[10px] [&_pre]:bg-muted [&_pre]:p-2 [&_pre]:rounded [&_pre]:text-[10px] [&_pre]:overflow-x-auto [&_a]:text-primary [&_a]:underline [&_ul]:pl-4 [&_ul]:list-disc [&_ol]:pl-4 [&_li]:my-0.5 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-2 [&_blockquote]:italic [&_hr]:border-border [&_hr]:my-2"
      dangerouslySetInnerHTML={{ __html: html }}
    />
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
    <div className="rounded border border-border hover:border-primary/30 transition-colors">
      <div className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm truncate" title={model.id}>
              {shortName}
            </div>
            <div className="text-xs text-muted-foreground">{model.author}</div>
            <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
              {model.size && <span title="Model size (safetensors)" className="font-semibold text-foreground">{model.size}</span>}
              <span title="Downloads">{formatNumber(model.downloads)} downloads</span>
              <span title="Likes">{model.likes} likes</span>
              {timeAgo(model.lastModified) && <span>{timeAgo(model.lastModified)}</span>}
            </div>
            {model.note && (
              <p className="text-[10px] text-muted-foreground mt-1 line-clamp-2 whitespace-pre-line">{model.note}</p>
            )}
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
                onClick={(e) => { e.stopPropagation(); onDownload() }}
                disabled={isDownloading}
                className="px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40 whitespace-nowrap"
              >
                {isDownloading ? 'Downloading...' : 'Download'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
