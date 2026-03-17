import { useState, useEffect, useCallback } from 'react'
import { Download, Copy, Loader2, ImageIcon, Pencil, RefreshCw } from 'lucide-react'
import type { ImageGenerationInfo } from './ImageTab'

interface ImageGalleryProps {
  generations: ImageGenerationInfo[]
  generating: boolean
  mode?: 'generate' | 'edit'
  onRegenerate?: (gen: ImageGenerationInfo) => void
}

export function ImageGallery({ generations, generating, mode, onRegenerate }: ImageGalleryProps) {
  if (generations.length === 0 && !generating) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center px-8">
        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${
          mode === 'edit' ? 'bg-violet-500/10' : 'bg-primary/10'
        }`}>
          {mode === 'edit'
            ? <Pencil className="h-8 w-8 text-violet-400" />
            : <ImageIcon className="h-8 w-8 text-primary" />
          }
        </div>
        <h3 className="text-lg font-semibold mb-2">
          {mode === 'edit' ? 'Edit your first image' : 'Generate your first image'}
        </h3>
        <p className="text-sm text-muted-foreground max-w-sm">
          {mode === 'edit'
            ? 'Upload a source image and type a prompt below to edit it with the selected model.'
            : 'Type a prompt below and click Generate to create an image with the selected model.'
          }
        </p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-4">
      <div className={`grid gap-4 ${
        mode === 'edit'
          ? 'grid-cols-1 md:grid-cols-2'
          : 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3'
      }`} style={{ minWidth: 0 }}>
        {generations.map((gen) => (
          <ImageCard key={gen.id} generation={gen} onRegenerate={onRegenerate} />
        ))}

        {/* Loading skeleton while generating */}
        {generating && (
          <div className="border border-border rounded-lg overflow-hidden animate-pulse">
            <div className="aspect-square bg-muted flex items-center justify-center">
              <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
            </div>
            <div className="p-3 space-y-2">
              <div className="h-3 bg-muted rounded w-3/4" />
              <div className="h-3 bg-muted rounded w-1/2" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ImageCard({ generation, onRegenerate }: { generation: ImageGenerationInfo; onRegenerate?: (gen: ImageGenerationInfo) => void }) {
  const [imageData, setImageData] = useState<string | null>(null)
  const [sourceData, setSourceData] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [hovered, setHovered] = useState(false)

  const isEdit = !!generation.sourceImagePath

  useEffect(() => {
    let cancelled = false
    const promises: Promise<void>[] = []

    // Load output image
    promises.push(
      window.api.image.readFile(generation.imagePath).then((data: string | null) => {
        if (!cancelled) setImageData(data)
      }).catch(() => {})
    )

    // Load source image for edits
    if (generation.sourceImagePath) {
      promises.push(
        window.api.image.readFile(generation.sourceImagePath).then((data: string | null) => {
          if (!cancelled) setSourceData(data)
        }).catch(() => {})
      )
    }

    Promise.all(promises).then(() => {
      if (!cancelled) setLoading(false)
    })

    return () => { cancelled = true }
  }, [generation.imagePath, generation.sourceImagePath])

  const handleSave = useCallback(async () => {
    await window.api.image.saveFile(generation.imagePath)
  }, [generation.imagePath])

  const handleCopySeed = useCallback(() => {
    if (generation.seed != null) {
      navigator.clipboard.writeText(generation.seed.toString()).catch(() => {})
    }
  }, [generation.seed])

  return (
    <div
      className="border border-border rounded-lg overflow-hidden group"
      style={{ minWidth: 240 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Image display */}
      {isEdit ? (
        // Edit layout: large result image with small source thumbnail overlay
        <div className="relative">
          {/* Main result image (full size) */}
          <div className="aspect-square bg-muted relative">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
              </div>
            ) : imageData ? (
              <img src={imageData} alt="Edited" className="w-full h-full object-contain" />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs">
                Failed to load
              </div>
            )}
            <span className="absolute top-1.5 right-1.5 text-[9px] px-1.5 py-0.5 bg-violet-600/80 text-white rounded font-medium">Edited</span>

            {/* Source image thumbnail (bottom-left corner) */}
            {sourceData && (
              <div className="absolute bottom-2 left-2 w-16 h-16 rounded border-2 border-white/60 overflow-hidden shadow-lg">
                <img src={sourceData} alt="Original" className="w-full h-full object-contain" />
                <span className="absolute bottom-0 left-0 right-0 text-[7px] text-center bg-black/60 text-white py-px">Original</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        // Standard single-image layout for generations
        <div className="aspect-square bg-muted relative">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
            </div>
          ) : imageData ? (
            <img
              src={imageData}
              alt={generation.prompt}
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs">
              Failed to load
            </div>
          )}

          {/* Hover overlay for seed copy */}
          {hovered && !loading && imageData && generation.seed != null && (
            <div className="absolute top-1.5 right-1.5">
              <button
                onClick={handleCopySeed}
                className="px-2 py-1 bg-black/60 text-white rounded text-xs flex items-center gap-1 hover:bg-black/80 transition-colors backdrop-blur-sm"
                title="Copy seed"
              >
                <Copy className="h-3 w-3" />
                {generation.seed}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Metadata */}
      <div className="p-3">
        <div className="flex items-center gap-1 mb-1">
          {isEdit && (
            <span className="text-[9px] px-1 py-0 rounded bg-violet-500/15 text-violet-400 flex-shrink-0">Edit</span>
          )}
          <p className="text-xs text-foreground line-clamp-2" title={generation.prompt}>
            {generation.prompt}
          </p>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span>{generation.width}x{generation.height}</span>
          <span>{generation.steps} steps</span>
          {generation.strength != null && (
            <span>str: {generation.strength}</span>
          )}
          {generation.elapsedSeconds != null && (
            <span>{generation.elapsedSeconds.toFixed(1)}s</span>
          )}
          {generation.seed != null && (
            <span>seed: {generation.seed}</span>
          )}
        </div>

        {/* Action buttons — always visible */}
        {!loading && imageData && (
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border">
            {onRegenerate && (
              <button
                onClick={() => onRegenerate(generation)}
                className={`flex-1 py-1.5 rounded text-xs font-medium flex items-center justify-center gap-1.5 transition-colors ${
                  isEdit
                    ? 'bg-violet-500/15 text-violet-400 hover:bg-violet-500/25'
                    : 'bg-primary/10 text-primary hover:bg-primary/20'
                }`}
                title="Use this image as starting point for next generation"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Iterate
              </button>
            )}
            <button
              onClick={handleSave}
              className="flex-1 py-1.5 rounded text-xs font-medium flex items-center justify-center gap-1.5 bg-muted text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              title="Save image to disk"
            >
              <Download className="h-3.5 w-3.5" />
              Save
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
