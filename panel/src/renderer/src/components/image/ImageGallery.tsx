import { useState, useEffect, useCallback } from 'react'
import { Download, Copy, Loader2, ImageIcon } from 'lucide-react'
import type { ImageGenerationInfo } from './ImageTab'

interface ImageGalleryProps {
  generations: ImageGenerationInfo[]
  generating: boolean
}

export function ImageGallery({ generations, generating }: ImageGalleryProps) {
  if (generations.length === 0 && !generating) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center px-8">
        <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
          <ImageIcon className="h-8 w-8 text-primary" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Generate your first image</h3>
        <p className="text-sm text-muted-foreground max-w-sm">
          Type a prompt below and click Generate to create an image with the selected model.
        </p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {generations.map((gen) => (
          <ImageCard key={gen.id} generation={gen} />
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

function ImageCard({ generation }: { generation: ImageGenerationInfo }) {
  const [imageData, setImageData] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [hovered, setHovered] = useState(false)

  useEffect(() => {
    let cancelled = false
    window.api.image.readFile(generation.imagePath).then((data: string | null) => {
      if (!cancelled) {
        setImageData(data)
        setLoading(false)
      }
    }).catch(() => {
      if (!cancelled) setLoading(false)
    })
    return () => { cancelled = true }
  }, [generation.imagePath])

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
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Image */}
      <div className="aspect-square bg-muted relative">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
          </div>
        ) : imageData ? (
          <img
            src={imageData}
            alt={generation.prompt}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs">
            Failed to load
          </div>
        )}

        {/* Hover overlay */}
        {hovered && !loading && imageData && (
          <div className="absolute inset-0 bg-black/40 flex items-end justify-end p-2 gap-1.5">
            <button
              onClick={handleSave}
              className="px-2 py-1 bg-white/90 text-black rounded text-xs flex items-center gap-1 hover:bg-white transition-colors"
              title="Save image"
            >
              <Download className="h-3 w-3" />
              Save
            </button>
            {generation.seed != null && (
              <button
                onClick={handleCopySeed}
                className="px-2 py-1 bg-white/90 text-black rounded text-xs flex items-center gap-1 hover:bg-white transition-colors"
                title="Copy seed"
              >
                <Copy className="h-3 w-3" />
                Seed
              </button>
            )}
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="p-3">
        <p className="text-xs text-foreground line-clamp-2 mb-1" title={generation.prompt}>
          {generation.prompt}
        </p>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span>{generation.width}x{generation.height}</span>
          <span>{generation.steps} steps</span>
          {generation.elapsedSeconds != null && (
            <span>{generation.elapsedSeconds.toFixed(1)}s</span>
          )}
          {generation.seed != null && (
            <span>seed: {generation.seed}</span>
          )}
        </div>
      </div>
    </div>
  )
}
