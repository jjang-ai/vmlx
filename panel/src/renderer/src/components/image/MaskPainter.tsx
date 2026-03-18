import { useState, useRef, useEffect, useCallback } from 'react'
import { Paintbrush, Eraser, RotateCcw, Check, X, Minus, Plus, Square } from 'lucide-react'

interface MaskPainterProps {
  /** Source image as data URL */
  imageDataUrl: string
  /** Called when user confirms the mask. Returns mask as base64 PNG (white = edit area, black = keep) */
  onConfirm: (maskBase64: string) => void
  /** Called when user cancels mask painting */
  onCancel: () => void
}

/**
 * Canvas-based mask painting tool for Fill inpainting.
 * User paints white areas over the image to mark regions for editing.
 * White = area to fill/edit, black = area to keep unchanged.
 */
export function MaskPainter({ imageDataUrl, onConfirm, onCancel }: MaskPainterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const [brushSize, setBrushSize] = useState(30)
  const [tool, setTool] = useState<'brush' | 'eraser' | 'rect'>('brush')
  const [drawing, setDrawing] = useState(false)
  const rectStart = useRef<{ x: number; y: number } | null>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const lastPos = useRef<{ x: number; y: number } | null>(null)

  // Load the source image
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setImageLoaded(true)
    }
    img.src = imageDataUrl
  }, [imageDataUrl])

  // Initialize canvases when image loads
  useEffect(() => {
    if (!imageLoaded || !imgRef.current) return
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas) return

    const img = imgRef.current
    // Scale to fit container while maintaining aspect ratio
    const maxW = 640
    const maxH = 640
    const scale = Math.min(maxW / img.width, maxH / img.height, 1)
    const w = Math.round(img.width * scale)
    const h = Math.round(img.height * scale)

    canvas.width = w
    canvas.height = h
    maskCanvas.width = img.width  // Full resolution for mask output
    maskCanvas.height = img.height

    // Initialize mask canvas to black (keep everything)
    const maskCtx = maskCanvas.getContext('2d')!
    maskCtx.fillStyle = '#000000'
    maskCtx.fillRect(0, 0, img.width, img.height)

    redraw()
  }, [imageLoaded])

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas || !imgRef.current) return

    const ctx = canvas.getContext('2d')!
    const img = imgRef.current

    // Draw source image
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Overlay mask with semi-transparent red
    ctx.save()
    ctx.globalAlpha = 0.4
    // Draw mask scaled to display size
    ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height)
    ctx.restore()

    // Also overlay as red tint for better visibility
    const maskCtx = maskCanvas.getContext('2d')!
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
    const displayCtx = ctx

    // Create a colored overlay from the mask
    const overlay = ctx.createImageData(canvas.width, canvas.height)
    const scaleX = maskCanvas.width / canvas.width
    const scaleY = maskCanvas.height / canvas.height

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const mx = Math.floor(x * scaleX)
        const my = Math.floor(y * scaleY)
        const mi = (my * maskCanvas.width + mx) * 4
        const oi = (y * canvas.width + x) * 4

        if (maskData.data[mi] > 128) {
          // White in mask = area to edit — show as red overlay
          overlay.data[oi] = 255     // R
          overlay.data[oi + 1] = 50  // G
          overlay.data[oi + 2] = 50  // B
          overlay.data[oi + 3] = 100 // A
        }
      }
    }
    ctx.putImageData(overlay, 0, 0)
    // Redraw image underneath
    ctx.globalCompositeOperation = 'destination-over'
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    ctx.globalCompositeOperation = 'source-over'
  }, [])

  const getCanvasPos = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    }
  }, [])

  const paintAt = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas || !imgRef.current) return

    const maskCtx = maskCanvas.getContext('2d')!
    // Scale display coords to mask coords
    const scaleX = maskCanvas.width / canvas.width
    const scaleY = maskCanvas.height / canvas.height
    const mx = x * scaleX
    const my = y * scaleY
    const mr = brushSize * scaleX

    maskCtx.beginPath()
    maskCtx.arc(mx, my, mr, 0, Math.PI * 2)
    maskCtx.fillStyle = tool === 'brush' ? '#ffffff' : '#000000'
    maskCtx.fill()

    redraw()
  }, [brushSize, tool, redraw])

  const paintLine = useCallback((from: { x: number; y: number }, to: { x: number; y: number }) => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas || !imgRef.current) return

    const maskCtx = maskCanvas.getContext('2d')!
    const scaleX = maskCanvas.width / canvas.width
    const scaleY = maskCanvas.height / canvas.height
    const mr = brushSize * scaleX

    maskCtx.beginPath()
    maskCtx.strokeStyle = tool === 'brush' ? '#ffffff' : '#000000'
    maskCtx.lineWidth = mr * 2
    maskCtx.lineCap = 'round'
    maskCtx.lineJoin = 'round'
    maskCtx.moveTo(from.x * scaleX, from.y * scaleY)
    maskCtx.lineTo(to.x * scaleX, to.y * scaleY)
    maskCtx.stroke()

    redraw()
  }, [brushSize, tool, redraw])

  const paintRect = useCallback((from: { x: number; y: number }, to: { x: number; y: number }) => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')!
    const scaleX = maskCanvas.width / canvas.width
    const scaleY = maskCanvas.height / canvas.height

    const x = Math.min(from.x, to.x) * scaleX
    const y = Math.min(from.y, to.y) * scaleY
    const w = Math.abs(to.x - from.x) * scaleX
    const h = Math.abs(to.y - from.y) * scaleY

    maskCtx.fillStyle = '#ffffff'
    maskCtx.fillRect(x, y, w, h)
    redraw()
  }, [redraw])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setDrawing(true)
    const pos = getCanvasPos(e)
    lastPos.current = pos
    if (tool === 'rect') {
      rectStart.current = pos
    } else {
      paintAt(pos.x, pos.y)
    }
  }, [getCanvasPos, paintAt, tool])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!drawing) return
    const pos = getCanvasPos(e)
    if (tool === 'rect') {
      // Preview rectangle on display canvas only
      redraw()
      const canvas = canvasRef.current
      if (canvas && rectStart.current) {
        const ctx = canvas.getContext('2d')!
        ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)'
        ctx.lineWidth = 2
        ctx.setLineDash([6, 3])
        ctx.strokeRect(
          rectStart.current.x, rectStart.current.y,
          pos.x - rectStart.current.x, pos.y - rectStart.current.y
        )
        ctx.setLineDash([])
      }
    } else if (lastPos.current) {
      paintLine(lastPos.current, pos)
    }
    lastPos.current = pos
  }, [drawing, getCanvasPos, paintLine, tool, redraw])

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    if (tool === 'rect' && rectStart.current && drawing) {
      const pos = getCanvasPos(e)
      paintRect(rectStart.current, pos)
      rectStart.current = null
    }
    setDrawing(false)
    lastPos.current = null
  }, [tool, drawing, getCanvasPos, paintRect])

  const handleClear = useCallback(() => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return
    const ctx = maskCanvas.getContext('2d')!
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
    redraw()
  }, [redraw])

  const handleConfirm = useCallback(() => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return
    // Export mask as PNG base64
    const dataUrl = maskCanvas.toDataURL('image/png')
    onConfirm(dataUrl)
  }, [onConfirm])

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Toolbar */}
      <div className="flex items-center gap-2 bg-muted rounded-lg px-3 py-1.5">
        <button
          onClick={() => setTool('brush')}
          className={`p-1.5 rounded transition-colors ${tool === 'brush' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
          title="Paint mask (white = area to edit)"
        >
          <Paintbrush className="h-4 w-4" />
        </button>
        <button
          onClick={() => setTool('rect')}
          className={`p-1.5 rounded transition-colors ${tool === 'rect' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
          title="Rectangle select — drag a box over the area to fill"
        >
          <Square className="h-4 w-4" />
        </button>
        <button
          onClick={() => setTool('eraser')}
          className={`p-1.5 rounded transition-colors ${tool === 'eraser' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
          title="Erase mask (remove painted area)"
        >
          <Eraser className="h-4 w-4" />
        </button>

        <div className="w-px h-5 bg-border mx-1" />

        {/* Brush size */}
        <button onClick={() => setBrushSize(s => Math.max(5, s - 5))} className="p-1 text-muted-foreground hover:text-foreground" title="Smaller brush">
          <Minus className="h-3.5 w-3.5" />
        </button>
        <span className="text-xs text-muted-foreground w-8 text-center" title="Brush size in pixels">{brushSize}px</span>
        <button onClick={() => setBrushSize(s => Math.min(100, s + 5))} className="p-1 text-muted-foreground hover:text-foreground" title="Larger brush">
          <Plus className="h-3.5 w-3.5" />
        </button>

        <div className="w-px h-5 bg-border mx-1" />

        <button onClick={handleClear} className="p-1.5 text-muted-foreground hover:text-foreground rounded" title="Clear entire mask">
          <RotateCcw className="h-4 w-4" />
        </button>

        <div className="w-px h-5 bg-border mx-1" />

        <button onClick={handleConfirm} className="px-3 py-1 bg-primary text-primary-foreground rounded text-xs font-medium flex items-center gap-1" title="Apply mask and edit">
          <Check className="h-3.5 w-3.5" /> Apply
        </button>
        <button onClick={onCancel} className="px-2 py-1 text-muted-foreground hover:text-destructive rounded text-xs" title="Cancel mask painting">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Help text */}
      <p className="text-[10px] text-muted-foreground">
        Paint over areas you want to edit (shown in red). The model will fill/replace only the painted areas.
      </p>

      {/* Canvas */}
      <div className="relative border border-border rounded-lg overflow-hidden" style={{ cursor: tool === 'brush' ? 'crosshair' : 'cell' }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="block"
        />
        {/* Hidden mask canvas (full resolution) */}
        <canvas ref={maskCanvasRef} className="hidden" />

        {/* Brush cursor preview */}
        {!drawing && (
          <div
            className="pointer-events-none absolute rounded-full border-2 border-white/60"
            style={{
              width: brushSize * 2,
              height: brushSize * 2,
              transform: 'translate(-50%, -50%)',
              display: 'none', // Shown via CSS :hover in production
            }}
          />
        )}
      </div>
    </div>
  )
}
