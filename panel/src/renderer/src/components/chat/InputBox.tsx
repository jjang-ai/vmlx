import { useState, useRef, useCallback, useEffect, KeyboardEvent, DragEvent, ClipboardEvent } from 'react'
import { Paperclip, Send, Square, ImagePlus, X } from 'lucide-react'
import { VoiceChat } from './VoiceChat'

export interface ImageAttachment {
  id: string
  dataUrl: string
  name: string
  type: string
}

interface InputBoxProps {
  onSend: (message: string, attachments?: ImageAttachment[]) => void
  onAbort?: () => void
  disabled?: boolean
  loading?: boolean
  sessionEndpoint?: { host: string; port: number }
  sessionId?: string
}

export function InputBox({ onSend, onAbort, disabled, loading, sessionEndpoint, sessionId }: InputBoxProps) {
  const [message, setMessage] = useState('')
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea based on content
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [message])

  // Auto-focus textarea when component mounts or loading completes
  useEffect(() => {
    if (!loading && !disabled) {
      textareaRef.current?.focus()
    }
  }, [loading, disabled])

  const handleSend = () => {
    if ((message.trim() || attachments.length > 0) && !disabled) {
      onSend(message, attachments.length > 0 ? attachments : undefined)
      setMessage('')
      setAttachments([])
      // Reset file input so the same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
    if (e.key === 'Escape' && loading && onAbort) {
      onAbort()
    }
  }

  const addFiles = useCallback((files: FileList | File[]) => {
    const imageFiles = Array.from(files).filter(f =>
      f.type.startsWith('image/') && f.size <= 10 * 1024 * 1024
    )
    for (const file of imageFiles) {
      const reader = new FileReader()
      reader.onload = () => {
        setAttachments(prev => [...prev, {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          dataUrl: reader.result as string,
          name: file.name,
          type: file.type
        }])
      }
      reader.onerror = () => {
        console.error('Failed to read file:', file.name, reader.error)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const handlePaste = useCallback((e: ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return
    const imageItems = Array.from(items).filter(i => i.type.startsWith('image/'))
    if (imageItems.length === 0) return
    e.preventDefault()
    const files = imageItems.map(item => item.getAsFile()).filter(Boolean) as File[]
    addFiles(files)
  }, [addFiles])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer?.files) {
      addFiles(e.dataTransfer.files)
    }
  }, [addFiles])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    // Only clear drag state when leaving the container entirely (not entering a child)
    if (e.currentTarget.contains(e.relatedTarget as Node)) return
    setIsDragOver(false)
  }, [])

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id))
  }

  const handleTranscription = useCallback((text: string) => {
    setMessage(prev => prev ? prev + ' ' + text : text)
  }, [])

  return (
    <div
      className={`relative border-t border-border p-4 transition-colors ${isDragOver ? 'bg-primary/5' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {/* Drag overlay */}
      {isDragOver && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/80 border-2 border-dashed border-primary/40 rounded-lg m-1 pointer-events-none">
          <div className="flex flex-col items-center gap-1 text-primary">
            <ImagePlus className="h-6 w-6" />
            <span className="text-xs font-medium">Drop image to attach</span>
          </div>
        </div>
      )}

      {attachments.length > 0 && (
        <div className="flex gap-2 mb-3 flex-wrap">
          {attachments.map(att => (
            <div key={att.id} className="relative group">
              <img
                src={att.dataUrl}
                alt={att.name}
                className="h-20 w-20 object-cover rounded-lg border border-border"
              />
              <button
                onClick={() => removeAttachment(att.id)}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="h-3 w-3" />
              </button>
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[9px] px-1 truncate rounded-b-lg">
                {att.name}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/gif,image/webp"
          multiple
          className="hidden"
          onChange={(e) => { if (e.target.files) addFiles(e.target.files) }}
        />
        <div className="flex items-center gap-1">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled && !loading}
            className="p-2 rounded-lg hover:bg-accent disabled:opacity-40 text-muted-foreground hover:text-foreground transition-colors"
            title="Attach image"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <VoiceChat
            onTranscription={handleTranscription}
            endpoint={sessionEndpoint}
            sessionId={sessionId}
            disabled={disabled && !loading}
          />
        </div>
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={loading ? "Waiting for response..." : "Message..."}
          disabled={disabled && !loading}
          className="flex-1 resize-none px-4 py-2.5 bg-background border border-input rounded-xl focus:outline-none focus:ring-2 focus:ring-ring/50 min-h-[42px] max-h-[200px] text-sm leading-relaxed"
          rows={1}
        />
        {loading ? (
          <button
            onClick={onAbort}
            className="p-2.5 bg-destructive text-destructive-foreground rounded-xl hover:bg-destructive/90 transition-colors flex-shrink-0"
            title="Stop generating (Esc)"
          >
            <Square className="h-4 w-4" />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={disabled || (!message.trim() && attachments.length === 0)}
            className="p-2.5 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 disabled:opacity-30 transition-colors flex-shrink-0"
            title="Send message (Enter)"
          >
            <Send className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  )
}
