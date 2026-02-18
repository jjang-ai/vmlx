import { useState, KeyboardEvent } from 'react'

interface InputBoxProps {
  onSend: (message: string) => void
  onAbort?: () => void
  disabled?: boolean
  loading?: boolean
}

export function InputBox({ onSend, onAbort, disabled, loading }: InputBoxProps) {
  const [message, setMessage] = useState('')

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message)
      setMessage('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
    // Escape to stop generation
    if (e.key === 'Escape' && loading && onAbort) {
      onAbort()
    }
  }

  return (
    <div className="border-t border-border p-4">
      <div className="flex gap-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={loading ? "Waiting for response... (Esc to stop)" : "Type your message... (Shift+Enter for new line)"}
          disabled={disabled && !loading}
          className="flex-1 resize-none px-4 py-3 bg-background border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring min-h-[60px] max-h-[200px]"
          rows={3}
        />
        {loading ? (
          <button
            onClick={onAbort}
            className="px-6 py-3 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 font-medium"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={disabled || !message.trim()}
            className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            Send
          </button>
        )}
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        {loading ? 'Press Esc or click Stop to cancel' : 'Press Enter to send, Shift+Enter for new line'}
      </p>
    </div>
  )
}
