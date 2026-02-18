import { useState, useEffect, useCallback, createContext, useContext } from 'react'

interface ToastMessage {
  id: number
  type: 'error' | 'warning' | 'info' | 'success'
  title: string
  detail?: string
}

interface ToastContextValue {
  showToast: (type: ToastMessage['type'], title: string, detail?: string) => void
}

const ToastContext = createContext<ToastContextValue>({ showToast: () => {} })

export function useToast() {
  return useContext(ToastContext)
}

let nextId = 0

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  const showToast = useCallback((type: ToastMessage['type'], title: string, detail?: string) => {
    const id = nextId++
    setToasts(prev => [...prev, { id, type, title, detail }])
    // Auto-dismiss after 6s (errors stay longer at 10s)
    const ms = type === 'error' ? 10000 : 6000
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), ms)
  }, [])

  const dismiss = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {/* Toast container — fixed at bottom-right */}
      {toasts.length > 0 && (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md">
          {toasts.map(toast => (
            <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
          ))}
        </div>
      )}
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onDismiss }: { toast: ToastMessage; onDismiss: (id: number) => void }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Trigger enter animation
    requestAnimationFrame(() => setVisible(true))
  }, [])

  const colors = {
    error: 'border-destructive/50 bg-destructive/10 text-destructive-foreground',
    warning: 'border-warning/50 bg-warning/10 text-foreground',
    info: 'border-primary/50 bg-primary/10 text-foreground',
    success: 'border-primary/50 bg-primary/10 text-foreground',
  }

  const icons = {
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
    success: '✓',
  }

  const iconColors = {
    error: 'text-destructive',
    warning: 'text-warning',
    info: 'text-primary',
    success: 'text-primary',
  }

  return (
    <div
      className={`border rounded-lg px-4 py-3 shadow-lg backdrop-blur-sm transition-all duration-200 ${colors[toast.type]} ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <span className={`text-sm font-bold flex-shrink-0 mt-0.5 ${iconColors[toast.type]}`}>
          {icons[toast.type]}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium">{toast.title}</p>
          {toast.detail && (
            <p className="text-xs text-muted-foreground mt-1 break-words font-mono">{toast.detail}</p>
          )}
        </div>
        <button
          onClick={() => onDismiss(toast.id)}
          className="text-muted-foreground hover:text-foreground text-xs flex-shrink-0 ml-2"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
