import { Settings, Square, RefreshCw, PanelLeftOpen } from 'lucide-react'

type ServerStatus = 'stopped' | 'starting' | 'running' | 'error'

interface ImageTopBarProps {
  model: string | null
  quantize: number
  status: ServerStatus
  port: number | null
  onSettings: () => void
  onStop: () => void
  onChangeModel: () => void
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
}

export function ImageTopBar({
  model,
  quantize,
  status,
  port,
  onSettings,
  onStop,
  onChangeModel,
  sidebarCollapsed,
  onToggleSidebar
}: ImageTopBarProps) {
  const quantizeLabel = quantize === 0 ? 'Full' : `${quantize}-bit`

  return (
    <div className="h-11 border-b border-border flex items-center justify-between px-3 bg-background flex-shrink-0">
      <div className="flex items-center gap-2">
        {sidebarCollapsed && (
          <button
            onClick={onToggleSidebar}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors mr-1"
            title="Show history"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        )}

        {/* Model name + quantize */}
        <button
          onClick={onChangeModel}
          className="text-sm font-medium hover:text-primary transition-colors"
          title="Change model"
        >
          {model || 'No model selected'}
        </button>
        {model && (
          <span className="text-[10px] px-1.5 py-0.5 bg-muted rounded-full text-muted-foreground">
            {quantizeLabel}
          </span>
        )}
        <button
          onClick={onChangeModel}
          className="text-[11px] text-muted-foreground hover:text-foreground px-1.5 py-0.5 rounded hover:bg-accent transition-colors"
        >
          Change
        </button>

        {/* Status indicator */}
        <div className="flex items-center gap-1.5 ml-2">
          <div className={`w-2 h-2 rounded-full ${
            status === 'running' ? 'bg-green-500' :
            status === 'starting' ? 'bg-yellow-500 animate-pulse' :
            status === 'error' ? 'bg-red-500' :
            'bg-gray-400'
          }`} />
          <span className="text-xs text-muted-foreground">
            {status === 'running' && port ? `Running on :${port}` :
             status === 'starting' ? 'Starting...' :
             status === 'error' ? 'Error' :
             'Stopped'}
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        <button
          onClick={onSettings}
          className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Settings"
        >
          <Settings className="h-4 w-4" />
        </button>
        {(status === 'running' || status === 'starting') && (
          <button
            onClick={onStop}
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-destructive transition-colors"
            title={status === 'starting' ? 'Cancel loading' : 'Stop server'}
          >
            <Square className="h-4 w-4" />
          </button>
        )}
        {status === 'error' && (
          <button
            onClick={onChangeModel}
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title="Retry"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  )
}
