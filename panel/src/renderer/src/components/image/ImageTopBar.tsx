import { useState, useEffect } from 'react'
import { Settings, Square, RefreshCw, PanelLeftOpen, FolderOpen, ScrollText } from 'lucide-react'

function formatElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

import { useTranslation } from '../../i18n'
import type { ImageServerStatus } from '../../../../shared/imageCapabilities'

type ServerStatus = ImageServerStatus

interface ImageTopBarProps {
  model: string | null
  displayModelName?: string | null
  quantize: number
  status: ServerStatus
  port: number | null
  generating?: boolean
  mode: 'generate' | 'edit'
  onSettings: () => void
  onLogs: () => void
  onStop: () => void
  onWake: () => void
  onChangeModel: () => void
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
}

export function ImageTopBar({
  model,
  displayModelName,
  quantize,
  status,
  port,
  mode,
  generating,
  onSettings,
  onLogs,
  onStop,
  onWake,
  onChangeModel,
  sidebarCollapsed,
  onToggleSidebar
}: ImageTopBarProps) {
  const { t } = useTranslation()
  const quantizeLabel = quantize === 0 ? t('image.topbar.quantFull') : `${quantize}-bit`
  const [loadingElapsed, setLoadingElapsed] = useState(0)

  // Elapsed time counter when model is loading
  useEffect(() => {
    if (status !== 'starting') {
      setLoadingElapsed(0)
      return
    }
    const interval = setInterval(() => {
      setLoadingElapsed(prev => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [status])
  const displaySource = displayModelName || model
  const displayName = displaySource ? (displaySource.includes('/') ? displaySource.split('/').pop() : displaySource) : t('image.topbar.noModelSelected')

  return (
    <div className="h-11 border-b border-border flex items-center justify-between px-3 bg-background flex-shrink-0">
      <div className="flex items-center gap-2">
        {sidebarCollapsed && (
          <button
            onClick={onToggleSidebar}
            data-vmlx-control="image-toggle-sidebar"
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors mr-1"
            title={t('image.topbar.historyTitle')}
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        )}

        {/* Selecting a model opens inspection; it never starts a preset. */}
        <button
          onClick={onChangeModel}
          data-vmlx-control="image-switch-model"
          disabled={generating || status === 'starting'}
          className="min-w-0 flex items-center gap-2 text-sm font-medium hover:text-primary disabled:opacity-50 disabled:cursor-not-allowed"
          title={generating ? t('image.topbar.cannotSwitchTitle') : t('image.picker.chooseFolderTitle')}
        >
          <span className="truncate max-w-[200px]">{displayName}</span>
          <FolderOpen className="h-3.5 w-3.5 flex-shrink-0" />
        </button>

        {/* Mode badge */}
        {model && mode === 'edit' && (
          <span className="text-[10px] px-1.5 py-0.5 bg-violet-500/15 text-violet-400 rounded-full font-medium">{t('image.topBar.imageEdit')}</span>
        )}
        {model && mode === 'generate' && (
          <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/15 text-blue-400 rounded-full font-medium">{t('image.topBar.imageGen')}</span>
        )}
        {model && (
          <span className="text-[10px] px-1.5 py-0.5 bg-muted rounded-full text-muted-foreground">
            {quantizeLabel}
          </span>
        )}

        {/* Status indicator */}
        <div data-vmlx-control="image-runtime-status" data-vmlx-state={status} className="flex items-center gap-1.5 ml-2">
          <div className={`w-2 h-2 rounded-full ${
            status === 'running' ? 'bg-green-500' :
            status === 'starting' ? 'bg-yellow-500 animate-pulse' :
            status === 'error' ? 'bg-red-500' :
            'bg-gray-400'
          }`} />
          <span className="text-xs text-muted-foreground">
            {status === 'running' && port ? t('image.topbar.runningOnPort', { port }) :
             status === 'starting' ? `${t('chat.interface.loadingBanner')} ${formatElapsed(loadingElapsed)}` :
             status === 'error' ? t('status.error') :
             status === 'standby' ? t('status.sleeping') :
             t('status.stopped')}
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        <button
          onClick={onLogs}
          data-vmlx-control="image-logs"
          className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title={t('image.topbar.logsTitle')}
        >
          <ScrollText className="h-4 w-4" />
        </button>
        <button
          onClick={onSettings}
          data-vmlx-control="image-settings"
          className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title={t('image.topbar.settingsTitle')}
        >
          <Settings className="h-4 w-4" />
        </button>
        {status === 'standby' && (
          <button onClick={onWake} data-vmlx-control="image-wake"
            className="px-2 py-1.5 text-xs hover:bg-accent text-foreground"
            title={t('sessions.card.wake')}>
            {t('sessions.card.wake')}
          </button>
        )}
        {(status === 'running' || status === 'starting' || status === 'standby') && (
          <button
            onClick={onStop}
            data-vmlx-control="image-stop"
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-destructive transition-colors"
            title={status === 'starting' ? t('image.topbar.cancelLoadingTitle') : t('image.topbar.stopServerTitle')}
          >
            <Square className="h-4 w-4" />
          </button>
        )}
        {status === 'error' && (
          <button
            onClick={onChangeModel}
            data-vmlx-control="image-retry"
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title={t('image.topbar.retryTitle')}
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  )
}
