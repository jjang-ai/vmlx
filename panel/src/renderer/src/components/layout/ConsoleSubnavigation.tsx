import { useAppState } from '../../contexts/AppStateContext'
import { useTranslation } from '../../i18n'
import { consolePageForMode } from '../../lib/consoleNavigation'
import type { AppMode } from '../../types/app-state'

export function ConsoleSubnavigation() {
  const { state, setMode } = useAppState()
  const { t } = useTranslation()
  const page = consolePageForMode(state.mode)
  const items: Array<[AppMode, string]> = page === 'chat'
    ? [['chat', t('app.mode.chat')], ['image', t('app.mode.image')]]
    : page === 'server'
      ? [['server', t('app.mode.server')], ['api', t('app.mode.api')]]
      : [['tools', t('console.libraryTools')], ['models', t('console.findDownload')]]
  return (
    <nav aria-label={t('console.sections')} data-vmlx-section="console-subnavigation"
      className="flex shrink-0 gap-1 border-b border-border px-4 py-1 bg-background">
      {items.map(([mode, label]) => (
        <button key={mode} data-vmlx-control={`section-${mode}`} aria-current={state.mode === mode ? 'page' : undefined}
          onClick={() => setMode(mode)}
          className={`px-3 py-1 text-xs border-b-2 transition-colors ${state.mode === mode
            ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground'}`}>
          {label}
        </button>
      ))}
    </nav>
  )
}
