import { useAppState } from '../../contexts/AppStateContext'
import { useTranslation } from '../../i18n'
import { consolePageForMode } from '../../lib/consoleNavigation'
import { Sidebar } from './Sidebar'

interface ConsoleSidebarProps {
  onChatSelect: (chatId: string, modelPath: string) => void
  onNewChat: () => void
}

export function ConsoleSidebar({ onChatSelect, onNewChat }: ConsoleSidebarProps) {
  const { state, setMode, dispatch } = useAppState()
  const { t } = useTranslation()
  const page = consolePageForMode(state.mode)
  const collapsed = state.sidebarCollapsed
  return (
    <aside data-vmlx-section="console-sidebar"
      className={`shrink-0 border-r border-border bg-background flex flex-col min-h-0 ${collapsed ? 'w-14' : 'w-[244px] max-[900px]:w-[200px]'}`}>
      <div className={`py-6 ${collapsed ? 'px-3' : 'px-6'}`} aria-label="vMLX">
        <span className="font-mono text-2xl tracking-tighter"><span className="text-primary">v</span>{collapsed ? '_' : 'mlx_'}</span>
      </div>
      <nav aria-label={t('console.pages')} className="px-2 pb-5 border-b border-border">
        {([
          ['chat', '01', t('console.chatImages')],
          ['server', '02', t('console.serversApi')],
          ['models', '03', t('console.models')],
        ] as const).map(([mode, number, label]) => {
          const active = page === mode
          return <button key={mode} data-vmlx-control={`mode-${mode}`}
            data-vmlx-state={active ? 'active' : 'inactive'} aria-current={active ? 'page' : undefined}
            title={label} aria-label={label}
            onClick={() => {
              if (mode === 'models') setMode('tools')
              else {
                setMode(mode)
                if (mode === 'server' && state.serverPanel === 'about') dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })
              }
            }}
            className={`w-full min-h-10 flex items-center gap-3 text-left text-xs border-l-2 px-3 ${active
              ? 'border-primary bg-accent text-foreground' : 'border-transparent text-muted-foreground hover:bg-card hover:text-foreground'}`}>
            <span className="tabular-nums text-muted-foreground">{number}</span>
            {!collapsed && <span>{label}</span>}
          </button>
        })}
      </nav>
      {/* Retain history/search state across primary-page changes. */}
      <div className="flex-1 min-h-0" hidden={collapsed}>
        <Sidebar embedded collapsed={false} currentChatId={state.activeChatId}
          onNewChat={() => { setMode('chat'); onNewChat() }}
          onChatSelect={(id, path) => { setMode('chat'); onChatSelect(id, path) }} />
      </div>
      <button data-vmlx-control="console-preferences" title={t('console.preferences')}
        aria-label={t('console.preferences')}
        className="mt-auto border-t border-border px-4 py-4 text-xs text-left text-muted-foreground hover:bg-card hover:text-foreground"
        onClick={() => { setMode('server'); dispatch({ type: 'SET_SERVER_PANEL', panel: 'about' }) }}>
        {collapsed ? '⚙' : t('console.preferences')}
      </button>
    </aside>
  )
}
