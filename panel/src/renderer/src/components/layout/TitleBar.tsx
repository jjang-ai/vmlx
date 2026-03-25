import { useState, useRef, useEffect } from 'react'
import { MessageSquare, Server, Wrench, Code2, ImageIcon, PanelLeftClose, PanelLeft, Info, Terminal } from 'lucide-react'
import { ThemeToggle } from '../ui/theme-toggle'
import { useAppState } from '../../contexts/AppStateContext'
import { useTranslation, LOCALE_NAMES, LOCALE_FLAGS, type Locale } from '../../i18n'


export function TitleBar() {
  const { state, setMode, dispatch } = useAppState()
  const { t, locale, setLocale } = useTranslation()

  return (
    <div
      className="flex items-center h-10 bg-card border-b border-border flex-shrink-0"
      style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
    >
      {/* macOS traffic light spacer + sidebar toggle */}
      <div
        className="flex items-center gap-1 pl-[72px] pr-2"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        {state.mode === 'chat' && (
          <button
            onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
            className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
            title={state.sidebarCollapsed ? t('app.sidebar.show') : t('app.sidebar.hide')}
          >
            {state.sidebarCollapsed
              ? <PanelLeft className="h-3.5 w-3.5" />
              : <PanelLeftClose className="h-3.5 w-3.5" />
            }
          </button>
        )}
      </div>

      {/* Center: mode toggle */}
      <div className="flex-1 flex justify-center">
        <div
          className="flex items-center bg-muted/80 rounded-lg p-0.5 gap-0.5 border border-border/30 shadow-sm"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
        >
          <ModeButton
            active={state.mode === 'code'}
            onClick={() => setMode('code')}
            icon={<Terminal className="h-3 w-3" />}
            label={t('app.mode.code')}
          />
          <ModeButton
            active={state.mode === 'chat'}
            onClick={() => setMode('chat')}
            icon={<MessageSquare className="h-3 w-3" />}
            label={t('app.mode.chat')}
          />
          <ModeButton
            active={state.mode === 'server'}
            onClick={() => {
              setMode('server')
              if (state.serverPanel === 'about') dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })
            }}
            icon={<Server className="h-3 w-3" />}
            label={t('app.mode.server')}
          />
          <ModeButton
            active={state.mode === 'tools'}
            onClick={() => setMode('tools')}
            icon={<Wrench className="h-3 w-3" />}
            label={t('app.mode.tools')}
          />
          <ModeButton
            active={state.mode === 'image'}
            onClick={() => setMode('image')}
            icon={<ImageIcon className="h-3 w-3" />}
            label={t('app.mode.image')}
          />
          <ModeButton
            active={state.mode === 'api'}
            onClick={() => setMode('api')}
            icon={<Code2 className="h-3 w-3" />}
            label={t('app.mode.api')}
          />
        </div>
      </div>

      {/* Right: language picker + about + theme toggle */}
      <div
        className="flex items-center gap-1 px-3"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <ThemeToggle />
        <LanguagePicker locale={locale} setLocale={setLocale} />
        <button
          onClick={() => {
            dispatch({ type: 'SET_MODE', mode: 'server' })
            dispatch({ type: 'SET_SERVER_PANEL', panel: 'about' })
          }}
          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
          title={t('app.about.settings')}
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

function LanguagePicker({ locale, setLocale }: { locale: Locale; setLocale: (l: Locale) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [open])

  const locales: Locale[] = ['en', 'zh', 'ko', 'ja', 'es']

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors text-sm leading-none"
        title="Language"
      >
        {LOCALE_FLAGS[locale]}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 bg-popover border border-border rounded-lg shadow-lg py-1 z-50 min-w-[140px]">
          {locales.map((l) => (
            <button
              key={l}
              onClick={() => { setLocale(l); setOpen(false) }}
              className={`w-full text-left px-3 py-1.5 text-sm flex items-center gap-2 hover:bg-accent transition-colors ${
                locale === l ? 'bg-primary/10 font-medium' : ''
              }`}
            >
              <span>{LOCALE_FLAGS[l]}</span>
              <span>{LOCALE_NAMES[l]}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function ModeButton({ active, onClick, icon, label }: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 ${
        active
          ? 'bg-background text-foreground shadow-md shadow-primary/10'
          : 'text-muted-foreground/70 hover:text-foreground hover:bg-background/50'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
