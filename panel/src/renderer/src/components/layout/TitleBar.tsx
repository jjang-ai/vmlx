import { MessageSquare, Server, Wrench, Code2, ImageIcon, PanelLeftClose, PanelLeft, Info, Terminal } from 'lucide-react'
import { ThemeToggle } from '../ui/theme-toggle'
import { useAppState } from '../../contexts/AppStateContext'


export function TitleBar() {
  const { state, setMode, dispatch } = useAppState()

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
            title={state.sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
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
          className="flex items-center bg-muted rounded-md p-0.5 gap-0.5"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
        >
          <ModeButton
            active={state.mode === 'code'}
            onClick={() => setMode('code')}
            icon={<Terminal className="h-3 w-3" />}
            label="Code"
          />
          <ModeButton
            active={state.mode === 'chat'}
            onClick={() => setMode('chat')}
            icon={<MessageSquare className="h-3 w-3" />}
            label="Chat"
          />
          <ModeButton
            active={state.mode === 'server'}
            onClick={() => {
              setMode('server')
              if (state.serverPanel === 'about') dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })
            }}
            icon={<Server className="h-3 w-3" />}
            label="Server"
          />
          <ModeButton
            active={state.mode === 'tools'}
            onClick={() => setMode('tools')}
            icon={<Wrench className="h-3 w-3" />}
            label="Tools"
          />
          <ModeButton
            active={state.mode === 'image'}
            onClick={() => setMode('image')}
            icon={<ImageIcon className="h-3 w-3" />}
            label="Image"
          />
          <ModeButton
            active={state.mode === 'api'}
            onClick={() => setMode('api')}
            icon={<Code2 className="h-3 w-3" />}
            label="API"
          />
        </div>
      </div>

      {/* Right: about + theme toggle */}
      <div
        className="flex items-center gap-1 px-3"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <ThemeToggle />
        <button
          onClick={() => {
            dispatch({ type: 'SET_MODE', mode: 'server' })
            dispatch({ type: 'SET_SERVER_PANEL', panel: 'about' })
          }}
          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
          title="About & Settings"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </div>
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
      className={`flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded transition-all ${
        active
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}
