import { createContext, useContext, useReducer, useEffect } from 'react'
import type { AppState, AppAction, AppMode } from '../types/app-state'

const initialState: AppState = {
  mode: 'chat',
  activeChatId: null,
  activeSessionId: null,
  serverPanel: 'dashboard',
  serverSessionId: null,
  sidebarCollapsed: false,
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, mode: action.mode }
    case 'OPEN_CHAT':
      return { ...state, activeChatId: action.chatId, activeSessionId: action.sessionId }
    case 'CLOSE_CHAT':
      return { ...state, activeChatId: null, activeSessionId: null }
    case 'SET_SERVER_PANEL':
      return { ...state, serverPanel: action.panel, serverSessionId: action.sessionId ?? state.serverSessionId }
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed }
    case 'RESTORE_STATE':
      return { ...state, ...action.state }
    default:
      return state
  }
}

interface AppStateContextValue {
  state: AppState
  dispatch: React.Dispatch<AppAction>
  setMode: (mode: AppMode) => void
  openChat: (chatId: string, sessionId: string) => void
}

const AppStateContext = createContext<AppStateContextValue>(null!)

export function useAppState() {
  return useContext(AppStateContext)
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState)

  // Restore persisted state on mount
  useEffect(() => {
    const restore = async () => {
      try {
        const mode = await window.api.settings.get('appMode')
        const sidebar = await window.api.settings.get('sidebarCollapsed')
        const lastChat = await window.api.settings.get('lastActiveChatId')
        const lastSession = await window.api.settings.get('lastActiveSessionId')
        dispatch({
          type: 'RESTORE_STATE',
          state: {
            mode: (mode as AppMode) || 'chat',
            sidebarCollapsed: sidebar === 'true',
            activeChatId: lastChat || null,
            activeSessionId: lastSession || null,
          },
        })
      } catch { /* first launch, use defaults */ }
    }
    restore()
  }, [])

  // Persist state changes
  useEffect(() => {
    window.api.settings.set('appMode', state.mode).catch(() => {})
    window.api.settings.set('sidebarCollapsed', String(state.sidebarCollapsed)).catch(() => {})
    if (state.activeChatId) {
      window.api.settings.set('lastActiveChatId', state.activeChatId).catch(() => {})
    }
    if (state.activeSessionId) {
      window.api.settings.set('lastActiveSessionId', state.activeSessionId).catch(() => {})
    }
  }, [state.mode, state.sidebarCollapsed, state.activeChatId, state.activeSessionId])

  const setMode = (mode: AppMode) => dispatch({ type: 'SET_MODE', mode })
  const openChat = (chatId: string, sessionId: string) => dispatch({ type: 'OPEN_CHAT', chatId, sessionId })

  return (
    <AppStateContext.Provider value={{ state, dispatch, setMode, openChat }}>
      {children}
    </AppStateContext.Provider>
  )
}
