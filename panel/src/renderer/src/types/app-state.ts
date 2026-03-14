export type AppMode = 'chat' | 'server' | 'tools'

export type ServerPanel = 'dashboard' | 'session' | 'create' | 'settings' | 'about'

export type ToolsPanel = 'dashboard' | 'inspector' | 'doctor' | 'converter'

export interface AppState {
  mode: AppMode
  // Chat mode
  activeChatId: string | null
  activeSessionId: string | null
  // Server mode
  serverPanel: ServerPanel
  serverSessionId: string | null
  serverInitialModelPath: string | null
  // Tools mode
  toolsPanel: ToolsPanel
  toolsModelPath: string | null
  // Layout
  sidebarCollapsed: boolean
}

export type AppAction =
  | { type: 'SET_MODE'; mode: AppMode }
  | { type: 'OPEN_CHAT'; chatId: string; sessionId: string }
  | { type: 'CLOSE_CHAT' }
  | { type: 'SET_SERVER_PANEL'; panel: ServerPanel; sessionId?: string; modelPath?: string }
  | { type: 'SET_TOOLS_PANEL'; panel: ToolsPanel; modelPath?: string | null }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'RESTORE_STATE'; state: Partial<AppState> }
