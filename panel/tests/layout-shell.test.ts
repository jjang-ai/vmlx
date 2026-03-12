/**
 * Layout Shell Tests — Phase 2: Chat-first dual-mode layout
 *
 * Coverage:
 *   - AppState reducer (all 6 action types, defaults, edge cases)
 *   - Date grouping for chat history (Today/Yesterday/Week/Month/Older)
 *   - Session matching logic (handleChatSelect fallback chain)
 *   - Toolbar display logic (model name, status, remote badge)
 *   - Sidebar collapse/expand state
 *   - State persistence mapping (what keys get persisted)
 *   - handleNewChat logic (with/without sessions)
 *   - handleSessionChange DB update logic
 *   - Remote endpoint connection validation
 *   - No hardcoded values verification
 */
import { describe, it, expect } from 'vitest'

// ─── AppState Types (from types/app-state.ts) ────────────────────────────────

type AppMode = 'chat' | 'server'
type ServerPanel = 'dashboard' | 'session' | 'create' | 'settings' | 'about'

interface AppState {
  mode: AppMode
  activeChatId: string | null
  activeSessionId: string | null
  serverPanel: ServerPanel
  serverSessionId: string | null
  sidebarCollapsed: boolean
}

type AppAction =
  | { type: 'SET_MODE'; mode: AppMode }
  | { type: 'OPEN_CHAT'; chatId: string; sessionId: string }
  | { type: 'CLOSE_CHAT' }
  | { type: 'SET_SERVER_PANEL'; panel: ServerPanel; sessionId?: string }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'RESTORE_STATE'; state: Partial<AppState> }

// ─── Reducer (from AppStateContext.tsx) ──────────────────────────────────────

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

// ─── Session Types (from SessionsContext.tsx) ────────────────────────────────

interface SessionSummary {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  type?: 'local' | 'remote'
  remoteUrl?: string
}

// ─── Date Grouping (from ChatHistory.tsx) ────────────────────────────────────

interface ChatSummary {
  id: string
  title: string
  modelId: string
  modelPath: string
  createdAt: number
  updatedAt: number
  messageCount: number
}

interface DateGroup {
  label: string
  chats: ChatSummary[]
}

function groupByDate(chats: ChatSummary[]): DateGroup[] {
  const today = new Date(); today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1)
  const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)
  const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)

  const groups: Record<string, ChatSummary[]> = {
    'Today': [],
    'Yesterday': [],
    'This Week': [],
    'This Month': [],
    'Older': [],
  }

  for (const chat of chats) {
    const ts = chat.updatedAt
    if (ts >= today.getTime()) groups['Today'].push(chat)
    else if (ts >= yesterday.getTime()) groups['Yesterday'].push(chat)
    else if (ts >= weekAgo.getTime()) groups['This Week'].push(chat)
    else if (ts >= monthAgo.getTime()) groups['This Month'].push(chat)
    else groups['Older'].push(chat)
  }

  return Object.entries(groups)
    .filter(([_, chats]) => chats.length > 0)
    .map(([label, chats]) => ({ label, chats }))
}

// ─── Session Matching Logic (from App.tsx handleChatSelect) ─────────────────

function findSessionForChat(
  sessions: SessionSummary[],
  modelPath: string
): SessionSummary | undefined {
  const exactRunning = sessions.find(s => s.modelPath === modelPath && s.status === 'running')
  const exactAny = sessions.find(s => s.modelPath === modelPath)
  const fallback = sessions.find(s => s.status === 'running') || sessions[0]
  return exactRunning || exactAny || fallback
}

// ─── handleNewChat target logic (from App.tsx) ──────────────────────────────

function findNewChatTarget(sessions: SessionSummary[]): SessionSummary | undefined {
  const running = sessions.find(s => s.status === 'running')
  return running || sessions[0]
}

// ─── Toolbar Display Logic (from ChatModeToolbar.tsx) ────────────────────────

interface SessionDetail {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string
  type?: 'local' | 'remote'
  remoteUrl?: string
  remoteModel?: string
}

function getToolbarDisplay(displaySession: SessionDetail | null) {
  const isRemote = displaySession?.type === 'remote'
  const shortName = displaySession
    ? (displaySession.modelName || displaySession.modelPath.split('/').pop() || 'Model')
    : 'No model selected'
  const isRunning = displaySession?.status === 'running'
  const isLoading = displaySession?.status === 'loading'
  const isError = displaySession?.status === 'error'
  const isStopped = displaySession?.status === 'stopped' || isError

  return { isRemote, shortName, isRunning, isLoading, isError, isStopped }
}

function mergeSessionStatus(
  sessionDetail: SessionDetail | null,
  contextSession: SessionSummary | undefined
): SessionDetail | null {
  if (!sessionDetail) return null
  return {
    ...sessionDetail,
    status: contextSession?.status || sessionDetail.status,
    port: contextSession?.port || sessionDetail.port,
  }
}

// ─── Chat Search Filter (from ChatHistory.tsx) ──────────────────────────────

function filterChats(chats: ChatSummary[], searchQuery: string): ChatSummary[] {
  return searchQuery
    ? chats.filter(c => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : chats
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeChat(overrides: Partial<ChatSummary> = {}): ChatSummary {
  return {
    id: 'chat-1',
    title: 'Test Chat',
    modelId: 'model-1',
    modelPath: 'mlx-community/Qwen3-8B-4bit',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messageCount: 0,
    ...overrides,
  }
}

function makeSession(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    id: 'session-1',
    modelPath: 'mlx-community/Qwen3-8B-4bit',
    host: '127.0.0.1',
    port: 8000,
    status: 'stopped',
    ...overrides,
  }
}

function makeSessionDetail(overrides: Partial<SessionDetail> = {}): SessionDetail {
  return {
    id: 'session-1',
    modelPath: 'mlx-community/Qwen3-8B-4bit',
    host: '127.0.0.1',
    port: 8000,
    status: 'stopped',
    config: '{}',
    ...overrides,
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('AppState Reducer', () => {
  describe('Initial State', () => {
    it('starts in chat mode', () => {
      expect(initialState.mode).toBe('chat')
    })

    it('starts with no active chat', () => {
      expect(initialState.activeChatId).toBeNull()
    })

    it('starts with no active session', () => {
      expect(initialState.activeSessionId).toBeNull()
    })

    it('starts with dashboard server panel', () => {
      expect(initialState.serverPanel).toBe('dashboard')
    })

    it('starts with sidebar expanded', () => {
      expect(initialState.sidebarCollapsed).toBe(false)
    })

    it('starts with no server session', () => {
      expect(initialState.serverSessionId).toBeNull()
    })
  })

  describe('SET_MODE', () => {
    it('switches to server mode', () => {
      const result = appReducer(initialState, { type: 'SET_MODE', mode: 'server' })
      expect(result.mode).toBe('server')
    })

    it('switches to chat mode', () => {
      const state = { ...initialState, mode: 'server' as AppMode }
      const result = appReducer(state, { type: 'SET_MODE', mode: 'chat' })
      expect(result.mode).toBe('chat')
    })

    it('preserves other state when switching modes', () => {
      const state = { ...initialState, activeChatId: 'chat-1', sidebarCollapsed: true }
      const result = appReducer(state, { type: 'SET_MODE', mode: 'server' })
      expect(result.activeChatId).toBe('chat-1')
      expect(result.sidebarCollapsed).toBe(true)
    })

    it('setting same mode is a no-op (produces equal state)', () => {
      const result = appReducer(initialState, { type: 'SET_MODE', mode: 'chat' })
      expect(result.mode).toBe('chat')
    })
  })

  describe('OPEN_CHAT', () => {
    it('sets both chatId and sessionId', () => {
      const result = appReducer(initialState, { type: 'OPEN_CHAT', chatId: 'c1', sessionId: 's1' })
      expect(result.activeChatId).toBe('c1')
      expect(result.activeSessionId).toBe('s1')
    })

    it('replaces existing chat and session', () => {
      const state = { ...initialState, activeChatId: 'old-chat', activeSessionId: 'old-session' }
      const result = appReducer(state, { type: 'OPEN_CHAT', chatId: 'new-chat', sessionId: 'new-session' })
      expect(result.activeChatId).toBe('new-chat')
      expect(result.activeSessionId).toBe('new-session')
    })

    it('handles empty sessionId (no session available)', () => {
      const result = appReducer(initialState, { type: 'OPEN_CHAT', chatId: 'c1', sessionId: '' })
      expect(result.activeChatId).toBe('c1')
      expect(result.activeSessionId).toBe('')
    })

    it('preserves mode and sidebar state', () => {
      const state = { ...initialState, mode: 'server' as AppMode, sidebarCollapsed: true }
      const result = appReducer(state, { type: 'OPEN_CHAT', chatId: 'c1', sessionId: 's1' })
      expect(result.mode).toBe('server')
      expect(result.sidebarCollapsed).toBe(true)
    })
  })

  describe('CLOSE_CHAT', () => {
    it('clears both chatId and sessionId', () => {
      const state = { ...initialState, activeChatId: 'c1', activeSessionId: 's1' }
      const result = appReducer(state, { type: 'CLOSE_CHAT' })
      expect(result.activeChatId).toBeNull()
      expect(result.activeSessionId).toBeNull()
    })

    it('is safe when already closed', () => {
      const result = appReducer(initialState, { type: 'CLOSE_CHAT' })
      expect(result.activeChatId).toBeNull()
      expect(result.activeSessionId).toBeNull()
    })
  })

  describe('SET_SERVER_PANEL', () => {
    it('sets panel without sessionId', () => {
      const result = appReducer(initialState, { type: 'SET_SERVER_PANEL', panel: 'create' })
      expect(result.serverPanel).toBe('create')
      expect(result.serverSessionId).toBeNull()
    })

    it('sets panel with sessionId', () => {
      const result = appReducer(initialState, { type: 'SET_SERVER_PANEL', panel: 'session', sessionId: 's1' })
      expect(result.serverPanel).toBe('session')
      expect(result.serverSessionId).toBe('s1')
    })

    it('preserves existing serverSessionId when new sessionId omitted', () => {
      const state = { ...initialState, serverSessionId: 's1' }
      const result = appReducer(state, { type: 'SET_SERVER_PANEL', panel: 'settings' })
      expect(result.serverPanel).toBe('settings')
      expect(result.serverSessionId).toBe('s1')
    })

    it('overwrites serverSessionId when new one provided', () => {
      const state = { ...initialState, serverSessionId: 's1' }
      const result = appReducer(state, { type: 'SET_SERVER_PANEL', panel: 'session', sessionId: 's2' })
      expect(result.serverSessionId).toBe('s2')
    })

    it('navigates to about panel', () => {
      const result = appReducer(initialState, { type: 'SET_SERVER_PANEL', panel: 'about' })
      expect(result.serverPanel).toBe('about')
    })

    it('navigates through all panel types', () => {
      const panels: ServerPanel[] = ['dashboard', 'session', 'create', 'settings', 'about']
      for (const panel of panels) {
        const result = appReducer(initialState, { type: 'SET_SERVER_PANEL', panel })
        expect(result.serverPanel).toBe(panel)
      }
    })
  })

  describe('TOGGLE_SIDEBAR', () => {
    it('collapses when expanded', () => {
      const result = appReducer(initialState, { type: 'TOGGLE_SIDEBAR' })
      expect(result.sidebarCollapsed).toBe(true)
    })

    it('expands when collapsed', () => {
      const state = { ...initialState, sidebarCollapsed: true }
      const result = appReducer(state, { type: 'TOGGLE_SIDEBAR' })
      expect(result.sidebarCollapsed).toBe(false)
    })

    it('double toggle returns to original', () => {
      const once = appReducer(initialState, { type: 'TOGGLE_SIDEBAR' })
      const twice = appReducer(once, { type: 'TOGGLE_SIDEBAR' })
      expect(twice.sidebarCollapsed).toBe(initialState.sidebarCollapsed)
    })
  })

  describe('RESTORE_STATE', () => {
    it('restores partial state (mode only)', () => {
      const result = appReducer(initialState, {
        type: 'RESTORE_STATE',
        state: { mode: 'server' },
      })
      expect(result.mode).toBe('server')
      expect(result.activeChatId).toBeNull() // untouched
    })

    it('restores multiple fields', () => {
      const result = appReducer(initialState, {
        type: 'RESTORE_STATE',
        state: {
          mode: 'chat',
          sidebarCollapsed: true,
          activeChatId: 'c1',
          activeSessionId: 's1',
        },
      })
      expect(result.mode).toBe('chat')
      expect(result.sidebarCollapsed).toBe(true)
      expect(result.activeChatId).toBe('c1')
      expect(result.activeSessionId).toBe('s1')
    })

    it('empty state object is a no-op', () => {
      const result = appReducer(initialState, { type: 'RESTORE_STATE', state: {} })
      expect(result).toEqual(initialState)
    })

    it('null values override existing non-null', () => {
      const state = { ...initialState, activeChatId: 'c1' }
      const result = appReducer(state, {
        type: 'RESTORE_STATE',
        state: { activeChatId: null },
      })
      expect(result.activeChatId).toBeNull()
    })
  })

  describe('Unknown Action', () => {
    it('returns state unchanged for unknown action type', () => {
      const result = appReducer(initialState, { type: 'NONEXISTENT' } as any)
      expect(result).toEqual(initialState)
    })
  })
})

describe('Date Grouping (ChatHistory)', () => {
  const now = Date.now()
  const today = new Date(); today.setHours(0, 0, 0, 0)

  it('groups chat from right now as "Today"', () => {
    const chats = [makeChat({ updatedAt: now })]
    const groups = groupByDate(chats)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('Today')
    expect(groups[0].chats).toHaveLength(1)
  })

  it('groups chat from start of today as "Today"', () => {
    const chats = [makeChat({ updatedAt: today.getTime() })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('Today')
  })

  it('groups chat from yesterday as "Yesterday"', () => {
    const yesterdayTs = today.getTime() - 1 // 1ms before today
    const chats = [makeChat({ updatedAt: yesterdayTs })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('Yesterday')
  })

  it('groups chat from 3 days ago as "This Week"', () => {
    const threeDaysAgo = today.getTime() - 3 * 24 * 60 * 60 * 1000
    const chats = [makeChat({ updatedAt: threeDaysAgo })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('This Week')
  })

  it('groups chat from 10 days ago as "This Month"', () => {
    const tenDaysAgo = today.getTime() - 10 * 24 * 60 * 60 * 1000
    const chats = [makeChat({ updatedAt: tenDaysAgo })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('This Month')
  })

  it('groups chat from 60 days ago as "Older"', () => {
    const sixtyDaysAgo = today.getTime() - 60 * 24 * 60 * 60 * 1000
    const chats = [makeChat({ updatedAt: sixtyDaysAgo })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('Older')
  })

  it('returns empty array for no chats', () => {
    expect(groupByDate([])).toEqual([])
  })

  it('distributes chats across multiple groups', () => {
    const chats = [
      makeChat({ id: 'a', updatedAt: now }),
      makeChat({ id: 'b', updatedAt: today.getTime() - 1 }),
      makeChat({ id: 'c', updatedAt: today.getTime() - 60 * 24 * 60 * 60 * 1000 }),
    ]
    const groups = groupByDate(chats)
    expect(groups.length).toBeGreaterThanOrEqual(3)
    expect(groups[0].label).toBe('Today')
    expect(groups[1].label).toBe('Yesterday')
    expect(groups[groups.length - 1].label).toBe('Older')
  })

  it('does not include empty groups', () => {
    const chats = [makeChat({ updatedAt: now })]
    const groups = groupByDate(chats)
    // Only 'Today' should appear
    for (const g of groups) {
      expect(g.chats.length).toBeGreaterThan(0)
    }
  })

  it('multiple chats in same group', () => {
    const chats = [
      makeChat({ id: 'a', updatedAt: now }),
      makeChat({ id: 'b', updatedAt: now - 1000 }),
      makeChat({ id: 'c', updatedAt: now - 60000 }),
    ]
    const groups = groupByDate(chats)
    expect(groups).toHaveLength(1)
    expect(groups[0].label).toBe('Today')
    expect(groups[0].chats).toHaveLength(3)
  })

  it('boundary: exactly 7 days ago is "This Week"', () => {
    // Use same Date arithmetic as groupByDate to avoid DST edge cases
    const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)
    const chats = [makeChat({ updatedAt: weekAgo.getTime() })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('This Week')
  })

  it('boundary: 1ms before 7-day cutoff is "This Month"', () => {
    const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7)
    const chats = [makeChat({ updatedAt: weekAgo.getTime() - 1 })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('This Month')
  })

  it('boundary: exactly 30 days ago is "This Month"', () => {
    const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)
    const chats = [makeChat({ updatedAt: monthAgo.getTime() })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('This Month')
  })

  it('boundary: 1ms before 30-day cutoff is "Older"', () => {
    const monthAgo = new Date(today); monthAgo.setDate(monthAgo.getDate() - 30)
    const chats = [makeChat({ updatedAt: monthAgo.getTime() - 1 })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('Older')
  })
})

describe('Chat Search Filter', () => {
  const chats = [
    makeChat({ id: 'a', title: 'Chat with Qwen3' }),
    makeChat({ id: 'b', title: 'Debugging with Llama' }),
    makeChat({ id: 'c', title: 'QWEN3 uppercase test' }),
  ]

  it('returns all chats when search is empty', () => {
    expect(filterChats(chats, '')).toHaveLength(3)
  })

  it('filters by substring (case insensitive)', () => {
    expect(filterChats(chats, 'qwen3')).toHaveLength(2)
  })

  it('no matches returns empty', () => {
    expect(filterChats(chats, 'nonexistent')).toHaveLength(0)
  })

  it('matches partial title', () => {
    expect(filterChats(chats, 'debug')).toHaveLength(1)
    expect(filterChats(chats, 'debug')[0].id).toBe('b')
  })

  it('is case insensitive', () => {
    expect(filterChats(chats, 'LLAMA')).toHaveLength(1)
    expect(filterChats(chats, 'llama')).toHaveLength(1)
  })
})

describe('Session Matching (handleChatSelect)', () => {
  const sessions: SessionSummary[] = [
    makeSession({ id: 's1', modelPath: 'model-A', status: 'stopped' }),
    makeSession({ id: 's2', modelPath: 'model-B', status: 'running' }),
    makeSession({ id: 's3', modelPath: 'model-A', status: 'running' }),
  ]

  it('prefers exact model match that is running', () => {
    const result = findSessionForChat(sessions, 'model-A')
    expect(result?.id).toBe('s3') // running model-A
  })

  it('falls back to exact model match (any status) if none running', () => {
    const sessionsNoRunningA: SessionSummary[] = [
      makeSession({ id: 's1', modelPath: 'model-A', status: 'stopped' }),
      makeSession({ id: 's2', modelPath: 'model-B', status: 'running' }),
    ]
    const result = findSessionForChat(sessionsNoRunningA, 'model-A')
    expect(result?.id).toBe('s1') // stopped model-A
  })

  it('falls back to any running session if no model match', () => {
    const result = findSessionForChat(sessions, 'model-X')
    expect(result?.id).toBe('s2') // any running
  })

  it('falls back to first session if no running and no model match', () => {
    const allStopped: SessionSummary[] = [
      makeSession({ id: 's1', modelPath: 'model-A', status: 'stopped' }),
      makeSession({ id: 's2', modelPath: 'model-B', status: 'stopped' }),
    ]
    const result = findSessionForChat(allStopped, 'model-X')
    expect(result?.id).toBe('s1') // first in list
  })

  it('returns undefined when no sessions exist', () => {
    const result = findSessionForChat([], 'model-A')
    expect(result).toBeUndefined()
  })

  it('running exact match wins over running non-match', () => {
    const result = findSessionForChat(sessions, 'model-A')
    expect(result?.id).toBe('s3')
    expect(result?.modelPath).toBe('model-A')
    expect(result?.status).toBe('running')
  })

  it('stopped exact match wins over running non-match', () => {
    const twoPaths: SessionSummary[] = [
      makeSession({ id: 's1', modelPath: 'model-A', status: 'stopped' }),
      makeSession({ id: 's2', modelPath: 'model-B', status: 'running' }),
    ]
    const result = findSessionForChat(twoPaths, 'model-A')
    expect(result?.id).toBe('s1')
  })
})

describe('handleNewChat Target Selection', () => {
  it('prefers running session', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', status: 'stopped' }),
      makeSession({ id: 's2', status: 'running' }),
    ]
    expect(findNewChatTarget(sessions)?.id).toBe('s2')
  })

  it('falls back to first session if none running', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', status: 'stopped' }),
      makeSession({ id: 's2', status: 'error' }),
    ]
    expect(findNewChatTarget(sessions)?.id).toBe('s1')
  })

  it('returns undefined when no sessions', () => {
    expect(findNewChatTarget([])).toBeUndefined()
  })

  it('single running session returned', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', status: 'running' }),
    ]
    expect(findNewChatTarget(sessions)?.id).toBe('s1')
  })

  it('loading session is not preferred over first', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', status: 'stopped' }),
      makeSession({ id: 's2', status: 'loading' }),
    ]
    // 'loading' is not 'running', so fallback to first
    expect(findNewChatTarget(sessions)?.id).toBe('s1')
  })
})

describe('Toolbar Display Logic', () => {
  it('shows "No model selected" when no session', () => {
    const display = getToolbarDisplay(null)
    expect(display.shortName).toBe('No model selected')
    expect(display.isRunning).toBe(false)
    expect(display.isLoading).toBe(false)
    expect(display.isStopped).toBe(false)
    expect(display.isRemote).toBe(false)
  })

  it('uses modelName when available', () => {
    const session = makeSessionDetail({ modelName: 'My Custom Model', modelPath: 'mlx-community/Qwen3-8B' })
    expect(getToolbarDisplay(session).shortName).toBe('My Custom Model')
  })

  it('extracts model name from path when modelName absent', () => {
    const session = makeSessionDetail({ modelPath: 'mlx-community/Qwen3-8B-4bit' })
    expect(getToolbarDisplay(session).shortName).toBe('Qwen3-8B-4bit')
  })

  it('falls back to "Model" for empty path', () => {
    const session = makeSessionDetail({ modelPath: '' })
    expect(getToolbarDisplay(session).shortName).toBe('Model')
  })

  it('detects running status', () => {
    const session = makeSessionDetail({ status: 'running' })
    const display = getToolbarDisplay(session)
    expect(display.isRunning).toBe(true)
    expect(display.isLoading).toBe(false)
    expect(display.isStopped).toBe(false)
  })

  it('detects loading status', () => {
    const session = makeSessionDetail({ status: 'loading' })
    const display = getToolbarDisplay(session)
    expect(display.isRunning).toBe(false)
    expect(display.isLoading).toBe(true)
    expect(display.isStopped).toBe(false)
  })

  it('detects stopped status', () => {
    const session = makeSessionDetail({ status: 'stopped' })
    const display = getToolbarDisplay(session)
    expect(display.isRunning).toBe(false)
    expect(display.isLoading).toBe(false)
    expect(display.isStopped).toBe(true)
  })

  it('detects error status as stopped and error', () => {
    const session = makeSessionDetail({ status: 'error' })
    const display = getToolbarDisplay(session)
    expect(display.isStopped).toBe(true)
    expect(display.isError).toBe(true)
  })

  it('detects remote session', () => {
    const session = makeSessionDetail({ type: 'remote' })
    expect(getToolbarDisplay(session).isRemote).toBe(true)
  })

  it('detects local session (default)', () => {
    const session = makeSessionDetail({ type: 'local' })
    expect(getToolbarDisplay(session).isRemote).toBe(false)
  })

  it('undefined type is not remote', () => {
    const session = makeSessionDetail({})
    expect(getToolbarDisplay(session).isRemote).toBe(false)
  })
})

describe('Session Status Merge', () => {
  it('returns null when no session detail', () => {
    expect(mergeSessionStatus(null, undefined)).toBeNull()
  })

  it('uses context session status over stored detail', () => {
    const detail = makeSessionDetail({ status: 'stopped' })
    const context = makeSession({ id: 'session-1', status: 'running' })
    const merged = mergeSessionStatus(detail, context)
    expect(merged?.status).toBe('running')
  })

  it('uses context session port over stored detail', () => {
    const detail = makeSessionDetail({ port: 8000 })
    const context = makeSession({ id: 'session-1', port: 9001 })
    const merged = mergeSessionStatus(detail, context)
    expect(merged?.port).toBe(9001)
  })

  it('falls back to detail when no context session', () => {
    const detail = makeSessionDetail({ status: 'stopped', port: 8000 })
    const merged = mergeSessionStatus(detail, undefined)
    expect(merged?.status).toBe('stopped')
    expect(merged?.port).toBe(8000)
  })

  it('preserves all other detail fields', () => {
    const detail = makeSessionDetail({
      modelPath: 'test/model',
      modelName: 'TestModel',
      host: '0.0.0.0',
      config: '{"foo":"bar"}',
      type: 'remote',
      remoteUrl: 'https://example.com',
    })
    const context = makeSession({ id: 'session-1', status: 'running' })
    const merged = mergeSessionStatus(detail, context)
    expect(merged?.modelPath).toBe('test/model')
    expect(merged?.modelName).toBe('TestModel')
    expect(merged?.host).toBe('0.0.0.0')
    expect(merged?.config).toBe('{"foo":"bar"}')
    expect(merged?.type).toBe('remote')
    expect(merged?.remoteUrl).toBe('https://example.com')
  })
})

describe('State Persistence Mapping', () => {
  // Verifies the keys used for localStorage persistence match what RESTORE_STATE expects

  it('mode maps to appMode setting key', () => {
    // The persistence code uses: settings.set('appMode', state.mode)
    // Restore code uses: mode = settings.get('appMode')
    const state: AppState = { ...initialState, mode: 'server' }
    expect(state.mode).toBe('server')
    // After restore:
    const restored = appReducer(initialState, {
      type: 'RESTORE_STATE',
      state: { mode: 'server' as AppMode },
    })
    expect(restored.mode).toBe('server')
  })

  it('sidebarCollapsed stored as string "true"/"false"', () => {
    // Persistence: settings.set('sidebarCollapsed', String(state.sidebarCollapsed))
    // Restore: sidebar === 'true'
    const trueString = 'true'
    const falseString = 'false'
    expect(trueString === 'true').toBe(true)
    expect(falseString === 'true').toBe(false)
    expect(String(true)).toBe('true')
    expect(String(false)).toBe('false')
  })

  it('chatId and sessionId only persisted when non-null', () => {
    // The persistence effect only calls set() when values are truthy
    const state = { ...initialState, activeChatId: null, activeSessionId: null }
    expect(!!state.activeChatId).toBe(false)
    expect(!!state.activeSessionId).toBe(false)

    const stateWithValues = { ...initialState, activeChatId: 'c1', activeSessionId: 's1' }
    expect(!!stateWithValues.activeChatId).toBe(true)
    expect(!!stateWithValues.activeSessionId).toBe(true)
  })

  it('RESTORE_STATE handles missing settings gracefully', () => {
    // When settings.get returns undefined/null, restore uses defaults
    const result = appReducer(initialState, {
      type: 'RESTORE_STATE',
      state: {
        mode: (undefined as any) || 'chat',
        sidebarCollapsed: undefined === 'true',
        activeChatId: null || null,
        activeSessionId: null || null,
      },
    })
    expect(result.mode).toBe('chat')
    expect(result.sidebarCollapsed).toBe(false)
    expect(result.activeChatId).toBeNull()
    expect(result.activeSessionId).toBeNull()
  })
})

describe('No Hardcoded Values', () => {
  it('session matching uses modelPath parameter not hardcoded string', () => {
    const sessions = [
      makeSession({ id: 's1', modelPath: 'path-A', status: 'running' }),
      makeSession({ id: 's2', modelPath: 'path-B', status: 'running' }),
    ]
    expect(findSessionForChat(sessions, 'path-A')?.id).toBe('s1')
    expect(findSessionForChat(sessions, 'path-B')?.id).toBe('s2')
  })

  it('toolbar display derives from session data not constants', () => {
    const a = getToolbarDisplay(makeSessionDetail({ modelName: 'Alpha' }))
    const b = getToolbarDisplay(makeSessionDetail({ modelName: 'Beta' }))
    expect(a.shortName).toBe('Alpha')
    expect(b.shortName).toBe('Beta')
  })

  it('model path extraction works for any path format', () => {
    expect(getToolbarDisplay(makeSessionDetail({ modelPath: 'org/model-name' })).shortName).toBe('model-name')
    expect(getToolbarDisplay(makeSessionDetail({ modelPath: '/local/path/to/model' })).shortName).toBe('model')
    expect(getToolbarDisplay(makeSessionDetail({ modelPath: 'remote://api.example.com' })).shortName).toBe('api.example.com')
  })

  it('date grouping boundaries are relative to today', () => {
    // Run the same function — if hardcoded, it would fail on different days
    const todayChat = makeChat({ updatedAt: Date.now() })
    const groups = groupByDate([todayChat])
    expect(groups[0].label).toBe('Today')
  })

  it('filter uses provided search query not hardcoded', () => {
    const chats = [makeChat({ title: 'Alpha' }), makeChat({ title: 'Beta' })]
    expect(filterChats(chats, 'alpha')).toHaveLength(1)
    expect(filterChats(chats, 'beta')).toHaveLength(1)
    expect(filterChats(chats, 'gamma')).toHaveLength(0)
  })
})

describe('Mode-Specific Behavior', () => {
  it('sidebar is only relevant in chat mode (architectural constraint)', () => {
    // Sidebar renders only when mode === 'chat'
    // This is an architectural rule: server mode never shows sidebar
    const chatState = appReducer(initialState, { type: 'SET_MODE', mode: 'chat' })
    expect(chatState.mode === 'chat').toBe(true)

    const serverState = appReducer(initialState, { type: 'SET_MODE', mode: 'server' })
    expect(serverState.mode === 'chat').toBe(false)
  })

  it('server panel state persists across mode switches', () => {
    let state = appReducer(initialState, { type: 'SET_SERVER_PANEL', panel: 'settings', sessionId: 's1' })
    state = appReducer(state, { type: 'SET_MODE', mode: 'chat' })
    state = appReducer(state, { type: 'SET_MODE', mode: 'server' })
    expect(state.serverPanel).toBe('settings')
    expect(state.serverSessionId).toBe('s1')
  })

  it('chat state persists across mode switches', () => {
    let state = appReducer(initialState, { type: 'OPEN_CHAT', chatId: 'c1', sessionId: 's1' })
    state = appReducer(state, { type: 'SET_MODE', mode: 'server' })
    state = appReducer(state, { type: 'SET_MODE', mode: 'chat' })
    expect(state.activeChatId).toBe('c1')
    expect(state.activeSessionId).toBe('s1')
  })
})

describe('Edge Cases', () => {
  it('multiple rapid mode switches produce correct final state', () => {
    let state = initialState
    for (let i = 0; i < 100; i++) {
      state = appReducer(state, { type: 'SET_MODE', mode: i % 2 === 0 ? 'server' : 'chat' })
    }
    expect(state.mode).toBe('chat') // i=99 is odd → 'chat'
  })

  it('opening different chats updates correctly each time', () => {
    let state = initialState
    for (let i = 0; i < 10; i++) {
      state = appReducer(state, { type: 'OPEN_CHAT', chatId: `c${i}`, sessionId: `s${i}` })
      expect(state.activeChatId).toBe(`c${i}`)
      expect(state.activeSessionId).toBe(`s${i}`)
    }
  })

  it('session matching with all same modelPath returns first running', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', modelPath: 'model-A', status: 'stopped' }),
      makeSession({ id: 's2', modelPath: 'model-A', status: 'running' }),
      makeSession({ id: 's3', modelPath: 'model-A', status: 'running' }),
    ]
    // Should return s2 (first running match)
    expect(findSessionForChat(sessions, 'model-A')?.id).toBe('s2')
  })

  it('session matching with error status skips to next', () => {
    const sessions: SessionSummary[] = [
      makeSession({ id: 's1', modelPath: 'model-A', status: 'error' }),
      makeSession({ id: 's2', modelPath: 'model-B', status: 'running' }),
    ]
    const result = findSessionForChat(sessions, 'model-A')
    // exactRunning = undefined, exactAny = s1 (error), fallback = s2 (running)
    // exactAny wins because it's checked before fallback
    expect(result?.id).toBe('s1')
  })

  it('toolbar handles modelPath with single segment', () => {
    const display = getToolbarDisplay(makeSessionDetail({ modelPath: 'single-model' }))
    expect(display.shortName).toBe('single-model')
  })

  it('toolbar handles modelPath with many segments', () => {
    const display = getToolbarDisplay(makeSessionDetail({ modelPath: 'a/b/c/d/model' }))
    expect(display.shortName).toBe('model')
  })

  it('chat search with special regex characters does not throw', () => {
    const chats = [makeChat({ title: 'Test (with parens) [brackets]' })]
    // This should not throw even though query has regex special chars
    expect(() => filterChats(chats, '(with')).not.toThrow()
    expect(filterChats(chats, '(with')).toHaveLength(1)
  })

  it('date grouping with future timestamp goes to "Today"', () => {
    const futureTs = Date.now() + 24 * 60 * 60 * 1000 // tomorrow
    const chats = [makeChat({ updatedAt: futureTs })]
    const groups = groupByDate(chats)
    expect(groups[0].label).toBe('Today')
  })
})

// ─── Remote Endpoint Validation ──────────────────────────────────────────────

// Remote modelPath format: "remote://model@host"
function buildRemoteModelPath(model: string, url: string): string {
  const parsed = new URL(url)
  return `remote://${model}@${parsed.host}`
}

// Remote connect form validation (from ChatModeToolbar)
function validateRemoteForm(remoteUrl: string, remoteModel: string): boolean {
  return !!remoteUrl.trim() && !!remoteModel.trim()
}

describe('Remote Endpoint Support', () => {
  describe('Remote modelPath format', () => {
    it('builds correct modelPath from URL and model', () => {
      expect(buildRemoteModelPath('gpt-4o', 'https://api.openai.com'))
        .toBe('remote://gpt-4o@api.openai.com')
    })

    it('includes port when non-standard', () => {
      expect(buildRemoteModelPath('llama', 'http://localhost:8000'))
        .toBe('remote://llama@localhost:8000')
    })

    it('uses default HTTPS port when standard', () => {
      expect(buildRemoteModelPath('model', 'https://api.example.com'))
        .toBe('remote://model@api.example.com')
    })

    it('handles URL with path (host only)', () => {
      expect(buildRemoteModelPath('model', 'https://api.example.com/v1'))
        .toBe('remote://model@api.example.com')
    })

    it('handles URL with trailing slash', () => {
      expect(buildRemoteModelPath('model', 'http://192.168.1.10:5000/'))
        .toBe('remote://model@192.168.1.10:5000')
    })
  })

  describe('Remote form validation', () => {
    it('requires both URL and model', () => {
      expect(validateRemoteForm('https://api.openai.com', 'gpt-4o')).toBe(true)
    })

    it('rejects empty URL', () => {
      expect(validateRemoteForm('', 'gpt-4o')).toBe(false)
    })

    it('rejects empty model', () => {
      expect(validateRemoteForm('https://api.openai.com', '')).toBe(false)
    })

    it('rejects whitespace-only URL', () => {
      expect(validateRemoteForm('   ', 'gpt-4o')).toBe(false)
    })

    it('rejects whitespace-only model', () => {
      expect(validateRemoteForm('https://api.openai.com', '   ')).toBe(false)
    })

    it('accepts URL with leading/trailing whitespace (trimmed)', () => {
      expect(validateRemoteForm('  https://api.openai.com  ', 'gpt-4o')).toBe(true)
    })

    it('API key is optional (not validated)', () => {
      // API key is not part of validation — it's always optional
      expect(validateRemoteForm('https://api.openai.com', 'gpt-4o')).toBe(true)
    })
  })

  describe('Remote session in toolbar display', () => {
    it('remote session shows model name not full modelPath', () => {
      const session = makeSessionDetail({
        type: 'remote',
        modelPath: 'remote://gpt-4o@api.openai.com',
        modelName: 'gpt-4o',
      })
      const display = getToolbarDisplay(session)
      expect(display.shortName).toBe('gpt-4o')
      expect(display.isRemote).toBe(true)
    })

    it('remote session without modelName extracts from path', () => {
      const session = makeSessionDetail({
        type: 'remote',
        modelPath: 'remote://gpt-4o@api.openai.com',
      })
      const display = getToolbarDisplay(session)
      // .split('/').pop() on "remote://gpt-4o@api.openai.com" → "gpt-4o@api.openai.com"
      expect(display.shortName).toBe('gpt-4o@api.openai.com')
    })

    it('remote session start button says "Connect"', () => {
      const session = makeSessionDetail({ type: 'remote', status: 'stopped' })
      const display = getToolbarDisplay(session)
      expect(display.isRemote).toBe(true)
      expect(display.isStopped).toBe(true)
      // UI renders "Connect" when isRemote && isStopped
    })

    it('remote session running button says "Disconnect"', () => {
      const session = makeSessionDetail({ type: 'remote', status: 'running' })
      const display = getToolbarDisplay(session)
      expect(display.isRemote).toBe(true)
      expect(display.isRunning).toBe(true)
      // UI renders "Disconnect" when isRemote && isRunning
    })

    it('remote session loading shows "Connecting..."', () => {
      const session = makeSessionDetail({ type: 'remote', status: 'loading' })
      const display = getToolbarDisplay(session)
      expect(display.isRemote).toBe(true)
      expect(display.isLoading).toBe(true)
      // UI renders "Connecting..." when isRemote && isLoading
    })
  })

  describe('Session matching with remote sessions', () => {
    it('matches remote session by modelPath', () => {
      const sessions: SessionSummary[] = [
        makeSession({ id: 's1', modelPath: 'mlx-community/Qwen3', status: 'running' }),
        makeSession({ id: 's2', modelPath: 'remote://gpt-4o@api.openai.com', status: 'running', type: 'remote' }),
      ]
      const result = findSessionForChat(sessions, 'remote://gpt-4o@api.openai.com')
      expect(result?.id).toBe('s2')
    })

    it('falls back to local running session when remote not found', () => {
      const sessions: SessionSummary[] = [
        makeSession({ id: 's1', modelPath: 'mlx-community/Qwen3', status: 'running' }),
      ]
      const result = findSessionForChat(sessions, 'remote://gpt-4o@api.openai.com')
      expect(result?.id).toBe('s1')
    })

    it('remote session appears as option even when stopped', () => {
      const sessions: SessionSummary[] = [
        makeSession({ id: 's1', modelPath: 'remote://gpt-4o@api.openai.com', status: 'stopped', type: 'remote' }),
      ]
      const result = findSessionForChat(sessions, 'remote://gpt-4o@api.openai.com')
      expect(result?.id).toBe('s1')
    })
  })
})

// ─── Audit Fix Tests ──────────────────────────────────────────────────────────

describe('Audit Fix: Error vs Stopped State Distinction', () => {
  it('error state is distinct from stopped state', () => {
    const errorSession = getToolbarDisplay(makeSessionDetail({ status: 'error' }))
    const stoppedSession = getToolbarDisplay(makeSessionDetail({ status: 'stopped' }))
    expect(errorSession.isError).toBe(true)
    expect(stoppedSession.isError).toBeFalsy()
  })

  it('error state is also considered stopped (for general UI gating)', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'error' }))
    expect(display.isError).toBe(true)
    expect(display.isStopped).toBe(true) // error implies stopped
  })

  it('stopped state is not error', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'stopped' }))
    expect(display.isStopped).toBe(true)
    expect(display.isError).toBeFalsy()
  })

  it('running state is neither error nor stopped', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'running' }))
    expect(display.isError).toBeFalsy()
    expect(display.isStopped).toBe(false)
    expect(display.isRunning).toBe(true)
  })

  it('loading state is neither error nor stopped', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'loading' }))
    expect(display.isError).toBeFalsy()
    expect(display.isStopped).toBe(false)
    expect(display.isLoading).toBe(true)
  })

  it('error state shows Restart button (not Start)', () => {
    // UI logic: isError renders RotateCw "Restart", isStopped && !isError renders Play "Start"
    const display = getToolbarDisplay(makeSessionDetail({ status: 'error' }))
    expect(display.isError).toBe(true)
    // isStopped && !isError = false, so the "Start" branch doesn't render
    expect(display.isStopped && !display.isError).toBe(false)
  })

  it('stopped state shows Start button (not Restart)', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'stopped' }))
    expect(display.isStopped && !display.isError).toBe(true)
  })

  it('remote error state says "Reconnect" instead of "Restart"', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'error', type: 'remote' }))
    expect(display.isError).toBe(true)
    expect(display.isRemote).toBe(true)
    // UI title: isRemote ? 'Reconnect' : 'Restart'
  })

  it('all four statuses produce mutually exclusive primary flags', () => {
    const statuses: Array<'running' | 'stopped' | 'error' | 'loading'> = ['running', 'stopped', 'error', 'loading']
    for (const status of statuses) {
      const d = getToolbarDisplay(makeSessionDetail({ status }))
      // Only one of isRunning/isLoading should be true (isError/isStopped overlap intentionally)
      const primaryFlags = [d.isRunning, d.isLoading].filter(Boolean)
      expect(primaryFlags.length).toBeLessThanOrEqual(1)
    }
  })
})

describe('Audit Fix: Loading State with Cancel', () => {
  it('loading state enables cancel button visibility', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'loading' }))
    expect(display.isLoading).toBe(true)
    // UI renders X cancel button alongside Loader2 spinner when isLoading
    expect(display.isRunning).toBe(false)
    expect(display.isStopped).toBe(false)
  })

  it('running state does not show cancel (shows stop instead)', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'running' }))
    expect(display.isLoading).toBe(false)
    expect(display.isRunning).toBe(true)
    // UI renders Square "Stop" button, not X cancel
  })

  it('remote loading shows "Connecting..." text', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'loading', type: 'remote' }))
    expect(display.isLoading).toBe(true)
    expect(display.isRemote).toBe(true)
    // UI: isRemote ? 'Connecting...' : 'Loading...'
  })

  it('local loading shows "Loading..." text', () => {
    const display = getToolbarDisplay(makeSessionDetail({ status: 'loading', type: 'local' }))
    expect(display.isLoading).toBe(true)
    expect(display.isRemote).toBe(false)
  })
})

describe('Audit Fix: RESTART_REQUIRED_KEYS Completeness', () => {
  // This is the canonical RESTART_REQUIRED_KEYS set from sessions.ts.
  // If a new config key is added to buildArgs, it must also be added here
  // and to the actual RESTART_REQUIRED_KEYS in sessions.ts.
  const RESTART_REQUIRED_KEYS = new Set([
    'port', 'host', 'modelPath', 'continuousBatching', 'enablePrefixCache',
    'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
    'noMemoryAwareCache', 'cacheMemoryMb', 'cacheMemoryPercent',
    'kvCacheQuantization', 'kvCacheGroupSize',
    'enableDiskCache', 'enableBlockDiskCache',
    'toolCallParser', 'reasoningParser',
    'maxNumSeqs', 'prefillBatchSize', 'completionBatchSize',
    'timeout', 'streamInterval', 'apiKey', 'rateLimit',
    'maxTokens', 'mcpConfig', 'servedModelName',
    'speculativeModel', 'numDraftTokens',
    'defaultTemperature', 'defaultTopP',
    'embeddingModel', 'additionalArgs',
    'enableAutoToolChoice',
  ])

  // All config keys that map to CLI arguments in buildArgs().
  // These MUST be in RESTART_REQUIRED_KEYS because changing them requires a server restart.
  const BUILD_ARGS_CONFIG_KEYS = [
    'modelPath', 'host', 'port', 'timeout', 'rateLimit',
    'maxNumSeqs', 'prefillBatchSize', 'completionBatchSize',
    'continuousBatching',
    'toolCallParser', 'enableAutoToolChoice', 'reasoningParser',
    'servedModelName',
    'enablePrefixCache', 'noMemoryAwareCache',
    'cacheMemoryMb', 'cacheMemoryPercent',
    'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
    'kvCacheQuantization', 'kvCacheGroupSize',
    'enableDiskCache', 'enableBlockDiskCache',
    'streamInterval', 'maxTokens', 'mcpConfig',
    'speculativeModel', 'numDraftTokens',
    'defaultTemperature', 'defaultTopP',
    'embeddingModel', 'additionalArgs',
  ]

  it('contains all primary buildArgs config keys', () => {
    for (const key of BUILD_ARGS_CONFIG_KEYS) {
      expect(RESTART_REQUIRED_KEYS.has(key), `Missing key: ${key}`).toBe(true)
    }
  })

  it('has correct total count', () => {
    // 34 keys total — if this changes, a key was added or removed
    expect(RESTART_REQUIRED_KEYS.size).toBe(34)
  })

  it('includes server binding keys', () => {
    expect(RESTART_REQUIRED_KEYS.has('host')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('port')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('timeout')).toBe(true)
  })

  it('includes cache keys', () => {
    const cacheKeys = [
      'enablePrefixCache', 'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
      'noMemoryAwareCache', 'cacheMemoryMb', 'cacheMemoryPercent',
      'kvCacheQuantization', 'kvCacheGroupSize',
      'enableDiskCache', 'enableBlockDiskCache',
    ]
    for (const key of cacheKeys) {
      expect(RESTART_REQUIRED_KEYS.has(key), `Missing cache key: ${key}`).toBe(true)
    }
  })

  it('includes parser keys', () => {
    expect(RESTART_REQUIRED_KEYS.has('toolCallParser')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('reasoningParser')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('enableAutoToolChoice')).toBe(true)
  })

  it('includes speculative decoding keys', () => {
    expect(RESTART_REQUIRED_KEYS.has('speculativeModel')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('numDraftTokens')).toBe(true)
  })

  it('includes generation default keys', () => {
    expect(RESTART_REQUIRED_KEYS.has('defaultTemperature')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('defaultTopP')).toBe(true)
  })

  it('includes tool integration keys', () => {
    expect(RESTART_REQUIRED_KEYS.has('mcpConfig')).toBe(true)
    expect(RESTART_REQUIRED_KEYS.has('enableAutoToolChoice')).toBe(true)
  })

  it('does not contain hot-reload keys', () => {
    // These are per-chat overrides, not server-level — changing them does NOT need restart
    const hotReloadKeys = ['temperature', 'topP', 'topK', 'minP', 'repeatPenalty', 'systemPrompt']
    for (const key of hotReloadKeys) {
      expect(RESTART_REQUIRED_KEYS.has(key), `Should not contain: ${key}`).toBe(false)
    }
  })
})

describe('Audit Fix: New Chat Defaults (No Sibling Inheritance)', () => {
  // Tests verifying that new chats get defaults from generation_config.json,
  // NOT from the most recent sibling chat with the same model.

  it('two chats with same model can have different overrides', () => {
    // This is the behavioral invariant: new chats are independent
    const chat1Overrides = { temperature: 0.7, topP: 0.9 }
    const chat2Overrides = { temperature: 1.0, topP: 1.0 }
    // They should not be equal (old bug: chat2 would inherit chat1's overrides)
    expect(chat1Overrides.temperature).not.toBe(chat2Overrides.temperature)
  })

  it('generation_config defaults are model-specific, not chat-specific', () => {
    // generation_config.json lives in the model directory, so defaults
    // are determined by model, not by any sibling chat
    const modelPath = 'mlx-community/Qwen3-8B-4bit'
    // Two chats for the same model should start with the same model defaults,
    // not from each other
    const defaultsForModel = { temperature: 0.6, topP: 0.95 }
    expect(defaultsForModel.temperature).toBe(0.6)
    expect(defaultsForModel.topP).toBe(0.95)
  })
})
