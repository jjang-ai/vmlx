import type { AppMode } from '../types/app-state'

export type ConsolePage = 'chat' | 'server' | 'models'

// Keep persisted modes and existing navigation events compatible. The new
// primary pages group the existing controllers instead of rewriting history.
export function consolePageForMode(mode: AppMode): ConsolePage {
  if (mode === 'image' || mode === 'chat' || mode === 'code') return 'chat'
  if (mode === 'tools' || mode === 'models') return 'models'
  return 'server'
}

export function restoreAppMode(value: string | null | undefined): AppMode {
  // The removed Code page was a placeholder, not a conversation store.
  if (value === 'code') return 'chat'
  return ['chat', 'image', 'server', 'api', 'tools', 'models'].includes(value ?? '')
    ? value as AppMode
    : 'chat'
}
