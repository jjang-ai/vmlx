/**
 * Per-model settings — stored in SQLite, passed as CLI flags when spawning.
 *
 * Settings are keyed by model_path (HuggingFace repo or local path).
 * Each setting maps to a vmlx-engine CLI flag when spawning via ProcessManager.
 */

import { ipcMain } from 'electron'
import { db } from '../database'

export interface ModelSettings {
  model_path: string
  alias?: string
  temperature?: number
  top_p?: number
  max_tokens?: number
  ttl_minutes?: number
  pinned: boolean
  port?: number
  cache_quant?: string   // 'q4' | 'q8' | 'none'
  disk_cache_enabled: boolean
  reasoning_mode: string // 'auto' | 'on' | 'off'
}

/**
 * Convert stored DB row to ModelSettings.
 */
function fromRow(row: Record<string, any>): ModelSettings {
  return {
    model_path: row.model_path,
    alias: row.alias ?? undefined,
    temperature: row.temperature ?? undefined,
    top_p: row.top_p ?? undefined,
    max_tokens: row.max_tokens ?? undefined,
    ttl_minutes: row.ttl_minutes ?? undefined,
    pinned: !!row.pinned,
    port: row.port ?? undefined,
    cache_quant: row.cache_quant ?? undefined,
    disk_cache_enabled: !!row.disk_cache_enabled,
    reasoning_mode: row.reasoning_mode ?? 'auto',
  }
}

/**
 * Register IPC handlers for model settings.
 */
export function registerModelSettingsHandlers(): void {
  ipcMain.handle('model-settings:get', (_e, modelPath: string) => {
    const row = db.getModelSettings(modelPath)
    return row ? fromRow(row) : null
  })

  ipcMain.handle('model-settings:getAll', () => {
    return db.getAllModelSettings().map(fromRow)
  })

  ipcMain.handle('model-settings:save', (_e, modelPath: string, settings: Partial<ModelSettings>) => {
    if (typeof modelPath !== 'string' || !modelPath.trim()) {
      return { success: false, error: 'model_path is required' }
    }

    // Sanitize numeric fields — clamp to valid ranges
    const sanitized: Partial<ModelSettings> = {}
    if (settings.alias !== undefined) sanitized.alias = String(settings.alias).slice(0, 200)
    if (settings.temperature !== undefined) {
      const t = Number(settings.temperature)
      if (!isNaN(t)) sanitized.temperature = Math.max(0, Math.min(2, t))
    }
    if (settings.top_p !== undefined) {
      const p = Number(settings.top_p)
      if (!isNaN(p)) sanitized.top_p = Math.max(0, Math.min(1, p))
    }
    if (settings.max_tokens !== undefined) {
      const m = Math.round(Number(settings.max_tokens))
      if (!isNaN(m) && m > 0) sanitized.max_tokens = Math.min(m, 1_000_000)
    }
    if (settings.ttl_minutes !== undefined) {
      const t = Math.round(Number(settings.ttl_minutes))
      if (!isNaN(t) && t >= 0) sanitized.ttl_minutes = t
    }
    if (settings.pinned !== undefined) sanitized.pinned = !!settings.pinned
    if (settings.port !== undefined) {
      const p = Math.round(Number(settings.port))
      if (!isNaN(p) && p >= 1024 && p <= 65535) sanitized.port = p
    }
    if (settings.cache_quant !== undefined) {
      const valid = ['q4', 'q8', 'none']
      sanitized.cache_quant = valid.includes(settings.cache_quant) ? settings.cache_quant : undefined
    }
    if (settings.disk_cache_enabled !== undefined) sanitized.disk_cache_enabled = !!settings.disk_cache_enabled
    if (settings.reasoning_mode !== undefined) {
      const valid = ['auto', 'on', 'off']
      sanitized.reasoning_mode = valid.includes(settings.reasoning_mode) ? settings.reasoning_mode : 'auto'
    }

    db.saveModelSettings(modelPath, { ...sanitized, model_path: modelPath })
    return { success: true }
  })

  ipcMain.handle('model-settings:delete', (_e, modelPath: string) => {
    db.deleteModelSettings(modelPath)
    return { success: true }
  })
}
