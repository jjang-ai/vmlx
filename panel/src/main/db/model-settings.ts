/**
 * Per-model settings — stored in SQLite for model/session metadata.
 *
 * Settings are keyed by model_path (HuggingFace repo or local path).
 * Sampling and thinking choices live in bundle metadata, per-chat overrides,
 * or explicit API request parameters. They are intentionally not exposed here.
 */

import { ipcMain } from 'electron'
import { db } from '../database'

export interface ModelSettings {
  model_path: string
  alias?: string
  ttl_minutes?: number
  pinned: boolean
  port?: number
  cache_quant?: string   // 'q4' | 'q8' | 'none'
  disk_cache_enabled: boolean
}

/**
 * Convert stored DB row to ModelSettings.
 */
function fromRow(row: Record<string, any>): ModelSettings {
  return {
    model_path: row.model_path,
    alias: row.alias ?? undefined,
    ttl_minutes: row.ttl_minutes ?? undefined,
    pinned: !!row.pinned,
    port: row.port ?? undefined,
    cache_quant: row.cache_quant ?? undefined,
    disk_cache_enabled: !!row.disk_cache_enabled,
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

    // Model settings are launch/session metadata. Sampling and thinking
    // choices are per-chat/API overrides and must not be stored per model.
    const sanitized: Partial<ModelSettings> = {}
    if (settings.alias !== undefined) sanitized.alias = String(settings.alias).slice(0, 200)
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

    db.saveModelSettings(modelPath, { ...sanitized, model_path: modelPath })
    return { success: true }
  })

  ipcMain.handle('model-settings:delete', (_e, modelPath: string) => {
    db.deleteModelSettings(modelPath)
    return { success: true }
  })
}
