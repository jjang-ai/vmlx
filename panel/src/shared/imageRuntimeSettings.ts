import { getDefaultGuidance, getDefaultSteps } from './imageModels'

export interface ImageRuntimeSettings {
  steps: number; width: number; height: number; guidance: number
  negativePrompt: string; seed?: number; count: number; quantize: number; strength: number
}
type Preferences = Omit<ImageRuntimeSettings, 'seed' | 'quantize'>
export interface ImageSettingsOwner { sessionId: string; modelId: string; quantize: number }
interface Store {
  version: 2
  legacyOwner?: string
  sessions: Record<string, { modelId: string; values: Partial<Preferences> }>
}
export interface SettingsStorage {
  getSetting(key: string): string | undefined
  setSetting(key: string, value: string): void
}
export const IMAGE_SETTINGS_KEY = 'image_settings_v2'

export function defaultImageRuntimeSettings(modelId: string, quantize: number): ImageRuntimeSettings {
  return { steps: getDefaultSteps(modelId), guidance: getDefaultGuidance(modelId),
    width: 1024, height: 1024, negativePrompt: '', seed: undefined,
    count: 1, quantize, strength: 0.8 }
}

function preferences(value: unknown): Partial<Preferences> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const raw = value as Record<string, unknown>, result: Partial<Preferences> = {}
  for (const key of ['steps', 'width', 'height', 'count'] as const) {
    const v = raw[key]
    if (typeof v === 'number' && Number.isSafeInteger(v) && v > 0) result[key] = v
  }
  for (const key of ['guidance', 'strength'] as const) {
    const v = raw[key]
    if (typeof v === 'number' && Number.isFinite(v) && v >= 0) result[key] = v
  }
  if (typeof raw.negativePrompt === 'string') result.negativePrompt = raw.negativePrompt
  return result
}

function readStore(storage: SettingsStorage): Store {
  const raw = storage.getSetting(IMAGE_SETTINGS_KEY)
  if (!raw) return { version: 2, sessions: {} }
  const parsed = JSON.parse(raw)
  if (parsed?.version !== 2 || !parsed.sessions || typeof parsed.sessions !== 'object' || Array.isArray(parsed.sessions))
    throw new Error('Image settings record is invalid or uses an unsupported version; saved data was preserved')
  return parsed
}

/** Synchronous main-process read/migrate/write: owner and preferences publish in one DB setting update. */
export function loadImageRuntimeSettings(storage: SettingsStorage, owner: ImageSettingsOwner, adoptLegacy = false): ImageRuntimeSettings {
  const store = readStore(storage)
  const saved = store.sessions[owner.sessionId]
  let values = saved?.modelId === owner.modelId ? preferences(saved.values) : {}
  if (!saved && adoptLegacy && !store.legacyOwner) {
    let legacy: unknown
    try { legacy = JSON.parse(storage.getSetting('image_settings') || 'null') } catch { legacy = null }
    values = preferences(legacy)
    if (Object.keys(values).length) {
      store.legacyOwner = owner.sessionId
      store.sessions[owner.sessionId] = { modelId: owner.modelId, values }
      // Do not delete or rewrite the old global record.
      storage.setSetting(IMAGE_SETTINGS_KEY, JSON.stringify(store))
    }
  }
  return { ...defaultImageRuntimeSettings(owner.modelId, owner.quantize), ...values }
}

export function saveImageRuntimeSettings(storage: SettingsStorage, owner: ImageSettingsOwner, value: unknown): void {
  const store = readStore(storage)
  store.sessions[owner.sessionId] = { modelId: owner.modelId, values: preferences(value) }
  storage.setSetting(IMAGE_SETTINGS_KEY, JSON.stringify(store))
}

