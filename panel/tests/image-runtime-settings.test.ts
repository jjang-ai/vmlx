import { describe, expect, it } from 'vitest'
import { defaultImageRuntimeSettings, loadImageRuntimeSettings, saveImageRuntimeSettings, IMAGE_SETTINGS_KEY } from '../src/shared/imageRuntimeSettings'
function storage(initial: Record<string, string> = {}) {
  const entries = new Map(Object.entries(initial))
  return { entries, getSetting: (k: string) => entries.get(k), setSetting: (k: string, v: string) => { entries.set(k, v) } }
}
const qwen = { sessionId: 'q8-session', modelId: 'qwen-image-edit', quantize: 8 }
describe('image runtime settings ownership', () => {
  it('uses canonical Qwen defaults, not a q8 basename or placeholder4/3.5', () => {
    expect(loadImageRuntimeSettings(storage(), qwen)).toMatchObject({ steps: 28, guidance: 4, quantize: 8 })
  })
  it('retains zero guidance and explicit dimensions without persisting seed or precision', () => {
    const db = storage()
    saveImageRuntimeSettings(db, qwen, { steps: 9, width: 768, height: 512, guidance: 0, seed: 42, quantize: 4 })
    expect(loadImageRuntimeSettings(db, qwen)).toMatchObject({ steps: 9, width: 768, height: 512, guidance: 0, seed: undefined, quantize: 8 })
    expect(db.getSetting(IMAGE_SETTINGS_KEY)).not.toMatch(/seed|quantize/)
  })
  it('does not carry Qwen controls into a different session or changed architecture', () => {
    const db = storage()
    saveImageRuntimeSettings(db, qwen, { steps: 70, guidance: 18 })
    const schnell = { sessionId: 'schnell-session', modelId: 'schnell', quantize: 4 }
    expect(loadImageRuntimeSettings(db, schnell)).toEqual(defaultImageRuntimeSettings('schnell', 4))
    expect(loadImageRuntimeSettings(db, { ...qwen, modelId: 'schnell' })).toEqual(defaultImageRuntimeSettings('schnell', 8))
  })
  it('migrates legacy data once on adoption, preserving the original and exact zeros', () => {
    const legacy = JSON.stringify({ steps: 17, guidance: 0, negativePrompt: 'blur', seed: 99 })
    const db = storage({ image_settings: legacy })
    expect(loadImageRuntimeSettings(db, qwen, true)).toMatchObject({ steps: 17, guidance: 0, negativePrompt: 'blur', seed: undefined })
    expect(loadImageRuntimeSettings(db, { sessionId: 'other', modelId: 'schnell', quantize: 4 }, true).steps).toBe(4)
    expect(db.getSetting('image_settings')).toBe(legacy)
    expect(JSON.parse(db.getSetting(IMAGE_SETTINGS_KEY)!).legacyOwner).toBe(qwen.sessionId)
  })
  it('does not apply ambiguous global preferences to a newly selected model', () => {
    expect(loadImageRuntimeSettings(storage({ image_settings: '{"steps":80}' }), qwen).steps).toBe(28)
  })

  it('does not import legacy preferences on remount after explicit selection with no edits', () => {
    const db = storage({ image_settings: '{"steps":80,"guidance":19}' })
    loadImageRuntimeSettings(db, qwen, false)
    expect(loadImageRuntimeSettings(db, qwen, true).steps).toBe(28)
    expect(loadImageRuntimeSettings(db, qwen, true).guidance).toBe(4)
  })
  it('ignores malformed legacy fields instead of spreading arbitrary keys into settings', () => {
    const db = storage({ image_settings: '{"steps":-1,"guidance":0,"width":"bad","quantize":3,"seed":99}' })
    expect(loadImageRuntimeSettings(db, qwen, true)).toEqual({ ...defaultImageRuntimeSettings(qwen.modelId, 8), guidance: 0 })
  })
  it('preserves unreadable or future-version records without writing over them', () => {
    for (const value of ['bad json', '{"version":3,"sessions":{}}']) {
      const db = storage({ [IMAGE_SETTINGS_KEY]: value })
      expect(() => saveImageRuntimeSettings(db, qwen, { steps: 8 })).toThrow()
      expect(db.getSetting(IMAGE_SETTINGS_KEY)).toBe(value)
    }
  })
  it('publishes legacy owner and preferences in a single setting write', () => {
    const db = storage({ image_settings: '{"steps":12}' })
    const writes: string[] = []
    const original = db.setSetting
    db.setSetting = (k, v) => { writes.push(k); original(k, v) }
    loadImageRuntimeSettings(db, qwen, true)
    expect(writes).toEqual([IMAGE_SETTINGS_KEY])
  })
})
