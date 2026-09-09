import { describe, expect, it } from 'vitest'
import { imageRuntimeSnapshot } from '../src/shared/imageCapabilities'

describe('image model readiness, not merely HTTP liveness', () => {
  it('requires loaded image evidence for Running', () => {
    expect(imageRuntimeSnapshot({ status: 'healthy' }).status).not.toBe('running')
    expect(imageRuntimeSnapshot({ status: 'healthy', model_loaded: false, image: { loaded: false } }).status).not.toBe('running')
    expect(imageRuntimeSnapshot({ status: 'healthy', model_loaded: true, model_type: 'text' }).status).not.toBe('running')
    expect(imageRuntimeSnapshot({ status: 'healthy', model_loaded: true, image: { loaded: true, mode: 'edit' } }))
      .toMatchObject({ status: 'running', capabilities: { loaded: true, mode: 'edit' } })
  })
  it('keeps no-model initialization and explicit reload in Starting', () => {
    expect(imageRuntimeSnapshot({ status: 'no_model' }).status).toBe('starting')
    expect(imageRuntimeSnapshot({ status: 'standby_deep', wake_in_progress: true }).status).toBe('starting')
  })
  it('preserves observed loaded capabilities while keeping both sleep depths non-ready', () => {
    for (const status of ['standby_soft', 'standby_deep']) {
      const image = { loaded: true, edit_strength: false, mask: 'none' }
      expect(imageRuntimeSnapshot({ status, model_loaded: true, image }))
        .toEqual({ status: 'standby', capabilities: image })
      for (const body of [
        { status, model_loaded: false, image },
        { status, model_loaded: true, image: { loaded: false } },
        { status },
      ]) expect(imageRuntimeSnapshot(body)).toEqual({ status: 'standby', capabilities: null })
    }
  })
  it('does not promote malformed, failed, or contradictory payloads', () => {
    for (const body of [null, {}, [], { status: 'error' }, { status: 'healthy', model_loaded: false, image: { loaded: true } }]) {
      expect(imageRuntimeSnapshot(body)).toEqual({ status: 'error', capabilities: null })
    }
  })
  it('supports explicit legacy image loaded evidence without inventing capabilities', () => {
    expect(imageRuntimeSnapshot({ status: 'healthy', model_type: 'image', model_loaded: true }))
      .toEqual({ status: 'running', capabilities: null })
  })
})
