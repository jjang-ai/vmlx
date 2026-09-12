import { describe, expect, it } from 'vitest'
import {
  RUNTIME_VIDEO_CAPABLE_FAMILIES,
  isRuntimeVideoCapable,
  modalitiesIncludeVideo,
} from '../src/shared/videoCapableFamilies'
import { normalizeDetectedFamilyName } from '../src/shared/detectedFamilyNames'

// S21 (post-1655 campaign): the drawer hid the video controls on a Muse
// Glimmer session because visibility came from a hard-coded family list that
// never learned "muse-glimmer", while the engine served the session's videos
// and honoured video_token_budget over the API. Visibility now comes from the
// bundle's declared modalities OR the family list, either one sufficing.
describe('video controls visibility (S21)', () => {
  it('uses the exact live report ahead of offline family or bundle guesses', () => {
    expect(isRuntimeVideoCapable({ normalizedFamily: 'glm5-next', liveRuntimeModalities: ['text', 'vision', 'video'] })).toBe(true)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'qwen4-exp', runtimeModalities: ['video'], liveRuntimeModalities: ['text'] })).toBe(false)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'qwen4-exp', liveRuntimeModalities: ['vision'] })).toBe(false)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'qwen4-exp', liveRuntimeModalities: undefined })).toBe(true)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'glm5-next', liveRuntimeModalities: null })).toBe(false)
  })
  it('shows the controls for Muse Glimmer through the engine family spelling', () => {
    const normalized = normalizeDetectedFamilyName('muse_glimmer')
    expect(isRuntimeVideoCapable({ normalizedFamily: normalized })).toBe(true)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'muse-glimmer' })).toBe(true)
  })

  it('keeps every previously listed family', () => {
    for (const family of ['qwen3-vl', 'qwen3.5', 'qwen3.5-moe', 'qwen4-exp', 'qwen4_exp', 'qwen2-vl', 'gemma4', 'nemotron-h', 'mistral3', 'mistral4', 'pixtral', 'kimi-k25']) {
      expect(RUNTIME_VIDEO_CAPABLE_FAMILIES).toContain(family)
      expect(isRuntimeVideoCapable({ normalizedFamily: family })).toBe(true)
    }
  })

  it('shows the controls for an unlisted family whose bundle declares video', () => {
    expect(isRuntimeVideoCapable({ normalizedFamily: 'some-new-vlm', runtimeModalities: ['text', 'vision', 'video'] })).toBe(true)
    expect(isRuntimeVideoCapable({ normalizedFamily: undefined, runtimeModalities: ['TEXT', 'Video'] })).toBe(true)
  })

  it('hides the controls when neither the family nor the bundle says video', () => {
    expect(isRuntimeVideoCapable({ normalizedFamily: 'zaya1-vl', runtimeModalities: ['text', 'vision'] })).toBe(false)
    expect(isRuntimeVideoCapable({ normalizedFamily: normalizeDetectedFamilyName('zaya1_vl') })).toBe(false)
    expect(isRuntimeVideoCapable({ normalizedFamily: 'glm5-next', runtimeModalities: ['text'] })).toBe(false)
    expect(isRuntimeVideoCapable({})).toBe(false)
  })

  it('reads only a real modality list', () => {
    expect(modalitiesIncludeVideo(undefined)).toBe(false)
    expect(modalitiesIncludeVideo(null)).toBe(false)
    expect(modalitiesIncludeVideo([])).toBe(false)
    expect(modalitiesIncludeVideo(['video'])).toBe(true)
  })
})
