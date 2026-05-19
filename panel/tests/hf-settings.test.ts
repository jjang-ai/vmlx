import { describe, expect, it } from 'vitest'

import {
  HF_CANONICAL_ENDPOINT,
  normalizeHfEndpointSetting,
  normalizeHfTokenSetting,
} from '../src/shared/hfSettings'

describe('HuggingFace persisted settings', () => {
  it('normalizes tokens before passing them to downloads or API calls', () => {
    expect(normalizeHfTokenSetting(null)).toBeNull()
    expect(normalizeHfTokenSetting('')).toBeNull()
    expect(normalizeHfTokenSetting('   ')).toBeNull()
    expect(normalizeHfTokenSetting('  hf_example_token  ')).toBe('hf_example_token')
  })

  it('keeps empty or invalid mirror settings from poisoning GUI downloads', () => {
    expect(normalizeHfEndpointSetting(null)).toBeNull()
    expect(normalizeHfEndpointSetting('')).toBeNull()
    expect(normalizeHfEndpointSetting('   ')).toBeNull()
    expect(normalizeHfEndpointSetting('hf-mirror.com')).toBeNull()
    expect(normalizeHfEndpointSetting('file:///tmp/not-hf')).toBeNull()
  })

  it('canonicalizes valid HF-compatible endpoints without changing the default', () => {
    expect(normalizeHfEndpointSetting(' https://hf-mirror.com/// ')).toBe('https://hf-mirror.com')
    expect(normalizeHfEndpointSetting(`${HF_CANONICAL_ENDPOINT}/`)).toBe(HF_CANONICAL_ENDPOINT)
  })
})
