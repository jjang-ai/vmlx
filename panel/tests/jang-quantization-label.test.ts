import { describe, expect, it } from 'vitest'

import { formatJangQuantizationLabel } from '../src/shared/jangQuantization'

describe('JANG quantization labels', () => {
  it('derives JANGTQ1 bits but labels it unsupported', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      quantization: { profile: 'JANGTQ1' },
    })).toBe('JANGTQ1 (1b, unsupported)')
  })

  it('derives JANGTQ2 and JANGTQ4 bits from profile when explicit bits are absent', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      quantization: { profile: 'JANGTQ2' },
    })).toBe('JANGTQ2 (2b)')
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      quantization: { profile: 'JANGTQ4' },
    })).toBe('JANGTQ4 (4b)')
  })

  it('keeps explicit JANG actual bits for affine JANG profiles', () => {
    expect(formatJangQuantizationLabel({
      format: 'jang',
      quantization: { profile: 'JANG_2L', actual_bits: 2.73, target_bits: 2 },
    })).toBe('JANG_2L (2.73b)')
  })

  it('keeps explicit JANG bits when bundles stamp weight_format instead of format', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'jang',
      quantization: { profile: 'JANG_4M', actual_bits: 4.45, target_bits: 4 },
    })).toBe('JANG_4M (4.45b)')
  })

  it('falls back without leaking undefined when profile has no known bit mapping', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      quantization: { profile: 'JANGTQ_K' },
    })).toBe('JANGTQ_K')
  })
})
