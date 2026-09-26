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

  it('uses the top-level profile shape emitted by current JANG sidecars', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      profile: 'JANGTQ2',
      quantization: { bits: undefined },
    })).toBe('JANGTQ2 (2b)')
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

  it('labels affine JANG sidecars as JANG profiles rather than JANGTQ', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'affine',
      profile: 'JANG_2L_GS64',
      quantization: { target_bits: 2 },
    })).toBe('JANG_2L_GS64 (2b)')
  })

  it('does not mislabel mixed routed JANG profiles with their affine container bits', () => {
    expect(formatJangQuantizationLabel({
      format: 'jang',
      quantization: {
        profile: 'JANG_2K',
        bits: 8,
        routed_avg_bits: 2.3333333333333335,
      },
    })).toBe('JANG_2K (2.33b routed)')
  })

  it('falls back without leaking undefined when profile has no known bit mapping', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq',
      quantization: { profile: 'JANGTQ_K' },
    })).toBe('JANGTQ_K')
  })
})


describe('JANGH legacy bundle identity', () => {
  it('names v2 separately while retaining original JANGTQ', () => {
    expect(formatJangQuantizationLabel({ format: 'jangtq2' })).toBe('JANGH')
    expect(formatJangQuantizationLabel({ weight_format: 'jangtq2' })).toBe('JANGH')
    expect(formatJangQuantizationLabel({ format: 'jangtq' })).toBe('JANGTQ')
  })
})
