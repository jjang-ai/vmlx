import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  calculatePrefillTps,
  parseServerDecodeUsage,
  selectFinalDecodeTps,
  summarizeServerDecodePasses,
} from '../src/shared/chatMetrics'

describe('chat prefill TPS', () => {
  const base = {}
  it('uses the engine processed-token window independently of queueing and cache restore', () => {
    expect(calculatePrefillTps({ ...base, prefillUsage: {
      tokens: 107, seconds: 0.2, scope: 'model_prefill_and_prompt_state',
    } })).toBe('535.0')
  })
  it('does not invent prefill throughput from usage and TTFT', () => {
    expect(calculatePrefillTps(base)).toBeUndefined()
  })
  it.each([
    { tokens: 0, seconds: 1 }, { tokens: -1, seconds: 1 },
    { tokens: 10, seconds: 0 }, { tokens: 10, seconds: Infinity },
    { tokens: 1.5, seconds: 1 }, { tokens: '10', seconds: 1 },
  ])('rejects invalid measured receipts %j', receipt => {
    expect(calculatePrefillTps({ ...base, prefillUsage: {
      ...receipt, scope: 'model_prefill_and_prompt_state',
    } })).toBeUndefined()
  })
  it('rejects unknown timing definitions', () => {
    expect(calculatePrefillTps({ ...base, prefillUsage: {
      tokens: 107, seconds: 0.2, scope: 'ttft',
    } })).toBeUndefined()
  })
  it('routes live, final, and abort through the measured receipt', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    expect(source.match(/prefillUsage: currentPrefillUsage/g)).toHaveLength(3)
    expect(source).toContain('currentPrefillUsage = undefined;')
  })
})

describe('final chat decode TPS', () => {
  it('uses authoritative per-request decode windows for a tool-loop exchange', () => {
    const first = parseServerDecodeUsage({
      output_tokens: 85,
      vmlx_decode: {
        tokens: 84,
        seconds: 1.85,
        tokens_per_second: 45.405,
      },
    })
    const followUp = parseServerDecodeUsage({
      output_tokens: 237,
      vmlx_decode: {
        tokens: 236,
        seconds: 5.25,
        tokens_per_second: 44.952,
      },
    })

    expect(summarizeServerDecodePasses([first, followUp])).toEqual({
      outputTokens: 322,
      decodeTokens: 320,
      decodeSeconds: 7.1,
      tokensPerSecond: 320 / 7.1,
    })
  })

  it('rejects malformed server decode telemetry instead of fabricating a rate', () => {
    expect(
      parseServerDecodeUsage({
        output_tokens: 237,
        vmlx_decode: { tokens: 236, seconds: 0 },
      }),
    ).toBeUndefined()
    expect(summarizeServerDecodePasses([undefined])).toBeUndefined()
  })

  it('keeps cumulative multi-iteration throughput when only the final tail is slow', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 49.6,
        rollingTps: [48.8, 49.1, 50.3, 49.4, 8.3],
        lastRollingTps: 8.3,
      }),
    ).toBe(49.6)
  })

  it('rejects an impossible cumulative burst from buffered output', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 261,
        rollingTps: [42.7, 42.9, 43.1],
        lastRollingTps: 43.1,
      }),
    ).toBe(42.9)
  })

  it('falls back cleanly when only one timing source is available', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 0,
        rollingTps: [],
        lastRollingTps: 37.5,
      }),
    ).toBe(37.5)
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 31.25,
        rollingTps: [],
      }),
    ).toBe(31.25)
  })
})
