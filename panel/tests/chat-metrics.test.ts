import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  calculateCachedPromptTps,
  calculatePrefillTps,
  estimateColdPrefillTps,
  summarizePrefillPasses,
  ExchangeFirstTokenClock,
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
  it('routes live, final, and abort through the measured receipt and both footer chips', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    expect(source.match(/= exchangePrefillSpeed\(\)/g)).toHaveLength(1)
    expect(source.match(/= footerPrefillRates\(/g)).toHaveLength(3)
    for (const spread of ['...prefillRates,', '...finalPrefillRates,', '...abortPrefillRates,']) {
      expect(source).toContain(spread)
    }
    expect(source).toContain('completedPrefillPasses.push(currentPrefillUsage)')
    expect(source).toContain('currentPrefillUsage = undefined;')
    expect(source).toContain('firstPass ??= { promptTokens, cachedTokens };')
    // Not serverSendsUsage: tool execution resets it before sendFollowUp, which
    // hid the cached chip on every tool turn (Spark pps row 2026-10-11).
    expect(source).toContain('const counts = first.promptTokens > 0 ? {')
  })
})

describe('footer pp/s chips (cold + cached reuse)', () => {
  it('estimates cold pp/s from TTFT only without a cache hit', () => {
    expect(estimateColdPrefillTps({ promptTokens: 4000, cachedTokens: 0, ttftSeconds: 5 })).toBe('800.0')
    expect(estimateColdPrefillTps({ promptTokens: 4000, cachedTokens: 3990, ttftSeconds: 0.4 })).toBeUndefined()
    expect(estimateColdPrefillTps({ promptTokens: 4000, ttftSeconds: 0 })).toBeUndefined()
  })
  it('reports cached pp/s as the whole prompt over TTFT on a cache hit only', () => {
    expect(calculateCachedPromptTps({ promptTokens: 9000, cachedTokens: 8960, ttftSeconds: 0.36 })).toBe('25000.0')
    expect(calculateCachedPromptTps({ promptTokens: 9000, cachedTokens: 0, ttftSeconds: 0.36 })).toBeUndefined()
    expect(calculateCachedPromptTps({ promptTokens: 9000, cachedTokens: 10, ttftSeconds: 0 })).toBeUndefined()
  })
})

describe('final chat decode TPS', () => {
  it('accepts Chat completion counts and requires timing for every tool pass', () => {
    const pass = parseServerDecodeUsage({ completion_tokens: 12,
      vmlx_decode: { tokens: 8, seconds: 2, tokens_per_second: 999 } })
    expect(pass?.tokensPerSecond).toBe(4)
    expect(summarizeServerDecodePasses([pass, undefined])).toBeUndefined()
    expect(summarizeServerDecodePasses([undefined, pass])).toBeUndefined()
    expect(summarizeServerDecodePasses([pass, pass])?.outputTokens).toBe(24)
  })

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


describe('whole tool-turn latency and prefill', () => {
  const receipt = (tokens: number, seconds: number) => ({
    scope: 'model_prefill_and_prompt_state', tokens, seconds,
  })
  it('keeps slow initial TTFT and weights a fast tool continuation by engine time', () => {
    const clock = new ExchangeFirstTokenClock()
    clock.beginPass(1000)
    clock.recordToken(21000)
    clock.beginPass(31000)
    clock.recordToken(32650)
    expect(clock.seconds()).toBe(20)
    expect(summarizePrefillPasses([receipt(2000, 18), receipt(107, 0.97)]))
      .toBe((2107 / 18.97).toFixed(1))
  })
  it('withholds exchange rate when any pass is missing its authoritative receipt', () => {
    expect(summarizePrefillPasses([undefined, receipt(107, 0.97)])).toBeUndefined()
    expect(summarizePrefillPasses([receipt(2000, 18), undefined])).toBeUndefined()
    expect(summarizePrefillPasses([{ tokens: 2000, seconds: 18 }, receipt(107, 0.97)]))
      .toBeUndefined()
  })
  it('keeps a tool-only first event when visible text arrives in a later pass', () => {
    const clock = new ExchangeFirstTokenClock()
    clock.beginPass(1000)
    clock.recordToken(3500) // structured tool name/arguments, no visible text
    clock.beginPass(8000)
    clock.recordToken(8200) // tool-result answer
    expect(clock.seconds()).toBe(2.5)
  })
  it('preserves first TTFT when a later request aborts before generation', () => {
    const clock = new ExchangeFirstTokenClock()
    clock.beginPass(100)
    clock.recordToken(900)
    clock.beginPass(2000)
    expect(clock.seconds()).toBe(0.8)
    expect(new ExchangeFirstTokenClock().seconds()).toBe(0)
  })
  it('handles measured zero-new-token passes without inventing throughput', () => {
    expect(summarizePrefillPasses([receipt(0, 0.1), receipt(100, 0.9)])).toBe('100.0')
    expect(summarizePrefillPasses([receipt(0, 0)])).toBeUndefined()
    expect(summarizePrefillPasses([receipt(100, 0)])).toBeUndefined()
    expect(summarizePrefillPasses([receipt(100, 2)])).toBe('50.0')
  })
})
