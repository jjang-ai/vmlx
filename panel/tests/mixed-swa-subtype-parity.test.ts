import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { isMixedSwaBundle } from '../src/shared/storedKvQuantPolicy'

const form = readFileSync(
  resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf-8')

/**
 * step3p7_full_sliding_kv was listed as mixed-SWA in three of four places:
 * mixedSwaBlockDiskOnlySupported, subtypeRequiresPagedCache and the ENGINE's
 * own detector (mllm_scheduler returns true for it alongside mixed_swa_kv) —
 * but NOT in mixedSwaCacheActive, which drives the cache codec badge and its
 * description. Step-3.7 therefore showed a generic "AUTO" badge with the
 * engine-native copy, for a family everything else treats as mixed-SWA.
 *
 * That drift is now structurally impossible rather than merely asserted:
 * mixedSwaCacheActive and mixedSwaBlockDiskOnlySupported were two identical
 * hand-copied chains and both now read from shared/storedKvQuantPolicy. The
 * checks below moved to the policy (membership) plus a delegation check on the
 * form (nobody re-inlines it). subtypeRequiresPagedCache is a DIFFERENT
 * question — whether the subtype forces the paged pool — so it keeps its own
 * literal list and its own assertion.
 */
describe('mixed-SWA subtype membership lives in one policy', () => {
  it.each(['step3p7_full_sliding_kv', 'mixed_swa_kv'])(
    'the shared policy classifies %s as mixed-SWA',
    (subtype) => {
      expect(isMixedSwaBundle({ cacheSubtype: subtype })).toBe(true)
    },
  )

  it('rotating_kv and the schema/arch hints classify too', () => {
    expect(isMixedSwaBundle({ cacheType: 'rotating_kv' })).toBe(true)
    expect(
      isMixedSwaBundle({ architectureHints: { cacheSchema: 'mixed_swa_kv_v1' } }),
    ).toBe(true)
    expect(
      isMixedSwaBundle({ architectureHints: { attentionArch: 'full_and_sliding_kv' } }),
    ).toBe(true)
  })

  it('the form delegates its remaining consumer to the policy', () => {
    // The display-only consumer (mixedSwaCacheActive, which drove the removed
    // cache-codec badge) is gone with the badge; the one consumer left still
    // reads the shared policy rather than an inlined chain.
    expect(form).toContain('shared/storedKvQuantPolicy')
    expect(form).toContain('const mixedSwaBundle = isMixedSwaBundle(')
    expect(form).toContain('const mixedSwaBlockDiskOnlySupported = mixedSwaBundle')
    expect(form).not.toContain('mixedSwaCacheActive')
  })

  it('neither consumer re-inlines the subtype chain', () => {
    const inlined =
      /detectedCacheType === 'rotating_kv' \|\|\s*\n\s*detectedCacheSubtype === 'mixed_swa_kv'/
    expect(inlined.test(form)).toBe(false)
  })

  it('the retired RAM tier can no longer be forced back on by any subtype', () => {
    // subtypeRequiresPagedCache was retired with the RAM tier itself: no
    // detected family may force the paged cache back on. The membership list
    // both consumers share now lives in shared/storedKvQuantPolicy.ts.
    expect(form.indexOf('const subtypeRequiresPagedCache =')).toBe(-1)
    expect(form).toContain('const architectureRequiresPagedCache = false')
    const policy = readFileSync(
      resolve(__dirname, '..', 'src/shared/storedKvQuantPolicy.ts'), 'utf-8')
    expect(policy).toContain("'step3p7_full_sliding_kv'")
    expect(policy).toContain("'mixed_swa_kv'")
  })
})

describe('the engine agrees these are the mixed-SWA subtypes', () => {
  it('mllm_scheduler classifies both the same way', () => {
    const sched = readFileSync(
      resolve(__dirname, '../../vmlx_engine/mllm_scheduler.py'), 'utf-8')
    const start = sched.indexOf('"mixed_swa_kv"')
    const block = sched.slice(start - 200, start + 300)
    expect(block).toContain('"step3p7_full_sliding_kv"')
  })
})
