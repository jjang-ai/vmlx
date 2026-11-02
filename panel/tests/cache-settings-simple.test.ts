import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import en from '../src/renderer/src/i18n/locales/en.json'
import es from '../src/renderer/src/i18n/locales/es.json'
import ja from '../src/renderer/src/i18n/locales/ja.json'
import ko from '../src/renderer/src/i18n/locales/ko.json'
import zh from '../src/renderer/src/i18n/locales/zh.json'

/**
 * Eric (2026-11-02): "there should only be % of max ssd for ssd caching
 * allowed or % of ram used for new prefix cache blocks cuz the rest is too
 * hard to understand." The Server Settings prefix-cache area shows exactly two
 * choices, SSD cache (default on, max % of disk) and RAM cache (default off,
 * max % of RAM). Everything else (block size, max blocks, paged toggles, RAM
 * copy, TTL, entry counts, byte limits, tier notes) is gone from the UI; the
 * saved keys and CLI flags are untouched.
 */
const form = readFileSync(
  resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'),
  'utf8',
)
const prefixSection = form.slice(
  form.indexOf('sectionKey="prefixCache"'),
  form.indexOf('sectionKey="power"'),
)
const LOCALES: Record<string, any> = { en, es, ja, ko, zh }

describe('Server Settings prefix cache: exactly two choices', () => {
  it('renders an SSD cache toggle with one percent-of-disk number and a RAM cache toggle with one percent-of-RAM number', () => {
    expect(prefixSection).toContain('data-vmlx-control="ssd-cache"')
    expect(prefixSection).toContain("label={t('sessions.config.ssdCache')}")
    expect(prefixSection).toContain('cacheControlUpdatesForSsdToggle(v, cacheControlState, cacheLane)')
    expect(prefixSection).toContain('settingKey="blockDiskCacheMaxPercent"')
    expect(prefixSection).toContain('data-vmlx-control="ram-cache"')
    expect(prefixSection).toContain("label={t('sessions.config.ramCache')}")
    expect(prefixSection).toContain('cacheControlUpdatesForRamToggle(v, cacheControlState, cacheLane)')
    expect(prefixSection).toContain('settingKey="cacheMemoryPercent"')
    // One slider each, and only while that choice is on.
    expect(prefixSection.match(/settingKey="blockDiskCacheMaxPercent"/g)).toHaveLength(1)
    expect(prefixSection.match(/settingKey="cacheMemoryPercent"/g)).toHaveLength(1)
    expect(prefixSection).toContain('{simpleCacheTier.ssdOn && (')
    expect(prefixSection).toContain('{simpleCacheTier.ramOn && (')
  })

  it('no longer shows the knobs nobody asked for', () => {
    for (const removed of [
      'settingKey="pagedCacheBlockSize"',
      'settingKey="maxCacheBlocks"',
      'settingKey="cacheMemoryMb"',
      'settingKey="cacheTtlMinutes"',
      'settingKey="prefixCacheSize"',
      'settingKey="prefixCacheMaxBytes"',
      'settingKey="diskCacheMaxGb"',
      "onChange('noMemoryAwareCache'",
      "onChange('enablePrefixCache'",
      "onChange('blockDiskCacheDir'",
      "onChange('diskCacheDir'",
      'ramCopyWithSsd',
      'cacheTierRamOnly',
      'cacheTierSsdOnly',
      'cacheTierRamSsd',
      'cacheTierNativeSsd',
      'pagedCacheHint',
      'blockDiskOnlyBudgetNote',
      'legacyEntryCountCache',
      'blockDiskCapacity',
      'data-vmlx-control="ram-copy-cache"',
      'data-vmlx-control="block-disk-cache"',
      'data-vmlx-section="typed-disk-cache"',
      "label={t('sessions.cache.blockDiskCache')}",
      "label={t('sessions.config.enableDiskCache')}",
      'resolveCacheControlPolicy',
      'cacheControlUpdatesForPagedToggle',
      'cacheControlUpdatesForBlockDiskToggle',
      'cacheControlUpdatesForDiskToggle',
      'pagedCacheControlsState',
      'resolvePagedCacheCapacity',
    ]) {
      expect(form, removed).not.toContain(removed)
    }
  })

  it('defaults: SSD cache on at the shared default percent, RAM cache off', () => {
    const defaults = form.slice(form.indexOf('export const DEFAULT_CONFIG'), form.indexOf('export const RESET_CONFIG'))
    expect(defaults).toMatch(/^\s*enablePrefixCache: true,\s*$/m)
    expect(defaults).toMatch(/^\s*enableBlockDiskCache: true,\s*$/m)
    expect(defaults).toMatch(/^\s*usePagedCache: false,\s*$/m)
    expect(defaults).toMatch(/^\s*blockDiskCacheMaxPercent: DEFAULT_BLOCK_DISK_CACHE_PERCENT,\s*$/m)
    expect(defaults).toMatch(/^\s*cacheMemoryPercent: 15,\s*$/m)
  })

  it.each(Object.keys(LOCALES))('%s carries the two plain labels and their one-number sliders', (name) => {
    const cfg = LOCALES[name].sessions.config
    for (const key of ['ssdCache', 'ssdCacheTooltip', 'ramCache', 'ramCacheTooltip', 'blockCacheMaxPercent', 'blockCacheMaxPercentTooltip', 'ramCacheMaxPercent', 'ramCacheMaxPercentTooltip', 'cacheOffNote', 'ssdOnlyLaneNote']) {
      expect(typeof cfg[key], `${name} sessions.config.${key}`).toBe('string')
      expect(cfg[key].length).toBeGreaterThan(3)
    }
    expect(cfg.blockCacheMaxPercent).toMatch(/%/)
    expect(cfg.ramCacheMaxPercent).toMatch(/%/)
    expect(cfg.ramCacheMaxPercent).toMatch(/RAM/)
    // The removed knobs' strings are gone in every language, not just English.
    for (const removed of ['ramCopyWithSsd', 'cacheTierRamOnly', 'cacheTierSsdOnly', 'cacheTierRamSsd', 'pagedCacheHint', 'blockDiskOnlyBudgetNote', 'legacyEntryCountCache', 'maxCacheEntries', 'cacheTtlMinutes', 'blockSizeTokens', 'maxCacheBlocks', 'cacheMemoryLimitMb']) {
      expect(cfg[removed], `${name} sessions.config.${removed}`).toBeUndefined()
    }
    // The tooltips never use the internal words that made the old controls unreadable.
    for (const key of ['ssdCacheTooltip', 'ramCacheTooltip', 'blockCacheMaxPercentTooltip', 'ramCacheMaxPercentTooltip', 'ramCacheTradeoffNotice']) {
      expect(cfg[key], `${name} ${key}`).not.toMatch(/paged|L1|L2|block[- ]?disk|typed companion|namespace/i)
    }
  })
})
