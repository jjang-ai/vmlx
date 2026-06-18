import { describe, expect, it } from 'vitest'

import {
  pagedCacheCapacityText,
  pagedCacheControlsState,
  pagedCacheMemoryIgnoredText,
} from '../src/shared/cacheCapacityDisplay'

describe('cache capacity display helpers', () => {
  it('shows effective paged-cache capacity as block size times max blocks', () => {
    expect(pagedCacheCapacityText({ blockSize: 64, maxBlocks: 1000 })).toBe(
      'Effective paged capacity: 64 tokens/block x 1000 blocks = 64,000 tokens',
    )
    expect(pagedCacheCapacityText({ blockSize: 256, maxBlocks: 64 })).toBe(
      'Effective paged capacity: 256 tokens/block x 64 blocks = 16,384 tokens',
    )
  })

  it('names MB and percent cache controls as ignored while paged cache is active', () => {
    expect(pagedCacheMemoryIgnoredText).toContain('Cache Memory Limit')
    expect(pagedCacheMemoryIgnoredText).toContain('Cache Memory %')
    expect(pagedCacheMemoryIgnoredText).toContain('ignored while paged cache is active')
    expect(pagedCacheMemoryIgnoredText).toContain('Max Cache Blocks')
  })

  it('derives disabled/ignored control state from effective paged cache state', () => {
    expect(pagedCacheControlsState(true)).toEqual({
      memoryBudgetControlsDisabled: true,
      cacheTtlDisabled: true,
      memoryBudgetIgnored: true,
    })
    expect(pagedCacheControlsState(false)).toEqual({
      memoryBudgetControlsDisabled: false,
      cacheTtlDisabled: false,
      memoryBudgetIgnored: false,
    })
  })
})
