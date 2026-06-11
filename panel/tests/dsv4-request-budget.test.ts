import { describe, expect, it } from 'vitest'
import {
  dsv4FinalizerTokens,
  dsv4OutputBudget,
} from '../src/shared/dsv4RequestBudget'

describe('DSV4 chat request budget helpers', () => {
  it('preserves explicit DSV4 max token budgets without adding hidden finalizer tokens', () => {
    expect(dsv4OutputBudget(128, true, 'deepseek-v4')).toBe(128)
    expect(dsv4FinalizerTokens(true, 'deepseek-v4')).toBeUndefined()
  })

  it('ignores legacy session DSV4 finalizer overrides so decode stays natural', () => {
    expect(dsv4FinalizerTokens(true, 'deepseek-v4', 4096)).toBeUndefined()
    expect(dsv4FinalizerTokens(undefined, 'deepseek-v4', 4096.9)).toBeUndefined()
  })

  it('does not synthesize a DSV4 max token budget when the request leaves it to server/model defaults', () => {
    expect(dsv4OutputBudget(undefined, true, 'deepseek-v4')).toBeUndefined()
    expect(dsv4FinalizerTokens(true, 'deepseek-v4', 0)).toBeUndefined()
    expect(dsv4FinalizerTokens(true, 'deepseek-v4', -1)).toBeUndefined()
    expect(dsv4FinalizerTokens(true, 'deepseek-v4', Number.NaN)).toBeUndefined()
  })

  it('DSV4 request budget rejects invalid caps and floors fractional explicit caps', () => {
    for (const value of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, '8192', null]) {
      expect(dsv4OutputBudget(value, true, 'deepseek-v4')).toBeUndefined()
    }

    expect(dsv4OutputBudget(8192.9, true, 'deepseek-v4')).toBe(8192)
  })

  it('does not change explicit Max reasoning budgets', () => {
    expect(dsv4OutputBudget(128, true, 'deepseek-v4', 'max')).toBe(128)
  })

  it('does not apply DSV4 thinking budgets when thinking is explicitly off', () => {
    expect(dsv4OutputBudget(128, false, 'deepseek-v4')).toBe(128)
    expect(dsv4FinalizerTokens(false, 'deepseek-v4', 4096)).toBeUndefined()
  })

  it('raises undersized Gemma4 reasoning tool budgets without changing other turns', () => {
    expect(dsv4OutputBudget(128, true, 'gemma4', undefined, true)).toBe(512)
    expect(dsv4OutputBudget(128, true, 'gemma4-text', undefined, true)).toBe(512)
    expect(dsv4OutputBudget(512, true, 'gemma4', undefined, true)).toBe(512)
    expect(dsv4OutputBudget(768, true, 'gemma4', undefined, true)).toBe(768)
    expect(dsv4OutputBudget(128, false, 'gemma4', undefined, true)).toBe(128)
    expect(dsv4OutputBudget(128, true, 'gemma4', undefined, false)).toBe(128)
    expect(dsv4OutputBudget(128, true, 'qwen3_5_moe', undefined, true)).toBe(128)
  })
})
