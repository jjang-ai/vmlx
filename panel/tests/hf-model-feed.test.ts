import { describe, expect, it } from 'vitest'
import { hfModelFeedPath, validateHfModelFeed } from '../src/shared/hfModelFeed'

describe('author model feeds', () => {
  for (const author of ['JANGQ-AI', 'dealignai']) {
    for (const sort of ['createdAt', 'lastModified'] as const) {
      it(`${author} requests newest ${sort} without a popularity or collection filter`, () => {
        const url = new URL(hfModelFeedPath(author, sort), 'https://huggingface.co')
        expect(url.pathname).toBe('/api/models')
        expect(Object.fromEntries(url.searchParams)).toEqual({ author, sort, direction: '-1', limit: '100', full: 'true' })
      })
    }
  }
  it('rejects arbitrary scopes and sort values at the IPC boundary', () => {
    expect(() => hfModelFeedPath('../other', 'createdAt')).toThrow()
    expect(() => hfModelFeedPath('dealignai', 'downloads' as any)).toThrow()
  })
  it('keeps author order, missing metadata and distinct formats without asserting compatibility', () => {
    const models = [{ id: 'dealignai/new-JANG_4M', createdAt: '2026-09-01' }, { id: 'dealignai/old-JANGTQ', tags: ['gguf'] }]
    expect(validateHfModelFeed([...models, models[0], null, { id: 'other/model' }, { id: 'dealignai/profile' }], 'dealignai')).toEqual(models)
  })
  it('distinguishes an empty feed from an invalid response', () => {
    expect(validateHfModelFeed([], 'dealignai')).toEqual([])
    expect(() => validateHfModelFeed({ error: 'rate limited' }, 'dealignai')).toThrow()
  })
  it('filters on the name rather than the JANGQ-AI owner and retains newest order', () => {
    const rows = ['Qwen-NVFP4', 'Qwen-JANG_2L', 'Qwen-EXL3', 'MiniCPM-jang8m', 'Old-JANGTQ', 'Model-bf16'].map(name => ({ id: `JANGQ-AI/${name}` }))
    expect(validateHfModelFeed(rows, 'JANGQ-AI')).toEqual([rows[1], rows[3], rows[4]])
    expect(validateHfModelFeed([{ modelId: 'dealignai/Model-JANG_4M' }, { id: 'dealignai/Model-EXL3' }], 'dealignai')).toEqual([{ modelId: 'dealignai/Model-JANG_4M' }])
  })
})
