export const HF_MODEL_FEED_AUTHORS = { jang: 'JANGQ-AI', uncensored: 'dealignai' } as const
export type HfModelFeedSort = 'createdAt' | 'lastModified'

/** Fixed author scope; authentication and mirror routing remain main-process owned. */
export function hfModelFeedPath(author: string, sort: HfModelFeedSort): string {
  if (!Object.values(HF_MODEL_FEED_AUTHORS).some(value => value === author)) {
    throw new Error('Unsupported model feed author')
  }
  if (sort !== 'createdAt' && sort !== 'lastModified') throw new Error('Unsupported model feed sort')
  return `/api/models?${new URLSearchParams({ author, sort, direction: '-1', limit: '100', full: 'true' })}`
}

export function validateHfModelFeed(value: unknown, author: string): Record<string, any>[] {
  if (!Array.isArray(value)) throw new Error('Invalid model feed response')
  const seen = new Set<string>()
  return value.filter(model => {
    const id = model?.id || model?.modelId
    if (typeof id !== 'string' || !id.toLowerCase().startsWith(`${author.toLowerCase()}/`) || seen.has(id)) return false
    // HF exposes the account's profile README as a model-list repository.
    if (id.toLowerCase() === `${author.toLowerCase()}/profile`) return false
    seen.add(id)
    return true
  })
}
