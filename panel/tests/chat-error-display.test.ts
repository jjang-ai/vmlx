import { describe, expect, it } from 'vitest'
import {
  projectedMetalHeadroomChatErrorContent,
} from '../src/shared/chatErrorDisplay'

describe('chat error display policy', () => {
  it('turns projected Metal headroom rejects into visible assistant content', () => {
    const content = projectedMetalHeadroomChatErrorContent(
      'API error: 413 - Requested max output tokens exceed projected safe Metal headroom: requested=8192, safe_cap=1. Reduce max_tokens/max_output_tokens, context length, paged cache blocks, or load a smaller model. Set VMLX_METAL_PROJECTED_OUTPUT_GUARD=0 only for explicit developer diagnostics; disabling it accepts Metal OOM / kernel-panic risk.',
    )

    expect(content).toContain('Generation blocked')
    expect(content).toContain('requested=8192')
    expect(content).toContain('safe_cap=1')
    expect(content).toContain('Metal OOM / kernel-panic risk')
  })

  it('does not convert ordinary connection errors into assistant content', () => {
    expect(projectedMetalHeadroomChatErrorContent('fetch failed')).toBeNull()
    expect(projectedMetalHeadroomChatErrorContent('Server connection lost')).toBeNull()
  })
})
