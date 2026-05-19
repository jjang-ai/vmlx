import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  appendReasoningDelta,
  markReasoningToolBoundary,
  reasoningSegmentsForDisplay,
  visibleReasoningSegments,
} from '../src/shared/interleavedReasoning'

describe('interleaved reasoning segments', () => {
  it('keeps separate reasoning segments around tool calls instead of replacing the first segment', () => {
    let segments = appendReasoningDelta([], 'Plan first.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'After tool, inspect result.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'Final synthesis.')

    expect(visibleReasoningSegments(segments)).toEqual([
      'Plan first.',
      'After tool, inspect result.',
      'Final synthesis.',
    ])
  })

  it('does not add empty duplicate segment boundaries for repeated tool status updates', () => {
    let segments = appendReasoningDelta([], 'Need shell.')
    segments = markReasoningToolBoundary(segments)
    segments = markReasoningToolBoundary(segments)
    segments = markReasoningToolBoundary(segments)

    expect(segments).toEqual(['Need shell.', ''])
    expect(visibleReasoningSegments(segments)).toEqual(['Need shell.'])
  })

  it('replaces old reasoning segments during live interleaved streaming, then can show all after completion', () => {
    let segments = appendReasoningDelta([], 'First plan before tools.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'Second plan after tool results.')

    expect(reasoningSegmentsForDisplay(segments, { liveReplace: true })).toEqual([
      'Second plan after tool results.',
    ])
    expect(reasoningSegmentsForDisplay(segments, { liveReplace: false })).toEqual([
      'First plan before tools.',
      'Second plan after tool results.',
    ])
  })

  it('marks a resumed reasoning segment as active again in the renderer', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')

    expect(source).toContain('setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: false }))')
  })
})
