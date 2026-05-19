export type ReasoningSegments = string[]

export function appendReasoningDelta(
  segments: ReasoningSegments,
  delta: string,
): ReasoningSegments {
  if (!delta) return segments
  const next = [...segments]
  if (next.length === 0) {
    next.push(delta)
  } else {
    next[next.length - 1] += delta
  }
  return next
}

export function markReasoningToolBoundary(
  segments: ReasoningSegments,
): ReasoningSegments {
  if (segments.length === 0) return segments
  const next = [...segments]
  const last = next[next.length - 1]
  if (last && last.trim().length > 0) {
    next.push('')
  }
  return next
}

export function visibleReasoningSegments(
  segments?: ReasoningSegments | null,
): ReasoningSegments {
  return (segments || []).filter((segment) => segment.trim().length > 0)
}

export function reasoningSegmentsForDisplay(
  segments?: ReasoningSegments | null,
  options?: { liveReplace?: boolean },
): ReasoningSegments {
  const visible = visibleReasoningSegments(segments)
  if (options?.liveReplace && visible.length > 1) {
    return [visible[visible.length - 1]]
  }
  return visible
}

export function joinReasoningSegments(
  segments?: ReasoningSegments | null,
): string {
  return visibleReasoningSegments(segments).join('\n\n')
}
