const PROJECTED_METAL_HEADROOM_RE =
  /Requested max output tokens exceed projected safe Metal headroom/i

export function projectedMetalHeadroomChatErrorContent(
  message: string | undefined | null,
): string | null {
  const raw = String(message || "").trim()
  if (!PROJECTED_METAL_HEADROOM_RE.test(raw)) return null

  const detail = raw
    .replace(/^Failed to send message:\s*/i, "")
    .replace(/^API error:\s*413\s*-\s*/i, "")
    .trim()

  return `Generation blocked: ${detail}`
}
