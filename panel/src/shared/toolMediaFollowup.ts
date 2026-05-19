export type ToolMediaFollowupContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'video_url'; video_url: { url: string } }

export function buildToolMediaFollowupContent(
  imageDataUrls: string[],
  videoDataUrls: string[],
): ToolMediaFollowupContentPart[] | null {
  if (imageDataUrls.length === 0 && videoDataUrls.length === 0) return null

  return [
    {
      type: 'text',
      text: 'Here is the media from the tool results above.',
    },
    ...imageDataUrls.map((url) => ({
      type: 'image_url' as const,
      image_url: { url },
    })),
    ...videoDataUrls.map((url) => ({
      type: 'video_url' as const,
      video_url: { url },
    })),
  ]
}
