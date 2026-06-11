import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { buildToolMediaFollowupContent } from '../src/shared/toolMediaFollowup'

const repoRoot = new URL('..', import.meta.url).pathname

describe('tool media follow-up routing', () => {
  it('exposes read_video as a file tool next to read_image', () => {
    const registry = readFileSync(`${repoRoot}/src/main/tools/registry.ts`, 'utf8')
    const chat = readFileSync(`${repoRoot}/src/main/ipc/chat.ts`, 'utf8')

    expect(registry).toContain("name: 'read_image'")
    expect(registry).toContain("name: 'read_video'")
    expect(chat).toContain('"read_image"')
    expect(chat).toContain('"read_video"')
  })

  it('injects image and video tool results as multimodal follow-up content parts', () => {
    const chat = readFileSync(`${repoRoot}/src/main/ipc/chat.ts`, 'utf8')
    const helper = readFileSync(`${repoRoot}/src/shared/toolMediaFollowup.ts`, 'utf8')

    expect(chat).toContain('pendingImageDataUrls')
    expect(chat).toContain('pendingVideoDataUrls')
    expect(chat).toContain('result.imageDataUrl')
    expect(chat).toContain('result.videoDataUrl')
    expect(chat).toContain('chatIsMultimodal || (isRemote && !modelForceTextOnly)')
    expect(chat).toContain('Skipping tool media bytes for text-only local session')
    expect(chat).toContain('buildToolMediaFollowupContent')
    expect(helper).toContain("type: 'image_url'")
    expect(helper).toContain("type: 'video_url'")
  })

  it('keeps filesystem media-reader tools out of direct attachment requests', () => {
    const chat = readFileSync(`${repoRoot}/src/main/ipc/chat.ts`, 'utf8')

    expect(chat).toContain('DIRECT_MEDIA_ATTACHMENT_TOOL_RULE')
    expect(chat).toContain('hasDirectMediaAttachments?: boolean')
    expect(chat).toContain('context.hasDirectMediaAttachments')
    expect(chat).toContain('disabled.add("read_image")')
    expect(chat).toContain('disabled.add("read_video")')
    expect(chat).toContain('hasDirectMediaAttachments: hasMediaAttachments')
  })

  it('builds real multimodal follow-up content in text-image-video order', () => {
    const content = buildToolMediaFollowupContent(
      ['data:image/png;base64,AAAA'],
      ['data:video/mp4;base64,BBBB'],
    )

    expect(content).toEqual([
      { type: 'text', text: 'Here is the media from the tool results above.' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
      { type: 'video_url', video_url: { url: 'data:video/mp4;base64,BBBB' } },
    ])
  })

  it('does not create an empty media follow-up when no tool returned media bytes', () => {
    expect(buildToolMediaFollowupContent([], [])).toBeNull()
  })

  it('keeps media bytes out of plain tool text and returns data URLs separately', () => {
    const executor = readFileSync(`${repoRoot}/src/main/tools/executor.ts`, 'utf8')

    expect(executor).toContain('imageDataUrl?: string')
    expect(executor).toContain('videoDataUrl?: string')
    expect(executor).toContain("case 'read_image'")
    expect(executor).toContain("case 'read_video'")
    expect(executor).toContain('video/mp4')
    expect(executor).toContain('data:${mime};base64')
    expect(executor).toContain('The video has been attached for visual analysis.')
  })
})
