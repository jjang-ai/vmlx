import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

describe('download icon-only cancellation controls', () => {
  it('labels both paused and queued cancel buttons with the existing translated action', () => {
    const source = readFileSync(join(process.cwd(), 'src/renderer/src/components/DownloadsView.tsx'), 'utf8')
    const buttons = [...source.matchAll(/<button\b[\s\S]*?<\/button>/g)]
      .map(match => match[0]).filter(button => button.includes('<X '))
    expect(buttons).toHaveLength(2)
    for (const button of buttons) {
      expect(button).toContain("aria-label={t('common.cancel')}")
      expect(button).toContain("title={t('common.cancel')}")
      expect(button).toContain('window.api.models.cancelDownload(item.jobId)')
    }
  })
})
