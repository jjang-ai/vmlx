import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const chatSource = readFileSync(join(process.cwd(), 'src/main/ipc/chat.ts'), 'utf8')

describe('failed media message rollback', () => {
  it('removes a failed media user message so later text turns do not replay the image', () => {
    expect(chatSource).toContain('failed oversized media turn')
    expect(chatSource).toContain('rolled_back_failed_media_user_message')
    expect(chatSource).toContain('hasMediaAttachments &&')
    expect(chatSource).toContain('!hadVisibleActivity')
    expect(chatSource).toContain('!wasAborted')
    expect(chatSource).toContain('db.deleteMessage(userMessage.id)')
  })

  it('also removes media turns that complete with empty warning responses', () => {
    expect(chatSource).toContain('mediaWarningWithoutVisibleActivity')
    expect(chatSource).toContain('finalResponseWarnings.length > 0')
    expect(chatSource).toContain('rolled_back_empty_warning_media_user_message')
    expect(chatSource).toContain('Media request failed before visible output')
    expect(chatSource).toContain('Do not persist the failed media user turn')
    expect(chatSource).toContain('text-only prompt replays the same image')
  })
})
