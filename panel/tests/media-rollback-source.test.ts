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
})
