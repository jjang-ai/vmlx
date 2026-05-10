import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

vi.mock('electron', () => ({
  ipcMain: { handle: vi.fn() },
  dialog: { showOpenDialog: vi.fn() },
  BrowserWindow: { getAllWindows: () => [] },
}))

vi.mock('../src/main/database', () => ({
  db: {
    getSetting: vi.fn(() => undefined),
    saveBookmark: vi.fn(),
  },
}))

import { readGenerationDefaults } from '../src/main/ipc/models'

const createdDirs: string[] = []

function makeModelDir(files: Record<string, unknown>): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-generation-defaults-'))
  createdDirs.push(dir)
  for (const [name, payload] of Object.entries(files)) {
    writeFileSync(join(dir, name), JSON.stringify(payload, null, 2))
  }
  return dir
}

afterEach(() => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir) rmSync(dir, { recursive: true, force: true })
  }
})

describe('readGenerationDefaults JANG sampling defaults', () => {
  it('uses chat repetition penalty when bundle default reasoning mode is not thinking', async () => {
    const dir = makeModelDir({
      'jang_config.json': {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            temperature: 0.6,
            top_p: 0.95,
            repetition_penalty_thinking: 1.0,
            repetition_penalty_chat: 1.05,
            max_new_tokens: 4096,
          },
        },
      },
    })

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.6,
      topP: 0.95,
      repeatPenalty: 1.05,
      maxNewTokens: 4096,
      source: 'jang_config',
    })
  })

  it('uses thinking repetition penalty when bundle default reasoning mode is thinking', async () => {
    const dir = makeModelDir({
      'jang_config.json': {
        chat: {
          reasoning: { default_mode: 'thinking' },
          sampling_defaults: {
            repetition_penalty_thinking: 1.0,
            repetition_penalty_chat: 1.05,
          },
        },
      },
    })

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      repeatPenalty: 1.0,
      source: 'jang_config',
    })
  })
})
