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

function makeModelDir(
  files: Record<string, unknown>,
  prefix = 'vmlx-generation-defaults-',
): string {
  const dir = mkdtempSync(join(tmpdir(), prefix))
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

describe('readGenerationDefaults generation_config defaults', () => {
  it('uses standard MLX generation_config.json for non-JANG and VLM bundles', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.02,
        repetition_penalty: 1.05,
        max_new_tokens: 8192,
        max_thinking_tokens: 4096,
      },
    }, 'vmlx-generation-defaults-mlx-vl-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.8,
      topP: 0.9,
      topK: 40,
      minP: 0.02,
      repeatPenalty: 1.05,
      maxNewTokens: 8192,
      maxThinkingTokens: 4096,
      source: 'generation_config',
    })
  })

  it('normalizes disabled top_k sentinels to Off/0 for UI and requests', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        temperature: 1.0,
        top_p: 1.0,
        top_k: -1,
      },
    }, 'vmlx-generation-defaults-disabled-topk-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 1.0,
      topP: 1.0,
      topK: 0,
      source: 'generation_config',
    })
  })

  it('treats generation_config do_sample=false as effective greedy sampling', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        do_sample: false,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 40,
        max_new_tokens: 2048,
      },
    }, 'vmlx-generation-defaults-do-sample-false-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      doSample: false,
      temperature: 0,
      topP: 1,
      topK: 0,
      maxNewTokens: 2048,
      source: 'generation_config',
    })
  })

  it('lets JANG chat sampling metadata override generation_config.json', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        do_sample: false,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        repetition_penalty: 1.1,
      },
      'jang_config.json': {
        chat: {
          sampling_defaults: {
            temperature: 0.55,
            top_p: 0.92,
            top_k: 64,
            repetition_penalty: 1.0,
          },
        },
      },
    }, 'vmlx-generation-defaults-jang-override-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.55,
      topP: 0.92,
      topK: 64,
      repeatPenalty: 1.0,
      source: 'jang_config',
    })
    const defaults = await readGenerationDefaults(dir)
    expect(defaults?.doSample).toBeUndefined()
  })

  it('surfaces Step Flash JANG greedy chat metadata instead of treating zero temperature as missing', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        bos_token_id: 0,
        eos_token_id: [1, 2, 128007],
      },
      'jang_config.json': {
        chat: {
          reasoning: {
            default_mode: 'no_think',
            parser: 'qwen3',
          },
          sampling_defaults: {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
          },
          tool_calling: {
            parser: 'step3p5',
            supported: true,
          },
        },
      },
    }, 'vmlx-generation-defaults-step37-flash-jang-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.0,
      topP: 1.0,
      topK: 0,
      source: 'jang_config',
    })
  })

  it('returns null when neither metadata file defines sampling defaults', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        bos_token_id: 1,
        eos_token_id: 2,
      },
      'config.json': {
        model_type: 'qwen3',
      },
    }, 'vmlx-generation-defaults-empty-')

    await expect(readGenerationDefaults(dir)).resolves.toBeNull()
  })

  it('marks thinking-budget unsupported when the template has thinking but no budget variable', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        max_thinking_tokens: 4096,
      },
    }, 'vmlx-generation-defaults-thinking-budget-unsupported-')
    writeFileSync(
      join(dir, 'chat_template.jinja'),
      '{% if enable_thinking %}<|think|>{% endif %}',
    )

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      thinkingBudgetSupported: false,
      source: 'generation_config',
    })
    const defaults = await readGenerationDefaults(dir)
    expect(defaults?.maxThinkingTokens).toBeUndefined()
  })

  it('surfaces maxThinkingTokens only when the template consumes thinking_budget', async () => {
    const dir = makeModelDir({
      'generation_config.json': {
        max_thinking_tokens: 4096,
      },
    }, 'vmlx-generation-defaults-thinking-budget-supported-')
    writeFileSync(
      join(dir, 'chat_template.jinja'),
      '{% if enable_thinking %}{{ thinking_budget }}<|think|>{% endif %}',
    )

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      maxThinkingTokens: 4096,
      thinkingBudgetSupported: true,
      source: 'generation_config',
    })
  })
})

describe('readGenerationDefaults JANG sampling defaults', () => {
  it('uses Ling CRACK bundle startup defaults without hidden temperature override', async () => {
    const dir = makeModelDir({
      'jang_config.json': {
        chat: {
          sampling_defaults: {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.0,
          },
        },
      },
    }, 'vmlx-generation-defaults-Ling-2.6-flash-JANGTQ2-CRACK-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.6,
      topP: 0.95,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.0,
      source: 'jang_config',
    })
  })

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
            thinking_budget: 2048,
          },
        },
      },
    })

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.6,
      topP: 0.95,
      repeatPenalty: 1.05,
      maxNewTokens: 4096,
      maxThinkingTokens: 2048,
      source: 'jang_config',
    })
  })

  it('uses neutral generic repetition penalty for DSV4 direct chat defaults', async () => {
    const dir = makeModelDir({
      'config.json': {
        model_type: 'deepseek_v4',
      },
      'jang_config.json': {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            temperature: 0.6,
            top_p: 0.95,
            repetition_penalty: 1.0,
            repetition_penalty_thinking: 1.0,
            repetition_penalty_chat: 1.05,
            max_new_tokens: 4096,
          },
        },
      },
    }, 'vmlx-generation-defaults-dsv4-')

    await expect(readGenerationDefaults(dir)).resolves.toMatchObject({
      temperature: 0.6,
      topP: 0.95,
      repeatPenalty: 1.0,
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
