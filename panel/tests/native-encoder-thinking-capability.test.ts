import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'
const dirs: string[] = []
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })
function bundle(model_type: string, tokenizer: object, template?: string) {
  const dir = mkdtempSync(join(tmpdir(), 'native-thinking-')); dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type }))
  writeFileSync(join(dir, 'tokenizer_config.json'), JSON.stringify(tokenizer))
  if (template !== undefined) writeFileSync(join(dir, 'chat_template.jinja'), template)
  return dir
}
describe('native encoder thinking capability', () => {
  it('exposes DSV4 native chat and thinking without a Jinja template', () => {
    const dir = bundle('deepseek_v4', { tokenizer_class: 'PreTrainedTokenizerFast' })
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify({ chat: {
      encoder: 'encoding_dsv4', reasoning: { supported: true, modes: ['chat', 'thinking'],
        default_mode: 'thinking', default_effort: 'low', reasoning_effort_levels: ['low', 'high', 'max'] },
    } }))
    expect(detectModelConfigFromDir(dir)).toMatchObject({ honorsEnableThinking: true,
      supportsInstructMode: true, supportedReasoningEfforts: ['low', 'high', 'max'] })
  })
  it.each([
    { encoder: 'other_encoder', modes: ['chat', 'thinking'] },
    { encoder: 'encoding_dsv4', modes: ['thinking'] },
  ])('does not invent an unsupported native contract: %j', ({ encoder, modes }) => {
    const dir = bundle('deepseek_v4', {})
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify({ chat: {
      encoder, reasoning: { supported: true, modes },
    } }))
    expect(detectModelConfigFromDir(dir).honorsEnableThinking).toBe(false)
  })
  it('preserves genuine template support', () => {
    expect(detectModelConfigFromDir(bundle('qwen2', { chat_template: '{% if enable_thinking %}think{% endif %}' })).honorsEnableThinking).toBe(true)
  })
  it('preserves a genuine template without a toggle', () => {
    expect(detectModelConfigFromDir(bundle('qwen2', {}, '{{ messages }}')).honorsEnableThinking).toBe(false)
  })
  it('does not enable an explicitly unsupported family on missing template', () => {
    expect(detectModelConfigFromDir(bundle('openpangu_v2', {})).honorsEnableThinking).toBe(false)
  })
})
