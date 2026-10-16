import { describe, expect, it } from 'vitest'
import { mkdtempSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'

describe('Nemotron-3.5 stamped reasoning modes', () => {
  it('surfaces medium + high (default high) from capabilities.reasoning_modes', () => {
    const dir = mkdtempSync(join(tmpdir(), 'nemo-modes-'))
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'nemotron_h', hybrid_override_pattern: 'M*E' }))
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify({
      format: 'jang', capabilities: {
        family: 'nemotron_h', reasoning_parser: 'deepseek_r1', tool_parser: 'qwen3_coder', supports_thinking: true,
        reasoning_modes: { on: { enable_thinking: true }, medium: { enable_thinking: true, medium_effort: true }, off: { enable_thinking: false } },
      },
    }))
    const cfg = detectModelConfigFromDir(dir)
    expect(cfg.supportedReasoningEfforts).toEqual(['medium', 'high'])
    expect(cfg.defaultReasoningEffort).toBe('high')
    expect(cfg.toolParser).toBe('qwen3_coder')
  })
})
