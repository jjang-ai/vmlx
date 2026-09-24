import { describe, expect, it } from 'vitest'
import { applyReasoningRequestFields } from '../src/shared/reasoningEffortPolicy'
import { sanitizeChatOverrides } from '../src/main/chat-override-policy'
import { buildChatSettingsResetOverrides } from '../src/shared/chatSettingsResetPolicy'

describe('native adaptive chat setting', () => {
  it.each(['completions', 'responses'] as const)('persists explicit intent into every %s tool continuation', wireApi => {
    const saved = sanitizeChatOverrides({ chatId: 'chat', thinkingMode: 'adaptive' as const })
    for (let turn = 0; turn < 3; turn++) {
      const body: Record<string, any> = {}
      applyReasoningRequestFields(body, { ...saved, isRemote: false, sessionHasReasoningParser: true, supportsAdaptiveThinking: true, wireApi })
      expect(body).toEqual({ thinking_mode: 'adaptive', chat_template_kwargs: { thinking_mode: 'adaptive' } })
    }
  })
  it('distinguishes explicit adaptive from inherited Auto', () => {
    const body = {}
    applyReasoningRequestFields(body, { isRemote: false, sessionHasReasoningParser: true, supportsAdaptiveThinking: true })
    expect(body).toEqual({})
    expect(buildChatSettingsResetOverrides({ thinkingMode: 'adaptive', workingDirectory: '/tmp' })).toEqual({ workingDirectory: '/tmp' })
  })
  it.each([false, undefined])('rejects stale adaptive setting without advertised support=%s', supportsAdaptiveThinking => {
    const body = {}
    expect(() => applyReasoningRequestFields(body, { thinkingMode: 'adaptive', isRemote: false, sessionHasReasoningParser: true, supportsAdaptiveThinking })).toThrow('does not advertise')
    expect(body).toEqual({})
  })
  it.each([true, false])('rejects contradictory boolean=%s', enableThinking => {
    expect(() => sanitizeChatOverrides({ chatId: 'chat', thinkingMode: 'adaptive', enableThinking })).toThrow('conflicts')
  })
  it('does not send vMLX adaptive extensions to generic remote providers', () => {
    expect(() => applyReasoningRequestFields({}, { thinkingMode: 'adaptive', isRemote: true, sessionHasReasoningParser: false, supportsAdaptiveThinking: true, remoteReasoningFormat: 'openai' })).toThrow('does not advertise')
    const body = {}
    applyReasoningRequestFields(body, { thinkingMode: 'adaptive', isRemote: true, sessionHasReasoningParser: false, supportsAdaptiveThinking: true, remoteReasoningFormat: 'vmlx' })
    expect(body).toHaveProperty('thinking_mode', 'adaptive')
  })
})

describe('adaptive capability discovery', () => {
  it('uses explicit remote capability rather than model names', async () => {
    const { detectedConfigFromRemoteCapabilities } = await import('../src/shared/remoteModelCapabilities')
    expect(detectedConfigFromRemoteCapabilities({ family: 'minimax_m3', supports_thinking: true, native_thinking_modes: ['enabled', 'disabled', 'adaptive'] })?.supportsAdaptiveThinking).toBe(true)
    expect(detectedConfigFromRemoteCapabilities({ family: 'minimax_m3', supports_thinking: true, native_thinking_modes: [] })?.supportsAdaptiveThinking).toBe(false)
    expect(detectedConfigFromRemoteCapabilities({ family: 'minimax_m3', supports_thinking: true })?.supportsAdaptiveThinking).toBeUndefined()
  })
  it('derives local support from a renamed bundle architecture', async () => {
    const fs = await import('node:fs')
    const os = await import('node:os')
    const path = await import('node:path')
    const { detectModelConfigFromDir } = await import('../src/main/model-config-registry')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'adaptive-renamed-'))
    try {
      fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify({model_type:'minimax_m3_vl'}))
      fs.writeFileSync(path.join(dir, 'chat_template.jinja'), '{% if thinking_mode == "enabled" %}think{% elif thinking_mode == "adaptive" %}decide{% endif %}')
      expect(detectModelConfigFromDir(dir).supportsAdaptiveThinking).toBe(true)
      expect(detectModelConfigFromDir(dir).honorsEnableThinking).toBe(true)
    } finally { fs.rmSync(dir, {recursive:true,force:true}) }
  })
})
