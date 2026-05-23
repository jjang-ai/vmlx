import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import {
  buildNewChatInheritedOverrides,
  sanitizeChatOverrides,
  type ChatOverridePolicyInput,
} from '../src/main/chat-override-policy'

const baseExisting: ChatOverridePolicyInput = {
  chatId: 'new-chat',
  temperature: 0.6,
  topP: 0.8,
  topK: 20,
  minP: 0.05,
  maxTokens: 4096,
  repeatPenalty: 1.05,
  enableThinking: undefined,
}

describe('new-chat override inheritance policy', () => {
  it('inherits coding/tool settings from the previous chat without making sampling, reasoning, or prompt text sticky', () => {
    const inherited = buildNewChatInheritedOverrides(baseExisting, {
      chatId: 'old-chat',
      temperature: 1.0,
      topP: 1.0,
      topK: 1,
      minP: 0.2,
      maxTokens: 128,
      maxThinkingTokens: 4096,
      repeatPenalty: 1.3,
      stopSequences: '<bad>',
      wireApi: 'completions',
      builtinToolsEnabled: true,
      shellEnabled: true,
      webSearchEnabled: true,
      braveSearchEnabled: false,
      fetchUrlEnabled: true,
      fileToolsEnabled: false,
      searchToolsEnabled: true,
      gitEnabled: false,
      utilityToolsEnabled: true,
      maxToolIterations: 6,
      workingDirectory: '/Users/example/project',
      hideToolStatus: true,
      toolResultMaxChars: 16000,
      systemPrompt: 'sticky prompt text that should stay chat-scoped',
      enableThinking: true,
      reasoningEffort: 'max',
    })

    expect(inherited).toMatchObject({
      chatId: 'new-chat',
      temperature: 0.6,
      topP: 0.8,
      topK: 20,
      minP: 0.05,
      maxTokens: 4096,
      repeatPenalty: 1.05,
      builtinToolsEnabled: true,
      shellEnabled: true,
      webSearchEnabled: true,
      braveSearchEnabled: false,
      fetchUrlEnabled: true,
      fileToolsEnabled: false,
      searchToolsEnabled: true,
      gitEnabled: false,
      utilityToolsEnabled: true,
      maxToolIterations: 6,
      workingDirectory: '/Users/example/project',
      hideToolStatus: true,
      toolResultMaxChars: 16000,
    })
    expect(inherited.stopSequences).toBeUndefined()
    expect(inherited.wireApi).toBeUndefined()
    expect(inherited.enableThinking).toBeUndefined()
    expect(inherited.reasoningEffort).toBeUndefined()
    expect(inherited.systemPrompt).toBeUndefined()
  })

  it('treats auto-applied default profiles as tool presets, not hidden sampler or reasoning presets', () => {
    const inherited = buildNewChatInheritedOverrides(baseExisting, {
      chatId: 'profile-default',
      temperature: 2.0,
      topP: 0.1,
      topK: 1,
      minP: 0.9,
      maxTokens: 128,
      repeatPenalty: 2.0,
      systemPrompt: 'sticky profile prompt',
      stopSequences: '<stop>',
      wireApi: 'completions',
      enableThinking: true,
      reasoningEffort: 'max',
      builtinToolsEnabled: true,
      shellEnabled: true,
      fileToolsEnabled: true,
      workingDirectory: '/Users/example/code',
    })

    expect(inherited).toMatchObject({
      chatId: 'new-chat',
      temperature: 0.6,
      topP: 0.8,
      topK: 20,
      minP: 0.05,
      maxTokens: 4096,
      repeatPenalty: 1.05,
      builtinToolsEnabled: true,
      shellEnabled: true,
      fileToolsEnabled: true,
      workingDirectory: '/Users/example/code',
    })
    expect(inherited.systemPrompt).toBeUndefined()
    expect(inherited.stopSequences).toBeUndefined()
    expect(inherited.wireApi).toBeUndefined()
    expect(inherited.enableThinking).toBeUndefined()
    expect(inherited.reasoningEffort).toBeUndefined()
  })

  it('default profiles cannot make maxTokens sticky on clean new chats', () => {
    const inherited = buildNewChatInheritedOverrides({ chatId: 'new-chat' }, {
      chatId: 'profile-default',
      maxTokens: 32768,
      maxThinkingTokens: 32768,
      temperature: 2.0,
      repeatPenalty: 2.0,
      systemPrompt: 'sticky prompt text that should stay chat-scoped',
      builtinToolsEnabled: true,
      shellEnabled: true,
      workingDirectory: '/Users/example/code',
    })

    expect(inherited).toMatchObject({
      chatId: 'new-chat',
      builtinToolsEnabled: true,
      shellEnabled: true,
      workingDirectory: '/Users/example/code',
    })
    expect(inherited.maxTokens).toBeUndefined()
    expect(inherited.maxThinkingTokens).toBeUndefined()
    expect(inherited.temperature).toBeUndefined()
    expect(inherited.repeatPenalty).toBeUndefined()
    expect(inherited.systemPrompt).toBeUndefined()
  })

  it('new chats preserve model-owned maxTokens while refusing inherited output caps', () => {
    const inherited = buildNewChatInheritedOverrides({
      chatId: 'new-chat',
      maxTokens: 4096,
      temperature: 0.9,
    }, {
      chatId: 'previous-chat',
      maxTokens: 32768,
      temperature: 2.0,
      repeatPenalty: 2.0,
      systemPrompt: 'sticky prompt text that should stay chat-scoped',
      builtinToolsEnabled: true,
      shellEnabled: true,
      workingDirectory: '/Users/example/code',
    })

    expect(inherited).toMatchObject({
      chatId: 'new-chat',
      maxTokens: 4096,
      temperature: 0.9,
      builtinToolsEnabled: true,
      shellEnabled: true,
      workingDirectory: '/Users/example/code',
    })
    expect(inherited.maxTokens).not.toBe(32768)
    expect(inherited.repeatPenalty).toBeUndefined()
    expect(inherited.systemPrompt).toBeUndefined()
  })

  it('does not overwrite derived model defaults with undefined inherited tool values', () => {
    const inherited = buildNewChatInheritedOverrides(baseExisting, {
      chatId: 'old-chat',
      builtinToolsEnabled: undefined,
      shellEnabled: undefined,
      maxToolIterations: undefined,
      workingDirectory: undefined,
    })

    expect(inherited).toEqual(baseExisting)
  })

  it('does not let chat:setOverrides rewrite global model generation or reasoning defaults', () => {
    const chatIpcSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/ipc/chat.ts'),
      'utf8',
    )
    const setOverridesHandler = chatIpcSource.slice(
      chatIpcSource.indexOf('"chat:setOverrides"'),
      chatIpcSource.indexOf('ipcMain.handle("chat:getOverrides"'),
    )

    expect(setOverridesHandler).not.toContain('db.saveModelSettings')
    expect(setOverridesHandler).not.toContain('reasoning_mode')
    expect(setOverridesHandler).not.toContain('Synced')
  })

  it('chat:setOverrides treats maxTokens 0 or lower as Auto instead of a one-token cap', () => {
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 0 }).maxTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: -20 }).maxTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 512 }).maxTokens).toBe(512)
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 2_000_000 }).maxTokens).toBe(1_000_000)
  })

  it('chat:setOverrides treats maxThinkingTokens as an explicit thinking budget, not an output cap', () => {
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: 0 }).maxThinkingTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: -20 }).maxThinkingTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: Number.NaN }).maxThinkingTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: '4096' as any }).maxThinkingTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: 4096.9 }).maxThinkingTokens).toBe(4096)
    expect(sanitizeChatOverrides({ chatId: 'chat', maxThinkingTokens: 2_000_000 }).maxThinkingTokens).toBe(1_000_000)

    const sanitized = sanitizeChatOverrides({
      chatId: 'chat',
      maxTokens: 512,
      maxThinkingTokens: 4096,
    })
    expect(sanitized.maxTokens).toBe(512)
    expect(sanitized.maxThinkingTokens).toBe(4096)
  })

  it('chat:setOverrides rejects non-finite or non-numeric maxTokens instead of poisoning server defaults', () => {
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: Number.NaN }).maxTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: Number.POSITIVE_INFINITY }).maxTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: '1024' as any }).maxTokens).toBeUndefined()
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 512.9 }).maxTokens).toBe(512)
  })

  it('chat maxTokens save path cannot mutate session startup maxTokens', () => {
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 8192 }).maxTokens).toBe(8192)

    const chatIpcSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/ipc/chat.ts'),
      'utf8',
    )
    const setOverridesHandler = chatIpcSource.slice(
      chatIpcSource.indexOf('"chat:setOverrides"'),
      chatIpcSource.indexOf('ipcMain.handle("chat:getOverrides"'),
    )

    expect(setOverridesHandler).toContain('sanitizeChatOverrides')
    expect(setOverridesHandler).toContain('db.setChatOverrides')
    expect(setOverridesHandler).not.toContain('sessionManager')
    expect(setOverridesHandler).not.toContain('saveModelSettings')
    expect(setOverridesHandler).not.toContain('saveSession')
    expect(setOverridesHandler).not.toContain('model_settings')

    const sessionsSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/sessions.ts'),
      'utf8',
    )
    const performanceLaunchBlock = sessionsSource.slice(
      sessionsSource.indexOf('// Performance'),
      sessionsSource.indexOf('// Tool integration'),
    )

    expect(performanceLaunchBlock).toContain('const maxTokens = finitePositiveInteger(config.maxTokens)')
    expect(performanceLaunchBlock).toContain("args.push('--max-tokens', maxTokens.toString())")
    expect(performanceLaunchBlock).toContain('const maxContextLength = finitePositiveInteger(config.maxContextLength)')
    expect(performanceLaunchBlock).toContain("args.push('--max-prompt-tokens', maxContextLength.toString())")
    expect(performanceLaunchBlock).not.toContain('overrides.maxTokens')
    expect(performanceLaunchBlock).not.toContain('chatOverrides')
    expect(performanceLaunchBlock).not.toContain('getChatOverrides')

    const modelSettingsSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/db/model-settings.ts'),
      'utf8',
    )
    expect(modelSettingsSource).toContain('choices are per-chat/API overrides and must not be stored per model')
    expect(modelSettingsSource).not.toContain('maxTokens')
    expect(modelSettingsSource).not.toContain('max_tokens')
    expect(modelSettingsSource).not.toContain('reasoning_mode')
  })

  it('persisted chat maxTokens cannot relaunch server with a new startup maxTokens', () => {
    const databaseSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/database.ts'),
      'utf8',
    )
    const setChatOverridesBlock = databaseSource.slice(
      databaseSource.indexOf('setChatOverrides(overrides'),
      databaseSource.indexOf('getChatOverrides(chatId'),
    )
    const saveModelSettingsBlock = databaseSource.slice(
      databaseSource.indexOf('saveModelSettings(modelPath'),
      databaseSource.indexOf('deleteModelSettings(modelPath'),
    )
    const sessionsSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/sessions.ts'),
      'utf8',
    )
    const performanceLaunchBlock = sessionsSource.slice(
      sessionsSource.indexOf('// Performance'),
      sessionsSource.indexOf('// Tool integration'),
    )

    expect(setChatOverridesBlock).toContain('INSERT OR REPLACE INTO chat_overrides')
    expect(setChatOverridesBlock).toContain('max_tokens')
    expect(setChatOverridesBlock).toContain('max_thinking_tokens')
    expect(setChatOverridesBlock).toContain('overrides.maxTokens')
    expect(setChatOverridesBlock).toContain('overrides.maxThinkingTokens')
    expect(setChatOverridesBlock).not.toContain('sessions')
    expect(setChatOverridesBlock).not.toContain('model_settings')
    expect(setChatOverridesBlock).not.toContain('saveModelSettings')
    expect(setChatOverridesBlock).not.toContain('saveSession')

    expect(saveModelSettingsBlock).toContain('max_tokens')
    expect(saveModelSettingsBlock).toContain('null')
    expect(saveModelSettingsBlock).not.toContain('settings.maxTokens')
    expect(saveModelSettingsBlock).not.toContain('settings.max_tokens')

    expect(performanceLaunchBlock).toContain('finitePositiveInteger(config.maxTokens)')
    expect(performanceLaunchBlock).toContain("args.push('--max-tokens', maxTokens.toString())")
    expect(performanceLaunchBlock).not.toContain('getChatOverrides')
    expect(performanceLaunchBlock).not.toContain('chat_overrides')
    expect(performanceLaunchBlock).not.toContain('overrides.maxTokens')
    expect(performanceLaunchBlock).not.toContain('overrides.maxThinkingTokens')
  })

  it('server startup maxTokens and chat maxTokens remain independent when both are set', () => {
    expect(sanitizeChatOverrides({ chatId: 'chat', maxTokens: 8192 }).maxTokens).toBe(8192)

    const sessionsSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/sessions.ts'),
      'utf8',
    )
    const performanceLaunchBlock = sessionsSource.slice(
      sessionsSource.indexOf('// Performance'),
      sessionsSource.indexOf('// Tool integration'),
    )

    expect(performanceLaunchBlock).toContain('const maxTokens = finitePositiveInteger(config.maxTokens)')
    expect(performanceLaunchBlock).toContain("args.push('--max-tokens', maxTokens.toString())")
    expect(performanceLaunchBlock).not.toContain('overrides.maxTokens')
    expect(performanceLaunchBlock).not.toContain('chatOverrides')

    const chatIpcSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/ipc/chat.ts'),
      'utf8',
    )
    const requestBuildBlock = chatIpcSource.slice(
      chatIpcSource.indexOf('const resolvedOutputBudget = dsv4OutputBudget('),
      chatIpcSource.indexOf('if (messageContent.length === 0)'),
    )

    expect(requestBuildBlock).toContain('overrides?.maxTokens')
    expect(requestBuildBlock).toContain('overrides?.maxThinkingTokens')
    expect(requestBuildBlock).toContain('max_output_tokens: resolvedOutputBudget')
    expect(requestBuildBlock).toContain('max_tokens: resolvedOutputBudget')
    expect(requestBuildBlock).toContain('max_thinking_tokens')
    expect(requestBuildBlock).toContain('thinking_budget')
    expect(requestBuildBlock).not.toContain('config.maxTokens')
    expect(requestBuildBlock).not.toContain('config.maxThinkingTokens')
    expect(requestBuildBlock).not.toContain('session.config.maxTokens')
  })

  it('chat:setOverrides rejects malformed sampler and tool numeric overrides instead of forcing hidden values', () => {
    const sanitized = sanitizeChatOverrides({
      chatId: 'chat',
      temperature: Number.NaN,
      topP: Number.POSITIVE_INFINITY,
      topK: '40' as any,
      minP: Number.NEGATIVE_INFINITY,
      repeatPenalty: '1.1' as any,
      maxToolIterations: Number.NaN,
      toolResultMaxChars: Number.POSITIVE_INFINITY,
    })

    expect(sanitized.temperature).toBeUndefined()
    expect(sanitized.topP).toBeUndefined()
    expect(sanitized.topK).toBeUndefined()
    expect(sanitized.minP).toBeUndefined()
    expect(sanitized.repeatPenalty).toBeUndefined()
    expect(sanitized.maxToolIterations).toBeUndefined()
    expect(sanitized.toolResultMaxChars).toBeUndefined()
  })

  it('wires starred default profiles through the tool-only new-chat inheritance policy', () => {
    const chatIpcSource = fs.readFileSync(
      path.resolve(__dirname, '../src/main/ipc/chat.ts'),
      'utf8',
    )
    const createHandler = chatIpcSource.slice(
      chatIpcSource.indexOf('"chat:create"'),
      chatIpcSource.indexOf('ipcMain.handle("chat:getByModel"'),
    )
    const defaultProfileBranch = createHandler.slice(
      createHandler.indexOf('if (defaultProfile)'),
      createHandler.indexOf('} else if (modelPath)'),
    )

    expect(defaultProfileBranch).toContain('buildNewChatInheritedOverrides')
    expect(defaultProfileBranch).not.toContain('Object.entries(defaultProfile)')
  })
})
