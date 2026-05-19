import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import {
  buildNewChatInheritedOverrides,
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
