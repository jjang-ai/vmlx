export interface ChatSettingsCompatibilityOverrides {
  enableThinking?: boolean
  reasoningEffort?: 'low' | 'medium' | 'high' | 'max'
  builtinToolsEnabled?: boolean
}

export interface ChatSettingsCompatibilityInput {
  messageCount: number
  savedChatModelPath?: string
  currentModelPath?: string
  overrides: ChatSettingsCompatibilityOverrides
  reasoningParser?: string
  toolParser?: string
}

function basename(path?: string): string {
  if (!path) return 'unknown model'
  return path.replace(/\/+$/, '').split('/').pop() || path
}

function samePath(a?: string, b?: string): boolean {
  if (!a || !b) return true
  return a.replace(/\/+$/, '') === b.replace(/\/+$/, '')
}

function parserUsesEffortLevels(parser?: string): boolean {
  return parser === 'openai_gptoss' || parser === 'mistral'
}

export function buildChatSettingsCompatibilityWarnings(input: ChatSettingsCompatibilityInput): string[] {
  const { messageCount, savedChatModelPath, currentModelPath, overrides, reasoningParser, toolParser } = input
  if (messageCount <= 0) return []

  const warnings: string[] = []

  if (!samePath(savedChatModelPath, currentModelPath)) {
    warnings.push(
      `This chat was started on ${basename(savedChatModelPath)} but is now attached to ${basename(currentModelPath)}. Review saved per-chat settings before continuing.`,
    )
  }

  if (overrides.enableThinking === true && !reasoningParser) {
    warnings.push('Saved Thinking On cannot take effect because this model has no detected reasoning parser.')
  }

  if (overrides.reasoningEffort) {
    if (!reasoningParser) {
      warnings.push(
        `Saved reasoning effort "${overrides.reasoningEffort}" cannot take effect because this model has no detected reasoning parser.`,
      )
    } else if (reasoningParser === 'mistral' && overrides.reasoningEffort !== 'high') {
      warnings.push(`Saved reasoning effort "${overrides.reasoningEffort}" is not supported by Mistral. Use Auto or High.`)
    } else if (!parserUsesEffortLevels(reasoningParser)) {
      warnings.push(`Saved reasoning effort "${overrides.reasoningEffort}" is not used by ${reasoningParser}. Reset the chat setting or switch to Auto.`)
    }
  }

  if (overrides.builtinToolsEnabled === true && !toolParser) {
    warnings.push('Built-in tools are enabled, but this model has no detected tool parser. Tool calls may not round-trip.')
  }

  return warnings
}
