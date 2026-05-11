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
  detectedFamily?: string
}

export function isThinkingBlockedForModel(modelPath?: string, detectedFamily?: string): boolean {
  const family = String(detectedFamily || '').toLowerCase().replace(/_/g, '-')
  const name = String(modelPath || '').toLowerCase()
  if (family !== 'zaya' && family !== 'zaya1-vl' && !name.includes('zaya')) return false
  return name.includes('jangtq2')
}

function basename(path?: string): string {
  if (!path) return 'unknown model'
  return path.replace(/\/+$/, '').split('/').pop() || path
}

function samePath(a?: string, b?: string): boolean {
  if (!a || !b) return true
  return a.replace(/\/+$/, '') === b.replace(/\/+$/, '')
}

function parserUsesEffortLevels(parser?: string, detectedFamily?: string): boolean {
  if (detectedFamily === 'hy3') return true
  return parser === 'openai_gptoss' || parser === 'mistral'
}

function parserAcceptsEffort(effort: string, parser?: string, detectedFamily?: string): boolean {
  if (detectedFamily === 'hy3') {
    return effort === 'low' || effort === 'high'
  }
  if (parser === 'mistral') {
    return effort === 'high'
  }
  return true
}

export function buildChatSettingsCompatibilityWarnings(input: ChatSettingsCompatibilityInput): string[] {
  const { messageCount, savedChatModelPath, currentModelPath, overrides, reasoningParser, toolParser, detectedFamily } = input
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
  if (overrides.enableThinking === true && isThinkingBlockedForModel(currentModelPath, detectedFamily)) {
    warnings.push('Saved Thinking On is blocked for ZAYA JANGTQ2 because that experimental 2-bit tier fails strict chat coherency.')
  }

  if (overrides.reasoningEffort) {
    if (!reasoningParser) {
      warnings.push(
        `Saved reasoning effort "${overrides.reasoningEffort}" cannot take effect because this model has no detected reasoning parser.`,
      )
    } else if (!parserAcceptsEffort(overrides.reasoningEffort, reasoningParser, detectedFamily)) {
      const modelName = detectedFamily === 'hy3' ? 'Hy3' : reasoningParser === 'mistral' ? 'Mistral' : reasoningParser
      warnings.push(`Saved reasoning effort "${overrides.reasoningEffort}" is not supported by ${modelName}. Use Auto or High.`)
    } else if (!parserUsesEffortLevels(reasoningParser, detectedFamily)) {
      warnings.push(`Saved reasoning effort "${overrides.reasoningEffort}" is not used by ${reasoningParser}. Reset the chat setting or switch to Auto.`)
    }
  }

  if (overrides.builtinToolsEnabled === true && !toolParser) {
    warnings.push('Built-in tools are enabled, but this model has no detected tool parser. Tool calls may not round-trip.')
  }

  return warnings
}
