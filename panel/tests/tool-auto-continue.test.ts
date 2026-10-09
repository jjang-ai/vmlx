import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  applyPostToolRequestFields,
  captureToolRequestFields,
  isToolAuthorizedForCurrentTurn,
  isToolNameProvidedForCurrentTurn,
  noVisibleToolAnswerWarning,
  requiredToolChoiceNamesForCurrentTurn,
  requestedExactFinalToolNames,
  requestedOnceToolNames,
  requestedScopedToolNames,
  repeatsOnlyCompletedExactlyOnceTools,
  replaySafeToolCallKey,
  requestsBoundedFinalAnswerAfterToolResult,
  requestsExactTextOnlyWithoutToolUse,
  requestsNoToolCalls,
  requestsPrivateReasoningWithoutToolUse,
  scopeToolDefinitionsByName,
  shouldAutoContinueAfterToolUse,
  shouldFinishZayaAppleScriptToolRound,
  shouldPlanCompletedToolAnswerPass,
  toolChoiceForCurrentTurn,
  unavailableRequestedToolNames,
} from '../src/shared/toolAutoContinue'

describe('no-visible tool answer diagnostic', () => {
  it('reports the actual ten-iteration ceiling without inventing recovery or tool success', () => {
    expect(noVisibleToolAnswerWarning({
      toolIteration: 10,
      maxToolIterations: 10,
      recoveryAttempted: false,
    })).toBe('The tool loop reached its iteration limit (10) without a visible answer. No direct-answer recovery was attempted.')
  })

  it('reports attempted recovery without claiming it completed or ran exactly once', () => {
    expect(noVisibleToolAnswerWarning({
      toolIteration: 1,
      maxToolIterations: 10,
      recoveryAttempted: true,
    })).toBe('The tool loop ended without a visible answer after a direct-answer recovery was attempted.')
  })

  it('distinguishes early termination without recovery from the ceiling', () => {
    expect(noVisibleToolAnswerWarning({
      toolIteration: 1,
      maxToolIterations: 10,
      recoveryAttempted: false,
    })).toBe('The tool loop ended without a visible answer before reaching its iteration limit. No direct-answer recovery was attempted.')
  })

  it('preserves both facts when recovery was attempted and a custom limit was reached', () => {
    expect(noVisibleToolAnswerWarning({
      toolIteration: 3,
      maxToolIterations: 3,
      recoveryAttempted: true,
    })).toBe('The tool loop reached its iteration limit (3) without a visible answer. A direct-answer recovery was attempted.')
  })

  it('wires observed loop and recovery state into the existing warning/status path', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const diagnostic = source.indexOf('const noVisibleAnswerWarning = noVisibleToolAnswerWarning({')
    expect(diagnostic).toBeGreaterThan(-1)
    expect(source.slice(diagnostic, diagnostic + 240)).toContain('maxToolIterations: MAX_TOOL_ITERATIONS')
    expect(source.slice(diagnostic, diagnostic + 240)).toContain('recoveryAttempted: finalAnswerRecovery')
    expect(source).not.toContain('after one direct-answer recovery.')
    expect(source.slice(diagnostic, diagnostic + 550)).toContain('new Set([...(responseWarnings || []), noVisibleAnswerWarning])')
    expect(source.slice(diagnostic, diagnostic + 550)).toContain('noVisibleAnswerWarning,')
  })
})

describe('tool auto-continue policy', () => {
  it('keeps every completed exactly-once continuation on the cache-stable render', () => {
    expect(
      shouldPlanCompletedToolAnswerPass({
        directAnswerAfterSingleTool: false,
        exactFinalToolCount: 0,
        exactFinalToolsComplete: false,
        exactlyOnceToolCount: 1,
        exactlyOnceToolsComplete: true,
      }),
    ).toBe(true)

    expect(
      shouldPlanCompletedToolAnswerPass({
        directAnswerAfterSingleTool: false,
        exactFinalToolCount: 0,
        exactFinalToolsComplete: false,
        exactlyOnceToolCount: 1,
        exactlyOnceToolsComplete: false,
      }),
    ).toBe(false)
  })

  it('keeps the local fulfilled-tool render stable and reserves schema removal for recovery', () => {
    const tool = {
      type: 'function',
      name: 'file_info',
      parameters: { type: 'object' },
    }
    const choice = { type: 'function', name: 'file_info' }
    const previous = captureToolRequestFields({ tools: [tool], tool_choice: choice })

    const localPlanned: Record<string, unknown> = { tools: [] }
    applyPostToolRequestFields(localPlanned, {
      finalAnswerRecovery: false,
      plannedDirectAnswerPass: true,
      isRemote: false,
      previous,
    })
    expect(localPlanned).toEqual({ tools: [tool], tool_choice: choice })

    const remotePlanned: Record<string, unknown> = { tools: [] }
    applyPostToolRequestFields(remotePlanned, {
      finalAnswerRecovery: false,
      plannedDirectAnswerPass: true,
      isRemote: true,
      previous,
    })
    expect(remotePlanned).toEqual({ tools: [tool] })

    applyPostToolRequestFields(localPlanned, {
      finalAnswerRecovery: true,
      plannedDirectAnswerPass: true,
      isRemote: false,
      previous,
    })
    expect(localPlanned).toEqual({ tool_choice: 'none' })
  })

  it('continues when a model stops after tools with no visible response', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: '',
        iterationTokenCount: 0,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('continues short content only when the model hit the length limit', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'partial sentence',
        iterationTokenCount: 4,
        finishReason: 'length',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('does not duplicate a short normal final answer after tool results', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'Done after tools.',
        iterationTokenCount: 4,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(false)
  })

  it('routes a replay-only exactly-once pass to bounded answer recovery', () => {
    const completed = new Set(['get_current_datetime'])
    expect(
      repeatsOnlyCompletedExactlyOnceTools(
        ['GET_CURRENT_DATETIME'],
        ['get_current_datetime'],
        completed,
      ),
    ).toBe(true)
    expect(
      repeatsOnlyCompletedExactlyOnceTools(
        ['get_current_datetime', 'file_info'],
        ['get_current_datetime'],
        completed,
      ),
    ).toBe(false)
    expect(
      repeatsOnlyCompletedExactlyOnceTools(
        ['file_info'],
        ['get_current_datetime'],
        completed,
      ),
    ).toBe(false)
  })

  it('finishes the specialized ZAYA AppleScript bundle after its native action result', () => {
    expect(
      shouldFinishZayaAppleScriptToolRound(true, ['run_applescript']),
    ).toBe(true)
    expect(
      shouldFinishZayaAppleScriptToolRound(false, ['run_applescript']),
    ).toBe(false)
    expect(
      shouldFinishZayaAppleScriptToolRound(true, ['run_applescript', 'read_file']),
    ).toBe(false)
    expect(shouldFinishZayaAppleScriptToolRound(true, [])).toBe(false)
  })

  it('deduplicates only consecutive identical replay-safe reads', () => {
    const first = replaySafeToolCallKey('read_file', {
      path: 'README.md',
      line_start: 1,
      line_end: 20,
    })
    const reordered = replaySafeToolCallKey('READ_FILE', {
      line_end: 20,
      path: 'README.md',
      line_start: 1,
    })

    expect(first).toBe(reordered)
    expect(
      replaySafeToolCallKey('read_file', {
        path: 'README.md',
        line_start: 2,
        line_end: 20,
      }),
    ).not.toBe(first)
    expect(replaySafeToolCallKey('write_file', { path: 'README.md' })).toBeUndefined()
    expect(replaySafeToolCallKey('run_command', { command: 'pwd' })).toBeUndefined()
    expect(replaySafeToolCallKey('fetch_url', { url: 'https://example.com' })).toBeUndefined()
  })

  it('wires replay-safe read dedupe before builtin execution and breaks on errors', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const key = source.indexOf('const replaySafeKey = toolAuthorized')
    const dedupe = source.indexOf('lastReplaySafeToolResultKey === replaySafeKey', key)
    const reset = source.indexOf('lastReplaySafeToolResultKey = null;', dedupe)
    const execute = source.indexOf('const result = await executeBuiltinTool(', dedupe)
    const retain = source.indexOf('if (replaySafeKey && !result.is_error)', execute)

    expect(key).toBeGreaterThan(-1)
    expect(dedupe).toBeGreaterThan(key)
    expect(reset).toBeGreaterThan(dedupe)
    expect(execute).toBeGreaterThan(reset)
    expect(retain).toBeGreaterThan(execute)
    expect(source).toContain(
      '[CHAT] Deduplicated consecutive replay-safe tool call: ${tc.function.name}',
    )
    expect(source).toContain(
      '// Invalid arguments cannot be compared safely and therefore',
    )
    expect(source).toContain(
      '// A rejected tool call is still an intervening action in the',
    )
  })

  it('recognizes explicit exact-final tool scopes', () => {
    expect(
      requestedExactFinalToolNames(
        'Call file_info exactly once. After the tool result, reply exactly DONE and nothing else.',
      ),
    ).toEqual(['file_info'])

    expect(
      requestedExactFinalToolNames(
        'Use the built-in file_info tool exactly once on panel/package.json. Think briefly, then your visible answer must be exactly two lines: DONE and RESULT.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames(
        'Use exactly one file_info tool call to inspect generation_config.json.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames('Do not use file_info exactly once. Answer directly.'),
    ).toEqual([])
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use exactly one file_info tool call. After the tool result, answer briefly.',
        ['file_info'],
      ),
    ).toBe(true)
    expect(
      requestedExactFinalToolNames(
        'Continue this same chat. Call the built-in file_info tool exactly once with path pyproject.toml. After the real tool result, reply exactly B1-NONE-MT2-DONE and nothing else.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedExactFinalToolNames(
        'Call the built-in file_info tool exactly once with path panel/package.json. After its result, reply exactly B1-ELECTRON-TOOL-TEMPLATE1-DONE and nothing else.',
      ),
    ).toEqual(['file_info'])
    const liveLfmContract =
      'Call the built-in file_info tool exactly once with path panel/package.json. Use the real tool. Then reply exactly two visible lines: R18-LFM-EPOCH-ACCEPT-1-DONE and FILE=panel/package.json SIZE=<human-readable tool size>. Do not add other visible text.'
    expect(requestedExactFinalToolNames(liveLfmContract)).toEqual(['file_info'])
    const scoped = scopeToolDefinitionsByName(
      [
        { function: { name: 'file_info' } },
        { function: { name: 'write_file' } },
      ],
      requestedExactFinalToolNames(liveLfmContract),
    )
    const scopedNames = scoped.map(tool => tool.function.name)
    expect(scopedNames).toEqual(['file_info'])
    expect(isToolNameProvidedForCurrentTurn('file_info', scopedNames)).toBe(true)
    expect(isToolNameProvidedForCurrentTurn('write_file', scopedNames)).toBe(false)
    expect(
      requestedExactFinalToolNames(
        'First call file_info exactly once with path panel/package.json. After that result, call run_command exactly once with command pwd. Use exactly those two tools in that order. After both tool results, reply exactly B1-CURRENT-MULTI3-DONE.',
      ),
    ).toEqual(['file_info', 'run_command'])
    expect(
      requestedExactFinalToolNames(
        'Call file_info exactly once. Use tools as needed, then reply exactly DONE.',
      ),
    ).toEqual([])
    expect(
      requestedExactFinalToolNames(
        'Call file_info exactly once. You may call other tools, then reply exactly DONE.',
      ),
    ).toEqual([])
    expect(
      requestedExactFinalToolNames(
        'Call file_info exactly once with path panel/package.json. Call read_file with path AGENTS.md. Then reply exactly DONE.',
        ['write_file', 'read_file', 'file_info'],
      ),
    ).toEqual(['file_info', 'read_file'])
  })

  it('retires explicitly named exactly-once tools even without exact final wording', () => {
    expect(
      requestedOnceToolNames(
        'Call read_file with path README.md exactly once. After the tool result, reply on one line.',
      ),
    ).toEqual(['read_file'])
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Call read_file with path README.md exactly once. After the tool result, reply on one line.',
        ['read_file'],
      ),
    ).toBe(true)
    expect(
      requestedOnceToolNames(
        'Call read_file with path README.md. Then call file_info exactly once.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames(
        'Call read_file with path README.md and then call file_info exactly once.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames(
        'Call the built-in file_info tool exactly once with path panel/package.json. After the real tool result, report the human-readable size.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedExactFinalToolNames(
        'Call the built-in file_info tool exactly once with path panel/package.json. After the real tool result, report the human-readable size.',
      ),
    ).toEqual([])
    expect(
      requestedOnceToolNames(
        'First call file_info exactly once. Then call run_command exactly once.',
      ),
    ).toEqual(['file_info', 'run_command'])
    expect(
      requestedOnceToolNames(
        'Call the built-in `file_info` tool exactly once with path panel/package.json.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames(
        'Call the built in `file_info` function exactly once with path panel/package.json.',
      ),
    ).toEqual(['file_info'])
    expect(
      requestedOnceToolNames(
        'Use the run_command tool exactly once with this exact command: printf test.',
      ),
    ).toEqual(['run_command'])
    expect(
      requestedOnceToolNames(
        'Invoke file_info exactly once, then execute run_command exactly once.',
      ),
    ).toEqual(['file_info', 'run_command'])
    expect(
      requestedExactFinalToolNames(
        'Call the built-in `file_info` tool exactly once. Then reply exactly DONE.',
        ['file_info', 'write_file'],
      ),
    ).toEqual(['file_info'])
  })

  it('recognizes a bounded post-tool answer without requiring exact final wording', () => {
    const releasePrompt =
      'Use the run_command tool exactly once with this exact command: printf %s LIVE > probe.txt. After the tool result is returned, reply briefly in English and include LIVE once.'
    expect(requestsBoundedFinalAnswerAfterToolResult(releasePrompt)).toBe(true)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use run_command exactly once. After the tool result, use tools as needed and reply when done.',
      ),
    ).toBe(false)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use run_command exactly once and use additional tools if useful. After the tool result, reply briefly.',
      ),
    ).toBe(false)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use run_command exactly once. After the tool result, use more tools if needed, then reply briefly.',
      ),
    ).toBe(false)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use run_command exactly once. After the tool result, call file_info and reply briefly.',
        ['run_command', 'file_info'],
      ),
    ).toBe(false)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Use run_command exactly once, then call file_info exactly once. After both tool results, reply briefly.',
        ['run_command', 'file_info'],
      ),
    ).toBe(true)
    expect(
      requestsBoundedFinalAnswerAfterToolResult(
        'Discuss what a tool result means and reply briefly.',
      ),
    ).toBe(false)
  })

  it('keeps an open-ended catalog and matches provided names case-insensitively', () => {
    const tools = [
      { function: { name: 'file_info' } },
      { function: { name: 'write_file' } },
    ]
    expect(scopeToolDefinitionsByName(tools, [])).toEqual(tools)
    expect(isToolNameProvidedForCurrentTurn('FILE_INFO', ['file_info'])).toBe(true)
    expect(isToolNameProvidedForCurrentTurn('write_file', ['file_info'])).toBe(false)
  })

  it('maps local no-tool and singular exact-tool authorization onto the wire API', () => {
    expect(toolChoiceForCurrentTurn(true, [], 'responses')).toBe('none')
    expect(toolChoiceForCurrentTurn(true, ['file_info'], 'chat')).toBe('none')
    expect(toolChoiceForCurrentTurn(false, ['file_info'], 'responses')).toEqual({
      type: 'function',
      name: 'file_info',
    })
    expect(toolChoiceForCurrentTurn(false, ['file_info'], 'chat')).toEqual({
      type: 'function',
      function: { name: 'file_info' },
    })
    expect(toolChoiceForCurrentTurn(false, [], 'responses')).toBeUndefined()
    expect(
      toolChoiceForCurrentTurn(false, ['file_info', 'run_command'], 'chat'),
    ).toEqual({
      type: 'function',
      function: { name: 'file_info' },
    })
    expect(
      toolChoiceForCurrentTurn(
        false,
        requiredToolChoiceNamesForCurrentTurn(
          [],
          requestedOnceToolNames(
            'Use the run_command tool exactly once with command pwd. After the tool result, reply briefly.',
          ),
        ),
        'chat',
      ),
    ).toEqual({
      type: 'function',
      function: { name: 'run_command' },
    })
    expect(
      toolChoiceForCurrentTurn(
        false,
        requiredToolChoiceNamesForCurrentTurn(
          [],
          requestedOnceToolNames(
            'Use run_command exactly once and use more tools if needed.',
          ),
        ),
        'chat',
      ),
    ).toEqual({
      type: 'function',
      function: { name: 'run_command' },
    })
    expect(
      requiredToolChoiceNamesForCurrentTurn(
        ['file_info'],
        ['run_command'],
      ),
    ).toEqual(['file_info'])
    expect(
      requestedScopedToolNames(
        'Use exactly one file_info tool call to inspect generation_config.json. Do not guess. After the real tool result, reason briefly, then answer exactly DONE.',
        ['file_info', 'run_command'],
      ),
    ).toEqual(['file_info'])
    expect(
      requestedScopedToolNames(
        'Use run_command exactly once and use more tools if needed.',
        ['file_info', 'run_command'],
      ),
    ).toEqual([])
    expect(
      requestedScopedToolNames(
        'Use file_info exactly once. Do not use additional tools.',
        ['file_info', 'run_command'],
      ),
    ).toEqual(['file_info'])
    expect(
      requestedScopedToolNames(
        'Use file_info exactly once. Do not use run_command.',
        ['file_info', 'run_command'],
      ),
    ).toEqual(['file_info'])
    expect(
      requestedScopedToolNames(
        'Use filesystem__read_file exactly once.',
        ['file_info', 'run_command'],
      ),
    ).toEqual(['filesystem__read_file'])
  })

  it('rejects explicit required tools that are absent from the filtered catalog', () => {
    expect(
      unavailableRequestedToolNames(
        ['file_info', 'READ_FILE'],
        ['read_file', 'run_command'],
      ),
    ).toEqual(['file_info'])
    expect(
      unavailableRequestedToolNames(['FILE_INFO'], ['file_info']),
    ).toEqual([])
  })

  it('keeps remote no-tool compatibility but enforces singular remote tool contracts', () => {
    expect(toolChoiceForCurrentTurn(true, [], 'chat', true)).toBeUndefined()
    expect(toolChoiceForCurrentTurn(true, [], 'responses', true)).toBeUndefined()
    expect(toolChoiceForCurrentTurn(false, [], 'chat', true)).toBeUndefined()
    expect(toolChoiceForCurrentTurn(false, ['file_info'], 'chat', true)).toEqual({
      type: 'function',
      function: { name: 'file_info' },
    })
    expect(
      toolChoiceForCurrentTurn(false, ['file_info'], 'responses', true),
    ).toEqual({
      type: 'function',
      name: 'file_info',
    })
  })

  it('authorizes no MCP execution on forbidden or exact built-in turns', () => {
    expect(
      isToolAuthorizedForCurrentTurn('filesystem__read_file', [], true, []),
    ).toBe(false)
    expect(
      isToolAuthorizedForCurrentTurn(
        'file_info',
        ['file_info'],
        false,
        ['file_info'],
      ),
    ).toBe(true)
    expect(
      isToolAuthorizedForCurrentTurn(
        'filesystem__read_file',
        ['file_info'],
        false,
        ['file_info'],
      ),
    ).toBe(false)
    expect(
      isToolAuthorizedForCurrentTurn(
        'filesystem__read_file',
        ['file_info'],
        false,
        [],
      ),
    ).toBe(true)
    expect(
      isToolAuthorizedForCurrentTurn(
        'filesystem__read_file',
        ['file_info'],
        false,
        ['file_info'],
      ),
    ).toBe(false)
  })

  it('maps an explicit current-turn no-tool directive to the API contract', () => {
    expect(
      requestsNoToolCalls(
        '[FOLLOW] Do not call any tool. Use only the previous result.',
      ),
    ).toBe(true)
    expect(
      requestsNoToolCalls(
        '[FOLLOW] Do not call a tool. Use only the previous result.',
      ),
    ).toBe(true)
    expect(requestsNoToolCalls('Please do not use the tool again.')).toBe(true)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        'Do not call a tool. Reply exactly NO-TOOL-ARTICLE-OK.',
      ),
    ).toBe(true)
    expect(requestsNoToolCalls('Please never use tools. Answer directly.')).toBe(true)
    expect(
      requestsNoToolCalls(
        'Do not call another tool. Using the prior tool results, reply briefly.',
      ),
    ).toBe(true)
    expect(requestsNoToolCalls('Do not use additional tools.')).toBe(true)
    expect(
      requestsNoToolCalls(
        'Describe the attached video; do not call any tool.',
      ),
    ).toBe(true)
    expect(
      requestsNoToolCalls(
        'Use only the prior image: without calling any tool, answer directly.',
      ),
    ).toBe(true)
    expect(
      requestsNoToolCalls(
        '[FOLLOW] Without tools, retrieve the value from the previous turn.',
      ),
    ).toBe(true)
    expect(requestsNoToolCalls('Without using any tools, answer directly.')).toBe(true)
    expect(requestsNoToolCalls('Now solve this without tools: what is the probability?')).toBe(true)
    expect(requestsNoToolCalls('Please answer the question without using any tools.')).toBe(true)
    expect(requestsNoToolCalls('Respond without tools, using the history.')).toBe(true)
    expect(requestsNoToolCalls('Explain why someone says "solve this without tools".')).toBe(false)
    expect(requestsNoToolCalls('Solve this without tools unless a file is missing.')).toBe(false)
    expect(requestsNoToolCalls('Can this model solve this without tools?')).toBe(false)
    expect(
      requestsNoToolCalls(
        '[FOLLOW] Without calling any tool, recall the previous result.',
      ),
    ).toBe(true)
    expect(
      requestsNoToolCalls('Do not call any tool unless the file is missing.'),
    ).toBe(false)
    expect(
      requestsNoToolCalls('Explain why someone might say "do not call any tool".'),
    ).toBe(false)
    expect(
      requestsNoToolCalls('Explain the phrase "without tools" in this policy.'),
    ).toBe(false)
  })

  it('treats strict exact-answer prompts without a tool directive as text-only turns', () => {
    expect(
      requestsExactTextOnlyWithoutToolUse(
        '[LAG-UI-GLOBAL1] Think privately if useful, then reply exactly LAG-UI-GLOBAL1-DONE.',
      ),
    ).toBe(true)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        '[LAG-S21-UI-RAIL] Think privately in one short sentence. Visible answer exactly LAG-S21-UI-RAIL-DONE.',
      ),
    ).toBe(true)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        'Privately solve it. Your only visible answer must be SECOND-PASS-DONE.',
      ),
    ).toBe(true)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        'Call the built-in file_info tool exactly once with path panel/package.json. After the tool result, reply exactly DONE.',
      ),
    ).toBe(false)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        'Call the built-in `file_info` tool exactly once with path panel/package.json. After the tool result, reply exactly DONE.',
      ),
    ).toBe(false)
    expect(
      requestsExactTextOnlyWithoutToolUse(
        'You must use the tool. After its function result, reply exactly DONE.',
      ),
    ).toBe(false)
    expect(
      requestsExactTextOnlyWithoutToolUse('Use tools as needed, then reply exactly DONE.'),
    ).toBe(false)
  })

  it('treats private reasoning calculation probes as text-only even when tools are enabled', () => {
    expect(
      requestsPrivateReasoningWithoutToolUse(
        '[LAG-S21] Privately calculate 143 times 27 and double-check it. Then answer one concise sentence ending with PASS.',
      ),
    ).toBe(true)
    expect(
      requestsPrivateReasoningWithoutToolUse(
        'Do not expose reasoning. Compute the modulo and answer with only the marker.',
      ),
    ).toBe(true)
    expect(
      requestsPrivateReasoningWithoutToolUse(
        'After the private calculation, answer exactly THREE-LINE-PASS.',
      ),
    ).toBe(true)
    expect(
      requestsPrivateReasoningWithoutToolUse(
        'Call the built-in file_info tool exactly once with path panel/package.json. You must use the tool.',
      ),
    ).toBe(false)
    expect(
      requestsPrivateReasoningWithoutToolUse(
        'Inspect the repo and fix the failing test.',
      ),
    ).toBe(false)
  })

  it('omits unusable tool schemas and suppresses the generic tool prompt when requested', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const latestUserText = latestUserMessageText(messages)')
    expect(source).not.toContain(
      '.find((m: any) => m?.role === "user" && typeof m.content === "string")',
    )
    expect(source).toContain('requestsNoToolCalls(latestUserText)')
    expect(source).toContain('requestsExactTextOnlyWithoutToolUse(latestUserText)')
    expect(source).toContain('requestsPrivateReasoningWithoutToolUse(latestUserText)')
    expect(source).toContain('const attachBuiltinToolsForCurrentTurn =')
    expect(source).toContain('!exactTextOnlyNoToolTurn')
    expect(source).toContain('!privateReasoningNoToolTurn')
    expect(source).toContain(
      'const currentPromptAlreadyForbidsTools = userForbidsToolCalls',
    )
    expect(source).not.toContain(
      'const currentPromptAlreadyForbidsTools =\n        userForbidsToolCalls ||',
    )
    expect(source.match(/if \(attachBuiltinToolsForCurrentTurn\)/g) || []).toHaveLength(2)
    expect(source).toContain('toolChoiceForCurrentTurn(')
    expect(source).toContain('const requestToolChoice = currentTurnToolChoice()')
    expect(source).toContain('const remainingExactlyOnceBuiltinTools =')
    expect(source).toContain('requiredToolChoiceNamesForCurrentTurn(')
    expect(source).toContain('unavailableRequestedToolNames(')
    expect(source).toContain('disabled or unavailable for this turn')
    expect(source).toContain('requestedScopedToolNames(')
    expect(source).toContain('exactlyOnceToolsComplete')
    expect(source).toContain('obj.tool_choice = requestToolChoice')
    expect(source).toContain('applyPostToolRequestFields(obj')
  })

  it('checks the terminal AppleScript round before sending a follow-up', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const policy = source.indexOf('shouldFinishZayaAppleScriptToolRound(')
    const terminalBreak = source.indexOf('if (finishAfterNativeToolResult)', policy)
    const followUp = source.indexOf('if (!(await sendFollowUp())) break;', policy)

    expect(policy).toBeGreaterThan(-1)
    expect(terminalBreak).toBeGreaterThan(policy)
    expect(followUp).toBeGreaterThan(terminalBreak)
  })

  it('increments the auto-continue counter once per follow-up attempt', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('shouldAutoContinueAfterToolUse({'),
      source.indexOf('const hasContent = fullContent.trim().length > 0'),
    )

    expect(branch.match(/autoContinueCount\+\+/g) || []).toHaveLength(1)
  })

  it('uses one answer-only recovery instead of repeating reasoning-only tool follow-ups', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const MAX_AUTO_CONTINUES = 1')
    expect(source).toContain('let finalAnswerRecovery = false')
    expect(source).toContain('finalAnswerRecovery = true')
    expect(source).toContain('applyPostToolRequestFields(obj')
    expect(source).toContain('obj.enable_thinking = false')
    expect(source).toContain(
      'const noVisibleAnswerWarning = noVisibleToolAnswerWarning({',
    )
    expect(source).toContain('recoveryAttempted: finalAnswerRecovery')
  })

  it('preserves the completed tool render before the exact-final follow-up', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const planned = source.indexOf('plannedDirectAnswerPass =')
    const followUp = source.indexOf('if (!(await sendFollowUp())) break;', planned)
    const policyStart = source.indexOf('const applyPostToolAnswerPolicy =')
    const policyEnd = source.indexOf('if (useResponsesApi)', policyStart)
    const policy = source.slice(policyStart, policyEnd)
    const authorizationStart = source.indexOf(
      'const toolAuthorized = isToolAuthorizedForCurrentTurn(',
    )
    const authorization = source.slice(
      authorizationStart,
      source.indexOf(');', authorizationStart) + 2,
    )

    expect(source).toContain('exactFinalBuiltinToolNames.length === 1')
    expect(source).toContain('requestedExactFinalToolNames(')
    expect(source).toContain('unscopedCurrentTurnToolDefinitions')
    expect(source).toContain('requestedOnceToolNames(latestUserText)')
    expect(source).toContain('completedExactlyOnceTools.has(name)')
    expect(source).toContain('Duplicate ${tc.function.name} call was not executed')
    expect(source).toContain('const availableToolDefinitions = () =>')
    expect(source).toContain('completedExactFinalTools.add(normalizedToolName)')
    expect(source).toContain('exactFinalToolNames.every((name) =>')
    expect(authorization).toContain('currentToolNames')
    expect(authorization).not.toContain('availableToolDefinitions()')
    expect(policy).toContain('applyPostToolRequestFields(obj')
    expect(policy).toContain('previous: previousToolRequestFields')
    expect(policy).toContain('if (!finalAnswerRecovery) return')
    expect(policy.indexOf('if (!finalAnswerRecovery) return')).toBeLessThan(
      policy.indexOf('obj.enable_thinking = false'),
    )
    expect(planned).toBeGreaterThan(-1)
    expect(followUp).toBeGreaterThan(planned)
    expect(source).toContain('"X-vMLX-Tool-Choice-Fulfilled": "1"')
    expect(source).toContain('logRequestShape(followUpBody, "follow_up")')
  })

  it('marks exactly-once completion only after valid authorized arguments', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const loopStart = source.indexOf('for (const tc of receivedToolCalls)')
    const loopEnd = source.indexOf(
      'if (exactFinalToolNames.includes(normalizedToolName))',
      loopStart,
    )
    const loop = source.slice(loopStart, loopEnd)
    const parse = loop.indexOf('toolArgs = JSON.parse(')
    const authorize = loop.indexOf(
      'const toolAuthorized = isToolAuthorizedForCurrentTurn(',
    )
    const authorizedBranch = loop.indexOf('if (toolAuthorized) {', authorize)
    const mark = loop.indexOf(
      'completedExactlyOnceTools.add(normalizedToolName)',
      authorizedBranch,
    )
    const executing = loop.indexOf('emitToolStatus(\n                  "executing"', authorizedBranch)
    const rejectedBranch = loop.indexOf('if (!toolAuthorized) {', authorize)

    expect(loopStart).toBeGreaterThan(-1)
    expect(loopEnd).toBeGreaterThan(loopStart)
    expect(parse).toBeGreaterThan(-1)
    expect(authorize).toBeGreaterThan(parse)
    expect(authorizedBranch).toBeGreaterThan(authorize)
    expect(mark).toBeGreaterThan(authorizedBranch)
    expect(executing).toBeGreaterThan(mark)
    expect(rejectedBranch).toBeGreaterThan(executing)
    expect(loop.slice(0, parse)).not.toContain(
      'completedExactlyOnceTools.add(normalizedToolName)',
    )
  })

  it('resets token timing at follow-up stream boundaries and counts long in-stream gaps', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const followUp = source.slice(
      source.indexOf('const sendFollowUp = async'),
      source.indexOf('// ─── Helper: execute tool calls', source.indexOf('const sendFollowUp = async')),
    )

    expect(followUp).toContain('lastTokenTime = null')
    expect(source).toContain('if (gap > 0) generationMs += gap')
    expect(source).not.toContain('if (gap < 5000) generationMs += gap')
  })

  it('uses cumulative agent-loop timing without accepting a buffered usage burst', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const liveTpsHistory: number[] = []')
    expect(source).toContain('liveTpsHistory.push(liveTps)')
    expect(source).toContain('const finalTps =')
    expect(source).toContain('serverDecodeSummary?.tokensPerSecond ??')
    expect(source).toContain('selectFinalDecodeTps({')
    expect(source).toContain('cumulativeTps: cumulativeDecodeTps')
    expect(source).toContain('rollingTps: liveTpsHistory')
  })

  it('drops only superseded empty-response warnings after a successful recovery', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const start = source.indexOf('if (\n          finalAnswerRecovery &&')
    const end = source.indexOf('if (\n          toolIteration > 0 &&', start)
    const branch = source.slice(start, end)

    expect(start).toBeGreaterThan(-1)
    expect(branch).toContain('allGeneratedContent.trim() || fullContent.trim()')
    expect(branch).toContain('dropSupersededRecoveryWarnings(responseWarnings)')
  })

  it('resets text-chat tool streaming state before chained follow-up requests', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('receivedToolCalls = [];'),
      source.indexOf('if (!(await sendFollowUp())) break;', source.indexOf('receivedToolCalls = [];')),
    )

    for (const required of [
      'receivedToolCalls = []',
      'fullContent = ""',
      'rawAccumulated = ""',
      'lastFinishReason = undefined',
      'clientToolCallBuffering = false',
      'clientSideThinkParsing = false',
      'serverSendsUsage = false',
      'currentEventType = ""',
      'seenResponsesApiEvents.clear()',
    ]) {
      expect(branch).toContain(required)
    }
  })

  it('clears Responses tool-call buffers before follow-up and on stalled buffering', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const toolLoopStart = source.indexOf('await executeToolCalls()')
    const followUpStart = source.indexOf('if (!(await sendFollowUp())) break;', toolLoopStart)
    const followUpBranch = source.slice(toolLoopStart, followUpStart)

    expect(toolLoopStart).toBeGreaterThan(-1)
    expect(followUpStart).toBeGreaterThan(toolLoopStart)
    expect(followUpBranch).toContain('receivedToolCalls = []')
    expect(followUpBranch).toContain('clientToolCallBuffering = false')
    expect(followUpBranch.indexOf('receivedToolCalls = []')).toBeLessThan(
      followUpBranch.indexOf('clientToolCallBuffering = false'),
    )

    const stallStart = source.indexOf('Tool call generation stalled')
    const stallBranch = source.slice(stallStart, source.indexOf('await rdr.cancel()', stallStart) + 200)

    expect(stallStart).toBeGreaterThan(-1)
    expect(stallBranch).toContain('clientToolCallBuffering = false')
    expect(stallBranch).toContain('await rdr.cancel()')
    expect(stallBranch).not.toContain('executeToolCalls')
  })

  it('responses stream parser accepts data-only event types from parsed payloads', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const responsesEventType =')
    expect(source).toContain(
      'typeof parsed.type === "string" ? parsed.type : currentEventType',
    )

    const functionCallBranch = source.slice(
      source.indexOf('// Handle function_call items (tool calls) from Responses API'),
      source.indexOf('// Real-time usage from response.usage events'),
    )

    expect(functionCallBranch).toContain(
      'responsesEventType === "response.output_item.done"',
    )
  })

  it('loopback remote sessions use node streaming fetch for SSE', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('function isLoopbackUrl')
    expect(source).toContain('const useNodeStreamingFetch =')
    expect(source).toContain('!isRemote || isLoopbackUrl(apiUrl)')
    expect(source).toContain('!isRemote || isLoopbackUrl(url)')
  })

  it('suppresses generic agentic instructions for native DSV4, ZAYA, and LFM2 prompts', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('function shouldSuppressGenericAgenticPromptForNativeTools')
    expect(source).toContain('detectedFamily === "zaya"')
    expect(source).toContain('detectedFamily === "zaya1-vl"')
    expect(source).toContain('detectedFamily === "zaya1_vl"')
    expect(source).toContain('detectedFamily === "lfm2"')
    expect(source).toContain('detectedFamily === "deepseek-v4"')
    expect(source).toContain('modelNameOrPath')
    expect(source).toContain('modelName.includes("zaya")')
    expect(source).toContain('modelName.includes("lfm2")')
    expect(source).not.toContain('modelName.includes("deepseek-v4")')
    expect(source).not.toContain('modelName.includes("dsv4")')
    expect(source).toContain('await readDetectedModelConfig(chat.modelPath)')
    expect(source).toContain('const suppressGenericAgenticToolPromptForNativeTools =')
    expect(source).toContain(
      'chat.modelPath || resolvedSession?.remoteModel || chat.modelId',
    )

    const promptBranch = source.slice(
      source.indexOf('const suppressGenericAgenticToolPromptForNativeTools ='),
      source.indexOf('// No default system prompt injected'),
    )
    expect(promptBranch).toContain('!suppressGenericAgenticToolPromptForNativeTools')
    expect(promptBranch).toContain(
      'chat.modelPath || resolvedSession?.remoteModel || chat.modelId',
    )
    expect(promptBranch).toContain('AGENTIC_SYSTEM_PROMPT + directMediaAttachmentRule')
    expect(promptBranch).toContain('directMediaAttachmentRule.trim()')
    expect(source).toContain('const scopedCurrentTurnBuiltinToolNames =')
    expect(source).toContain(
      'scopedCurrentTurnToolNames.filter((name) => isBuiltinTool(name))',
    )
  })

  it('panel max tool iterations caps tool loops', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10;'),
      source.indexOf('if (toolIteration > 0 || collectedToolStatuses.length > 0)'),
    )

    expect(branch).toContain('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10')
    expect(branch).toContain('while (toolIteration < MAX_TOOL_ITERATIONS)')
    expect(branch).toContain('toolIteration++')
  })
})
