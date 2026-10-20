import { describe, expect, it } from 'vitest'
import { appendReasoningDelta, markReasoningToolBoundary } from '../src/shared/interleavedReasoning'
import { replayPersistedAssistantHistory } from '../src/shared/toolHistoryReplay'
import { REASONING_WITHOUT_ANSWER_NOTICE } from '../src/shared/responsesStreamRecovery'

describe('persisted assistant tool-history replay', () => {
  it.each([true, false])('preserves a completed empty generation boundary (Responses=%s)', (responses) => {
    const rejected = {
      content: '',
      generationRecordJson: JSON.stringify({
        status: 'completed', passes: [{ submittedAt: 123 }], finishReason: 'incomplete',
      }),
    }
    const prior = { role: 'user', content: 'Diagnostic control sequence' }
    const followup = { role: 'user', content: 'Answer the separate history question' }
    const history = [prior, ...replayPersistedAssistantHistory(rejected, responses), followup]
    expect(history.map(item => item.role)).toEqual(['user', 'assistant', 'user'])
    expect(history[1]).toEqual(responses
      ? { type: 'message', role: 'assistant', content: '' }
      : { role: 'assistant', content: '' })
  })

  it.each(['{}', '{broken', JSON.stringify({ status: 'pending', passes: [] }),
    JSON.stringify({ status: 'completed', passes: [] })])('does not invent a generation for an empty placeholder %s', (record) => {
    for (const wire of [true, false]) expect(replayPersistedAssistantHistory({
      content: '', generationRecordJson: record,
    }, wire)).toEqual([])
  })

  it('keeps a diagnostic reasoning-only bubble out of model history on both wires', () => {
    const exhausted = {
      content: REASONING_WITHOUT_ANSWER_NOTICE,
      reasoningContent: 'unfinished planning',
      warningsJson: JSON.stringify([
        'This response produced reasoning only (no visible message, no tool calls). The maximum output-token limit was reached.',
      ]),
    }
    expect(replayPersistedAssistantHistory(exhausted, true)).toEqual([
      { type: 'message', role: 'assistant', content: '' },
    ])
    expect(replayPersistedAssistantHistory(exhausted, false)).toEqual([
      { role: 'assistant', content: '', reasoning_content: 'unfinished planning' },
    ])
  })

  it('preserves a literal model answer equal to the notice without a diagnostic warning', () => {
    expect(replayPersistedAssistantHistory({
      content: REASONING_WITHOUT_ANSWER_NOTICE,
      reasoningContent: 'quote the requested text',
    }, false)[0].content).toBe(REASONING_WITHOUT_ANSWER_NOTICE)
  })

  const row = {
    content: 'D4-CURRENT-TO1-DONE',
    reasoningContent: 'inspect\n\nreport',
    reasoningSegmentsJson: JSON.stringify(['inspect', 'report']),
    toolCallsJson: JSON.stringify([
      {
        phase: 'calling',
        toolName: 'file_info',
        toolCallId: 'call_1',
        iteration: 0,
      },
    ]),
    toolCallsOaiJson: JSON.stringify([
      {
        id: 'call_1',
        type: 'function',
        function: {
          name: 'file_info',
          arguments: '{"path":"panel/package.json"}',
        },
      },
    ]),
    toolResultsOaiJson: JSON.stringify([
      { tool_call_id: 'call_1', content: 'Size: 5.2 KB' },
    ]),
  }

  it('keeps zero-reasoning tool passes ahead of final reasoning on both wires', () => {
    let segments = markReasoningToolBoundary([], 1)
    segments = markReasoningToolBoundary(segments, 1)
    segments = appendReasoningDelta(segments, 'Reason after real result.')
    expect(segments).toEqual(['', 'Reason after real result.'])
    const saved = { ...row, reasoningSegmentsJson: JSON.stringify(segments) }
    const responses = replayPersistedAssistantHistory(saved, true)
    expect(responses.map(item => item.type)).toEqual([
      'function_call', 'function_call_output', 'reasoning', 'output_text',
    ])
    const chat = replayPersistedAssistantHistory(saved, false)
    expect(chat[0].reasoning_content).toBeUndefined()
    expect(chat[2].reasoning_content).toBe('Reason after real result.')
    expect(markReasoningToolBoundary(markReasoningToolBoundary([], 1), 2))
      .toEqual(['', '', ''])
  })

  it('replays Responses reasoning around the real call and result', () => {
    expect(replayPersistedAssistantHistory(row, true)).toEqual([
      {
        type: 'reasoning',
        content: [{ type: 'reasoning', text: 'inspect' }],
      },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'file_info',
        arguments: '{"path":"panel/package.json"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'Size: 5.2 KB',
      },
      {
        type: 'reasoning',
        content: [{ type: 'reasoning', text: 'report' }],
      },
      { type: 'output_text', text: 'D4-CURRENT-TO1-DONE' },
    ])
  })

  it('replays Chat Completions reasoning_content on each assistant turn', () => {
    expect(replayPersistedAssistantHistory(row, false)).toEqual([
      {
        role: 'assistant',
        content: null,
        tool_calls: [JSON.parse(row.toolCallsOaiJson)[0]],
        reasoning_content: 'inspect',
      },
      {
        role: 'tool',
        tool_call_id: 'call_1',
        content: 'Size: 5.2 KB',
      },
      {
        role: 'assistant',
        content: 'D4-CURRENT-TO1-DONE',
        reasoning_content: 'report',
      },
    ])
  })

  it('uses call iteration metadata for sequential multi-tool ordering', () => {
    const multi = {
      ...row,
      content: 'DONE',
      reasoningSegmentsJson: JSON.stringify(['first', 'second', '']),
      toolCallsJson: JSON.stringify([
        { phase: 'calling', toolCallId: 'call_1', iteration: 0 },
        { phase: 'calling', toolCallId: 'call_2', iteration: 1 },
      ]),
      toolCallsOaiJson: JSON.stringify([
        JSON.parse(row.toolCallsOaiJson)[0],
        {
          id: 'call_2',
          type: 'function',
          function: { name: 'run_command', arguments: '{"command":"pwd"}' },
        },
      ]),
      toolResultsOaiJson: JSON.stringify([
        { tool_call_id: 'call_1', content: 'Size: 5.2 KB' },
        { tool_call_id: 'call_2', content: '/repo' },
      ]),
    }
    const replay = replayPersistedAssistantHistory(multi, true)

    expect(replay.map((item) => item.type)).toEqual([
      'reasoning',
      'function_call',
      'function_call_output',
      'reasoning',
      'function_call',
      'function_call_output',
      'output_text',
    ])
    expect(replay[4].call_id).toBe('call_2')
  })

  it('keeps a Responses reasoning-only assistant row as a boundary without replaying stale hidden text', () => {
    expect(
      replayPersistedAssistantHistory(
        {
          content: '',
          reasoningSegmentsJson: JSON.stringify([
            'stale private reasoning about a previous user prompt',
          ]),
        },
        true,
      ),
    ).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: '',
      },
    ])
  })

  it('omits stale private reasoning for an explicit no-tool turn while preserving real history', () => {
    expect(
      replayPersistedAssistantHistory(row, true, { includeReasoning: false }),
    ).toEqual([
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'file_info',
        arguments: '{"path":"panel/package.json"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'Size: 5.2 KB',
      },
      { type: 'output_text', text: 'D4-CURRENT-TO1-DONE' },
    ])

    expect(
      replayPersistedAssistantHistory(
        {
          content: '',
          reasoningContent: 'stale requirement to call a tool',
        },
        true,
        { includeReasoning: false },
      ),
    ).toEqual([{ type: 'message', role: 'assistant', content: '' }])
  })
})

describe('live-loop iteration numbering (tool steps numbered from 1)', () => {
  it('maps persisted segments by step rank so the tool step keeps its own reasoning and the final reasoning survives', () => {
    // Shape persisted by the live loop on 2026-09-06 (dev app, Qwen3.8 4S): the
    // "generating" status is iteration 0 and the first "calling" status is
    // iteration 1; segments are [tool-step reasoning, final reasoning].
    const row = {
      content: 'Magic word: PELICAN-7731 (4 entries).',
      reasoningContent: 'plan the two calls\n\nreport the result',
      reasoningSegmentsJson: JSON.stringify(['plan the two calls', 'report the result']),
      toolCallsJson: JSON.stringify([
        { phase: 'generating', toolName: '', iteration: 0 },
        { phase: 'calling', toolName: 'read_file', toolCallId: 'call_a', iteration: 1 },
        { phase: 'calling', toolName: 'run_command', toolCallId: 'call_b', iteration: 1 },
      ]),
      toolCallsOaiJson: JSON.stringify([
        { id: 'call_a', type: 'function', function: { name: 'read_file', arguments: '{"path":"README.md"}' } },
        { id: 'call_b', type: 'function', function: { name: 'run_command', arguments: '{"command":"ls -la ."}' } },
      ]),
      toolResultsOaiJson: JSON.stringify([
        { tool_call_id: 'call_a', content: 'The magic word is PELICAN-7731.' },
        { tool_call_id: 'call_b', content: 'total 16 ...' },
      ]),
    }
    const replay = replayPersistedAssistantHistory(row, true)
    expect(replay.map((item) => item.type)).toEqual([
      'reasoning', 'function_call', 'function_call', 'function_call_output', 'function_call_output', 'reasoning', 'output_text',
    ])
    expect(replay[0].content[0].text).toBe('plan the two calls')
    expect(replay[5].content[0].text).toBe('report the result')
    const chat = replayPersistedAssistantHistory(row, false)
    expect(chat[0].reasoning_content).toBe('plan the two calls')
    expect(chat[chat.length - 1].reasoning_content).toBe('report the result')
  })
})

it('replays Naive native content separators from the recorded wire history on both APIs', () => {
  const call = { id: 'native-call', type: 'function', function: { name: 'read_file', arguments: '{"path":"a"}' } }
  const row = {
    content: 'Answer.',
    reasoningSegmentsJson: JSON.stringify(['\nThink.\n', '\nDone.\n']),
    toolCallsOaiJson: JSON.stringify([call]),
    toolResultsOaiJson: JSON.stringify([{ tool_call_id: call.id, content: 'file data' }]),
    generationRecordJson: JSON.stringify({
      passes: [{ family: 'naive_n05_flash' }],
      nativeFinalContent: '\n\nAnswer.',
      toolExchange: [
        { type: 'output_text', text: '\n\n' },
        { type: 'function_call', call_id: call.id },
        { type: 'function_call_output', call_id: call.id, output: 'file data' },
      ],
    }),
  }
  const chat = replayPersistedAssistantHistory(row, false)
  expect(chat[0].content).toBe('\n\n')
  expect(chat[2].content).toBe('\n\nAnswer.')
  const responses = replayPersistedAssistantHistory(row, true)
  expect(responses.filter(x => x.type === 'output_text').map(x => x.text)).toEqual(['\n\n', '\n\nAnswer.'])
  const legacy = replayPersistedAssistantHistory({ ...row, generationRecordJson: '{}' }, false)
  expect(legacy[0].content).toBe(null)
  expect(legacy[2].content).toBe('Answer.')
})
