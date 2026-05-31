import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { MessageBubble } from '../src/renderer/src/components/chat/MessageBubble'

vi.mock('dompurify', () => ({
  default: {
    sanitize: (html: string) => html,
  },
}))

const baseMessage = {
  id: 'assistant-1',
  role: 'assistant' as const,
  content: '',
  timestamp: Date.now(),
}

function renderBubble(props: Record<string, unknown>): string {
  return renderToStaticMarkup(React.createElement(MessageBubble as any, props))
}

describe('interleaved reasoning rendered display', () => {
  it('live-replaces previous reasoning segments while streaming and shows all after completion', () => {
    const segments = [
      'First reasoning segment before tool.',
      'Second reasoning segment after tool.',
    ]

    const streaming = renderBubble({
      message: baseMessage,
      isStreaming: true,
      reasoningSegments: segments,
      reasoningDone: false,
      isLastAssistant: true,
    })

    expect(streaming).not.toContain('First reasoning segment before tool.')
    expect(streaming).toContain('Second reasoning segment after tool.')

    const completed = renderBubble({
      message: baseMessage,
      isStreaming: false,
      reasoningSegments: segments,
      reasoningDone: true,
      isLastAssistant: true,
    })

    expect(completed).toContain('First reasoning segment before tool.')
    expect(completed).toContain('Second reasoning segment after tool.')
  })

  it('renders reasoning and structured tool status without leaking raw tool parser markup', () => {
    const html = renderBubble({
      message: {
        ...baseMessage,
        content: [
          'I will inspect the file.',
          '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/a.txt"}}</tool_call>',
          'The file says hello.',
        ].join('\n'),
      },
      isStreaming: false,
      reasoningSegments: [
        'Need to inspect the file before answering.',
        'Tool returned the relevant text.',
      ],
      reasoningDone: true,
      toolStatuses: [
        {
          phase: 'calling',
          toolName: 'read_file',
          toolCallId: 'call-read-1',
          detail: '{"path":"/tmp/a.txt"}',
          contentOffset: 25,
          timestamp: 1,
        },
        {
          phase: 'result',
          toolName: 'read_file',
          toolCallId: 'call-read-1',
          detail: 'hello',
          timestamp: 2,
        },
      ],
      isLastAssistant: true,
    })

    expect(html).toContain('Need to inspect the file before answering.')
    expect(html).toContain('Tool returned the relevant text.')
    expect(html).toContain('Read')
    expect(html).toContain('/tmp/a.txt')
    expect(html).toContain('The file says hello.')
    expect(html).not.toContain('<tool_call')
    expect(html).not.toContain('</tool_call>')
    expect(html).not.toContain('zyphra_tool_call')
    expect(html).not.toContain('&lt;tool_call')
  })
})
