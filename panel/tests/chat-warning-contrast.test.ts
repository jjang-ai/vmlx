import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('chat warning contrast tokens', () => {
  it('uses readable surface text for both tinted warning containers', () => {
    const source = readFileSync(resolve('src/renderer/src/components/chat/MessageBubble.tsx'), 'utf8')
    const containers = [...source.matchAll(/className="([^"]*bg-warning\/10[^"]*)"/g)].map(match => match[1])
    expect(containers).toHaveLength(2)
    for (const classes of containers) {
      expect(classes.split(' ')).toContain('text-foreground')
      expect(classes).not.toContain('text-warning-foreground')
    }
  })
})
