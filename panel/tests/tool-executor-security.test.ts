import { afterEach, describe, expect, it, vi } from 'vitest'
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

vi.mock('electron', () => ({
  clipboard: {
    readText: vi.fn(() => ''),
    writeText: vi.fn(),
  },
}))

vi.mock('../src/main/database', () => ({
  db: {
    getSetting: vi.fn(() => undefined),
  },
}))

import { executeBuiltinTool } from '../src/main/tools/executor'

const roots: string[] = []

function makeRoot(prefix: string): string {
  const root = mkdtempSync(join(tmpdir(), prefix))
  roots.push(root)
  return root
}

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop()!, { recursive: true, force: true })
  }
})

describe('built-in tool executor path sandbox', () => {
  it('blocks write_file and edit_file traversal outside the working directory', async () => {
    const workingDir = makeRoot('vmlx-tool-sandbox-work-')
    const outside = makeRoot('vmlx-tool-sandbox-outside-')
    const outsideFile = join(outside, 'owned.txt')
    writeFileSync(outsideFile, 'original', 'utf8')

    const relEscape = `../${outside.split('/').pop()}/owned.txt`

    const writeResult = await executeBuiltinTool(
      'write_file',
      { path: relEscape, content: 'changed' },
      workingDir,
    )
    expect(writeResult.is_error).toBe(true)
    expect(writeResult.content).toContain('Path escapes working directory')
    expect(readFileSync(outsideFile, 'utf8')).toBe('original')

    const editResult = await executeBuiltinTool(
      'edit_file',
      { path: relEscape, search_text: 'original', replacement_text: 'changed' },
      workingDir,
    )
    expect(editResult.is_error).toBe(true)
    expect(editResult.content).toContain('Path escapes working directory')
    expect(readFileSync(outsideFile, 'utf8')).toBe('original')
  })

  it('blocks symlink-parent escapes for newly created files', async () => {
    const workingDir = makeRoot('vmlx-tool-sandbox-work-')
    const outside = makeRoot('vmlx-tool-sandbox-outside-')
    const link = join(workingDir, 'outside-link')
    symlinkSync(outside, link, 'dir')

    const result = await executeBuiltinTool(
      'write_file',
      { path: 'outside-link/new.txt', content: 'escape' },
      workingDir,
    )

    expect(result.is_error).toBe(true)
    expect(result.content).toContain('Path escapes working directory')
  })

  it('allows normal nested writes inside the working directory', async () => {
    const workingDir = makeRoot('vmlx-tool-sandbox-work-')
    const result = await executeBuiltinTool(
      'write_file',
      { path: 'nested/ok.txt', content: 'inside' },
      workingDir,
    )

    expect(result.is_error).toBe(false)
    expect(readFileSync(join(workingDir, 'nested', 'ok.txt'), 'utf8')).toBe('inside')
  })

  it('blocks absolute-path writes and edits outside the working directory', async () => {
    const workingDir = makeRoot('vmlx-tool-sandbox-work-')
    const outside = makeRoot('vmlx-tool-sandbox-outside-')
    const outsideFile = join(outside, 'owned.txt')
    writeFileSync(outsideFile, 'original\nsecond\n', 'utf8')

    const cases: Array<{ tool: string; args: Record<string, any> }> = [
      { tool: 'write_file', args: { path: outsideFile, content: 'changed' } },
      { tool: 'edit_file', args: { path: outsideFile, search_text: 'original', replacement_text: 'changed' } },
      { tool: 'patch_file', args: { path: outsideFile, patch: '@@ -1,1 +1,1 @@\n-original\n+changed' } },
      { tool: 'batch_edit', args: { path: outsideFile, edits: [{ search_text: 'original', replacement_text: 'changed' }] } },
      { tool: 'insert_text', args: { path: outsideFile, line: 1, text: 'changed' } },
      { tool: 'replace_lines', args: { path: outsideFile, start_line: 1, end_line: 1, text: 'changed' } },
      { tool: 'apply_regex', args: { path: outsideFile, pattern: 'original', replacement: 'changed' } },
    ]

    for (const { tool, args } of cases) {
      const result = await executeBuiltinTool(tool, args, workingDir)
      expect(result.is_error, tool).toBe(true)
      expect(result.content, tool).toContain('Path escapes working directory')
      expect(readFileSync(outsideFile, 'utf8'), tool).toBe('original\nsecond\n')
    }
  })

  it('blocks directory and copy/move/delete mutations outside the working directory', async () => {
    const workingDir = makeRoot('vmlx-tool-sandbox-work-')
    const outside = makeRoot('vmlx-tool-sandbox-outside-')
    const insideFile = join(workingDir, 'inside.txt')
    const outsideFile = join(outside, 'owned.txt')
    const outsideCopy = join(outside, 'copy.txt')
    writeFileSync(insideFile, 'inside', 'utf8')
    writeFileSync(outsideFile, 'original', 'utf8')

    const cases: Array<{ tool: string; args: Record<string, any>; expectedFile?: string }> = [
      { tool: 'create_directory', args: { path: join(outside, 'new-dir') }, expectedFile: join(outside, 'new-dir') },
      { tool: 'delete_file', args: { path: outsideFile } },
      { tool: 'move_file', args: { source: insideFile, destination: join(outside, 'moved.txt') }, expectedFile: join(outside, 'moved.txt') },
      { tool: 'copy_file', args: { source: insideFile, destination: outsideCopy }, expectedFile: outsideCopy },
    ]

    for (const { tool, args, expectedFile } of cases) {
      const result = await executeBuiltinTool(tool, args, workingDir)
      expect(result.is_error, tool).toBe(true)
      expect(result.content, tool).toContain('Path escapes working directory')
      expect(readFileSync(outsideFile, 'utf8'), tool).toBe('original')
      if (expectedFile) expect(existsSync(expectedFile), tool).toBe(false)
      expect(readFileSync(insideFile, 'utf8'), tool).toBe('inside')
    }
  })
})
