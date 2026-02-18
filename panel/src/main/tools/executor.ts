/**
 * Built-in tool executor — runs coding tools locally in the Electron main process.
 * All file operations are sandboxed to the working directory.
 */

import { readFileSync, writeFileSync, copyFileSync, renameSync, unlinkSync, rmdirSync, mkdirSync, readdirSync, statSync, existsSync, realpathSync } from 'fs'
import { resolve, relative, dirname, basename, isAbsolute, join } from 'path'
import { execSync, execFileSync } from 'child_process'
import { db } from '../database'

export interface ToolResult {
  content: string
  is_error: boolean
}

// ─── Security ────────────────────────────────────────────────────────────────

/** Resolve path relative to working directory. Blocks directory traversal and symlink escape. */
function resolvePath(workingDir: string, userPath: string): string {
  const resolved = isAbsolute(userPath) ? resolve(userPath) : resolve(workingDir, userPath)
  // Resolve symlinks to prevent escape via symlink chains
  let realResolved: string
  try {
    realResolved = realpathSync(resolved)
  } catch {
    // Path doesn't exist yet (e.g., write_file creating new file) — resolve parent
    const parent = dirname(resolved)
    try {
      realResolved = join(realpathSync(parent), basename(resolved))
    } catch {
      realResolved = resolved
    }
  }
  const realWorkingDir = realpathSync(workingDir)
  const rel = relative(realWorkingDir, realResolved)
  if (rel.startsWith('..') || (isAbsolute(rel) && !realResolved.startsWith(realWorkingDir))) {
    throw new Error(`Path escapes working directory: ${userPath}`)
  }
  return realResolved
}

// ─── Tool Result Limits ──────────────────────────────────────────────────────

const MAX_TOOL_RESULT_CHARS = 50000 // ~50KB — prevents context overflow on large files/outputs

function truncateResult(content: string): string {
  if (content.length <= MAX_TOOL_RESULT_CHARS) return content
  return content.slice(0, MAX_TOOL_RESULT_CHARS) + `\n\n[Truncated — showing first ${MAX_TOOL_RESULT_CHARS} of ${content.length} characters]`
}

// ─── Main Entry ──────────────────────────────────────────────────────────────

export async function executeBuiltinTool(
  toolName: string,
  args: Record<string, any>,
  workingDir: string
): Promise<ToolResult> {
  try {
    let result: ToolResult
    switch (toolName) {
      case 'read_file':
        result = readFile(args.path, workingDir, args.offset, args.limit); break
      case 'write_file':
        result = writeFile(args.path, args.content, workingDir); break
      case 'edit_file':
        result = editFile(args.path, args.search_text, args.replacement_text, workingDir, args.replace_all); break
      case 'patch_file':
        result = patchFile(args.path, args.patch, workingDir); break
      case 'batch_edit':
        result = batchEdit(args.path, args.edits, workingDir); break
      case 'get_diagnostics':
        result = getDiagnostics(args.path || '.', args.tool, workingDir); break
      case 'ddg_search':
        result = await ddgSearch(args.query, args.count); break
      case 'list_directory':
        result = listDirectory(args.path || '.', args.recursive ?? false, workingDir); break
      case 'search_files':
        result = searchFiles(args.pattern, args.path || '.', args.glob, workingDir); break
      case 'find_files':
        result = findFiles(args.pattern, args.path || '.', workingDir); break
      case 'file_info':
        result = fileInfo(args.path, workingDir); break
      case 'create_directory':
        result = createDirectory(args.path, workingDir); break
      case 'delete_file':
        result = deleteFile(args.path, workingDir); break
      case 'move_file':
        result = moveFile(args.source, args.destination, workingDir); break
      case 'copy_file':
        result = copyFile(args.source, args.destination, workingDir); break
      case 'run_command':
        result = runCommand(args.command, workingDir); break
      case 'web_search':
        result = await webSearch(args.query, args.count); break
      case 'fetch_url':
        result = await fetchUrl(args.url, args.max_length); break
      default:
        return { content: `Unknown tool: ${toolName}`, is_error: true }
    }
    // Truncate large results to prevent context overflow in follow-up requests
    result.content = truncateResult(result.content)
    return result
  } catch (err: any) {
    return { content: err.message || String(err), is_error: true }
  }
}

// ─── Tool Implementations ────────────────────────────────────────────────────

function readFile(path: string, workingDir: string, offset?: number, limit?: number): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `File not found: ${path}`, is_error: true }
  }
  const stat = statSync(fullPath)
  if (stat.isDirectory()) {
    return { content: `Path is a directory, not a file: ${path}. Use list_directory instead.`, is_error: true }
  }
  // Check file size before reading — reject binary/huge files early
  if (stat.size > 10 * 1024 * 1024) {
    return { content: `File is too large (${formatBytes(stat.size)}). Use run_command with head/tail to read portions.`, is_error: true }
  }
  const content = readFileSync(fullPath, 'utf-8')
  const allLines = content.split('\n')
  const totalLines = allLines.length

  // Apply offset/limit (1-based offset)
  const startLine = Math.max(1, offset || 1)
  const maxLines = limit || 2000
  const endLine = Math.min(totalLines, startLine + maxLines - 1)
  const slice = allLines.slice(startLine - 1, endLine)

  const numbered = slice.map((line, i) => `${String(startLine + i).padStart(5)} | ${line}`).join('\n')

  let header = `File: ${path} (${totalLines} lines)`
  if (startLine > 1 || endLine < totalLines) {
    header += ` — showing lines ${startLine}–${endLine}`
  }
  if (endLine < totalLines) {
    header += `\n[${totalLines - endLine} more lines. Use offset=${endLine + 1} to continue reading.]`
  }

  return { content: `${header}\n\n${numbered}`, is_error: false }
}

function writeFile(path: string, content: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (content === undefined || content === null) return { content: 'Missing required parameter: content', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  mkdirSync(dirname(fullPath), { recursive: true })
  writeFileSync(fullPath, content, 'utf-8')
  const lines = content.split('\n').length
  const bytes = Buffer.byteLength(content, 'utf-8')
  return { content: `Wrote ${path} (${lines} lines, ${bytes} bytes)`, is_error: false }
}

function editFile(
  path: string,
  searchText: string,
  replacementText: string,
  workingDir: string,
  replaceAll?: boolean
): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (!searchText) return { content: 'Missing required parameter: search_text', is_error: true }
  if (replacementText === undefined || replacementText === null) {
    return { content: 'Missing required parameter: replacement_text', is_error: true }
  }

  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `File not found: ${path}`, is_error: true }
  }
  const content = readFileSync(fullPath, 'utf-8')

  // Exact match
  const idx = content.indexOf(searchText)
  if (idx === -1) {
    const allLines = content.split('\n')
    const showLines = Math.min(allLines.length, 300)
    const preview = allLines.slice(0, showLines).map((l, i) => `${String(i + 1).padStart(5)} | ${l}`).join('\n')
    const truncMsg = allLines.length > showLines ? `\n[...${allLines.length - showLines} more lines]` : ''
    return {
      content: `search_text not found in ${path} (${allLines.length} lines). The search_text must match EXACTLY including whitespace and indentation. Here is the file content:\n\n${preview}${truncMsg}`,
      is_error: true
    }
  }

  if (replaceAll) {
    // Replace all occurrences
    const newContent = content.split(searchText).join(replacementText)
    const occurrences = (content.split(searchText).length - 1)
    writeFileSync(fullPath, newContent, 'utf-8')
    return {
      content: `Edited ${path}: replaced ${occurrences} occurrence(s)`,
      is_error: false
    }
  }

  // Single replacement: check for uniqueness
  const secondIdx = content.indexOf(searchText, idx + 1)
  if (secondIdx !== -1) {
    return {
      content: `search_text appears multiple times in ${path}. Provide more surrounding context to make it unique, or use replace_all=true to replace all occurrences.`,
      is_error: true
    }
  }

  const newContent = content.replace(searchText, replacementText)
  writeFileSync(fullPath, newContent, 'utf-8')

  const oldLines = searchText.split('\n').length
  const newLines = replacementText.split('\n').length
  return {
    content: `Edited ${path}: replaced ${oldLines} line(s) with ${newLines} line(s)`,
    is_error: false
  }
}

function listDirectory(path: string, recursive: boolean, workingDir: string): ToolResult {
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `Directory not found: ${path}`, is_error: true }
  }

  const entries: string[] = []
  let fileCount = 0
  const MAX_ENTRIES = 500

  function walk(dir: string, depth: number): void {
    if (fileCount >= MAX_ENTRIES) return
    let items: any[]
    try {
      items = readdirSync(dir, { withFileTypes: true }) as any[]
    } catch {
      return
    }
    // Sort: dirs first, then files
    items.sort((a, b) => {
      if (a.isDirectory() && !b.isDirectory()) return -1
      if (!a.isDirectory() && b.isDirectory()) return 1
      return a.name.localeCompare(b.name)
    })
    const indent = '  '.repeat(depth)
    for (const item of items) {
      if (fileCount >= MAX_ENTRIES) {
        entries.push(`${indent}... (truncated at ${MAX_ENTRIES} entries)`)
        return
      }
      const itemPath = join(dir, item.name)
      fileCount++
      if (item.isDirectory()) {
        entries.push(`${indent}${item.name}/`)
        if (recursive && !item.name.startsWith('.') && item.name !== 'node_modules') {
          walk(itemPath, depth + 1)
        }
      } else {
        try {
          const stat = statSync(itemPath)
          const kb = (stat.size / 1024).toFixed(1)
          entries.push(`${indent}${item.name}  (${kb} KB)`)
        } catch {
          entries.push(`${indent}${item.name}`)
        }
      }
    }
  }

  walk(fullPath, 0)
  return { content: `Directory: ${path}\n\n${entries.join('\n')}`, is_error: false }
}

function searchFiles(
  pattern: string,
  path: string,
  glob: string | undefined,
  workingDir: string
): ToolResult {
  if (!pattern) return { content: 'Missing required parameter: pattern', is_error: true }
  const searchDir = resolvePath(workingDir, path)

  // Build ripgrep args (use execFileSync to prevent shell injection)
  const rgArgs = ['--color=never', '--line-number', '--no-heading', '-m', '100']
  if (glob) rgArgs.push('--glob', glob)
  rgArgs.push('--', pattern, searchDir)

  try {
    const output = execFileSync('rg', rgArgs, {
      encoding: 'utf-8',
      maxBuffer: 10 * 1024 * 1024,
      timeout: 30000
    })
    // Make paths relative to working dir for cleaner output
    const cleaned = output.replace(new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), '')
    return { content: `Search: "${pattern}"${glob ? ` in ${glob}` : ''}\n\n${cleaned}`, is_error: false }
  } catch (err: any) {
    if (err.status === 1) {
      return { content: `No matches found for "${pattern}"${glob ? ` in ${glob}` : ''}`, is_error: false }
    }
    if (err.code === 'ENOENT' || (err.message && err.message.includes('ENOENT'))) {
      // Fallback to grep if ripgrep not installed (also use execFileSync)
      try {
        const grepArgs = ['-rn', '--color=never']
        if (glob) grepArgs.push('--include', glob)
        grepArgs.push('--', pattern, searchDir)
        const output = execFileSync('grep', grepArgs, {
          encoding: 'utf-8',
          maxBuffer: 10 * 1024 * 1024,
          timeout: 30000
        })
        const cleaned = output.replace(new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), '')
        return { content: `Search: "${pattern}"\n\n${cleaned}`, is_error: false }
      } catch (grepErr: any) {
        if (grepErr.status === 1) return { content: `No matches found for "${pattern}"`, is_error: false }
        return { content: `Search failed: ${grepErr.message}`, is_error: true }
      }
    }
    return { content: `Search failed: ${err.message}`, is_error: true }
  }
}

function findFiles(pattern: string, path: string, workingDir: string): ToolResult {
  if (!pattern) return { content: 'Missing required parameter: pattern', is_error: true }
  const searchDir = resolvePath(workingDir, path)

  // Use fd if available, fallback to find
  try {
    const fdArgs = ['--color=never', '--type', 'f', '--glob', pattern, searchDir]
    const output = execFileSync('fd', fdArgs, {
      encoding: 'utf-8',
      maxBuffer: 10 * 1024 * 1024,
      timeout: 30000
    })
    const cleaned = output.replace(new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), '')
    const count = cleaned.trim() ? cleaned.trim().split('\n').length : 0
    return { content: `Found ${count} file(s) matching "${pattern}":\n\n${cleaned}`, is_error: false }
  } catch (err: any) {
    if (err.status === 1) {
      return { content: `No files found matching "${pattern}"`, is_error: false }
    }
    // Fallback to find command
    try {
      const findArgs = [searchDir, '-name', pattern, '-type', 'f', '-not', '-path', '*/node_modules/*', '-not', '-path', '*/.git/*']
      const output = execFileSync('find', findArgs, {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024,
        timeout: 30000
      })
      const cleaned = output.replace(new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), '')
      const count = cleaned.trim() ? cleaned.trim().split('\n').length : 0
      return { content: `Found ${count} file(s) matching "${pattern}":\n\n${cleaned}`, is_error: false }
    } catch (findErr: any) {
      return { content: `Find failed: ${findErr.message}`, is_error: true }
    }
  }
}

function fileInfo(path: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `Path not found: ${path}`, is_error: true }
  }
  const stat = statSync(fullPath)
  const type = stat.isDirectory() ? 'directory' : stat.isSymbolicLink() ? 'symlink' : 'file'
  const size = stat.isDirectory() ? '-' : formatBytes(stat.size)
  const modified = new Date(stat.mtime).toISOString()
  const mode = '0' + (stat.mode & 0o777).toString(8)
  return {
    content: `Path: ${path}\nType: ${type}\nSize: ${size}\nModified: ${modified}\nPermissions: ${mode}`,
    is_error: false
  }
}

function createDirectory(path: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  mkdirSync(fullPath, { recursive: true })
  return { content: `Created directory: ${path}`, is_error: false }
}

function deleteFile(path: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `Path not found: ${path}`, is_error: true }
  }
  const stat = statSync(fullPath)
  if (stat.isDirectory()) {
    try {
      rmdirSync(fullPath) // Only removes empty directories
    } catch (err: any) {
      if (err.code === 'ENOTEMPTY') {
        return { content: `Directory is not empty: ${path}. Remove contents first or use run_command with rm -rf.`, is_error: true }
      }
      throw err
    }
    return { content: `Deleted directory: ${path}`, is_error: false }
  }
  unlinkSync(fullPath)
  return { content: `Deleted file: ${path}`, is_error: false }
}

function moveFile(source: string, destination: string, workingDir: string): ToolResult {
  if (!source) return { content: 'Missing required parameter: source', is_error: true }
  if (!destination) return { content: 'Missing required parameter: destination', is_error: true }
  const srcPath = resolvePath(workingDir, source)
  const dstPath = resolvePath(workingDir, destination)
  if (!existsSync(srcPath)) {
    return { content: `Source not found: ${source}`, is_error: true }
  }
  mkdirSync(dirname(dstPath), { recursive: true })
  renameSync(srcPath, dstPath)
  return { content: `Moved ${source} → ${destination}`, is_error: false }
}

function copyFile(source: string, destination: string, workingDir: string): ToolResult {
  if (!source) return { content: 'Missing required parameter: source', is_error: true }
  if (!destination) return { content: 'Missing required parameter: destination', is_error: true }
  const srcPath = resolvePath(workingDir, source)
  const dstPath = resolvePath(workingDir, destination)
  if (!existsSync(srcPath)) {
    return { content: `Source not found: ${source}`, is_error: true }
  }
  const stat = statSync(srcPath)
  if (stat.isDirectory()) {
    return { content: `Cannot copy a directory. Use run_command with cp -r instead.`, is_error: true }
  }
  mkdirSync(dirname(dstPath), { recursive: true })
  copyFileSync(srcPath, dstPath)
  return { content: `Copied ${source} → ${destination}`, is_error: false }
}

function runCommand(command: string, workingDir: string): ToolResult {
  if (!command) return { content: 'Missing required parameter: command', is_error: true }
  try {
    const output = execSync(command, {
      cwd: workingDir,
      encoding: 'utf-8',
      shell: '/bin/sh',
      timeout: 60000,
      maxBuffer: 10 * 1024 * 1024
    })
    return { content: `$ ${command}\n\n${output}`, is_error: false }
  } catch (err: any) {
    const stdout = err.stdout?.toString() || ''
    const stderr = err.stderr?.toString() || ''
    const combined = [stdout, stderr ? `STDERR:\n${stderr}` : ''].filter(Boolean).join('\n\n')
    if (err.killed) {
      return { content: `$ ${command}\n\nCommand timed out after 60 seconds\n\n${combined}`, is_error: true }
    }
    return { content: `$ ${command}\n\nExit code: ${err.status ?? 'unknown'}\n\n${combined}`, is_error: true }
  }
}

// ─── Patch / Batch / Diagnostics ─────────────────────────────────────────────

function patchFile(path: string, patch: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (!patch) return { content: 'Missing required parameter: patch', is_error: true }

  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `File not found: ${path}`, is_error: true }
  }

  const content = readFileSync(fullPath, 'utf-8')
  const lines = content.split('\n')

  // Parse unified diff hunks
  const hunkRe = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/
  const patchLines = patch.split('\n')
  const hunks: { oldStart: number; removes: string[]; adds: string[] }[] = []
  let currentHunk: { oldStart: number; removes: string[]; adds: string[] } | null = null

  for (const pl of patchLines) {
    const m = pl.match(hunkRe)
    if (m) {
      if (currentHunk) hunks.push(currentHunk)
      currentHunk = { oldStart: parseInt(m[1], 10), removes: [], adds: [] }
    } else if (currentHunk) {
      if (pl.startsWith('-')) {
        currentHunk.removes.push(pl.slice(1))
      } else if (pl.startsWith('+')) {
        currentHunk.adds.push(pl.slice(1))
      }
      // Context lines (space prefix) and "\ No newline" are skipped
    }
  }
  if (currentHunk) hunks.push(currentHunk)

  if (hunks.length === 0) {
    return { content: 'No valid hunks found in patch. Use unified diff format with @@ hunk headers.', is_error: true }
  }

  // Apply hunks in reverse order so line numbers stay valid
  hunks.sort((a, b) => b.oldStart - a.oldStart)
  let applied = 0

  for (const hunk of hunks) {
    const startIdx = hunk.oldStart - 1 // Convert 1-based to 0-based

    // Verify the lines to remove match
    let match = true
    for (let i = 0; i < hunk.removes.length; i++) {
      if (startIdx + i >= lines.length || lines[startIdx + i] !== hunk.removes[i]) {
        match = false
        break
      }
    }

    if (!match) {
      // Try fuzzy match: search nearby (±5 lines) for the removed block
      let found = -1
      for (let offset = -5; offset <= 5; offset++) {
        const tryIdx = startIdx + offset
        if (tryIdx < 0) continue
        let ok = true
        for (let i = 0; i < hunk.removes.length; i++) {
          if (tryIdx + i >= lines.length || lines[tryIdx + i] !== hunk.removes[i]) {
            ok = false; break
          }
        }
        if (ok) { found = tryIdx; break }
      }
      if (found === -1) {
        return {
          content: `Hunk at line ${hunk.oldStart} failed to apply — expected lines not found:\n${hunk.removes.map(l => `- ${l}`).join('\n')}`,
          is_error: true
        }
      }
      lines.splice(found, hunk.removes.length, ...hunk.adds)
    } else {
      lines.splice(startIdx, hunk.removes.length, ...hunk.adds)
    }
    applied++
  }

  writeFileSync(fullPath, lines.join('\n'), 'utf-8')
  return { content: `Patched ${path}: applied ${applied} hunk(s)`, is_error: false }
}

function batchEdit(
  path: string,
  edits: Array<{ search_text: string; replacement_text: string }>,
  workingDir: string
): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (!edits || !Array.isArray(edits) || edits.length === 0) {
    return { content: 'Missing or empty edits array', is_error: true }
  }

  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) {
    return { content: `File not found: ${path}`, is_error: true }
  }

  let content = readFileSync(fullPath, 'utf-8')
  const results: string[] = []

  for (let i = 0; i < edits.length; i++) {
    const edit = edits[i]
    if (!edit.search_text) {
      results.push(`Edit ${i + 1}: skipped (empty search_text)`)
      continue
    }
    const idx = content.indexOf(edit.search_text)
    if (idx === -1) {
      results.push(`Edit ${i + 1}: FAILED — search_text not found`)
      // Continue applying remaining edits — partial success is more useful than total failure
      continue
    }
    // Check uniqueness
    const secondIdx = content.indexOf(edit.search_text, idx + 1)
    if (secondIdx !== -1) {
      results.push(`Edit ${i + 1}: FAILED — search_text matches multiple locations`)
      continue
    }
    content = content.slice(0, idx) + edit.replacement_text + content.slice(idx + edit.search_text.length)
    results.push(`Edit ${i + 1}: OK`)
  }

  writeFileSync(fullPath, content, 'utf-8')
  const okCount = results.filter(r => r.includes('OK')).length
  return {
    content: `Batch edit ${path}: ${okCount}/${edits.length} succeeded\n${results.join('\n')}`,
    is_error: okCount === 0
  }
}

function getDiagnostics(path: string, tool: string | undefined, workingDir: string): ToolResult {
  const fullPath = resolvePath(workingDir, path)

  // Auto-detect tool if not specified
  if (!tool || tool === 'auto') {
    if (existsSync(join(workingDir, 'tsconfig.json'))) {
      tool = 'tsc'
    } else if (existsSync(join(workingDir, '.eslintrc.json')) ||
               existsSync(join(workingDir, '.eslintrc.js')) ||
               existsSync(join(workingDir, 'eslint.config.js')) ||
               existsSync(join(workingDir, 'eslint.config.mjs'))) {
      tool = 'eslint'
    } else if (path.endsWith('.py') || existsSync(join(workingDir, 'pyproject.toml'))) {
      tool = 'python'
    } else {
      return { content: 'Could not auto-detect diagnostic tool. Specify tool: "tsc", "eslint", or "python".', is_error: true }
    }
  }

  try {
    let cmd: string
    let args: string[]
    switch (tool) {
      case 'tsc':
        // Check for npx first; run tsc --noEmit on the file or project
        cmd = 'npx'
        args = ['tsc', '--noEmit', '--pretty', 'false']
        if (path !== '.' && !statSync(fullPath).isDirectory()) {
          // Type-checking a single file needs --isolatedModules or just check the project
          args = ['tsc', '--noEmit', '--pretty', 'false']
        }
        break
      case 'eslint':
        cmd = 'npx'
        args = ['eslint', '--format', 'compact', fullPath]
        break
      case 'python':
        cmd = 'python3'
        args = ['-m', 'py_compile', fullPath]
        break
      default:
        return { content: `Unknown diagnostic tool: ${tool}. Use "tsc", "eslint", or "python".`, is_error: true }
    }

    const output = execFileSync(cmd, args, {
      cwd: workingDir,
      encoding: 'utf-8',
      timeout: 30000,
      maxBuffer: 5 * 1024 * 1024
    })

    // Clean paths to relative
    const cleaned = output.replace(
      new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), ''
    )
    return { content: cleaned.trim() || `${tool}: No errors found.`, is_error: false }
  } catch (err: any) {
    // tsc and eslint return non-zero on errors — that's expected
    const stdout = err.stdout?.toString() || ''
    const stderr = err.stderr?.toString() || ''
    const combined = (stdout + '\n' + stderr).trim()
    if (combined) {
      const cleaned = combined.replace(
        new RegExp(workingDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '/', 'g'), ''
      )
      return { content: `${tool} diagnostics:\n\n${cleaned}`, is_error: false }
    }
    return { content: `${tool} failed: ${err.message}`, is_error: true }
  }
}

// ─── Web Tools ───────────────────────────────────────────────────────────────

/** Read Brave API key from SQLite settings table */
function getBraveApiKey(): string | null {
  const key = db.getSetting('braveApiKey')
  if (key) return key
  return process.env.BRAVE_API_KEY || null
}

async function webSearch(query: string, count?: number): Promise<ToolResult> {
  if (!query) return { content: 'Missing required parameter: query', is_error: true }
  const apiKey = getBraveApiKey()
  if (!apiKey) {
    return {
      content: 'Brave Search API key not configured. Go to About → API Keys in the app to set your Brave Search API key.',
      is_error: true
    }
  }

  const numResults = Math.min(count || 5, 20)
  const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${numResults}`

  try {
    const res = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': apiKey
      }
    })

    if (!res.ok) {
      const errText = await res.text()
      return { content: `Brave Search error (${res.status}): ${errText}`, is_error: true }
    }

    const data = await res.json() as any
    const results = data.web?.results || []

    if (results.length === 0) {
      return { content: `No results found for "${query}"`, is_error: false }
    }

    const formatted = results.map((r: any, i: number) => {
      const desc = r.description || ''
      return `${i + 1}. ${r.title}\n   ${r.url}\n   ${desc}`
    }).join('\n\n')

    return { content: `Search: "${query}" (${results.length} results)\n\n${formatted}`, is_error: false }
  } catch (err: any) {
    return { content: `Web search failed: ${err.message}`, is_error: true }
  }
}

async function fetchUrl(url: string, maxLength?: number): Promise<ToolResult> {
  if (!url) return { content: 'Missing required parameter: url', is_error: true }

  // Basic URL validation
  try {
    new URL(url)
  } catch {
    return { content: `Invalid URL: ${url}`, is_error: true }
  }

  try {
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; vMLX/1.0)',
        'Accept': 'text/html,application/json,text/plain,*/*'
      },
      signal: AbortSignal.timeout(30000)
    })

    if (!res.ok) {
      return { content: `HTTP ${res.status}: ${res.statusText} for ${url}`, is_error: true }
    }

    const contentType = res.headers.get('content-type') || ''
    let text = await res.text()

    // Strip HTML tags if content is HTML
    if (contentType.includes('html')) {
      // Remove script/style blocks first
      text = text.replace(/<script[\s\S]*?<\/script>/gi, '')
      text = text.replace(/<style[\s\S]*?<\/style>/gi, '')
      text = text.replace(/<nav[\s\S]*?<\/nav>/gi, '')
      text = text.replace(/<header[\s\S]*?<\/header>/gi, '')
      text = text.replace(/<footer[\s\S]*?<\/footer>/gi, '')
      // Convert common elements
      text = text.replace(/<br\s*\/?>/gi, '\n')
      text = text.replace(/<\/p>/gi, '\n\n')
      text = text.replace(/<\/div>/gi, '\n')
      text = text.replace(/<\/h[1-6]>/gi, '\n\n')
      text = text.replace(/<li>/gi, '- ')
      text = text.replace(/<\/li>/gi, '\n')
      // Strip remaining tags
      text = text.replace(/<[^>]+>/g, '')
      // Decode common entities
      text = text.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ')
      // Collapse whitespace
      text = text.replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim()
    }

    // Truncate
    const maxChars = maxLength || 20000
    let truncated = false
    if (text.length > maxChars) {
      text = text.slice(0, maxChars)
      truncated = true
    }

    const header = `URL: ${url} (${contentType.split(';')[0]})`
    const footer = truncated ? `\n\n[Truncated at ${maxChars} chars. ${text.length} of total fetched.]` : ''

    return { content: `${header}\n\n${text}${footer}`, is_error: false }
  } catch (err: any) {
    return { content: `Fetch failed: ${err.message}`, is_error: true }
  }
}

async function ddgSearch(query: string, count?: number): Promise<ToolResult> {
  if (!query) return { content: 'Missing required parameter: query', is_error: true }
  const numResults = Math.min(count || 5, 10)

  // DuckDuckGo HTML search — free, no API key
  const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
  try {
    const res = await fetch(searchUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      },
      signal: AbortSignal.timeout(15000)
    })

    if (!res.ok) {
      return { content: `DuckDuckGo search failed (${res.status})`, is_error: true }
    }

    const html = await res.text()

    // Parse results from DDG HTML
    const results: { title: string; url: string; snippet: string }[] = []
    // DDG HTML wraps each result in <div class="result">
    const resultBlocks = html.split(/class="result\s/)
    for (let i = 1; i < resultBlocks.length && results.length < numResults; i++) {
      const block = resultBlocks[i]
      // Extract title from <a class="result__a" ...>
      const titleMatch = block.match(/class="result__a"[^>]*>([^<]+)</)
      // Extract URL from <a class="result__url" href="...">
      const urlMatch = block.match(/class="result__url"[^>]*href="([^"]*)"/) ||
                       block.match(/class="result__a"[^>]*href="([^"]*)"/)
      // Extract snippet from <a class="result__snippet" ...>
      const snippetMatch = block.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/)

      if (titleMatch && urlMatch) {
        let url = urlMatch[1]
        // DDG wraps URLs in a redirect — extract actual URL
        const uddg = url.match(/uddg=([^&]+)/)
        if (uddg) url = decodeURIComponent(uddg[1])
        // Clean snippet: strip HTML tags and entities
        let snippet = snippetMatch ? snippetMatch[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").trim() : ''
        results.push({
          title: titleMatch[1].trim(),
          url,
          snippet
        })
      }
    }

    if (results.length === 0) {
      return { content: `No results found for "${query}"`, is_error: false }
    }

    const formatted = results.map((r, i) => {
      return `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet}`
    }).join('\n\n')

    return { content: `DuckDuckGo: "${query}" (${results.length} results)\n\n${formatted}`, is_error: false }
  } catch (err: any) {
    return { content: `DuckDuckGo search failed: ${err.message}`, is_error: true }
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}
