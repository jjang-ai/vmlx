/**
 * Built-in tool executor — runs coding tools locally in the Electron main process.
 * All file operations are sandboxed to the working directory.
 */

import { readFileSync, writeFileSync, copyFileSync, renameSync, unlinkSync, rmdirSync, mkdirSync, readdirSync, statSync, existsSync, realpathSync } from 'fs'
import { resolve, relative, dirname, basename, isAbsolute, join, sep } from 'path'
import { execFileSync, spawn, ChildProcess } from 'child_process'
import { clipboard } from 'electron'
import { db } from '../database'

// ─── Spawned Process Tracking ────────────────────────────────────────────────

interface SpawnedEntry {
  proc: ChildProcess
  stdout: string
  stderr: string
  running: boolean
  startedAt: number
}

const spawnedProcesses = new Map<string, SpawnedEntry>()

export interface ToolResult {
  content: string
  is_error: boolean
  /** For read_image: base64 data URL to inject as a multimodal content part in VLM follow-ups */
  imageDataUrl?: string
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
  if (rel.startsWith('..') || (isAbsolute(rel) && realResolved !== realWorkingDir && !realResolved.startsWith(realWorkingDir + sep))) {
    throw new Error(`Path escapes working directory: ${userPath}`)
  }
  return realResolved
}

// ─── Tool Result Limits ──────────────────────────────────────────────────────

const DEFAULT_MAX_TOOL_RESULT_CHARS = 50000 // ~50KB — prevents context overflow on large files/outputs

function truncateResult(content: string, maxChars?: number): string {
  const limit = maxChars && maxChars > 0 ? maxChars : DEFAULT_MAX_TOOL_RESULT_CHARS
  if (content.length <= limit) return content
  return content.slice(0, limit) + `\n\n[Truncated — showing first ${limit} of ${content.length} characters]`
}

// ─── Main Entry ──────────────────────────────────────────────────────────────

export async function executeBuiltinTool(
  toolName: string,
  args: Record<string, any>,
  workingDir: string,
  maxResultChars?: number
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
        result = await runCommand(args.command, workingDir); break
      case 'web_search':
        result = await webSearch(args.query, args.count); break
      case 'fetch_url':
        result = await fetchUrl(args.url, args.max_length); break
      case 'insert_text':
        result = insertText(args.path, args.line, args.text, workingDir); break
      case 'replace_lines':
        result = replaceLines(args.path, args.start_line, args.end_line, args.text, workingDir); break
      case 'get_tree':
        result = getTree(args.path || '.', args.max_depth || 4, workingDir); break
      case 'apply_regex':
        result = applyRegex(args.pattern, args.replacement, args.path, args.glob, workingDir); break
      case 'read_image':
        result = readImage(args.path, workingDir); break
      case 'spawn_process':
        result = spawnProcess(args.command, workingDir); break
      case 'get_process_output':
        result = getProcessOutput(args.pid); break
      case 'diff_files':
        result = diffFiles(args.path_a, args.path_b, workingDir); break
      case 'count_tokens':
        result = countTokens(args.text); break
      case 'clipboard_read':
        result = clipboardRead(); break
      case 'clipboard_write':
        result = clipboardWrite(args.text); break
      case 'git':
        result = gitCommand(args.command, workingDir); break
      case 'get_current_datetime':
        result = getCurrentDatetime(); break
      // ask_user is handled in chat.ts (needs IPC to renderer), not here
      default:
        return { content: `Unknown tool: ${toolName}`, is_error: true }
    }
    // Truncate large results to prevent context overflow in follow-up requests
    result.content = truncateResult(result.content, maxResultChars)
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

  let header = `File: ${fullPath} (${totalLines} lines)`
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
  return { content: `Wrote ${fullPath} (${lines} lines, ${bytes} bytes)`, is_error: false }
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
  return { content: `Created directory: ${fullPath}`, is_error: false }
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

async function runCommand(command: string, workingDir: string): Promise<ToolResult> {
  if (!command) return { content: 'Missing required parameter: command', is_error: true }
  return new Promise((resolve) => {
    let stdout = '', stderr = ''
    let killReason = ''
    const proc = spawn('/bin/sh', ['-c', command], {
      cwd: workingDir,
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe']
    })
    proc.stdout?.on('data', (d: Buffer) => {
      stdout += d.toString()
      if (!killReason && stdout.length > 10 * 1024 * 1024) { killReason = 'Output exceeded 10MB limit'; proc.kill() }
    })
    proc.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString()
      if (!killReason && stderr.length > 10 * 1024 * 1024) { killReason = 'Stderr exceeded 10MB limit'; proc.kill() }
    })
    const timer = setTimeout(() => { if (!killReason) { killReason = 'Command timed out after 60 seconds'; proc.kill() } }, 60000)
    proc.on('close', (code, signal) => {
      clearTimeout(timer)
      if (killReason) {
        const combined = [stdout, stderr ? `STDERR:\n${stderr}` : ''].filter(Boolean).join('\n\n')
        resolve({ content: `$ ${command}\n\n${killReason}\n\n${combined}`, is_error: true })
      } else if (code === 0) {
        resolve({ content: `$ ${command}\n\n${stdout}`, is_error: false })
      } else if (code === null && signal) {
        // Killed by external signal (e.g., OOM killer sends SIGKILL)
        const combined = [stdout, stderr ? `STDERR:\n${stderr}` : ''].filter(Boolean).join('\n\n')
        resolve({ content: `$ ${command}\n\nProcess killed by signal ${signal}\n\n${combined}`, is_error: true })
      } else {
        const combined = [stdout, stderr ? `STDERR:\n${stderr}` : ''].filter(Boolean).join('\n\n')
        resolve({ content: `$ ${command}\n\nExit code: ${code}\n\n${combined}`, is_error: true })
      }
    })
    proc.on('error', (err: Error) => {
      clearTimeout(timer)
      resolve({ content: `$ ${command}\n\nProcess error: ${err.message}`, is_error: true })
    })
  })
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

  const okCount = results.filter(r => r.includes('OK')).length
  if (okCount > 0) {
    writeFileSync(fullPath, content, 'utf-8')
  }
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
        // Always run project-wide tsc --noEmit (single-file checking requires --isolatedModules
        // and misses cross-file type errors). The path argument selects auto-detect only.
        cmd = 'npx'
        args = ['tsc', '--noEmit', '--pretty', 'false']
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

  // URL validation: only allow http/https schemes
  try {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return { content: `Unsupported URL scheme: ${parsed.protocol} — only http and https are allowed`, is_error: true }
    }
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
    const originalLength = text.length
    let truncated = false
    if (text.length > maxChars) {
      text = text.slice(0, maxChars)
      truncated = true
    }

    const header = `URL: ${url} (${contentType.split(';')[0]})`
    const footer = truncated ? `\n\n[Truncated at ${maxChars} chars. Full response was ${originalLength} chars.]` : ''

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

// ─── Line-Based Editing ──────────────────────────────────────────────────────

function insertText(path: string, line: number, text: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (text === undefined || text === null) return { content: 'Missing required parameter: text', is_error: true }
  if (line === undefined || line === null) return { content: 'Missing required parameter: line', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) return { content: `File not found: ${path}`, is_error: true }

  const content = readFileSync(fullPath, 'utf-8')
  const lines = content.split('\n')
  const insertIdx = (line <= 0 || line > lines.length) ? lines.length : line - 1
  const newLines = text.split('\n')
  lines.splice(insertIdx, 0, ...newLines)
  writeFileSync(fullPath, lines.join('\n'), 'utf-8')
  return { content: `Inserted ${newLines.length} line(s) at line ${insertIdx + 1} in ${path}`, is_error: false }
}

function replaceLines(path: string, startLine: number, endLine: number, text: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  if (text === undefined || text === null) return { content: 'Missing required parameter: text', is_error: true }
  if (!Number.isFinite(startLine) || !Number.isFinite(endLine)) {
    return { content: 'Missing or invalid start_line/end_line parameters', is_error: true }
  }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) return { content: `File not found: ${path}`, is_error: true }

  const content = readFileSync(fullPath, 'utf-8')
  const lines = content.split('\n')
  if (startLine < 1 || startLine > lines.length) {
    return { content: `start_line ${startLine} out of range (file has ${lines.length} lines)`, is_error: true }
  }
  if (endLine < startLine || endLine > lines.length) {
    return { content: `end_line ${endLine} out of range (start_line=${startLine}, file has ${lines.length} lines)`, is_error: true }
  }

  const newLines = text.split('\n')
  const removedCount = endLine - startLine + 1
  lines.splice(startLine - 1, removedCount, ...newLines)
  writeFileSync(fullPath, lines.join('\n'), 'utf-8')
  return { content: `Replaced lines ${startLine}-${endLine} (${removedCount} lines) with ${newLines.length} line(s) in ${path}`, is_error: false }
}

// ─── Project Tree ────────────────────────────────────────────────────────────

function getTree(path: string, maxDepth: number, workingDir: string): ToolResult {
  const dir = resolvePath(workingDir, path)
  if (!existsSync(dir)) return { content: `Directory not found: ${path}`, is_error: true }

  // Try git ls-files for .gitignore-aware listing
  try {
    const output = execFileSync('git', ['ls-files', '--others', '--cached', '--exclude-standard'], {
      cwd: dir, encoding: 'utf-8', timeout: 10000
    })
    if (output.trim()) {
      const files = output.trim().split('\n').sort()
      const tree = buildTreeFromPaths(files, maxDepth)
      return { content: `Project tree: ${path}\n\n${tree}`, is_error: false }
    }
  } catch { /* not a git repo - fallback */ }

  // Fallback: manual tree walk
  const SKIP_DIRS = new Set(['node_modules', '.git', '__pycache__', '.venv', 'venv', '.tox', 'dist', 'build', '.next', '.cache', '.DS_Store'])
  const treeLines: string[] = [path + '/']
  let count = 0
  const MAX = 500

  function walk(d: string, prefix: string, depth: number): void {
    if (depth > maxDepth || count >= MAX) return
    let items: any[]
    try { items = readdirSync(d, { withFileTypes: true }) as any[] } catch { return }
    items = items.filter(i => !SKIP_DIRS.has(i.name) && (!i.name.startsWith('.') || i.name === '.env'))
    items.sort((a, b) => {
      if (a.isDirectory() && !b.isDirectory()) return -1
      if (!a.isDirectory() && b.isDirectory()) return 1
      return a.name.localeCompare(b.name)
    })
    for (let i = 0; i < items.length && count < MAX; i++) {
      const item = items[i]
      const isLast = i === items.length - 1
      count++
      if (item.isDirectory()) {
        treeLines.push(`${prefix}${isLast ? '└── ' : '├── '}${item.name}/`)
        walk(join(d, item.name), prefix + (isLast ? '    ' : '│   '), depth + 1)
      } else {
        treeLines.push(`${prefix}${isLast ? '└── ' : '├── '}${item.name}`)
      }
    }
    if (count >= MAX) treeLines.push(`${prefix}... (truncated at ${MAX} entries)`)
  }

  walk(dir, '', 1)
  return { content: treeLines.join('\n'), is_error: false }
}

function buildTreeFromPaths(files: string[], maxDepth: number): string {
  const root: Record<string, any> = {}
  for (const file of files) {
    const parts = file.split('/')
    let node = root
    for (let i = 0; i < parts.length; i++) {
      if (i === parts.length - 1) { node[parts[i]] = null }
      else { if (!node[parts[i]]) node[parts[i]] = {}; node = node[parts[i]] }
    }
  }

  const renderLines: string[] = ['.']
  function render(node: Record<string, any>, prefix: string, depth: number): void {
    if (depth > maxDepth) { renderLines.push(`${prefix}...`); return }
    const entries = Object.keys(node).sort((a, b) => {
      const aDir = node[a] !== null, bDir = node[b] !== null
      if (aDir && !bDir) return -1
      if (!aDir && bDir) return 1
      return a.localeCompare(b)
    })
    for (let i = 0; i < entries.length; i++) {
      const name = entries[i]
      const isLast = i === entries.length - 1
      const connector = isLast ? '└── ' : '├── '
      const extension = isLast ? '    ' : '│   '
      if (node[name] !== null) {
        renderLines.push(`${prefix}${connector}${name}/`)
        render(node[name], prefix + extension, depth + 1)
      } else {
        renderLines.push(`${prefix}${connector}${name}`)
      }
    }
  }
  render(root, '', 1)
  return renderLines.join('\n')
}

// ─── Regex Replace ───────────────────────────────────────────────────────────

function applyRegex(pattern: string, replacement: string, path: string, glob: string | undefined, workingDir: string): ToolResult {
  if (!pattern) return { content: 'Missing required parameter: pattern', is_error: true }
  if (replacement === undefined) return { content: 'Missing required parameter: replacement', is_error: true }

  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) return { content: `Path not found: ${path}`, is_error: true }

  let regex: RegExp
  try { regex = new RegExp(pattern, 'g') } catch (e) {
    return { content: `Invalid regex: ${(e as Error).message}`, is_error: true }
  }

  const stat = statSync(fullPath)
  const targetFiles: string[] = []
  if (stat.isFile()) {
    targetFiles.push(fullPath)
  } else {
    try {
      const fdArgs = ['--color=never', '--type', 'f']
      if (glob) fdArgs.push('--glob', glob)
      fdArgs.push(fullPath)
      const output = execFileSync('fd', fdArgs, { encoding: 'utf-8', timeout: 10000 })
      targetFiles.push(...output.trim().split('\n').filter(Boolean))
    } catch {
      try {
        const findArgs = [fullPath, '-type', 'f', '-not', '-path', '*/node_modules/*', '-not', '-path', '*/.git/*']
        if (glob) findArgs.push('-name', glob)
        const output = execFileSync('find', findArgs, { encoding: 'utf-8', timeout: 10000 })
        targetFiles.push(...output.trim().split('\n').filter(Boolean))
      } catch (e) {
        return { content: `Failed to find files: ${(e as Error).message}`, is_error: true }
      }
    }
  }

  const results: string[] = []
  let totalReplacements = 0
  for (const file of targetFiles) {
    try {
      const content = readFileSync(file, 'utf-8')
      const matches = content.match(regex)
      if (matches && matches.length > 0) {
        writeFileSync(file, content.replace(regex, replacement), 'utf-8')
        results.push(`${relative(workingDir, file)}: ${matches.length} replacement(s)`)
        totalReplacements += matches.length
      }
    } catch { /* skip binary/unreadable files */ }
  }

  if (totalReplacements === 0) {
    return { content: `No matches found for /${pattern}/ in ${path}${glob ? ` (${glob})` : ''}`, is_error: false }
  }
  return { content: `Applied /${pattern}/: ${totalReplacements} replacement(s) in ${results.length} file(s)\n\n${results.join('\n')}`, is_error: false }
}

// ─── Image Reading ───────────────────────────────────────────────────────────

function readImage(path: string, workingDir: string): ToolResult {
  if (!path) return { content: 'Missing required parameter: path', is_error: true }
  const fullPath = resolvePath(workingDir, path)
  if (!existsSync(fullPath)) return { content: `File not found: ${path}`, is_error: true }

  const ext = path.toLowerCase().split('.').pop()
  const mimeTypes: Record<string, string> = {
    png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
    gif: 'image/gif', webp: 'image/webp', svg: 'image/svg+xml'
  }
  const mime = mimeTypes[ext || '']
  if (!mime) return { content: `Unsupported image format: .${ext}. Supported: png, jpg, gif, webp, svg`, is_error: true }

  const stat = statSync(fullPath)
  if (stat.size > 10 * 1024 * 1024) return { content: `Image too large: ${formatBytes(stat.size)} (max 10MB)`, is_error: true }

  const base64 = readFileSync(fullPath).toString('base64')
  const dataUrl = `data:${mime};base64,${base64}`
  // Return metadata as content (safe for context) + image data URL separately
  // for VLM follow-ups. The dataUrl is injected as a multimodal content part
  // by chat.ts so VL models can actually "see" the image.
  return {
    content: `Image loaded: ${path} (${mime}, ${formatBytes(stat.size)}). The image has been attached for visual analysis.`,
    is_error: false,
    imageDataUrl: dataUrl
  }
}

// ─── Background Process ──────────────────────────────────────────────────────

/** Clean up finished processes to prevent unbounded memory growth. */
function cleanupFinishedProcesses(): void {
  const MAX_FINISHED_AGE = 600_000 // 10 minutes after stop
  const now = Date.now()
  for (const [id, entry] of spawnedProcesses) {
    if (!entry.running && now - entry.startedAt > MAX_FINISHED_AGE) {
      spawnedProcesses.delete(id)
    }
  }
}

function spawnProcess(command: string, workingDir: string): ToolResult {
  if (!command) return { content: 'Missing required parameter: command', is_error: true }

  // Clean up old finished processes before creating new ones
  cleanupFinishedProcesses()

  const id = `proc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`

  const proc = spawn('/bin/sh', ['-c', command], {
    cwd: workingDir,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  const entry: SpawnedEntry = { proc, stdout: '', stderr: '', running: true, startedAt: Date.now() }
  proc.stdout?.on('data', (d: Buffer) => {
    entry.stdout += d.toString()
    if (entry.stdout.length > 100000) entry.stdout = entry.stdout.slice(-50000)
  })
  proc.stderr?.on('data', (d: Buffer) => {
    entry.stderr += d.toString()
    if (entry.stderr.length > 100000) entry.stderr = entry.stderr.slice(-50000)
  })
  proc.on('close', () => { entry.running = false })
  proc.on('error', (err: Error) => { entry.stderr += `\nProcess error: ${err.message}`; entry.running = false })

  // Auto-kill after 5 minutes
  setTimeout(() => { if (entry.running) { try { proc.kill() } catch { } } }, 300000)
  spawnedProcesses.set(id, entry)

  return { content: `Started process: ${command}\nPID: ${id}\nUse get_process_output(pid="${id}") to check output.`, is_error: false }
}

function getProcessOutput(pid: string): ToolResult {
  if (!pid) return { content: 'Missing required parameter: pid', is_error: true }
  const entry = spawnedProcesses.get(pid)
  if (!entry) return { content: `Process not found: ${pid}. It may have been cleaned up or never existed.`, is_error: true }

  const status = entry.running ? 'RUNNING' : 'STOPPED'
  const elapsed = ((Date.now() - entry.startedAt) / 1000).toFixed(1)
  let output = `Process ${pid} - ${status} (${elapsed}s elapsed)\n`
  if (entry.stdout) output += `\nSTDOUT:\n${entry.stdout.slice(-20000)}`
  if (entry.stderr) output += `\nSTDERR:\n${entry.stderr.slice(-10000)}`
  if (!entry.stdout && !entry.stderr) output += '\n(no output yet)'
  return { content: output, is_error: false }
}

// ─── Diff ────────────────────────────────────────────────────────────────────

function diffFiles(pathA: string, pathB: string | undefined, workingDir: string): ToolResult {
  if (!pathA) return { content: 'Missing required parameter: path_a', is_error: true }

  if (pathB) {
    const fullA = resolvePath(workingDir, pathA)
    const fullB = resolvePath(workingDir, pathB)
    if (!existsSync(fullA)) return { content: `File not found: ${pathA}`, is_error: true }
    if (!existsSync(fullB)) return { content: `File not found: ${pathB}`, is_error: true }
    try {
      const output = execFileSync('diff', ['-u', fullA, fullB], {
        encoding: 'utf-8', timeout: 10000, maxBuffer: 5 * 1024 * 1024
      })
      return { content: output || `Files are identical: ${pathA} and ${pathB}`, is_error: false }
    } catch (err: any) {
      if (err.status === 1 && err.stdout) {
        return { content: `Diff: ${pathA} vs ${pathB}\n\n${err.stdout}`, is_error: false }
      }
      return { content: `Diff failed: ${err.message}`, is_error: true }
    }
  } else {
    try {
      const output = execFileSync('git', ['diff', 'HEAD', '--', pathA], {
        cwd: workingDir, encoding: 'utf-8', timeout: 10000
      })
      return { content: output || `No changes from HEAD: ${pathA}`, is_error: false }
    } catch (err: any) {
      if (err.stdout) return { content: `Git diff: ${pathA}\n\n${err.stdout}`, is_error: false }
      return { content: `Git diff failed (not a git repo?): ${err.message}`, is_error: true }
    }
  }
}

// ─── Token Estimation ────────────────────────────────────────────────────────

function countTokens(text: string): ToolResult {
  if (!text) return { content: 'Missing required parameter: text', is_error: true }
  const charEstimate = Math.ceil(text.length / 4)
  const words = text.split(/\s+/).filter(Boolean).length
  const wordEstimate = Math.ceil(words * 1.3)
  const estimate = Math.ceil((charEstimate + wordEstimate) / 2)
  return {
    content: `Text: ${text.length} chars, ${words} words\nEstimated tokens: ~${estimate} (char-based: ~${charEstimate}, word-based: ~${wordEstimate})`,
    is_error: false
  }
}

// ─── Clipboard ───────────────────────────────────────────────────────────────

function clipboardRead(): ToolResult {
  const text = clipboard.readText()
  if (!text) return { content: '(clipboard is empty)', is_error: false }
  return { content: `Clipboard (${text.length} chars):\n\n${text}`, is_error: false }
}

function clipboardWrite(text: string): ToolResult {
  if (text === undefined || text === null) return { content: 'Missing required parameter: text', is_error: true }
  clipboard.writeText(text)
  return { content: `Written ${text.length} characters to clipboard.`, is_error: false }
}

// ─── Date/Time ──────────────────────────────────────────────────────────────

function getCurrentDatetime(): ToolResult {
  const now = new Date()
  const dateStr = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true })
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  return { content: `${dateStr}, ${timeStr} (${timezone})`, is_error: false }
}

// ─── Git ─────────────────────────────────────────────────────────────────────

function gitCommand(command: string, workingDir: string): ToolResult {
  if (!command) return { content: 'Missing required parameter: command', is_error: true }

  // Block shell metacharacters that could enable command injection via /bin/sh -c.
  // Includes newlines (\n, \r) which bypass single-line metachar checks but still
  // inject separate commands when passed to /bin/sh -c, and < > for redirection.
  if (/[;|&`$(){}<>\n\r]/.test(command)) {
    return { content: `Blocked: "git ${command}" contains shell metacharacters. Use run_command for complex shell commands.`, is_error: true }
  }

  // Block destructive operations
  const dangerous = [
    /push\s+.*--force/, /push\s+-f\b/,
    /reset\s+--hard/,
    /clean\s+-f/, /clean\s+.*-fd/,
    /branch\s+-D\b/
  ]
  for (const re of dangerous) {
    if (re.test(command)) {
      return { content: `Blocked: "git ${command}" is a destructive operation. Use run_command if you really need this.`, is_error: true }
    }
  }

  // Use shell execution via /bin/sh -c to handle quoted args correctly
  // (e.g., git commit -m "fix: some bug" would break with naive split)
  try {
    const output = execFileSync('/bin/sh', ['-c', `git ${command}`], {
      cwd: workingDir,
      encoding: 'utf-8',
      timeout: 30000,
      maxBuffer: 10 * 1024 * 1024
    })
    return { content: `$ git ${command}\n\n${output}`, is_error: false }
  } catch (err: any) {
    const stdout = err.stdout?.toString() || ''
    const stderr = err.stderr?.toString() || ''
    const combined = [stdout, stderr ? `STDERR:\n${stderr}` : ''].filter(Boolean).join('\n\n')
    if (err.status === 1 && command.startsWith('diff')) {
      return { content: `$ git ${command}\n\n${combined}`, is_error: false }
    }
    return { content: `$ git ${command}\n\nExit code: ${err.status ?? 'unknown'}\n\n${combined}`, is_error: true }
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}
