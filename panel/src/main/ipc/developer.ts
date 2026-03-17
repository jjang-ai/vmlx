import { ipcMain, BrowserWindow, dialog } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { sessionManager } from '../sessions'
import { db } from '../database'

let activeProcess: ChildProcess | null = null
let cancelled = false

function buildCliEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: undefined }
  try {
    const hfToken = db.getSetting('hf_api_key')
    if (hfToken) env.HF_TOKEN = hfToken
  } catch { /* DB not ready yet — skip token injection */ }
  return env
}

/** Resolve the CLI spawn command + args using the same path as sessions.ts */
function resolveCliSpawn(subcommandArgs: string[]): { cmd: string; args: string[]; env: Record<string, string | undefined> } {
  const engineResult = sessionManager.findEnginePath()
  const env = buildCliEnv()
  if (engineResult?.type === 'bundled') {
    return {
      cmd: engineResult.pythonPath,
      args: ['-s', '-m', 'vmlx_engine.cli', ...subcommandArgs],
      env,
    }
  }
  if (engineResult?.type === 'system') {
    return {
      cmd: engineResult.binaryPath,
      args: subcommandArgs,
      env,
    }
  }
  // Last resort: bare python3 (may not have vmlx_engine installed)
  return {
    cmd: 'python3',
    args: ['-s', '-m', 'vmlx_engine.cli', ...subcommandArgs],
    env,
  }
}

function emitLog(getWin: () => BrowserWindow | null, data: string) {
  const win = getWin()
  if (win && !win.isDestroyed()) {
    win.webContents.send('developer:log', { data })
  }
}

function emitComplete(getWin: () => BrowserWindow | null, result: { success: boolean; cancelled?: boolean; error?: string }) {
  const win = getWin()
  if (win && !win.isDestroyed()) {
    win.webContents.send('developer:complete', result)
  }
}

/** Run a quick CLI command that buffers output and returns it. 30s timeout. */
async function runQuickCommand(subcommand: string, args: string[]): Promise<{ success: boolean; output: string; error?: string }> {
  const spawn_info = resolveCliSpawn([subcommand, ...args])
  return new Promise((resolve) => {
    const proc = spawn(spawn_info.cmd, spawn_info.args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: spawn_info.env,
    })
    let stdout = ''
    let stderr = ''
    let resolved = false

    proc.stdout?.on('data', (d) => { stdout += d.toString() })
    proc.stderr?.on('data', (d) => { stderr += d.toString() })

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true
        proc.kill('SIGKILL')
        resolve({ success: false, output: '', error: 'Command timed out after 30 seconds' })
      }
    }, 30000)

    proc.on('close', (code) => {
      clearTimeout(timer)
      if (!resolved) {
        resolved = true
        resolve({
          success: code === 0,
          output: stdout,
          error: code !== 0 ? stderr || 'Command failed' : undefined,
        })
      }
    })
    proc.on('error', (err) => {
      clearTimeout(timer)
      if (!resolved) {
        resolved = true
        resolve({ success: false, output: '', error: err.message })
      }
    })
  })
}

/** Run a long-running CLI command that streams log output via IPC events. */
async function runStreamingCommand(
  getWin: () => BrowserWindow | null,
  cliArgs: string[],
  failureMessage: string,
): Promise<{ success: boolean; cancelled?: boolean; error?: string }> {
  if (activeProcess) {
    return { success: false, error: 'Another operation is already running' }
  }

  const spawn_info = resolveCliSpawn(cliArgs)
  cancelled = false

  return new Promise((resolve) => {
    const proc = spawn(spawn_info.cmd, spawn_info.args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: spawn_info.env,
    })
    activeProcess = proc

    proc.stdout?.on('data', (data) => emitLog(getWin, data.toString()))
    proc.stderr?.on('data', (data) => emitLog(getWin, data.toString()))

    proc.on('close', (code) => {
      activeProcess = null
      if (cancelled) {
        emitComplete(getWin, { success: false, cancelled: true })
        resolve({ success: false, cancelled: true, error: 'Cancelled' })
      } else {
        const success = code === 0
        emitComplete(getWin, { success })
        resolve({ success, cancelled: false, error: success ? undefined : failureMessage })
      }
    })
    proc.on('error', (err) => {
      activeProcess = null
      emitComplete(getWin, { success: false, error: err.message })
      resolve({ success: false, error: err.message })
    })
  })
}

export function registerDeveloperHandlers(getWin: () => BrowserWindow | null) {
  ipcMain.handle('developer:info', async (_e, modelPath: string) => {
    return runQuickCommand('info', [modelPath])
  })

  ipcMain.handle('developer:doctor', async (_e, modelPath: string, options: { noInference?: boolean }) => {
    const args = ['doctor', modelPath]
    if (options?.noInference) args.push('--no-inference')
    return runStreamingCommand(getWin, args, 'Doctor found issues')
  })

  ipcMain.handle('developer:convert', async (_e, args: {
    model: string
    output?: string
    bits: number
    groupSize: number
    mode?: string
    dtype?: string
    force?: boolean
    skipVerify?: boolean
    trustRemoteCode?: boolean
    jangProfile?: string
    jangMethod?: string
    calibrationMethod?: string
    imatrixPath?: string
    useAwq?: boolean
  }) => {
    const cliArgs = ['convert', args.model]

    if (args.jangProfile) {
      // JANG adaptive quantization
      cliArgs.push('--jang-profile', args.jangProfile)
      if (args.jangMethod) cliArgs.push('--jang-method', args.jangMethod)
      if (args.calibrationMethod && args.calibrationMethod !== 'weights') {
        cliArgs.push('--calibration-method', args.calibrationMethod)
      }
      if (args.imatrixPath) cliArgs.push('--imatrix-path', args.imatrixPath)
      if (args.useAwq) cliArgs.push('--use-awq')
    } else {
      // MLX uniform quantization
      cliArgs.push('--bits', args.bits.toString(), '--group-size', args.groupSize.toString())
      if (args.mode && args.mode !== 'default') cliArgs.push('--mode', args.mode)
      if (args.dtype) cliArgs.push('--dtype', args.dtype)
    }

    if (args.output) cliArgs.push('--output', args.output)
    if (args.force) cliArgs.push('--force')
    if (args.skipVerify) cliArgs.push('--skip-verify')
    if (args.trustRemoteCode) cliArgs.push('--trust-remote-code')
    return runStreamingCommand(getWin, cliArgs, 'Conversion failed')
  })

  ipcMain.handle('developer:cancelOp', async () => {
    if (activeProcess) {
      cancelled = true
      activeProcess.kill('SIGTERM')
      const proc = activeProcess
      setTimeout(() => {
        try { proc.kill('SIGKILL') } catch { /* already dead */ }
      }, 3000)
      return { success: true }
    }
    return { success: false, error: 'No active operation' }
  })

  ipcMain.handle('developer:browseOutputDir', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory'],
      title: 'Select Output Directory',
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })
}

export function killActiveOperation() {
  if (activeProcess) {
    const proc = activeProcess
    activeProcess = null
    cancelled = true
    proc.kill('SIGTERM')
    setTimeout(() => {
      try { proc.kill('SIGKILL') } catch { /* already dead */ }
    }, 3000)
  }
}
