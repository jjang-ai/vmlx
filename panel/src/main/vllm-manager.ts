import { exec as execCallback, spawn, execSync, ChildProcess } from 'child_process'
import { promisify } from 'util'
import { existsSync, readFileSync, realpathSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import { app } from 'electron'

const exec = promisify(execCallback)

export type InstallMethod = 'uv' | 'pip' | 'brew' | 'conda' | 'manual' | 'bundled' | 'unknown'

export interface VllmInstallation {
  installed: boolean
  path?: string
  version?: string
  method?: InstallMethod
  bundled?: boolean
}

export interface AvailableInstaller {
  method: 'uv' | 'pip'
  path: string
  label: string
}

// Common installation paths — uv first (recommended), then pip/brew/conda
const SEARCH_PATHS = [
  join(homedir(), '.local', 'bin', 'vmlx-engine'),     // uv tool / pip --user
  '/opt/homebrew/bin/vmlx-engine',                      // Homebrew (Apple Silicon)
  '/usr/local/bin/vmlx-engine',                         // Homebrew (Intel)
  '/usr/bin/vmlx-engine',                               // System pip
  join(homedir(), 'miniforge3', 'bin', 'vmlx-engine'), // Miniforge
  join(homedir(), 'anaconda3', 'bin', 'vmlx-engine'),  // Anaconda
  join(homedir(), 'miniconda3', 'bin', 'vmlx-engine')  // Miniconda
]

/**
 * Get the path to bundled Python interpreter (standalone distribution).
 * Returns null if not packaged or bundled Python doesn't exist.
 */
export function getBundledPythonPath(): string | null {
  if (!app.isPackaged) return null
  const pythonPath = join(process.resourcesPath, 'bundled-python', 'python', 'bin', 'python3')
  if (existsSync(pythonPath)) return pythonPath
  return null
}

/**
 * Check if vmlx-engine is installed and where
 */
export async function checkVllmInstallation(): Promise<VllmInstallation> {
  console.log('[vLLM Manager] Checking installation...')

  // 0. Check bundled Python first (standalone distribution)
  const bundledPython = getBundledPythonPath()
  if (bundledPython) {
    try {
      const ver = execSync(`"${bundledPython}" -s -c "import vmlx_engine; print(vmlx_engine.__version__)"`, {
        encoding: 'utf-8',
        timeout: 10000,
        env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
      }).trim()
      console.log(`[vLLM Manager] Found bundled Python with vmlx_engine ${ver}`)
      return { installed: true, path: bundledPython, version: ver, method: 'bundled', bundled: true }
    } catch (_) {
      console.log('[vLLM Manager] Bundled Python found but vmlx_engine import failed, falling through to system')
    }
  }

  // 1. Check common paths
  for (const path of SEARCH_PATHS) {
    if (existsSync(path)) {
      console.log(`[vLLM Manager] Found at: ${path}`)
      const version = await getVersionFromBinary(path)
      const method = detectInstallMethod(path)
      return { installed: true, path, version, method }
    }
  }

  // 2. Check PATH
  try {
    const result = await exec('which vmlx-engine')
    const path = result.stdout.trim()

    if (path) {
      console.log(`[vLLM Manager] Found in PATH: ${path}`)
      const version = await getVersionFromBinary(path)
      const method = detectInstallMethod(path)
      return { installed: true, path, version, method }
    }
  } catch (_) {
    // Not in PATH
  }

  // 3. Not found
  console.log('[vLLM Manager] Not installed')
  return { installed: false }
}

/**
 * Get version from vmlx-engine binary
 */
async function getVersionFromBinary(path: string): Promise<string> {
  // Get version via Python package metadata (works with editable installs)
  try {
    const shebangResult = await exec(`head -1 "${path}"`)
    const shebang = shebangResult.stdout.trim().replace(/^#\!/, '').trim()
    if (shebang) {
      const pyResult = await exec(`"${shebang}" -c "import importlib.metadata; print(importlib.metadata.version('vmlx-engine'))"`)
      const ver = pyResult.stdout.trim()
      if (/^\d+\.\d+\.\d+/.test(ver)) {
        console.log(`[vLLM Manager] Version: ${ver}`)
        return ver
      }
    }
  } catch (_) { /* fallback below */ }

  // Fallback: try --version flag
  try {
    const result = await exec(`"${path}" --version 2>&1`)
    const match = (result.stdout || result.stderr).match(/(\d+\.\d+\.\d+)/)
    if (match) {
      console.log(`[vLLM Manager] Version: ${match[1]}`)
      return match[1]
    }
  } catch (_) { /* not supported */ }

  return 'unknown'
}

/**
 * Detect installation method from path
 */
function detectInstallMethod(path: string): InstallMethod {
  if (path.includes('uv/tools') || path.includes('uv\\tools')) {
    return 'uv'
  }
  // Check if this is a uv-managed binary (symlink in ~/.local/bin pointing to uv/tools)
  if (path.includes('.local/bin') || path.includes('.local\\bin')) {
    try {
      const resolved = realpathSync(path)
      if (resolved.includes('uv/tools') || resolved.includes('uv\\tools')) {
        return 'uv'
      }
    } catch (_) { }
  }
  if (path.includes('homebrew') || path.includes('Homebrew')) {
    return 'brew'
  }
  if (path.includes('.local') || path.includes('site-packages')) {
    return 'pip'
  }
  if (path.includes('conda') || path.includes('miniforge')) {
    return 'conda'
  }
  if (path.includes('/usr/local') || path.includes('/usr/bin')) {
    return 'manual'
  }
  return 'unknown'
}

/**
 * Detect available install methods on this system.
 * Returns ordered list: uv first (preferred), then pip.
 */
export async function detectAvailableInstallers(): Promise<AvailableInstaller[]> {
  const installers: AvailableInstaller[] = []

  // Check for uv
  const uvPaths = [
    join(homedir(), '.local', 'bin', 'uv'),
    '/opt/homebrew/bin/uv',
    '/usr/local/bin/uv'
  ]
  for (const uvPath of uvPaths) {
    if (existsSync(uvPath)) {
      installers.push({ method: 'uv', path: uvPath, label: 'uv (Recommended)' })
      break
    }
  }
  if (installers.length === 0) {
    try {
      const result = await exec('which uv')
      const uvPath = result.stdout.trim()
      if (uvPath) {
        installers.push({ method: 'uv', path: uvPath, label: 'uv (Recommended)' })
      }
    } catch (_) { }
  }

  // Check for pip3 with Python >= 3.10
  const pipPaths = [
    '/opt/homebrew/bin/pip3',
    '/usr/local/bin/pip3',
    '/usr/bin/pip3'
  ]
  for (const pipPath of pipPaths) {
    if (existsSync(pipPath)) {
      // Verify it uses Python >= 3.10
      try {
        const result = await exec(`"${pipPath}" --version 2>&1`)
        const match = result.stdout.match(/python (\d+\.\d+)/i)
        if (match) {
          const [major, minor] = match[1].split('.').map(Number)
          if (major > 3 || (major === 3 && minor >= 10)) {
            installers.push({ method: 'pip', path: pipPath, label: `pip (Python ${match[1]})` })
            break
          }
        }
      } catch (_) { }
    }
  }

  return installers
}

/**
 * Find the bundled vmlx-engine source directory.
 * In packaged app: Resources/vmlx-engine-source/
 * In dev mode: monorepo root (../  from panel/)
 */
export function getBundledSourcePath(): string | null {
  // Packaged app: extraResources lands in process.resourcesPath
  if (app.isPackaged) {
    const bundled = join(process.resourcesPath, 'vmlx-engine-source')
    if (existsSync(join(bundled, 'pyproject.toml')) && existsSync(join(bundled, 'vmlx_engine'))) {
      return bundled
    }
  }

  // Dev mode: monorepo root is one level up from panel/
  const devPath = join(app.getAppPath(), '..')
  if (existsSync(join(devPath, 'pyproject.toml')) && existsSync(join(devPath, 'vmlx_engine'))) {
    return devPath
  }

  return null
}

/**
 * Build the command+args for install or upgrade.
 * Prefers bundled source over PyPI to carry our custom patches.
 */
function buildInstallCommand(
  method: 'uv' | 'pip',
  action: 'install' | 'upgrade',
  installerPath?: string
): { cmd: string; args: string[] } {
  const bundledSource = getBundledSourcePath()
  // Use bundled source path if available, otherwise fall back to PyPI package name
  const pkg = bundledSource || 'vmlx-engine'

  if (method === 'uv') {
    const cmd = installerPath || 'uv'
    if (action === 'install') {
      return { cmd, args: ['tool', 'install', pkg] }
    } else {
      // uv tool upgrade doesn't support local paths — reinstall with --force
      return bundledSource
        ? { cmd, args: ['tool', 'install', '--force', pkg] }
        : { cmd, args: ['tool', 'upgrade', 'vmlx-engine'] }
    }
  } else {
    const cmd = installerPath || 'pip3'
    if (action === 'install') {
      return { cmd, args: ['install', '--user', pkg] }
    } else {
      return { cmd, args: ['install', '--upgrade', '--user', pkg] }
    }
  }
}

// Track active install process for cancellation
let activeInstall: ChildProcess | null = null

/**
 * Install or upgrade vmlx-engine with streaming output.
 * Calls onLog for each line of output, onComplete when done.
 *
 * method='bundled-update' reinstalls vmlx-engine from bundled source into bundled Python
 * (fast, no-deps reinstall for engine updates).
 */
export function installVllmStreaming(
  method: 'uv' | 'pip' | 'bundled-update',
  action: 'install' | 'upgrade',
  installerPath: string | undefined,
  onLog: (data: string) => void,
  onComplete: (result: { success: boolean; error?: string }) => void
): void {
  // MAS App Sandbox completely forbids downloading executable dependencies or writing to Contents/Resources
  if (app.isPackaged && process.mas) {
    onComplete({
      success: false,
      error: 'In-app engine updates are disabled in the Mac App Store version due to App Sandbox constraints. Please update vMLX via the App Store to receive engine updates.'
    })
    return
  }

  if (activeInstall) {
    onComplete({ success: false, error: 'An install/update is already in progress' })
    return
  }

  let cmd: string
  let args: string[]

  if (method === 'bundled-update') {
    const bundledPython = getBundledPythonPath()
    const sourcePath = getBundledSourcePath()
    if (!bundledPython || !sourcePath) {
      onComplete({ success: false, error: 'Bundled Python or source not found' })
      return
    }
    cmd = bundledPython
    // -s: suppress user site-packages to ensure pip installs into bundled env only
    args = ['-s', '-m', 'pip', 'install', '--force-reinstall', '--no-deps', sourcePath]
  } else {
    const built = buildInstallCommand(method, action, installerPath)
    cmd = built.cmd
    args = built.args
  }

  const fullCmd = `${cmd} ${args.join(' ')}`
  console.log(`[vLLM Manager] Running: ${fullCmd}`)
  onLog(`$ ${fullCmd}\n`)

  const installEnv: Record<string, string | undefined> = { ...process.env }
  if (method === 'bundled-update') {
    // Isolate bundled Python from user's system packages
    installEnv.PYTHONNOUSERSITE = '1'
    installEnv.PYTHONPATH = ''
  }
  const proc = spawn(cmd, args, {
    env: installEnv,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  activeInstall = proc

  proc.stdout?.on('data', (data: Buffer) => {
    onLog(data.toString())
  })
  proc.stderr?.on('data', (data: Buffer) => {
    onLog(data.toString())
  })

  proc.on('exit', (code) => {
    activeInstall = null
    if (code === 0) {
      console.log('[vLLM Manager] Install/update completed successfully')
      onComplete({ success: true })
    } else {
      console.error(`[vLLM Manager] Install/update failed with code ${code}`)
      onComplete({ success: false, error: `Process exited with code ${code}` })
    }
  })

  proc.on('error', (err) => {
    activeInstall = null
    console.error('[vLLM Manager] Install/update error:', err)
    onComplete({ success: false, error: err.message })
  })
}

/**
 * Check if the bundled engine source is newer than the installed version.
 * Used for auto-update on startup.
 */
export function checkEngineVersion(): { current: string; bundled: string; needsUpdate: boolean } {
  const bundledPython = getBundledPythonPath()
  if (!bundledPython) return { current: '', bundled: '', needsUpdate: false }

  let current = ''
  try {
    current = execSync(`"${bundledPython}" -s -c "import vmlx_engine; print(getattr(vmlx_engine, '__version__', 'unknown'))"`, {
      encoding: 'utf-8',
      timeout: 10000,
      env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
    }).trim()
  } catch (_) {
    // Can't import vmlx_engine at all — needs install, not just update
    current = 'unknown'
  }

  const sourcePath = getBundledSourcePath()
  if (!sourcePath) return { current, bundled: '', needsUpdate: false }

  let bundled = ''
  try {
    const pyproject = readFileSync(join(sourcePath, 'pyproject.toml'), 'utf-8')
    const match = pyproject.match(/version\s*=\s*"(.+?)"/)
    bundled = match?.[1] || ''
  } catch (_) {
    return { current, bundled: '', needsUpdate: false }
  }

  // Update if versions differ OR if current version is unknown (old install without __version__)
  const needsUpdate = !!(bundled && (current !== bundled))
  console.log(`[vLLM Manager] Engine version check: installed=${current}, source=${bundled}, needsUpdate=${needsUpdate}`)
  return { current, bundled, needsUpdate }
}

/**
 * Cancel an active install/update.
 */
export function cancelInstall(): boolean {
  if (activeInstall) {
    activeInstall.kill('SIGTERM')
    activeInstall = null
    return true
  }
  return false
}

