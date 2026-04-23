import { exec as execCallback, spawn, execSync, execFileSync, ChildProcess } from 'child_process'
import { promisify } from 'util'
import { createHash } from 'crypto'
import { existsSync, readFileSync, readdirSync, realpathSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import { app } from 'electron'

const exec = promisify(execCallback)

export type InstallMethod = 'uv' | 'pip' | 'brew' | 'conda' | 'manual' | 'bundled' | 'unknown'

export interface EngineInstallation {
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
 * Read the installed vmlx-engine version from filesystem metadata without
 * spawning Python (which can take >10s on cold boot and time out, falsely
 * reporting the engine as missing). Looks up the first `vmlx*.dist-info` or
 * `vmlx_engine*.dist-info` directory next to the package and parses the
 * `Version:` line from its METADATA file.
 *
 * mlxstudio#84: previously we executed `python3 -s -c "import vmlx_engine;
 * print(vmlx_engine.__version__)"` with a 10 s timeout. MLX + mlx_vlm pull
 * ~200 MB of shared libs on first import and routinely blow past 10 s on
 * cold-disk M2. The timeout throws → we wrongly report installed=false and
 * the renderer shows "Inference engine not found".
 */
function getBundledEngineVersionFromFilesystem(): string | null {
  if (!app.isPackaged) return null
  try {
    const sitePackages = join(
      process.resourcesPath,
      'bundled-python',
      'python',
      'lib',
      'python3.12',
      'site-packages',
    )
    if (!existsSync(sitePackages)) return null

    // Must also have the actual package dir — dist-info alone is a zombie
    const pkgDir = join(sitePackages, 'vmlx_engine')
    if (!existsSync(join(pkgDir, '__init__.py'))) return null

    for (const entry of readdirSync(sitePackages)) {
      if (!entry.endsWith('.dist-info')) continue
      if (!entry.startsWith('vmlx-') && !entry.startsWith('vmlx_engine-')) continue
      const metadata = join(sitePackages, entry, 'METADATA')
      if (!existsSync(metadata)) continue
      const content = readFileSync(metadata, 'utf-8')
      const match = content.match(/^Version:\s*(.+)$/m)
      if (match && /^\d+\.\d+\.\d+/.test(match[1].trim())) {
        return match[1].trim()
      }
    }
  } catch (_) {
    /* fall through */
  }
  return null
}

/**
 * Check if vmlx-engine is installed and where
 */
export async function checkEngineInstallation(): Promise<EngineInstallation> {
  console.log('[Engine Manager] Checking installation...')

  // 0. Check bundled Python first (standalone distribution).
  // mlxstudio#84: use filesystem metadata instead of spawning Python — the
  // subprocess import timed out on cold boot and wrongly reported missing.
  const bundledPython = getBundledPythonPath()
  if (bundledPython) {
    const fsVer = getBundledEngineVersionFromFilesystem()
    if (fsVer) {
      console.log(`[Engine Manager] Found bundled Python with vmlx_engine ${fsVer} (from dist-info)`)
      return { installed: true, path: bundledPython, version: fsVer, method: 'bundled', bundled: true }
    }
    console.log('[Engine Manager] Bundled Python present but vmlx_engine dist-info missing; trying system')
  }

  // 1. Check common paths
  for (const path of SEARCH_PATHS) {
    if (existsSync(path)) {
      console.log(`[Engine Manager] Found at: ${path}`)
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
      console.log(`[Engine Manager] Found in PATH: ${path}`)
      const version = await getVersionFromBinary(path)
      const method = detectInstallMethod(path)
      return { installed: true, path, version, method }
    }
  } catch (_) {
    // Not in PATH
  }

  // 3. Not found
  console.log('[Engine Manager] Not installed')
  return { installed: false }
}

/**
 * Get version from vmlx-engine binary
 */
async function getVersionFromBinary(path: string): Promise<string> {
  // Get version via Python package metadata (works with editable installs)
  try {
    const { readFileSync } = await import('fs')
    const firstLine = readFileSync(path, 'utf-8').split('\n')[0]
    const shebang = firstLine.trim().replace(/^#\!/, '').trim()
    // Validate shebang is a plausible Python path (no shell metacharacters)
    if (shebang && /^[/\w.\-]+$/.test(shebang)) {
      const { execFile: execFileCb } = await import('child_process')
      const { promisify } = await import('util')
      const execFileAsync = promisify(execFileCb)
      const pyResult = await execFileAsync(shebang, ['-c', "import importlib.metadata; print(importlib.metadata.version('vmlx-engine'))"])
      const ver = pyResult.stdout.trim()
      if (/^\d+\.\d+\.\d+/.test(ver)) {
        console.log(`[Engine Manager] Version: ${ver}`)
        return ver
      }
    }
  } catch (_) { /* fallback below */ }

  // Fallback: try --version flag
  try {
    const result = await exec(`"${path}" --version 2>&1`)
    const match = (result.stdout || result.stderr).match(/(\d+\.\d+\.\d+)/)
    if (match) {
      console.log(`[Engine Manager] Version: ${match[1]}`)
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
function getBundledSourcePath(): string | null {
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
export function installEngineStreaming(
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
  console.log(`[Engine Manager] Running: ${fullCmd}`)
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
      console.log('[Engine Manager] Install/update completed successfully')
      onComplete({ success: true })
    } else {
      console.error(`[Engine Manager] Install/update failed with code ${code}`)
      onComplete({ success: false, error: `Process exited with code ${code}` })
    }
  })

  proc.on('error', (err) => {
    activeInstall = null
    console.error('[Engine Manager] Install/update error:', err)
    onComplete({ success: false, error: err.message })
  })
}

/**
 * Hash key Python source files to detect code changes without version bump.
 * Hashes pyproject.toml + all .py files in vmlx_engine/ (non-recursive top level + key subdirs).
 */
function hashSourceFiles(basePath: string): string | null {
  try {
    const hash = createHash('sha256')
    const engineDir = join(basePath, 'vmlx_engine')
    if (!existsSync(engineDir)) return null

    // Hash pyproject.toml
    const pyproject = join(basePath, 'pyproject.toml')
    if (existsSync(pyproject)) hash.update(readFileSync(pyproject))

    // Hash all .py files in vmlx_engine/ (top-level only for speed)
    for (const f of readdirSync(engineDir).sort()) {
      if (f.endsWith('.py')) {
        hash.update(readFileSync(join(engineDir, f)))
      }
    }
    // Also hash key subdirectories
    for (const subdir of ['utils', 'reasoning', 'tool_parsers', 'api', 'engine', 'commands']) {
      const sub = join(engineDir, subdir)
      if (existsSync(sub)) {
        for (const f of readdirSync(sub).sort()) {
          if (f.endsWith('.py')) hash.update(readFileSync(join(sub, f)))
        }
      }
    }
    return hash.digest('hex').slice(0, 16)
  } catch (_) {
    return null
  }
}

/**
 * Check if the bundled engine source is newer than the installed version.
 * Used for auto-update on startup.
 */
export function checkEngineVersion(): { current: string; bundled: string; needsUpdate: boolean } {
  const bundledPython = getBundledPythonPath()
  if (!bundledPython) return { current: '', bundled: '', needsUpdate: false }

  // mlxstudio#84: prefer filesystem dist-info lookup over spawning Python.
  // The subprocess import timed out on cold boot (>10 s for MLX + mlx_vlm),
  // reporting current='unknown', which triggered a pip --force-reinstall
  // every startup. That reinstall runs *into the signed app bundle* where
  // site-packages is read-only — it uninstalled vmlx_engine then failed to
  // write the new copy, leaving the user with a broken install and the
  // "Inference engine not found" error on next session launch.
  //
  // Also: `vmlx_engine.__version__` is hardcoded "1.0.3" inside the package
  // __init__.py (independent of the wheel Version), so even when the import
  // did succeed, current !== bundled was always true → reinstall every time.
  let current = getBundledEngineVersionFromFilesystem() || ''
  if (!current) {
    try {
      current = execFileSync(
        bundledPython,
        ['-s', '-c', "import importlib.metadata as m; print(m.version('vmlx'))"],
        {
          encoding: 'utf-8',
          timeout: 30000,
          env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
        },
      ).trim()
    } catch (_) {
      current = 'unknown'
    }
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

  // mlxstudio#84: only trigger update when we know the installed version AND
  // it differs from source. Previously `current='unknown' !== bundled` was
  // treated as "needs update" and kicked off a futile pip reinstall into the
  // read-only signed app bundle, corrupting the existing install.
  // Hard gate: in the packaged app, pip install into bundled site-packages
  // ALWAYS fails (signed bundle + extended attrs), so `needsUpdate` must be
  // false regardless of version comparison. Keep the live version diff for
  // dev-tree runs where the bundled python is user-writable.
  let needsUpdate = !!(
    bundled
    && current
    && current !== 'unknown'
    && current !== bundled
    && !app.isPackaged
  )

  // mlxstudio#84: the hash-based content check below is only useful in dev
  // mode. In the packaged app, site-packages is inside the signed bundle and
  // a pip --force-reinstall will fail with EPERM — looping the check would
  // just corrupt the install. Skip it for packaged builds where the bundled
  // source and installed copy both come from the same DMG.
  if (!needsUpdate && bundled && sourcePath && !app.isPackaged) {
    // Hash key source files to detect code changes without version bump
    const sourceHash = hashSourceFiles(sourcePath)
    if (sourceHash) {
      try {
        const installed = execFileSync(bundledPython, ['-s', '-c', 'import vmlx_engine; import pathlib; print(pathlib.Path(vmlx_engine.__file__).parent)'], {
          encoding: 'utf-8', timeout: 30000,
          env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
        }).trim()
        if (installed && existsSync(installed)) {
          const installedHash = hashSourceFiles(join(installed, '..'))
          if (installedHash && installedHash !== sourceHash) {
            needsUpdate = true
            console.log(`[Engine Manager] Source content changed (hash mismatch) — triggering update`)
          }
        }
      } catch (_) { /* hash comparison optional — fall back to version check */ }
    }
  }

  console.log(`[Engine Manager] Engine version check: installed=${current}, source=${bundled}, needsUpdate=${needsUpdate}`)
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
