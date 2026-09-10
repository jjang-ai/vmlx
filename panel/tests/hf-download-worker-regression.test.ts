import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { spawnSync } from 'child_process'
import { describe, expect, it } from 'vitest'

function extractDownloadWorkerScript(): string {
  const source = readFileSync(join(process.cwd(), 'src/main/ipc/models.ts'), 'utf-8')
  const marker = 'const script = ['
  const markerIndex = source.indexOf(marker)
  expect(markerIndex).toBeGreaterThanOrEqual(0)

  const arrayStart = source.indexOf('[', markerIndex)
  const joinMarker = '].join("\\n")'
  const arrayEnd = source.indexOf(joinMarker, arrayStart)
  expect(arrayStart).toBeGreaterThanOrEqual(0)
  expect(arrayEnd).toBeGreaterThan(arrayStart)

  const arrayLiteral = source.slice(arrayStart, arrayEnd + 1)
  const lines = Function(`return ${arrayLiteral}`)()
  expect(Array.isArray(lines)).toBe(true)
  return lines.join('\n')
}

function writeFakeHubPackage(root: string): void {
  const packageDir = join(root, 'huggingface_hub')
  mkdirSync(packageDir, { recursive: true })
  writeFileSync(
    join(packageDir, '__init__.py'),
    [
      'import os',
      'from .utils import GatedRepoError, RepositoryNotFoundError',
      '',
      'class Entry:',
      '    def __init__(self, rfilename, size):',
      '        self.rfilename = rfilename',
      '        self.size = size',
      '',
      'def _auth_error():',
      "    return GatedRepoError('401 invalid token')",
      '',
      'class HfApi:',
      '    def __init__(self, endpoint=None):',
      '        self.endpoint = endpoint',
      '    def list_repo_tree(self, repo_id, token=None, recursive=True):',
      '        if self.endpoint:',
      "            raise RuntimeError('mirror unavailable')",
      '        if token:',
      '            raise _auth_error()',
      "        return [Entry('config.json', 4)]",
      '',
      'def hf_hub_download(repo_id, filename, local_dir, token=None, endpoint=None, local_dir_use_symlinks=False, tqdm_class=None):',
      '    if endpoint:',
      "        raise RuntimeError('mirror unavailable')",
      '    if token:',
      '        raise _auth_error()',
      '    os.makedirs(local_dir, exist_ok=True)',
      '    path = os.path.join(local_dir, filename)',
      '    progress = tqdm_class(total=4, initial=0)',
      '    progress.refresh()',
      "    with open(path, 'wb') as f:",
      "        f.write(b'test')",
      '    progress.update(4)',
      '    progress.close()',
      '    return path',
      '',
    ].join('\n'),
  )
  writeFileSync(
    join(packageDir, 'utils.py'),
    [
      'class _Response:',
      '    status_code = 401',
      '',
      'class GatedRepoError(Exception):',
      '    def __init__(self, message):',
      '        super().__init__(message)',
      '        self.response = _Response()',
      '',
      'class RepositoryNotFoundError(Exception):',
      '    def __init__(self, message):',
      '        super().__init__(message)',
      '        self.response = _Response()',
      '',
    ].join('\n'),
  )
}

describe('HuggingFace download worker fallback', () => {
  it('reports missing dependencies as a structured error before any transfer', () => {
    const result = spawnSync(process.env.PYTHON || 'python3',
      ['-I', '-S', '-c', extractDownloadWorkerScript(), 'test/model', '/unused', '', ''],
      { encoding: 'utf-8', timeout: 10000 })
    expect(result.status).toBe(1)
    expect(result.stderr).toBe('')
    expect(JSON.parse(result.stdout)).toEqual({
      status: 'error',
      error: expect.stringContaining('Download runtime dependency unavailable:'),
    })
  })

  it('uses the engine project venv in development, retaining bundled precedence', async () => {
    const source = readFileSync(join(process.cwd(), 'src/main/ipc/models.ts'), 'utf-8')
    const body = source.split('async function getPythonPath(): Promise<string> {')[1].split('\n  async function processQueue')[0]
    const resolve = (bundled: string | null, dev: string | null) => Function(
      'getBundledPythonPath', 'getDevelopmentProjectVenv', 'access',
      'return (async function() {' + body + ')()'
    )(() => bundled, () => dev ? { pythonPath: dev } : null, async () => {})
    expect(await resolve('/bundle/python', '/project/python')).toBe('/bundle/python')
    expect(await resolve(null, '/project/python')).toBe('/project/python')
    expect(await resolve(null, null)).toBe('python3')
  })

  it('preserves failed download state and diagnostic in the history renderer', () => {
    const source = readFileSync(join(process.cwd(), 'src/renderer/src/components/DownloadsView.tsx'), 'utf-8')
    const errorHandler = source.split('const unsubError =')[1].split('const unsubStart =')[0]
    expect(errorHandler).toContain("status: 'error' as const")
    expect(errorHandler).not.toContain("status: 'cancelled'")
    expect(source).toContain('error: c.error')
    expect(source).toContain('data-vmlx-download-status={item.status}')
    expect(source).toContain('queue.length === 0 && paused.length === 0 && completed.length === 0')
    expect(source).toContain("{item.status === 'error' && item.error &&")
  })

  it('supports refresh while recovering from a stale backup endpoint plus stale token', () => {
    const workerScript = extractDownloadWorkerScript()
    const root = mkdtempSync(join(tmpdir(), 'vmlx-hf-worker-'))
    const fakeHubRoot = join(root, 'fake-hub')
    const downloadDir = join(root, 'download')

    try {
      writeFakeHubPackage(fakeHubRoot)
      const result = spawnSync(
        process.env.PYTHON || 'python3',
        [
          '-B',
          '-s',
          '-u',
          '-c',
          workerScript,
          'test-org/test-model',
          downloadDir,
          'http://127.0.0.1:9',
          // repo subfolder: the worker takes it as its fourth argument (empty = whole repo)
          '',
        ],
        {
          encoding: 'utf-8',
          env: {
            PATH: process.env.PATH || '',
            PYTHONPATH: fakeHubRoot,
            HF_TOKEN: 'stale-token',
          },
          timeout: 10000,
        },
      )

      expect(result.status).toBe(0)
      expect(result.stderr).toBe('')
      expect(result.stdout).toContain('"type": "fallback"')
      expect(result.stdout).toContain('"type": "file_progress"')
      expect(result.stdout).toContain('"status": "complete"')
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
