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
      "    with open(path, 'wb') as f:",
      "        f.write(b'test')",
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
  it('recovers public downloads from a stale backup endpoint plus stale token', () => {
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
      expect(result.stdout).toContain('"status": "complete"')
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
