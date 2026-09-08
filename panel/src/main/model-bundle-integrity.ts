import { execFile } from 'child_process'

export type BundleIntegrityEnginePath =
  | { type: 'bundled'; pythonPath: string }
  | { type: 'development'; pythonPath: string; sourceRoot: string }
  | { type: 'system'; binaryPath: string; sourceRoot?: string }

export interface BundleIntegrityReport {
  schema: 'vmlx-model-bundle-integrity-v1'
  status: 'ok'
  bundle: string
  shards: number
  tensors: number
  misaligned_tensors: number
  alignment_contract: 'atomic_realign'
  repairs: string[]
  warnings: string[]
  cache_hit: boolean
}

export interface BundleIntegrityInvocation {
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
}

export function buildBundleIntegrityInvocation(
  engine: BundleIntegrityEnginePath,
  modelPath: string,
): BundleIntegrityInvocation {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONDONTWRITEBYTECODE: '1',
    PYTHONNOUSERSITE: '1',
    PYTHONPATH: undefined,
  }
  const checkArgs = ['bundle-check', modelPath, '--json']
  if (engine.type === 'system') {
    if (engine.sourceRoot) env.PYTHONPATH = engine.sourceRoot
    return { command: engine.binaryPath, args: checkArgs, env }
  }
  if (engine.type === 'development') env.PYTHONPATH = engine.sourceRoot
  return {
    command: engine.pythonPath,
    args: ['-B', '-s', '-m', 'vmlx_engine.cli', ...checkArgs],
    env,
  }
}

export function parseBundleIntegrityReport(output: string): BundleIntegrityReport {
  let value: unknown
  try {
    value = JSON.parse(output.trim())
  } catch (error) {
    throw new Error(
      `vmlx-engine returned an invalid bundle-integrity report: ${String(error)}`,
    )
  }
  if (!value || typeof value !== 'object') {
    throw new Error('vmlx-engine returned an empty bundle-integrity report')
  }
  const report = value as Partial<BundleIntegrityReport> & { error?: unknown }
  if (report.status !== 'ok') {
    const detail = typeof report.error === 'string' ? report.error : 'unknown failure'
    throw new Error(`Model bundle integrity check failed: ${detail}`)
  }
  if (
    report.schema !== 'vmlx-model-bundle-integrity-v1' ||
    typeof report.bundle !== 'string' ||
    !Number.isInteger(report.shards) ||
    !Number.isInteger(report.tensors) ||
    !Number.isInteger(report.misaligned_tensors) ||
    report.alignment_contract !== 'atomic_realign' ||
    !Array.isArray(report.repairs) ||
    !Array.isArray(report.warnings) ||
    typeof report.cache_hit !== 'boolean'
  ) {
    throw new Error('vmlx-engine returned an incomplete bundle-integrity report')
  }
  return report as BundleIntegrityReport
}

export function runModelBundleIntegrityPreflight(
  engine: BundleIntegrityEnginePath,
  modelPath: string,
  onProgress?: (line: string) => void,
): Promise<BundleIntegrityReport> {
  const invocation = buildBundleIntegrityInvocation(engine, modelPath)
  return new Promise((resolve, reject) => {
    const child = execFile(
      invocation.command,
      invocation.args,
      {
        encoding: 'utf8',
        env: invocation.env,
        maxBuffer: 4 * 1024 * 1024,
        // A one-time streaming shard repair can exceed three minutes on USB
        // storage. Progress is surfaced below; original shards stay intact.
        timeout: 3_600_000,
      },
      (error, stdout, stderr) => {
        if (error) {
          const candidates = [stdout, stderr]
            .map(value => String(value || '').trim())
            .filter(Boolean)
          for (const candidate of candidates) {
            try {
              parseBundleIntegrityReport(candidate)
            } catch (parseError) {
              if (
                parseError instanceof Error &&
                parseError.message.startsWith('Model bundle integrity check failed:')
              ) {
                reject(parseError)
                return
              }
            }
          }
          reject(
            new Error(
              `Model bundle integrity preflight could not run: ${error.message}. ` +
              'Repair or re-download the bundle before loading it.',
            ),
          )
          return
        }
        try {
          resolve(parseBundleIntegrityReport(String(stdout)))
        } catch (parseError) {
          reject(parseError)
        }
      },
    )
    let pending = ''
    child.stderr?.on('data', (chunk: string | Buffer) => {
      pending += String(chunk)
      const lines = pending.split('\n')
      pending = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('[BUNDLE-ALIGNMENT] ')) onProgress?.(line)
      }
    })
  })
}
