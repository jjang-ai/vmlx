import { execFileSync } from 'child_process'
import { existsSync, readdirSync, readFileSync, statSync } from 'fs'
import { join } from 'path'

/** Estimate local model file bytes recursively. Returns 0 if unknown. */
export function estimateModelFileBytes(modelPath: string): number {
  try {
    const entries = readdirSync(modelPath, { withFileTypes: true })
    let totalBytes = 0
    for (const entry of entries) {
      const fullPath = join(modelPath, entry.name)
      if (entry.isDirectory()) {
        totalBytes += estimateModelFileBytes(fullPath)
      } else if (entry.isFile()) {
        totalBytes += statSync(fullPath).size
      }
    }
    return totalBytes
  } catch (_) {
    return 0
  }
}

/** Conservative full-resident estimate for non-mmap bundles. */
export function estimateModelMemory(modelPath: string): number {
  const fileBytes = estimateModelFileBytes(modelPath)
  if (fileBytes <= 0) return 0
  return Math.round(fileBytes * 1.3)
}

export const JANG_MMAP_LAUNCH_RESIDENT_RATIO = 0.7

export function isLazyMmapJangBundle(modelPath: string): boolean {
  try {
    if (existsSync(join(modelPath, 'jang_config.json'))) return true
    const config = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf8'))
    const weightFormat = String(config?.weight_format || config?.quantization?.format || '').toLowerCase()
    if (['jang', 'jangtq', 'mxtq', 'affine', 'jjqf', 'mxq'].includes(weightFormat)) return true
  } catch (_) {
    // Fall through to extension check below.
  }

  try {
    const entries = readdirSync(modelPath, { withFileTypes: true })
    return entries.some(entry =>
      entry.isFile() &&
      (entry.name.endsWith('.jang.safetensors') ||
        entry.name.endsWith('.mxtq.safetensors') ||
        entry.name.endsWith('.mxq.safetensors') ||
        entry.name.endsWith('.jjqf.safetensors'))
    )
  } catch (_) {
    return false
  }
}

export function estimateModelLaunchResidentBytes(modelPath: string, modelFileBytes: number, totalBytes: number): number {
  if (modelFileBytes <= 0) return 0
  if (isLazyMmapJangBundle(modelPath)) {
    const mmapResidentBytes = Math.round(modelFileBytes * JANG_MMAP_LAUNCH_RESIDENT_RATIO)
    return totalBytes > 0 ? Math.min(mmapResidentBytes, totalBytes) : mmapResidentBytes
  }
  return Math.round(modelFileBytes * 1.3)
}

export function formatGb(bytes: number): string {
  return (bytes / 1e9).toFixed(1)
}

export const MODEL_LAUNCH_SAFETY_RESERVE_BYTES = 16 * 1024 ** 3
export const UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV = 'VMLX_ALLOW_UNSAFE_MODEL_LAUNCH'
export const UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV = 'VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH'

export interface LaunchAdmissionOptions {
  lazyMmap?: boolean
  reclaimableBytes?: number
  totalBytes?: number
}

export function unsafeModelLaunchOverrideEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV] === '1' || env[UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV] === '1'
}

export function unsafeModelLaunchOverrideHint(): string {
  return `${UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV}=1 (or legacy ${UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV}=1)`
}

export function unsafeModelLaunchReason(
  modelSizeBytes: number,
  availableBytes: number,
  env: NodeJS.ProcessEnv = process.env,
  options: LaunchAdmissionOptions = {},
): string | null {
  if (modelSizeBytes <= 0 || availableBytes <= 0) return null
  if (unsafeModelLaunchOverrideEnabled(env)) return null
  const effectiveAvailableBytes = effectiveLaunchAvailableBytes(availableBytes, options)
  if (modelSizeBytes <= effectiveAvailableBytes) return null
  return (
    `estimated launch resident ~${formatGb(modelSizeBytes)} GB exceeds ` +
    `currently free RAM ${formatGb(availableBytes)} GB`
  )
}

export function effectiveLaunchAvailableBytes(
  availableBytes: number,
  options: LaunchAdmissionOptions = {},
): number {
  if (!options.lazyMmap || availableBytes <= 0) return availableBytes
  const reclaimableBytes = Math.max(0, options.reclaimableBytes || 0)
  if (reclaimableBytes <= 0) return availableBytes
  const totalBytes = Math.max(0, options.totalBytes || 0)
  const reclaimableCap = totalBytes > 0 ? Math.round(totalBytes * 0.15) : reclaimableBytes
  return availableBytes + Math.min(reclaimableBytes, reclaimableCap)
}

export function estimateMacReclaimableMemoryBytesFromVmStat(output: string): number {
  const pageSizeMatch = output.match(/page size of\s+(\d+)\s+bytes/i)
  const pageSize = pageSizeMatch ? Number(pageSizeMatch[1]) : 4096
  if (!Number.isFinite(pageSize) || pageSize <= 0) return 0

  const pagesFor = (label: string): number => {
    const match = output.match(new RegExp(`${label}:\\s+(\\d+)\\.`, 'i'))
    if (!match) return 0
    const pages = Number(match[1])
    return Number.isFinite(pages) && pages > 0 ? pages : 0
  }

  // Node's os.freemem() already includes free/speculative memory on macOS.
  // Count only conservative extra reclaimable pages so lazy-mmap launches do
  // not get blocked by cache-heavy but low-pressure systems.
  return (pagesFor('Pages inactive') + pagesFor('Pages purgeable')) * pageSize
}

export function estimateMacReclaimableMemoryBytes(): number {
  if (process.platform !== 'darwin') return 0
  try {
    const output = execFileSync('vm_stat', [], {
      encoding: 'utf8',
      timeout: 1000,
    })
    return estimateMacReclaimableMemoryBytesFromVmStat(output)
  } catch (_) {
    return 0
  }
}

export function modelLaunchReserveWarning(modelSizeBytes: number, availableBytes: number): string | null {
  if (modelSizeBytes <= 0 || availableBytes <= 0) return null
  const requiredFreeBytes = modelSizeBytes + MODEL_LAUNCH_SAFETY_RESERVE_BYTES
  if (requiredFreeBytes <= availableBytes) return null
  return (
    `estimated launch resident ~${formatGb(modelSizeBytes)} GB leaves less ` +
    `than 16.0 GB system safety headroom (${formatGb(availableBytes)} GB free)`
  )
}
