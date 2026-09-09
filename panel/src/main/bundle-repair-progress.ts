import { basename } from 'path'

export interface BundleRepairMessage {
  label: string
  labelKey: string
  labelParams?: Record<string, string | number>
}

const stages: Record<string, [string, string]> = {
  MISALIGNED_DETECTED: ['bundleRepairDetected', 'Misaligned tensors detected: {shard}'],
  COPYING: ['bundleRepairCopying', 'Repairing {shard}: copying{detail}'],
  VALIDATED: ['bundleRepairValidated', 'Repairing {shard}: tensor payloads validated'],
  TRANSACTION_COMMITTED: ['bundleRepairCommitted', 'Repairing {shard}: validated replacement committed'],
  REPAIRED_ON_DISK: ['bundleRepairFinished', 'Repaired {shard}; continuing bundle check'],
  INTERRUPTED_COPY_DISCARDED: ['bundleRepairRecovered', 'Removed interrupted copy for {shard}; original preserved'],
  REPAIR_FAILED: ['bundleRepairFailed', 'Bundle repair failed: {shard}. See the error log.'],
}

// Per-start state, never shared between sessions. The checker remains the
// authority: unknown/malformed messages cannot manufacture progress or success.
export function createBundleRepairProgressReporter(
  publish: (message: BundleRepairMessage, isNotice: boolean) => void,
): (line: string) => void {
  let announced = false
  return line => {
    if (!line.startsWith('[BUNDLE-ALIGNMENT] ')) return
    let event: Record<string, unknown>
    try {
      const parsed = JSON.parse(line.slice('[BUNDLE-ALIGNMENT] '.length))
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return
      event = parsed
    } catch { return }
    if (typeof event.stage !== 'string' || typeof event.shard !== 'string') return
    const stage = Object.prototype.hasOwnProperty.call(stages, event.stage) ? stages[event.stage] : undefined
    if (!stage) return
    const shard = basename(event.shard).replace(/[\r\n\t]/g, ' ')
    if (!shard) return
    if (!announced) {
      announced = true
      publish({
        label: 'This bundle needs an alignment repair before loading. This is a one-time step for the current files; unchanged files will not need it again. Please wait while each shard is copied, validated and atomically replaced.',
        labelKey: 'main.loadProgress.bundleRepairNotice',
      }, true)
    }
    const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
    const copied = event.copied_bytes, total = event.payload_bytes
    const detail = event.stage === 'COPYING' && finite(copied) && finite(total) && total > 0 && copied <= total
      ? ' (' + (copied / 1048576).toFixed(1) + ' / ' + (total / 1048576).toFixed(1) + ' MiB)' : ''
    const labelParams = { shard, detail }
    publish({
      label: stage[1].replace('{shard}', shard).replace('{detail}', detail),
      labelKey: 'main.loadProgress.' + stage[0],
      labelParams,
    }, false)
  }
}
