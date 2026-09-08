// One-time dated warning for Qwen3.8-Flash-Next JANG models (model_type
// qwen4_exp — base JANGQ-AI / OsaurusAI and dealignai CRACK alike):
// the models were re-uploaded with an MTP fix, and users who downloaded
// before that date must REDOWNLOAD THE ENTIRE MODEL to get it.
//
// Uses read-only local bundle checks, no network requests or runtime mutation.
// When an affected model is loaded, the
// renderer shows the warning once; any interaction persists the dismissal
// for this fix date, so it never reappears. Bumping MTP_FIX_DATE for a
// future fix shows the warning once more.

import { BrowserWindow } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { db } from './database'
import { detectModelConfigFromDir } from './model-config-registry'

// The date the fixed Flash-Next bundles were published. Bump on a future fix.
export const MTP_FIX_DATE = '2026-09-04'

export interface MtpComponentUpdate {
  repoId: string
  modelPath: string
  /** The fix date this warning is about (doubles as the dismissal key). */
  remoteFingerprint: string
}

function dismissKey(repoId: string): string {
  return `mtp_component_dismissed:${repoId}`
}

// At most one warning per model per app run, on top of the persisted
// dismissal — repeated loads can never stack popups.
const promptedThisRun = new Set<string>()

function isFlashNextQwen4Exp(modelPath: string): boolean {
  try {
    // MODEL_TYPE_TO_FAMILY maps qwen4_exp and qwen4_exp_text -> qwen4-exp,
    // which covers every Flash-Next tier regardless of publishing org.
    return (
      String(detectModelConfigFromDir(modelPath)?.family ?? '').toLowerCase() ===
      'qwen4-exp'
    )
  } catch {
    return false
  }
}

// <org>/<name> from the bundle path — matches how the model scanner ids
// user-dir bundles (JANGQ-AI/..., dealignai/..., OsaurusAI/...).
function repoIdFromModelPath(modelPath: string): string | null {
  const parts = modelPath.split('/').filter(Boolean)
  if (parts.length < 2) return null
  return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`
}

// The fixed bundles' contract, checked locally with three read-only looks:
// mtp.* keys present in the weight index, the calibrated proposal-head
// sidecar under mtp_draft/, and a stamp whose draft_artifact points at it.
// A bundle missing ANY of these is an older/broken MTP layout — the user
// must redownload the entire model to utilize MTP. A conformant bundle
// never warns. This inspects files only; it changes nothing about how MTP
// is detected, loaded, or used.
function bundleHasFixedMtpComponent(modelPath: string): boolean {
  try {
    const index = JSON.parse(
      readFileSync(join(modelPath, 'model.safetensors.index.json'), 'utf8'),
    ) as { weight_map?: Record<string, string> }
    const hasMtpKeys = Object.keys(index.weight_map ?? {}).some((k) =>
      k.startsWith('mtp.'),
    )
    if (!hasMtpKeys) return false

    const sidecarRel = 'mtp_draft/vmlx_mtp_proposal_head.safetensors'
    if (!existsSync(join(modelPath, sidecarRel))) return false

    const stamp = JSON.parse(
      readFileSync(join(modelPath, 'vmlx_mtp_proposal_head.json'), 'utf8'),
    ) as { draft_artifact?: { file?: string } }
    return stamp.draft_artifact?.file === sidecarRel
  } catch {
    return false
  }
}

function bundleExplicitlyOmitsMtp(modelPath: string): boolean {
  try {
    const config = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf8'))
    const index = JSON.parse(readFileSync(join(modelPath, 'model.safetensors.index.json'), 'utf8'))
    // Some intentionally stripped artifacts explicitly declare this contract.
    // Never infer it from JANG tier/name, or hide real indexed MTP components.
    return config.jang_config?.mtp?.mtp_mode === 'none'
      && index.weight_map != null && typeof index.weight_map === 'object'
      && !Object.keys(index.weight_map).some(k => k.startsWith('mtp.'))
  } catch { return false }
}

/**
 * Fire-and-forget: when the user loads a Flash-Next JANG model whose bundle
 * is an older/broken MTP layout (missing the fixed component), show the
 * one-time dated redownload warning unless already dismissed for this fix
 * date. A bundle that carries the fix never warns. Never throws; never
 * blocks or delays the load.
 */
export function checkMtpComponentUpdateOnLoad(
  getWindow: () => BrowserWindow | null,
  modelPath: string,
): void {
  if (process.env.VMLX_SKIP_MTP_COMPONENT_CHECK === '1') return
  void (async () => {
    try {
      if (!modelPath || !isFlashNextQwen4Exp(modelPath)) return
      if (bundleExplicitlyOmitsMtp(modelPath)) return
      if (bundleHasFixedMtpComponent(modelPath)) return
      const repoId = repoIdFromModelPath(modelPath)
      if (!repoId) return
      const runKey = `${repoId}@${MTP_FIX_DATE}`
      if (promptedThisRun.has(runKey)) return
      let dismissed: string | undefined
      try {
        dismissed = db.getSetting(dismissKey(repoId))
      } catch {
        dismissed = undefined
      }
      if (dismissed === MTP_FIX_DATE) return

      const win = getWindow()
      if (win && !win.isDestroyed()) {
        promptedThisRun.add(runKey)
        const update: MtpComponentUpdate = {
          repoId,
          modelPath,
          remoteFingerprint: MTP_FIX_DATE,
        }
        win.webContents.send('models:mtpComponentUpdateAvailable', update)
      }
    } catch (err) {
      console.log(
        `[MTP-UPDATE] warning skipped: ${(err as Error)?.message ?? err}`,
      )
    }
  })()
}

export function dismissMtpComponentUpdate(
  repoId: string,
  remoteFingerprint: string,
): void {
  try {
    db.setSetting(dismissKey(repoId), remoteFingerprint)
  } catch (err) {
    console.log(`[MTP-UPDATE] dismiss failed: ${(err as Error)?.message ?? err}`)
  }
}

// Exposed for tests.
export const __test = { repoIdFromModelPath, bundleHasFixedMtpComponent, bundleExplicitlyOmitsMtp }
