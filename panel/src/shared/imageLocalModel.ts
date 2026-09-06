/**
 * Local image-model directories chosen by the user (custom path or Browse).
 *
 * A typed or browsed path is a directory on disk, not a registry key. The
 * image server start path used to look the string up in the downloaded-model
 * registry and, on a miss, tell the user to "download" a model that already
 * sits on their external drive. This module answers, from the filesystem only:
 *   - is this a local model directory (mflux layout: transformer/ + text_encoder/
 *     or a diffusers model_index.json)?
 *   - which precision does the bundle actually carry? Read from the bundle
 *     itself (mflux writes `quantization_level` into the safetensors header,
 *     diffusers-style bundles carry `quantization_config.bits`), then from the
 *     folder name, and only then from what the picker requested. mflux ignores
 *     a requested level that conflicts with a pre-quantized bundle, so the
 *     stored level is the only truthful thing to show and to pass along.
 *   - is it a multi-variant folder (q3/ q4/ q8/ ... each holding a model), and
 *     which variant matches the requested precision?
 *   - if the path is gone, is that because its external volume is not mounted
 *     (keep the registration, say so) or because the folder was removed?
 * Pure with respect to the injected `fs` so it can be unit-tested on temp dirs.
 */
import { closeSync, existsSync, openSync, readSync, readdirSync, readFileSync, statSync } from 'fs'
import { homedir } from 'os'
import { basename, isAbsolute, join, resolve } from 'path'

export interface LocalImageModelFs {
  existsSync: (p: string) => boolean
  isDirectory: (p: string) => boolean
  readdirSync: (p: string) => string[]
  /** Parsed `__metadata__` of a safetensors file (header only), or null. */
  readSafetensorsMetadata: (p: string) => Record<string, string> | null
  /** Parsed JSON file, or null when missing/invalid. */
  readJson: (p: string) => unknown
}

/** Header-only read: 8-byte little-endian length, then a JSON header whose `__metadata__` holds mflux's stamps. */
export function readSafetensorsHeaderMetadata(p: string): Record<string, string> | null {
  let fd: number | null = null
  try {
    fd = openSync(p, 'r')
    const lenBuf = Buffer.alloc(8)
    if (readSync(fd, lenBuf, 0, 8, 0) !== 8) return null
    const len = Number(lenBuf.readBigUInt64LE(0))
    if (!Number.isFinite(len) || len <= 0 || len > 64 * 1024 * 1024) return null
    const headerBuf = Buffer.alloc(len)
    const got = readSync(fd, headerBuf, 0, len, 8)
    const header = JSON.parse(headerBuf.subarray(0, got).toString('utf8'))
    const meta = header && typeof header === 'object' ? header.__metadata__ : null
    return meta && typeof meta === 'object' ? (meta as Record<string, string>) : null
  } catch {
    return null
  } finally {
    if (fd !== null) {
      try { closeSync(fd) } catch { /* ignore */ }
    }
  }
}

const defaultFs: LocalImageModelFs = {
  existsSync,
  isDirectory: (p) => {
    try {
      return statSync(p).isDirectory()
    } catch {
      return false
    }
  },
  readdirSync: (p) => readdirSync(p),
  readSafetensorsMetadata: readSafetensorsHeaderMetadata,
  readJson: (p) => {
    try {
      return JSON.parse(readFileSync(p, 'utf8'))
    } catch {
      return null
    }
  },
}

export type QuantizeSource = 'metadata' | 'config' | 'name' | 'requested'

export interface LocalImageModelVariant {
  path: string
  quantize: number | null
  quantizeSource: QuantizeSource | null
  name: string
}

export type LocalImageModelResolution =
  | { kind: 'model'; path: string; quantize: number | null; quantizeSource: QuantizeSource | null; mfluxVersion: string | null; siblings?: LocalImageModelVariant[] }
  | { kind: 'variants'; path: string; variants: LocalImageModelVariant[]; requestedQuantize: number }
  | { kind: 'not-a-model-directory'; path: string }
  | { kind: 'path-missing'; path: string; volume: string | null }

function bitsOrNull(value: unknown): number | null {
  const bits = typeof value === 'string' ? Number(value.trim()) : typeof value === 'number' ? value : NaN
  return Number.isInteger(bits) && bits >= 2 && bits <= 16 ? bits : null
}

/** "…-8bit", "…_4bit", "…-q8", "q6", "…-8-bit" → bits; null when the name says nothing. */
export function inferImageQuantizeFromName(name: string): number | null {
  const base = basename(name)
  const m =
    base.match(/(?:^|[-_.])(\d{1,2})[-_ ]?bit(?:$|[-_.])/i) ||
    base.match(/(?:^|[-_.])q(\d{1,2})(?:$|[-_.])/i) ||
    base.match(/^q(\d{1,2})$/i)
  return m ? bitsOrNull(m[1]) : null
}

export function expandUserPath(input: string): string {
  const trimmed = input.trim()
  if (trimmed === '~') return homedir()
  if (trimmed.startsWith('~/')) return join(homedir(), trimmed.slice(2))
  return trimmed
}

/** True only for something that names a filesystem location, never for a Hugging Face id like `org/repo`. */
export function looksLikeLocalPath(input: string): boolean {
  const t = input.trim()
  return t.startsWith('/') || t.startsWith('~') || t.startsWith('./') || t.startsWith('../')
}

/**
 * For a path under /Volumes/<name>/…, the volume name when that volume is not
 * mounted right now; null otherwise. Lets callers tell "drive unplugged" from
 * "folder deleted" instead of collapsing both into "not downloaded".
 */
export function unmountedVolume(p: string, fs: LocalImageModelFs = defaultFs): string | null {
  const m = p.match(/^\/Volumes\/([^/]+)(?:\/|$)/)
  if (!m) return null
  return fs.isDirectory(`/Volumes/${m[1]}`) ? null : m[1]
}

function isModelDirectory(dir: string, fs: LocalImageModelFs): boolean {
  if (fs.existsSync(join(dir, 'model_index.json'))) return true
  return fs.isDirectory(join(dir, 'transformer')) && (fs.isDirectory(join(dir, 'text_encoder')) || fs.isDirectory(join(dir, 'vae')))
}

export interface BundleQuantization {
  bits: number | null
  source: QuantizeSource | null
  mfluxVersion: string | null
}

/**
 * The precision the bundle itself declares. mflux stamps every shard header
 * with `quantization_level` (and `mflux_version`); a diffusers-style bundle may
 * carry `quantization_config.bits` in config.json. A full-precision bundle
 * declares nothing, and the folder name is then the last honest hint.
 */
export function readBundleQuantization(dir: string, fs: LocalImageModelFs = defaultFs): BundleQuantization {
  const transformer = join(dir, 'transformer')
  if (fs.isDirectory(transformer)) {
    const shards = fs.readdirSync(transformer).filter((n) => n.endsWith('.safetensors')).sort()
    for (const shard of shards) {
      const meta = fs.readSafetensorsMetadata(join(transformer, shard))
      if (!meta) continue
      const mfluxVersion = typeof meta.mflux_version === 'string' ? meta.mflux_version : null
      const bits = bitsOrNull(meta.quantization_level)
      if (bits !== null) return { bits, source: 'metadata', mfluxVersion }
      if (mfluxVersion) return { bits: null, source: null, mfluxVersion }
      break
    }
  }
  const cfg = fs.readJson(join(dir, 'config.json')) as { quantization_config?: { bits?: unknown } } | null
  const cfgBits = cfg && typeof cfg === 'object' ? bitsOrNull(cfg.quantization_config?.bits) : null
  if (cfgBits !== null) return { bits: cfgBits, source: 'config', mfluxVersion: null }
  const nameBits = inferImageQuantizeFromName(dir)
  if (nameBits !== null) return { bits: nameBits, source: 'name', mfluxVersion: null }
  return { bits: null, source: null, mfluxVersion: null }
}

function describeModel(dir: string, requestedQuantize: number, fs: LocalImageModelFs): Extract<LocalImageModelResolution, { kind: 'model' }> {
  const declared = readBundleQuantization(dir, fs)
  if (declared.bits !== null) {
    return { kind: 'model', path: dir, quantize: declared.bits, quantizeSource: declared.source, mfluxVersion: declared.mfluxVersion }
  }
  // Nothing declared: a full-precision bundle. The picker's request then
  // means "quantize on the fly", which mflux honours.
  const quantize = requestedQuantize > 0 ? requestedQuantize : null
  return { kind: 'model', path: dir, quantize, quantizeSource: quantize === null ? null : 'requested', mfluxVersion: declared.mfluxVersion }
}

/**
 * Resolve a user-chosen path. Returns `null` when the input is not a local path
 * at all (a Hugging Face id or a bare model name), so callers fall through to
 * the registry. Otherwise reports a model directory, a folder of variants, a
 * directory that holds no recognisable model, or a missing path (with the
 * unmounted volume named when that is the reason).
 */
export function resolveLocalImageModelDirectory(
  input: string,
  requestedQuantize: number,
  fs: LocalImageModelFs = defaultFs,
): LocalImageModelResolution | null {
  if (!looksLikeLocalPath(input)) return null
  const expanded = expandUserPath(input)
  const dir = isAbsolute(expanded) ? expanded : resolve(expanded)
  if (!fs.existsSync(dir) || !fs.isDirectory(dir)) {
    return { kind: 'path-missing', path: dir, volume: unmountedVolume(dir, fs) }
  }

  if (isModelDirectory(dir, fs)) return withSiblings(describeModel(dir, requestedQuantize, fs), fs)

  const variants: LocalImageModelVariant[] = []
  for (const name of fs.readdirSync(dir).sort()) {
    const sub = join(dir, name)
    if (!fs.isDirectory(sub) || !isModelDirectory(sub, fs)) continue
    const declared = readBundleQuantization(sub, fs)
    variants.push({ path: sub, quantize: declared.bits, quantizeSource: declared.source, name })
  }
  if (variants.length === 0) return { kind: 'not-a-model-directory', path: dir }
  const match = requestedQuantize > 0 ? variants.find((v) => v.quantize === requestedQuantize) : undefined
  if (match) return { ...describeModel(match.path, requestedQuantize, fs), siblings: variants.filter((v) => v.path !== match.path) }
  if (variants.length === 1) return describeModel(variants[0].path, requestedQuantize, fs)
  return { kind: 'variants', path: dir, variants, requestedQuantize }
}

/** Other model variants living next to `res.path` (q3/ q4/ q8/ … under one parent), if any. */
function withSiblings(res: Extract<LocalImageModelResolution, { kind: 'model' }>, fs: LocalImageModelFs): Extract<LocalImageModelResolution, { kind: 'model' }> {
  const parent = resolve(res.path, '..')
  if (!fs.isDirectory(parent)) return res
  const siblings: LocalImageModelVariant[] = []
  for (const name of fs.readdirSync(parent).sort()) {
    const sub = join(parent, name)
    if (sub === res.path || !fs.isDirectory(sub) || !isModelDirectory(sub, fs)) continue
    const declared = readBundleQuantization(sub, fs)
    siblings.push({ path: sub, quantize: declared.bits, quantizeSource: declared.source, name })
  }
  return siblings.length ? { ...res, siblings } : res
}

/**
 * Edit classes at 4-bit and below: measured 2026-09-06 on Qwen-Image-Edit-mflux
 * through mflux directly (no vMLX): q4 returned noise and ignored the
 * instruction on a synthetic and a photographic source; q8 followed it.
 * Returns the best higher-precision sibling to offer, or null.
 */
export const EDIT_LOW_PRECISION_MAX_BITS = 4
export function editPrecisionAlternative(res: Extract<LocalImageModelResolution, { kind: 'model' }>): LocalImageModelVariant | null {
  if (res.quantize === null || res.quantize > EDIT_LOW_PRECISION_MAX_BITS) return null
  const better = (res.siblings || []).filter((v) => v.quantize === null || v.quantize >= 8).sort((a, b) => (a.quantize ?? 99) - (b.quantize ?? 99))
  return better[0] || null
}

export function describeVariants(res: Extract<LocalImageModelResolution, { kind: 'variants' }>): string {
  return `"${basename(res.path)}" holds several model variants: ${variantList(res)}. Pick one variant folder as the model path, or choose a matching precision.`
}

export function variantList(res: Extract<LocalImageModelResolution, { kind: 'variants' }>): string {
  return res.variants.map((v) => (v.quantize ? `${v.name} (${v.quantize}-bit)` : v.name)).join(', ')
}

/**
 * Stable error identity for the renderer: a code the UI translates, the
 * parameters the translation needs, and an English fallback message.
 */
export interface LocalImageModelError {
  code: 'variants' | 'notAModelDirectory' | 'volumeUnavailable' | 'pathMissing'
  params: Record<string, string>
  message: string
}

export function localImageModelError(res: Exclude<LocalImageModelResolution, { kind: 'model' }>): LocalImageModelError {
  switch (res.kind) {
    case 'variants':
      return { code: 'variants', params: { name: basename(res.path), variants: variantList(res) }, message: describeVariants(res) }
    case 'not-a-model-directory':
      return {
        code: 'notAModelDirectory',
        params: { path: res.path },
        message: `"${res.path}" exists but holds no image model (expected transformer/ with text_encoder/ or vae/, or a model_index.json).`,
      }
    case 'path-missing':
      if (res.volume) {
        return {
          code: 'volumeUnavailable',
          params: { path: res.path, volume: res.volume },
          message: `"${res.path}" is on the volume "${res.volume}", which is not mounted. Connect the drive and try again.`,
        }
      }
      return { code: 'pathMissing', params: { path: res.path }, message: `"${res.path}" does not exist.` }
  }
}
