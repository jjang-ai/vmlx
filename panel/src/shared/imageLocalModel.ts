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
import { basename, dirname, isAbsolute, join, resolve } from 'path'
import { getImageModel, resolveImageModelFromDirectoryName, type ImageModelDef } from './imageModels'

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
  | { kind: 'unsupported-format'; path: string; configPath: string; format: string }

/** Explicit incompatible declarations are stronger than a filename bit hint.
 * This is a rejection gate, not proof that every other tensor layout loads. */
function unsupportedImageFormat(dir: string, fs: LocalImageModelFs): Extract<LocalImageModelResolution, { kind: 'unsupported-format' }> | null {
  const roles = new Set(['transformer', 'unconditional_transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'vae'])
  const index = fs.readJson(join(dir, 'model_index.json'))
  if (index && typeof index === 'object' && !Array.isArray(index)) {
    for (const [key, value] of Object.entries(index)) {
      if (!key.startsWith('_') && basename(key) === key && key !== '.' && key !== '..' && Array.isArray(value)) roles.add(key)
    }
  }
  for (const configPath of ['config.json', ...[...roles].map(role => join(role, 'config.json'))]) {
    const config = fs.readJson(join(dir, configPath))
    if (!config || typeof config !== 'object' || Array.isArray(config)) continue
    const quant = (config as Record<string, unknown>).quantization_config
    if (!quant || typeof quant !== 'object' || Array.isArray(quant)) continue
    const q = quant as Record<string, unknown>
    const method = typeof q.quant_method === 'string' ? q.quant_method.trim().toLowerCase() : ''
    const bnbType = typeof q.bnb_4bit_quant_type === 'string' ? q.bnb_4bit_quant_type.trim().toLowerCase() : ''
    if (method === 'bitsandbytes' || bnbType === 'nf4' || bnbType === 'fp4') {
      return { kind: 'unsupported-format', path: dir, configPath, format: bnbType ? `bitsandbytes/${bnbType}` : 'bitsandbytes' }
    }
  }
  return null
}

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
    const directoryShards = fs.readdirSync(transformer)
      .filter((n) => n.endsWith('.safetensors') && !n.startsWith('._')).sort()
    // Match mflux WeightLoader: a usable weight index owns the checkpoint.
    // Leftover shards from an older export must not choose its precision.
    // An unusable index falls back to the directory, as the loader does.
    const index = fs.readJson(join(transformer, 'model.safetensors.index.json')) as { weight_map?: unknown } | null
    const map = index && typeof index === 'object' ? index.weight_map : null
    const names = map && typeof map === 'object' && !Array.isArray(map) ? Object.values(map) : []
    const validIndex = names.length > 0 && names.every((n) =>
      typeof n === 'string' && !!n && basename(n) === n)
    const indexed = validIndex ? [...new Set(names as string[])].sort()
      .filter(n => fs.existsSync(join(transformer, n)) && !fs.isDirectory(join(transformer, n))) : []
    const shards = indexed.length ? indexed : directoryShards
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

  if (isModelDirectory(dir, fs)) {
    const unsupported = unsupportedImageFormat(dir, fs)
    if (unsupported) return unsupported
    return withSiblings(describeModel(dir, requestedQuantize, fs), fs)
  }

  const variants: LocalImageModelVariant[] = []
  for (const name of fs.readdirSync(dir).sort()) {
    const sub = join(dir, name)
    if (!fs.isDirectory(sub) || !isModelDirectory(sub, fs)) continue
    const declared = readBundleQuantization(sub, fs)
    variants.push({ path: sub, quantize: declared.bits, quantizeSource: declared.source, name })
  }
  if (variants.length === 0) return { kind: 'not-a-model-directory', path: dir }
  const match = requestedQuantize > 0 ? variants.find((v) => v.quantize === requestedQuantize) : undefined
  if (match) return resolveLocalImageModelDirectory(match.path, requestedQuantize, fs)
  if (variants.length === 1) return resolveLocalImageModelDirectory(variants[0].path, requestedQuantize, fs)
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

/**
 * The registry entry (mflux class, canonical name, defaults) for a LOCAL
 * folder. A precision variant folder ("q8", "8bit") says nothing about the
 * model, so its bundle root is consulted too. Before this, the warning's
 * "Use q8" action started `…/Qwen-Image-Edit-mflux/q8` without a class and the
 * engine refused with "Cannot determine mflux class".
 */
export function resolveImageModelForLocalDirectory(path: string): ImageModelDef | undefined {
  // A declared pipeline takes precedence over a directory label. Do not infer
  // an ambiguous Flux variant from the pipeline class alone.
  const index = defaultFs.readJson(join(path, 'model_index.json')) as { _class_name?: string } | null
  const config = defaultFs.readJson(join(path, 'config.json')) as { _class_name?: string; original_model?: string } | null
  const pipelines: Record<string, string> = {
    QwenImagePipeline: 'qwen-image',
    QwenImageEditPipeline: 'qwen-image-edit',
    ZImageTurboPipeline: 'z-image-turbo',
    FluxKontextPipeline: 'kontext',
    FluxFillPipeline: 'fill',
  }
  const pipeline = index?._class_name || config?._class_name
  if (pipeline && pipelines[pipeline]) {
    return getImageModel(pipelines[pipeline])
  }
  if (pipeline === 'FluxPipeline') {
    // Older mflux exports carry this in config.json. FluxPipeline alone
    // cannot distinguish Schnell from Dev, including after a folder rename.
    if (typeof config?.original_model === 'string') {
      const declared = resolveImageModelFromDirectoryName(basename(config.original_model))
      if (declared?.id === 'schnell' || declared?.id === 'dev') return declared
    }
  } else if (pipeline) {
    // An unknown declared architecture is stronger evidence than a filename.
    // Leave it unresolved for an explicit supported adapter choice.
    return undefined
  }
  const base = basename(path)
  const direct = resolveImageModelFromDirectoryName(base)
  if (direct) return direct
  if (isVariantFolderName(base)) return resolveImageModelFromDirectoryName(basename(dirname(path)))
  return undefined
}

/** Shared by folder preview and launch; unknown architecture is not Flux1. */
export function inspectLocalImageModel(input: string, requestedQuantize = 0) {
  const resolution = resolveLocalImageModelDirectory(input, requestedQuantize)
  if (!resolution) return { success: false as const, error: 'Select a local model folder.' }
  if (resolution.kind !== 'model') {
    const error = localImageModelError(resolution)
    return { success: false as const, error: error.message, errorCode: error.code, errorParams: error.params }
  }
  const model = resolveImageModelForLocalDirectory(resolution.path)
  return {
    success: true as const,
    path: resolution.path,
    quantize: resolution.quantize ?? 0,
    quantizeSource: resolution.quantizeSource,
    mfluxVersion: resolution.mfluxVersion,
    model,
  }
}

/** "q8", "8bit", "8-bit", "4_bit", "int4": a precision name, not a model name. */
export function isVariantFolderName(name: string): boolean {
  return /^(?:q\d{1,2}|(?:int|fp|bf)\d{1,2}|\d{1,2}[-_ ]?bit)$/i.test(name.trim())
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
  code: 'variants' | 'notAModelDirectory' | 'volumeUnavailable' | 'pathMissing' | 'unsupportedImageFormat'
  params: Record<string, string>
  message: string
}

export function localImageModelError(res: Exclude<LocalImageModelResolution, { kind: 'model' }>): LocalImageModelError {
  switch (res.kind) {
    case 'unsupported-format':
      return {
        code: 'unsupportedImageFormat',
        params: { path: res.path, config: res.configPath, format: res.format },
        message: `"${res.path}" declares ${res.format} in ${res.configPath}. This format is not supported by the mflux image runtime. Select an mflux-compatible export; changing the adapter or requested bit size does not convert the weights.`,
      }
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
