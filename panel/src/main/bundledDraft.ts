import { openSync, closeSync, readSync, fstatSync, readdirSync, readFileSync } from 'fs'
import { basename, join } from 'path'

// Header-only admission; the Python loader still strictly checks every tensor.
// Never read multi-GB weight payloads into Electron to discover a draft.
function tensorsIn(file: string): Record<string, any> {
  const fd = openSync(file, 'r')
  try {
    const size = fstatSync(fd).size
    const prefix = Buffer.alloc(8)
    if (readSync(fd, prefix, 0, 8, 0) !== 8) throw Error('Truncated weight prefix')
    const length = Number(prefix.readBigUInt64LE())
    if (!Number.isSafeInteger(length) || length <= 0 || length > 256 * 1024 * 1024 || length + 8 > size) throw Error('Invalid weight header length')
    const bytes = Buffer.alloc(length)
    let read = 0
    while (read < length) {
      const n = readSync(fd, bytes, read, length - read, 8 + read)
      if (!n) throw Error('Truncated weight header')
      read += n
    }
    const header = JSON.parse(bytes.toString('utf8'))
    delete header.__metadata__
    const dtypeBytes: Record<string, number> = { BOOL: 1, U8: 1, I8: 1, F8_E4M3: 1, F8_E5M2: 1, I16: 2, U16: 2, F16: 2, BF16: 2, I32: 4, U32: 4, F32: 4, I64: 8, U64: 8, F64: 8 }
    const intervals: number[][] = []
    for (const t of Object.values(header) as any[]) {
      if (!Array.isArray(t.shape) || t.shape.some((n: unknown) => !Number.isSafeInteger(n) || Number(n) < 0)) throw Error('Invalid tensor shape')
      if (!Array.isArray(t.data_offsets) || t.data_offsets.length !== 2) throw Error('Invalid tensor offsets')
      const [start, end] = t.data_offsets
      const count = t.shape.reduce((a: number, b: number) => a * b, 1)
      if (![start, end, count].every(Number.isSafeInteger) || start < 0 || end < start || end > size - length - 8 || !dtypeBytes[t.dtype] || end - start !== count * dtypeBytes[t.dtype]) throw Error('Invalid tensor payload')
      intervals.push([start, end])
    }
    intervals.sort((a, b) => a[0] - b[0] || a[1] - b[1])
    let end = 0
    for (const range of intervals) {
      if (range[0] !== end) throw Error('Overlapping or missing tensor payload')
      end = range[1]
    }
    if (end !== size - length - 8) throw Error('Incomplete tensor payload')
    return header
  } finally { closeSync(fd) }
}

/**
 * MiniMax-M3.1 `DSparkMiniMaxDraftModel` drafter (engine dspark/drafter.is_minimax_config): geometry under
 * `text_config`, taps in `dspark_target_layer_ids`, tensors under the release's module prefix.
 */
function isMinimaxDraftConfig(cfg: any): boolean {
  const archs: unknown[] = Array.isArray(cfg?.architectures) ? cfg.architectures : []
  return archs.includes('DSparkMiniMaxDraftModel') || 'dspark_block_size' in (cfg?.text_config || cfg || {})
}

/** engine bundled_draft._draft_geometry: the flat DFlash / DSpark layout or the MiniMax-M3.1 layout. */
function draftGeometry(cfg: any) {
  if (isMinimaxDraftConfig(cfg)) {
    const text = cfg.text_config || cfg
    return { hidden: text.hidden_size, vocab: text.vocab_size, layers: text.num_hidden_layers, minimax: true,
      taps: text.dspark_target_layer_ids ?? text.dspark_config?.target_layer_ids,
      numTargetLayers: text.num_target_layers ?? cfg.num_target_layers }
  }
  return { hidden: cfg.hidden_size, vocab: cfg.vocab_size, layers: cfg.num_hidden_layers, minimax: false,
    taps: cfg.dflash_config?.target_layer_ids, numTargetLayers: cfg.num_target_layers }
}

/** engine dspark/drafter.minimax_tensor_name: the release's tensor names under the drafter's module names. */
function minimaxTensorName(name: string): string {
  const prefix = 'language_model.model.dspark.'
  const stripped = (name.startsWith(prefix) ? name.slice(prefix.length) : name).split('.decoder_layer.').join('.')
  return stripped.startsWith('final_norm.') ? 'norm.' + stripped.slice('final_norm.'.length) : stripped
}

/** engine bundled_draft._check_fc: the drafter's fc projection fits hidden x (hidden * taps). */
function fcFits(tensors: Record<string, any>, quant: any, hidden: number, width: number): boolean {
  const fc = tensors['fc.weight']
  if ('fc.weight_scale_inv' in tensors) {
    // MXFP8 as released: e4m3 bytes + one e8m0 scale per 32 inputs.
    return ['U8', 'F8_E4M3'].includes(fc?.dtype) && JSON.stringify(fc.shape) === JSON.stringify([hidden, width]) &&
      width % 32 === 0 && JSON.stringify(tensors['fc.weight_scale_inv'].shape) === JSON.stringify([hidden, width / 32])
  }
  if ('fc.scales' in tensors) {
    // mlx-lm's per-module entry for fc (keyed by its live path), else the block's top-level settings.
    const q = quant.fc && typeof quant.fc === 'object' ? quant.fc : quant
    const mode = q.mode || 'affine'
    if (mode === 'mxfp8') {
      if (![undefined, 8].includes(q.bits) || ![undefined, 32].includes(q.group_size) || width % 32 || fc?.dtype !== 'U32') return false
      if (JSON.stringify(tensors['fc.scales'].shape) !== JSON.stringify([hidden, width / 32]) || 'fc.biases' in tensors) return false
      width = width * 8 / 32
    } else if (mode === 'affine') {
      if (![2, 3, 4, 5, 6, 8].includes(q.bits) || ![32, 64, 128].includes(q.group_size)) return false
      if (width * q.bits % 32 || width % q.group_size || fc?.dtype !== 'U32') return false
      for (const key of ['fc.scales', 'fc.biases']) {
        if (JSON.stringify(tensors[key]?.shape) !== JSON.stringify([hidden, width / q.group_size])) return false
      }
      width = width * q.bits / 32
    } else return false
  } else if (!['F16', 'BF16', 'F32'].includes(fc?.dtype)) return false
  return JSON.stringify(fc.shape) === JSON.stringify([hidden, width])
}

/** Mirrors engine bundled_draft admission, including stored affine / MXFP8 fc geometry and the MiniMax layout. */
export function bundledDraftFits(draft: string, target: string): boolean {
  try {
    const cfg = JSON.parse(readFileSync(join(draft, 'config.json'), 'utf8'))
    const root = JSON.parse(readFileSync(join(target, 'config.json'), 'utf8'))
    const text = root.text_config || root
    const { hidden_size: hidden, vocab_size: vocab, num_hidden_layers: layers } = text
    if (![hidden, vocab, layers].every(n => Number.isSafeInteger(n) && n > 0)) return false
    const geo = draftGeometry(cfg)
    if (geo.hidden !== hidden || !(geo.vocab > 0 && geo.vocab <= vocab)) return false
    const taps = geo.taps
    if (!Array.isArray(taps) || !taps.length || taps.some(i => !Number.isInteger(i) || i < 0 || i >= layers)) return false
    if (geo.numTargetLayers !== undefined && geo.numTargetLayers !== null && geo.numTargetLayers !== layers) return false
    const rename = geo.minimax ? minimaxTensorName : (name: string) => name
    const files = readdirSync(draft)
    const shards = files.filter(f => f.endsWith('.safetensors') && !f.startsWith('._'))
    if (!shards.length) return false
    const tensors: Record<string, any> = {}
    const owners: Record<string, string> = {}
    for (const file of shards) {
      const header = tensorsIn(join(draft, file))
      const names = Object.keys(header).map(rename)
      if (new Set(names).size !== names.length || names.some(key => key in tensors)) return false
      Object.keys(header).forEach((key, i) => { tensors[names[i]] = header[key]; owners[key] = file })
    }
    if (!fcFits(tensors, cfg.quantization || cfg.quantization_config || {}, hidden, hidden * taps.length)) return false
    if (!Number.isSafeInteger(geo.layers) || geo.layers <= 0 || geo.layers > Object.keys(tensors).length) return false
    for (let i = 0; i < geo.layers; i++) {
      if (!tensors[`layers.${i}.self_attn.q_proj.weight`]) return false
    }
    for (const index of files.filter(f => f.endsWith('.safetensors.index.json') && !f.startsWith('._'))) {
      const mapping = JSON.parse(readFileSync(join(draft, index), 'utf8')).weight_map
      for (const [name, file] of Object.entries(mapping)) {
        if (typeof file !== 'string' || basename(file) !== file || owners[name] !== file) return false
      }
    }
    return true
  } catch { return false }
}
