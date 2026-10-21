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

/** Mirrors engine bundled_draft admission, including stored affine fc geometry. */
export function bundledDraftFits(draft: string, target: string): boolean {
  try {
    const cfg = JSON.parse(readFileSync(join(draft, 'config.json'), 'utf8'))
    const root = JSON.parse(readFileSync(join(target, 'config.json'), 'utf8'))
    const text = root.text_config || root
    const { hidden_size: hidden, vocab_size: vocab, num_hidden_layers: layers } = text
    if (![hidden, vocab, layers].every(n => Number.isSafeInteger(n) && n > 0)) return false
    if (cfg.hidden_size !== hidden || !(cfg.vocab_size > 0 && cfg.vocab_size <= vocab)) return false
    const taps = cfg.dflash_config?.target_layer_ids
    if (!Array.isArray(taps) || !taps.length || taps.some(i => !Number.isInteger(i) || i < 0 || i >= layers)) return false
    if (cfg.num_target_layers !== undefined && cfg.num_target_layers !== layers) return false
    const files = readdirSync(draft)
    const shards = files.filter(f => f.endsWith('.safetensors') && !f.startsWith('._'))
    if (!shards.length) return false
    const tensors: Record<string, any> = {}
    const owners: Record<string, string> = {}
    for (const file of shards) {
      const header = tensorsIn(join(draft, file))
      if (Object.keys(header).some(key => key in tensors)) return false
      Object.assign(tensors, header)
      for (const key of Object.keys(header)) owners[key] = file
    }
    const fc = tensors['fc.weight']
    let width = hidden * taps.length
    if ('fc.scales' in tensors) {
      const q = cfg.quantization || cfg.quantization_config || {}
      if ((q.mode || 'affine') !== 'affine' || ![2, 3, 4, 5, 6, 8].includes(q.bits) || ![32, 64, 128].includes(q.group_size)) return false
      if (width * q.bits % 32 || width % q.group_size || fc?.dtype !== 'U32') return false
      for (const key of ['fc.scales', 'fc.biases']) {
        if (JSON.stringify(tensors[key]?.shape) !== JSON.stringify([hidden, width / q.group_size])) return false
      }
      width = width * q.bits / 32
    } else if (!['F16', 'BF16', 'F32'].includes(fc?.dtype)) return false
    if (JSON.stringify(fc.shape) !== JSON.stringify([hidden, width])) return false
    if (!Number.isSafeInteger(cfg.num_hidden_layers) || cfg.num_hidden_layers <= 0 || cfg.num_hidden_layers > Object.keys(tensors).length) return false
    for (let i = 0; i < cfg.num_hidden_layers; i++) {
      if (!tensors[`layers.${i}.self_attn.q_proj.weight`]) return false
    }
    for (const index of files.filter(f => f.endsWith('.safetensors.index.json'))) {
      const mapping = JSON.parse(readFileSync(join(draft, index), 'utf8')).weight_map
      for (const [name, file] of Object.entries(mapping)) {
        if (typeof file !== 'string' || basename(file) !== file || owners[name] !== file) return false
      }
    }
    return true
  } catch { return false }
}
