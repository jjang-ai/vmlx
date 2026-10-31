import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

/** Small complete safetensor payload for header-admission tests, not inference. */
export function addDraftAdmissionWeights(root: string, child: string): void {
  const target = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
  target.text_config = { hidden_size: 32, vocab_size: 64, num_hidden_layers: 2 }
  writeFileSync(join(root, 'config.json'), JSON.stringify(target))
  const cfg = JSON.parse(readFileSync(join(root, child, 'config.json'), 'utf8'))
  Object.assign(cfg, { hidden_size: 32, vocab_size: 64, num_hidden_layers: 1,
    dflash_config: { ...cfg.dflash_config, target_layer_ids: [0] } })
  writeFileSync(join(root, child, 'config.json'), JSON.stringify(cfg))
  const header = Buffer.from(JSON.stringify({
    'fc.weight': { dtype: 'F16', shape: [32, 32], data_offsets: [0, 2048] },
    'layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [32, 32], data_offsets: [2048, 4096] }
  }))
  const prefix = Buffer.alloc(8)
  prefix.writeBigUInt64LE(BigInt(header.length))
  writeFileSync(join(root, child, 'model.safetensors'), Buffer.concat([prefix, header, Buffer.alloc(4096)]))
}

/**
 * A MiniMax-M3.1-shaped bundle (`minimax_m3_vl` target, `<bundle>/<child>` drafter): `DSparkMiniMaxDraftModel` config
 * with its geometry under `text_config`, taps in `dspark_target_layer_ids`, tensors under the release's
 * `language_model.model.dspark.*` / `.decoder_layer.` names, MXFP8 linears (`weight_scale_inv`) and a BF16 fc.
 * Small complete payload for header-admission tests, not inference.
 */
export function writeMinimaxDsparkBundle(root: string, child = 'dspark', modelType = 'minimax_m3_vl'): void {
  writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: modelType,
    text_config: { hidden_size: 32, vocab_size: 64, num_hidden_layers: 4 } }))
  mkdirSync(join(root, child), { recursive: true })
  writeFileSync(join(root, child, 'config.json'), JSON.stringify({
    architectures: ['DSparkMiniMaxDraftModel'], model_type: 'minimax_m3_vl', hidden_size: 32,
    quantization_config: { quant_method: 'mxfp8', weight_block_size: [1, 32] },
    text_config: { architectures: ['DSparkMiniMaxDraftModel'], hidden_size: 32, vocab_size: 64, num_hidden_layers: 1,
      dspark_block_size: 7, dspark_target_layer_ids: [1, 3],
      dflash_config: { num_target_layers: 4, block_size: 7, target_layer_ids: [1, 3] } },
  }))
  const p = 'language_model.model.dspark.'
  const specs: Array<[string, string, number[], number]> = [
    [`${p}fc.weight`, 'BF16', [32, 64], 2],
    [`${p}final_norm.weight`, 'BF16', [32], 2],
    [`${p}layers.0.decoder_layer.self_attn.q_proj.weight`, 'F8_E4M3', [32, 32], 1],
    [`${p}layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv`, 'U8', [32, 1], 1],
  ]
  const header: Record<string, unknown> = {}
  let offset = 0
  for (const [name, dtype, shape, bytes] of specs) {
    const size = shape.reduce((a, b) => a * b, 1) * bytes
    header[name] = { dtype, shape, data_offsets: [offset, offset + size] }
    offset += size
  }
  const json = Buffer.from(JSON.stringify(header))
  const prefix = Buffer.alloc(8)
  prefix.writeBigUInt64LE(BigInt(json.length))
  const file = 'model-dspark-00001.safetensors'
  writeFileSync(join(root, child, file), Buffer.concat([prefix, json, Buffer.alloc(offset)]))
  writeFileSync(join(root, child, 'model.safetensors.index.json'), JSON.stringify({
    weight_map: Object.fromEntries(specs.map(([name]) => [name, file])) }))
}
