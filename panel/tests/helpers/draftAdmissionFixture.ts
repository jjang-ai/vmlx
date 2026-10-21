import { readFileSync, writeFileSync } from 'node:fs'
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
