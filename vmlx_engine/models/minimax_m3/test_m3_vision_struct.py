"""Isolated STRUCTURAL test for the M3 VL vision stack (random weights, no bundle).
Validates forward wiring + shapes only (NOT numerics)."""
import sys, json
import mlx.core as mx
sys.path.insert(0, "/tmp")
import minimax_m3_vl as V

cfg = json.load(open(sys.argv[1] + "/config.json"))
print("vision hidden:", cfg["vision_config"]["hidden_size"],
      "layers:", cfg["vision_config"]["num_hidden_layers"],
      "heads:", cfg["vision_config"]["num_attention_heads"],
      "projector_hidden_size:", cfg.get("projector_hidden_size"))

stack = V.MiniMaxM3VLVisionStack(cfg)
mx.eval(stack.parameters())

# synthetic: grid [1,4,4] -> 16 patches; pixel_values[16, 3*2*14*14=1176]
gt, gh, gw = 1, 4, 4
N = gt * gh * gw
pv = mx.random.normal((N, 3 * 2 * 14 * 14))
grid = mx.array([[gt, gh, gw]], dtype=mx.int32)
print("input pixel_values:", pv.shape, "grid_thw:", grid.tolist())

out = stack(pv, grid)
mx.eval(out)
merge = cfg["vision_config"].get("img_token_compression_config", {}).get("spatial_merge_size", 2)
expect = (N // (merge ** 2), cfg.get("projector_hidden_size", 6144))
print("output:", out.shape, "expected:", expect)
assert tuple(out.shape) == expect, f"SHAPE MISMATCH {out.shape} != {expect}"
print("STRUCT TEST PASS — forward wiring + shapes OK")

# also print the parameter paths to compare against bundle weight names
paths = []
def walk(d, pfx=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, f"{pfx}.{k}" if pfx else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            walk(v, f"{pfx}.{i}")
    else:
        paths.append(pfx)
walk(stack.parameters())
print("=== sample param paths (vs bundle names) ===")
for p in sorted(paths):
    if any(t in p for t in ("patch_embedding", "pre_layrnorm", "encoder.layers.0.",
                            "multi_modal_projector", "patch_merge")):
        print("  ", p)
