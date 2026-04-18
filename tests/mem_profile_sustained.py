"""Sustained multi-request memory profile to detect accumulation leaks.

mx.eval below = MLX tensor realization, not Python built-in.
"""
import argparse, gc, os, time
from pathlib import Path
import mlx.core as mx
import psutil


def mem(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss/1e9
    a = mx.get_active_memory()/1e9
    pk = mx.get_peak_memory()/1e9
    c = mx.get_cache_memory()/1e9
    print(f"  [{tag:32s}] RSS={rss:6.2f}GB active={a:6.2f}GB peak={pk:6.2f}GB cache={c:6.2f}GB",
          flush=True)


def png(w, h, rgb):
    from PIL import Image
    p = Path(f"/tmp/_sust_{w}_{rgb[0]:03d}.png")
    Image.new("RGB", (w, h), rgb).save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--clear-each", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    mem("C0 startup")

    from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
    model, processor = load_jangtq_vlm_model(args.model_path)
    mem("C1 after load")

    if args.force_eval:
        import mlx.utils as _mu
        flat = _mu.tree_flatten(model.parameters())
        for i in range(0, len(flat), 200):
            mx.eval(*[v for _, v in flat[i:i+200]])
        mem("C2 after force eval")

    for it in range(args.iters):
        img_path = png(args.img_size, args.img_size,
                       (50 + (it*30)%200, 100, 150))
        prompt = ("<|im_start|>user\n"
                  "<|vision_start|><|image_pad|><|vision_end|>"
                  f"Describe request {it}.<|im_end|>\n"
                  "<|im_start|>assistant\n")
        inputs = processor(text=[prompt], images=[[str(img_path)]],
                           return_tensors="mlx", padding=True)
        t = time.perf_counter()
        kw = {k: v for k, v in inputs.items() if k != "input_ids"}
        out = model(inputs["input_ids"], **kw)
        mx.eval(out)
        dt = time.perf_counter() - t
        mem(f"iter {it:02d} ({dt:.2f}s)")
        del inputs, out, kw
        if args.clear_each:
            gc.collect()
            try: mx.clear_cache()
            except AttributeError: mx.metal.clear_cache()

    print(f"\nWall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
