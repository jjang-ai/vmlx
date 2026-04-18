"""MXFP4 (unsloth-like) load via mlx_vlm.utils.load_model — matches LM Studio path.

Note: mx.eval below is MLX tensor realization, not Python's built-in.
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


def png(w, h):
    from PIL import Image
    p = Path(f"/tmp/_mxfp4_{w}.png")
    Image.new("RGB", (w, h), (200,50,50)).save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--lazy", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    mem("C0 startup")

    from mlx_vlm.utils import load_model, load_processor
    print(f"\nLoading {args.model_path} ...", flush=True)
    t = time.perf_counter()
    model = load_model(Path(args.model_path), lazy=args.lazy)
    processor = load_processor(Path(args.model_path), add_generation_prompt=True)
    print(f"  load time: {time.perf_counter()-t:.2f}s", flush=True)
    mem("C1 after load")

    img_path = png(args.img_size, args.img_size)
    prompt = ("<|im_start|>user\n"
              "<|vision_start|><|image_pad|><|vision_end|>"
              "Describe this image briefly.<|im_end|>\n"
              "<|im_start|>assistant\n")
    inputs = processor(text=[prompt], images=[[str(img_path)]],
                       return_tensors="mlx", padding=True)
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape} dtype={v.dtype}")

    t = time.perf_counter()
    kw = {k: v for k, v in inputs.items() if k != "input_ids"}
    out = model(inputs["input_ids"], **kw)
    mx.eval(out)
    print(f"  image prefill: {time.perf_counter()-t:.2f}s", flush=True)
    mem("C4 after image prefill")

    print(f"\nWall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
