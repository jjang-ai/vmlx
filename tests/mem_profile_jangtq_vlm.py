"""Memory profile harness for JANGTQ VLM load + image + video processing.

Records RSS + MLX active/peak/cache at checkpoints:
  C0  startup (imports only)
  C1  after load_jangtq_vlm_model
  C2  after force-materialize (--force-eval)
  C3  after mx.clear_cache (--clear-cache)
  C4  after single-image prefill
  C5  after multi-frame (video) prefill

mx.eval below = MLX tensor realization, not Python eval.
"""
import argparse
import gc
import os
import time
from pathlib import Path

import mlx.core as mx
import psutil


def _mem(tag: str):
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1e9
    try:
        active = mx.get_active_memory() / 1e9
    except Exception:
        active = -1.0
    try:
        peak = mx.get_peak_memory() / 1e9
    except Exception:
        peak = -1.0
    try:
        cache = mx.get_cache_memory() / 1e9
    except Exception:
        cache = -1.0
    print(f"  [{tag:32s}] RSS={rss_gb:6.2f}GB  active={active:6.2f}GB  "
          f"peak={peak:6.2f}GB  cache={cache:6.2f}GB", flush=True)
    return {"tag": tag, "rss": rss_gb, "active": active, "peak": peak, "cache": cache}


def _png(w, h, rgb=(200, 50, 50)):
    from PIL import Image
    img = Image.new("RGB", (w, h), rgb)
    fp = Path(f"/tmp/_memprobe_{w}x{h}.png")
    img.save(fp)
    return fp


def _video_frames(n, w, h):
    from PIL import Image
    paths = []
    for i in range(n):
        img = Image.new("RGB", (w, h), (20 + (i * 7) % 200, 100, 150))
        fp = Path(f"/tmp/_memprobe_vf{i}.png")
        img.save(fp)
        paths.append(str(fp))
    return paths


# Qwen3-VL image marker — from chat_template.jinja:
# '<|vision_start|><|image_pad|><|vision_end|>' for image,
# '<|vision_start|><|video_pad|><|vision_end|>' for video.
def _build_prompt(media_marker: str, text: str) -> str:
    return (f"<|im_start|>user\n{media_marker}{text}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def _do_prefill(model, inputs):
    t = time.perf_counter()
    kw = {k: v for k, v in inputs.items() if k != "input_ids"}
    out = model(inputs["input_ids"], **kw)
    mx.eval(out)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--frame-size", type=int, default=384)
    ap.add_argument("--skip-image", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    log = []
    t0 = time.perf_counter()

    log.append(_mem("C0 startup"))

    from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
    print(f"\nLoading {args.model_path} ...", flush=True)
    t = time.perf_counter()
    model, processor = load_jangtq_vlm_model(args.model_path)
    print(f"  load time: {time.perf_counter()-t:.2f}s", flush=True)
    log.append(_mem("C1 after load"))

    if args.force_eval:
        import mlx.utils as _mu
        print("  force materialize params ...", flush=True)
        flat = _mu.tree_flatten(model.parameters())
        for i in range(0, len(flat), 200):
            mx.eval(*[v for _, v in flat[i:i + 200]])
        log.append(_mem("C2 after force eval"))

    if args.clear_cache:
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            mx.metal.clear_cache()
        log.append(_mem("C3 after clear_cache"))

    # --------- IMAGE PREFILL ---------
    if not args.skip_image:
        print(f"\nImage prefill (solid {args.img_size}x{args.img_size}) ...", flush=True)
        img_path = _png(args.img_size, args.img_size)
        prompt = _build_prompt(
            "<|vision_start|><|image_pad|><|vision_end|>",
            "Describe this image briefly.",
        )
        try:
            inputs = processor(
                text=[prompt], images=[[str(img_path)]],
                return_tensors="mlx", padding=True,
            )
            for k, v in inputs.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape} dtype={v.dtype}")
            dt = _do_prefill(model, inputs)
            print(f"  image prefill: {dt:.2f}s", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  image prefill threw ({type(e).__name__}): {e}", flush=True)
        log.append(_mem("C4 after image prefill"))

    # --------- VIDEO PREFILL ---------
    if not args.skip_video:
        print(f"\nVideo prefill ({args.frames} frames @ {args.frame_size}) ...", flush=True)
        frame_paths = _video_frames(args.frames, args.frame_size, args.frame_size)
        prompt = _build_prompt(
            "<|vision_start|><|video_pad|><|vision_end|>",
            "Describe what's happening in this video.",
        )
        try:
            vinputs = processor(
                text=[prompt], videos=[frame_paths],
                return_tensors="mlx", padding=True,
            )
            for k, v in vinputs.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={v.shape} dtype={v.dtype}")
            dt = _do_prefill(model, vinputs)
            print(f"  video prefill: {dt:.2f}s", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  video prefill threw ({type(e).__name__}): {e}", flush=True)
        log.append(_mem("C5 after video prefill"))

    print("\n=== SUMMARY ===", flush=True)
    for row in log:
        print(f"  {row['tag']:32s} RSS={row['rss']:6.2f}  "
              f"active={row['active']:6.2f}  peak={row['peak']:6.2f}  "
              f"cache={row['cache']:6.2f}")
    print(f"\nWall: {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
