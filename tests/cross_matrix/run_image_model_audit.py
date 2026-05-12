#!/usr/bin/env python3
"""Live image-model audit for vMLX/mflux server rows.

This is intentionally a live gate, not a unit test. It launches the same
`vmlx_engine.cli serve` command the app uses for local image models, calls the
OpenAI-compatible image endpoint, records the output size/timing, then shuts
the server down before the next row.

The runner keeps image diffusion on the native mflux/MLX path. JANGTQ
TurboQuant/MPP/NAX acceleration is only expected for text/VL JANGTQ bundles
with `jang_config.json` / `*.tq_packed` weights, not for these diffusers/mflux
image directories.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OUT_DIR = Path("/tmp/vmlx_family_audit")
IMAGE_ROOT = Path.home() / ".mlxstudio" / "models" / "image"


@dataclass(frozen=True)
class ImageRow:
    row_id: str
    model_id: str
    display_name: str
    path: str
    mode: str
    mflux_class: str
    quantize: int | None
    endpoint: str
    steps: int
    guidance: float
    size: str = "64x64"
    n: int = 1
    expected_local: bool = True
    requires_mask: bool = False


ROWS: dict[str, ImageRow] = {
    "schnell_4bit": ImageRow(
        row_id="schnell_4bit",
        model_id="schnell",
        display_name="Flux Schnell 4-bit",
        path=str(IMAGE_ROOT / "FLUX.1-schnell-mflux-4bit"),
        mode="generate",
        mflux_class="Flux1",
        quantize=4,
        endpoint="/v1/images/generations",
        steps=1,
        guidance=0.0,
    ),
    "z_image_turbo_4bit": ImageRow(
        row_id="z_image_turbo_4bit",
        model_id="z-image-turbo",
        display_name="Z-Image Turbo 4-bit",
        path=str(IMAGE_ROOT / "Z-Image-Turbo-mflux-4bit"),
        mode="generate",
        mflux_class="ZImage",
        quantize=4,
        endpoint="/v1/images/generations",
        steps=1,
        guidance=1.0,
    ),
    "klein_4b_4bit": ImageRow(
        row_id="klein_4b_4bit",
        model_id="flux2-klein-4b",
        display_name="FLUX.2 Klein 4B 4-bit",
        path=str(IMAGE_ROOT / "FLUX.2-klein-4B-mflux-4bit"),
        mode="generate",
        mflux_class="Flux2Klein",
        quantize=4,
        endpoint="/v1/images/generations",
        steps=4,
        guidance=3.5,
    ),
    "qwen_image_4bit": ImageRow(
        row_id="qwen_image_4bit",
        model_id="qwen-image",
        display_name="Qwen Image 4-bit",
        path=str(IMAGE_ROOT / "qwen-image-mflux-4bit"),
        mode="generate",
        mflux_class="QwenImage",
        quantize=4,
        endpoint="/v1/images/generations",
        steps=4,
        guidance=4.0,
    ),
    "klein_9b_full": ImageRow(
        row_id="klein_9b_full",
        model_id="flux2-klein-9b",
        display_name="FLUX.2 Klein 9B full",
        path=str(IMAGE_ROOT / "FLUX.2-klein-9B"),
        mode="generate",
        mflux_class="Flux2Klein",
        quantize=None,
        endpoint="/v1/images/generations",
        steps=4,
        guidance=3.5,
    ),
    "qwen_image_edit_full": ImageRow(
        row_id="qwen_image_edit_full",
        model_id="qwen-image-edit",
        display_name="Qwen Image Edit full",
        path=str(IMAGE_ROOT / "Qwen-Image-Edit"),
        mode="edit",
        mflux_class="QwenImageEdit",
        quantize=None,
        endpoint="/v1/images/edits",
        steps=4,
        guidance=4.0,
    ),
    "flux_fill_full": ImageRow(
        row_id="flux_fill_full",
        model_id="fill",
        display_name="Flux Fill full",
        path=str(IMAGE_ROOT / "FLUX.1-Fill-dev"),
        mode="edit",
        mflux_class="Flux1Fill",
        quantize=None,
        endpoint="/v1/images/edits",
        steps=4,
        guidance=30.0,
        expected_local=False,
        requires_mask=True,
    ),
}


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_health(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            return _http_json("GET", f"{base_url}/health", timeout=2.0)
        except Exception as exc:  # server not ready yet
            last_error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout:.1f}s: {last_error}")


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def _make_input_images(tmp_dir: Path) -> tuple[str, str]:
    from PIL import Image, ImageDraw

    source = tmp_dir / "source.png"
    mask = tmp_dir / "mask.png"

    img = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((18, 18, 46, 46), fill=(0, 64, 255))
    img.save(source)

    mask_img = Image.new("L", (64, 64), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.rectangle((20, 20, 44, 44), fill=255)
    mask_img.save(mask)

    return _file_b64(source), _file_b64(mask)


def _file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _request_payload(row: ImageRow, tmp_dir: Path) -> dict[str, Any]:
    if row.mode == "generate":
        return {
            "model": row.model_id,
            "prompt": "a simple blue square app icon on a white background",
            "n": row.n,
            "size": row.size,
            "steps": row.steps,
            "guidance": row.guidance,
            "seed": 123,
            "response_format": "b64_json",
        }

    image_b64, mask_b64 = _make_input_images(tmp_dir)
    payload: dict[str, Any] = {
        "model": row.model_id,
        "prompt": "turn the blue square into a red square",
        "image": image_b64,
        "n": row.n,
        "size": row.size,
        "steps": row.steps,
        "guidance": row.guidance,
        "seed": 123,
    }
    if row.requires_mask:
        payload["mask"] = mask_b64
        payload["prompt"] = "fill the masked square with a red color"
    return payload


def _image_output_summary(response: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    import io

    data = response.get("data") or []
    images = []
    for item in data:
        b64 = item.get("b64_json") or ""
        decoded_len = 0
        image_format = None
        image_size = None
        image_error = None
        try:
            raw = base64.b64decode(b64)
            decoded_len = len(raw)
            with Image.open(io.BytesIO(raw)) as img:
                image_format = img.format
                image_size = list(img.size)
                img.verify()
        except Exception as exc:
            image_error = str(exc)
        images.append(
            {
                "b64_len": len(b64),
                "decoded_bytes": decoded_len,
                "format": image_format,
                "size": image_size,
                "image_error": image_error,
                "seed": item.get("seed"),
                "has_revised_prompt": bool(item.get("revised_prompt")),
            }
        )
    return {
        "image_count": len(images),
        "images": images,
        "usage": response.get("usage"),
    }


def run_row(row: ImageRow, py: str, port: int, load_timeout: float, request_timeout: float) -> dict[str, Any]:
    started = time.time()
    path = Path(row.path)
    result: dict[str, Any] = {
        "row": asdict(row),
        "path_exists": path.exists(),
        "path_size_gb": None,
        "acceleration_expected": "mflux_native_not_jangtq_mpp_nax",
        "started_at": started,
        "ok": False,
    }

    if path.exists():
        try:
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            result["path_size_gb"] = round(size / (1024**3), 3)
        except Exception:
            pass
    if not path.exists():
        result["skipped"] = True
        result["skip_reason"] = "model path missing"
        return result
    if not row.expected_local:
        required = ["transformer", "text_encoder", "vae"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            result["skipped"] = True
            result["skip_reason"] = f"model path incomplete, missing {missing}"
            return result

    log_path = OUT_DIR / f"image_{row.row_id}_{int(started)}.log"
    out_json = OUT_DIR / f"image_{row.row_id}_{int(started)}.json"
    cmd = [
        py,
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        str(int(max(request_timeout + 60, 300))),
        "--image-mode",
        row.mode,
        "--served-model-name",
        row.model_id,
        "--mflux-class",
        row.mflux_class,
        "--log-level",
        "INFO",
    ]
    if row.quantize is not None:
        cmd.extend(["--image-quantize", str(row.quantize)])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory(prefix=f"vmlx_img_{row.row_id}_") as td:
        tmp_dir = Path(td)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(OUT_DIR),
                env=env,
            )
            result["pid"] = proc.pid
            result["log"] = str(log_path)
            result["artifact"] = str(out_json)
            base_url = f"http://127.0.0.1:{port}"
            try:
                health = _wait_health(base_url, load_timeout)
                result["health_after_load"] = health
                payload = _request_payload(row, tmp_dir)
                req_started = time.time()
                response = _http_json(
                    "POST",
                    f"{base_url}{row.endpoint}",
                    payload,
                    timeout=request_timeout,
                )
                elapsed = time.time() - req_started
                result["request_elapsed_s"] = round(elapsed, 3)
                result["response_summary"] = _image_output_summary(response)
                result["ok"] = (
                    result["response_summary"]["image_count"] == row.n
                    and all(
                        img["decoded_bytes"] > 512
                        and img["format"] == "PNG"
                        and img["size"] == [64, 64]
                        and not img["image_error"]
                        for img in result["response_summary"]["images"]
                    )
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                result["error"] = f"HTTP {exc.code}: {body}"
            except Exception as exc:
                result["error"] = repr(exc)
            finally:
                _terminate(proc)

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    result["process_returncode"] = proc.returncode
    result["resource_tracker_warning"] = "resource_tracker" in log_text or "leaked semaphore" in log_text
    result["runtime_warning"] = "RuntimeWarning" in log_text
    if result.get("ok") and (result["resource_tracker_warning"] or result["runtime_warning"]):
        result["ok"] = False
    result["log_tail"] = "\n".join(log_text.splitlines()[-80:])
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows",
        default=",".join(ROWS.keys()),
        help=f"Comma-separated rows to run. Available: {','.join(ROWS.keys())}",
    )
    ap.add_argument("--py", default="panel/bundled-python/python/bin/python3")
    ap.add_argument("--port", type=int, default=8158)
    ap.add_argument("--load-timeout", type=float, default=480.0)
    ap.add_argument("--request-timeout", type=float, default=480.0)
    ap.add_argument("--out", default=str(OUT_DIR / "image_model_audit.json"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    py = str(Path(args.py).expanduser().resolve())
    selected = [r.strip() for r in args.rows.split(",") if r.strip()]
    unknown = [r for r in selected if r not in ROWS]
    if unknown:
        print(f"Unknown rows: {unknown}", file=sys.stderr)
        return 2

    results = []
    for offset, row_id in enumerate(selected):
        row = ROWS[row_id]
        print(f"==> {row_id}: {row.display_name}", flush=True)
        result = run_row(
            row,
            py=py,
            port=args.port + offset,
            load_timeout=args.load_timeout,
            request_timeout=args.request_timeout,
        )
        results.append(result)
        status = "PASS" if result.get("ok") else ("SKIP" if result.get("skipped") else "FAIL")
        detail = result.get("error") or result.get("skip_reason") or result.get("response_summary")
        print(f"    {status}: {detail}", flush=True)

    summary = {
        "created_at": time.time(),
        "rows": selected,
        "results": results,
        "all_ok_or_skipped": all(r.get("ok") or r.get("skipped") for r in results),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0 if summary["all_ok_or_skipped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
