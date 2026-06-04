#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import types
import wave
from pathlib import Path
from typing import Any

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmlx_engine.models.mllm import MLXMultimodalLM

ROWS = {
    "mxfp4": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP4",
    "mxfp8": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8",
    "mxfp8_attnfp16": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE",
    "mxfp8_attnfp16_l024": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE",
    "jang4m": "/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
}

VARIANTS = [
    {"name": "baseline"},
    {"name": "input_scale_0_25", "input_scale": 0.25},
    {"name": "input_scale_0_5", "input_scale": 0.5},
    {"name": "input_scale_2", "input_scale": 2.0},
    {"name": "output_scale_0_25", "output_scale": 0.25},
    {"name": "output_scale_0_5", "output_scale": 0.5},
    {"name": "output_scale_2", "output_scale": 2.0},
    {"name": "output_scale_4", "output_scale": 4.0},
    {"name": "zero_audio", "output_scale": 0.0},
]

BAD_MARKERS = (
    "transcriptionno",
    "cannot hear",
    "cannot process audio",
    "no audio",
    "no sound",
    "thought",
    "<|channel>",
)


def make_speech_wav() -> Path:
    aiff = None
    wav_path = None
    try:
        fd, aiff = tempfile.mkstemp(suffix=".aiff")
        os.close(fd)
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["say", "-o", aiff, "audio present"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff, wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return Path(wav_path)
    except Exception:
        if wav_path:
            try:
                Path(wav_path).unlink()
            except OSError:
                pass
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sample_rate = 16000
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(int(0.8 * sample_rate)):
                amp = int(0.25 * 32767 * math.sin(2 * math.pi * 440.0 * i / sample_rate))
                wf.writeframesraw(struct.pack("<h", amp))
        return Path(wav_path)
    finally:
        if aiff:
            try:
                Path(aiff).unlink()
            except OSError:
                pass


def audio_message(audio_path: Path) -> list[dict[str, Any]]:
    data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the attached audio. Reply with only the spoken words.",
                },
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
            ],
        }
    ]


def no_audio_message() -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "The current user message has no audio attachment. Answer exactly NONE.",
        },
        {"role": "user", "content": "Is there an audio attachment in this current message?"},
    ]


def verdict(text: str) -> dict[str, Any]:
    lower = text.lower()
    ok = "audio" in lower and "present" in lower and not any(m in lower for m in BAD_MARKERS)
    return {"ok": ok, "bad_markers": [m for m in BAD_MARKERS if m in lower]}


def stats_for_array(x: mx.array) -> dict[str, Any]:
    xf = x.astype(mx.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(mx.min(xf).item()),
        "max": float(mx.max(xf).item()),
        "mean": float(mx.mean(xf).item()),
        "rms": float(mx.sqrt(mx.mean(xf * xf)).item()),
    }


def patch_audio_transform(model: Any, variant: dict[str, Any], capture: dict[str, Any]) -> None:
    original = model.get_audio_features

    def wrapped(self, input_features, input_features_mask=None):
        capture.setdefault("input", stats_for_array(input_features))
        x = input_features
        if "input_scale" in variant:
            x = x * float(variant["input_scale"])
        out = original(x, input_features_mask)
        capture.setdefault("projected_before", stats_for_array(out))
        if "output_scale" in variant:
            out = out * float(variant["output_scale"])
        capture.setdefault("projected_after", stats_for_array(out))
        return out

    model.get_audio_features = types.MethodType(wrapped, model)


def run_row(row: str, variants: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    path = ROWS[row]
    audio_path = make_speech_wav()
    result: dict[str, Any] = {"row": row, "path": path, "variants": []}
    try:
        engine = MLXMultimodalLM(path, enable_cache=False)
        t0 = time.perf_counter()
        engine.load()
        result["load_sec"] = round(time.perf_counter() - t0, 3)
        original_get_audio = engine.model.get_audio_features
        for variant in variants:
            engine.model.get_audio_features = original_get_audio
            capture: dict[str, Any] = {}
            if variant.get("name") != "baseline":
                patch_audio_transform(engine.model, variant, capture)
            started = time.perf_counter()
            try:
                out = engine.chat(
                    audio_message(audio_path),
                    max_tokens=max_tokens,
                    temperature=0,
                    enable_thinking=False,
                    use_cache=False,
                )
                text = out.text
                error = None
            except Exception as exc:
                text = ""
                error = repr(exc)
            item = {
                "name": variant["name"],
                "variant": variant,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "text": text,
                "error": error,
                "verdict": verdict(text),
                "audio_stats": capture,
            }
            result["variants"].append(item)
        engine.model.get_audio_features = original_get_audio
        no_audio = engine.chat(
            no_audio_message(),
            max_tokens=32,
            temperature=0,
            enable_thinking=False,
            use_cache=False,
        )
        result["no_audio_text"] = no_audio.text
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass
    result["status"] = "pass" if any(v["verdict"]["ok"] for v in result["variants"]) else "fail"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", choices=sorted(ROWS), default=["mxfp4", "mxfp8"])
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out", default="build/current-gemma4-12b-audio-transform-probe.json")
    args = parser.parse_args()

    variants = VARIANTS
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in VARIANTS if v["name"] in wanted]
    summary = {"rows": args.rows, "results": []}
    for row in args.rows:
        summary["results"].append(run_row(row, variants, args.max_tokens))
    summary["status"] = "pass" if all(r["status"] == "pass" for r in summary["results"]) else "fail"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print("status=" + summary["status"])
    for row in summary["results"]:
        print(f"{row['row']}: status={row['status']} load_sec={row.get('load_sec')}")
        for variant in row["variants"]:
            print(f"  {variant['name']}: ok={variant['verdict']['ok']} text={variant['text']!r} err={variant['error']}")
        print(f"  no_audio={row.get('no_audio_text')!r}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
