#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(os.environ.get("VMLINUX_BENCH_PYTHON", sys.executable))
if not PYTHON.is_absolute():
    PYTHON = (ROOT / PYTHON).resolve()

ROWS: dict[str, dict[str, str]] = {
    "mxfp4": {
        "name": "mxfp4_audio",
        "served_name": "gemma4-12b-mxfp4-audio-smoke",
        "path": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP4",
    },
    "mxfp8": {
        "name": "mxfp8_audio",
        "served_name": "gemma4-12b-mxfp8-audio-smoke",
        "path": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8",
    },
    "mxfp8_attnfp16": {
        "name": "mxfp8_attnfp16_audio",
        "served_name": "gemma4-12b-mxfp8-attnfp16-audio-smoke",
        "path": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-CANDIDATE",
    },
    "mxfp8_attnfp16_l024": {
        "name": "mxfp8_attnfp16_l024_audio",
        "served_name": "gemma4-12b-mxfp8-attnfp16-l024-audio-smoke",
        "path": "/Users/example/models/OsaurusAI/gemma-4-12B-it-MXFP8-ATTNFP16-L0-24-FP16-CANDIDATE",
    },
    "jang4m": {
        "name": "jang4m_audio",
        "served_name": "gemma4-12b-jang4m-audio-smoke",
        "path": "/Users/example/models/JANGQ-AI/gemma-4-12B-it-JANG_4M",
    },
}

_CHANNEL_LEAKS = ("<|channel>", "<channel|>", "<|start>", "<|end>")
_NEGATIVE_AUDIO_MARKERS = (
    "no audio",
    "not audio",
    "no sound",
    "no_audio",
    "not attached",
    "no attachment",
    "none",
    "pas de fichier audio",
    "pas d'audio",
    "aucun audio",
    "senza audio",
    "non c",
    "cannot hear",
    "cannot process audio",
)
_BAD_AUDIO_QUALITY_MARKERS = (
    "thought",
    "transcriptionno",
)


def make_tone_wav_b64(*, seconds: float = 0.8, sample_rate: int = 16000, hz: float = 440.0) -> str:
    samples = int(seconds * sample_rate)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(samples):
            amp = int(0.25 * 32767 * math.sin(2 * math.pi * hz * i / sample_rate))
            wf.writeframesraw(struct.pack("<h", amp))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_speech_audio_b64() -> tuple[str, str, str]:
    """Return (base64, format, expected phrase) for a local speech fixture."""
    tmp = None
    wav = None
    try:
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".aiff")
        os.close(fd)
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["say", "-o", tmp, "audio present"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", tmp, wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        data = Path(wav).read_bytes()
        return base64.b64encode(data).decode("ascii"), "wav", "audio present"
    except Exception:
        return make_tone_wav_b64(), "wav", "tone"
    finally:
        if tmp:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
        if wav:
            try:
                Path(wav).unlink()
            except OSError:
                pass


def request_json(method: str, url: str, body: Any | None = None, *, timeout: float = 180.0) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.perf_counter() - t0
            return resp.status, json.loads(raw.decode("utf-8")) if raw else None, elapsed
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0


def extract_content(resp: Any) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "")


def build_command(row: dict[str, str], port: int, out_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        row["served_name"],
        "--timeout",
        "240",
        "--max-num-seqs",
        "1",
        "--prefill-batch-size",
        "512",
        "--prefill-step-size",
        "1024",
        "--completion-batch-size",
        "256",
        "--no-continuous-batching",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-dir",
        str((out_dir / "block_cache").resolve()),
        "--block-disk-cache-max-gb",
        "2",
        "--kv-cache-quantization",
        "none",
        "--default-enable-thinking",
        "false",
        "--is-mllm",
        "--log-level",
        "INFO",
    ]


def terminate(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=10)


def wait_ready(base_url: str, proc: subprocess.Popen[Any], timeout_s: float) -> dict[str, Any]:
    end = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}")
        code, body, _ = request_json("GET", f"{base_url}/health", timeout=5.0)
        last = body
        if code == 200 and isinstance(body, dict) and body.get("model_loaded"):
            return body
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready: {last}")


def has_modality(body: Any, modality: str) -> bool:
    if not isinstance(body, dict):
        return False
    modalities = body.get("modalities")
    if isinstance(modalities, list) and modality in {str(x).lower() for x in modalities}:
        return True
    caps = body.get("capabilities") if isinstance(body.get("capabilities"), dict) else {}
    modalities = caps.get("modalities") if isinstance(caps.get("modalities"), list) else []
    return modality in {str(x).lower() for x in modalities}


def run_row(
    row_key: str,
    *,
    port: int,
    load_timeout: float,
    out_base: Path,
    expect_audio_gated: bool = False,
) -> dict[str, Any]:
    row = dict(ROWS[row_key])
    out_dir = out_base.with_name(out_base.name + "_artifacts") / row["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    cmd = build_command(row, port, out_dir)
    result: dict[str, Any] = {
        "row": row_key,
        "name": row["name"],
        "path": row["path"],
        "command": cmd,
        "log": str(log_path),
        "status": "starting",
        "failures": [],
        "expected_audio_gated": expect_audio_gated,
    }
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{port}"
    try:
        try:
            result["health"] = wait_ready(base_url, proc, load_timeout)
        except Exception as exc:
            result["status"] = "load_failed"
            result["failures"].append({"reason": "load_failed", "error": repr(exc)})
            return result

        code, capabilities, _ = request_json(
            "GET",
            f"{base_url}/v1/models/{urllib.parse.quote(row['served_name'], safe='')}/capabilities",
            timeout=30.0,
        )
        result["capabilities"] = {"code": code, "body": capabilities}
        if code != 200:
            result["failures"].append({"reason": "capabilities_http_status", "code": code})
        audio_advertised = has_modality(capabilities, "audio")
        result["audio_advertised"] = audio_advertised
        if not audio_advertised:
            result["failures"].append({"reason": "audio_not_advertised", "capabilities": capabilities})
        if expect_audio_gated and audio_advertised:
            result["failures"].append({"reason": "audio_unexpectedly_advertised", "capabilities": capabilities})
        if expect_audio_gated and not audio_advertised:
            code, cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
            result["cache_stats_final"] = {"code": code, "body": cache_stats}
            log_text = log_path.read_text(errors="replace")
            result["server_log_tail"] = log_text[-6000:]
            result["expected_audio_gate_observed"] = True
            result["unexpected_failures"] = []
            result["status"] = "pass"
            return result

        audio_b64, audio_format, expected_audio = make_speech_audio_b64()
        audio_payload = {
            "model": row["served_name"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe the attached audio. Reply with only the spoken words."
                            ),
                        },
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
            "enable_thinking": False,
        }
        code, audio_resp, audio_elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            audio_payload,
            timeout=240.0,
        )
        audio_content = extract_content(audio_resp)
        result["audio_request"] = {
            "code": code,
            "elapsed_sec": audio_elapsed,
            "content": audio_content,
            "body_head": str(audio_resp)[:1000],
            "expected_audio": expected_audio,
            "audio_format": audio_format,
        }
        if code != 200:
            result["failures"].append({"reason": "audio_http_status", "code": code, "body": audio_resp})
        if not audio_content.strip():
            result["failures"].append({"reason": "audio_empty_content"})
        if any(marker in audio_content for marker in _CHANNEL_LEAKS):
            result["failures"].append({"reason": "audio_channel_marker_leak", "content": audio_content[:300]})
        audio_lower = audio_content.lower()
        if any(marker in audio_lower for marker in _NEGATIVE_AUDIO_MARKERS):
            result["failures"].append({"reason": "audio_negative_answer", "content": audio_content[:300]})
        if any(marker in audio_lower for marker in _BAD_AUDIO_QUALITY_MARKERS):
            result["failures"].append({"reason": "audio_bad_quality_marker", "content": audio_content[:300]})
        if (
            expected_audio == "audio present"
            and ("audio" not in audio_lower or "present" not in audio_lower)
        ):
            result["failures"].append({"reason": "audio_transcription_missing", "expected": expected_audio, "content": audio_content[:300]})
        if expected_audio == "tone" and "tone" not in audio_lower and "sound" not in audio_lower:
            result["failures"].append({"reason": "audio_tone_not_acknowledged", "content": audio_content[:300]})

        no_audio_payload = {
            "model": row["served_name"],
            "messages": [
                {
                    "role": "system",
                    "content": "The current user message has no audio attachment. Answer exactly NONE.",
                },
                {"role": "user", "content": "Is there an audio attachment in this current message?"},
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "enable_thinking": False,
        }
        code, no_audio_resp, no_audio_elapsed = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            no_audio_payload,
            timeout=180.0,
        )
        no_audio_content = extract_content(no_audio_resp)
        result["no_audio_request"] = {
            "code": code,
            "elapsed_sec": no_audio_elapsed,
            "content": no_audio_content,
            "body_head": str(no_audio_resp)[:1000],
        }
        if code != 200:
            result["failures"].append({"reason": "no_audio_http_status", "code": code, "body": no_audio_resp})
        if "none" not in no_audio_content.lower() and "no" not in no_audio_content.lower():
            result["failures"].append({"reason": "no_audio_isolation_failed", "content": no_audio_content[:300]})

        code, cache_stats, _ = request_json("GET", f"{base_url}/v1/cache/stats", timeout=30.0)
        result["cache_stats_final"] = {"code": code, "body": cache_stats}
        log_text = log_path.read_text(errors="replace")
        result["server_log_tail"] = log_text[-6000:]
        if "audio=yes" not in log_text and "input_audio" not in log_text and "has_audio" not in log_text:
            result["failures"].append({"reason": "server_log_missing_audio_marker"})
        if expect_audio_gated and not audio_advertised:
            expected_gate_reasons = {
                "audio_not_advertised",
                "audio_http_status",
                "audio_empty_content",
                "audio_transcription_missing",
            }
            unexpected_failures = [
                failure
                for failure in result["failures"]
                if failure.get("reason") not in expected_gate_reasons
            ]
            result["expected_audio_gate_observed"] = True
            result["unexpected_failures"] = unexpected_failures
            result["status"] = "pass" if not unexpected_failures else "fail"
        else:
            result["status"] = "pass" if not result["failures"] else "fail"
        return result
    finally:
        terminate(proc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", choices=sorted(ROWS), default=sorted(ROWS))
    parser.add_argument("--port", type=int, default=8958)
    parser.add_argument("--load-timeout", type=float, default=420.0)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--expect-audio-gated", nargs="*", choices=sorted(ROWS), default=[])
    parser.add_argument(
        "--row-cooldown",
        type=float,
        default=5.0,
        help="Seconds to wait between row server teardowns and the next model load.",
    )
    parser.add_argument("--out", default="build/current-gemma4-12b-audio-smoke.json")
    args = parser.parse_args()

    out = Path(args.out)
    results = []
    for i, row in enumerate(args.rows):
        result = run_row(
            row,
            port=args.port + i,
            load_timeout=args.load_timeout,
            out_base=out,
            expect_audio_gated=row in set(args.expect_audio_gated),
        )
        results.append(result)
        if args.stop_on_failure and result["status"] != "pass":
            break
        if i + 1 < len(args.rows) and args.row_cooldown > 0:
            time.sleep(args.row_cooldown)
    summary = {
        "status": "pass" if all(r.get("status") == "pass" for r in results) and len(results) == len(args.rows) else "fail",
        "rows": args.rows,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print("status=" + summary["status"])
    for result in results:
        print(f"{result['name']}: status={result['status']} failures={result['failures']}")
        print(f"  audio_content={result.get('audio_request', {}).get('content')!r}")
        print(f"  no_audio_content={result.get('no_audio_request', {}).get('content')!r}")
        print(f"  log={result['log']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
