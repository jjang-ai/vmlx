#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'
sys.path.insert(0, str(ROOT))

from bench.final_live_model_gate import image_message, video_message, request_json, extract_text  # noqa: E402


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def make_audio_b64(out: Path) -> tuple[str | None, str | None]:
    aiff = out / 'blue.aiff'
    wav = out / 'blue.wav'
    try:
        subprocess.run(['say', '-o', str(aiff), 'blue'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', str(aiff), str(wav)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return base64.b64encode(wav.read_bytes()).decode('ascii'), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='/Users/example/models/dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK')
    ap.add_argument('--port', type=int, default=8940)
    ap.add_argument('--out', default='docs/internal/release-gates/20260510_nemotron_omni_media_live_gate')
    ap.add_argument('--load-timeout', type=int, default=900)
    ap.add_argument('--request-timeout', type=int, default=900)
    ap.add_argument('--keep-running', action='store_true')
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    base = f'http://127.0.0.1:{args.port}'
    cmd = [
        str(PYTHON), '-B', '-s', '-m', 'vmlx_engine.cli', 'serve', args.model_path,
        '--host', '127.0.0.1', '--port', str(args.port), '--timeout', '300',
        '--max-num-seqs', '1', '--prefill-batch-size', '512', '--prefill-step-size', '2048', '--completion-batch-size', '512',
        '--continuous-batching', '--use-paged-cache', '--paged-cache-block-size', '64', '--max-cache-blocks', '1000',
        '--enable-block-disk-cache', '--block-disk-cache-max-gb', '10', '--stream-interval', '1',
        '--served-model-name', 'nemotron-omni', '--log-level', 'INFO',
        '--tool-call-parser', 'nemotron', '--reasoning-parser', 'deepseek_r1', '--default-enable-thinking', 'false',
        '--max-tokens', '32768', '--default-temperature', '0.60', '--default-top-p', '0.95', '--default-repetition-penalty', '1.10',
    ]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    log_path = out / 'server.log'
    write(out / 'start.json', {'command': cmd, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    with log_path.open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)

    result: dict[str, Any] = {'requests': [], 'checks': []}
    def check(name: str, ok: bool, detail: Any) -> None:
        result['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})

    try:
        for _ in range(args.load_timeout):
            if proc.poll() is not None:
                result['status'] = 'FAIL'
                result['reason'] = f'exited during load rc={proc.returncode}'
                result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
                write(out / 'SUMMARY.json', result)
                return 1
            code, health, _ = request_json('GET', base + '/health', timeout=2)
            if code == 200 and isinstance(health, dict):
                result['health_initial'] = health
                write(out / 'health_initial.json', health)
                break
            time.sleep(1)
        else:
            result['status'] = 'FAIL'
            result['reason'] = 'load timeout'
            result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
            write(out / 'SUMMARY.json', result)
            return 1

        code, caps, _ = request_json('GET', base + '/v1/models/nemotron-omni/capabilities', timeout=30)
        write(out / 'capabilities.json', {'code': code, 'body': caps})
        check('capabilities_available', code == 200, caps)
        code, stats0, _ = request_json('GET', base + '/v1/cache/stats', timeout=30)
        write(out / 'cache_initial.json', {'code': code, 'body': stats0})
        check('cache_stats_initial_available', code == 200, stats0)

        probes: list[tuple[str, list[dict[str, Any]], int]] = []
        turn1 = [{'role': 'user', 'content': 'Remember these facts: color blue, animal cat. Reply exactly: noted.'}]
        probes.append(('turn1_memory', turn1, 96))
        probes.append(('image_blue', image_message('This is a tiny solid-color image. What color is it? Reply with one color word.'), 96))
        probes.append(('video_blue', video_message('This is a short solid-color video. What color are the frames? Reply with one color word.'), 96))
        audio_b64, audio_error = make_audio_b64(out)
        if audio_b64:
            probes.append((
                'audio_blue',
                [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'The audio says one color word. Transcribe the word only.'},
                        {'type': 'input_audio', 'input_audio': {'data': audio_b64, 'format': 'wav'}},
                    ],
                }],
                96,
            ))
        else:
            check('audio_fixture_created', False, audio_error)

        turn1_content = ''
        for name, messages, max_tokens in probes:
            body = {'model': 'nemotron-omni', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0.0, 'stream': False, 'enable_thinking': False}
            if name == 'video_blue':
                body['video_fps'] = 2
                body['video_max_frames'] = 4
            t0 = time.perf_counter()
            code, resp, elapsed = request_json('POST', base + '/v1/chat/completions', body, timeout=args.request_timeout)
            content, reasoning, usage = extract_text(resp)
            elapsed = elapsed or (time.perf_counter() - t0)
            write(out / f'{name}.json', {'request': body, 'code': code, 'elapsed_sec': elapsed, 'response': resp, 'content': content, 'reasoning': reasoning, 'usage': usage})
            result['requests'].append({'name': name, 'code': code, 'elapsed_sec': elapsed, 'content': content, 'reasoning': reasoning, 'usage': usage, 'head': content[:300], 'tail': content[-300:]})
            check(name + '_http_ok', code == 200, {'content': content, 'reasoning': reasoning, 'usage': usage})
            if name == 'turn1_memory':
                turn1_content = content
                turn2_messages = [
                    turn1[0],
                    {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': 'Reply exactly: color=blue, animal=cat'},
                ]
                probes.append(('turn2_recall', turn2_messages, 96))
            if name in {'image_blue', 'video_blue', 'audio_blue'}:
                check(name + '_mentions_blue', 'blue' in (content or '').lower(), content)
            if name == 'turn1_memory':
                check('turn1_noted_exact_or_contains', 'noted' in (content or '').lower(), content)
            if name == 'turn2_recall':
                low = (content or '').lower()
                check('turn2_recall_color_animal', 'blue' in low and 'cat' in low, content)

        code, stats1, _ = request_json('GET', base + '/v1/cache/stats', timeout=30)
        write(out / 'cache_final.json', {'code': code, 'body': stats1})
        check('cache_stats_final_available', code == 200, stats1)
        result['cache_final'] = stats1
        result['status'] = 'PASS' if all(c['ok'] for c in result['checks']) else 'FAIL'
        result['passed'] = result['status'] == 'PASS'
        result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
        write(out / 'SUMMARY.json', result)
        print(f"{result['status']} checks={sum(1 for c in result['checks'] if c['ok'])}/{len(result['checks'])} out={out}")
        for req in result['requests']:
            print(req['name'], req['code'], repr(req['content'][:240]))
        return 0 if result['passed'] else 2
    finally:
        if not args.keep_running:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    raise SystemExit(main())
