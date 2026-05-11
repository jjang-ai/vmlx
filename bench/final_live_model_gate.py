#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path('/Users/example/mlx/vllm-mlx')
PYTHON = ROOT / '.venv/bin/python'
OUT_BASE = ROOT / 'docs/internal/release-gates/20260510_final_live_model_gate'

BLUE_PNG = (
    'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAI0lEQVR4nGNkYPjPQApgIkk1w6g'
    'G4gATkergYFQDMYDkUAIAPjABH26QQDYAAAAASUVORK5CYII='
)

ROWS: dict[str, dict[str, Any]] = {
    'zaya_mxfp4': {
        'path': '/Users/example/models/JANGQ/ZAYA1-8B-MXFP4',
        'model': 'zaya-mxfp4',
        'is_mllm': False,
        'family': 'zaya',
        'expect_cache': 'zaya_cca',
        'tool_parser': 'zaya_xml',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'zaya_jangtq2': {
        'path': '/Users/example/models/JANGQ/ZAYA1-8B-JANGTQ2',
        'model': 'zaya-jangtq2',
        'is_mllm': False,
        'family': 'zaya',
        'expect_cache': 'zaya_cca',
        'tool_parser': 'zaya_xml',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'zaya_vl_mxfp4': {
        'path': '/Users/example/models/JANGQ/ZAYA1-VL-8B-MXFP4',
        'model': 'zaya-vl-mxfp4',
        'is_mllm': True,
        'family': 'zaya1_vl',
        'expect_cache': 'zaya_cca',
        'tool_parser': 'zaya_xml',
        'reasoning_parser': 'qwen3',
        'thinking': False,
        'vl': True,
    },
    'zaya_vl_jangtq2': {
        'path': '/Users/example/models/JANGQ/ZAYA1-VL-8B-JANGTQ2',
        'model': 'zaya-vl-jangtq2',
        'is_mllm': True,
        'family': 'zaya1_vl',
        'expect_cache': 'zaya_cca',
        'tool_parser': 'zaya_xml',
        'reasoning_parser': 'qwen3',
        'thinking': False,
        'vl': True,
        'video': True,
    },
    'minimax_m27': {
        'path': '/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ',
        'model': 'minimax',
        'is_mllm': False,
        'family': 'minimax',
        'expect_cache': 'tq_kv',
        'tool_parser': 'minimax',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'minimax_m27_k': {
        'path': '/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K',
        'model': 'minimax-k',
        'is_mllm': False,
        'family': 'minimax',
        'expect_cache': 'tq_kv',
        'tool_parser': 'minimax',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'qwen36_35b_moe': {
        'path': '/Users/example/models/Qwen3.6-35B-A3B-4bit',
        'model': 'qwen36-35b',
        'is_mllm': True,
        'family': 'qwen3_5_moe',
        'expect_cache': 'hybrid_ssm',
        'tool_parser': 'qwen',
        'reasoning_parser': 'qwen3',
        'thinking': False,
        'vl': True,
        'video': True,
    },
    'ling_tq': {
        'path': '/Users/example/models/JANGQ/Ling-2.6-flash-JANGTQ',
        'model': 'ling',
        'is_mllm': False,
        'family': 'ling',
        'expect_cache': 'hybrid_ssm',
        'tool_parser': 'deepseek',
        'reasoning_parser': 'deepseek_r1',
        'thinking': False,
    },
    'hy3_jangtq2': {
        'path': '/Users/example/models/JANGQ/Hy3-preview-JANGTQ2',
        'model': 'hy3',
        'is_mllm': False,
        'family': 'hy_v3',
        'expect_cache': 'tq_kv',
        'tool_parser': 'hunyuan',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'laguna_tq': {
        'path': '/Users/example/models/JANGQ/Laguna-XS.2-JANGTQ',
        'model': 'laguna',
        'is_mllm': False,
        'family': 'laguna',
        'expect_cache': 'tq_kv',
        'tool_parser': 'qwen',
        'reasoning_parser': 'qwen3',
        'thinking': False,
    },
    'kimi_small_tq': {
        'path': '/Users/example/models/JANGQ/Kimi-K2.6-Small-JANGTQ',
        'model': 'kimi-small',
        'is_mllm': False,
        'family': 'kimi_k25',
        'expect_cache': 'tq_kv',
        'tool_parser': 'kimi',
        'reasoning_parser': 'deepseek_r1',
        'thinking': False,
    },
    'gemma4_26b': {
        'path': '/Users/example/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK',
        'model': 'gemma4',
        'is_mllm': True,
        'family': 'gemma4',
        'expect_cache': 'vl',
        'tool_parser': 'gemma4',
        'reasoning_parser': 'gemma4',
        'thinking': False,
        'vl': True,
    },
    'nemotron_omni_mxfp4': {
        'path': '/Users/example/models/dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK',
        'model': 'nemotron-omni',
        'is_mllm': False,
        'family': 'nemotron_h',
        'expect_cache': 'hybrid_ssm',
        'tool_parser': 'nemotron',
        'reasoning_parser': 'deepseek_r1',
        'thinking': False,
    },
    'dsv4_mixed': {
        'path': '/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-V3-F32-MIXED',
        'model': 'dsv4',
        'is_mllm': False,
        'family': 'deepseek_v4',
        'expect_cache': 'deepseek_v4_composite',
        'tool_parser': 'dsml',
        'reasoning_parser': 'deepseek_r1',
        'thinking': False,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def bundle_defaults(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gen = read_json(path / 'generation_config.json')
    if isinstance(gen.get('temperature'), (int, float)):
        out['temperature'] = float(gen['temperature'])
    if isinstance(gen.get('top_p'), (int, float)):
        out['top_p'] = float(gen['top_p'])
    if isinstance(gen.get('repetition_penalty'), (int, float)):
        out['repetition_penalty'] = float(gen['repetition_penalty'])
    if isinstance(gen.get('max_new_tokens'), int) and gen['max_new_tokens'] > 0:
        out['max_tokens'] = int(gen['max_new_tokens'])

    jang = read_json(path / 'jang_config.json')
    sampling = ((jang.get('chat') or {}).get('sampling_defaults') or {}) if isinstance(jang, dict) else {}
    if isinstance(sampling, dict):
        if isinstance(sampling.get('temperature'), (int, float)):
            out['temperature'] = float(sampling['temperature'])
        if isinstance(sampling.get('top_p'), (int, float)):
            out['top_p'] = float(sampling['top_p'])
        rep = sampling.get('repetition_penalty_chat', sampling.get('repetition_penalty'))
        if isinstance(rep, (int, float)):
            out['repetition_penalty'] = float(rep)
        if isinstance(sampling.get('max_new_tokens'), int) and sampling['max_new_tokens'] > 0:
            out['max_tokens'] = int(sampling['max_new_tokens'])
    return out


def request_json(method: str, url: str, body: Any | None = None, timeout: int = 120) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            elapsed = time.perf_counter() - t0
            if not raw:
                return r.status, None, elapsed
            try:
                return r.status, json.loads(raw), elapsed
            except Exception:
                return r.status, raw.decode('utf-8', 'replace'), elapsed
    except Exception as e:
        return 0, {'error': f'{type(e).__name__}: {e}'}, time.perf_counter() - t0


def extract_text(resp: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(resp, dict):
        return '', '', {}
    usage = resp.get('usage') if isinstance(resp.get('usage'), dict) else {}
    choices = resp.get('choices') or []
    if choices:
        msg = choices[0].get('message') or {}
        return str(msg.get('content') or ''), str(msg.get('reasoning_content') or msg.get('reasoning') or ''), usage
    output = resp.get('output') or []
    chunks=[]
    for item in output if isinstance(output, list) else []:
        if isinstance(item, dict):
            for c in item.get('content') or []:
                if isinstance(c, dict) and c.get('text'):
                    chunks.append(str(c.get('text')))
    return ''.join(chunks), str(resp.get('reasoning') or ''), usage


def looks_like_generated_error(content: str) -> bool:
    lower = content.lower()
    return (
        lower.startswith('generation failed:')
        or lower.startswith('[generation error:')
        or 'traceback' in lower
        or 'runtimeerror:' in lower
    )


def normalize_reply_text(content: str) -> str:
    return " ".join(content.strip().lower().split()).rstrip(".! ")


def recall_has_expected_format(content: str) -> bool:
    import re
    lower = content.lower()
    return (
        re.search(r"\bcolor\s*=\s*blue\b", lower) is not None
        and re.search(r"\banimal\s*=\s*cat\b", lower) is not None
    )


def image_message(prompt: str) -> list[dict[str, Any]]:
    return [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + BLUE_PNG}},
        ],
    }]


def blue_video_data_url() -> str:
    import cv2
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()
    try:
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            2.0,
            (32, 32),
        )
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not open mp4v VideoWriter')
        for _ in range(8):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)  # BGR blue.
            writer.write(frame)
        writer.release()
        data = Path(tmp.name).read_bytes()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return 'data:video/mp4;base64,' + base64.b64encode(data).decode('ascii')


def video_message(prompt: str) -> list[dict[str, Any]]:
    return [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'video_url', 'video_url': {'url': blue_video_data_url()}},
        ],
    }]


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_row(row_id: str, row: dict[str, Any], port: int, out_base: Path, load_timeout: int, keep: bool) -> dict[str, Any]:
    out = out_base / row_id
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(row['path'])
    defaults = bundle_defaults(model_path)
    args = [
        str(PYTHON), '-B', '-s', '-m', 'vmlx_engine.cli', 'serve', str(model_path),
        '--host', '127.0.0.1', '--port', str(port), '--timeout', '300',
        '--max-num-seqs', '1', '--prefill-batch-size', '512', '--prefill-step-size', '2048', '--completion-batch-size', '512',
        '--continuous-batching', '--use-paged-cache', '--paged-cache-block-size', '64', '--max-cache-blocks', '1000',
        '--enable-block-disk-cache', '--block-disk-cache-max-gb', '10', '--stream-interval', '1',
        '--served-model-name', row['model'], '--log-level', 'INFO',
    ]
    if row.get('is_mllm'):
        args.append('--is-mllm')
    if row.get('tool_parser'):
        args += ['--tool-call-parser', row['tool_parser']]
    if row.get('reasoning_parser'):
        args += ['--reasoning-parser', row['reasoning_parser']]
    args += ['--default-enable-thinking', 'true' if row.get('thinking') else 'false']
    if defaults.get('temperature') is not None:
        args += ['--default-temperature', f"{defaults['temperature']:.2f}"]
    if defaults.get('top_p') is not None:
        args += ['--default-top-p', f"{defaults['top_p']:.2f}"]
    if defaults.get('repetition_penalty') is not None:
        args += ['--default-repetition-penalty', f"{defaults['repetition_penalty']:.2f}"]
    args += ['--max-tokens', str(min(int(defaults.get('max_tokens') or 32768), 32768))]

    env = dict(os.environ)
    env.pop('JANGTQ_TOPK_OVERRIDE', None)
    env['PYTHONPATH'] = str(ROOT)
    if row.get('expect_cache') == 'zaya_cca':
        env['VMLINUX_ZAYA_ENABLE_TYPED_CCA_CACHE'] = '1'
    log_path = out / 'server.log'
    result: dict[str, Any] = {
        'row_id': row_id,
        'row': row,
        'command': args,
        'defaults_from_bundle': defaults,
        'env_overrides': {k: env[k] for k in ('VMLINUX_ZAYA_ENABLE_TYPED_CCA_CACHE',) if k in env},
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'checks': [],
        'requests': [],
    }
    write(out / 'start.json', result)
    with log_path.open('w') as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    base = f'http://127.0.0.1:{port}'
    loaded = False
    for _ in range(load_timeout):
        if proc.poll() is not None:
            result['status'] = 'FAIL'
            result['reason'] = f'exited during load rc={proc.returncode}'
            result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
            write(out / 'SUMMARY.json', result)
            return result
        code, health, _ = request_json('GET', base + '/health', timeout=2)
        if code == 200 and isinstance(health, dict):
            loaded = True
            result['health_initial'] = health
            write(out / 'health_initial.json', health)
            break
        time.sleep(1)
    if not loaded:
        result['status'] = 'FAIL'
        result['reason'] = 'load timeout'
        result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
        proc.terminate()
        write(out / 'SUMMARY.json', result)
        return result

    def check(name: str, ok: bool, detail: Any) -> None:
        result['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})

    code, caps, _ = request_json('GET', base + '/v1/models/' + urllib.parse.quote(row['model'], safe='') + '/capabilities', timeout=20)
    write(out / 'capabilities.json', {'code': code, 'body': caps})
    native = (((caps or {}).get('cache') or {}).get('native') or {}) if isinstance(caps, dict) else {}
    check('capabilities_available', code == 200, caps)
    check('expected_family_visible', row.get('family') in json.dumps(caps).lower() or row.get('family') in json.dumps(result.get('health_initial', {})).lower(), {'family': row.get('family'), 'caps': caps, 'health': result.get('health_initial')})

    code, stats0, _ = request_json('GET', base + '/v1/cache/stats', timeout=20)
    write(out / 'cache_initial.json', {'code': code, 'body': stats0})
    check('cache_stats_available', code == 200, stats0)

    history = [{'role': 'user', 'content': 'Remember these facts: color blue, animal cat. Reply exactly: noted.'}]
    if row.get('vl'):
        probes = [
            ('vl_blue_image', image_message('This is a tiny solid-color image. What color is it? Reply with one color word.')),
        ]
        if row.get('video'):
            probes.append(('vl_blue_video', video_message('This is a short solid-color video. What color are the frames? Reply with one color word.')))
        probes.append(('turn1_memory', history))
    else:
        probes = [('turn1_memory', history)]
    for name, messages in probes:
        body = {'model': row['model'], 'messages': messages, 'max_tokens': 96, 'temperature': 0.0, 'stream': False, 'enable_thinking': row.get('thinking', False)}
        if name == 'vl_blue_video':
            body['video_fps'] = 2
            body['video_max_frames'] = 4
        code, resp, elapsed = request_json('POST', base + '/v1/chat/completions', body, timeout=180)
        content, reasoning, usage = extract_text(resp)
        write(out / f'{name}.json', {'request': body, 'code': code, 'elapsed_sec': elapsed, 'response': resp, 'content': content, 'reasoning': reasoning, 'usage': usage})
        completion_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        rough_tps = (float(completion_tokens) / elapsed) if completion_tokens and elapsed > 0 else None
        result['requests'].append({'name': name, 'code': code, 'elapsed_sec': elapsed, 'completion_tps_rough': rough_tps, 'content_chars': len(content), 'reasoning_chars': len(reasoning), 'usage': usage, 'head': content[:200], 'tail': content[-200:]})
        check(name + '_http_ok', code == 200, {'content': content[:300], 'reasoning': reasoning[:120], 'usage': usage})
        check(name + '_visible_nonempty', bool(content.strip()) and not looks_like_generated_error(content), {'content': content[:300], 'reasoning_chars': len(reasoning)})
        if name == 'vl_blue_image':
            check(
                'vl_blue_image_exact',
                normalize_reply_text(content) == 'blue',
                {'content': content[:300]},
            )
        if name == 'vl_blue_video':
            check(
                'vl_blue_video_exact',
                normalize_reply_text(content) == 'blue',
                {'content': content[:300]},
            )
        if name == 'turn1_memory' and code == 200:
            check(
                'turn1_exact_noted',
                normalize_reply_text(content) == 'noted',
                {'content': content[:300]},
            )
            history.append({'role': 'assistant', 'content': content or 'noted'})

    history.append({'role': 'user', 'content': 'What color and animal did I ask you to remember? Reply in the format color=..., animal=... .'})
    body = {'model': row['model'], 'messages': history, 'max_tokens': 96, 'temperature': 0.0, 'stream': False, 'enable_thinking': row.get('thinking', False)}
    code, resp, elapsed = request_json('POST', base + '/v1/chat/completions', body, timeout=180)
    content, reasoning, usage = extract_text(resp)
    write(out / 'turn2_recall.json', {'request': body, 'code': code, 'elapsed_sec': elapsed, 'response': resp, 'content': content, 'reasoning': reasoning, 'usage': usage})
    completion_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
    rough_tps = (float(completion_tokens) / elapsed) if completion_tokens and elapsed > 0 else None
    result['requests'].append({'name': 'turn2_recall', 'code': code, 'elapsed_sec': elapsed, 'completion_tps_rough': rough_tps, 'content_chars': len(content), 'reasoning_chars': len(reasoning), 'usage': usage, 'head': content[:200], 'tail': content[-200:]})
    lower = content.lower()
    check('turn2_recall_http_ok', code == 200, {'content': content[:300], 'reasoning': reasoning[:120], 'usage': usage})
    check('turn2_recall_contains_facts', 'blue' in lower and 'cat' in lower and not looks_like_generated_error(content), {'content': content, 'reasoning': reasoning[:200]})
    check('turn2_recall_exact_format', recall_has_expected_format(content), {'content': content, 'reasoning': reasoning[:200]})

    code, stats1, _ = request_json('GET', base + '/v1/cache/stats', timeout=20)
    write(out / 'cache_final.json', {'code': code, 'body': stats1})
    check('cache_stats_final_available', code == 200, stats1)
    stats_text = json.dumps(stats1).lower()
    expect = row.get('expect_cache')
    if expect == 'zaya_cca':
        check('zaya_typed_cache_visible', 'zaya_cca' in stats_text or 'zaya_cca' in json.dumps(native).lower(), {'native': native, 'stats': stats1})
        check('generic_tqkv_not_used_for_zaya', 'turboquant_kv_cache' not in stats_text or not (((stats1 or {}).get('turboquant_kv_cache') or {}).get('enabled')), stats1)
    elif expect == 'hybrid_ssm':
        check('hybrid_ssm_visible', 'hybrid_ssm' in stats_text or 'hybrid_ssm' in json.dumps(native).lower(), {'native': native, 'stats': stats1})
    elif expect == 'deepseek_v4_composite':
        check('dsv4_composite_visible', 'deepseek_v4' in stats_text or 'deepseek_v4' in json.dumps(native).lower(), {'native': native, 'stats': stats1})
    elif expect == 'tq_kv':
        tq = ((stats1 or {}).get('turboquant_kv_cache') or {}) if isinstance(stats1, dict) else {}
        check('turboquant_kv_visible', bool(tq.get('enabled')) or 'turboquant' in stats_text, {'tq': tq, 'stats': stats1})
        check('single_active_visible', 'single_active' in stats_text or 'single_active' in json.dumps(result.get('health_initial', {})).lower(), {'stats': stats1, 'health': result.get('health_initial')})

    failures = [c for c in result['checks'] if not c['ok']]
    result['status'] = 'PASS' if not failures else 'FAIL'
    result['failures'] = failures
    result['log_tail'] = log_path.read_text(errors='ignore')[-6000:]
    write(out / 'SUMMARY.json', result)
    lines = [f"# {row_id} live gate", '', f"status: {result['status']}", '', '## Requests']
    for req in result['requests']:
        lines.append(f"- {req['name']}: code={req['code']} elapsed={req['elapsed_sec']:.2f}s chars={req['content_chars']} usage={req.get('usage')} head={req.get('head')!r}")
    lines += ['', '## Failures']
    if failures:
        lines += [f"- {f['name']}: {f['detail']}" for f in failures]
    else:
        lines.append('- none')
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    if keep:
        result['kept_running_pid'] = proc.pid
    else:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--port', type=int, default=8910)
    ap.add_argument('--out', default=str(OUT_BASE))
    ap.add_argument('--load-timeout', type=int, default=600)
    ap.add_argument('--keep-running', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_results=[]
    for i, row_id in enumerate([r.strip() for r in args.rows.split(',') if r.strip()]):
        if row_id not in ROWS:
            raise SystemExit(f'unknown row {row_id}; known={sorted(ROWS)}')
        res = run_row(row_id, ROWS[row_id], args.port + i, out, args.load_timeout, args.keep_running)
        all_results.append(res)
        print(row_id, res.get('status'), 'failures=', len(res.get('failures') or []), flush=True)
    write(out / 'SUMMARY.json', {'rows': all_results, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})

if __name__ == '__main__':
    main()
