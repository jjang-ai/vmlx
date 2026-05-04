#!/usr/bin/env python3
"""Production-scale vMLX family audit runner.

This is intentionally explicit and conservative. It has two phases:

1. Static bundle audit: config/generation/jang metadata, quant bit maps,
   special EOS/template fields, cache-risk flags, and safetensor shard summary.
2. Optional live audit: spawn one model at a time, run the same small matrix
   across Chat Completions, Responses, Anthropic Messages, Ollama chat,
   reasoning on/off, tool-history continuation shape, and cache stats.

The live phase is opt-in with --live because these models are huge. Missing
models are reported as SKIP, never silently passed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path("/Users/eric/mlx/vllm-mlx")
DEFAULT_PY = Path(
    "/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3"
)
OUT_DIR = Path("/tmp/vmlx_family_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelRow:
    id: str
    label: str
    path: str
    family: str
    kind: str = "llm"  # llm | vl | omni | audio
    slow: bool = False
    expect_reasoning: bool = False
    expect_tool_parser: str | None = None
    cache_profile: str = "default"  # default | dsv4_composite | vl | hybrid_ssm
    live_supported: bool = True
    unsupported_reason: str | None = None
    notes: list[str] = field(default_factory=list)


ROWS: list[ModelRow] = [
    ModelRow(
        id="dsv4_tq",
        label="DeepSeek-V4-Flash JANGTQ",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "DSV4 SWA+CSA/HSA heterogenous cache; paged/block L2 must use "
            "deepseek_v4_v6 composite-state serialization, not generic KV blocks"
        ],
    ),
    ModelRow(
        id="dsv4_tq2",
        label="DeepSeek-V4-Flash JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ2",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        slow=True,
        notes=[
            "DSV4 canonical encoder + EOS [end,user,assistant] required; "
            "DeepseekV4Cache paged/L2 v4 schema must be active"
        ],
    ),
    ModelRow(
        id="dsv4_2l",
        label="DeepSeek-V4-Flash JANG_2L",
        path="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANG_2L",
        family="deepseek_v4",
        expect_reasoning=True,
        expect_tool_parser="dsml",
        cache_profile="dsv4_composite",
        live_supported=False,
        unsupported_reason=(
            "Local bundle is corrupt: live load fails with safetensors invalid "
            "data offsets in layers.28.ffn.experts.197.w3.biases. Re-download "
            "or regenerate before treating this row as a runtime target."
        ),
        slow=True,
    ),
    ModelRow(
        id="qwen36_moe_tq4",
        label="Qwen3.6-35B-A3B JANGTQ4 hybrid MoE+SSM",
        path="/Volumes/EricsLLMDrive/jangq-ai/Qwen3.6-35B-A3B-JANGTQ4",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_moe_crack",
        label="Qwen3.6-35B-A3B JANGTQ CRACK hybrid MoE+SSM",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-35B-A3B-JANGTQ-CRACK",
        family="qwen3_5_moe",
        expect_reasoning=True,
        expect_tool_parser="qwen",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="qwen36_dense_mxfp4",
        label="Qwen3.6-27B MXFP4 CRACK dense",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-MXFP4-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="qwen36_dense_jang",
        label="Qwen3.6-27B JANG_4M CRACK dense",
        path="/Volumes/EricsLLMDrive/dealignai/Qwen3.6-27B-JANG_4M-CRACK",
        family="qwen3_5",
        expect_reasoning=True,
        expect_tool_parser="qwen",
    ),
    ModelRow(
        id="nemotron_omni_tq2",
        label="Nemotron-3-Nano-Omni-30B-A3B JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2",
        family="nemotron_omni",
        kind="omni",
        expect_reasoning=True,
        expect_tool_parser="nemotron",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="nemotron_omni_tq4",
        label="Nemotron-3-Nano-Omni-30B-A3B JANGTQ4",
        path="/Volumes/EricsLLMDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4",
        family="nemotron_omni",
        kind="omni",
        expect_reasoning=True,
        expect_tool_parser="nemotron",
        cache_profile="hybrid_ssm",
        slow=True,
    ),
    ModelRow(
        id="laguna_tq",
        label="Laguna-XS.2 JANGTQ",
        path="/Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Laguna-XS.2-JANGTQ",
        family="laguna",
        expect_tool_parser=None,
        slow=True,
    ),
    ModelRow(
        id="mistral_medium35_tq",
        label="Mistral-Medium-3.5-128B JANGTQ2",
        path="/Volumes/EricsLLMDrive/jangq-ai/JANGQ-AI/Mistral-Medium-3.5-128B-JANGTQ",
        family="mistral3",
        kind="vl",
        expect_reasoning=False,
        expect_tool_parser="mistral",
        cache_profile="vl",
        live_supported=False,
        unsupported_reason=(
            "Known non-production bundle: 2-bit dense JANGTQ Mistral 3.5 "
            "loads structurally but full prefill is decode-kernel bound and "
            "text quality degenerates. Use Mistral-Medium-3.5-128B-mxfp4 "
            "until a JANGTQ4 bundle/runtime is validated."
        ),
        slow=True,
        notes=[
            "Structural-only regression row. Do not treat as a coherent production target.",
            "Pixtral-style multimodal Mistral path; vision passthrough fp16",
        ],
    ),
    ModelRow(
        id="mistral_medium35_mxfp4",
        label="Mistral-Medium-3.5-128B MXFP4",
        path="/Volumes/EricsLLMDrive/jangq-ai/OsaurusAI/Mistral-Medium-3.5-128B-mxfp4",
        family="mistral3",
        kind="vl",
        expect_reasoning=False,
        expect_tool_parser="mistral",
        cache_profile="vl",
        slow=True,
        notes=[
            "Production Mistral 3.5 target. Direct runtime verified: "
            "368-token prefill ~7s, first decode ~0.7s, coherent 'ok'.",
            "Pixtral vision metadata present; Python runtime is currently text-first.",
        ],
    ),
    ModelRow(
        id="mistral_small4_crack",
        label="Mistral-Small-4-119B JANG_6M CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/Mistral-Small-4-119B-JANG_6M-CRACK",
        family="mistral",
        expect_tool_parser="mistral",
        slow=True,
    ),
    ModelRow(
        id="gemma4_crack",
        label="Gemma-4-26B-A4B-it JANG_4M CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/Gemma-4-26B-A4B-it-JANG_4M-CRACK",
        family="gemma4",
        kind="vl",
        expect_reasoning=True,
        expect_tool_parser="gemma4",
        cache_profile="vl",
        slow=True,
    ),
    ModelRow(
        id="minimax_m27_tq_crack",
        label="MiniMax-M2.7 JANGTQ CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/MiniMax-M2.7-JANGTQ-CRACK",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        slow=True,
        notes=[
            "Pure-attention large MoE; validates long prompt KV/paged/L2 behavior "
            "separately from hybrid SSM rows."
        ],
    ),
    ModelRow(
        id="minimax_m27_tq4_crack",
        label="MiniMax-M2.7 JANGTQ4 CRACK",
        path="/Volumes/EricsLLMDrive/dealignai/MiniMax-M2.7-JANGTQ4-CRACK",
        family="minimax",
        expect_reasoning=True,
        expect_tool_parser="minimax",
        live_supported=False,
        unsupported_reason=(
            "Local 4-bit MiniMax bundle loads at ~117GB model working set and "
            "trips the Metal working-set guard on this 128GB Mac "
            "(live 2026-05-03: active 110.4GB, 103% of 107.5GB cap). "
            "Use the validated 2-bit JANGTQ row or a larger unified-memory host."
        ),
        slow=True,
        notes=[
            "Higher-bit MiniMax runtime target; keep in the matrix so static "
            "bit/layer/template drift is visible even when live pass is skipped.",
            "Live load also showed quant_shape_inference patched 250 mixed-precision "
            "modules where config.json claimed uniform 4-bit.",
        ],
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}
    return {}


def summarize_safetensors(model_dir: Path, max_files: int = 9999) -> dict[str, Any]:
    files = sorted(model_dir.glob("*.safetensors"))
    total_gb = sum(p.stat().st_size for p in files) / (1024**3)
    summary: dict[str, Any] = {
        "files": len(files),
        "total_gb": round(total_gb, 2),
        "layer_ids_seen": [],
        "sample_keys": [],
        "key_prefix_counts": {},
    }
    if not files:
        return summary
    try:
        from safetensors import safe_open
    except Exception as exc:
        summary["safetensors_import_error"] = str(exc)
        return summary

    layer_ids: set[int] = set()
    prefixes: dict[str, int] = {}
    sample: list[str] = []
    for p in files[:max_files]:
        try:
            with safe_open(str(p), framework="np") as f:
                for key in f.keys():
                    if len(sample) < 30:
                        sample.append(key)
                    head = key.split(".", 2)[0]
                    if head == "layers":
                        parts = key.split(".", 3)
                        if len(parts) > 1 and parts[1].isdigit():
                            layer_ids.add(int(parts[1]))
                            head = ".".join(parts[:3]) if len(parts) > 2 else "layers"
                    prefixes[head] = prefixes.get(head, 0) + 1
        except Exception as exc:
            summary.setdefault("file_errors", []).append(
                {"file": p.name, "error": f"{type(exc).__name__}: {exc}"}
            )
    summary["layer_ids_seen"] = [min(layer_ids), max(layer_ids), len(layer_ids)] if layer_ids else []
    summary["sample_keys"] = sample
    summary["key_prefix_counts"] = dict(sorted(prefixes.items(), key=lambda kv: (-kv[1], kv[0]))[:40])
    return summary


def static_audit(row: ModelRow) -> dict[str, Any]:
    model_dir = Path(row.path)
    cfg = read_json(model_dir / "config.json")
    gen = read_json(model_dir / "generation_config.json")
    jang = read_json(model_dir / "jang_config.json")
    tok_cfg = read_json(model_dir / "tokenizer_config.json")
    chat_template_file = model_dir / "chat_template.jinja"
    registry: dict[str, Any] = {}
    try:
        sys.path.insert(0, str(ROOT))
        from vmlx_engine.model_config_registry import get_model_config_registry

        rcfg = get_model_config_registry().lookup(str(model_dir))
        registry = {
            "family_name": rcfg.family_name,
            "reasoning_parser": rcfg.reasoning_parser,
            "tool_parser": rcfg.tool_parser,
            "cache_type": rcfg.cache_type,
            "think_in_template": rcfg.think_in_template,
            "eos_tokens": rcfg.eos_tokens,
        }
    except Exception as exc:
        registry = {"lookup_error": f"{type(exc).__name__}: {exc}"}

    layer_types = cfg.get("layer_types") or cfg.get("layer_type") or []
    if isinstance(layer_types, list):
        layer_type_counts = {x: layer_types.count(x) for x in sorted(set(map(str, layer_types)))}
    else:
        layer_type_counts = {}

    chat = jang.get("chat") if isinstance(jang.get("chat"), dict) else {}
    reasoning = chat.get("reasoning") if isinstance(chat.get("reasoning"), dict) else {}
    sampling = chat.get("sampling_defaults") if isinstance(chat.get("sampling_defaults"), dict) else {}
    safetensors_summary = summarize_safetensors(model_dir)

    eos = {
        "config": cfg.get("eos_token_id"),
        "generation": gen.get("eos_token_id"),
        "tokenizer_eos": tok_cfg.get("eos_token"),
        "tokenizer_has_template": bool(tok_cfg.get("chat_template")),
        "chat_template_file": chat_template_file.is_file(),
        "registry_eos_tokens": registry.get("eos_tokens"),
    }

    issues: list[str] = []
    if model_dir.is_dir() and safetensors_summary.get("files", 0) == 0:
        issues.append("bundle has no safetensors weights; structural metadata only")
    if not row.live_supported and row.unsupported_reason:
        issues.append(row.unsupported_reason)
    if row.family == "deepseek_v4":
        if jang.get("weight_format") == "mxtq":
            bits = jang.get("mxtq_bits", {})
            if bits.get("routed_expert") not in (2, 4, 8):
                issues.append("DSV4 routed_expert mxtq_bits missing/unexpected")
            if sampling.get("repetition_penalty_thinking") is None:
                issues.append("DSV4 missing chat.sampling_defaults.repetition_penalty_thinking")
        if "<｜Assistant｜>" not in str(registry.get("eos_tokens")):
            issues.append("DSV4 bundle EOS config does not list Assistant marker; engine registry must add it")
        if not tok_cfg.get("chat_template") and not chat_template_file.is_file():
            issues.append("DSV4 missing chat template file/config; canonical encoder shim still required")
    has_reasoning_surface = bool(
        reasoning.get("supported")
        or tok_cfg.get("chat_template")
        or chat_template_file.is_file()
        or registry.get("reasoning_parser")
    )
    if row.expect_reasoning and not has_reasoning_surface:
        issues.append("expected reasoning model but no reasoning metadata/template found")
    if row.kind in ("vl", "omni") and not any(
        k in (str(cfg) + str(jang) + str(tok_cfg)).lower()
        for k in ("vision", "image", "audio", "video", "mm", "omni", "radio", "parakeet")
    ):
        issues.append("multimodal row did not expose obvious multimodal metadata; loader detection must be live-tested")

    return {
        "id": row.id,
        "label": row.label,
        "path": row.path,
        "exists": model_dir.is_dir(),
        "family_expected": row.family,
        "kind": row.kind,
        "config": {
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "num_key_value_heads": cfg.get("num_key_value_heads"),
            "sliding_window": cfg.get("sliding_window"),
            "window_size": cfg.get("window_size"),
            "kv_lora_rank": cfg.get("kv_lora_rank"),
            "layer_type_counts": layer_type_counts,
            "quantization": cfg.get("quantization") or cfg.get("quantization_config"),
        },
        "generation": {
            "eos_token_id": gen.get("eos_token_id"),
            "pad_token_id": gen.get("pad_token_id"),
            "max_new_tokens": gen.get("max_new_tokens"),
            "temperature": gen.get("temperature"),
            "top_p": gen.get("top_p"),
        },
        "jang": {
            "weight_format": jang.get("weight_format"),
            "mxtq_bits": jang.get("mxtq_bits"),
            "quantization": jang.get("quantization"),
            "sampling_defaults": sampling,
            "reasoning": reasoning,
            "runtime": jang.get("runtime"),
        },
        "eos": eos,
        "registry": registry,
        "safetensors": safetensors_summary,
        "issues": issues,
        "live_supported": row.live_supported,
        "unsupported_reason": row.unsupported_reason,
        "notes": row.notes,
    }


def http_json(method: str, url: str, body: Any | None = None, timeout: int = 240) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            if not raw:
                return r.status, None
            return r.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def extract_chat_text(resp: Any) -> tuple[str, str, str]:
    try:
        ch = resp["choices"][0]
        msg = ch.get("message") or {}
        return (
            msg.get("content") or "",
            msg.get("reasoning_content") or "",
            ch.get("finish_reason") or "",
        )
    except Exception:
        return "", "", ""


def extract_responses_text(resp: Any) -> str:
    if isinstance(resp, dict):
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"]
        out = resp.get("output")
        if isinstance(out, list):
            texts: list[str] = []
            for item in out:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            texts.append(c.get("text", ""))
            return "\n".join(texts)
    return ""


def extract_responses_function_calls(resp: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(resp, dict):
        return calls
    out = resp.get("output")
    if not isinstance(out, list):
        return calls
    for item in out:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def has_duplicate_block(text: str, min_len: int = 80) -> bool:
    compact = re.sub(r"\s+", " ", text)
    if len(compact) < min_len * 2:
        return False
    for size in (240, 160, 100, 80):
        if len(compact) < size * 2:
            continue
        seen: set[str] = set()
        for i in range(0, len(compact) - size, max(16, size // 4)):
            block = compact[i : i + size]
            if block in seen:
                return True
            seen.add(block)
    return False


def post_chat(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None,
    max_tokens: int = 180,
    temperature: float = 0.6,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": False,
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    code, resp = http_json("POST", f"{base}/v1/chat/completions", body)
    return {"code": code, "body": resp, "request": body}


def live_audit(row: ModelRow, py: Path, port: int, timeout_load: int, keep_running: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": row.id,
        "label": row.label,
        "path": row.path,
        "status": "UNKNOWN",
        "checks": [],
    }
    model_dir = Path(row.path)
    if not model_dir.is_dir():
        result["status"] = "SKIP"
        result["reason"] = "path missing"
        return result
    if not row.live_supported:
        result["status"] = "SKIP"
        result["reason"] = row.unsupported_reason or "row marked live unsupported"
        return result

    log = OUT_DIR / f"{row.id}_{int(time.time())}.log"
    cmd = [
        str(py),
        "-s",
        "-m",
        "vmlx_engine.cli",
        "serve",
        row.path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        "300",
        "--continuous-batching",
        "--max-num-seqs",
        "5",
        "--prefill-batch-size",
        "1024",
        "--prefill-step-size",
        "2048",
        "--completion-batch-size",
        "1024",
        "--cache-memory-percent",
        "0.2",
        "--stream-interval",
        "1",
        "--max-tokens",
        "32768",
        "--default-temperature",
        "0.6",
        "--default-top-p",
        "0.95",
        "--default-repetition-penalty",
        "1.10",
        "--enable-prefix-cache",
        "--use-paged-cache",
        "--paged-cache-block-size",
        "64",
        "--max-cache-blocks",
        "1000",
        "--enable-block-disk-cache",
        "--block-disk-cache-max-gb",
        "10",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--enable-auto-tool-choice",
    ]

    subprocess.run(["pkill", "-9", "-f", "vmlx_engine.cli"], capture_output=True)
    time.sleep(2)
    with log.open("w") as lf:
        lf.write("# " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    result["pid"] = proc.pid
    result["log"] = str(log)

    base = f"http://127.0.0.1:{port}"
    loaded = False
    for _ in range(timeout_load):
        if proc.poll() is not None:
            result["status"] = "FAIL"
            result["reason"] = f"process exited during load: {proc.returncode}"
            result["log_tail"] = log.read_text(errors="ignore")[-4000:]
            return result
        code, health = http_json("GET", f"{base}/health", timeout=2)
        if code == 200 and isinstance(health, dict):
            loaded = True
            result["health"] = health
            break
        time.sleep(1)
    if not loaded:
        result["status"] = "FAIL"
        result["reason"] = "load timeout"
        result["log_tail"] = log.read_text(errors="ignore")[-4000:]
        proc.terminate()
        return result

    def check(name: str, ok: bool, detail: Any) -> None:
        result["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    log_text = log.read_text(errors="ignore")
    if row.family == "deepseek_v4":
        check(
            "dsv4_paged_cache_composite_enabled",
            "DeepseekV4Cache-aware paged prefix cache enabled" in log_text
            and "deepseek_v4_v6" in log_text,
            "DSV4 must keep paged/L2 cache on and use the deepseek_v4 v4 composite-state schema",
        )
        check(
            "dsv4_canonical_encoder_shim",
            "DSV4 chat-template shim installed" in log_text,
            "tokenizer.apply_chat_template must route through encoding_dsv4",
        )
        check(
            "dsv4_multi_eos",
            "128804" in log_text and "128803" in log_text,
            "expected EOS includes end/user/assistant markers",
        )

    code, stats0 = http_json("GET", f"{base}/v1/cache/stats", timeout=20)
    check("cache_stats_available", code == 200, stats0)

    model = (
        result.get("health", {}).get("model_name")
        or result.get("health", {}).get("model")
        or row.label
    )
    cap_model = urllib.parse.quote(str(model), safe="")
    code, caps = http_json(
        "GET", f"{base}/v1/models/{cap_model}/capabilities", timeout=20
    )
    supported_modes = (
        caps.get("supported_modes", []) if code == 200 and isinstance(caps, dict) else []
    )
    expected_modes = ["instruct"]
    if row.expect_reasoning:
        expected_modes.append("reasoning")
    if row.family == "deepseek_v4":
        expected_modes.append("max")
    check(
        "model_capabilities_endpoint",
        code == 200
        and all(mode in supported_modes for mode in expected_modes)
        and (
            row.family != "deepseek_v4"
            or caps.get("cache", {}).get("dsv4_composite_state") is True
        ),
        {"code": code, "caps": caps},
    )
    history: list[dict[str, Any]] = []
    r1 = post_chat(
        base,
        model,
        [{"role": "user", "content": "Remember these facts: color blue, pet cat. Reply exactly: noted."}],
        thinking=False,
        max_tokens=80,
    )
    c1, reasoning1, finish1 = extract_chat_text(r1["body"])
    check(
        "chat_thinking_off_basic",
        r1["code"] == 200 and "noted" in c1.lower() and not has_duplicate_block(c1 + reasoning1),
        {"finish": finish1, "content": c1[:300], "reasoning": reasoning1[:160], "code": r1["code"]},
    )
    history += r1["request"]["messages"] + [{"role": "assistant", "content": c1}]

    r2 = post_chat(
        base,
        model,
        history + [{"role": "user", "content": "What color did I say? Answer in one word."}],
        thinking=True if row.expect_reasoning else False,
        max_tokens=220,
    )
    c2, reasoning2, finish2 = extract_chat_text(r2["body"])
    check(
        "chat_thinking_on_or_reasoning_toggle_recall",
        r2["code"] == 200 and "blue" in (c2 + reasoning2).lower() and not has_duplicate_block(c2 + reasoning2),
        {
            "finish": finish2,
            "content": c2[:300],
            "reasoning_chars": len(reasoning2),
            "reasoning_head": reasoning2[:240],
            "code": r2["code"],
        },
    )

    if row.family == "deepseek_v4":
        code, max_resp = http_json(
            "POST",
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use max thinking. What is 3 plus 4? Include the final digit.",
                    }
                ],
                "thinking_mode": "max",
                "max_tokens": 320,
                "temperature": 0.6,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=240,
        )
        max_content, max_reasoning, max_finish = extract_chat_text(max_resp)
        max_joined = (max_content + "\n" + max_reasoning).lower()
        check(
            "dsv4_thinking_mode_max",
            code == 200
            and "7" in max_joined
            and not has_duplicate_block(max_joined),
            {
                "code": code,
                "finish": max_finish,
                "content": max_content[:300],
                "reasoning_chars": len(max_reasoning),
                "reasoning_head": max_reasoning[:240],
            },
        )

    # Responses API, including persisted tool-history shape. This first
    # check verifies replay of an already-completed tool turn; it deliberately
    # provides no tools and sets tool_choice=none so the model must use the
    # existing function_call_output instead of making a fresh call.
    responses_input = [
        {"role": "user", "content": "We used a directory tool. Remember this."},
        {
            "type": "function_call",
            "call_id": "call_audit_1",
            "name": "list_directory",
            "arguments": json.dumps({"path": "."}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_audit_1",
            "output": "README.md\npyproject.toml\nvmlx_engine",
        },
        {"type": "output_text", "text": "README.md"},
        {
            "type": "message",
            "role": "user",
            "content": (
                "Based only on the previous tool result text, copy exactly one "
                "listed string from this set: README.md, pyproject.toml, "
                "vmlx_engine. Do not call a tool. Reply with only that string."
            ),
        },
    ]
    code, resp = http_json(
        "POST",
        f"{base}/v1/responses",
        {
            "model": model,
            "input": responses_input,
            "max_output_tokens": 160,
            "temperature": 0.3,
            "stream": False,
            "enable_thinking": False,
            "tool_choice": "none",
        },
        timeout=240,
    )
    resp_text = extract_responses_text(resp)
    check(
        "responses_tool_history_continuation",
        code == 200 and any(x in resp_text.lower() for x in ("readme", "pyproject", "vmlx_engine")),
        {"code": code, "text": resp_text[:400], "raw_keys": list(resp.keys()) if isinstance(resp, dict) else None},
    )

    # Responses API auto tool choice. When tools ARE supplied and the model
    # emits its native tool-call syntax, the endpoint must parse it into a
    # structured Responses `function_call` item instead of returning raw
    # `<tool_call>` text.
    code, resp_tool = http_json(
        "POST",
        f"{base}/v1/responses",
        {
            "model": model,
            "input": "Use the list_directory tool for path '.' and do not answer in prose.",
            "tools": [
                {
                    "type": "function",
                    "name": "list_directory",
                    "description": "List files in a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            "tool_choice": "auto",
            "max_output_tokens": 160,
            "temperature": 0.1,
            "stream": False,
            "enable_thinking": False,
        },
        timeout=240,
    )
    fcalls = extract_responses_function_calls(resp_tool)
    raw_tool_text = extract_responses_text(resp_tool)
    check(
        "responses_auto_tool_choice_structured",
        code == 200
        and bool(fcalls)
        and any(c.get("name") == "list_directory" for c in fcalls)
        and "<tool_call>" not in raw_tool_text,
        {
            "code": code,
            "function_calls": fcalls,
            "text": raw_tool_text[:300],
            "raw_keys": list(resp_tool.keys()) if isinstance(resp_tool, dict) else None,
        },
    )

    code, anth = http_json(
        "POST",
        f"{base}/v1/messages",
        {
            "model": model,
            "max_tokens": 120,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly: anthropic-ok"}],
            "enable_thinking": False,
        },
        timeout=240,
    )
    anth_text = json.dumps(anth)[:1000].lower()
    check("anthropic_messages_basic", code == 200 and "anthropic-ok" in anth_text, {"code": code, "body": anth})

    code, ollama = http_json(
        "POST",
        f"{base}/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly: ollama-ok"}],
            "options": {"temperature": 0.0, "num_predict": 80},
        },
        timeout=240,
    )
    ollama_text = json.dumps(ollama)[:1000].lower()
    check("ollama_chat_basic", code == 200 and "ollama-ok" in ollama_text, {"code": code, "body": ollama})

    # Cache exact-hit coherence. Multi-turn memory is covered by the chat
    # recall checks above; this check isolates cache mechanics with a prompt
    # that stops cleanly and produces >3 tokens so the scheduler does not
    # skip cache donation as a short single-token benchmark response.
    cache_messages = [
        {
            "role": "user",
            "content": (
                "Cache audit exact-hit check. Reply exactly: blue blue blue blue."
            ),
        }
    ]
    r_cache = post_chat(
        base,
        model,
        cache_messages,
        thinking=False,
        max_tokens=80,
        temperature=0.1,
    )
    c_cache, _, _ = extract_chat_text(r_cache["body"])
    r_cache_repeat = post_chat(
        base,
        model,
        cache_messages,
        thinking=False,
        max_tokens=80,
        temperature=0.1,
    )
    c_cache_repeat, _, _ = extract_chat_text(r_cache_repeat["body"])
    _repeat_usage = (
        r_cache_repeat.get("body", {}).get("usage", {})
        if isinstance(r_cache_repeat.get("body"), dict)
        else {}
    )
    _repeat_details = (
        _repeat_usage.get("prompt_tokens_details") or {}
        if isinstance(_repeat_usage, dict)
        else {}
    )
    code, stats1 = http_json("GET", f"{base}/v1/cache/stats", timeout=20)
    _sched_cache = stats1.get("scheduler_cache", {}) if code == 200 and isinstance(stats1, dict) else {}
    _block_disk = stats1.get("block_disk_cache", {}) if code == 200 and isinstance(stats1, dict) else {}
    _cache_hit_observed = (
        int(_repeat_details.get("cached_tokens") or 0) > 0
        or int(_sched_cache.get("cache_hits") or _sched_cache.get("hits") or 0) > 0
        or int(_block_disk.get("disk_hits") or 0) > 0
    )
    _cache_bypass_expected = (
        row.family == "gemma4"
        and "Prefix cache is auto-bypassed on every request" in log_text
    )
    check(
        "cache_second_turn_coherent",
        r_cache["code"] == 200
        and r_cache_repeat["code"] == 200
        and c_cache.lower().count("blue") >= 3
        and c_cache_repeat.lower().count("blue") >= 3
        and not has_duplicate_block(c_cache + c_cache_repeat)
        and (_cache_hit_observed or _cache_bypass_expected),
        {
            "content": c_cache[:240],
            "repeat_content": c_cache_repeat[:240],
            "repeat_usage": _repeat_usage,
            "cache_hit_observed": _cache_hit_observed,
            "cache_bypass_expected": _cache_bypass_expected,
            "cache_stats": stats1 if code == 200 else stats0,
        },
    )

    failures = [c for c in result["checks"] if not c["ok"]]
    result["status"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures

    if keep_running:
        result["kept_running"] = True
    else:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="Comma-separated row IDs. Default: all.")
    ap.add_argument("--live", action="store_true", help="Actually start models and hit APIs.")
    ap.add_argument("--skip-slow", action="store_true", help="Skip rows tagged slow.")
    ap.add_argument("--py", default=str(DEFAULT_PY), help="Python executable to run vmlx_engine.")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--keep-running", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "production_family_audit.json"))
    args = ap.parse_args()

    rows = ROWS
    if args.rows:
        wanted = {x.strip() for x in args.rows.split(",") if x.strip()}
        rows = [r for r in rows if r.id in wanted]
    if args.skip_slow:
        rows = [r for r in rows if not r.slow]

    results: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": args.py,
        "live": args.live,
        "rows": [],
    }
    for row in rows:
        print(f"\n=== {row.id}: {row.label}")
        static = static_audit(row)
        print(
            "static:",
            "exists=" + str(static["exists"]),
            "model_type=" + str(static["config"].get("model_type")),
            "bits=" + str(static["jang"].get("mxtq_bits") or static["config"].get("quantization")),
            "issues=" + str(static["issues"]),
        )
        item: dict[str, Any] = {"static": static}
        if args.live:
            live = live_audit(
                row,
                Path(args.py),
                args.port,
                args.load_timeout,
                keep_running=args.keep_running,
            )
            print("live:", live["status"], "failures=", len(live.get("failures", [])))
            for chk in live.get("checks", []):
                print(" ", "OK" if chk["ok"] else "FAIL", chk["name"])
            item["live"] = live
        results["rows"].append(item)
        if args.live and args.keep_running:
            print("--keep-running set; stopping after first live row")
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {out}")

    if args.live:
        failed = [
            r
            for r in results["rows"]
            if r.get("live", {}).get("status") not in (None, "PASS", "SKIP")
        ]
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
