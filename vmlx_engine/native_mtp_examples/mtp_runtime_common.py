#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_NATIVE_MTP_DEPTH = 3
NATIVE_MTP_DEPTH_ENVS = ("VMLINUX_NATIVE_MTP_DEPTH", "VMLX_NATIVE_MTP_DEPTH")
NATIVE_MTP_ENABLE_ENVS = ("VMLINUX_NATIVE_MTP", "VMLX_NATIVE_MTP")
LIVE_RUN_ACK_ENV = "VMLINUX_NATIVE_MTP_LIVE_RUN_ACK"
LIVE_RUN_ACK_VALUE = "I_UNDERSTAND_THIS_LOADS_MODEL_WEIGHTS"
LIVE_RUN_WARNING = (
    "WARNING: native MTP server probes can load large model weights and consume "
    "substantial RAM. These examples default to dry-run and must not be used for "
    "large live runs unless you explicitly opt in."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return value if isinstance(value, dict) else {}


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def clamp_native_mtp_depth(raw: Any, *, default: int | None = None) -> int | None:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(3, value))


def effective_depth_from_env(
    env: dict[str, str] | None = None,
    *,
    default: int = DEFAULT_NATIVE_MTP_DEPTH,
) -> tuple[int, str]:
    source_env = env if env is not None else os.environ
    for name in NATIVE_MTP_DEPTH_ENVS:
        if name in source_env:
            depth = clamp_native_mtp_depth(source_env.get(name), default=default)
            return int(depth if depth is not None else default), name
    return default, "default"


def build_native_mtp_env(
    *,
    enabled: bool = True,
    depth: Any = None,
    base_env: dict[str, str] | None = None,
    trace: bool = False,
    debug_tokens: bool = False,
    use_tuning: bool = True,
) -> dict[str, str]:
    env = dict(base_env or {})
    enabled_value = "1" if enabled else "0"
    for name in NATIVE_MTP_ENABLE_ENVS:
        env[name] = enabled_value
    clamped = clamp_native_mtp_depth(depth)
    if clamped is not None:
        for name in NATIVE_MTP_DEPTH_ENVS:
            env[name] = str(clamped)
    env["VMLINUX_NATIVE_MTP_TRACE"] = "1" if trace else "0"
    env["VMLINUX_NATIVE_MTP_DEBUG_TOKENS"] = "1" if debug_tokens else "0"
    env["VMLINUX_NATIVE_MTP_USE_TUNING"] = "1" if use_tuning else "0"
    return env


def tuning_depth(model_dir: str | Path | None) -> tuple[int | None, str | None]:
    if not model_dir:
        return None, None
    tuning = read_json_file(Path(model_dir) / "vmlx_mtp_tuning.json")
    if not tuning:
        return None, None
    candidates: list[tuple[str, Any]] = []
    native_mtp = tuning.get("native_mtp")
    if isinstance(native_mtp, dict):
        allowed = (
            native_mtp.get("blocked") is not True
            and native_mtp.get("validated") is not False
            and native_mtp.get("output_equivalent") is not False
        )
        if allowed:
            candidates.append(("native_mtp.best_depth", native_mtp.get("best_depth")))
    best_native = tuning.get("best_native_mtp_depth")
    if isinstance(best_native, dict):
        candidates.append(("best_native_mtp_depth.best_depth", best_native.get("best_depth")))
    candidates.append(("best_depth", tuning.get("best_depth")))

    for source, raw_depth in candidates:
        depth = clamp_native_mtp_depth(raw_depth)
        if depth is not None:
            return depth, f"vmlx_mtp_tuning.json:{source}"
    return None, None


def inspect_bundle_no_load(model_dir: str | Path) -> dict[str, Any]:
    from vmlx_engine.native_mtp import inspect_native_mtp_bundle

    return inspect_native_mtp_bundle(Path(model_dir))


def shell_join(parts: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def env_shell_prefix(env: dict[str, str]) -> str:
    return " ".join(
        f"{shlex.quote(name)}={shlex.quote(str(value))}"
        for name, value in sorted(env.items())
    )


def command_with_env(env: dict[str, str], command: Sequence[str | Path]) -> str:
    prefix = env_shell_prefix(env)
    command_text = shell_join(command)
    return f"{prefix} {command_text}" if prefix else command_text


def build_server_command(
    model_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "native-mtp-local",
    depth: Any = None,
    mllm: bool = False,
    disable_prefix_cache: bool = False,
    extra_args: Iterable[str] = (),
    python: str | Path | None = None,
) -> dict[str, Any]:
    env = build_native_mtp_env(enabled=True, depth=depth)
    command: list[str | Path] = [
        python or sys.executable,
        "-m",
        "vmlx_engine.cli",
        "serve",
        str(model_dir),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        model_name,
    ]
    if mllm:
        command.append("--is-mllm")
    if disable_prefix_cache:
        command.append("--disable-prefix-cache")
    command.extend(str(item) for item in extra_args)
    if depth is None:
        tuned_depth, tuned_source = tuning_depth(model_dir)
        reported_depth = tuned_depth if tuned_depth is not None else DEFAULT_NATIVE_MTP_DEPTH
        depth_source = tuned_source or "runtime_default"
    else:
        reported_depth = clamp_native_mtp_depth(depth, default=DEFAULT_NATIVE_MTP_DEPTH)
        depth_source = "argument"
    return {
        "dry_run": True,
        "warning": LIVE_RUN_WARNING,
        "env": env,
        "command": [str(item) for item in command],
        "shell": command_with_env(env, command),
        "native_mtp_depth": reported_depth,
        "native_mtp_depth_source": depth_source,
    }


def live_run_allowed(env: dict[str, str] | None = None, *, allow_live: bool = False) -> bool:
    source_env = env if env is not None else os.environ
    return bool(allow_live and source_env.get(LIVE_RUN_ACK_ENV) == LIVE_RUN_ACK_VALUE)


def require_live_run_allowed(
    env: dict[str, str] | None = None,
    *,
    allow_live: bool = False,
) -> None:
    if not live_run_allowed(env, allow_live=allow_live):
        raise SystemExit(
            f"{LIVE_RUN_WARNING}\nRefusing to execute. Pass --allow-live and set "
            f"{LIVE_RUN_ACK_ENV}={LIVE_RUN_ACK_VALUE}."
        )


def command_matrix_rows(
    model_dir: str | Path,
    *,
    depths: Iterable[Any] = (1, 2, 3),
    include_disabled: bool = False,
    include_default: bool = True,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "native-mtp-local",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_default:
        rows.append(
            {
                "label": "native_mtp_default_depth",
                "depth": DEFAULT_NATIVE_MTP_DEPTH,
                "depth_source": "default_or_validated_tuning",
                **build_server_command(
                    model_dir,
                    host=host,
                    port=port,
                    model_name=model_name,
                    depth=None,
                ),
            }
        )
    seen: set[int] = set()
    for raw_depth in depths:
        depth = clamp_native_mtp_depth(raw_depth)
        if depth is None or depth in seen:
            continue
        seen.add(depth)
        row = build_server_command(
            model_dir,
            host=host,
            port=port,
            model_name=model_name,
            depth=depth,
        )
        row.update(
            {
                "label": f"native_mtp_d{depth}",
                "depth": depth,
                "depth_source": "env_aliases_clamped_1_to_3",
            }
        )
        rows.append(row)
    if include_disabled:
        env = build_native_mtp_env(enabled=False, depth=None)
        command = build_server_command(
            model_dir,
            host=host,
            port=port,
            model_name=model_name,
            depth=None,
        )["command"]
        rows.append(
            {
                "label": "native_mtp_disabled",
                "depth": None,
                "depth_source": "disabled",
                "dry_run": True,
                "warning": LIVE_RUN_WARNING,
                "env": env,
                "command": command,
                "shell": command_with_env(env, command),
            }
        )
    return rows
