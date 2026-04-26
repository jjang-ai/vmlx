"""
Configuration source for command-line arguments.

CLI arguments take highest priority over all other sources.

Usage:
    # In server startup
    args = parse_cli_args([
        "--codebook-memory-limit-mb", "65536",
        "--kv-cache-quantization", "turboquant",
        "--turboquant-key-bits", "4",
        "--hybrid-ssm-recompute", "full",
        "--use-metal",
    ])

    config_manager = ConfigManager(model_name="Qwen3.5-35B-A3B-CODEBOOK-TEST")
    cli_overrides = parse_cli_args()
    config_manager.apply_cli_overrides(cli_overrides)
"""

import argparse
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# CLI argument definitions
CLI_ARGS = [
    # Memory settings
    ("--memory-total-budget-percent", "memory.total_budget_percent", float),
    ("--kv-cache-quantization", "memory.kv_cache.quantization", str),
    ("--kv-cache-memory-limit-mb", "memory.kv_cache.memory_limit_mb", int),
    ("--codebook-memory-limit-mb", "memory.codebook_cache.memory_limit_mb", int),
    ("--codebook-disk-cache-dir", "memory.codebook_cache.disk_cache_dir", str),
    ("--codebook-disk-max-gb", "memory.codebook_cache.disk_max_gb", int),
    ("--prefix-cache-enabled", "memory.prefix_cache.enabled", bool),
    ("--prefix-cache-memory-limit-mb", "memory.prefix_cache.memory_limit_mb", int),
    ("--paged-cache-enabled", "memory.paged_cache.enabled", bool),
    ("--paged-cache-block-size", "memory.paged_cache.block_size", int),
    ("--disk-cache-enabled", "memory.disk_cache.enabled", bool),
    ("--disk-cache-dir", "memory.disk_cache.cache_dir", str),
    ("--disk-cache-max-gb", "memory.disk_cache.max_gb", float),
    # Codebook settings
    ("--codebook-enabled", "codebook.enabled", str),
    ("--codebook-loading", "codebook.loading", str),
    ("--codebook-kernel", "codebook.kernel", str),
    ("--codebook-metal-threads", "codebook.kernel_config.threads_per_threadgroup", int),
    ("--codebook-metal-batch-size", "codebook.kernel_config.max_batch_size", int),
    # TurboQuant settings
    ("--turboquant-key-bits", "turboquant.default_key_bits", int),
    ("--turboquant-value-bits", "turboquant.default_value_bits", int),
    ("--turboquant-critical-key-bits", "turboquant.critical_key_bits", int),
    ("--turboquant-critical-value-bits", "turboquant.critical_value_bits", int),
    (
        "--turboquant-critical-layers",
        "turboquant.critical_layers",
        str,
    ),  # comma-separated
    ("--turboquant-sink-tokens", "turboquant.sink_tokens", int),
    ("--turboquant-seed", "turboquant.seed", int),
    # Hybrid settings
    ("--hybrid-ssm-recompute", "hybrid.ssm_recompute", str),
    ("--hybrid-checkpoint-interval", "hybrid.checkpoint_interval_tokens", int),
    ("--hybrid-detection", "hybrid.detection", str),
    # Inference settings
    ("--max-concurrent-requests", "inference.max_concurrent_requests", int),
    ("--prefill-batch-size", "inference.prefill_batch_size", int),
    ("--completion-batch-size", "inference.completion_batch_size", int),
    ("--max-tokens", "inference.max_tokens", int),
    ("--max-tokens-per-step", "inference.max_tokens_per_step", int),
    ("--sampling-temperature", "inference.sampling.temperature", float),
    ("--sampling-top-p", "inference.sampling.top_p", float),
    ("--sampling-top-k", "inference.sampling.top_k", int),
    ("--sampling-min-p", "inference.sampling.min_p", float),
    ("--sampling-repetition-penalty", "inference.sampling.repetition_penalty", float),
    # Kernel settings
    ("--use-metal", "kernel.use_metal", bool),
    ("--no-metal", "kernel.use_metal", bool),  # negation
    ("--num-threads", "kernel.num_threads", int),
    ("--compute-precision", "kernel.compute_precision", str),
    ("--kv-compute-precision", "kernel.kv_compute_precision", str),
    # Debug settings
    ("--debug-log-cache-hits", "debug.log_cache_hits", bool),
    ("--debug-log-cache-misses", "debug.log_cache_misses", bool),
    ("--debug-log-memory-usage", "debug.log_memory_usage", bool),
    ("--debug-profile-kernel-times", "debug.profile_kernel_times", bool),
    ("--debug-stats-interval", "debug.stats_interval_seconds", int),
    # Config file
    ("--config", "___config_file___", str),
]


class CLIConfigSource:
    """Parses and holds command-line argument overrides."""

    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self._overrides: Dict[str, Any] = {}
        self._config_file: Optional[str] = None
        if args:
            self._parse_args(args)

    def _parse_args(self, args: Dict[str, Any]):
        """Parse a dict of arguments (from argparse or direct)."""
        self._overrides = {}

        for cli_arg, config_path, arg_type in CLI_ARGS:
            # Handle negation flags
            normalized_key = cli_arg.lstrip("-").replace("-", "_")

            if cli_arg == "--no-metal":
                if normalized_key in args:
                    self._overrides[config_path] = not bool(args[normalized_key])
            elif normalized_key in args:
                value = args[normalized_key]
                if value is None:
                    continue

                # Special case: critical_layers as comma-separated string
                if config_path == "turboquant.critical_layers":
                    try:
                        value = [int(x.strip()) for x in value.split(",")]
                    except ValueError:
                        logger.warning(f"Invalid critical_layers value: {value}")
                        continue

                # Type coercion
                if arg_type == bool:
                    value = (
                        value
                        if isinstance(value, bool)
                        else value.lower() in ("true", "1", "yes", "on")
                    )
                elif arg_type == int:
                    value = int(value)
                elif arg_type == float:
                    value = float(value)

                self._overrides[config_path] = value

            # Handle explicit flags (--flag without value)
            elif cli_arg.startswith("--no-"):
                normalized = cli_arg[5:].replace("-", "_")
                if normalized in args and args[normalized]:
                    self._overrides[config_path] = False

    def load(self) -> Dict[str, Any]:
        """Return parsed CLI overrides as nested dict."""
        if not self._overrides:
            return {}

        # Convert flat keys to nested dict
        result: Dict[str, Any] = {}
        for key, value in self._overrides.items():
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result

    def get_overrides(self) -> Dict[str, Any]:
        """Return flat dict of CLI overrides for display."""
        return self._overrides.copy()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argparse parser with all vMLX configuration options."""
    parser = argparse.ArgumentParser(
        description="vMLX Inference Engine Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vmlx serve Qwen3.5-35B-A3B-CODEBOOK-TEST \\
      --codebook-memory-limit-mb 65536 \\
      --kv-cache-quantization turboquant \\
      --turboquant-key-bits 4 \\
      --hybrid-ssm-recompute full \\
      --use-metal

  vmlx serve Gemma-4-26B-A4B \\
      --codebook-enabled false \\
      --max-concurrent-requests 128 \\
      --sampling-temperature 0.8
        """,
    )

    # Memory settings group
    mem_group = parser.add_argument_group("Memory Settings")
    mem_group.add_argument(
        "--memory-total-budget-percent",
        type=float,
        help="Total memory budget as fraction of RAM (default: 0.85)",
    )
    mem_group.add_argument(
        "--kv-cache-quantization",
        choices=["none", "q4", "q8", "turboquant"],
        help="KV cache quantization method",
    )
    mem_group.add_argument(
        "--codebook-memory-limit-mb",
        type=int,
        help="Codebook cache memory limit in MB (0 = unlimited)",
    )
    mem_group.add_argument(
        "--codebook-disk-cache-dir", type=str, help="Codebook disk cache directory"
    )
    mem_group.add_argument(
        "--codebook-disk-max-gb", type=int, help="Codebook disk cache max size in GB"
    )

    # TurboQuant settings group
    tq_group = parser.add_argument_group("TurboQuant Settings")
    tq_group.add_argument(
        "--turboquant-key-bits",
        type=int,
        choices=[2, 3, 4, 5, 6],
        help="Default key bits for TurboQuant (default: 3)",
    )
    tq_group.add_argument(
        "--turboquant-value-bits",
        type=int,
        choices=[2, 3, 4, 5, 6],
        help="Default value bits for TurboQuant (default: 3)",
    )
    tq_group.add_argument(
        "--turboquant-critical-layers",
        type=str,
        help="Critical layers for higher precision (comma-separated, e.g., '0,1,2,-1')",
    )
    tq_group.add_argument(
        "--turboquant-sink-tokens",
        type=int,
        help="Number of sink tokens to keep at full precision (default: 4)",
    )

    # Hybrid settings group
    hybrid_group = parser.add_argument_group("Hybrid Architecture Settings")
    hybrid_group.add_argument(
        "--hybrid-ssm-recompute",
        choices=["full", "checkpoint"],
        help="SSM recompute strategy on cache miss",
    )
    hybrid_group.add_argument(
        "--hybrid-detection",
        type=str,
        help="Layer type detection ('auto' or path to layer_types.json)",
    )

    # Inference settings group
    inf_group = parser.add_argument_group("Inference Settings")
    inf_group.add_argument(
        "--max-concurrent-requests",
        type=int,
        help="Maximum concurrent requests (default: 256)",
    )
    inf_group.add_argument(
        "--prefill-batch-size", type=int, help="Prefill batch size (default: 8)"
    )
    inf_group.add_argument(
        "--completion-batch-size", type=int, help="Completion batch size (default: 32)"
    )
    inf_group.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate (default: 4096)"
    )
    inf_group.add_argument(
        "--sampling-temperature", type=float, help="Sampling temperature (default: 0.7)"
    )
    inf_group.add_argument(
        "--sampling-top-p", type=float, help="Nucleus sampling top-p (default: 0.9)"
    )

    # Kernel settings group
    kernel_group = parser.add_argument_group("Kernel Settings")
    kernel_group.add_argument(
        "--use-metal",
        action="store_true",
        default=None,
        help="Use Metal kernels (default: True)",
    )
    kernel_group.add_argument(
        "--no-metal", action="store_true", help="Disable Metal kernels"
    )
    kernel_group.add_argument(
        "--compute-precision",
        choices=["float16", "bfloat16", "float32"],
        help="Compute precision",
    )
    kernel_group.add_argument(
        "--kv-compute-precision",
        choices=["float16", "bfloat16", "float32"],
        help="KV cache compute precision",
    )

    # Debug settings group
    debug_group = parser.add_argument_group("Debug Settings")
    debug_group.add_argument(
        "--debug-log-cache-hits", action="store_true", help="Log cache hits"
    )
    debug_group.add_argument(
        "--debug-log-cache-misses", action="store_true", help="Log cache misses"
    )
    debug_group.add_argument(
        "--debug-log-memory-usage",
        action="store_true",
        help="Log memory usage periodically",
    )
    debug_group.add_argument(
        "--debug-profile-kernel-times",
        action="store_true",
        help="Profile kernel execution times",
    )

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    return parser


def parse_cli_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse command-line arguments into a dict."""
    parser = create_argument_parser()
    parsed = parser.parse_args(args)

    # Convert Namespace to dict, excluding None values
    result = {}
    for key, value in vars(parsed).items():
        if value is not None:
            # Convert underscores to hyphens for CLI keys
            cli_key = f"--{key.replace('_', '-')}"
            result[cli_key] = value

    return result
