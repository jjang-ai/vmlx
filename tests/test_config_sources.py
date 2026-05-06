# SPDX-License-Identifier: Apache-2.0
"""Config source regressions for the legacy config manager.

The main serve CLI owns production startup, but this config subsystem is still
importable and should not advertise or preserve stale cache modes.
"""

import pytest


def test_cli_source_accepts_parse_cli_args_dashed_keys():
    from vmlx_engine.config.sources.cli_source import (
        CLIConfigSource,
        parse_cli_args,
    )

    parsed = parse_cli_args(
        [
            "--kv-cache-quantization",
            "q4",
            "--turboquant-key-bits",
            "4",
            "--use-metal",
        ]
    )
    loaded = CLIConfigSource(args=parsed).load()

    assert loaded["memory"]["kv_cache"]["quantization"] == "q4"
    assert loaded["turboquant"]["default_key_bits"] == 4
    assert loaded["kernel"]["use_metal"] is True


def test_cli_source_accepts_normalized_argparse_keys():
    from vmlx_engine.config.sources.cli_source import CLIConfigSource

    loaded = CLIConfigSource(
        args={
            "kv_cache_quantization": "q8",
            "turboquant_key_bits": "4",
        }
    ).load()

    assert loaded["memory"]["kv_cache"]["quantization"] == "q8"
    assert loaded["turboquant"]["default_key_bits"] == 4


def test_config_cli_no_longer_accepts_turboquant_as_kv_mode():
    from vmlx_engine.config.sources.cli_source import parse_cli_args

    with pytest.raises(SystemExit):
        parse_cli_args(["--kv-cache-quantization", "turboquant"])


def test_kv_cache_config_maps_legacy_turboquant_alias_to_q4():
    from vmlx_engine.config.models import KVCacheConfig

    cfg = KVCacheConfig(quantization="turboquant")

    assert cfg.quantization == "q4"


def test_env_source_accepts_documented_and_legacy_kv_quant_names(monkeypatch):
    from vmlx_engine.config.sources.env_source import EnvConfigSource

    monkeypatch.setenv("VMLX_MEMORY_KV_CACHE_QUANTIZATION", "q8")
    loaded = EnvConfigSource().load()
    assert loaded["memory"]["kv_cache"]["quantization"] == "q8"

    monkeypatch.delenv("VMLX_MEMORY_KV_CACHE_QUANTIZATION")
    monkeypatch.setenv("VMLX_KV_CACHE_QUANTIZATION", "q4")
    loaded = EnvConfigSource().load()
    assert loaded["memory"]["kv_cache"]["quantization"] == "q4"
