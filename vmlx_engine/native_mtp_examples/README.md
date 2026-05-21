# Native MTP No-Load Examples

These scripts are reusable, no-load examples for inspecting and planning vMLX native MTP work. They read metadata, construct dry-run command matrices, and parse synthetic or captured logs. They do not load model weights unless `server_smoke.py --execute` is used with the explicit live-run guard.

Current runtime contract:

- Qwen 3.5/3.6 dense and MoE families are the advertised native-MTP runtime-supported families.
- DSV4/DeepSeek bundles can contain MTP metadata, but DSV4 native MTP is not currently a claimed active runtime path until custom generator support exists and real server logs prove activation and acceptance.
- Default runtime draft depth is 3.
- `VMLINUX_NATIVE_MTP_DEPTH` and `VMLX_NATIVE_MTP_DEPTH` are aliases and clamp to `1..3`.
- A validated `vmlx_mtp_tuning.json` can override the default depth when no depth env override is present.

Examples:

```bash
python -m vmlx_engine.native_mtp_examples.inspect_mtp_metadata /path/to/model --json
python -m vmlx_engine.native_mtp_examples.generate_server_command /path/to/model --depth 3
python -m vmlx_engine.native_mtp_examples.env_flag_matrix /path/to/model --include-disabled
python -m vmlx_engine.native_mtp_examples.parse_mtp_logs /path/to/server.log --json
python -m vmlx_engine.native_mtp_examples.server_smoke /path/to/model --depth 3
```

`server_smoke.py` prints a dry-run server command plus `curl` checks by default. It refuses live execution unless both safeguards are present:

```bash
VMLINUX_NATIVE_MTP_LIVE_RUN_ACK=I_UNDERSTAND_THIS_LOADS_MODEL_WEIGHTS \
  python -m vmlx_engine.native_mtp_examples.server_smoke /path/to/model --execute --allow-live
```

Do not use the live mode for large bundles without explicit approval and enough RAM headroom.
