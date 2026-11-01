# Bundled MLX dependency

`mlx-runtime.json` selects the upstream MLX source, package version, build tools,
and deployment targets. The engine does not carry a fork of MLX's kernels.
The source-build overlay changes only the Python distribution version so a
rebuild does not acquire today's date or impersonate an official release.

Build both `mlx` and `mlx-metal` from the same source and install them together.
`mlx-runtime.py prepare` records the source commit, target, toolchain, wheel
tags and SHA-256 hashes; `verify` rejects mismatched or modified artifacts.
Use a separate wheel directory for each version and platform. Retain its
provenance JSON with the release evidence.

The native flavor targets macOS 26.2 to include upstream NAX kernels. Its wheel
tag is `macosx_26_0_arm64`, because macOS wheel tags use the major OS version;
the app must declare the actual 26.2 minimum. The compatibility flavor retains
the macOS 14 target and does not advertise NAX on unsupported hardware.

To move to an official upstream release:

1. Verify that its tag contains every required `upstream_prs` merge commit.
2. Set `version` to the official version and `source` to `{"kind": "pypi"}`.
   Review deployment targets against that release's wheel requirements.
3. Prepare and verify the two official wheel pairs. The source-build overlay
   and compiler are no longer used. Keep the former artifact receipts.
4. Trace upstream changes to the engine's dispatch and version-qualified custom
   kernels. Validate affected numerical, model and packaging rows before
   changing their qualification gates. Do not blindly enable custom kernels
   for a new MLX version or rerun unrelated retained rows.

Upstream benchmark speedups are not application speed claims. Measure the
affected application paths on their actual hardware and model bundles.
