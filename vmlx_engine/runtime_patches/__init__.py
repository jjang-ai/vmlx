# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for upstream packages shipped alongside vMLX.

See ``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.1 & §1.2.

vMLX's release build applies these patches to the bundled Python at
packaging time (``panel/bundled-python/.../mlx_lm/models/deepseek_v3.py``).
The installers here also exist for users running ``vmlx_engine`` against
a system Python / user-managed ``mlx_lm`` — the installer refuses to
modify files under a ``vmlx/`` path, exactly mirroring
``jang_tools.kimi_prune.runtime_patch``'s refusal, so a stray dev-run
never corrupts the bundle.
"""
