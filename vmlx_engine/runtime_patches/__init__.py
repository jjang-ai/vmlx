# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for upstream packages shipped alongside vMLX.

See ``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.1 & §1.2 and
``research/DSV4-RUNTIME-ARCHITECTURE.md`` §3.

vMLX's release build applies these patches to the bundled Python at
packaging time (``panel/bundled-python/.../mlx_lm/models/deepseek_v3.py``).
The installers here also exist for users running ``vmlx_engine`` against
a system Python / user-managed ``mlx_lm`` — the installer refuses to
modify files under a ``vmlx/`` path, exactly mirroring
``jang_tools.kimi_prune.runtime_patch``'s refusal, so a stray dev-run
never corrupts the bundle.

Registered patches (auto-installed on ``import vmlx_engine.runtime_patches``):
  * ``kimi_k25_mla``          — Kimi K2.6 fp32 MLA L==1 SDPA cast
  * ``deepseek_v4_register``  — DSV4 mlx_lm.models.deepseek_v4 registration
"""

# Eagerly install every patch on first import so they land before any
# model load path walks config.json.model_type. Safe: each register()
# guards against already-patched state internally.
from . import deepseek_v4_register  # noqa: F401
