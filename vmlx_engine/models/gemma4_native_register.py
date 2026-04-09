# SPDX-License-Identifier: Apache-2.0
"""Register vendored Gemma 4 native text MoE module under the ``mlx_lm.models.*``
namespace so the upstream loader (``mlx_lm.utils.load_model``) can find it.

Background
----------
Native Gemma 4 text MoE (the architecture used by Gemma 4 26B) ships in
``mlx-lm >= 0.31.2``. vMLX is currently pinned at ``mlx-lm >= 0.30.2`` and the
4-agent audit decided to **local-port** new features rather than bump the
upstream pin (matches Agent 1's PrefixCacheManager port, Agent 2's
SequenceStateMachine port, and Agent 3's BatchMambaCache.lengths/advance
backport).

This module vendors two upstream files:
  - ``vmlx_engine/models/_gemma4_text_upstream.py``  (≈666 lines, the model)
  - ``vmlx_engine/models/_gemma4_upstream.py``       (≈92 lines, the wrapper)

Both are copied verbatim from the mlx-lm 0.31.2 source under Apple's MIT-style
copyright header. Their relative imports (``from .base import ...``) have been
rewritten to absolute imports (``from mlx_lm.models.base import ...``) so the
files can live outside the ``mlx_lm.models`` package.

The remaining gap is that ``mlx_lm.utils.load_model`` resolves the model class
via ``importlib.import_module(f"mlx_lm.models.{model_type}")``. To make that
work for ``model_type == "gemma4"`` on a 0.31.1 install, we register the
vendored modules under their expected dotted names in ``sys.modules`` BEFORE
the loader tries to import them.

Forward compatibility
---------------------
When the user upgrades to ``mlx-lm >= 0.31.2``, this register becomes a no-op:
the upstream ``mlx_lm.models.gemma4`` is imported normally before our register
runs (or after — either way, the ``setdefault`` semantics preserve whichever
came first). To revert the local-port entirely after the bump:

  1. Delete ``_gemma4_upstream.py`` and ``_gemma4_text_upstream.py``
  2. Delete this file
  3. Remove the ``register_gemma4_native()`` call from ``vmlx_engine/__init__.py``

That leaves vMLX consuming the upstream Gemma 4 directly.

Usage
-----
Call ``register_gemma4_native()`` once at process startup, before any model
load attempt. The function is idempotent.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("vmlx_engine")

_REGISTERED = False


def register_gemma4_native() -> bool:
    """Install vendored Gemma 4 modules into ``sys.modules``.

    Returns ``True`` if the register installed our vendored versions,
    ``False`` if upstream Gemma 4 is already present (which means the user is
    on ``mlx-lm >= 0.31.2`` and we should defer to upstream).

    The function is idempotent — calling it multiple times is a no-op after
    the first successful registration.
    """
    global _REGISTERED
    if _REGISTERED:
        return True

    # If upstream mlx-lm 0.31.2+ is installed, both modules already exist in
    # sys.modules (via normal import). Don't override — defer to upstream.
    upstream_present = False
    try:
        import mlx_lm.models  # noqa: F401
        # Try the import the way mlx_lm.utils.load_model would do it
        import importlib
        try:
            importlib.import_module("mlx_lm.models.gemma4")
            importlib.import_module("mlx_lm.models.gemma4_text")
            upstream_present = True
        except ModuleNotFoundError:
            upstream_present = False
    except ImportError:
        # mlx_lm not installed at all — caller should have errored earlier
        return False

    if upstream_present:
        logger.debug(
            "register_gemma4_native: upstream mlx_lm.models.gemma4 already present "
            "(mlx-lm >= 0.31.2 detected); not registering vendored copy"
        )
        _REGISTERED = True
        return False

    # Upstream is missing — register our vendored copies. Import order matters:
    # _gemma4_upstream imports `from . import _gemma4_text_upstream as gemma4_text`
    # so the text module must be importable first.
    try:
        from . import _gemma4_text_upstream as _vendored_text  # noqa: F401
        from . import _gemma4_upstream as _vendored_wrapper  # noqa: F401
    except Exception as e:  # pragma: no cover
        logger.warning(
            "register_gemma4_native: failed to import vendored Gemma 4 modules: %s",
            e,
        )
        return False

    # setdefault keeps any existing registration (defensive against race)
    sys.modules.setdefault("mlx_lm.models.gemma4_text", _vendored_text)
    sys.modules.setdefault("mlx_lm.models.gemma4", _vendored_wrapper)

    _REGISTERED = True
    logger.info(
        "Registered vendored Gemma 4 native text MoE modules under "
        "mlx_lm.models.gemma4 and mlx_lm.models.gemma4_text "
        "(local-port from mlx-lm 0.31.2 — see vmlx_engine/models/gemma4_native_register.py)"
    )
    return True
