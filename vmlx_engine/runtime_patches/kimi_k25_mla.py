# SPDX-License-Identifier: Apache-2.0
"""Kimi K2.6 MLA fp32-SDPA patch installer.

Patches ``mlx_lm.models.deepseek_v3.DeepseekV3Attention.__call__`` so
the L==1 MLA absorb path casts q/k/v/mask to ``mx.float32`` before
``scaled_dot_product_attention`` and back to bf16 afterwards.

Background: without this fix, Kimi K2.6 (``kimi_k25`` model type, built
on DeepseekV3) decoding at bf16 drifts ~7.0 in logit magnitude per
token and produces repetition loops after ~14 tokens on quantized
bundles (see GLM-5.1 / DSV3.2 history in
``research/VMLX-RUNTIME-FIXES.md``).

Three install modes, in order of preference:

1. **vMLX release build** — ``panel/scripts/bundle-python.sh`` edits
   ``panel/bundled-python/.../mlx_lm/models/deepseek_v3.py`` in place
   BEFORE signing the DMG. No runtime patch needed for shipped users.

2. **Import-time monkey-patch (this module's default)** — call
   :func:`install` from a bootstrap script. Idempotent: if the source
   file already contains the patch marker we no-op.

3. **File patch for a user-managed ``mlx_lm``** — :func:`install_file`
   copies ``research/deepseek_v3_patched.py`` (shipped in jang_tools'
   docs tree) over the target file with a timestamped backup. Refuses
   to touch any file under a ``vmlx/`` path (mirroring
   ``jang_tools.kimi_prune.runtime_patch``) so it never corrupts a
   bundled install — that's mode 1's job.

``install_file`` is a thin re-export of
``jang_tools.kimi_prune.runtime_patch.apply`` so the refusal guard and
backup contract stay in one place. ``install`` is net-new: it verifies
the patch is live at runtime and attempts a monkey-patch only when the
file-edit path would be rejected.
"""

from __future__ import annotations

import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PATCH_MARKER = "JANG fast fix"


def is_patched() -> bool:
    """Return True if ``mlx_lm.models.deepseek_v3`` source already
    contains the fp32-SDPA fix (via the build-time in-place edit).

    Runs ``inspect.getsource`` on the imported module so it works even
    when ``mlx_lm`` comes from a .pyc in the bundled Python.
    """
    try:
        import mlx_lm.models.deepseek_v3 as _dv3  # noqa: WPS433 runtime import
        import inspect

        src = inspect.getsource(_dv3.DeepseekV3Attention.__call__)
        return PATCH_MARKER in src and "q_sdpa" in src
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("is_patched() failed: %s", exc)
        return False


def install() -> bool:
    """Ensure the fp32-SDPA MLA fix is live in the current process.

    Returns ``True`` if the patch is active (either already present in
    the bundled file or newly monkey-patched into the class), ``False``
    if install was not possible (``mlx_lm`` not importable, etc).

    Idempotent — safe to call from every bootstrap, every process.
    """
    if is_patched():
        logger.debug("Kimi MLA patch already active")
        return True

    # Fallback: attempt a file patch only outside vMLX's bundle. Mirrors
    # jang_tools.kimi_prune.runtime_patch.apply's refusal contract so we
    # never corrupt a signed release artifact.
    target = _locate_target()
    if target is None:
        return False
    if _is_vmlx_bundle(target):
        logger.warning(
            "Kimi MLA patch refused: target %s is a vMLX-bundled file. "
            "Bundled releases apply this patch at build time; if you are "
            "seeing this message inside /Applications/vMLX.app, please "
            "file a bug — the build-time patch step was skipped.",
            target,
        )
        return False
    try:
        from jang_tools.kimi_prune.runtime_patch import apply as _apply
    except ImportError:
        logger.warning(
            "Kimi MLA patch needs jang_tools.kimi_prune.runtime_patch for "
            "file-install; import failed. Install jang_tools or use the "
            "bundled vMLX Python."
        )
        return False
    rc = _apply(dry_run=False)
    return rc == 0


def install_file(dry_run: bool = False) -> int:
    """Thin re-export of ``jang_tools.kimi_prune.runtime_patch.apply``.

    Use this from a command line when you want the backup-and-replace
    behavior. Returns the same ``int`` exit code as the underlying
    script (0 on success, 2 on error).
    """
    try:
        from jang_tools.kimi_prune.runtime_patch import apply as _apply
    except ImportError as _ie:
        raise ImportError(
            "vmlx_engine.runtime_patches.kimi_k25_mla.install_file requires "
            "`jang_tools.kimi_prune` in the active Python environment."
        ) from _ie
    return _apply(dry_run=dry_run)


def _locate_target() -> Optional[Path]:
    spec = find_spec("mlx_lm.models.deepseek_v3")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _is_vmlx_bundle(path: Path) -> bool:
    """Does this file live inside a vMLX-shipped Python bundle?

    Matches ``/vmlx/…``, ``/vmlx-…``, and ``vMLX.app/Contents/`` paths.
    Used to refuse file patches on shipped artifacts (the build-time
    step handles those).
    """
    s = str(path).lower()
    return (
        "/vmlx/" in s
        or "/vmlx-" in s
        or "vmlx.app/" in s
    )


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if the patch is already active, 2 otherwise. No edits.",
    )
    args = ap.parse_args(argv)
    if args.check:
        return 0 if is_patched() else 2
    return install_file(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
