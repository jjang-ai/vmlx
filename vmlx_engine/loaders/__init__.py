# SPDX-License-Identifier: Apache-2.0
"""JANGTQ model loaders — thin wrappers over jang_tools.

Exists so callers can follow the module layout described in
``research/KIMI-K2.6-VMLX-INTEGRATION.md`` §1.3–1.4 and still get the
stable jang_tools entry points bundled in vMLX's Python runtime.

Everything in this package is a re-export. The production loaders live
in ``jang_tools`` (bundled wheel); we import from them at call time so
version skews between a dev-installed ``jang_tools`` and the bundle
don't surprise us.
"""
