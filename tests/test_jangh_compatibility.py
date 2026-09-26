"""JANGH ownership must preserve old imports without duplicate kernel state."""
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["contract", "format", "install", "kernels", "payload", "runtime_identity", "switch"])
def test_legacy_module_is_canonical_module(name):
    canonical = importlib.import_module(f"vmlx_engine.jangh.{name}")
    legacy = importlib.import_module(f"vmlx_engine.jangtq2.{name}")
    assert legacy is canonical
    assert Path(legacy.__file__).parent.name == "jangh"


def test_install_entrypoints_are_aliases():
    from vmlx_engine.jangh import install
    assert install.install_jangh is install.install_jangtq2
    assert install.is_jangh is install.is_jangtq2


def test_identity_tracks_canonical_sources():
    import hashlib
    from vmlx_engine.jangh import runtime_identity
    digest = hashlib.sha256()
    for path in sorted(Path(runtime_identity.__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    assert runtime_identity.runtime_identity().startswith("jangh=" + digest.hexdigest() + ";")


def test_legacy_first_imports_in_fresh_process():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", """
import importlib
for name in ('contract', 'format', 'install', 'kernels', 'payload', 'runtime_identity', 'switch'):
    legacy = importlib.import_module('vmlx_engine.jangtq2.' + name)
    canonical = importlib.import_module('vmlx_engine.jangh.' + name)
    assert legacy is canonical, name
"""], check=True)
