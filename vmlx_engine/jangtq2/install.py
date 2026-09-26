"""Compatibility alias for :mod:`vmlx_engine.jangh.install`."""
import sys
from ..jangh import install as _implementation

sys.modules[__name__] = _implementation
