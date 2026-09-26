"""Compatibility alias for :mod:`vmlx_engine.jangh.contract`."""
import sys
from ..jangh import contract as _implementation

sys.modules[__name__] = _implementation
