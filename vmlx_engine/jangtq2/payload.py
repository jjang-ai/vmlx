"""Compatibility alias for :mod:`vmlx_engine.jangh.payload`."""
import sys
from ..jangh import payload as _implementation

sys.modules[__name__] = _implementation
