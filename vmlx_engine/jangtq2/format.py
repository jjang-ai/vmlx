"""Compatibility alias for :mod:`vmlx_engine.jangh.format`."""
import sys
from ..jangh import format as _implementation

sys.modules[__name__] = _implementation
