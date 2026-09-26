"""Compatibility alias for :mod:`vmlx_engine.jangh.switch`."""
import sys
from ..jangh import switch as _implementation

sys.modules[__name__] = _implementation
