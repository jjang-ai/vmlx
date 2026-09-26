"""Compatibility alias for :mod:`vmlx_engine.jangh.kernels`."""
import sys
from ..jangh import kernels as _implementation

sys.modules[__name__] = _implementation
