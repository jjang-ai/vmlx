"""Compatibility alias for :mod:`vmlx_engine.jangh.runtime_identity`."""
import sys
from ..jangh import runtime_identity as _implementation

sys.modules[__name__] = _implementation
