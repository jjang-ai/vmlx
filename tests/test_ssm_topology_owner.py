"""Topology must report the attached SSD store for both scheduler owners."""
from types import SimpleNamespace
import pytest
from vmlx_engine.server import _cache_topology_configuration
class EmptyRamCompanion:
    def __init__(self,disk): self._disk=disk
    def __len__(self): return 0
@pytest.mark.parametrize("owner,enabled",[("text",True),("mllm",True),("text",False),("mllm",False)])
def test_topology_reports_actual_ssm_disk_owner(owner,enabled):
    disk=object() if enabled else None
    scheduler=SimpleNamespace()
    if owner=="text": scheduler._ssm_state_cache=EmptyRamCompanion(disk)
    else: scheduler._ssm_companion_disk_store=disk
    result=_cache_topology_configuration(scheduler,{})
    assert result["instantiated"]["ssm_companion_l2"] is enabled

