"""CPU-only store outcome tests; compile the production method without MLX import.

The seam under test is enqueue-result reporting, not tensor clone/serialization.
Loading only the method prevents even package import from initializing Metal.
"""
import ast
import logging
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import Mock


def production_store():
    source = Path(__file__).parents[1] / 'vmlx_engine/utils/ssm_companion_cache.py'
    tree = ast.parse(source.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SSMCompanionCache')
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'store')
    namespace = dict(List=List, Any=Any, Optional=Optional, logger=logging.getLogger('ssm-admission-test'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['store']


class AdmissionReportingTest(unittest.TestCase):
    def invoke(self, admitted=True, error=None, ram=False, disk=True):
        states = [object()]
        writer = SimpleNamespace(store=Mock(return_value=admitted, side_effect=error)) if disk else None
        cache = SimpleNamespace(
            ram_enabled=ram, _disk=writer,
            _clone_states=Mock(return_value=states),
            _estimate_state_nbytes=lambda value: 64,
            _key=lambda *args, **kwargs: 'abc123',
            _prefix_hash=lambda *args, **kwargs: 'family',
            _max_bytes=None, _store=OrderedDict(), _entry_nbytes={},
            _total_nbytes=0, _length_index={}, _evict_if_needed=Mock(),
        )
        with self.assertLogs('ssm-admission-test', level='INFO') as logs:
            result = production_store()(cache, [1, 2], 2, states)
        self.assertIsNone(result)  # Existing store API does not change.
        return cache, logs.output

    def test_false_admission_is_not_logged_as_disk_success(self):
        for reason in ('queue full', 'byte budget', 'writer closing'):
            with self.subTest(reason=reason):
                cache, logs = self.invoke(admitted=False)
                self.assertTrue(any('disk=False' in line for line in logs), logs)
                self.assertFalse(any('disk=True' in line for line in logs), logs)
                self.assertEqual(cache._store, {})
                self.assertEqual(cache._disk.store.call_count, 1)

    def test_accepted_enqueue_is_reported(self):
        _, logs = self.invoke(admitted=True)
        self.assertTrue(any('disk=True' in line for line in logs), logs)

    def test_exception_remains_unsuccessful(self):
        _, logs = self.invoke(error=RuntimeError('write unavailable'))
        self.assertTrue(any('disk=False' in line for line in logs), logs)

    def test_rejected_disk_does_not_change_ram_admission(self):
        cache, logs = self.invoke(admitted=False, ram=True)
        self.assertTrue(any('disk=False' in line for line in logs), logs)
        self.assertIn('abc123', cache._store)
        self.assertEqual(cache._total_nbytes, 64)
        cache._evict_if_needed.assert_called_once()

    def test_ram_only_store_does_not_claim_disk(self):
        cache, logs = self.invoke(ram=True, disk=False)
        self.assertTrue(any('disk=False' in line for line in logs), logs)
        self.assertIn('abc123', cache._store)


if __name__ == '__main__':
    unittest.main()
