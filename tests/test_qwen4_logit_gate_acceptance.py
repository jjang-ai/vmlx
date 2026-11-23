"""Qualification must reject fallback-only runs and restore diagnostic wrappers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "qwen4_logit_gate", Path(__file__).parents[1] / "bench/qwen4_logit_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class QualificationTests(unittest.TestCase):
    def row(self, **changes):
        return dict(
            dict(arm="verify", context=8192, mean_kl=0.0, max_kl=0.0,
                 logit_rms=0.0, exact_logits=True, verifier_dispatches=1),
            **changes,
        )

    def test_fallback_is_not_qualification(self):
        self.assertFalse(gate.qualification_passes(self.row(verifier_dispatches=0)))
        self.assertTrue(gate.qualification_passes(self.row()))

    def test_exact_gate_cannot_be_replaced_by_tolerance(self):
        row = self.row(exact_logits=False, logit_rms=0.0001)
        self.assertTrue(gate.qualification_passes(row))
        self.assertFalse(gate.qualification_passes(row, require_exact=True))
        self.assertTrue(gate.qualification_passes(self.row(), require_exact=True))

    def test_existing_control_and_short_context_remain_valid(self):
        self.assertTrue(gate.qualification_passes(self.row(arm="repeat", verifier_dispatches=0)))
        self.assertTrue(gate.qualification_passes(self.row(context=1024, verifier_dispatches=0)))
        self.assertFalse(gate.qualification_passes(self.row(max_kl=0.051)))
        self.assertFalse(gate.qualification_passes(self.row(mean_kl=float("nan"))))

    def test_observer_preserves_result_and_restores_after_failure(self):
        result = object()
        values = iter([None, result])
        original = lambda *args, **kwargs: next(values)
        module = SimpleNamespace(qwen4_verify_sdpa=original)
        q = SimpleNamespace(shape=(1, 24, 3, 256), dtype="float16")
        k = SimpleNamespace(shape=(1, 2, 8192, 256))
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with gate.observe_verifier(module) as observed:
                self.assertIsNone(module.qwen4_verify_sdpa(q, k, k, None, scale=1))
                self.assertEqual(observed["dispatches"], 0)
                self.assertIs(module.qwen4_verify_sdpa(q, k, k, None, scale=1), result)
                self.assertEqual(observed["dispatches"], 1)
                self.assertEqual(observed["shapes"], [dict(rows=3, context=8192, dtype="float16")])
                raise RuntimeError("forward failed")
        self.assertIs(module.qwen4_verify_sdpa, original)


if __name__ == "__main__":
    unittest.main()
