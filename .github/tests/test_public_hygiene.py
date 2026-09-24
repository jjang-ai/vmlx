import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("public_hygiene", Path(__file__).parents[1] / "scripts/check_public_hygiene.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicDocumentBoundary(unittest.TestCase):
    def test_rejects_private_paths_even_without_content(self):
        for path in ("research/results.json", "docs/internal/notes.md", "docs/plans/work.md", "reports/run.json", "nested/HANDOFF.md"):
            with self.subTest(path=path):
                self.assertTrue(module.violations(path))

    def test_rejects_internal_document_renamed_to_public_path(self):
        for body in (b"# INTERNAL ONLY", b"> For Claude: REQUIRED SUB-SKILL", b"Run /Users/eric/model/probe.py"):
            with self.subTest(body=body):
                self.assertTrue(module.violations("docs/guide.md", body))

    def test_accepts_public_usage_and_runtime_internal_modules(self):
        for path, body in (("docs/api.md", b"Configure a private key through an environment variable."), ("jang-tools/_internal/jang_v3/encode.py", b""), ("tests/test_session_history.py", b""), ("README.md", b"python -m pip install jang")):
            with self.subTest(path=path):
                self.assertEqual(module.violations(path, body), [])


if __name__ == "__main__":
    unittest.main()
