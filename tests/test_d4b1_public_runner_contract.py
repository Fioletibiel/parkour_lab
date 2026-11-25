"""Source-free tests of the public D4B1 open-model runner's safety contract."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "lab" / "d4b1_open_model_native.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("d4b1_public_check", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicD4B1Contract(unittest.TestCase):
    def test_source_blobs_frozen(self):
        mod = load_runner()
        self.assertEqual(len(mod.MODELS), 2)
        self.assertEqual(mod.SOURCE_COMMIT,
                         "d9b05d470b1a481c222372c85b75772faf8f7792")
        self.assertEqual(mod.CONVERTER_COMMIT,
                         "8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f")

    def test_git_blob_hash_is_git_compatible(self):
        mod = load_runner()
        payload = b"public synthetic fixture"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.txt"
            path.write_bytes(payload)
            expected = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\\0" + payload
            ).hexdigest()
            self.assertEqual(mod.model_git_sha1(path), expected)

    def test_frozen_cohort_domains(self):
        mod = load_runner()
        self.assertEqual(len(mod.CORE_KNEES), len(mod.CORE_HIPS))
        self.assertEqual(len(mod.CORE_KNEES), 4)
        self.assertEqual(len(mod.EXTRA_KNEES), len(mod.EXTRA_HIPS))
        self.assertEqual(len(mod.EXTRA_KNEES), 2)
        self.assertTrue(all(2.0944 < a < 2.44346095279206
                            for pair in mod.EXTRA_KNEES for a in pair))
        self.assertEqual((mod.POS_LIMIT_M, mod.ANG_LIMIT_DEG),
                         (.001, .5))

    def test_nonfinite_rejection(self):
        mod = load_runner()
        for number in (float("nan"), float("inf"), -1, float("-inf")):
            with self.subTest(number=str(number)):
                with self.assertRaises(ValueError):
                    mod.check_finite_nonnegative(number)

    def test_no_private_checkout_or_upload(self):
        content = (ROOT / ".github" / "workflows" /
                   "d4b1-public-open-model-native.yml").read_text()
        self.assertIn("runs-on: ubuntu-24.04", content)
        self.assertIn("permissions:\n  contents: read", content)
        self.assertNotIn("Fioletibiel/Parkour", content)
        self.assertNotIn("upload-artifact", content)
        self.assertIn("original_open_models", content)
        self.assertIn("if: always()", content)

    def test_converter_called_with_active_venv_interpreter(self):
        tree = ast.parse(RUNNER.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "converter_command")
        calls = [x for x in ast.walk(fn) if isinstance(x, ast.Call)]
        subprocess_calls = [
            c for c in calls if isinstance(c.func, ast.Attribute)
            and c.func.attr == "run"
        ]
        self.assertEqual(len(subprocess_calls), 1)
        first = subprocess_calls[0].args[0]
        self.assertIsInstance(first, ast.List)
        self.assertIsInstance(first.elts[0], ast.Attribute)
        self.assertEqual(first.elts[0].attr, "executable")


if __name__ == "__main__":
    unittest.main()
