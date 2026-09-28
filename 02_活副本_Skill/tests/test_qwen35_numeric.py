"""T04 identity and partial-state regressions; NPU execution belongs to T12."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "_qwen35_numeric.py"
spec = importlib.util.spec_from_file_location("qwen35_numeric_test_target", HELPER)
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)


class IdentityAndPartialTests(unittest.TestCase):
    def fixture(self):
        root = ROOT.parent / "tmp" / "qwen35_numeric_tests"
        root.mkdir(parents=True, exist_ok=True)
        base = Path(tempfile.mkdtemp(prefix="numeric-", dir=root))
        checkout = base / "checkout"
        source = checkout / numeric.MODEL
        source.parent.mkdir(parents=True)
        source.write_text("# fixed target model\n", encoding="utf-8")
        doc = {"schema": "qwen35_migration.v1", "migration_id": "test-migration",
               "source": {"commit": "source-revision"},
               "target": {"commit": numeric.TARGET_COMMIT,
                          "source_files_sha256": {numeric.MODEL: hashlib.sha256(source.read_bytes()).hexdigest()}}}
        manifest = base / "migration_manifest.json"
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        return checkout, source, manifest

    def test_t03_identity_rejection_propagates(self):
        checkout, source, manifest = self.fixture()
        bundle = manifest.parent / "bundle"
        validate = mock.Mock(side_effect=ValueError("overlay identity mismatch"))
        verify_checkout = mock.Mock()
        with mock.patch.object(numeric, "_load_t03", return_value=(validate, verify_checkout)):
            with self.assertRaisesRegex(ValueError, "overlay identity mismatch"):
                numeric.verify_identity(manifest, bundle, checkout)
        validate.assert_called_once_with(bundle.resolve(), manifest.parent.resolve())
        verify_checkout.assert_not_called()
        with self.assertRaisesRegex(ValueError, "explicit migration manifest"):
            numeric.verify_identity(source, bundle, checkout)

    def test_t03_checkout_drift_rejection_propagates(self):
        checkout, source, manifest = self.fixture()
        bundle = manifest.parent / "bundle"
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        validate = mock.Mock(return_value=(doc, b"patched"))
        verify_checkout = mock.Mock(side_effect=ValueError("target checkout has other tracked changes"))
        with mock.patch.object(numeric, "_load_t03", return_value=(validate, verify_checkout)):
            with self.assertRaisesRegex(ValueError, "other tracked changes"):
                numeric.verify_identity(manifest, bundle, checkout)
        verify_checkout.assert_called_once_with(checkout.resolve(), require_patched=True)

    def test_missing_target_dependency_remains_partial_after_identity_check(self):
        checkout, source, manifest = self.fixture()
        torch_stub = types.ModuleType("torch")
        torch_stub.__version__ = "test-only"
        torch_stub.__file__ = __file__
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        receipt = {"target_commit": numeric.TARGET_COMMIT}
        with mock.patch.object(numeric, "verify_identity", return_value=(doc, receipt)), \
             mock.patch.object(numeric.importlib, "import_module", side_effect=ImportError("target dependencies absent")), \
             mock.patch.dict(sys.modules, {"torch": torch_stub}):
            rows = numeric.run(manifest, manifest.parent / "bundle", checkout, manifest.parent / "numeric-artifacts")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["status"] == "contract-only" for r in rows))
        self.assertTrue(all(r["validation_level"] == "dependency_unavailable" for r in rows))
        self.assertFalse(any("numeric_pass" in json.dumps(r) for r in rows))

    def test_decode_runtime_error_keeps_prefill_metrics(self):
        checkout, source, manifest = self.fixture()

        def run_until_decode(progress):
            progress["stage"] = "text model prefill"
            progress["metrics"]["prefill.hidden"] = {"shape": [1, 4, 128], "finite": True}
            progress["stage"] = "text model decode"
            raise RuntimeError("real decode failure")

        result = numeric.execute_check("target_text_model_cpu_eager_cache", run_until_decode,
                                       fixture={"seed": 9137}, identity={"migration_id": "fixture"},
                                       runtime={"torch_version": "fixture"}, artifact_dir=manifest.parent / "out")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failed_stage"], "text model decode")
        self.assertIn("prefill.hidden", result["completed_metrics"])
        self.assertIn("RuntimeError: real decode failure", result["err"])

    def test_artifact_write_failure_cannot_pass(self):
        checkout, source, manifest = self.fixture()
        with mock.patch.object(numeric.ArtifactWriter, "__init__", side_effect=OSError("artifact disk failure")):
            result = numeric.execute_check("target_gdn_triton_prefill", lambda progress: {"status": "forward_ok"},
                                           fixture={"seed": 1}, identity={}, runtime={},
                                           artifact_dir=manifest.parent / "out")
        self.assertEqual(result["status"], "error")
        self.assertIn("artifact disk failure", result["err"])

    def test_actual_cli_serializes_new_rows_without_legacy_shapes(self):
        checkout, source, manifest = self.fixture()
        outdir = manifest.parent / "cli"
        script = ROOT / "scripts" / "30_verify_ops.py"
        child = r'''
import json, runpy, sys, types
from unittest import mock
torch = types.ModuleType('torch')
nn = types.ModuleType('torch.nn')
functional = types.ModuleType('torch.nn.functional')
torch.device = lambda name: name
torch.nn = nn
nn.functional = functional
numeric = types.ModuleType('_qwen35_numeric')
numeric.run = lambda manifest, bundle, checkout, output: [
    {'op': 'target_cpu', 'status': 'forward_ok', 'validation_level': 'forward_cache',
     'metrics': {'prefill.hidden': {'shape': [1, 4, 128], 'finite': True}}},
    {'op': 'target_npu', 'status': 'contract-only', 'validation_level': 'not_executed',
     'reason': 'no NPU in control-flow fixture'},
]
sys.modules['_qwen35_numeric'] = numeric
script, outdir = sys.argv[1:]
def trace(frame, event, arg):
    if frame.f_code.co_filename == script and frame.f_code.co_name == 'main' and event == 'call':
        frame.f_globals['probe_backend'] = lambda: ('cpu', 'test-only CPU')
        frame.f_globals['run_matrix'] = lambda device: []
        sys.settrace(None)
    return trace
sys.argv = [script, '--out', outdir, '--migration-manifest', 'm.json',
            '--migration-bundle', 'bundle', '--target-checkout', 'checkout']
with mock.patch.dict(sys.modules, {'torch': torch, 'torch.nn': nn, 'torch.nn.functional': functional}):
    sys.settrace(trace)
    runpy.run_path(script, run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", child, str(script), str(outdir)],
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads((outdir / "ops_matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["status"], "PARTIAL")
        self.assertEqual([r["op"] for r in payload["matrix"]], ["target_cpu", "target_npu"])
        self.assertIn("VERIFY_PARTIAL", result.stdout)


if __name__ == "__main__":
    unittest.main()
