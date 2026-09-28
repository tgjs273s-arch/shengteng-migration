"""Offline regressions for environment checks; no NPU or downloads required."""

import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ARTIFACTS = SCRIPTS.parents[1] / "tmp" / "environment_tests"


def fixture_directory():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="environment-", dir=ARTIFACTS)


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_script("05_preflight.py")


class EnvironmentCheckTests(unittest.TestCase):
    def test_missing_dev_directory_does_not_crash_offline_probe(self):
        # Windows has no /dev; the old preflight called os.listdir('/dev') directly.
        with mock.patch("glob.glob", side_effect=FileNotFoundError("/dev")):
            self.assertEqual(preflight.davinci_nodes(), [])

    def test_preflight_reports_no_npu_when_dev_is_absent(self):
        directory = fixture_directory()
        preflight.ROWS.clear()
        with mock.patch.object(preflight.os, "listdir", side_effect=FileNotFoundError("/dev")), \
             mock.patch.object(preflight, "davinci_nodes", return_value=[]), \
             mock.patch.object(preflight, "npu_info", return_value={"visible": False, "chip_count": None}), \
             mock.patch.object(preflight, "torch_npu_info", return_value=None), \
             mock.patch.object(preflight, "cann_version", return_value=None), \
             mock.patch.object(preflight, "pkg_manager", return_value=None), \
             mock.patch.object(preflight, "missing_commands", return_value=[]), \
             mock.patch.object(preflight, "default_msmm_dir", return_value=directory), \
             mock.patch.object(preflight, "default_data_dir", return_value=directory), \
             mock.patch.object(preflight, "_http_ok", return_value="000"), \
             mock.patch.object(preflight, "degrade"), \
             mock.patch.object(preflight, "DEG_FILE", str(Path(directory) / "degradations.json")), \
             mock.patch.object(sys, "argv", ["05_preflight.py"]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = preflight.main()
        self.assertEqual(rc, 3)
        self.assertIn({"capability": "npu_visible", "status": "BLOCKED"},
                      [{"capability": row["capability"], "status": row["status"]}
                       for row in preflight.ROWS])

    def test_msmm_directory_without_trainer_is_not_ready(self):
        directory = fixture_directory()
        source = preflight.msmm_source_info(directory)
        self.assertFalse(source["entry_present"])
        self.assertIsNone(source["git_commit"])

    def test_msmm_entry_without_checkout_identity_is_explicit(self):
        directory = fixture_directory()
        entry = Path(directory) / "mindspeed_mm" / "fsdp" / "train" / "trainer.py"
        entry.parent.mkdir(parents=True)
        entry.write_text("# test fixture\n", encoding="utf-8")
        with mock.patch.object(preflight.subprocess, "run", side_effect=OSError("git unavailable")):
            source = preflight.msmm_source_info(directory)
        self.assertTrue(source["entry_present"])
        self.assertIsNone(source["git_commit"])
        self.assertIsNone(source["git_dirty"])

    def test_torch_npu_mismatch_and_unknown_are_not_match(self):
        self.assertEqual(preflight.torch_pair_status("2.7.1+cpu", "2.7.1.post10"), "MATCH")
        self.assertEqual(preflight.torch_pair_status("2.7.1", "2.8.0"), "MISMATCH")
        self.assertEqual(preflight.torch_pair_status("2.7.1", None), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
