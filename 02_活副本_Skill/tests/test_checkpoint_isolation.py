"""Offline P5 save isolation contracts; never launches torchrun or an NPU."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("p5_checkpoint_isolation", SKILL_ROOT / "scripts" / "50_train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def fixture_root():
    root = SKILL_ROOT.parent / "tmp" / "t08-checkpoint-isolation"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="case-", dir=root))


def config_bytes(save="./shared checkpoint", save_format=None):
    training = {"micro_batch_size": 4, "gradient_accumulation_steps": 1,
                "train_iters": 100, "load": None, "save": save}
    if save_format is not None:
        training["save_format"] = save_format
    return yaml.safe_dump({"training": training, "parallel": {"data_parallel_size": 2}},
                          allow_unicode=True, sort_keys=False).encode("utf-8")


class CheckpointIsolationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "fcntl runner requires Linux/WSL")
    def test_two_runs_do_not_share_save_directory_or_change_source(self):
        root = fixture_root()
        source = root / "config.yaml"
        source_bytes = config_bytes(save="./旧目录 中文 path's #: value")
        source.write_bytes(source_bytes)
        log = root / "train.log"
        records = []
        for _ in range(2):
            proc, run_dir = train.launch_training("exit 0", str(log), detached=False,
                                                   config_bytes=source.read_bytes())
            self.assertEqual(proc.wait(timeout=10), 0)
            record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
            snapshot_bytes = (Path(run_dir) / "effective_config.yaml").read_bytes()
            effective = yaml.safe_load(snapshot_bytes)
            save_path = str(Path(run_dir) / "checkpoints")
            self.assertEqual(effective["training"]["save"], save_path)
            self.assertEqual(record["checkpoint_save"]["source_value"],
                             "./旧目录 中文 path's #: value")
            self.assertEqual(record["checkpoint_save"]["effective_path"], save_path)
            self.assertEqual(record["checkpoint_save"]["actual_format"], "dcp")
            self.assertEqual(record["source_config_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(record["effective_config_sha256"], hashlib.sha256(snapshot_bytes).hexdigest())
            self.assertFalse(Path(save_path).exists(), "dummy runner never saved a checkpoint")
            records.append(record)
        self.assertNotEqual(records[0]["run_id"], records[1]["run_id"])
        self.assertNotEqual(records[0]["checkpoint_save"]["effective_path"],
                            records[1]["checkpoint_save"]["effective_path"])
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_null_and_false_keep_saving_disabled(self):
        root = fixture_root()
        for save in (None, False, ""):
            with self.subTest(save=save):
                source = config_bytes(save=save)
                snapshot, intent = train.snapshot_config_for_run(source, str(root / "p5-test"))
                self.assertNotIn("save", yaml.safe_load(snapshot)["training"])
                self.assertTrue(intent["source_present"])
                self.assertEqual(intent["source_value"], save)
                self.assertFalse(intent["enabled"])
                self.assertFalse(intent["effective_present"])
                self.assertIsNone(intent["effective_path"])
                self.assertIsNone(intent["actual_format"])

    def test_missing_save_key_preserves_source_bytes(self):
        root = fixture_root()
        source_doc = yaml.safe_load(config_bytes())
        del source_doc["training"]["save"]
        source = yaml.safe_dump(source_doc, allow_unicode=True, sort_keys=False).encode("utf-8")
        snapshot, intent = train.snapshot_config_for_run(source, str(root / "p5-test"))
        self.assertEqual(snapshot, source)
        self.assertFalse(intent["source_present"])
        self.assertFalse(intent["enabled"])

    @unittest.skipUnless(os.name == "posix", "fcntl runner requires Linux/WSL")
    def test_null_save_runner_keeps_snapshot_and_does_not_create_checkpoint_path(self):
        root = fixture_root()
        source = config_bytes(save=None)
        proc, run_dir = train.launch_training("exit 0", str(root / "null.log"),
                                               detached=False, config_bytes=source)
        self.assertEqual(proc.wait(timeout=10), 0)
        record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        snapshot = (Path(run_dir) / "effective_config.yaml").read_bytes()
        self.assertNotIn("save", yaml.safe_load(snapshot)["training"])
        self.assertEqual(record["source_config_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(record["effective_config_sha256"], hashlib.sha256(snapshot).hexdigest())
        self.assertFalse(record["checkpoint_save"]["enabled"])
        self.assertIsNone(record["checkpoint_save"]["effective_path"])
        self.assertFalse((Path(run_dir) / "checkpoints").exists())

    def test_hf_and_auto_save_format_do_not_reach_launch(self):
        root = fixture_root()
        for fmt in ("hf", "auto", "unknown"):
            with self.subTest(save_format=fmt):
                config = root / (fmt + ".yaml")
                config.write_bytes(config_bytes(save_format=fmt))
                stderr = io.StringIO()
                argv = ["50_train.py", "--config", str(config), "--log", str(root / "train.log")]
                with mock.patch.object(sys, "argv", argv), \
                     mock.patch.object(train, "launch_training") as launch, \
                     contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    rc = train.main()
                self.assertEqual(rc, 3)
                self.assertIn("training.save_format", stderr.getvalue())
                launch.assert_not_called()

    def test_explicit_dcp_is_accepted_but_save_still_isolated(self):
        root = fixture_root()
        snapshot, intent = train.snapshot_config_for_run(
            config_bytes(save="./old", save_format="dcp"), str(root / "p5-one"))
        self.assertEqual(yaml.safe_load(snapshot)["training"]["save"], intent["effective_path"])
        self.assertEqual(intent["actual_format"], "dcp")

    def test_zero_exit_and_complete_log_cannot_replace_save_receipt(self):
        root = fixture_root()
        config = root / "config.yaml"
        source = config_bytes(save="./old-shared-save")
        config.write_bytes(source)
        env = root / "env.json"
        env.write_text(json.dumps({"recommended_profile": {"world_size": 2},
                                   "capabilities": {"can_train": True}}), encoding="utf-8")
        log = root / "train.log"
        run_dir = root / "runs" / "p5-synthetic"
        run_dir.mkdir(parents=True)

        class ZeroProcess:
            pid = 123

            def wait(self, timeout):
                return 0

        def fake_launch(body, log_path, detached=True, config_bytes=None):
            snapshot, _ = train.snapshot_config_for_run(config_bytes, str(run_dir))
            (run_dir / "effective_config.yaml").write_bytes(snapshot)
            Path(log_path).write_text("".join(
                f"[Rank 0 | Local Rank 0] iteration {step}/100 | "
                f"consumed samples: {step * 8} | elapsed time per iteration (ms): 100.0 | "
                "learning rate: 1.0E-6 | global batch size: 8 | "
                "loss: 1.0 | grad norm: 2.0 |\n" for step in range(1, 101)), encoding="utf-8")
            # Deliberately omit run.json. A complete log is not save provenance.
            return ZeroProcess(), str(run_dir)

        argv = ["50_train.py", "--config", str(config), "--env", str(env),
                "--log", str(log), "--foreground"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(train, "launch_training", fake_launch), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = train.main()
        check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
        self.assertNotEqual(rc, 0)
        self.assertEqual(check["source_config"], str(config))
        self.assertEqual(check["source_config_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(check["config"], str(run_dir / "effective_config.yaml"))
        self.assertEqual(check["config_sha256"], hashlib.sha256(
            (run_dir / "effective_config.yaml").read_bytes()).hexdigest())
        self.assertEqual(check["source_config_sha256_after_run"],
                         check["source_config_sha256"])
        self.assertEqual(check["config_sha256_after_run"], check["config_sha256"])
        self.assertEqual(check["train_rc"], 0)
        self.assertEqual(check["selected_count"], 100)
        self.assertEqual(check["state"], "INCOMPLETE")
        self.assertIn("checkpoint_save_receipt_missing_or_changed", check["problems"])


if __name__ == "__main__":
    unittest.main()
