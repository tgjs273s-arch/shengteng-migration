"""P5 asset receipt contracts using synthetic inputs; no torchrun, download or NPU."""

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

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
spec = importlib.util.spec_from_file_location("p5_runtime_binding", SKILL / "scripts" / "50_train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
import _runtime_asset_binding as binding


def root():
    base = SKILL.parent / "tmp" / "t10-runtime-binding"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="case-", dir=base))


def config(save=None):
    return yaml.safe_dump({"training": {"micro_batch_size": 4,
                                         "gradient_accumulation_steps": 1,
                                         "train_iters": 100, "load": None, "save": save},
                           "parallel": {"data_parallel_size": 2}},
                          allow_unicode=True).encode("utf-8")


class AssetBindingTests(unittest.TestCase):
    def test_capture_checks_actual_p2_list_model_paths_and_inventory_digests(self):
        case = root()
        hf, dcp, coco = case / "hf", case / "dcp", case / "coco"
        for directory in (hf, dcp, coco / "train2017"):
            directory.mkdir(parents=True)
        llava, data = case / "llava.json", case / "dataset.json"
        llava.write_text("[]", encoding="utf-8")
        data.write_text("[]", encoding="utf-8")
        conversion = dcp / "migration_conversion_receipt.json"
        conversion.write_text('{"schema":"fixture"}', encoding="utf-8")
        records = {
            "hf_identity": {"status": "local_complete", "file_inventory_sha256": "h",
                            "config_sha256": "c", "index_sha256": "i",
                            "model_revision": "revision-fixture"},
            "dcp_identity": {"status": "local_structure_and_hashes", "file_inventory_sha256": "d"},
            "llava_identity": {"status": "local_content_verified", "json_sha256": "l",
                               "order_sha256": "lo", "row_count": 1},
            "data_identity": {"status": "local_content_verified", "json_sha256": "j",
                              "order_sha256": "jo", "row_count": 1,
                              "referenced_images_sha256": "image-a"},
        }
        declared = {"schema": "migrator_assets.v2", "attempt_id": "p4-fixture",
                    "migration_id": "migration-fixture", "migration_manifest_sha256": "m",
                    "migration_manifest_path": str(case / "manifest.json"),
                    "migration_identity_state": "validated_overlay_and_target",
                    "readiness": "local_complete_official_unverified", "missing_required": [],
                    "checks": [{"ok": True}], "conversion": {
                        "receipt_sha256": binding.sha256(conversion)},
                    "assets": {key: {"path": str(path)} for key, path in {
                        "weight_hf": hf, "weight_dcp": dcp, "llava_json": llava,
                        "converted_json": data, "coco_images": coco / "train2017"}.items()},
                    **records}
        assets = case / "assets.json"
        assets.write_text(json.dumps(declared), encoding="utf-8")
        source = {"model": {"model_name_or_path": str(hf)},
                  "training": {"load": str(dcp)},
                  "data": {"dataset_param": {
                      "preprocess_parameters": {"model_name_or_path": str(hf)},
                      "basic_parameters": {"dataset": [str(data)], "dataset_dir": str(coco)}}}}
        fake_p4 = mock.Mock()
        fake_p4.verified_migration.return_value = {
            "migration_id": "migration-fixture", "manifest_sha256": "m",
            "manifest_path": str(case / "manifest.json")}
        fake_p4._conversion_receipt.return_value = {"schema": "fixture"}
        target = {"target_commit": "commit-fixture", "target_files_sha256": {"code.py": "hash"}}
        patches = (mock.patch.object(binding, "_p4_module", return_value=fake_p4),
                   mock.patch.object(binding, "verify_target_checkout", return_value=target),
                   mock.patch.object(binding, "inspect_hf", return_value=records["hf_identity"]),
                   mock.patch.object(binding, "inspect_dcp", return_value=records["dcp_identity"]),
                   mock.patch.object(binding, "inspect_llava", return_value=records["llava_identity"]),
                   mock.patch.object(binding, "inspect_data", return_value=records["data_identity"]))
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            def capture(doc):
                return binding.capture(str(assets), "bundle", "overlay", str(case),
                                       yaml.safe_dump(doc).encode("utf-8"))
            observed = capture(source)
            self.assertEqual(observed["data_json_sha256"], "j")
            self.assertEqual(observed["referenced_images_sha256"], "image-a")
            self.assertEqual(observed["conversion_receipt_sha256"], binding.sha256(conversion))
            wrong = json.loads(json.dumps(source))
            wrong["data"]["dataset_param"]["basic_parameters"]["dataset"].append("other.json")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                capture(wrong)
            wrong = json.loads(json.dumps(source))
            wrong["model"]["model_name_or_path"] = str(case / "other-hf")
            with self.assertRaisesRegex(ValueError, "model path"):
                capture(wrong)
            altered = {**records["data_identity"], "referenced_images_sha256": "image-b"}
            with mock.patch.object(binding, "inspect_data", return_value=altered):
                with self.assertRaisesRegex(ValueError, "differs from P4"):
                    capture(source)

    def test_cli_group_required_before_launch(self):
        case = root()
        source = case / "config.yaml"
        source.write_bytes(config())
        with mock.patch.object(sys, "argv", ["50_train.py", "--config", str(source),
                                             "--assets-json", str(case / "assets.json")]), \
             mock.patch.object(train, "launch_training") as launch, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(train.main(), 2)
        launch.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "fcntl runner requires Linux/WSL")
    def test_precheck_failure_preserves_old_log_and_receipts(self):
        case = root() / "中文 path's #: value"
        case.mkdir()
        log = case / "train.log"
        log.write_text("previous run\n", encoding="utf-8")
        with mock.patch.object(binding, "capture", side_effect=ValueError("same-size asset drift")):
            with self.assertRaises(train.AssetPrecheckError) as caught:
                train.launch_training("exit 0", str(log), detached=False,
                                      config_bytes=config(save=None),
                                      asset_inputs=("assets", "bundle", "overlay", "target"))
        run_dir = Path(caught.exception.run_dir)
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        integrity = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
        self.assertEqual(log.read_text(encoding="utf-8"), "previous run\n")
        self.assertEqual(record["asset_binding"]["state"], "PRECHECK_FAILED")
        self.assertIsNone(record["pid"])
        self.assertIsNone(integrity["train_rc"])
        self.assertEqual(integrity["run_id"], record["run_id"])
        self.assertFalse((run_dir / "runner.log").exists())
        with open(str(log) + ".lock", "a") as lock:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @unittest.skipUnless(os.name == "posix", "fcntl runner requires Linux/WSL")
    def test_legacy_unbound_and_bound_save_disabled_keep_two_tuple(self):
        case = root()
        log = case / "log.txt"
        proc, run_dir = train.launch_training("exit 0", str(log), detached=False,
                                               config_bytes=config(save=None))
        self.assertEqual(proc.wait(timeout=10), 0)
        old = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(old["asset_binding"]["state"], "UNBOUND")
        self.assertFalse(old["checkpoint_save"]["enabled"])
        observed = {"assets_json_sha256": "a" * 64, "migration_id": "fixture",
                    "data_json_sha256": "b" * 64}
        with mock.patch.object(binding, "capture", return_value=observed):
            proc, run_dir = train.launch_training("exit 0", str(log), detached=False,
                                                   config_bytes=config(save=None),
                                                   asset_inputs=("assets", "bundle", "overlay", "target"))
            self.assertEqual(proc.wait(timeout=10), 0)
        current = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(current["asset_binding"]["state"], "PRECHECK_OK")
        self.assertEqual(current["asset_binding"]["prelaunch"], observed)
        self.assertFalse(current["checkpoint_save"]["enabled"])

    def test_postrun_same_size_content_drift_blocks_complete_log(self):
        case = root()
        source = case / "config.yaml"
        source.write_bytes(config(save=None))
        env = case / "env.json"
        env.write_text(json.dumps({"recommended_profile": {"world_size": 2},
                                   "capabilities": {"can_train": True}}), encoding="utf-8")
        log = case / "train.log"
        run_dir = case / "runs" / "p5-synthetic"
        run_dir.mkdir(parents=True)
        before = {"assets_json_sha256": "a" * 64, "migration_id": "fixture",
                  "data_json_sha256": "b" * 64, "data_bytes": 100}
        after = {**before, "data_json_sha256": "c" * 64}  # same byte count, different content

        class ZeroProcess:
            pid = 123

            def wait(self, timeout):
                return 0

        def fake_launch(body, log_path, detached=True, config_bytes=None, asset_inputs=None):
            snapshot, intent = train.snapshot_config_for_run(config_bytes, str(run_dir))
            (run_dir / "effective_config.yaml").write_bytes(snapshot)
            record = {"run_id": run_dir.name, "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                      "effective_config_sha256": hashlib.sha256(snapshot).hexdigest(),
                      "checkpoint_save": intent,
                      "asset_binding": {**binding.new_binding(), "state": "PRECHECK_OK", "prelaunch": before}}
            (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
            Path(log_path).write_text("".join(
                f"[Rank 0 | Local Rank 0] iteration {step}/100 | "
                f"consumed samples: {step * 8} | elapsed time per iteration (ms): 100.0 | "
                "learning rate: 1.0E-6 | global batch size: 8 | "
                "loss: 1.0 | grad norm: 2.0 |\n" for step in range(1, 101)), encoding="utf-8")
            return ZeroProcess(), str(run_dir)

        argv = ["50_train.py", "--config", str(source), "--env", str(env),
                "--log", str(log), "--foreground", "--assets-json", "assets",
                "--migration-bundle", "bundle", "--migration-overlay", "overlay"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(train, "launch_training", fake_launch), \
             mock.patch.object(binding, "capture", autospec=True, return_value=after) as capture, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = train.main()
        self.assertEqual(len(capture.call_args.args), 5)
        check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertNotEqual(rc, 0)
        self.assertEqual(check["train_rc"], 0)
        self.assertEqual(check["selected_count"], 100)
        self.assertEqual(check["state"], "INCOMPLETE")
        self.assertEqual(check["asset_binding"]["state"], "DRIFTED")
        self.assertEqual(record["asset_binding"]["postrun"], after)
        self.assertEqual(check["asset_binding"]["run_record_sha256"],
                         hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest())

    def test_postrun_matching_identity_can_complete_with_binding(self):
        case = root()
        source = case / "config.yaml"
        source.write_bytes(config(save=None))
        env = case / "env.json"
        env.write_text(json.dumps({"recommended_profile": {"world_size": 2},
                                   "capabilities": {"can_train": True}}), encoding="utf-8")
        log = case / "train.log"
        run_dir = case / "runs" / "p5-matching"
        run_dir.mkdir(parents=True)
        identity = {"assets_json_sha256": "a" * 64, "migration_id": "fixture",
                    "data_json_sha256": "b" * 64}

        class ZeroProcess:
            pid = 123

            def wait(self, timeout):
                return 0

        def fake_launch(body, log_path, detached=True, config_bytes=None, asset_inputs=None):
            snapshot, intent = train.snapshot_config_for_run(config_bytes, str(run_dir))
            (run_dir / "effective_config.yaml").write_bytes(snapshot)
            (run_dir / "run.json").write_text(json.dumps({
                "run_id": run_dir.name,
                "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "effective_config_sha256": hashlib.sha256(snapshot).hexdigest(),
                "checkpoint_save": intent,
                "asset_binding": {**binding.new_binding(), "state": "PRECHECK_OK",
                                  "prelaunch": identity}}), encoding="utf-8")
            Path(log_path).write_text("".join(
                f"[Rank 0 | Local Rank 0] iteration {step}/100 | "
                f"consumed samples: {step * 8} | elapsed time per iteration (ms): 100.0 | "
                "learning rate: 1.0E-6 | global batch size: 8 | "
                "loss: 1.0 | grad norm: 2.0 |\n" for step in range(1, 101)), encoding="utf-8")
            return ZeroProcess(), str(run_dir)

        argv = ["50_train.py", "--config", str(source), "--env", str(env),
                "--log", str(log), "--foreground", "--assets-json", "assets",
                "--migration-bundle", "bundle", "--migration-overlay", "overlay"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(train, "launch_training", fake_launch), \
             mock.patch.object(binding, "capture", autospec=True, return_value=identity) as capture, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = train.main()
        self.assertEqual(len(capture.call_args.args), 5)
        self.assertEqual(rc, 0)
        check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
        receipt = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(check["state"], "COMPLETE")
        self.assertEqual(check["asset_binding"]["state"], "VERIFIED")
        self.assertEqual(receipt["asset_binding"]["prelaunch"],
                         receipt["asset_binding"]["postrun"])

    def test_legacy_main_accepts_complete_fake_run_without_new_receipt(self):
        case = root()
        source = case / "config.yaml"
        source.write_bytes(config(save=None))
        env = case / "env.json"
        env.write_text(json.dumps({"recommended_profile": {"world_size": 2},
                                   "capabilities": {"can_train": True}}), encoding="utf-8")
        log = case / "train.log"
        run_dir = case / "runs" / "p5-old-fixture"
        run_dir.mkdir(parents=True)

        class ZeroProcess:
            pid = 123

            def wait(self, timeout):
                return 0

        def old_launch(body, log_path, detached=True, config_bytes=None):
            snapshot, _ = train.snapshot_config_for_run(config_bytes, str(run_dir))
            (run_dir / "effective_config.yaml").write_bytes(snapshot)
            Path(log_path).write_text("".join(
                f"[Rank 0 | Local Rank 0] iteration {step}/100 | "
                f"consumed samples: {step * 8} | elapsed time per iteration (ms): 100.0 | "
                "learning rate: 1.0E-6 | global batch size: 8 | "
                "loss: 1.0 | grad norm: 2.0 |\n" for step in range(1, 101)), encoding="utf-8")
            return ZeroProcess(), str(run_dir)

        argv = ["50_train.py", "--config", str(source), "--env", str(env),
                "--log", str(log), "--foreground"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(train, "launch_training", old_launch), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = train.main()
        check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 0)
        self.assertEqual(check["state"], "COMPLETE")
        self.assertEqual(check["asset_binding"]["state"], "UNBOUND")

    def test_parseable_run_receipt_with_wrong_binding_type_is_incomplete(self):
        case = root()
        source = case / "config.yaml"
        source.write_bytes(config(save=None))
        env = case / "env.json"
        env.write_text(json.dumps({"recommended_profile": {"world_size": 2},
                                   "capabilities": {"can_train": True}}), encoding="utf-8")
        log = case / "train.log"
        run_dir = case / "runs" / "p5-bad-binding"
        run_dir.mkdir(parents=True)

        class ZeroProcess:
            pid = 123

            def wait(self, timeout):
                return 0

        def fake_launch(body, log_path, detached=True, config_bytes=None, asset_inputs=None):
            snapshot, intent = train.snapshot_config_for_run(config_bytes, str(run_dir))
            (run_dir / "effective_config.yaml").write_bytes(snapshot)
            (run_dir / "run.json").write_text(json.dumps({
                "run_id": run_dir.name,
                "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "effective_config_sha256": hashlib.sha256(snapshot).hexdigest(),
                "checkpoint_save": intent, "asset_binding": bad_binding}), encoding="utf-8")
            Path(log_path).write_text("".join(
                f"[Rank 0 | Local Rank 0] iteration {step}/100 | "
                f"consumed samples: {step * 8} | elapsed time per iteration (ms): 100.0 | "
                "learning rate: 1.0E-6 | global batch size: 8 | "
                "loss: 1.0 | grad norm: 2.0 |\n" for step in range(1, 101)), encoding="utf-8")
            return ZeroProcess(), str(run_dir)

        argv = ["50_train.py", "--config", str(source), "--env", str(env),
                "--log", str(log), "--foreground", "--assets-json", "assets",
                "--migration-bundle", "bundle", "--migration-overlay", "overlay"]
        good = {**binding.new_binding(), "state": "PRECHECK_OK", "prelaunch": {"id": "one"}}
        cases = (
            (["fake VERIFIED"], "asset binding must be a JSON object"),
            ({**good, "prelaunch": ["bad"]}, "prelaunch/postrun structure invalid"),
            ({**good, "postrun": ["bad"]}, "prelaunch/postrun structure invalid"),
            ({**good, "problems": "bad"}, "asset binding problems must be a list"),
        )
        for bad_binding, reason in cases:
            with self.subTest(reason=reason, binding=bad_binding), \
                 mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(train, "launch_training", fake_launch), \
                 mock.patch.object(binding, "capture", autospec=True) as capture, \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = train.main()
                capture.assert_not_called()
            self.assertNotEqual(rc, 0)
            check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
            self.assertEqual(check["train_rc"], 0)
            self.assertEqual(check["selected_count"], 100)
            self.assertEqual(check["state"], "INCOMPLETE")
            self.assertEqual(check["asset_binding"]["state"], "DRIFTED")
            self.assertIn(reason, check["asset_binding"]["error"])
            self.assertEqual(check["asset_binding"]["run_record_sha256"],
                             hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
