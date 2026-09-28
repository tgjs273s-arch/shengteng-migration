"""Offline T08 contracts; no conversion, model download, or real reload."""

import importlib.util
import contextlib
import functools
import io
import inspect
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
import _model_export as export
import _runtime_asset_binding as runtime_binding

spec = importlib.util.spec_from_file_location("t08_export_cli", SKILL / "scripts" / "66_export_model.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def scratch():
    root = SKILL.parent / "tmp" / "t08-export-tests"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="case-", dir=root))


def put_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def shard(path, tensors):
    header = {}
    cursor = 0
    for key, shape in tensors.items():
        nbytes = 2
        for size in shape:
            nbytes *= size
        header[key] = {"dtype": "BF16", "shape": shape,
                       "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))


class ExportContracts(unittest.TestCase):
    def test_decorated_loader_source_uses_original_function(self):
        def actual_loader():
            return "dcp"

        original_path = Path(__file__).with_name("original_loader_source.py").resolve()
        actual_loader.__code__ = actual_loader.__code__.replace(
            co_filename=str(original_path))

        @functools.wraps(actual_loader)
        def decorated_loader():
            return actual_loader()

        self.assertEqual(Path(inspect.getfile(decorated_loader)).resolve(),
                         Path(__file__).resolve())
        self.assertNotEqual(Path(inspect.getfile(decorated_loader)).resolve(), original_path)
        self.assertEqual(cli.unwrapped_source_path(decorated_loader), original_path)

    def test_training_receipt_binds_run_config_log_and_save(self):
        root = scratch()
        run = root / "p5-current"
        run.mkdir()
        source = root / "source.yaml"
        source.write_text("training: {}", encoding="utf-8")
        effective = run / "effective_config.yaml"
        effective.write_text("training: {save: local}", encoding="utf-8")
        log = run / "train.log"
        log.write_text("iteration 1/1", encoding="utf-8")
        save = {"enabled": True, "actual_format": "dcp",
                "effective_path": str(run / "checkpoints")}
        pre = {"assets_json_sha256": "a" * 64, "migration_id": "example"}
        binding = {"schema": "p5_asset_binding.v1", "state": "VERIFIED",
                   "prelaunch": pre, "postrun": pre, "problems": []}
        put_json(run / "run.json", {"run_id": run.name, "checkpoint_save": save,
                "config_snapshot": str(effective), "log": str(log),
                "asset_binding": binding,
                "source_config_sha256": export.file_hash(source),
                "effective_config_sha256": export.file_hash(effective)})
        put_json(run / "train_integrity.json", {
            "run_id": run.name, "state": "COMPLETE", "train_rc": 0,
            "failure_markers": [], "checkpoint_save": save, "source_config": str(source),
            "source_config_sha256": export.file_hash(source),
            "source_config_sha256_after_run": export.file_hash(source),
            "config": str(effective), "config_train_iters": 1,
            "config_sha256": export.file_hash(effective),
            "config_snapshot_sha256_after_run": export.file_hash(effective),
            "log": str(log), "log_sha256": export.file_hash(log),
            "asset_binding": {"schema": "p5_asset_binding.v1", "state": "VERIFIED",
                "problems": [], "run_record": str(run / "run.json"),
                "run_record_sha256": export.file_hash(run / "run.json"),
                "assets_json_sha256": pre["assets_json_sha256"],
                "migration_id": pre["migration_id"]},
            "expected_start": 1, "expected_end": 1, "unique_steps": [1],
            "world_size": 1,
            "problems": []})
        self.assertEqual(export.training_receipt(run)["run_id"], run.name)
        corrupt = export.read_json(run / "run.json")
        corrupt["asset_binding"]["state"] = "UNBOUND"
        put_json(run / "run.json", corrupt)
        with self.assertRaisesRegex(export.ExportError, "asset binding"):
            export.training_receipt(run)
        corrupt["asset_binding"] = binding
        put_json(run / "run.json", corrupt)
        integrity = export.read_json(run / "train_integrity.json")
        integrity["asset_binding"]["run_record_sha256"] = export.file_hash(run / "run.json")
        put_json(run / "train_integrity.json", integrity)
        log.write_text("stale or changed", encoding="utf-8")
        with self.assertRaisesRegex(export.ExportError, "log"):
            export.training_receipt(run)
        log.write_text("iteration 1/1", encoding="utf-8")
        put_json(run / "run.json", {"run_id": run.name, "checkpoint_save":
                 {"enabled": False, "actual_format": None},
                 "source_config_sha256": export.file_hash(source),
                 "effective_config_sha256": export.file_hash(effective)})
        with self.assertRaisesRegex(export.ExportError, "save intent"):
            export.training_receipt(run)

    def test_selected_checkpoint_rejects_old_tracker_empty_and_missing_payload(self):
        root = scratch()
        save = root / "checkpoints"
        save.mkdir()
        train = {"expected_end": 100, "start": 1,
                 "unique_steps": list(range(1, 101)), "world_size": 1,
                 "config_snapshot_mtime_ns": 0, "integrity_mtime_ns": 2**64}
        with self.assertRaisesRegex(export.ExportError, "tracker"):
            export.selected_checkpoint(save, 100, train)
        (save / "latest_checkpointed_iteration.txt").write_text("99")
        with self.assertRaisesRegex(export.ExportError, "tracker"):
            export.selected_checkpoint(save, 100, train)
        (save / "latest_checkpointed_iteration.txt").write_text("100")
        with self.assertRaisesRegex(export.ExportError, "metadata"):
            export.selected_checkpoint(save, 100, train)
        with self.assertRaisesRegex(export.ExportError, "outside"):
            export.selected_checkpoint(save, 101, train)

    def test_dcp_keys_require_all_nonmtp_tensors_and_tied_head_shape(self):
        headers = {"model.language_model.embed_tokens.weight":
                   {"shape": [4, 2], "dtype": "BF16"}}
        for i in range(472):
            headers["model.layer.%03d.weight" % i] = {"shape": [2], "dtype": "BF16"}
        for i in range(15):
            headers["mtp.%02d.weight" % i] = {"shape": [2], "dtype": "BF16"}
        report = {"tensor_headers": headers}
        def tensor(shape):
            return types.SimpleNamespace(size=shape,
                properties=types.SimpleNamespace(dtype="torch.bfloat16"))
        states = {"model." + key: tensor(value["shape"]) for key, value in headers.items()
                  if not key.startswith("mtp.")}
        meta = types.SimpleNamespace(state_dict_metadata=states)
        kept, dtypes = export.expected_model_keys(report, meta)
        self.assertEqual((len(kept), len(dtypes)), (473, 473))
        del states["model.model.layer.000.weight"]
        with self.assertRaisesRegex(export.ExportError, "key mismatch"):
            export.expected_model_keys(report, meta)
        states["model.model.layer.000.weight"] = tensor([2])
        states["model.lm_head.weight"] = tensor([3, 2])
        with self.assertRaisesRegex(export.ExportError, "tied lm_head"):
            export.expected_model_keys(report, meta)

    def test_derived_index_removes_mtp_without_modifying_origin(self):
        root = scratch()
        origin = root / "origin"
        origin.mkdir()
        put_json(origin / "config.json", {"text_config": {"mtp_num_hidden_layers": 1}})
        (origin / "tokenizer.json").write_text("{}")
        index = {"metadata": {"total_size": 100},
                 "weight_map": {"a": "model-1.safetensors", "mtp.x": "model-1.safetensors"}}
        put_json(origin / "model.safetensors.index.json", index)
        before = export.file_hash(origin / "config.json")
        attempt = root / "attempt"
        attempt.mkdir()
        derived, evidence = export.derived_origin(origin, index, attempt,
                                                {"a": {"shape": [2], "dtype": "BF16"}},
                                                [{"path": "config.json"},
                                                 {"path": "tokenizer.json"}])
        self.assertEqual(export.file_hash(origin / "config.json"), before)
        self.assertEqual(export.read_json(derived / "config.json")["text_config"]["mtp_num_hidden_layers"], 0)
        self.assertEqual(export.read_json(derived / "model.safetensors.index.json")["weight_map"],
                         {"a": "model-1.safetensors"})
        self.assertEqual(export.read_json(derived / "model.safetensors.index.json")["metadata"]["total_size"], 4)
        self.assertTrue(evidence["sha256"])

    def test_export_rejects_processor_shard_mapping_and_tensor_mismatch(self):
        root = scratch()
        expected = {"a": {"shape": [2], "dtype": "BF16"}}
        files = ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
                 "tokenizer_config.json", "tokenizer.json")
        for name in files:
            put_json(root / name, {"text_config": {"mtp_num_hidden_layers": 0,
                                                   "mtp_num_layers": 0}}
                     if name == "config.json" else {"valid": True})
        put_json(root / "model.safetensors.index.json",
                 {"metadata": {"total_size": 4},
                  "weight_map": {"a": "model-1.safetensors"}})
        with self.assertRaisesRegex(export.ExportError, "shard set"):
            export.inspect_export(root, expected, {"a": "BF16"})
        shard(root / "model-1.safetensors", {"wrong": [2]})
        with self.assertRaisesRegex(export.ExportError, "misrouted"):
            export.inspect_export(root, expected, {"a": "BF16"})
        shard(root / "model-1.safetensors", {"a": [3]})
        with self.assertRaisesRegex(export.ExportError, "shape/dtype"):
            export.inspect_export(root, expected, {"a": "BF16"})
        shard(root / "model-1.safetensors", {"a": [2]})
        self.assertEqual(export.inspect_export(root, expected, {"a": "BF16"})["key_count"], 1)
        put_json(root / "model.safetensors.index.json",
                 {"metadata": {"total_size": 8},
                  "weight_map": {"a": "model-1.safetensors"}})
        with self.assertRaisesRegex(export.ExportError, "total_size"):
            export.inspect_export(root, expected, {"a": "BF16"})
        put_json(root / "model.safetensors.index.json",
                 {"metadata": {"total_size": 4},
                  "weight_map": {"a": "model-1.safetensors"}})
        (root / "tokenizer.json").unlink()
        with self.assertRaisesRegex(export.ExportError, "processor/config"):
            export.inspect_export(root, expected, {"a": "BF16"})

    def test_dcp_storage_reference_rejects_truncated_payload(self):
        from _asset_integrity import validate_dcp_storage_ranges
        root = scratch()
        (root / "part.distcp").write_bytes(b"abc")
        reference = {"item": types.SimpleNamespace(relative_path="part.distcp",
                    offset=1, length=4)}
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_dcp_storage_ranges(root, reference)

    def test_failed_cli_keeps_attempt_receipt(self):
        root = scratch()
        rc = cli.main(["--train-run-dir", str(root / "absent"),
                       "--assets-json", str(root / "absent.json"),
                       "--target-checkout", str(root / "absent-target"),
                       "--iteration", "1", "--out", str(root / "exports")])
        self.assertEqual(rc, 3)
        attempts = list((root / "exports").glob("attempt-*/model_artifact.json"))
        self.assertEqual(len(attempts), 1)
        record = export.read_json(attempts[0])
        self.assertEqual(record["status"], "FAILED")
        self.assertFalse(record["reload_verified"])

    def _control_flow_fixture(self, root):
        run = root / "runs" / "p5-current"
        run.mkdir(parents=True)
        source = root / "source.yaml"
        source.write_text("training: {train_iters: 1}", encoding="utf-8")
        effective = run / "effective_config.yaml"
        effective.write_text("training: {train_iters: 1, save: checkpoints}", encoding="utf-8")
        log = run / "train.log"
        log.write_text("iteration 1/1", encoding="utf-8")
        assets_json = root / "assets.json"
        put_json(assets_json, {"schema": "migrator_assets.v2", "attempt_id": "p4-now"})
        origin = root / "origin"
        origin.mkdir()
        put_json(origin / "config.json", {"text_config": {"mtp_num_hidden_layers": 1}})
        put_json(origin / "model.safetensors.index.json",
                 {"metadata": {"total_size": 2}, "weight_map": {"a": "a.safetensors"}})
        manifest = root / "manifest.json"
        put_json(manifest, {"migration_id": "migration-now"})
        target = root / "target"
        target.mkdir()
        bound = {"assets_json_path": str(assets_json),
                 "assets_json_sha256": export.file_hash(assets_json),
                 "p4_attempt_id": "p4-now", "migration_id": "migration-now",
                 "hf_file_inventory_sha256": "hf-id", "target_checkout": str(target),
                 "migration_bundle": str(root / "bundle"),
                 "migration_overlay": str(root / "overlay"),
                 "paths": {"weight_hf": str(origin)}}
        save = {"enabled": True, "actual_format": "dcp",
                "effective_path": str(run / "checkpoints")}
        put_json(run / "run.json", {"run_id": run.name, "log": str(log),
                "config_snapshot": str(effective), "checkpoint_save": save,
                "source_config_sha256": export.file_hash(source),
                "effective_config_sha256": export.file_hash(effective),
                "asset_binding": {"schema": "p5_asset_binding.v1", "state": "VERIFIED",
                    "problems": [], "prelaunch": bound, "postrun": bound}})
        put_json(run / "train_integrity.json", {
            "run_id": run.name, "state": "COMPLETE", "train_rc": 0,
            "failure_markers": [], "problems": [], "checkpoint_save": save,
            "source_config": str(source),
            "source_config_sha256": export.file_hash(source),
            "source_config_sha256_after_run": export.file_hash(source),
            "config": str(effective), "config_sha256": export.file_hash(effective),
            "config_snapshot_sha256_after_run": export.file_hash(effective),
            "config_train_iters": 1, "expected_start": 1, "expected_end": 1,
            "unique_steps": [1], "world_size": 1,
            "log": str(log), "log_sha256": export.file_hash(log),
            "asset_binding": {"schema": "p5_asset_binding.v1", "state": "VERIFIED",
                "problems": [], "run_record": str(run / "run.json"),
                "run_record_sha256": export.file_hash(run / "run.json"),
                "assets_json_sha256": export.file_hash(assets_json),
                "migration_id": "migration-now"}})
        assets = {"attempt_id": "p4-now", "migration_id": "migration-now",
                  "migration_manifest_path": str(manifest),
                  "migration_manifest_sha256": export.file_hash(manifest),
                  "conversion": {"target_commit": "fixed", "converter_sha256": "conv"}}
        report = {"model_revision": "revision", "config_sha256": "config-id",
                  "index_sha256": "index-id", "indexed_keys": 488,
                  "mtp_source_keys": ["mtp.%d" % n for n in range(15)]}
        origin_inventory = {"sha256": "hf-id", "files": [
            {"path": "config.json"}, {"path": "model.safetensors.index.json"}]}
        return run, assets_json, target, origin, bound, assets, report, origin_inventory, log

    def test_main_state_transitions_with_simulated_converter_and_reload(self):
        # These are control-flow simulations; they never prove a real model reload.
        for mode, wanted_rc, wanted_status in (
                ("exported", 0, "EXPORTED_RELOAD_UNVERIFIED"),
                ("verified", 0, "RELOAD_VERIFIED"),
                ("converter_rc", 3, "FAILED"),
                ("converter_error", 3, "FAILED"),
                ("train_drift", 3, "FAILED"),
                ("asset_drift", 3, "FAILED"),
                ("dcp_drift", 3, "FAILED")):
            with self.subTest(mode=mode):
                root = scratch()
                (run, assets_json, target, origin, bound, assets, report,
                 origin_inventory, log) = self._control_flow_fixture(root)
                dcp_inventory = {"files": [{"path": "iter_0000001/.metadata"}],
                                 "sha256": "dcp-id"}
                seen = {"capture": 0, "dcp": 0}

                def fake_capture(*_args):
                    seen["capture"] += 1
                    return dict(bound, assets_json_sha256="changed") if (
                        mode == "asset_drift" and seen["capture"] == 2) else bound

                def fake_selected(*_args):
                    seen["dcp"] += 1
                    got = dict(dcp_inventory, sha256="changed") if (
                        mode == "dcp_drift" and seen["dcp"] == 2) else dcp_inventory
                    return run / "checkpoints" / "iter_0000001", object(), got

                def fake_converter(argv, cwd, stdout, stderr, timeout, check):
                    stdout.write(b"simulated converter log\n")
                    output = Path(argv[argv.index("--save_hf_dir") + 1])
                    (output / "simulated.bin").write_bytes(b"exported")
                    if mode == "train_drift":
                        log.write_text("changed during export", encoding="utf-8")
                    if mode == "converter_error":
                        raise OSError("simulated converter launch failure")
                    return types.SimpleNamespace(returncode=7 if mode == "converter_rc" else 0)

                def fake_inspect(path, *_args):
                    return {"inventory": export.inventory(path, ["simulated.bin"]),
                            "key_count": 1, "config_sha256": "derived",
                            "index_sha256": "derived", "processor_files": [],
                            "storage_dtypes": {"BF16": 1}}

                target_id = {"target_commit": "fixed", "converter_current_sha256": "conv"}
                args = ["--train-run-dir", str(run), "--assets-json", str(assets_json),
                        "--target-checkout", str(target), "--iteration", "1",
                        "--out", str(root / "out")]
                if mode == "verified":
                    args.append("--verify-reload")
                with (mock.patch.object(cli, "source_assets", return_value=(
                        assets, origin, report, origin_inventory)),
                      mock.patch.object(runtime_binding, "capture", side_effect=fake_capture),
                      mock.patch.object(cli, "verify_target_checkout", return_value=target_id),
                      mock.patch.object(cli, "selected_checkpoint", side_effect=fake_selected),
                      mock.patch.object(cli, "expected_model_keys", return_value=(
                          {"a": {"shape": [1], "dtype": "BF16"}},
                          {"a": {"shape": [1], "dtype": "BF16"}})),
                      mock.patch.object(cli, "inspect_export", side_effect=fake_inspect),
                      mock.patch.object(cli, "verify_round_trip",
                                        return_value={"status": "verified"}),
                      mock.patch.object(cli.subprocess, "run", side_effect=fake_converter)):
                    rc = cli.main(args)
                self.assertEqual(rc, wanted_rc)
                receipts = list((root / "out").glob("attempt-*/model_artifact.json"))
                self.assertEqual(len(receipts), 1)
                result = export.read_json(receipts[0])
                self.assertEqual(result["status"], wanted_status)
                self.assertEqual(result["reload_verified"], mode == "verified")
                self.assertEqual(result["train_run"]["run_id"], run.name)
                converter_log = receipts[0].parent / "converter.log"
                self.assertTrue(converter_log.is_file())
                self.assertEqual(result["converter"]["log_sha256"],
                                 export.file_hash(converter_log))
                if wanted_status == "FAILED":
                    self.assertTrue(result["problems"])
                    self.assertTrue(result["partial_export_files"])

    def test_nonfinite_tolerance_is_rejected_before_attempt(self):
        root = scratch()
        args = ["--train-run-dir", "unused", "--assets-json", "unused",
                "--target-checkout", "unused", "--iteration", "1",
                "--out", str(root / "out")]
        for option, value in (("--atol", "nan"), ("--rtol", "inf"),
                              ("--atol", "-inf")):
            with (self.subTest(option=option, value=value),
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as caught:
                    cli.main(args + [option, value])
                self.assertEqual(caught.exception.code, 2)
        self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
