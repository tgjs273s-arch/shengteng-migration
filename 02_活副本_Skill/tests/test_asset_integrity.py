"""Synthetic P4 asset failures; no network, install, conversion or NPU."""

import importlib.util
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BASE = ROOT.parent / "tmp" / "asset_integrity_tests"
sys.path.insert(0, str(SCRIPTS))


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = importlib.import_module("_asset_integrity")
prepare = load_script("40_prepare_assets.py")


class AssetIntegrityTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="asset-", dir=BASE))
        degradation_path = mock.patch.object(prepare, "DEG_FILE", str(self.root / "degradations.json"))
        degradation_path.start()
        self.addCleanup(degradation_path.stop)

    def fake_hf(self):
        hf = self.root / "Qwen3.5-0.8B-hf"
        hf.mkdir()
        for name in assets.REQUIRED_HF_FILES:
            (hf / name).write_text("{}", encoding="utf-8")
        (hf / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
        (hf / "model-00001-of-00001.safetensors").write_bytes(b"abc")
        report = {"shards": ["model-00001-of-00001.safetensors"], "model_revision": "fixture",
                  "config_sha256": "config", "index_sha256": "index",
                  "header_keys": 488, "indexed_keys": 488, "mtp_source_keys": []}
        return hf, report

    def fake_dcp(self):
        dcp = self.root / "Qwen3.5-0.8B-dcp"
        (dcp / "release").mkdir(parents=True)
        (dcp / "latest_checkpointed_iteration.txt").write_text("release", encoding="utf-8")
        (dcp / "release" / ".metadata").write_bytes(b"synthetic metadata")
        return dcp

    def fake_json(self):
        coco = self.root / "coco"
        image = coco / "train2017" / "a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image1")
        data = self.root / "converted.json"
        rows = [{"images": ["./train2017/a.jpg"],
                 "messages": [{"role": "user", "content": "x"},
                              {"role": "assistant", "content": "y"}]}]
        data.write_text(json.dumps(rows), encoding="utf-8")
        return data, coco, image, rows

    def test_missing_processor_or_tokenizer_never_ready(self):
        hf, report = self.fake_hf()
        (hf / "preprocessor_config.json").unlink()
        self.assertIn("preprocessor_config.json", prepare.weights_missing_files(hf))
        self.assertFalse(prepare._weights_ready(hf))
        self.assertEqual(assets.inspect_hf(hf)["status"], "incomplete")
        (hf / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (hf / "tokenizer.json").unlink()
        self.assertIn("tokenizer.json", assets.inspect_hf(hf)["missing_files"])

    def test_bad_index_shape_or_missing_shard_propagates_strict_contract(self):
        hf, report = self.fake_hf()
        with mock.patch.object(assets, "metadata_contract", return_value=(report, {}, {})), \
             mock.patch.object(assets, "weight_headers_contract", side_effect=assets.WeightContractError("bad index")):
            self.assertIn("bad index", assets.inspect_hf(hf)["error"])
        with mock.patch.object(assets, "metadata_contract", return_value=(report, {}, {})), \
             mock.patch.object(assets, "weight_headers_contract", side_effect=assets.WeightContractError("critical tensor shape mismatch")):
            self.assertIn("shape mismatch", assets.inspect_hf(hf)["error"])
        (hf / "model-00001-of-00001.safetensors").unlink()
        with mock.patch.object(assets, "metadata_contract", return_value=(report, {}, {})), \
             mock.patch.object(assets, "weight_headers_contract", return_value=report):
            self.assertIn("model-00001-of-00001.safetensors", assets.inspect_hf(hf)["missing_files"])

    def test_same_size_hf_payload_change_changes_local_identity(self):
        hf, report = self.fake_hf()
        with mock.patch.object(assets, "metadata_contract", return_value=(report, {}, {})), \
             mock.patch.object(assets, "weight_headers_contract", return_value=report):
            before = assets.inspect_hf(hf)
            (hf / "model-00001-of-00001.safetensors").write_bytes(b"xyz")
            after = assets.inspect_hf(hf)
        self.assertNotEqual(before["file_inventory_sha256"], after["file_inventory_sha256"])
        self.assertEqual(after["upstream_payload_identity"], "unverified")

    def test_empty_dcp_and_receipt_payload_mismatch(self):
        dcp = self.fake_dcp()
        self.assertNotEqual(assets.inspect_dcp(dcp)["status"], "local_structure_and_hashes")
        payload = dcp / "release" / "__0_0.distcp"
        payload.write_bytes(b"abc")
        source = {"model_revision": "fixture", "config_sha256": "cfg", "index_sha256": "idx",
                  "file_inventory_sha256": "source-local", "mtp_source_keys": []}
        target = {"target_commit": "target", "converter_current_sha256": "converter",
                  "patch_identity": "patch", "target_files_sha256": {}}
        ranges = {"entry": types.SimpleNamespace(relative_path="__0_0.distcp", offset=0, length=3)}
        with mock.patch.object(assets, "_read_dcp_storage_data", return_value=ranges):
            receipt = prepare._conversion_receipt(dcp, source, target)
            (dcp / "migration_conversion_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            self.assertTrue(prepare._existing_conversion_verified(dcp, source, target))
            self.assertEqual(receipt["tie_weight_mapping"], prepare.TIE_MAPPING)
            tampered = dict(receipt, tie_weight_mapping={})
            (dcp / "migration_conversion_receipt.json").write_text(json.dumps(tampered), encoding="utf-8")
            self.assertFalse(prepare._existing_conversion_verified(dcp, source, target))
            (dcp / "migration_conversion_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            payload.write_bytes(b"xyz")
            self.assertFalse(prepare._existing_conversion_verified(dcp, source, target))

    def test_dcp_metadata_references_reject_missing_escape_and_truncation(self):
        dcp = self.fake_dcp()
        release = dcp / "release"
        payload = release / "__0_0.distcp"
        payload.write_bytes(b"abc")
        info = lambda name, offset, length: {"x": types.SimpleNamespace(
            relative_path=name, offset=offset, length=length)}
        self.assertEqual(assets.validate_dcp_storage_ranges(release, info("__0_0.distcp", 0, 3)),
                         {payload.resolve()})
        for rows in (info("missing.distcp", 0, 1), info("../outside.distcp", 0, 1),
                     info("__0_0.distcp", 1, 3), {}):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                assets.validate_dcp_storage_ranges(release, rows)
        with mock.patch.object(assets, "_read_dcp_storage_data",
                               side_effect=assets.DCPReaderUnavailable("torch absent")):
            self.assertEqual(assets.inspect_dcp(dcp)["status"], "metadata_unverified")
        with mock.patch.object(assets, "_read_dcp_storage_data",
                               side_effect=OSError("corrupt metadata")):
            self.assertEqual(assets.inspect_dcp(dcp)["status"], "incomplete")

    def test_data_missing_image_wrong_schema_content_and_order(self):
        data, coco, image, rows = self.fake_json()
        text_only = {"images": [], "messages": [{"role": "user", "content": "text only"}]}
        with mock.patch.object(assets, "_decode_image", return_value={"status": "verified", "format": "fixture"}):
            first = assets.inspect_data(data, coco)
            self.assertEqual(first["status"], "local_content_verified")
            self.assertEqual(first["official_same_bytes"], "unverified")
            self.assertEqual(first["image_inventory"][0]["path"], "train2017/a.jpg")
            image.write_bytes(b"image2")
            second = assets.inspect_data(data, coco)
            self.assertNotEqual(first["referenced_images_sha256"], second["referenced_images_sha256"])
            image.unlink()
            self.assertEqual(assets.inspect_data(data, coco)["status"], "incomplete")
            image.write_bytes(b"image2")
            data.write_text('{"images": []}', encoding="utf-8")
            self.assertEqual(assets.inspect_data(data, coco)["status"], "incomplete")
            data.write_text(json.dumps(rows + [text_only]), encoding="utf-8")
            ordered = assets.inspect_data(data, coco)
            data.write_text(json.dumps([text_only] + rows), encoding="utf-8")
            reordered = assets.inspect_data(data, coco)
        self.assertNotEqual(ordered["order_sha256"], reordered["order_sha256"])

    def test_bad_image_and_missing_decoder_cannot_claim_local_training_ready(self):
        data, coco, image, rows = self.fake_json()
        image.write_bytes(b"this is not a JPEG")
        observed = assets.inspect_data(data, coco)
        self.assertNotEqual(observed["status"], "local_content_verified")
        if observed["status"] == "decode_unverified":
            self.assertFalse(observed["image_decode_verified"])
        else:
            self.assertIn("invalid image", observed["error"])
        with mock.patch.object(assets, "_decode_image", return_value={"status": "unverified", "reason": "Pillow unavailable"}):
            unverified = assets.inspect_data(data, coco)
        self.assertEqual(unverified["status"], "decode_unverified")
        self.assertTrue(unverified["image_bytes_readable"])
        self.assertFalse(unverified["image_decode_verified"])

    def test_empty_dataset_and_empty_messages_are_invalid(self):
        data, coco, image, rows = self.fake_json()
        data.write_text("[]", encoding="utf-8")
        self.assertEqual(assets.inspect_data(data, coco)["status"], "incomplete")
        data.write_text(json.dumps([{"images": [], "messages": []}]), encoding="utf-8")
        self.assertEqual(assets.inspect_data(data, coco)["status"], "incomplete")
        llava = self.root / "llava.json"
        llava.write_text("[]", encoding="utf-8")
        self.assertEqual(assets.inspect_llava(llava)["status"], "incomplete")
        llava.write_text(json.dumps([{"conversations": []}]), encoding="utf-8")
        self.assertEqual(assets.inspect_llava(llava)["status"], "incomplete")

    def test_no_download_does_not_call_network_or_install(self):
        output = self.root / "out"
        argv = ["40_prepare_assets.py", "--no-download", "--model-dir", str(self.root),
                "--data-dir", str(self.root), "--msmm-dir", str(self.root), "--out", str(output)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(prepare, "_ms_download", side_effect=AssertionError("network")), \
             mock.patch.object(prepare, "pip_install", side_effect=AssertionError("install")), \
             mock.patch.object(prepare, "fetch_model", side_effect=AssertionError("download")), \
             mock.patch.object(prepare, "fetch_llava", side_effect=AssertionError("download")):
            self.assertEqual(prepare.main(), 3)
        result = json.loads((output / "assets.json").read_text(encoding="utf-8"))
        self.assertEqual(result["readiness"], "incomplete")
        self.assertTrue(result["missing_required"])

    def test_model_download_uses_fixed_hf_repo_commit_without_floating_fallback(self):
        target = str(self.root / "hf")
        with mock.patch.object(prepare, "_hf_snapshot_download", return_value=target) as download, \
             mock.patch.object(prepare, "_weights_ready", return_value=True), \
             mock.patch.object(prepare, "_ms_download", side_effect=AssertionError("mirror fallback")), \
             mock.patch.object(prepare, "pip_install", side_effect=AssertionError("install")):
            self.assertTrue(prepare.fetch_model(target))
        download.assert_called_once_with(repo_id="Qwen/Qwen3.5-0.8B",
                                         revision=prepare.MODEL_REVISION, local_dir=target)
        with mock.patch.object(prepare, "_hf_snapshot_download", side_effect=ImportError("hub absent")) as download, \
             mock.patch.object(prepare, "_ms_download", side_effect=AssertionError("mirror fallback")), \
             mock.patch.object(prepare, "pip_install", side_effect=AssertionError("install")):
            self.assertFalse(prepare.fetch_model(target))
        download.assert_called_once()

    def test_unpinned_converter_script_is_not_accepted(self):
        path = self.root / assets.CONVERTER_RELATIVE
        path.parent.mkdir(parents=True)
        path.write_text("# different conversion", encoding="utf-8")
        identity = assets.converter_identity(self.root)
        self.assertEqual(identity["status"], "mismatch")
        self.assertEqual(identity["target_commit"], assets.CONVERTER_COMMIT)

    def test_llava_same_size_content_change_and_bad_conversation(self):
        source = self.root / "llava.json"
        rows = [{"image": "a.jpg", "conversations": [{"from": "human", "value": "a"}]}]
        source.write_text(json.dumps(rows), encoding="utf-8")
        before = assets.inspect_llava(source)
        rows[0]["conversations"][0]["value"] = "b"
        source.write_text(json.dumps(rows), encoding="utf-8")
        after = assets.inspect_llava(source)
        self.assertNotEqual(before["json_sha256"], after["json_sha256"])
        self.assertEqual(after["official_same_bytes"], "unverified")
        rows[0]["conversations"][0]["from"] = "unknown"
        source.write_text(json.dumps(rows), encoding="utf-8")
        self.assertEqual(assets.inspect_llava(source)["status"], "incomplete")

    def test_invalid_migration_bundle_fails_before_asset_operations(self):
        output = self.root / "out"
        argv = ["40_prepare_assets.py", "--no-download", "--model-dir", str(self.root),
                "--data-dir", str(self.root), "--msmm-dir", str(self.root), "--out", str(output),
                "--migration-bundle", str(self.root / "fake-bundle"),
                "--migration-overlay", str(self.root / "fake-overlay")]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(prepare, "_ms_download", side_effect=AssertionError("network")):
            self.assertEqual(prepare.main(), 3)
        result = json.loads((output / "assets.json").read_text(encoding="utf-8"))
        self.assertEqual(result["migration_identity_state"], "invalid")

    def test_validated_overlay_must_match_installed_target(self):
        fake = types.SimpleNamespace(validate_overlay=lambda bundle, overlay: (
            {"migration_id": "fixture", "target": {"commit": "wrong",
             "converter_after_sha256": "converter"}}, b""))
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        with mock.patch.object(prepare.importlib.util, "spec_from_file_location", return_value=spec), \
             mock.patch.object(prepare.importlib.util, "module_from_spec", return_value=fake), \
             mock.patch.object(prepare, "verify_target_checkout", return_value={
                 "target_commit": "expected", "converter_current_sha256": "converter"}):
            with self.assertRaisesRegex(ValueError, "differs from installed target"):
                prepare.verified_migration(self.root, self.root, self.root)


if __name__ == "__main__":
    unittest.main()
