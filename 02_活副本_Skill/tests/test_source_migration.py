"""T03 fixed-source replay tests; requires QWEN35_REAL_BUNDLE from `22 ... fetch`."""

import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("migrate22", SCRIPTS / "22_migrate_qwen35.py")
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)
from _qwen35_weights import EMBED_KEY, WeightContractError, weight_headers_contract  # noqa: E402


@unittest.skipUnless(os.environ.get("QWEN35_REAL_BUNDLE"), "fetch pinned public source bundle first")
class FixedSourceMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ["QWEN35_REAL_BUNDLE"])
        migrate.validate_bundle(cls.bundle)

    def setUp(self):
        root = ROOT.parent / "tmp" / "migration_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="t03-", dir=root))

    def test_real_fixed_source_replay_and_idempotence(self):
        out = self.work / "out"
        migrate.apply(self.bundle, out)
        first = (out / "migration_manifest.json").read_bytes()
        migrate.apply(self.bundle, out)
        self.assertEqual(first, (out / "migration_manifest.json").read_bytes())
        manifest = json.loads(first)
        self.assertEqual(manifest["source"]["commit"], migrate.GPU_COMMIT)
        self.assertEqual(manifest["target"]["commit"], migrate.TARGET_COMMIT)
        self.assertEqual(manifest["reference_training_policy"]["mtp_num_layers"], 0)
        self.assertEqual(len(manifest["model_metadata"]["mtp_source_keys"]), 15)
        self.assertEqual(manifest["target"]["source_files_sha256"], migrate.TARGET_FILES_SHA256)
        self.assertEqual(manifest["execution_state"], "STATIC_APPLIED_RUNTIME_PENDING")
        self.assertFalse(manifest["complete_weights_verified"])
        self.assertIn("missing tied source weight", (out / "overlay" / migrate.CONVERTER).read_text(encoding="utf-8"))
        self.assertTrue((out / "upstream_model_diff.patch").stat().st_size > 0)

    def test_wrong_fixed_source_rejected_before_output(self):
        bundle = self.work / "tampered_bundle"
        shutil.copytree(self.bundle, bundle)
        path = bundle / "gpu/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
        path.write_bytes(path.read_bytes() + b"\n# tampered\n")
        out = self.work / "out"
        with self.assertRaises(migrate.MigrationError):
            migrate.apply(bundle, out)
        self.assertFalse((out / "migration_manifest.json").exists())

    def test_new_target_dependency_tamper_rejected(self):
        bundle = self.work / "tampered_bundle"
        shutil.copytree(self.bundle, bundle)
        path = bundle / "target/mindspeed_mm/fsdp/utils/register.py"
        path.write_bytes(path.read_bytes() + b"\n# tampered\n")
        with self.assertRaisesRegex(migrate.MigrationError, "source or target identity mismatch"):
            migrate.apply(bundle, self.work / "out")

    def test_tampered_overlay_and_manifest_rejected(self):
        out = self.work / "out"
        migrate.apply(self.bundle, out)
        manifest_path = out / "migration_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["target"]["converter_after_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(migrate.MigrationError):
            migrate.validate_overlay(self.bundle, out)

    def test_existing_dcp_without_source_receipt_is_rejected(self):
        spec = importlib.util.spec_from_file_location("prepare40", SCRIPTS / "40_prepare_assets.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        dcp = self.work / "dcp"
        (dcp / "release").mkdir(parents=True)
        (dcp / "latest_checkpointed_iteration.txt").write_text("release", encoding="utf-8")
        (dcp / "release" / ".metadata").write_bytes(b"synthetic metadata")
        self.assertTrue(prepare._dcp_release_ready(dcp))
        self.assertFalse(prepare._existing_conversion_verified(dcp,
                         {"model_revision": "fixture"}, {"target_commit": "fixture"}))

    def _synthetic_headers(self, omit=None, bad_embed=False):
        """Header-only fixture from the real pinned index; no model weights."""
        model = self.work / "model"
        model.mkdir()
        for name in ("config.json", "model.safetensors.index.json",
                     "preprocessor_config.json", "tokenizer_config.json"):
            shutil.copyfile(self.bundle / "model" / name, model / name)
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        header = {}
        offset = 0
        for key in index["weight_map"]:
            if key == omit:
                continue
            shape = [1]
            if key == EMBED_KEY:
                shape = [1, 1024] if bad_embed else [1]
            width = 2 * shape[0] * (shape[1] if len(shape) > 1 else 1)
            header[key] = {"dtype": "BF16", "shape": shape,
                           "data_offsets": [offset, offset + width]}
            offset += width
        raw = json.dumps(header).encode("utf-8")
        shard = next(iter(set(index["weight_map"].values())))
        (model / shard).write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * offset)
        return model

    def test_missing_tied_source_header_fails(self):
        model = self._synthetic_headers(omit=EMBED_KEY)
        with self.assertRaisesRegex(WeightContractError, "missing indexed tensors"):
            weight_headers_contract(model)

    def test_wrong_critical_shape_fails(self):
        model = self._synthetic_headers(bad_embed=True)
        with self.assertRaisesRegex(WeightContractError, "critical tensor shape mismatch"):
            weight_headers_contract(model)

    def test_invalid_tensor_span_fails_before_conversion(self):
        model = self._synthetic_headers()
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard = model / next(iter(set(index["weight_map"].values())))
        data = shard.read_bytes()
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + size])
        key = next(iter(header))
        header[key]["data_offsets"][1] += 1
        raw = json.dumps(header).encode("utf-8")
        shard.write_bytes(len(raw).to_bytes(8, "little") + raw + data[8 + size:])
        with self.assertRaisesRegex(WeightContractError, "offsets invalid"):
            weight_headers_contract(model)

    def test_dtype_byte_width_must_match_offsets(self):
        model = self._synthetic_headers()
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard = model / next(iter(set(index["weight_map"].values())))
        data = shard.read_bytes()
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + size])
        header[next(iter(header))]["dtype"] = "F32"
        raw = json.dumps(header).encode("utf-8")
        shard.write_bytes(len(raw).to_bytes(8, "little") + raw + data[8 + size:])
        with self.assertRaisesRegex(WeightContractError, "offsets invalid"):
            weight_headers_contract(model)

    def test_overlapping_tensor_offsets_fail(self):
        model = self._synthetic_headers()
        index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard = model / next(iter(set(index["weight_map"].values())))
        data = shard.read_bytes()
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + size])
        first, second = list(header)[:2]
        header[second]["data_offsets"] = header[first]["data_offsets"][:]
        raw = json.dumps(header).encode("utf-8")
        shard.write_bytes(len(raw).to_bytes(8, "little") + raw + data[8 + size:])
        with self.assertRaisesRegex(WeightContractError, "overlapping tensor offsets"):
            weight_headers_contract(model)


class MainExitCodeTests(unittest.TestCase):
    def test_new_target_registry_decorator_returns_class(self):
        module_name = "pinned_target.modeling_qwen3_5"
        model_module = types.ModuleType(module_name)
        model_cls = type("Qwen3_5ForConditionalGeneration", (), {"__module__": module_name})

        class Register:
            def get(self, key):
                if key != "qwen3_5":
                    raise KeyError(key)
                return model_cls

        model_module.Qwen3_5ForConditionalGeneration = model_cls
        with mock.patch.dict(sys.modules, {module_name: model_module}):
            self.assertIs(migrate._registered_model_class(model_module, Register()), model_cls)

    def test_registry_class_used_when_decorator_erases_module_attribute(self):
        module_name = "pinned_target.modeling_qwen3_5"
        model_module = types.ModuleType(module_name)

        class Register:
            def __init__(self):
                self._registry = {}

            def register(self, key):
                def decorator(cls):
                    self._registry[key] = cls
                return decorator

            def get(self, key):
                return self._registry[key]

        register = Register()
        model_cls = type("Qwen3_5ForConditionalGeneration", (), {"__module__": module_name})
        model_module.Qwen3_5ForConditionalGeneration = register.register("qwen3_5")(model_cls)
        self.assertIsNone(model_module.Qwen3_5ForConditionalGeneration)
        with mock.patch.dict(sys.modules, {module_name: model_module}):
            self.assertIs(migrate._registered_model_class(model_module, register), model_cls)

    def test_model_runtime_error_is_failure_not_missing_dependency(self):
        stderr = io.StringIO()
        with mock.patch.object(migrate, "runtime", side_effect=RuntimeError("model construction failed")):
            with redirect_stderr(stderr):
                rc = migrate.main(["runtime", "--bundle", "x", "--overlay", "y",
                                   "--target-checkout", "z", "--out", "w"])
        self.assertEqual(rc, 3)
        self.assertIn("MIGRATION_FAILED", stderr.getvalue())
        self.assertNotIn("MIGRATION_RUNTIME_UNAVAILABLE", stderr.getvalue())

    def test_confirmed_missing_import_uses_unavailable_code(self):
        stderr = io.StringIO()
        with mock.patch.object(migrate, "runtime", side_effect=migrate.DependencyUnavailable("accelerate missing")):
            with redirect_stderr(stderr):
                rc = migrate.main(["runtime", "--bundle", "x", "--overlay", "y",
                                   "--target-checkout", "z", "--out", "w"])
        self.assertEqual(rc, 4)
        self.assertIn("MIGRATION_RUNTIME_UNAVAILABLE", stderr.getvalue())

    def test_import_from_another_checkout_rejected(self):
        foreign = type("Module", (), {"__file__": "D:/other/mindspeed_mm/modeling_qwen3_5.py"})()
        with self.assertRaisesRegex(migrate.MigrationError, "outside pinned target checkout"):
            migrate._checked_module_path(foreign, "D:/expected/mindspeed_mm/modeling_qwen3_5.py")


if __name__ == "__main__":
    unittest.main()
