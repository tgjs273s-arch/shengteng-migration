"""Offline contracts only; a fake inference never proves NPU execution."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from _scene_inference import SceneError, canonical_hash, file_inventory, parse_scene_json, sha256, verify_artifact
import _scene_run


GOOD = '{"compliant":"否","step":"紧固","issue":"顺序异常","evidence":"画面可见",' \
       '"bbox":[1,2,10,12],"confidence":0.8}'
UNKNOWN = '{"compliant":"无法判断","step":"未识别","issue":"画面遮挡",' \
          '"evidence":"关键操作不可见","bbox":null,"confidence":0.2}'


class SceneContractTests(unittest.TestCase):
    def setUp(self):
        evidence_root = Path(__file__).resolve().parents[2] / "tmp" / "t09-scene-tests"
        evidence_root.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="case-", dir=evidence_root))
        self.attempt = self.root / "source-attempt"
        self.export = self.attempt / "export_hf"
        self.export.mkdir(parents=True)
        for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
                     "tokenizer_config.json", "tokenizer.json", "model-00001.safetensors"):
            (self.export / name).write_text("fixture-" + name, encoding="utf-8")
        evidence = self.attempt / "round_trip"
        evidence.mkdir(parents=True)
        (evidence / "input.json").write_text('{"text":"fixture"}', encoding="utf-8")
        for name in ("dcp_logits.f32le", "export_logits.f32le"):
            (evidence / name).write_bytes(b"fixture-logits")
        self.inventory = file_inventory(self.export)
        self.run_dir = self.root / "p5-fixture"
        selected = self.run_dir / "checkpoints" / "iter_0000100"
        selected.mkdir(parents=True)
        (selected.parent / "latest_checkpointed_iteration.txt").write_text("100", encoding="utf-8")
        (selected / ".metadata").write_bytes(b"metadata")
        self.checkpoint_inventory = file_inventory(selected.parent)
        (self.run_dir / "run.json").write_text("{}", encoding="utf-8")
        (self.run_dir / "train_integrity.json").write_text("{}", encoding="utf-8")
        self.assets = self.root / "assets.json"
        self.assets.write_text("{}", encoding="utf-8")
        self.migration = self.root / "migration_manifest.json"
        self.migration.write_text("{}", encoding="utf-8")
        comparison = {"status": "verified", "device": "cpu",
            "hf_model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration",
            "checkpoint_inventory_sha256": self.checkpoint_inventory["sha256"],
            "input_sha256": sha256(evidence / "input.json")}
        for key, name in (("dcp_logits", "dcp_logits.f32le"),
                          ("export_logits", "export_logits.f32le")):
            comparison[key] = {"path": str(evidence / name), "sha256": sha256(evidence / name)}
        (evidence / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
        identity = {"migration_id": "migration", "train_run_id": "p5-fixture",
                    "checkpoint_inventory_sha256": self.checkpoint_inventory["sha256"],
                    "export_inventory_sha256": self.inventory["sha256"]}
        self.manifest = {"schema": "model_artifact.v1", "status": "RELOAD_VERIFIED",
            "reload_verified": True, "post_export_identity_rechecked": True,
            "problems": [], "migration_id": "migration",
            "train_run": {"run_id": "p5-fixture", "run_json_sha256": sha256(self.run_dir / "run.json"),
                          "train_integrity_sha256": sha256(self.run_dir / "train_integrity.json")},
            "checkpoint": {"path": str(selected), "format": "dcp", "iteration": 100,
                           "inventory": self.checkpoint_inventory},
            "assets_json": {"path": str(self.assets), "sha256": sha256(self.assets)},
            "migration_manifest": {"path": str(self.migration), "sha256": sha256(self.migration)},
            "export_hf": {"path": str(self.export), "inventory": self.inventory},
            "round_trip": comparison,
            "model_artifact_id": "qwen35-0p8b-" + canonical_hash(identity)[:20]}
        self.artifact_path = self.attempt / "model_artifact.json"
        self.write_artifact()
        self.image = self.root / "frame.png"
        self.image.write_bytes(b"fixture-image")
        self.sop = self.root / "sop.txt"
        self.sop.write_text("按对角顺序紧固", encoding="utf-8")

    def write_artifact(self):
        self.artifact_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def args(self, *extra):
        return ["--out", str(self.root / "runs"), "--model-artifact", str(self.artifact_path),
                "--model-dir", str(self.export), "--processor-dir", str(self.export),
                "--image", str(self.image), "--sop", str(self.sop), *extra]

    def result(self):
        attempts = list((self.root / "runs").glob("attempt-*/inference_result.json"))
        self.assertEqual(len(attempts), 1)
        return json.loads(attempts[0].read_text(encoding="utf-8"))

    def fake_infer(self, raw, save=True):
        def run(_args, receipt, _image, _sop):
            folder = Path(receipt["attempt_dir"])
            if save:
                ids = folder / "generated_ids.json"
                ids.write_text('{"new_token_ids":[9]}', encoding="utf-8")
                output = folder / "raw_generation.txt"
                output.write_text(raw, encoding="utf-8")
                receipt["generated_ids"] = {"path": str(ids), "sha256": sha256(ids)}
                receipt["raw_generation"] = {"path": str(output), "sha256": sha256(output)}
            return raw, 20, 20
        return run

    def test_artifact_rejects_incomplete_and_unlisted_file(self):
        verify_artifact(self.artifact_path, self.export, self.export)
        self.manifest["reload_verified"] = False
        self.write_artifact()
        with self.assertRaises(SceneError):
            verify_artifact(self.artifact_path, self.export, self.export)
        self.manifest["reload_verified"] = True
        self.write_artifact()
        (self.export / "extra.bin").write_bytes(b"unlisted")
        with self.assertRaises(SceneError):
            verify_artifact(self.artifact_path, self.export, self.export)

    def test_artifact_rejects_wrong_id_and_evidence_change(self):
        self.manifest["model_artifact_id"] = "wrong"
        self.write_artifact()
        with self.assertRaises(SceneError):
            verify_artifact(self.artifact_path, self.export, self.export)
        self.manifest["model_artifact_id"] = "qwen35-0p8b-" + canonical_hash({
            "migration_id": "migration", "train_run_id": "p5-fixture",
            "checkpoint_inventory_sha256": self.checkpoint_inventory["sha256"],
            "export_inventory_sha256": self.inventory["sha256"]})[:20]
        self.write_artifact()
        (self.attempt / "round_trip" / "export_logits.f32le").write_bytes(b"changed")
        with self.assertRaises(SceneError):
            verify_artifact(self.artifact_path, self.export, self.export)

    def test_scene_json_contract(self):
        self.assertEqual(parse_scene_json(GOOD, 20, 20)["step"], "紧固")
        self.assertEqual(parse_scene_json(UNKNOWN, 20, 20)["compliant"], "无法判断")
        for raw in ("prefix" + GOOD, GOOD.replace("10,12", "30,12"),
                    GOOD.replace('"confidence":0.8', '"confidence":true'),
                    UNKNOWN.replace("null", "[1,2,10,12]")):
            with self.assertRaises(SceneError):
                parse_scene_json(raw, 20, 20)

    def test_npu_index_and_import_are_checked(self):
        current = SimpleNamespace(value=0)
        npu = SimpleNamespace(is_available=lambda: True,
            current_device=lambda: current.value,
            set_device=lambda device: setattr(current, "value", device.index))
        torch = SimpleNamespace(npu=npu,
            device=lambda name: SimpleNamespace(type="npu", index=int(name.split(":")[1])))
        with patch.object(_scene_run.importlib, "import_module",
                          return_value=SimpleNamespace(__version__="fixture")) as imported:
            device, version = _scene_run.checked_device(torch, "npu:1", False)
        imported.assert_called_once_with("torch_npu")
        self.assertEqual((device.index, version), (1, "fixture"))
        wrong = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), is_meta=False)
        with self.assertRaises(SceneError):
            _scene_run.checked_tensor_devices([wrong], device)
        with patch.object(_scene_run.importlib, "import_module", side_effect=ImportError("absent")):
            with self.assertRaisesRegex(SceneError, "torch_npu"):
                _scene_run.checked_device(torch, "npu:1", False)

    def test_decode_excludes_prompt_and_rejects_short_generation(self):
        class Row:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.values
        class Generated:
            ndim = 2
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))
            def __getitem__(self, index):
                return Row(self.values)
        processor = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["decoded-new"])
        with patch.object(processor, "batch_decode", wraps=processor.batch_decode) as decode:
            all_ids, new_ids, raw = _scene_run.decode_generated(
                processor, Generated([101, 102, 103, 901, 902]), 3)
        self.assertEqual((all_ids, new_ids, raw),
                         ([101, 102, 103, 901, 902], [901, 902], "decoded-new"))
        decode.assert_called_once_with([[901, 902]], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        with self.assertRaises(SceneError):
            _scene_run.decode_generated(processor, Generated([101, 102]), 3)
        with self.assertRaises(SceneError):
            _scene_run.decode_generated(processor, Generated([101, 102, 103]), 3)

    def test_cpu_diagnostic_and_npu_result_are_distinct(self):
        with patch.object(_scene_run, "infer", side_effect=self.fake_infer(GOOD)):
            self.assertEqual(_scene_run.main(self.args("--device", "cpu", "--cpu-diagnostic")), 0)
        result = self.result()
        self.assertEqual(result["status"], "EXECUTED_PARSED")
        self.assertEqual(result["execution_state"], "CPU_DIAGNOSTIC")
        self.assertFalse(result["real_inference_verified"])

    def test_evidence_insufficient_is_separate_from_parse_failure(self):
        with patch.object(_scene_run, "infer", side_effect=self.fake_infer(UNKNOWN)):
            self.assertEqual(_scene_run.main(self.args()), 0)
        result = self.result()
        self.assertEqual(result["status"], "EVIDENCE_INSUFFICIENT")
        self.assertEqual(result["business_validity"], "NOT_EVALUATED")

    def test_invalid_json_keeps_raw_generation(self):
        with patch.object(_scene_run, "infer", side_effect=self.fake_infer("prompt echo {bad}")):
            self.assertEqual(_scene_run.main(self.args()), 3)
        result = self.result()
        self.assertEqual(result["status"], "PARSE_FAILED")
        self.assertTrue(Path(result["raw_generation"]["path"]).is_file())

    def test_missing_raw_output_is_failure(self):
        with patch.object(_scene_run, "infer", side_effect=self.fake_infer(GOOD, save=False)):
            self.assertEqual(_scene_run.main(self.args()), 3)
        self.assertEqual(self.result()["status"], "FAILED")

    def test_missing_media_sop_and_wrong_model_leave_receipt(self):
        self.image.unlink()
        self.assertEqual(_scene_run.main(self.args()), 3)
        self.assertEqual(self.result()["status"], "FAILED")

    def test_missing_sop_is_recorded(self):
        self.sop.unlink()
        self.assertEqual(_scene_run.main(self.args()), 3)
        self.assertEqual(self.result()["status"], "FAILED")

    def test_cpu_fallback_without_diagnostic_flag_is_rejected(self):
        with patch.object(_scene_run, "infer") as mock_infer:
            self.assertEqual(_scene_run.main(self.args("--device", "cpu")), 3)
            mock_infer.assert_not_called()
        self.assertFalse(self.result()["real_inference_verified"])

    def test_wrong_model_path_is_rejected_before_generation(self):
        other = self.root / "other-model"
        other.mkdir()
        args = self.args()
        args[args.index("--model-dir") + 1] = str(other)
        with patch.object(_scene_run, "infer") as mock_infer:
            self.assertEqual(_scene_run.main(args), 3)
            mock_infer.assert_not_called()
        self.assertEqual(self.result()["status"], "FAILED")

    def test_legacy_positional_image_keeps_failure_receipt(self):
        self.assertEqual(_scene_run.main([str(self.image), "--out", str(self.root / "runs")]), 3)
        self.assertEqual(self.result()["status"], "FAILED")
        with self.assertRaises(SystemExit) as stopped:
            _scene_run.main([str(self.image), "--image", str(self.image),
                             "--out", str(self.root / "other-runs")])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
