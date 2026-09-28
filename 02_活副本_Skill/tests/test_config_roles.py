"""Reference/candidate configuration regressions; synthetic P0 input only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "20_plan_migration.py"
BASE = ROOT.parent / "tmp" / "config_role_tests"


class ConfigRoleTests(unittest.TestCase):
    def setUp(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="role-", dir=BASE))
        self.env = self.root / "env.json"
        self.points = self.root / "points.json"
        self.points.write_text('{"points":{"full_attn":[]}}', encoding="utf-8")

    def generate(self, profile, *extra):
        self.env.write_text(json.dumps({"path": "synthetic", "recommended_profile": profile}),
                            encoding="utf-8")
        output = self.root / ("out-" + str(len(list(self.root.glob("out-*")))))
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--env", str(self.env), "--points", str(self.points),
             "--out", str(output), *extra],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return proc, output

    @staticmethod
    def read(output):
        config_path = output / "train_config.yaml"
        config_bytes = config_path.read_bytes()
        manifest = json.loads((output / "config_manifest.json").read_text(encoding="utf-8"))
        assert manifest["effective_config"]["sha256"] == hashlib.sha256(config_bytes).hexdigest()
        reference_bytes = (output / "reference_config.yaml").read_bytes()
        assert manifest["reference_config"]["sha256"] == hashlib.sha256(reference_bytes).hexdigest()
        return yaml.safe_load(config_bytes), manifest

    def test_default_reference_does_not_inherit_p0_optimizations(self):
        profile = {"world_size": 2, "dp": 2, "mbs": 4, "gas": 1,
                   "operator_backend": "triton", "pregather": True,
                   "recompute": False, "enable_chunk_loss": False,
                   "enable_activation_offload": False, "save_format": "auto"}
        proc, output = self.generate(profile)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        config, manifest = self.read(output)
        self.assertEqual(manifest["role"], "reference")
        self.assertEqual(manifest["baseline_id"], "qwen35-0p8b-triton-20260724-100step-v1")
        self.assertEqual(manifest["official_rule_state"], "RULE_PENDING")
        self.assertIn("features.recompute", manifest["verified_reference_fields"])
        self.assertIn("training.save_format", manifest["template_default_fields_unverified"])
        self.assertIn("training.seed", manifest["template_default_fields_unverified"])
        differences = {row["field"] for row in manifest["differences_from_reference"]}
        self.assertNotIn("features.recompute", differences)
        self.assertNotIn("parallel.fsdp_plan.pregather", differences)
        self.assertFalse(config["parallel"]["fsdp_plan"]["pregather"])
        self.assertTrue(config["features"]["recompute"])
        self.assertTrue(config["features"]["enable_chunk_loss"])
        self.assertTrue(config["features"]["enable_activation_offload"])
        self.assertTrue(config["model"]["skip_flash_attn_recompute"])
        self.assertEqual(config["training"]["save_interval"], 100)

    def test_candidate_has_auditable_differences_and_exact_paths(self):
        profile = {"world_size": 2, "dp": 2, "mbs": 4, "gas": 1,
                   "operator_backend": "triton", "pregather": True, "prefetch": 2,
                   "recompute": False, "enable_chunk_loss": False,
                   "enable_activation_offload": False, "save_format": "dcp"}
        unusual = str(self.root / "中文 path's # colon: value $HOME")
        proc, output = self.generate(profile, "--config-role", "candidate",
                                     "--data-json", unusual + ".json", "--data-dir", unusual,
                                     "--weight-hf", unusual + " hf",
                                     "--weight-dcp", unusual + " dcp")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        config, manifest = self.read(output)
        differences = {row["field"]: row for row in manifest["differences_from_reference"]}
        self.assertEqual(manifest["role"], "candidate")
        self.assertEqual((differences["parallel.fsdp_plan.pregather"]["reference"],
                          differences["parallel.fsdp_plan.pregather"]["effective"]),
                         (False, True))
        for field in ("features.recompute", "features.enable_chunk_loss",
                      "features.enable_activation_offload"):
            self.assertEqual((differences[field]["reference"], differences[field]["effective"]),
                             (True, False))
        self.assertEqual(config["data"]["dataset_param"]["basic_parameters"]["dataset"],
                         [unusual + ".json"])
        self.assertEqual(config["data"]["dataset_param"]["basic_parameters"]["dataset_dir"],
                         unusual)
        self.assertEqual(config["training"]["load"], unusual + " dcp")
        self.assertEqual(differences["training.load"]["effective"], unusual + " dcp")
        self.assertEqual(differences["data.dataset_param.basic_parameters.dataset"]["effective"],
                         [unusual + ".json"])
        self.assertEqual(manifest["geometry"]["global_batch_size"], 8)
        self.assertEqual(differences["training.save_interval"]["effective"], 10000)
        self.assertEqual(differences["parallel.fsdp_plan.num_to_forward_prefetch"]["effective"], 2)
        self.assertIn("| pregather / forward prefetch / backward prefetch | True / 2 / 2 |",
                      (output / "migrate_plan.md").read_text(encoding="utf-8"))

    def test_reference_on_single_die_is_explicitly_blocked(self):
        profile = {"world_size": 1, "dp": 1, "mbs": 2, "gas": 4,
                   "operator_backend": "eager", "save_format": "hf"}
        proc, output = self.generate(profile)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        config, manifest = self.read(output)
        self.assertEqual(config["parallel"]["data_parallel_size"], 2)
        self.assertEqual(manifest["reference_feasibility"], "blocked")
        self.assertTrue(manifest["reference_feasibility_reasons"])
        self.assertIn("PLAN_BLOCKED", proc.stdout)

    def test_missing_probe_fields_stay_unknown(self):
        proc, output = self.generate({})
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        _, manifest = self.read(output)
        self.assertEqual(manifest["reference_feasibility"], "unknown")
        self.assertTrue(manifest["reference_feasibility_unknown"])

    def test_reference_records_step_and_custom_template_changes(self):
        profile = {"world_size": 2, "dp": 2, "mbs": 4, "gas": 1,
                   "operator_backend": "triton", "load_rank0_and_broadcast": False,
                   "save_format": "auto"}
        template = self.root / "custom.yaml"
        source = (ROOT / "config" / "templates" / "qwen3_5_0_8B_base.yaml").read_text(encoding="utf-8")
        template.write_text(source.replace("  seed: 42", "  seed: 43\n  new_empty: {}"), encoding="utf-8")
        proc, output = self.generate(profile, "--steps", "12", "--template", str(template))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        _, manifest = self.read(output)
        differences = {row["field"]: row for row in manifest["differences_from_reference"]}
        self.assertEqual(differences["training.train_iters"]["effective"], 12)
        self.assertEqual(differences["training.seed"]["effective"], 43)
        self.assertFalse(differences["training.new_empty"]["reference_present"])
        self.assertTrue(differences["training.new_empty"]["effective_present"])
        self.assertEqual(differences["training.new_empty"]["effective"], {})

    def test_diff_distinguishes_absent_from_null(self):
        spec = importlib.util.spec_from_file_location("config_roles_plan", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        changes = module.config_differences({"a": None, "b": {}}, {"c": None})
        by_field = {row["field"]: row for row in changes}
        self.assertEqual(set(by_field), {"a", "b", "c"})
        self.assertFalse(by_field["a"]["reference_present"])
        self.assertTrue(by_field["a"]["effective_present"])
        self.assertFalse(by_field["c"]["effective_present"])

    def test_invalid_candidate_geometry_refuses_output(self):
        profile = {"world_size": 2, "dp": 2, "mbs": 3, "gas": 1,
                   "operator_backend": "triton", "save_format": "dcp"}
        proc, output = self.generate(profile, "--config-role", "candidate")
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertFalse((output / "train_config.yaml").exists())
        self.assertFalse((output / "config_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
