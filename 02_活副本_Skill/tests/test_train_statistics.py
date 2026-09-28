"""Offline log-contract regressions. Fixtures below are synthetic unless noted."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT.parent / "tmp" / "t06-test"
FIXTURES.mkdir(parents=True, exist_ok=True)
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from _train_log import integrity, read_log  # noqa: E402

spec = importlib.util.spec_from_file_location("train_stage_t06", SCRIPTS / "50_train.py")
train_stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_stage)


@contextlib.contextmanager
def fixture_directory():
    """Keep synthetic fixtures and child output for failure review."""
    yield Path(tempfile.mkdtemp(prefix="case-", dir=FIXTURES))


def line(step, total, ms=100.0, gbs=8, rank=0):
    return (f"[Rank {rank} | Local Rank {rank}] iteration {step}/{total} | "
            f"consumed samples: {step * gbs} | elapsed time per iteration (ms): {ms} | "
            f"learning rate: 1.0E-6 | global batch size: {gbs} | "
            "loss: 1.0 | grad norm: 2.0 |\n")


class TrainingStatisticsTests(unittest.TestCase):
    def run_extract(self, content, window="50,100", expected_gbs=8):
        with fixture_directory() as tmp:
            log = Path(tmp) / "train.log"
            summary = Path(tmp) / "summary.json"
            log.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "95_extract_series.py"),
                 "--log", str(log), "--out", str(Path(tmp) / "series.csv"),
                 "--summary-json", str(summary), "--window", window,
                 "--expected-gbs", str(expected_gbs)], capture_output=True, text=True,
                encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            (Path(tmp) / "stdout.txt").write_text(result.stdout, encoding="utf-8")
            (Path(tmp) / "stderr.txt").write_text(result.stderr, encoding="utf-8")
            (Path(tmp) / "returncode.txt").write_text(str(result.returncode), encoding="utf-8")
            self.assertTrue(summary.exists(),
                            f"missing summary; fixture={tmp} rc={result.returncode} "
                            f"stdout={result.stdout!r} stderr={result.stderr!r}")
            return result, json.loads(summary.read_text(encoding="utf-8"))

    def test_saved_reference_statistics(self):
        # Real saved repository copy; the original external source is unavailable.
        log = ROOT / "examples" / "train" / "official_baseline.log"
        parsed = read_log(log)
        check = integrity(parsed, expected_end=100, expected_gbs=8)
        self.assertEqual(check["state"], "COMPLETE")
        result, doc = self.run_extract(log.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(doc["official_selection"]["metrics"]["mean_ms"], 2502.018)
        self.assertAlmostEqual(doc["official_selection"]["metrics"]["mean_throughput_samples_per_s"],
                               3.197419, places=5)
        self.assertAlmostEqual(doc["window_selection"]["metrics"]["mean_ms"],
                               649.2921568627451)
        self.assertAlmostEqual(doc["window_selection"]["metrics"]["median_ms"], 431.3)

    def test_first_200_then_last_100(self):
        content = "".join(line(step, 200, ms=float(step)) for step in range(1, 201))
        result, doc = self.run_extract(content)
        self.assertEqual(result.returncode, 0, result.stderr)
        chosen = doc["official_selection"]["metrics"]["steps"]
        self.assertEqual(chosen, list(range(101, 201)))
        self.assertEqual(doc["official_selection"]["metrics"]["mean_ms"], 150.5)

    def test_missing_duplicate_rank_bad_and_gbs_conflict(self):
        whole = "".join(line(step, 100) for step in range(1, 101))
        variants = [whole.replace(line(100, 100), ""),
                    whole.replace(line(100, 100), line(99, 100)),
                    whole + line(100, 100, rank=1),
                    whole.replace(line(70, 100), line(70, 100, ms="nan")),
                    whole.replace(line(80, 100), line(80, 100, gbs=4))]
        for content in variants:
            with self.subTest(content=content[-100:]):
                result, doc = self.run_extract(content)
                self.assertEqual(result.returncode, 3)
                self.assertEqual(doc["state"], "INCOMPLETE")
                self.assertIsNone(doc["official_selection"]["metrics"])

    def test_one_point_window_cannot_pass(self):
        content = "".join(line(step, 1) for step in (1,))
        result, doc = self.run_extract(content, window="1,1")
        self.assertEqual(result.returncode, 3)
        self.assertEqual(doc["steps_used"], 0)
        self.assertIsNone(doc["samples_per_s"])

    def test_unknown_resume_start_is_not_complete(self):
        with fixture_directory() as tmp:
            log = Path(tmp) / "resume.log"
            log.write_text("".join(line(step, 100) for step in range(51, 101)),
                           encoding="utf-8")
            check = integrity(read_log(log), expected_end=100, start_step=None,
                              expected_gbs=8)
            self.assertEqual(check["state"], "UNKNOWN")
            self.assertIsNone(check["missing_steps"])

    def test_target_load_tracker_start_semantics(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            self.assertEqual(train_stage.configured_start({}, str(root))[:2],
                             (1, "no_training_load"))
            self.assertIsNone(train_stage.configured_start({"load": str(root)}, str(root))[0])
            tracker = root / "latest_checkpointed_iteration.txt"
            tracker.write_text("release\n", encoding="utf-8")
            self.assertEqual(train_stage.configured_start({"load": str(root)}, str(root))[0], 1)
            tracker.write_text("0000042\n", encoding="utf-8")
            self.assertIsNone(train_stage.configured_start({"load": str(root)}, str(root))[0])
            tracker.write_text("broken\n", encoding="utf-8")
            self.assertIsNone(train_stage.configured_start({"load": str(root)}, str(root))[0])

    def test_relative_load_resolves_against_runner_workdir(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            workdir = root / "msmm"
            workdir.mkdir()
            intended = workdir / "same_name"
            unintended = root / "same_name"
            intended.mkdir()
            unintended.mkdir()
            (intended / "latest_checkpointed_iteration.txt").write_text("release\n", encoding="utf-8")
            (unintended / "latest_checkpointed_iteration.txt").write_text("42\n", encoding="utf-8")
            start, source, tracker = train_stage.configured_start({"load": "same_name"},
                                                                  str(workdir))
            self.assertEqual(start, 1)
            self.assertEqual(source, "release_tracker_initial_weights")
            self.assertEqual(Path(tracker), intended / "latest_checkpointed_iteration.txt")

    def test_reordered_steps_are_incomplete(self):
        with fixture_directory() as tmp:
            log = Path(tmp) / "out_of_order.log"
            log.write_text("".join(line(step, 100) for step in range(100, 0, -1)),
                           encoding="utf-8")
            check = integrity(read_log(log), expected_end=100, expected_gbs=8)
            self.assertEqual(check["state"], "INCOMPLETE")
            self.assertIn("non_increasing_step_order", check["problems"])
            self.assertTrue(check["out_of_order"])

    def test_p5_zero_exit_with_missing_step_is_failure(self):
        # Exercise the real P5 main success gate, with only process launch
        # replaced so the test never invokes torchrun or NPU.
        with fixture_directory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("training:\n  micro_batch_size: 4\n"
                              "  gradient_accumulation_steps: 1\n  train_iters: 100\n"
                              "parallel:\n  data_parallel_size: 2\n", encoding="utf-8")
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
                Path(log_path).write_text("".join(line(step, 100) for step in range(1, 100)),
                                          encoding="utf-8")
                (run_dir / "effective_config.yaml").write_bytes(config_bytes)
                return ZeroProcess(), str(run_dir)

            args = ["50_train.py", "--config", str(config), "--env", str(env),
                    "--log", str(log), "--foreground"]
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(train_stage, "launch_training", fake_launch), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = train_stage.main()
            self.assertNotEqual(result, 0)
            check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
            self.assertEqual(check["train_rc"], 0)
            self.assertEqual(check["state"], "INCOMPLETE")
            self.assertEqual(check["missing_steps"], [100])

    def test_p5_config_snapshot_identity_and_mutation(self):
        for mutation in (None, "input", "snapshot"):
            with self.subTest(mutation=mutation), fixture_directory() as tmp:
                root = Path(tmp)
                config = root / "config.yaml"
                config.write_text("training:\n  micro_batch_size: 4\n"
                                  "  gradient_accumulation_steps: 1\n  train_iters: 100\n"
                                  "parallel:\n  data_parallel_size: 2\n", encoding="utf-8")
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
                    snapshot = run_dir / "effective_config.yaml"
                    snapshot.write_bytes(config_bytes)
                    Path(log_path).write_text("".join(line(step, 100) for step in range(1, 101)),
                                              encoding="utf-8")
                    if mutation == "input":
                        config.write_text(config.read_text(encoding="utf-8") + "# changed\n",
                                          encoding="utf-8")
                    elif mutation == "snapshot":
                        snapshot.write_bytes(config_bytes + b"# changed\n")
                    return ZeroProcess(), str(run_dir)

                args = ["50_train.py", "--config", str(config), "--env", str(env),
                        "--log", str(log), "--foreground"]
                output = io.StringIO()
                with mock.patch.object(sys, "argv", args), \
                     mock.patch.object(train_stage, "launch_training", fake_launch), \
                     contextlib.redirect_stdout(output):
                    result = train_stage.main()
                (root / "p5.stdout.txt").write_text(output.getvalue(), encoding="utf-8")
                (root / "p5.returncode.txt").write_text(str(result), encoding="utf-8")
                check = json.loads((run_dir / "train_integrity.json").read_text(encoding="utf-8"))
                self.assertEqual(result == 0, mutation is None, f"fixture={root} {check}")
                self.assertEqual(check["state"] == "COMPLETE", mutation is None)


if __name__ == "__main__":
    unittest.main()
