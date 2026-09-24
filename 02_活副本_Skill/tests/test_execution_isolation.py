"""Local process regressions; no NPU, network, or recursive cleanup."""
import json
import os
from pathlib import Path
import time
import unittest
from test_stage_contracts import load_script, test_directory

train = load_script('50_train.py')


@unittest.skipUnless(os.name == 'posix', 'Ascend runner requires Linux')
class TrainingIsolationTests(unittest.TestCase):
    def test_parallel_runners_have_independent_exit_and_logs(self):
        with test_directory() as tmp:
            root = Path(tmp) / "中文 space ' $HOME"
            root.mkdir()
            a, ad = train.launch_training('sleep 0.1; echo train_rc=0; exit 7', str(root / 'a.log'))
            b, bd = train.launch_training('echo train_rc=7; exit 0', str(root / 'b.log'))
            self.assertEqual(a.wait(timeout=10), 7)
            self.assertEqual(b.wait(timeout=10), 0)
            self.assertNotEqual(ad, bd)
            self.assertIn('train_rc=0', (Path(ad) / 'runner.log').read_text())
            self.assertIn('train_rc=7', (Path(bd) / 'runner.log').read_text())
            self.assertEqual(json.loads((Path(ad) / 'run.json').read_text())['pid'], a.pid)

    def test_shared_log_is_locked_and_lock_releases_on_exit(self):
        with test_directory() as tmp:
            log = Path(tmp) / 'train.log'
            proc, _ = train.launch_training('sleep 0.5; exit 0', str(log))
            try:
                with self.assertRaises(BlockingIOError):
                    train.launch_training('exit 0', str(log))
            finally:
                self.assertEqual(proc.wait(timeout=10), 0)
            other, _ = train.launch_training('exit 0', str(log))
            self.assertEqual(other.wait(timeout=10), 0)

    def test_failed_command_cannot_leave_old_training_success(self):
        with test_directory() as tmp:
            log = Path(tmp) / 'train.log'
            log.write_text('iteration 1 / 1 old success')
            proc, _ = train.launch_training('exit 9', str(log), detached=False)
            self.assertEqual(proc.wait(timeout=10), 9)
            self.assertEqual(log.read_text(), '')


if __name__ == '__main__':
    unittest.main()
