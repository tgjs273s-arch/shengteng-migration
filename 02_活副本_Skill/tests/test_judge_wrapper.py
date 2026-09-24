"""Judge orchestration contracts, using isolated fake child tools."""
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import unittest
from unittest import mock
from test_stage_contracts import load_script, test_directory

judge = load_script('70_judge.py')
TOOLS = ['fingerprint_cfg.py', 'fingerprint_observed.py', 'judge_comparable.py']
SCHEMAS = ['sk04_fingerprint_cfg.v1', 'sk04_fingerprint_observed.v1', 'sk04_judge_verdict.v1']


class JudgeWrapperTests(unittest.TestCase):
    def exercise(self, root, bad=None, mode='rc', extra=(), color='red'):
        log = root / 'train.log'
        log.write_text('fixture')
        out = root / 'judge'
        out.mkdir(exist_ok=True)
        for name in ('fp.json', 'obs.json', 'verdict.json', 'judge_summary.json'):
            (out / name).write_text('{"old": true}')
        calls = []
        def child(script, args):
            calls.append(script)
            if script == bad and mode == 'rc':
                return 9, '', 'injected failure'
            if script in TOOLS:
                target = Path(args[args.index('--out') + 1])
                self.assertNotEqual(target.parent, out)
                if script == bad and mode == 'missing':
                    return 0, '', ''
                doc = {'schema': SCHEMAS[TOOLS.index(script)]}
                if script == TOOLS[-1]:
                    doc.update(verdict_id='fixture-id', open_gates=[], config_deviation={'count_mismatch': 2},
                               reachability={'verdict': 'fixture-verdict', 'level': 'none', 'color': color})
                target.write_text('broken' if script == bad and mode == 'malformed' else json.dumps(doc))
                return 0, '', ''  # Verdict must come from JSON, not a stdout regex.
            return 0, 'REGISTRY_APPEND_OK seq=1', ''
        evidence = dict(primary=dict(mbs=8, gas=1, dp=1), cli=None, cfg={}, log={},
                        gbs=8, primary_src='config', log_world=1)
        stdout = io.StringIO()
        with mock.patch.object(sys, 'argv', ['70_judge.py', '--log', str(log), '--out', str(out), *extra]), \
                mock.patch.object(judge, 'collect_world_evidence', return_value=evidence), \
                mock.patch.object(judge, 'resolve_world_size', return_value=(1, 'cli', [], None, False)), \
                mock.patch.object(judge, 'run_tool', side_effect=child), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            rc = judge.main()
        return rc, calls, json.loads((out / 'judge_summary.json').read_text()), stdout.getvalue()

    def test_each_child_failure_blocks_downstream_and_old_results(self):
        for tool in TOOLS:
            for mode in ('rc', 'missing', 'malformed'):
                with self.subTest(tool=tool, mode=mode), test_directory() as tmp:
                    rc, calls, doc, text = self.exercise(Path(tmp), tool, mode)
                    self.assertEqual(rc, 3)
                    self.assertEqual(calls[-1], tool)
                    self.assertEqual(doc['execution_state'], 'FAILED')
                    self.assertNotIn('JUDGE_DONE', text)

    def test_registry_failure_is_not_reported_as_success(self):
        with test_directory() as tmp:
            rc, calls, doc, text = self.exercise(Path(tmp), 'registry_append.py', extra=('--registry', str(Path(tmp)/'reg.json'), '--tag', 'one'))
            self.assertEqual(rc, 3)
            self.assertEqual(doc['failed_tool'], 'registry_append.py')
            self.assertNotIn('JUDGE_DONE', text)

    def test_business_colors_remain_valid_results_and_gate_codes(self):
        for color, code in [('green', 0), ('yellow', 3), ('red', 4)]:
            for gate in (False, True):
                with self.subTest(color=color, gate=gate), test_directory() as tmp:
                    rc, calls, doc, text = self.exercise(Path(tmp), color=color, extra=('--gate',) if gate else ())
                    self.assertEqual(rc, code if gate else 0)
                    self.assertEqual(doc['execution_state'], 'COMPLETED')
                    self.assertEqual(doc['verdict'], 'fixture-verdict')
                    self.assertEqual(doc['deviations'], 2)
                    self.assertIn('JUDGE_DONE', text)

    def test_real_child_chain_and_duplicate_registry_tag(self):
        with test_directory() as tmp:
            root = Path(tmp)
            log = root / 'synthetic.log'
            log.write_text('SYNTHETIC TEST ONLY -- NOT NPU EVIDENCE\n'
                           'iteration 1 / 100 | consumed samples: 8 | elapsed time per iteration (ms): 10 '
                           '| learning rate: 1e-5 | global batch size: 8 | loss: 2.0 | grad norm: 1.0 |\n')
            skill = Path(judge.SKILL_ROOT)
            command = [sys.executable, str(skill / 'scripts/70_judge.py'), '--log', str(log),
                       '--config', str(skill / 'sk04_judge/tests/fixtures/run_sk01_aligned_config.yaml'),
                       '--world-size', '1', '--out', str(root / 'judge'),
                       '--registry', str(root / 'registry.json'), '--tag', 'synthetic-test']
            def run():
                return subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                      env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, timeout=60)
            first = run()
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            summary = json.loads((root / 'judge/judge_summary.json').read_text())
            self.assertEqual(summary['execution_state'], 'COMPLETED')
            self.assertTrue((Path(summary['attempt_dir']) / 'verdict.json').is_file())
            registry_before = (root / 'registry.json').read_bytes()
            second = run()
            self.assertEqual(second.returncode, 3, second.stdout + second.stderr)
            self.assertNotIn('JUDGE_DONE', second.stdout)
            failed = json.loads((root / 'judge/judge_summary.json').read_text())
            self.assertEqual(failed['failed_tool'], 'registry_append.py')
            self.assertNotEqual(failed['attempt_dir'], summary['attempt_dir'])
            self.assertEqual((root / 'registry.json').read_bytes(), registry_before)

    def test_registry_requires_tag_before_any_child_runs(self):
        with test_directory() as tmp:
            rc, calls, _, text = self.exercise(Path(tmp), extra=('--registry', 'unused.json'))
            self.assertEqual(rc, 2)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
