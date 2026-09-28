"""T07 numeric diagnostics use synthetic inputs, never NPU acceptance evidence."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from test_stage_contracts import load_script, test_directory


judge = load_script('70_judge.py')


def row(step, loss, grad_norm):
    return {'iter': step, 'loss': loss, 'grad_norm': grad_norm}


class JudgeNumericTests(unittest.TestCase):
    def test_zero_and_near_zero_reference_are_not_fake_zero_percent(self):
        ours = {1: row(1, 1.0, 0.0), 2: row(2, 0.0, 2.0)}
        reference = {1: row(1, 0.0, 0.0), 2: row(2, 1e-13, 1.0)}
        metrics = judge.pointwise_metrics(ours, reference)
        self.assertEqual(metrics['loss']['relative_undefined_steps'], [1, 2])
        self.assertEqual(metrics['rows'][0]['loss_abs_err'], 1.0)
        self.assertIsNone(metrics['rows'][0]['loss_rel_err_pct'])
        self.assertIsNone(metrics['rows'][0]['gn_rel_err_pct'])
        self.assertEqual(metrics['rows'][1]['gn_rel_err_pct'], 100.0)

    def test_only_common_step_fifty_does_not_become_step_one(self):
        metrics = judge.pointwise_metrics({50: row(50, 2.0, 1.0)},
                                          {50: row(50, 1.0, 1.0)})
        self.assertEqual(metrics['common_steps'], [50])
        self.assertNotIn('step1_loss_ours', metrics)
        self.assertNotIn('step1_loss_official', metrics)

    def test_large_difference_is_measured_without_claiming_a_pass(self):
        metrics = judge.pointwise_metrics({1: row(1, 3.0, 4.0)},
                                          {1: row(1, 1.0, 2.0)})
        self.assertEqual(metrics['loss']['relative_pct']['max'], 200.0)
        self.assertEqual(metrics['grad_norm']['relative_pct']['max'], 100.0)
        self.assertNotIn('pass', metrics)

    def test_json_rows_are_the_csv_source_without_rounding_drift(self):
        metrics = judge.pointwise_metrics({1: row(1, 1.1, 2.2)},
                                          {1: row(1, 1.0, 2.0)})
        columns = ('iter', 'official_loss', 'ours_loss', 'loss_abs_err',
                   'loss_rel_err_pct', 'official_gn', 'ours_gn', 'gn_abs_err',
                   'gn_rel_err_pct')
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metrics['rows'])
        csv_row = next(csv.DictReader(io.StringIO(stream.getvalue())))
        json_row = json.loads(json.dumps(metrics['rows'][0]))
        for key in columns:
            self.assertEqual(csv_row[key], str(json_row[key]))

    def test_shared_parser_exposes_duplicate_and_nonfinite_steps(self):
        with test_directory() as tmp:
            path = Path(tmp) / 'synthetic.log'
            def line(step, loss):
                return (f'[Rank 0 | Local Rank 0] iteration {step}/2 | '
                        f'consumed samples: {step * 8} | '
                        'elapsed time per iteration (ms): 10 | '
                        'learning rate: 0 | global batch size: 8 | '
                        f'loss: {loss} | grad norm: 1 |')
            path.write_text('\n'.join([line(1, 1), line(1, 1), line(2, 'NaN')]),
                            encoding='utf-8')
            parsed = judge.read_log(path)
            state = judge.integrity(parsed, expected_end=2, expected_gbs=8)
            self.assertEqual(state['state'], 'INCOMPLETE')
            self.assertEqual(state['duplicate_steps'], [1])
            self.assertIn('bad_iteration_records', state['problems'])
            self.assertIn(2, state['missing_steps'])

    def test_baseline_alias_and_hash_pairing(self):
        info = judge.BASELINE_IDENTITIES['officialB']
        path = Path(judge.BASELINE_DIR) / 'officialB.yaml'
        reference = Path(info['log_path'])
        matched = judge.baseline_identity(info['canonical_id'], str(path), str(reference))
        self.assertEqual(matched['canonical_id'], info['canonical_id'])
        self.assertEqual(matched['identity_status'], 'MATCHED')
        with test_directory() as tmp:
            wrong = Path(tmp) / 'wrong.log'
            wrong.write_text('SYNTHETIC TEST ONLY', encoding='utf-8')
            mismatched = judge.baseline_identity('officialB', str(path), str(wrong))
            self.assertIn('baseline_log_hash_mismatch', mismatched['identity_issues'])
        old = judge.baseline_identity('officialA',
                                      str(Path(judge.BASELINE_DIR) / 'officialA.yaml'), None)
        self.assertEqual(old['canonical_id'], judge.LEGACY_A_ID)
        self.assertIn('reference_log_missing', old['identity_issues'])

    def test_observed_does_not_label_step_fifty_as_step_one(self):
        with test_directory() as tmp:
            root = Path(tmp)
            log = root / 'synthetic-step-50.log'
            result = root / 'observed.json'
            log.write_text(
                '[Rank 0 | Local Rank 0] iteration 50/50 | consumed samples: 400 | '
                'elapsed time per iteration (ms): 10 | learning rate: 0 | '
                'global batch size: 8 | loss: 1.5 | grad norm: 2 |\n',
                encoding='utf-8')
            script = (Path(judge.SKILL_ROOT) / 'sk04_judge/scripts/fingerprint_observed.py')
            completed = subprocess.run([sys.executable, str(script), '--log', str(log),
                                        '--out', str(result)], capture_output=True, text=True,
                                       encoding='utf-8', timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            observed = json.loads(result.read_text(encoding='utf-8'))['observed']
            self.assertIsNone(observed['step1_lr0']['iter'])
            self.assertIsNone(observed['step1_lr0']['loss'])

    def test_p7_cli_duplicate_step_without_reference_opens_integrity_gate(self):
        with test_directory() as tmp:
            root = Path(tmp)
            log = root / 'synthetic-duplicate.log'
            out = root / 'judge'
            def line(step, samples):
                return (f'[Rank 0 | Local Rank 0] iteration {step}/2 | '
                        f'consumed samples: {samples} | '
                        'elapsed time per iteration (ms): 10 | learning rate: 0 | '
                        'global batch size: 8 | loss: 2.0 | grad norm: 1.0 |')
            log.write_text('SYNTHETIC TEST ONLY -- NOT NPU EVIDENCE\n' + '\n'.join(
                [line(1, 8), line(1, 8), line(2, 16)]) + '\n', encoding='utf-8')
            skill = Path(judge.SKILL_ROOT)
            config = skill / 'sk04_judge/tests/fixtures/run_sk01_aligned_config.yaml'
            completed = subprocess.run(
                [sys.executable, str(skill / 'scripts/70_judge.py'), '--log', str(log),
                 '--config', str(config), '--world-size', '1', '--out', str(out)],
                capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads((out / 'judge_summary.json').read_text(encoding='utf-8'))
            observed = json.loads((Path(summary['attempt_dir']) / 'obs.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['execution_state'], 'COMPLETED')
            self.assertEqual(summary['candidate_integrity']['state'], 'INCOMPLETE')
            self.assertIn('candidate_integrity_incomplete', summary['numeric_open_gates'])
            self.assertIsNone(observed['observed']['step1_lr0']['loss'])
            self.assertIn('iteration_integrity', summary['open_gates'])
            self.assertNotEqual(summary['verdict'], 'POINTWISE_OK')

    def test_observed_cli_rejects_nonfinite_and_out_of_order_step_one_evidence(self):
        with test_directory() as tmp:
            root = Path(tmp)
            script = Path(judge.SKILL_ROOT) / 'sk04_judge/scripts/fingerprint_observed.py'
            def line(step, loss):
                return (f'[Rank 0 | Local Rank 0] iteration {step}/2 | '
                        f'consumed samples: {step * 8} | '
                        'elapsed time per iteration (ms): 10 | learning rate: 0 | '
                        f'global batch size: 8 | loss: {loss} | grad norm: 1 |')
            scenarios = {
                'nonfinite': [line(1, '1.0'), line(2, 'NaN')],
                'out-of-order': [line(2, '1.0'), line(1, '1.0')],
            }
            for name, lines in scenarios.items():
                with self.subTest(name=name):
                    log = root / f'{name}.log'
                    result = root / f'{name}.json'
                    log.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                    completed = subprocess.run(
                        [sys.executable, str(script), '--log', str(log), '--out', str(result)],
                        capture_output=True, text=True, encoding='utf-8', timeout=30)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    observed = json.loads(result.read_text(encoding='utf-8'))['observed']
                    self.assertEqual(observed['iteration_integrity']['state'], 'INCOMPLETE')
                    self.assertIsNone(observed['step1_lr0']['loss'])


if __name__ == '__main__':
    unittest.main()
