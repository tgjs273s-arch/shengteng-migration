"""Offline driver identity and A/B reportability examples; no training or download."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / 'scripts'
spec = importlib.util.spec_from_file_location('pipeline_integration_test', SCRIPTS / '_pipeline_integration.py')
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


class PipelineIntegrationTests(unittest.TestCase):
    def fixture(self):
        root = SKILL.parent / 'tmp' / 'pipeline_integration_tests'
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix='case-', dir=root))

    def test_current_59_and_63_shapes_report_performance_but_block_adoption(self):
        for schema, field in (('p59ab.v2', 'window_median_ms'),
                              ('t63_two_config_ab.v1', 'window_11_end_median_ms')):
            with self.subTest(schema=schema):
                root = self.fixture()
                runs = []
                for i in range(3):
                    runs.extend(({'tag': 'A%d' % i, 'variant': 'A', field: 100.0 + i},
                                 {'tag': 'B%d' % i, 'variant': 'B', field: 90.0 + i}))
                summary = root / 'ab.json'
                summary.write_text(json.dumps({'schema': schema, 'runs': runs,
                                               'official_comparable': False}), encoding='utf-8')
                noise = root / 'noise.json'
                noise.write_text(json.dumps({'schema': 'p59null.v1', 'runs': 3,
                                             'spread_pct': 1.0,
                                             'medians_ms': [100.0, 101.0, 99.0]}), encoding='utf-8')
                self.assertEqual(integration.candidate_inputs(
                    summary, noise, 'A', 'B'), 3)
                out = root / 'reportability.json'
                cmd = [sys.executable, str(SCRIPTS / '62_reportability.py'),
                       '--summary', str(summary), '--noise', str(noise),
                       '--baseline-variant', 'A', '--candidate-variant', 'B',
                       '--run-validity', 'UNVERIFIED', '--accuracy-validity', 'UNVERIFIED',
                       '--out', str(out)]
                result = subprocess.run(cmd, cwd=SKILL, capture_output=True, text=True,
                                        encoding='utf-8', timeout=20)
                self.assertIn(result.returncode, (3, 4, 5), result.stdout + result.stderr)
                report = json.loads(out.read_text(encoding='utf-8'))
                self.assertEqual(report['run_validity'], 'UNVERIFIED')
                self.assertEqual(report['accuracy_validity'], 'UNVERIFIED')
                self.assertFalse(report['adoption_ready'])
                self.assertFalse(report['roles_assumed'])
                self.assertEqual(report['summary_schema'], schema)
                self.assertEqual(report['pairs'], 3)

    def test_pair_roles_and_missing_sources_fail_explicitly(self):
        root = self.fixture()
        summary = root / 'ab.json'
        summary.write_text(json.dumps({'schema': 'p59ab.v2', 'runs': [
            {'variant': 'A', 'window_median_ms': 100},
            {'variant': 'B', 'window_median_ms': 90}]}), encoding='utf-8')
        noise = root / 'noise.json'
        noise.write_text(json.dumps({'schema': 'p59null.v1', 'runs': 3, 'spread_pct': 1.0}),
                         encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'distinct A/B'):
            integration.candidate_inputs(summary, noise, 'A', 'A')
        with self.assertRaises(FileNotFoundError):
            integration.candidate_inputs(root / 'missing.json', noise, 'A', 'B')

    def test_ledger_failure_makes_current_acceptance_fail(self):
        root = self.fixture()
        verdicts = root / 'verdicts.tsv'
        verdicts.write_text('', encoding='utf-8')
        out = root / 'summary.json'
        result = subprocess.run([sys.executable, str(SCRIPTS / '_pipeline_integration.py'),
                                 'aggregate', '--flow', 'diagnostic', '--verdicts', str(verdicts),
                                 '--out', str(out), '--ledger-bad', '1'],
                                cwd=SKILL, capture_output=True, text=True, encoding='utf-8', timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(summary['overall'], 'FAIL')
        self.assertIn('degradation_ledger_inconsistent', summary['open_gates'])


if __name__ == '__main__':
    unittest.main()
