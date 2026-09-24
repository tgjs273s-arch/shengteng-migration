"""Contract regressions; no model download or NPU required."""
import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


@contextmanager
def test_directory():
    artifacts = ROOT.parent / 'tmp' / 'contract_tests'
    artifacts.mkdir(parents=True, exist_ok=True)
    yield tempfile.mkdtemp(prefix='contract-', dir=artifacts)
    # Keep diagnostic files; no recursive cleanup under the project policy.


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load_script('20_plan_migration.py')
analyze = load_script('10_analyze_points.py')


class MigrationContractTests(unittest.TestCase):
    def test_real_p1_output_reaches_plan(self):
        with test_directory() as tmp:
            out = Path(tmp) / 'analyze'
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / '10_analyze_points.py'),
                 str(ROOT / 'refs/modeling_qwen3_5__transformers_v5.2.0.py'), '--out', str(out)],
                capture_output=True, text=True, encoding='utf-8',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((out / 'migrate_points.json').read_text(encoding='utf-8'))
            self.assertNotIn('SELF_CHECK', payload['source_tag'])
            rendered = plan.build_plan_md({}, {}, payload, 'train.yaml', [], [])
            self.assertEqual(len(payload['points']), 4)
            for category, hits in payload['points'].items():
                self.assertIn('| `%s` | %s |' % (category, len(hits)), rendered)
            self.assertIn(payload['source_tag'], rendered)

    def test_cli_data_and_weight_paths_roundtrip_in_yaml(self):
        import yaml
        with test_directory() as tmp:
            root=Path(tmp)
            env=root/'env.json'
            env.write_text(json.dumps({'path':'test_only','recommended_profile':
                {'world_size':2,'mbs':4,'gas':1,'save_format':'dcp'}}),encoding='utf-8')
            points=root/'points.json'
            points.write_text(json.dumps({'points':{'full_attn':[]}}),encoding='utf-8')
            unusual=str(root / "中文 data's # fragment: value $HOME")
            output=root/'plan'
            result=subprocess.run([sys.executable,str(SCRIPTS/'20_plan_migration.py'),
                '--env',str(env),'--points',str(points),'--out',str(output),
                '--data-json',unusual+'.json','--data-dir',unusual,
                '--weight-hf',unusual+' hf','--weight-dcp',unusual+' dcp'],
                capture_output=True,text=True,encoding='utf-8',
                env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            config=yaml.safe_load((output/'train_config.yaml').read_text(encoding='utf-8'))
            basic=config['data']['dataset_param']['basic_parameters']
            self.assertEqual(basic['dataset'],[unusual+'.json'])
            self.assertEqual(basic['dataset_dir'],unusual)
            self.assertEqual(config['model']['model_name_or_path'],unusual+' hf')
            self.assertEqual(config['data']['dataset_param']['preprocess_parameters']['model_name_or_path'],unusual+' hf')
            self.assertEqual(config['training']['load'],unusual+' dcp')

    def test_legacy_counts(self):
        for field in ('categories', 'summary'):
            self.assertEqual(plan.migration_counts({field: {'a': {'count': 2}, 'b': 0}}),
                             {'a': 2, 'b': 0})

    def test_invalid_contracts_fail(self):
        for payload in ([], {}, {'points': {}}, {'points': []},
                        {'points': {'a': 2}}, {'points': {'a': [{}]}},
                        {'points': {'a': [{'line': True, 'sym': 'x'}]}},
                        {'categories': {'a': -1}}, {'summary': {'a': True}},
                        {'points': None, 'summary': {'a': 1}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                plan.migration_counts(payload)

    def test_empty_hit_list_is_valid(self):
        self.assertEqual(plan.migration_counts({'points': {'full_attn': []}}), {'full_attn': 0})

    def test_selfcheck_source_is_preserved(self):
        payload = {'points': {'full_attn': []}, 'source_tag': 'SELF_CHECK'}
        self.assertIn('SELF_CHECK', plan.build_plan_md({}, {}, payload, 'x', [], []))

    def test_config_only_report_is_explicit(self):
        self.assertIn('迁移映射未验证', plan.build_plan_md({}, {}, {}, 'x', [], []))

    def test_bad_files_fail_before_output_and_preserve_existing_config(self):
        with test_directory() as tmp:
            root = Path(tmp)
            out = root / 'out'
            out.mkdir()
            config = out / 'train_config.yaml'
            config.write_text('sentinel', encoding='utf-8')
            cases = [('missing.json', None), ('broken.json', '{'), ('bad.json', '{"points": []}')]
            for filename, content in cases:
                path = root / filename
                if content is not None:
                    path.write_text(content, encoding='utf-8')
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / '20_plan_migration.py'),
                     '--points', str(path), '--out', str(out)],
                    capture_output=True, text=True, encoding='utf-8',
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('P1', result.stderr)
                self.assertEqual(config.read_text(encoding='utf-8'), 'sentinel')
                self.assertFalse((out / 'migrate_plan.md').exists())


class OperatorStatusTests(unittest.TestCase):
    @staticmethod
    def torch_stubs():
        modules = {name: types.ModuleType(name) for name in
                   ('torch', 'torch.nn', 'torch.nn.functional')}
        modules['torch'].device = lambda value: value
        modules['torch'].nn = modules['torch.nn']
        modules['torch.nn'].functional = modules['torch.nn.functional']
        return modules

    def setUp(self):
        with mock.patch.dict(sys.modules, self.torch_stubs()):
            self.ops = load_script('30_verify_ops.py')

    def test_success_and_partial_are_distinct(self):
        full = self.ops.summarize_matrix([{'status': 'forward_ok'}])
        partial = self.ops.summarize_matrix([
            {'status': 'forward_ok'}, {'status': 'contract-only'}])
        self.assertEqual(full['status'], 'PASSED')
        self.assertEqual(partial['status'], 'PARTIAL')
        self.assertEqual(partial['counts']['forward_ok'], 1)
        self.assertEqual(partial['exit_code'], 0)

    def test_contract_only_is_not_forward_success(self):
        result = self.ops.summarize_matrix([{'status': 'contract-only'}])
        self.assertEqual(result['status'], 'CONTRACT_ONLY')
        self.assertEqual(result['counts']['forward_ok'], 0)

    def test_empty_unknown_and_error_fail(self):
        for rows in ([], [{'status': 'unknown'}], [{'status': 'error'}]):
            with self.subTest(rows=rows):
                result = self.ops.summarize_matrix(rows)
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(result['exit_code'], 1)

    def test_operator_exception_is_captured_and_fails(self):
        # Inject a failing Conv3d rather than substitute an already-failed matrix.
        with mock.patch.object(self.ops.nn, 'Conv3d', create=True,
                               side_effect=RuntimeError('injected operator failure')):
            rows = self.ops.run_matrix('cpu')
        self.assertEqual(rows[0]['status'], 'error')
        self.assertIn('injected operator failure', rows[0]['err'])
        self.assertEqual(rows[-1]['status'], 'contract-only')
        self.assertEqual(self.ops.summarize_matrix(rows)['exit_code'], 1)

    def run_cli(self, tmp, rows):
        # Execute the actual __main__ guard with hardware calls replaced in an
        # isolated child process. This verifies the OS exit code and saved JSON.
        helper = r"""
import json, runpy, sys, types
from unittest import mock
modules = {name: types.ModuleType(name) for name in ('torch', 'torch.nn', 'torch.nn.functional')}
modules['torch'].device = lambda value: value
modules['torch'].nn = modules['torch.nn']
modules['torch.nn'].functional = modules['torch.nn.functional']
script, outdir, rows_json = sys.argv[1:]
rows = json.loads(rows_json)
def trace(frame, event, arg):
    if frame.f_code.co_filename == script and frame.f_code.co_name == 'main' and event == 'call':
        frame.f_globals['probe_backend'] = lambda: ('cpu', 'test-double CPU')
        frame.f_globals['run_matrix'] = lambda device: rows
        sys.settrace(None)
    return trace
sys.argv = [script, '--out', outdir]
with mock.patch.dict(sys.modules, modules):
    sys.settrace(trace)
    runpy.run_path(script, run_name='__main__')
"""
        return subprocess.run(
            [sys.executable, '-c', helper, str(SCRIPTS / '30_verify_ops.py'),
             tmp, json.dumps(rows)], capture_output=True, text=True, encoding='utf-8',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

    def test_cli_failure_saves_evidence_and_exits_nonzero(self):
        with test_directory() as tmp:
            result = self.run_cli(tmp, [{'op': 'broken', 'status': 'error', 'err': 'injected'}])
            self.assertEqual(result.returncode, 1, result.stderr)
            data = json.loads((Path(tmp) / 'ops_matrix.json').read_text(encoding='utf-8'))
            self.assertEqual(data['summary']['status'], 'FAILED')
            self.assertEqual(data['matrix'][0]['err'], 'injected')
            self.assertIn('VERIFY_FAILED', result.stdout)
            self.assertNotIn('3 类真 forward 贯通', result.stdout)

    def test_cli_partial_keeps_contract_boundary(self):
        rows = [{'op': 'real', 'status': 'forward_ok', 'in_shape': [1], 'out_shape': [1]},
                {'op': 'gdn', 'status': 'contract-only',
                 'shape_contract': {'in': '[B,T,H,K]', 'out': '[B,T,H,V]'}}]
        with test_directory() as tmp:
            result = self.run_cli(tmp, rows)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(tmp) / 'ops_matrix.json').read_text(encoding='utf-8'))
            self.assertEqual(data['summary']['status'], 'PARTIAL')
            self.assertEqual(data['summary']['counts']['forward_ok'], 1)
            self.assertIn('VERIFY_PARTIAL', result.stdout)


if __name__ == '__main__':
    unittest.main()
