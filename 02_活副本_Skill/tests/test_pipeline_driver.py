"""Exercise the real Bash driver with isolated fake stages, never real training."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('PIPELINE_TEST_BASH') or shutil.which('bash')
FAKE = r'''
import json, os
from pathlib import Path
import sys
name = Path(sys.argv[0]).name
stage = {'05_preflight.py':'PRE','00_probe_env.py':'P0','10_analyze_points.py':'P1',
 '20_plan_migration.py':'P2','30_verify_ops.py':'P3','40_prepare_assets.py':'P4',
 '50_train.py':'P5','60_bench.py':'P6','95_extract_series.py':'P6window','70_judge.py':'P7'}[name]
with open('calls.txt','a') as f: f.write(stage+'\n')
Path(stage+'.argv.json').write_text(json.dumps(sys.argv[1:]),encoding='utf-8')
def write(path, data):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data) if not isinstance(data,str) else data,encoding='utf-8')
if os.environ.get('NO_WRITE') == stage: sys.exit(0)
if os.environ.get('SLEEP_STAGE') == stage:
 import time
 time.sleep(5)
if stage=='PRE': write('out/preflight/capability.json', {'summary':{'OK':1}})
if stage=='P0': write('out/probe/env.json', {'path':'test_only','recommended_profile':{}})
if stage=='P1':
 write('out/analyze/migrate_points.json', {'points':{'full_attn':[]}})
 write('out/analyze/migrate_points_report.md','fake analysis')
if stage=='P2':
 write('out/plan/train_config.yaml','training:\n  micro_batch_size: 4\n')
 write('out/plan/migrate_plan.md','fake plan')
if stage=='P3':
 state = os.environ.get('OPS_STATE','partial')
 row={'op':'fake','status':'contract-only' if state=='partial' else ('error' if state=='error' else 'forward_ok')}
 write('out/verify/ops_matrix.json',{'matrix':[row], 'summary':{'status': {'partial':'CONTRACT_ONLY','error':'FAILED','ok':'PASSED'}[state]}})
if stage=='P4':
 write('out/assets/assets.json', {'assets':{},'data_comparability':'test_only'})
 print('ASSETS_OK')
if stage=='P5': write('out/train/train.log','iteration 1 fake only\n')
if stage=='P6': write('out/bench/round_1_baseline.json', {'scope':'test_only'})
if stage=='P6window':
 write('out/bench/loss_series.csv','iteration,loss\n50,1\n')
 write('out/bench/window_50_100.json', {'steps_used':1,'median_ms':1,'samples_per_s':1})
if stage=='P7':
 for name in ('fp','obs'): write('out/judge/'+name+'.json',{'test_only':True})
 write('out/judge/verdict.json',{'verdict_id':'test-id'})
 write('out/judge/judge_summary.json',{'verdict_id':'test-id','verdict':'NEEDS_EVIDENCE','level':'none'})
if os.environ.get('FAIL_STAGE') == stage: sys.exit(7)
'''
NAMES = ['05_preflight.py','00_probe_env.py','10_analyze_points.py','20_plan_migration.py',
         '30_verify_ops.py','40_prepare_assets.py','50_train.py','60_bench.py',
         '95_extract_series.py','70_judge.py']


@unittest.skipUnless(BASH, 'Bash is required for driver integration tests')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        artifacts = ROOT.parent / 'tmp' / 'pipeline_tests'
        artifacts.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='pipeline-test-', dir=artifacts))
        # Preserve artifacts for inspection; never recursively delete test directories.
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for name in ('run_from_zero.sh','_stage_state.py'):
            shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
        for name in NAMES:
            (scripts / name).write_text(FAKE, encoding='utf-8')
        shim = self.root / 'bin'
        shim.mkdir()
        # Resolve the test's interpreter instead of Windows Store's python3 alias.
        exe = Path(sys.executable).as_posix()
        if os.name == 'nt':
            exe = '/' + exe[0].lower() + exe[2:]
        wrapper = shim / 'python3'
        wrapper.write_text('#!/bin/bash\nexec "'+exe+'" "$@"\n', encoding='utf-8', newline='\n')
        wrapper.chmod(0o755)

    def run_driver(self, args=(), **settings):
        env = {**os.environ, 'PYTHONIOENCODING':'utf-8', **settings}
        result = subprocess.run(
            [BASH, '-c', 'export PATH="$PWD/bin:$PATH"; exec bash scripts/run_from_zero.sh "$@"',
             'pipeline-test', *args], cwd=self.root, env=env,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        self.last = result
        return result

    def statuses(self):
        text = (self.root/'out/logs/_verdicts.tsv').read_text(encoding='utf-8')
        return {cols[0]:cols[2] for cols in (line.split('\t') for line in text.splitlines())}

    def calls(self):
        p=self.root/'calls.txt'
        return p.read_text().splitlines() if p.exists() else []

    def test_training_failure_blocks_consumers_but_keeps_evidence(self):
        result=self.run_driver(FAIL_STAGE='P5')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        status=self.statuses()
        self.assertEqual(status['P5'],'FAIL')
        self.assertEqual(status['P6'],'BLOCKED')
        self.assertEqual(status['P7'],'BLOCKED')
        self.assertNotIn('P6',self.calls())
        self.assertTrue((self.root/'out/train/train.log').exists())

    def test_upstream_failure_allows_independent_diagnostics(self):
        result=self.run_driver(FAIL_STAGE='P1')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        status=self.statuses()
        self.assertEqual(status['P2'],'BLOCKED')
        self.assertEqual(status['P3'],'PARTIAL')
        self.assertEqual(status['P4'],'PASS')
        self.assertNotIn('P5',self.calls())

    def test_unchanged_old_artifacts_do_not_pass_even_with_zero_exit(self):
        out=self.root/'out/plan'
        out.mkdir(parents=True)
        for name in ('train_config.yaml','migrate_plan.md'):
            (out/name).write_text('old artifact',encoding='utf-8')
        result=self.run_driver(NO_WRITE='P2')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P2'],'FAIL')
        self.assertEqual((out/'train_config.yaml').read_text(),'old artifact')
        self.assertNotIn('P5',self.calls())

    def test_partial_is_not_whole_pipeline_success(self):
        result=self.run_driver()
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P3'],'PARTIAL')
        self.assertEqual(self.statuses()['P5'],'PASS')
        # Prefix receipts can be reused, but the partial boundary must survive.
        result=self.run_driver(('--from','P6'))
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P3'],'RESUMED')
        self.assertIn('PARTIAL=1',result.stdout)
        # A changed upstream artifact invalidates the receipt and all consumers.
        (self.root/'out/plan/train_config.yaml').write_text('tampered',encoding='utf-8')
        result=self.run_driver(('--from','P6'))
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P2'],'BLOCKED')
        self.assertEqual(self.statuses()['P6'],'BLOCKED')

    def test_early_multicommand_failure_is_not_hidden(self):
        result=self.run_driver(FAIL_STAGE='P6',OPS_STATE='ok')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertNotIn('P6window',self.calls())
        self.assertEqual(self.statuses()['P7'],'PASS')

    def test_bad_from_and_missing_receipts(self):
        result=self.run_driver(('--from','UNKNOWN'))
        self.assertEqual(result.returncode,2)
        self.assertFalse((self.root/'out').exists())
        result=self.run_driver(('--from','P5'))
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P5'],'BLOCKED')
        self.assertEqual(self.calls(),[])

    def test_custom_data_path_reaches_all_consumers(self):
        relative = "data 中文 # quote' $HOME"
        result=self.run_driver(('--data-dir',relative))
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        expected=str((self.root/relative).resolve())
        def arg(stage,key):
            values=json.loads((self.root/(stage+'.argv.json')).read_text(encoding='utf-8'))
            return values[values.index(key)+1]
        self.assertEqual(arg('P0','--data-dir'),expected)
        self.assertEqual(arg('P4','--data-dir'),expected)
        self.assertEqual(Path(arg('P2','--data-dir')),Path(expected)/'coco')
        for stage in ('P2','P7'):
            self.assertEqual(Path(arg(stage,'--data-json')),Path(expected)/'output_llava_coco_data.json')

    def test_asset_converter_shell_preserves_arguments(self):
        spec=importlib.util.spec_from_file_location('assets_for_test',ROOT/'scripts/40_prepare_assets.py')
        assets=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assets)
        values=["data 中文 # quote' $HOME", 'part with spaces']
        # Execute only a print stub, never the actual model/data converter.
        payload="import json,sys; print(json.dumps(sys.argv[1:],ensure_ascii=False))"
        command=assets.conversion_command(str(self.root),[sys.executable,'-c',payload,*values])
        if os.name=='nt':
            self.skipTest('Converter shell is Linux-only; run this case under WSL')
        result=subprocess.run([BASH,'-c',command],capture_output=True,text=True,encoding='utf-8',timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),values)

    def test_timeout_blocks_dependents(self):
        driver=self.root/'scripts/run_from_zero.sh'
        driver.write_text(driver.read_text(encoding='utf-8').replace(
            '"迁移点识别(AST)" 600', '"迁移点识别(AST)" 0.1'), encoding='utf-8')
        result=self.run_driver(SLEEP_STAGE='P1')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P1'],'TIMEOUT')
        self.assertEqual(self.statuses()['P2'],'BLOCKED')
        self.assertNotIn('P5',self.calls())

    def test_operator_error_cannot_pass_even_if_command_returns_zero(self):
        result=self.run_driver(OPS_STATE='error')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P3'],'FAIL')
        self.assertEqual(self.statuses()['P5'],'BLOCKED')


if __name__=='__main__':
    unittest.main()
