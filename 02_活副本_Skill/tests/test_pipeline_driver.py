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
import hashlib, json, os
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
 write('out/plan/train_config.yaml','training:\n  micro_batch_size: 4\n  gradient_accumulation_steps: 1\n  train_iters: 100\nparallel:\n  data_parallel_size: 2\n')
 write('out/plan/migrate_plan.md','fake plan')
 write('out/plan/reference_config.yaml','training:\n  train_iters: 100\n')
 write('out/plan/config_manifest.json',{'schema':'migrator_config.v1','role':'reference',
       'effective_config':{'sha256':hashlib.sha256(Path('out/plan/train_config.yaml').read_bytes()).hexdigest()},
       'reference_config':{'sha256':hashlib.sha256(Path('out/plan/reference_config.yaml').read_bytes()).hexdigest()}})
if stage=='P3':
 state = os.environ.get('OPS_STATE','partial')
 row={'op':'fake','status':'contract-only' if state=='partial' else ('error' if state=='error' else 'forward_ok')}
 summary={'partial':'CONTRACT_ONLY','error':'FAILED','ok':'PASSED'}[state]
 if os.environ.get('OPS_SUMMARY_BAD'): summary='PASSED'
 write('out/verify/ops_matrix.json',{'matrix':[row], 'summary':{'status':summary}})
if stage=='P4':
 write('out/assets/assets.json', {'schema':'migrator_assets.v2','assets':{},
       'missing_required':[],'migration_identity_state':'awaiting_T03_manifest_reference',
       'readiness':'local_complete_migration_unbound','data_comparability':'test_only'})
 print('ASSETS_OK')
if stage=='P5':
 write('out/train/train.log',''.join(
     f'[Rank 0 | Local Rank 0] iteration {step}/100 | consumed samples: {step * 8} | '
     'elapsed time per iteration (ms): 100.0 | learning rate: 1.0E-6 | '
     'global batch size: 8 | loss: 1.0 | grad norm: 2.0 |\n'
     for step in range(1,101)))
 run=Path('out/train/runs/p5-current'); run.mkdir(parents=True)
 config=Path('out/plan/train_config.yaml').read_bytes()
 snap=run/'effective_config.yaml'; snap.write_bytes(config)
 sha=hashlib.sha256(config).hexdigest()
 logsha=hashlib.sha256(Path('out/train/train.log').read_bytes()).hexdigest()
 write(run/'runner.sh','#!/bin/bash\nexit 0\n')
 write(run/'runner.log','train_rc=0\n')
 write(run/'run.json',{'run_id':run.name,'source_config_sha256':sha,
       'effective_config_sha256':sha,'log':str(Path('out/train/train.log').resolve()),
       'asset_binding':{'state':'UNBOUND'}})
 write(run/'train_integrity.json',{'schema':'train_integrity.v1','state':'COMPLETE',
       'train_rc':0,'run_id':run.name,'source_config_sha256':sha,
       'config':str(snap.resolve()),'config_sha256':sha,'world_size':2,
       'gbs_values':[8],'log_sha256':logsha,
       'log':str(Path('out/train/train.log').resolve()),
       'config_train_iters':100,'resume_start':1,'selected_count':100})
if stage=='P6': write('out/bench/round_1_baseline.json', {'scope':'test_only'})
if stage=='P6window':
 write('out/bench/loss_series.csv','iteration,loss\n50,1\n')
 write('out/bench/window_50_100.json', {'schema':'train_performance.v2',
       'state':'COMPLETE','source_type':'training_iteration_log',
       'log_sha256':hashlib.sha256(Path('out/train/train.log').read_bytes()).hexdigest(),
       'gbs':8,'gbs_source':'log_and_config_checked','integrity':{'state':'COMPLETE'},
       'official_selection':{'metrics':{'points':100}},
       'window_selection':{'metrics':{'points':51}},'steps_used':51,'median_ms':1,'samples_per_s':1})
if stage=='P7':
 attempt=Path('out/judge/attempt-current'); attempt.mkdir(parents=True)
 for name in ('fp','obs'): write('out/judge/'+name+'.json',{'test_only':True})
 write('out/judge/verdict.json',{'verdict_id':'test-id'})
 summary={'execution_state':'COMPLETED','attempt_dir':str(attempt.resolve()),
          'verdict_id':'test-id','verdict':'NEEDS_EVIDENCE','level':'none',
          'rule_status':'RULE_PENDING','numeric_acceptance':'NOT_DETERMINED',
          'numeric_open_gates':['rule_pending'],'comparability':{'verdict':'NEEDS_EVIDENCE'},
          'numeric_validity':'VALID_MEASUREMENT'}
 write('out/judge/judge_summary.json',summary)
 for name in ('fp','obs','verdict','judge_summary'):
  (attempt/(name+'.json')).write_bytes((Path('out/judge')/(name+'.json')).read_bytes())
 if os.environ.get('BREAK_AGGREGATE'):
  Path('out/stage_state/P2.json').write_text('{"status":"INVALID"}',encoding='utf-8')
if os.environ.get('FAIL_STAGE') == stage: sys.exit(7)
'''
NAMES = ['05_preflight.py','00_probe_env.py','10_analyze_points.py','20_plan_migration.py',
         '30_verify_ops.py','40_prepare_assets.py','50_train.py','60_bench.py',
         '95_extract_series.py','70_judge.py']

# Isolated fixture shims exercise the real driver/receipt wiring only. T09's
# actual artifact validation, model loading and NPU generation have separate tests.
FAKE_SCENE_VERIFY = r'''
import hashlib, json
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_artifact(manifest_path, model_dir, processor_dir):
 manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
 assert manifest['schema']=='model_artifact.v1' and manifest['status']=='RELOAD_VERIFIED'
 assert manifest['reload_verified'] is True
 assert Path(model_dir).resolve()==Path(processor_dir).resolve()==Path(manifest['export_hf']['path']).resolve()
 return manifest, {'sha256':'fixture-inventory'}, sha(manifest_path)
'''
FAKE_SCENE = r'''
import argparse, hashlib, json, uuid
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ap=argparse.ArgumentParser()
for name in ('model-artifact','model-dir','processor-dir','image','sop','device','attention','max-new-tokens','out'):
 ap.add_argument('--'+name, required=True)
ap.add_argument('--cpu-diagnostic',action='store_true')
a=ap.parse_args()
with open('calls.txt','a') as f: f.write('P10\n')
Path('P10.argv.json').write_text(json.dumps(vars(a)),encoding='utf-8')
attempt=Path(a.out).resolve()/('attempt-'+uuid.uuid4().hex)
attempt.mkdir(parents=True)
artifact=json.loads(Path(a.model_artifact).read_text(encoding='utf-8'))
evidence={}
for key,name,content in (('generated_ids','generated_ids.json','{"new_token_ids":[1]}'),
                         ('raw_generation','raw_generation.txt','{"compliant":"无法判断"}'),
                         ('parsed_result','parsed_result.json','{"compliant":"无法判断"}')):
 path=attempt/name; path.write_text(content,encoding='utf-8')
 evidence[key]={'path':str(path),'sha256':sha(path)}
result={'schema':'inference_result.v1','inference_run_id':attempt.name,
 'attempt_dir':str(attempt),'status':'EVIDENCE_INSUFFICIENT',
 'execution_state':'NPU_EXECUTED','real_inference_verified':True,
 'business_validity':'NOT_EVALUATED',**evidence,
 'model_artifact':{'path':str(Path(a.model_artifact).resolve()),
   'sha256':sha(a.model_artifact),'model_artifact_id':artifact['model_artifact_id'],
   'export_inventory_sha256':'fixture-inventory'},
 'inputs':{'image':{'path':str(Path(a.image).resolve()),'sha256':sha(a.image)},
           'sop':{'path':str(Path(a.sop).resolve()),'sha256':sha(a.sop)}}}
(attempt/'inference_result.json').write_text(json.dumps(result),encoding='utf-8')
print(attempt/'inference_result.json')
'''


@unittest.skipUnless(BASH, 'Bash is required for driver integration tests')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        artifacts = ROOT.parent / 'tmp' / 'pipeline_tests'
        artifacts.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='pipeline-test-', dir=artifacts))
        # Preserve artifacts for inspection; never recursively delete test directories.
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for name in ('run_from_zero.sh','_stage_state.py','_pipeline_integration.py','_train_log.py'):
            shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
        for name in NAMES:
            (scripts / name).write_text(FAKE, encoding='utf-8')
        (scripts / '_scene_inference.py').write_text(FAKE_SCENE_VERIFY, encoding='utf-8')
        (scripts / '65_scene.py').write_text(FAKE_SCENE, encoding='utf-8')
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
        self.assertEqual(status['P4'],'PARTIAL')
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
        self.assertEqual(self.statuses()['P5'],'PARTIAL')
        # Prefix receipts can be reused, but the partial boundary must survive.
        result=self.run_driver(('--from','P6'))
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P3'],'RESUMED')
        self.assertIn('PARTIAL=5',result.stdout)
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
        self.assertEqual(self.statuses()['P7'],'PARTIAL')

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

    def test_operator_summary_cannot_pass_contract_only_matrix(self):
        result=self.run_driver(OPS_SUMMARY_BAD='1')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P3'],'FAIL')
        self.assertEqual(self.statuses()['P5'],'BLOCKED')

    def test_default_diagnostic_leaves_model_and_scene_not_run(self):
        result=self.run_driver()
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertEqual(self.statuses()['P9'],'SKIP')
        self.assertEqual(self.statuses()['P10'],'SKIP')
        summary=json.loads((self.root/'out/acceptance/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['layers']['model_artifact'],'NOT_RUN')
        self.assertEqual(summary['layers']['inference'],'NOT_RUN')

    def test_inference_requires_explicit_verified_local_inputs(self):
        result=self.run_driver(('--flow','inference'))
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertEqual(self.calls(),[])
        result=self.run_driver(('--flow','inference','--image','missing.png','--sop','missing.txt',
                                '--model-artifact','missing.json','--model-dir','missing-model',
                                '--processor-dir','missing-model'))
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertEqual(self.calls(),[])

    def test_training_flow_rejects_unrelated_external_model(self):
        result=self.run_driver(('--flow','reference','--migration-bundle','bundle.json',
                                '--migration-overlay','overlay','--run-inference',
                                '--image','image.png','--sop','sop.txt',
                                '--model-artifact','other-run.json','--model-dir','other-model',
                                '--processor-dir','other-model'))
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertIn('同次 --export-model',result.stdout)
        self.assertEqual(self.calls(),[])

    def test_inference_only_writes_current_scene_attempt_without_prior_stages(self):
        exported=self.root/'model/attempt-source/export_hf'
        exported.mkdir(parents=True)
        (exported/'config.json').write_text('{}',encoding='utf-8')
        artifact=exported.parent/'model_artifact.json'
        artifact.write_text(json.dumps({'schema':'model_artifact.v1',
            'status':'RELOAD_VERIFIED','reload_verified':True,
            'model_artifact_id':'fixture-model','export_hf':{'path':str(exported)}}),encoding='utf-8')
        image=self.root/'scene.png'; image.write_bytes(b'fixture-image')
        sop=self.root/'sop.txt'; sop.write_text('fixture SOP',encoding='utf-8')
        result=self.run_driver(('--flow','inference','--model-artifact',str(artifact),
                                '--model-dir',str(exported),'--processor-dir',str(exported),
                                '--image',str(image),'--sop',str(sop)))
        self.assertEqual(result.returncode,3,result.stdout+result.stderr)
        self.assertEqual(self.calls(),['P10'])
        self.assertEqual(self.statuses(),{'P10':'PARTIAL'})
        argv=json.loads((self.root/'P10.argv.json').read_text(encoding='utf-8'))
        self.assertEqual(argv['model_artifact'],str(artifact))
        summary=json.loads((self.root/'out/acceptance/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['overall'],'PARTIAL')
        self.assertEqual(summary['layers']['inference']['execution_state'],'NPU_EXECUTED')
        self.assertEqual(summary['layers']['inference']['business_validity'],'NOT_EVALUATED')
        self.assertEqual(len(list((self.root/'out/scene').glob('attempt-*'))),1)

    def test_aggregation_failure_overwrites_old_acceptance_and_report_does_not_link_it(self):
        old=self.root/'out/acceptance/summary.json'
        old.parent.mkdir(parents=True)
        old.write_text('{"overall":"PASS","old":true}',encoding='utf-8')
        result=self.run_driver(BREAK_AGGREGATE='1')
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        summary=json.loads(old.read_text(encoding='utf-8'))
        self.assertEqual(summary['overall'],'FAIL')
        self.assertEqual(summary['open_gates'],['aggregation_failed'])
        report=(self.root/'out/from_zero_report.md').read_text(encoding='utf-8')
        self.assertIn('旧 out/acceptance/summary.json 不能作为本轮结论',report)
        self.assertNotIn('机器可读结果: [out/acceptance/summary.json]',report)


if __name__=='__main__':
    unittest.main()
