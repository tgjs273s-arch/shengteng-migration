"""Local pipeline receipts: no downloads, training, or device imports.

Receipts prove local artifact provenance, not current hardware availability or
external dataset/weight identity. Normal --resume refreshes PRE/P0/P4; --from only revalidates stored evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

OUTPUTS = {
    'PRE': ['out/preflight/capability.json'],
    'P0': ['out/probe/env.json'],
    'P1': ['out/analyze/migrate_points.json', 'out/analyze/migrate_points_report.md'],
    'P2': ['out/plan/train_config.yaml', 'out/plan/migrate_plan.md',
           'out/plan/reference_config.yaml', 'out/plan/config_manifest.json'],
    'P3': ['out/verify/ops_matrix.json'],
    'P4': ['out/assets/assets.json'],
    'P5': ['out/train/train.log'],
    'P6': ['out/bench/round_1_baseline.json', 'out/bench/loss_series.csv', 'out/bench/window_50_100.json'],
    'P7': ['out/judge/fp.json', 'out/judge/obs.json', 'out/judge/verdict.json', 'out/judge/judge_summary.json'],
    'P8': ['out/reportability/reportability.json'],
    'P9': [],
    'P10': [],
}
DEPS = {'PRE': [], 'P0': [], 'P1': [], 'P2': ['P0', 'P1'],
        'P3': ['P0'], 'P4': ['P0'], 'P5': ['PRE', 'P2', 'P3', 'P4'],
        'P6': ['P5'], 'P7': ['P2', 'P4', 'P5'],
        'P8': ['P5', 'P6', 'P7'], 'P9': ['P4', 'P5'], 'P10': []}
STATE_DIR = Path('out/stage_state')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_digest():
    h = hashlib.sha256()
    for root in ('scripts', 'config', 'refs', 'sk04_judge/scripts', 'sk04_judge/configs'):
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                h.update(path.as_posix().encode('utf-8'))
                h.update(digest(path).encode('ascii'))
    return h.hexdigest()


def receipt_path(stage):
    return STATE_DIR / (stage + '.json')


def read(stage):
    return json.loads(receipt_path(stage).read_text(encoding='utf-8'))


def write(stage, data):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    target = receipt_path(stage)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)


def context():
    return os.environ.get('PIPELINE_CONTEXT', '')


def flow():
    return os.environ.get('PIPELINE_FLOW', 'diagnostic')


def dependencies(stage):
    return DEPS[stage] + (['P9'] if stage == 'P10' and flow() != 'inference' else [])


def _inside(path, parent):
    try:
        return Path(path).resolve().is_relative_to(Path(parent).resolve())
    except (OSError, ValueError, TypeError):
        return False


def _dynamic_outputs(stage, item):
    if stage == 'P5':
        run_dir = item.get('run_dir')
        if not run_dir or not _inside(run_dir, 'out/train/runs') or not Path(run_dir).name.startswith('p5-'):
            raise ValueError('P5 run directory missing or outside current runs')
        return [str(Path(run_dir) / name) for name in
                ('run.json', 'train_integrity.json', 'effective_config.yaml', 'runner.sh', 'runner.log')]
    if stage == 'P7':
        attempt = item.get('attempt_dir')
        if not attempt or not _inside(attempt, 'out/judge') or not Path(attempt).name.startswith('attempt-'):
            raise ValueError('P7 attempt directory missing or outside current judge output')
        return [str(Path(attempt) / name) for name in
                ('fp.json', 'obs.json', 'verdict.json', 'judge_summary.json')]
    if stage in ('P9', 'P10'):
        attempt = item.get('attempt_dir')
        parent = 'out/model' if stage == 'P9' else 'out/scene'
        name = 'model_artifact.json' if stage == 'P9' else 'inference_result.json'
        if not attempt or not _inside(attempt, parent) or not Path(attempt).name.startswith('attempt-'):
            raise ValueError(stage + ' attempt missing or outside current output')
        files = sorted(str(p) for p in Path(attempt).rglob('*') if p.is_file())
        if str(Path(attempt) / name) not in files:
            raise ValueError(stage + ' primary receipt missing')
        return files
    return []


def _current_attempt(stage, item):
    if stage == 'P5':
        before = set(item.get('before_runs', []))
        root = Path('out/train/runs')
        new = [p for p in root.glob('p5-*') if p.is_dir() and p.name not in before]
        if len(new) != 1:
            raise ValueError('P5 must create exactly one new run directory')
        item['run_dir'] = str(new[0].resolve())
    elif stage == 'P7':
        summary = json.loads(Path('out/judge/judge_summary.json').read_text(encoding='utf-8'))
        attempt = summary.get('attempt_dir')
        before = set(item.get('before_attempts', []))
        if not attempt or Path(attempt).name in before:
            raise ValueError('P7 summary does not name a new attempt')
        item['attempt_dir'] = str(Path(attempt).resolve())
    elif stage == 'P9':
        before = set(item.get('before_model_attempts', []))
        root = Path('out/model')
        new = [p for p in root.glob('attempt-*') if p.is_dir() and p.name not in before]
        if len(new) != 1:
            raise ValueError('P9 must create exactly one new model attempt')
        item['attempt_dir'] = str(new[0].resolve())
    elif stage == 'P10':
        before = set(item.get('before_scene_attempts', []))
        root = Path('out/scene')
        new = [p for p in root.glob('attempt-*') if p.is_dir() and p.name not in before]
        if len(new) != 1:
            raise ValueError('P10 must create exactly one new scene attempt')
        item['attempt_dir'] = str(new[0].resolve())
    _dynamic_outputs(stage, item)


def _semantic_status(stage, item, live=False):
    """Stage execution status, distinct from official numeric/business acceptance."""
    if stage == 'P2':
        manifest = json.loads(Path('out/plan/config_manifest.json').read_text(encoding='utf-8'))
        effective = manifest.get('effective_config') or {}
        reference = manifest.get('reference_config') or {}
        if (manifest.get('schema') != 'migrator_config.v1' or
                effective.get('sha256') != digest('out/plan/train_config.yaml') or
                reference.get('sha256') != digest('out/plan/reference_config.yaml')):
            raise ValueError('P2 config manifest/file identity mismatch')
        expected_role = 'candidate' if flow() == 'candidate' else 'reference'
        if manifest.get('role') != expected_role:
            raise ValueError('P2 config role differs from requested flow')
        return 'PASS'
    if stage == 'P3':
        data = json.loads(Path('out/verify/ops_matrix.json').read_text(encoding='utf-8'))
        rows = data.get('matrix', [])
        if not rows or any(r.get('status') not in ('forward_ok', 'contract-only') for r in rows):
            raise ValueError('operator matrix contains an error or no rows')
        status = 'PARTIAL' if any(r['status'] == 'contract-only' for r in rows) else 'PASS'
        expected = ('PARTIAL' if any(r['status'] == 'forward_ok' for r in rows) else
                    'CONTRACT_ONLY') if status == 'PARTIAL' else 'PASSED'
        if data.get('summary', {}).get('status') != expected:
            raise ValueError('P3 matrix/summary disagreement')
        if flow() != 'diagnostic':
            if not all(data.get(k) for k in ('migration_manifest', 'migration_bundle', 'target_checkout')):
                raise ValueError('P3 target migration inputs missing')
            if live:
                from _qwen35_migration import verify_target_checkout
                verify_target_checkout(data['target_checkout'], require_patched=True)
        return status
    if stage == 'P4':
        data = json.loads(Path('out/assets/assets.json').read_text(encoding='utf-8'))
        if data.get('schema') != 'migrator_assets.v2' or data.get('missing_required'):
            raise ValueError('P4 assets incomplete')
        if flow() != 'diagnostic' and (data.get('readiness') != 'local_complete_official_unverified'
                                       or data.get('migration_identity_state') != 'validated_overlay_and_target'):
            raise ValueError('P4 migration/assets not bound for requested flow')
        return 'PARTIAL' if data.get('migration_identity_state') != 'validated_overlay_and_target' else 'PASS'
    if stage == 'P5':
        run_dir = Path(item['run_dir'])
        train = json.loads((run_dir / 'train_integrity.json').read_text(encoding='utf-8'))
        run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        binding = run.get('asset_binding') or {}
        from _train_log import read_log, integrity
        parsed = read_log('out/train/train.log')
        logged_gbs = train.get('gbs_values')
        gbs = logged_gbs[0] if isinstance(logged_gbs, list) and len(logged_gbs) == 1 else None
        observed = integrity(parsed, expected_end=train.get('config_train_iters'),
                             start_step=train.get('resume_start'), expected_gbs=gbs)
        if (train.get('state') != 'COMPLETE' or train.get('train_rc') != 0 or
                observed.get('state') != 'COMPLETE' or
                observed.get('selected_count') != train.get('selected_count') or
                train.get('run_id') != run.get('run_id') or train.get('run_id') != run_dir.name or
                train.get('source_config_sha256') != run.get('source_config_sha256') or
                train.get('config_sha256') != run.get('effective_config_sha256') or
                train.get('source_config_sha256') != digest('out/plan/train_config.yaml') or
                train.get('config_sha256') != digest(run_dir / 'effective_config.yaml') or
                train.get('log_sha256') != digest('out/train/train.log') or
                train.get('log') != str(Path('out/train/train.log').resolve()) or
                run.get('log') != str(Path('out/train/train.log').resolve())):
            raise ValueError('P5 run/config/log integrity mismatch')
        if flow() != 'diagnostic' and (binding.get('state') != 'VERIFIED' or
                                       train.get('asset_binding', {}).get('state') != 'VERIFIED'):
            raise ValueError('P5 current asset binding not verified')
        if binding.get('state') == 'VERIFIED' and (
                binding.get('prelaunch') != binding.get('postrun') or
                train.get('asset_binding', {}).get('run_record_sha256') != digest(run_dir / 'run.json')):
            raise ValueError('P5 claimed bound receipt does not match current run')
        if flow() != 'diagnostic' and live:
            from _runtime_asset_binding import capture
            before = binding.get('prelaunch')
            if (not isinstance(before, dict) or before != binding.get('postrun') or
                    train.get('asset_binding', {}).get('run_record_sha256') != digest(run_dir / 'run.json')):
                raise ValueError('P5 asset/run receipt changed')
            current = capture(before['assets_json_path'], before['migration_bundle'],
                              before['migration_overlay'], before['target_checkout'],
                              (run_dir / 'effective_config.yaml').read_bytes())
            if current != before:
                raise ValueError('P5 external model/data/target identity changed')
        return 'PARTIAL' if binding.get('state') != 'VERIFIED' else 'PASS'
    if stage == 'P6':
        perf = json.loads(Path('out/bench/window_50_100.json').read_text(encoding='utf-8'))
        from _pipeline_integration import effective_gbs
        gbs = effective_gbs(Path(read('P5')['run_dir']) / 'train_integrity.json')
        if (perf.get('schema') != 'train_performance.v2' or
                perf.get('log_sha256') != digest('out/train/train.log') or
                perf.get('source_type') != 'training_iteration_log' or
                perf.get('gbs') != gbs or
                perf.get('gbs_source') != 'log_and_config_checked' or
                perf.get('integrity', {}).get('state') != 'COMPLETE'):
            raise ValueError('P6 performance provenance mismatch')
        if perf.get('state') != 'COMPLETE':
            if flow() == 'diagnostic' and set(perf.get('problems') or []) <= {
                    'window_incomplete', 'official_selection_incomplete'}:
                return 'PARTIAL'
            raise ValueError('P6 training performance selection incomplete')
        official = perf.get('official_selection', {}).get('metrics') or {}
        window = perf.get('window_selection', {}).get('metrics') or {}
        if (official.get('points') != 100 or window.get('points') != 51 or
                perf.get('steps_used') != 51):
            raise ValueError('P6 GBS or 100/51 point selection unverified')
        return 'PARTIAL' if flow() == 'diagnostic' else 'PASS'
    if stage == 'P7':
        summary = json.loads(Path('out/judge/judge_summary.json').read_text(encoding='utf-8'))
        verdict = json.loads(Path('out/judge/verdict.json').read_text(encoding='utf-8'))
        attempt = Path(item['attempt_dir'])
        if (summary.get('execution_state') != 'COMPLETED' or
                Path(summary.get('attempt_dir', '')).resolve() != attempt.resolve() or
                not verdict.get('verdict_id') or summary.get('verdict_id') != verdict['verdict_id'] or
                json.loads((attempt / 'judge_summary.json').read_text(encoding='utf-8')) != summary):
            raise ValueError('P7 current attempt and summary disagree')
        if flow() != 'diagnostic' and summary.get('source_binding', {}).get('state') != 'LOCAL_BINDING_VERIFIED':
            raise ValueError('P7 source binding rejected')
        # Execution can finish with red/yellow, while the formal rule remains pending.
        return 'PARTIAL' if (summary.get('rule_status') != 'RULE_ACCEPTED' or
                             summary.get('numeric_acceptance') != 'PASS' or
                             summary.get('numeric_open_gates')) else 'PASS'
    if stage == 'P8':
        report = json.loads(Path('out/reportability/reportability.json').read_text(encoding='utf-8'))
        judge = json.loads(Path('out/judge/judge_summary.json').read_text(encoding='utf-8'))
        if (report.get('run_validity') != 'UNVERIFIED' or
                report.get('accuracy_validity') != 'UNVERIFIED' or
                judge.get('rule_status') != 'RULE_PENDING' or
                report.get('adoption_ready') is True):
            raise ValueError('P8 candidate/reportability identity or rule state mismatch')
        return 'PARTIAL'
    if stage == 'P9':
        artifact = json.loads((Path(item['attempt_dir']) / 'model_artifact.json').read_text(encoding='utf-8'))
        train = json.loads((Path(read('P5')['run_dir']) / 'train_integrity.json').read_text(encoding='utf-8'))
        assets = json.loads(Path('out/assets/assets.json').read_text(encoding='utf-8'))
        if (artifact.get('schema') != 'model_artifact.v1' or
                artifact.get('attempt_id') != Path(item['attempt_dir']).name or
                artifact.get('train_run', {}).get('run_id') != train.get('run_id') or
                artifact.get('migration_id') != assets.get('migration_id') or
                not artifact.get('model_artifact_id')):
            raise ValueError('P9 model artifact source identity mismatch')
        if artifact.get('status') != 'RELOAD_VERIFIED' or artifact.get('reload_verified') is not True:
            raise ValueError('P9 model reload not verified')
        if live:
            from _scene_inference import verify_artifact
            path = Path(item['attempt_dir']) / 'model_artifact.json'
            exported = (artifact.get('export_hf') or {}).get('path')
            if not exported:
                raise ValueError('P9 verified export path missing')
            verify_artifact(path, exported, exported)
        return 'PASS'
    if stage == 'P10':
        attempt = Path(item['attempt_dir'])
        result = json.loads((attempt / 'inference_result.json').read_text(encoding='utf-8'))
        artifact_path = os.environ.get('PIPELINE_MODEL_ARTIFACT')
        model_dir = os.environ.get('PIPELINE_INFER_MODEL_DIR')
        processor_dir = os.environ.get('PIPELINE_PROCESSOR_DIR')
        if not all((artifact_path, model_dir, processor_dir)):
            raise ValueError('P10 model inputs missing')
        from _scene_inference import verify_artifact
        artifact, inventory, artifact_sha = verify_artifact(
            artifact_path, model_dir, processor_dir)
        if flow() != 'inference' and (
                Path(read('P9')['attempt_dir']) / 'model_artifact.json').resolve() != Path(artifact_path).resolve():
            raise ValueError('P10 did not consume current P9 export')
        source = result.get('model_artifact') or {}
        if (result.get('schema') != 'inference_result.v1' or
                result.get('inference_run_id') != attempt.name or
                Path(result.get('attempt_dir', '')).resolve() != attempt.resolve() or
                result.get('status') not in ('EXECUTED_PARSED', 'EVIDENCE_INSUFFICIENT') or
                result.get('execution_state') not in ('NPU_EXECUTED', 'CPU_DIAGNOSTIC') or
                result.get('business_validity') != 'NOT_EVALUATED' or
                result.get('real_inference_verified') is not
                (result.get('execution_state') == 'NPU_EXECUTED') or
                source.get('path') != str(Path(artifact_path).resolve()) or
                source.get('sha256') != artifact_sha or
                source.get('model_artifact_id') != artifact['model_artifact_id'] or
                source.get('export_inventory_sha256') != inventory['sha256']):
            raise ValueError('P10 current inference/model receipt mismatch')
        for key, name in (('generated_ids', 'generated_ids.json'),
                          ('raw_generation', 'raw_generation.txt'),
                          ('parsed_result', 'parsed_result.json')):
            evidence = result.get(key) or {}
            path = attempt / name
            if (evidence.get('path') != str(path) or not path.is_file() or
                    evidence.get('sha256') != digest(path)):
                raise ValueError('P10 missing or changed raw/parsed evidence: ' + name)
        inputs = result.get('inputs') or {}
        for name in ('image', 'sop'):
            evidence = inputs.get(name) or {}
            path = Path(evidence.get('path', ''))
            if not path.is_file() or digest(path) != evidence.get('sha256'):
                raise ValueError('P10 current scene input changed: ' + name)
        return 'PARTIAL'  # Scene business quality has not been evaluated.
    return 'PASS'


def valid(stage):
    return _valid(stage, code_digest(), {})


def _valid(stage, code, checked):
    if stage in checked:
        return checked[stage]
    checked[stage] = False
    try:
        item = read(stage)
        if (item['status'] not in ('PASS', 'PARTIAL') or item['context'] != context()
                or item['code'] != code):
            return False
        expected = OUTPUTS[stage] + _dynamic_outputs(stage, item) + ['out/logs/' + stage + '.log']
        if set(item['outputs']) != set(expected):
            return False
        if any(not Path(p).is_file() or digest(p) != value for p, value in item['outputs'].items()):
            return False
        for dep in dependencies(stage):
            if not _valid(dep, code, checked) or item['deps'].get(dep) != read(dep)['attempt']:
                return False
        if _semantic_status(stage, item, live=True) != item['status']:
            return False
        checked[stage] = True
        return True
    except Exception:
        return False


def begin(stage):
    for dep in dependencies(stage):
        if not valid(dep):
            raise ValueError('unverified dependency: ' + dep)
    before = {p: Path(p).stat().st_mtime_ns if Path(p).is_file() else None for p in OUTPUTS[stage]}
    before_runs = sorted(p.name for p in Path('out/train/runs').glob('p5-*') if p.is_dir()) if stage == 'P5' else []
    before_attempts = sorted(p.name for p in Path('out/judge').glob('attempt-*') if p.is_dir()) if stage == 'P7' else []
    before_model_attempts = sorted(p.name for p in Path('out/model').glob('attempt-*') if p.is_dir()) if stage == 'P9' else []
    before_scene_attempts = sorted(p.name for p in Path('out/scene').glob('attempt-*') if p.is_dir()) if stage == 'P10' else []
    write(stage, {'status': 'RUNNING', 'attempt': uuid.uuid4().hex,
                  'context': context(), 'code': code_digest(), 'before': before,
                  'before_runs': before_runs, 'before_attempts': before_attempts,
                  'before_model_attempts': before_model_attempts,
                  'before_scene_attempts': before_scene_attempts,
                  'deps': {dep: read(dep)['attempt'] for dep in dependencies(stage)}})


def finish(stage):
    item = read(stage)
    if item['status'] != 'RUNNING' or item['context'] != context() or item['code'] != code_digest():
        raise ValueError('run context changed or no active attempt')
    for dep in dependencies(stage):
        if not valid(dep) or item['deps'].get(dep) != read(dep)['attempt']:
            raise ValueError('dependency changed during run: ' + dep)
    _current_attempt(stage, item)
    for path in OUTPUTS[stage]:
        p = Path(path)
        if not p.is_file() or not p.stat().st_size or p.stat().st_mtime_ns == item['before'][path]:
            raise ValueError('missing, empty, or unchanged old output: ' + path)
    for path in _dynamic_outputs(stage, item):
        p = Path(path)
        if not p.is_file() or not p.stat().st_size:
            raise ValueError('missing or empty current attempt output: ' + path)
    status = _semantic_status(stage, item)
    paths = OUTPUTS[stage] + _dynamic_outputs(stage, item) + ['out/logs/' + stage + '.log']
    item.update(status=status, outputs={p: digest(p) for p in paths})
    write(stage, item)
    print(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'begin', 'finish', 'invalidate', 'artifact'))
    parser.add_argument('stage', choices=OUTPUTS)
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    try:
        if args.action == 'check':
            if not valid(args.stage):
                return 1
            print(read(args.stage)['status'])
        elif args.action == 'artifact':
            if not valid(args.stage):
                raise ValueError('stage receipt invalid')
            item = read(args.stage)
            if args.stage == 'P5' and args.name in ('run_dir', 'run.json', 'train_integrity.json', 'effective_config.yaml'):
                print(item['run_dir'] if args.name == 'run_dir' else str(Path(item['run_dir']) / args.name))
            elif args.stage == 'P7' and args.name in ('attempt_dir', 'judge_summary.json'):
                print(item['attempt_dir'] if args.name == 'attempt_dir' else str(Path(item['attempt_dir']) / args.name))
            elif args.stage in ('P9', 'P10') and args.name in ('attempt_dir', 'model_artifact.json', 'inference_result.json'):
                if args.name != 'attempt_dir' and args.name != ('model_artifact.json' if args.stage == 'P9' else 'inference_result.json'):
                    raise ValueError('unknown dynamic artifact name')
                print(item['attempt_dir'] if args.name == 'attempt_dir' else str(Path(item['attempt_dir']) / args.name))
            else:
                raise ValueError('unknown dynamic artifact name')
        elif args.action == 'begin':
            begin(args.stage)
        elif args.action == 'finish':
            finish(args.stage)
        else:
            write(args.stage, {'status': 'INVALID'})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print('STAGE_STATE_ERROR: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
