"""Small read-only identity and geometry checks used by the P0–P7 driver."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    if not path:
        return None
    root = Path(path).resolve()
    if root.is_file():
        return {'path': str(root), 'sha256': digest(root)}
    if root.is_dir():
        return {'path': str(root), 'files': [
            {'path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()]}
    raise ValueError('missing input: %s' % path)


def optional_identity(path):
    if not path:
        return None
    if not Path(path).exists():
        return {'path': str(Path(path).resolve()), 'missing': True}
    return identity(path)


def context(args):
    infer = args.flow == 'inference' or args.run_inference == '1'
    data = {'flow': args.flow, 'mode': args.mode, 'steps': args.steps,
            'data_dir': str(Path(args.data_dir).resolve()),
            'asset_model_dir': str(Path(args.asset_model_dir).resolve()),
            'msmm_dir': str(Path(args.msmm_dir).resolve()),
            'no_download': args.no_download,
            'baseline': args.baseline,
            'baseline_log': (optional_identity(args.baseline_log) if args.flow in ('diagnostic', 'inference')
                             else identity(args.baseline_log)),
            'migration_bundle': identity(args.migration_bundle),
            'migration_overlay': identity(args.migration_overlay),
            'ab_summary': identity(args.ab_summary),
            'noise_summary': identity(args.noise_summary),
            'baseline_variant': args.baseline_variant,
            'candidate_variant': args.candidate_variant,
            'export_iteration': args.export_iteration,
            'export_model': args.export_model, 'run_inference': args.run_inference,
            'model_artifact': identity(args.model_artifact) if infer and args.model_artifact else None,
            'inference_model': identity(args.model_dir) if infer and args.model_dir else None,
            'processor': identity(args.processor_dir) if infer and args.processor_dir else None,
            'image': identity(args.image) if infer else None,
            'sop': identity(args.sop) if infer else None,
            'device': args.device, 'attention': args.attention,
            'max_new_tokens': args.max_new_tokens,
            'cpu_diagnostic': args.cpu_diagnostic}
    if args.flow != 'inference' and Path(args.baseline).is_file():
        data['baseline_file'] = identity(args.baseline)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def effective_gbs(train_integrity):
    import yaml
    train = json.loads(Path(train_integrity).read_text(encoding='utf-8'))
    if train.get('schema') != 'train_integrity.v1' or train.get('state') != 'COMPLETE':
        raise ValueError('P5 train integrity incomplete')
    snapshot = Path(train['config'])
    if digest(snapshot) != train.get('config_sha256'):
        raise ValueError('P5 effective config changed')
    config = yaml.safe_load(snapshot.read_text(encoding='utf-8'))
    training, parallel = config['training'], config['parallel']
    mbs = training['micro_batch_size']
    gas = training['gradient_accumulation_steps']
    dp = parallel['data_parallel_size']
    world = train['world_size']
    if any(type(n) is not int or n < 1 for n in (mbs, gas, dp, world)) or dp != world:
        raise ValueError('P5 batch geometry/world mismatch')
    gbs = mbs * gas * dp
    if train.get('gbs_values') != [gbs]:
        raise ValueError('P5 log GBS differs from effective config')
    return gbs


def candidate_inputs(summary_path, noise_path, baseline_variant, candidate_variant):
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    noise = json.loads(Path(noise_path).read_text(encoding='utf-8'))
    if summary.get('schema') not in ('p59ab.v2', 't63_two_config_ab.v1') or \
            noise.get('schema') != 'p59null.v1':
        raise ValueError('candidate comparison/noise schema unsupported')
    if (baseline_variant, candidate_variant) not in (('A', 'B'), ('B', 'A')):
        raise ValueError('baseline/candidate variants must explicitly be distinct A/B')
    source = Path(__file__).with_name('62_reportability.py')
    spec = importlib.util.spec_from_file_location('p62_for_pipeline', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _diffs, pairs, _official, _schema, _ids, reason = module.parse_summary(
        summary, baseline_variant=baseline_variant, candidate_variant=candidate_variant)
    if reason or not pairs:
        raise ValueError('candidate pairs unavailable: %s' % reason)
    # A/B aggregation may span many runs; it does not inherit one P5/P7 identity.
    return pairs


def export_dir(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'model_artifact.v1' or
            manifest.get('status') != 'RELOAD_VERIFIED' or
            manifest.get('reload_verified') is not True):
        raise ValueError('exported model has not passed T08 reload')
    result = Path(manifest['export_hf']['path']).resolve(strict=True)
    if not result.is_dir() or not result.is_relative_to(Path(manifest_path).resolve().parent):
        raise ValueError('model export outside verified attempt')
    return result


def write_summary(output, result):
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(target)


def aggregate(flow, verdicts, output, ledger_bad=False):
    import _stage_state as state
    rows = []
    for line in Path(verdicts).read_text(encoding='utf-8').splitlines():
        cols = line.split('\t')
        if len(cols) != 5:
            raise ValueError('invalid stage verdict row')
        rows.append({'stage': cols[0], 'status': cols[2], 'log': cols[4]})
    stages = {row['stage']: row for row in rows}
    evidence = {}
    for stage, row in stages.items():
        if row['status'] not in ('PASS', 'PARTIAL', 'RESUMED', 'PARTIAL_RESUMED'):
            continue
        if not state.valid(stage):
            raise ValueError('stage receipt changed before aggregation: ' + stage)
        receipt = state.read(stage)
        evidence[stage] = {'attempt': receipt['attempt'],
                           'receipt': str(state.receipt_path(stage)),
                           'receipt_sha256': digest(state.receipt_path(stage)),
                           'status': receipt['status']}
    def usable(name):
        return name in evidence
    layers = {
        'migration': 'STATIC_BOUND' if usable('P4') and
                     json.loads(Path('out/assets/assets.json').read_text(encoding='utf-8')).get(
                         'migration_identity_state') == 'validated_overlay_and_target' else 'UNVERIFIED',
        'operator_numeric': evidence.get('P3', {}).get('status', 'UNVERIFIED'),
        'asset_local': evidence.get('P4', {}).get('status', 'UNVERIFIED'),
        'training': evidence.get('P5', {}).get('status', 'UNVERIFIED'),
        'performance': evidence.get('P6', {}).get('status', 'UNVERIFIED'),
        'comparability': 'UNVERIFIED', 'numeric_validity': 'UNVERIFIED',
        'numeric_acceptance': 'NOT_DETERMINED', 'rule_status': 'RULE_PENDING',
        'reportability': 'NOT_APPLICABLE' if flow != 'candidate' else 'UNVERIFIED',
        'model_artifact': 'NOT_RUN', 'inference': 'NOT_RUN',
        'scene_quality': 'NOT_EVALUATED'}
    open_gates = []
    if usable('P7'):
        judge = json.loads(Path('out/judge/judge_summary.json').read_text(encoding='utf-8'))
        layers.update(comparability=judge.get('comparability'),
                      numeric_validity=judge.get('numeric_validity'),
                      numeric_acceptance=judge.get('numeric_acceptance'),
                      rule_status=judge.get('rule_status'))
        open_gates.extend(judge.get('numeric_open_gates') or [])
    if usable('P8'):
        report = json.loads(Path('out/reportability/reportability.json').read_text(encoding='utf-8'))
        layers['reportability'] = {'performance_pass': report.get('performance_pass'),
                                   'run_validity': report.get('run_validity'),
                                   'accuracy_validity': report.get('accuracy_validity'),
                                   'adoption_ready': report.get('adoption_ready')}
        open_gates.extend(['ab_run_provenance_unverified', 'ab_accuracy_provenance_unverified'])
    if usable('P9'):
        artifact = json.loads((Path(state.read('P9')['attempt_dir']) / 'model_artifact.json').read_text(
            encoding='utf-8'))
        layers['model_artifact'] = {'status': artifact.get('status'),
                                    'reload_verified': artifact.get('reload_verified'),
                                    'model_artifact_id': artifact.get('model_artifact_id')}
    if usable('P10'):
        scene = json.loads((Path(state.read('P10')['attempt_dir']) / 'inference_result.json').read_text(
            encoding='utf-8'))
        if layers['model_artifact'] == 'NOT_RUN':
            layers['model_artifact'] = {'status': 'RELOAD_VERIFIED',
                'model_artifact_id': scene['model_artifact']['model_artifact_id'],
                'source': 'external_verified_T08_artifact'}
        layers['inference'] = {'status': scene['status'],
            'execution_state': scene['execution_state'],
            'real_inference_verified': scene['real_inference_verified'],
            'business_validity': scene['business_validity']}
        open_gates.append('scene_business_quality_not_evaluated')
    failed = bool(ledger_bad) or any(row['status'] in ('FAIL', 'TIMEOUT', 'BLOCKED') for row in rows)
    requested_export = os.environ.get('PIPELINE_EXPORT_MODEL') == '1'
    requested_infer = flow == 'inference' or any(row['stage'] == 'P10' and row['status'] != 'SKIP' for row in rows)
    if requested_export and layers['model_artifact'] == 'NOT_RUN':
        open_gates.append('model_artifact_requested_unavailable')
    if requested_infer and layers['inference'] == 'NOT_RUN':
        open_gates.append('inference_requested_unavailable')
    if ledger_bad:
        open_gates.append('degradation_ledger_inconsistent')
    result = {'schema': 'pipeline_acceptance.v1', 'flow': flow,
              'context_sha256': os.environ.get('PIPELINE_CONTEXT'),
              'overall': 'FAIL' if failed else 'PARTIAL',
              'stages': rows, 'stage_evidence': evidence, 'layers': layers,
              'open_gates': sorted(set(open_gates))}
    write_summary(output, result)
    return result['overall']


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='action', required=True)
    p = sub.add_parser('context')
    for name in ('flow', 'mode', 'steps', 'data-dir', 'asset-model-dir', 'model-dir',
                 'processor-dir', 'model-artifact', 'image', 'sop', 'device',
                 'attention', 'max-new-tokens', 'export-model', 'run-inference',
                 'cpu-diagnostic', 'msmm-dir',
                 'baseline', 'baseline-log', 'migration-bundle', 'migration-overlay',
                 'ab-summary', 'noise-summary', 'baseline-variant', 'candidate-variant',
                 'export-iteration'):
        p.add_argument('--' + name, required=name in ('flow', 'mode', 'steps', 'data-dir',
                                                       'asset-model-dir', 'msmm-dir', 'baseline'))
    p.add_argument('--no-download', action='store_true')
    p = sub.add_parser('gbs')
    p.add_argument('--train-integrity', required=True)
    p = sub.add_parser('candidate')
    for name in ('summary', 'noise', 'baseline-variant', 'candidate-variant'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('aggregate')
    p.add_argument('--flow', required=True)
    p.add_argument('--verdicts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ledger-bad', choices=('0', '1'), default='0')
    p = sub.add_parser('export-dir')
    p.add_argument('--model-artifact', required=True)
    args = ap.parse_args()
    try:
        if args.action == 'context':
            print(context(args))
        elif args.action == 'gbs':
            print(effective_gbs(args.train_integrity))
        elif args.action == 'candidate':
            print(candidate_inputs(args.summary, args.noise,
                                   args.baseline_variant, args.candidate_variant))
        elif args.action == 'export-dir':
            print(export_dir(args.model_artifact))
        else:
            print(aggregate(args.flow, args.verdicts, args.out, args.ledger_bad == '1'))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if args.action == 'aggregate':
            try:
                write_summary(args.out, {'schema': 'pipeline_acceptance.v1',
                    'flow': args.flow, 'context_sha256': os.environ.get('PIPELINE_CONTEXT'),
                    'overall': 'FAIL', 'aggregation_error': str(exc),
                    'open_gates': ['aggregation_failed'], 'layers': {}, 'stage_evidence': {}})
            except OSError:
                pass
        print('PIPELINE_INPUT_ERROR: %s' % exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
