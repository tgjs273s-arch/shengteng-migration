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
    'P2': ['out/plan/train_config.yaml', 'out/plan/migrate_plan.md'],
    'P3': ['out/verify/ops_matrix.json'],
    'P4': ['out/assets/assets.json'],
    'P5': ['out/train/train.log'],
    'P6': ['out/bench/round_1_baseline.json', 'out/bench/loss_series.csv', 'out/bench/window_50_100.json'],
    'P7': ['out/judge/fp.json', 'out/judge/obs.json', 'out/judge/verdict.json', 'out/judge/judge_summary.json'],
}
DEPS = {'PRE': [], 'P0': [], 'P1': [], 'P2': ['P0', 'P1'],
        'P3': ['P0'], 'P4': ['P0'], 'P5': ['PRE', 'P2', 'P3', 'P4'],
        'P6': ['P5'], 'P7': ['P2', 'P4', 'P5']}
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
        expected = OUTPUTS[stage] + ['out/logs/' + stage + '.log']
        if set(item['outputs']) != set(expected):
            return False
        if any(not Path(p).is_file() or digest(p) != value for p, value in item['outputs'].items()):
            return False
        for dep in DEPS[stage]:
            if not _valid(dep, code, checked) or item['deps'].get(dep) != read(dep)['attempt']:
                return False
        checked[stage] = True
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def begin(stage):
    for dep in DEPS[stage]:
        if not valid(dep):
            raise ValueError('unverified dependency: ' + dep)
    before = {p: Path(p).stat().st_mtime_ns if Path(p).is_file() else None for p in OUTPUTS[stage]}
    write(stage, {'status': 'RUNNING', 'attempt': uuid.uuid4().hex,
                  'context': context(), 'code': code_digest(), 'before': before,
                  'deps': {dep: read(dep)['attempt'] for dep in DEPS[stage]}})


def finish(stage):
    item = read(stage)
    if item['status'] != 'RUNNING' or item['context'] != context() or item['code'] != code_digest():
        raise ValueError('run context changed or no active attempt')
    for dep in DEPS[stage]:
        if not valid(dep) or item['deps'].get(dep) != read(dep)['attempt']:
            raise ValueError('dependency changed during run: ' + dep)
    for path in OUTPUTS[stage]:
        p = Path(path)
        if not p.is_file() or not p.stat().st_size or p.stat().st_mtime_ns == item['before'][path]:
            raise ValueError('missing, empty, or unchanged old output: ' + path)
    status = 'PASS'
    if stage == 'P3':
        data = json.loads(Path(OUTPUTS[stage][0]).read_text(encoding='utf-8'))
        rows = data.get('matrix', [])
        if not rows or any(r.get('status') not in ('forward_ok', 'contract-only') for r in rows):
            raise ValueError('operator matrix contains an error or no rows')
        status = 'PARTIAL' if any(r['status'] == 'contract-only' for r in rows) else 'PASS'
        expected = ('PARTIAL' if any(r['status'] == 'forward_ok' for r in rows) else 'CONTRACT_ONLY') if status == 'PARTIAL' else 'PASSED'
        if data.get('summary', {}).get('status') != expected:
            raise ValueError('operator summary disagrees with matrix')
    if stage == 'P7':
        verdict = json.loads(Path('out/judge/verdict.json').read_text(encoding='utf-8'))
        summary = json.loads(Path('out/judge/judge_summary.json').read_text(encoding='utf-8'))
        if not verdict.get('verdict_id') or summary.get('verdict_id') != verdict['verdict_id']:
            raise ValueError('judge evidence and summary disagree')
    paths = OUTPUTS[stage] + ['out/logs/' + stage + '.log']
    item.update(status=status, outputs={p: digest(p) for p in paths})
    write(stage, item)
    print(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'begin', 'finish', 'invalidate'))
    parser.add_argument('stage', choices=OUTPUTS)
    args = parser.parse_args()
    try:
        if args.action == 'check':
            if not valid(args.stage):
                return 1
            print(read(args.stage)['status'])
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
