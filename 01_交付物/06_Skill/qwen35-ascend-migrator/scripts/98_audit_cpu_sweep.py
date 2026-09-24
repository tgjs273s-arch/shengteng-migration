#!/usr/bin/env python3
"""Audit T1 tar archives against raw logs and configurations without extracting files.

Diagnostic window 11-30 only. Incomplete runs are reported, never upgraded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import tarfile
import yaml

PAT = re.compile(r'iteration\s+(\d+)\s*/\s*30.*?elapsed time per iteration \(ms\):\s*([\d.]+)')


def check(record, raw, config, driver):
    cfg = yaml.safe_load(config)
    tr = cfg['training']
    if (cfg['parallel']['data_parallel_size'], tr['micro_batch_size'],
            tr['gradient_accumulation_steps'], tr['train_iters']) != (2, 4, 1, 30):
        raise ValueError('configuration geometry mismatch')
    if cfg['data']['dataloader_param']['num_workers'] != record['workers']:
        raise ValueError('configuration worker count mismatch')
    rows = [(int(i), float(ms)) for i, ms in PAT.findall(raw)]
    if [i for i, _ in rows] != list(range(1, 31)):
        raise ValueError('missing/duplicate/out-of-order steps')
    if record['rows'] != [list(row) for row in rows]:
        raise ValueError('result/raw iteration mismatch')
    if hashlib.sha256(config).hexdigest() != record['config_sha256']:
        raise ValueError('configuration hash mismatch')
    if re.findall(r'train_rc=(\d+)', driver) != ['0'] or record['train_rc'] != ['0'] or record['driver_rc'] != 0:
        raise ValueError('exit status not confirmed')
    if '507018' in raw or 'ChildFailedError' in raw:
        raise ValueError('training error in raw log')
    if not record['valid'] or record['gbs'] != 8 or record['window'] != [11, 30]:
        raise ValueError('wrong diagnostic contract')
    actual_gbs = re.findall(r'global batch size:\s*(\d+)', raw)
    if len(actual_gbs) != 30 or set(actual_gbs) != {'8'}:
        raise ValueError('runtime GBS not consistently 8')
    med = statistics.median([ms for i, ms in rows if 11 <= i <= 30])
    if med != record['median_ms']:
        raise ValueError('median mismatch')
    before, after = record['before'], record['after']
    if before['path'] != after['path']:
        raise ValueError('cgroup path mismatch')
    delta = {k: after['values'][k] - v for k, v in before['values'].items()}
    if any(v < 0 for v in delta.values()) or delta != record['delta']:
        raise ValueError('counter reset or delta mismatch')
    return {'tag': record['tag'], 'omp': record['omp_requested'], 'workers': record['workers'],
            'median_ms': med, 'per_sample_ms': med / 8,
            'nr_throttled_delta': delta.get('nr_throttled'),
            'throttled_ns_delta': delta.get('throttled_time'),
            'throttled_usec_delta': delta.get('throttled_usec'),
            'wall_seconds': record['wall_seconds'], 'source_sha256': record['source_sha256'],
            'thread_probe': record['torch_thread_probe'], 'window': [11, 30]}


def selftest():
    import copy
    cfg = yaml.safe_dump({'parallel': {'data_parallel_size': 2},
        'training': {'micro_batch_size': 4, 'gradient_accumulation_steps': 1, 'train_iters': 30},
        'data': {'dataloader_param': {'num_workers': 8}}}).encode()
    raw = '\n'.join('iteration %d/30 elapsed time per iteration (ms): 10 global batch size: 8' % i for i in range(1, 31))
    r = {'rows': [[i, 10.0] for i in range(1, 31)], 'config_sha256': hashlib.sha256(cfg).hexdigest(),
         'train_rc': ['0'], 'driver_rc': 0, 'valid': True, 'gbs': 8, 'window': [11, 30],
         'median_ms': 10.0, 'before': {'path': 'x', 'values': {'nr_throttled': 1}},
         'after': {'path': 'x', 'values': {'nr_throttled': 1}}, 'delta': {'nr_throttled': 0},
         'tag': 'synthetic', 'omp_requested': '8', 'workers': 8, 'wall_seconds': 1,
         'source_sha256': 'synthetic', 'torch_thread_probe': {}}
    assert check(r, raw, cfg, 'train_rc=0')['median_ms'] == 10
    bad = copy.deepcopy(r)
    bad['median_ms'] = 2
    wrong_worker = copy.deepcopy(r)
    wrong_worker['workers'] = 16
    wrong_geometry = cfg.replace(b'data_parallel_size: 2', b'data_parallel_size: 1')
    for rr, log, config, drv in [(bad, raw, cfg, 'train_rc=0'), (r, raw, cfg + b'\n', 'train_rc=0'),
            (r, raw, cfg, 'train_rc=1'), (r, raw.replace('size: 8', 'size: 4'), cfg, 'train_rc=0'),
            (wrong_worker, raw, cfg, 'train_rc=0'), (r, raw, wrong_geometry, 'train_rc=0')]:
        try:
            check(rr, log, config, drv)
        except ValueError:
            continue
        raise AssertionError('bad evidence accepted')
    print('CPU_AUDIT_SELFTEST_OK cases=7')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archives', nargs='+')
    ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.archives:
        ap.error('--archives is required')
    report = {'schema': 'cpu_sweep_audit.v1', 'diagnostic_only': True,
              'window': [11, 30], 'verified': [], 'incomplete': [], 'errors': []}
    for path in a.archives:
        with tarfile.open(path, 'r:gz') as tf:
            files = {m.name: m for m in tf.getmembers() if m.isfile()}
            def read(name):
                return tf.extractfile(files[name]).read()
            for name in sorted(files):
                if not name.endswith('/config.yaml'):
                    continue
                case = name.rsplit('/', 1)[0]
                if case + '/result.json' not in files:
                    report['incomplete'].append({'archive': str(path), 'case': case, 'reason': 'result/exit evidence absent'})
                    continue
                try:
                    r = json.loads(read(case + '/result.json'))
                    item = check(r, read(case + '/train.log').decode(errors='replace'),
                                 read(name), read(case + '/driver.log').decode(errors='replace'))
                    item['archive'] = str(path)
                    item['archive_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    report['verified'].append(item)
                except Exception as e:
                    report['errors'].append({'archive': str(path), 'case': case, 'error': str(e)})
    report['distinct_cases'] = len({r['tag'] for r in report['verified']})
    report['all_nine_observed'] = report['distinct_cases'] == 9 and not report['errors']
    report['limitations'] = ['30-step diagnostic runs, not official 50-100 metrics',
        'cgroup deltas cover whole run, including load/save',
        'different archives may span a container restart; no cross-epoch causal attribution',
        'separate thread probe is not a measurement inside the training rank',
        'no run-to-run confidence interval without repeated same-condition runs']
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(data, encoding='utf-8')
    print(data)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
