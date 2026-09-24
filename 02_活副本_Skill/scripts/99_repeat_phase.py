#!/usr/bin/env python3
"""Six serial 100-step CPU diagnostics, GBS=8; raw polling evidence and repeat uncertainty.

Requires explicit --execute and a new --out directory on Linux/NPU.
--audit-archive reads a tar.gz without extracting. No numerical accuracy verdict.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import subprocess
import sys
import tarfile
import time
import yaml

ORDER = ['default', '32', '32', 'default', 'default', '32']
PAT = re.compile(r'iteration\s+(\d+)\s*/\s*100.*?elapsed time per iteration \(ms\):\s*([\d.]+)')
spec = importlib.util.spec_from_file_location('cpu_sweep', Path(__file__).with_name('97_cpu_thread_sweep.py'))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def phase(last):
    if last == 0:
        return 'before_first_iteration_observed'
    if last < 49:
        return 'early_iterations_observed'
    if last < 100:
        return 'steady_interior_observed'
    return 'after_last_iteration_observed'


def phase_deltas(samples):
    if len(samples) < 2:
        raise ValueError('insufficient samples')
    groups = {}
    for a, b in zip(samples, samples[1:]):
        if a['identity'] != b['identity'] or b['last_iter'] < a['last_iter'] or b['t'] <= a['t']:
            raise ValueError('identity/time/iteration regression')
        d = sweep.delta(a['cpu'], b['cpu'])
        label = phase(a['last_iter']) if phase(a['last_iter']) == phase(b['last_iter']) else 'boundary_uncertain'
        g = groups.setdefault(label, {'seconds': 0., 'intervals': 0, 'max_gap_seconds': 0., 'delta': {k: 0 for k in d}})
        elapsed = b['t'] - a['t']
        g['seconds'] += elapsed
        g['max_gap_seconds'] = max(g['max_gap_seconds'], elapsed)
        g['intervals'] += 1
        for k, v in d.items():
            g['delta'][k] += v
    total = sweep.delta(samples[0]['cpu'], samples[-1]['cpu'])
    if any(sum(g['delta'][k] for g in groups.values()) != v for k, v in total.items()):
        raise ValueError('phase/total mismatch')
    return {'groups': groups, 'total': total}


def interval(values):
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('requires three finite independent run values')
    mean = st.mean(values)
    sd = st.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)  # Student t(.975, df=2)
    return {'n_runs': 3, 'values': values, 'mean': mean, 'sample_sd': sd,
            'min': min(values), 'max': max(values), 'mean_ci95_t_df2': [mean-half, mean+half],
            'assumption': 'independent approximately normal run values; n=3 is weak evidence'}


def analyze(raw, driver, config, source, samples, meta):
    cfg, base = yaml.safe_load(config), yaml.safe_load(source)
    sweep.geometry(cfg)
    sweep.geometry(base)
    expected = copy.deepcopy(base)
    expected['training']['train_iters'] = 100
    expected['training']['save'] = cfg['training']['save']
    expected['data']['dataloader_param']['num_workers'] = 24
    if cfg != expected or cfg['training'].get('save_format') != 'hf':
        raise ValueError('unexpected configuration change')
    if digest(config) != meta['config_sha256'] or digest(source) != meta['source_sha256']:
        raise ValueError('hash mismatch')
    rows = [(int(i), float(ms)) for i, ms in PAT.findall(raw)]
    if [i for i, ms in rows] != list(range(1, 101)) or any(not math.isfinite(ms) or ms <= 0 for i, ms in rows):
        raise ValueError('missing/invalid iterations')
    gbs = re.findall(r'global batch size:\s*(\d+)', raw)
    if len(gbs) != 100 or set(gbs) != {'8'}:
        raise ValueError('runtime GBS mismatch')
    if meta['driver_rc'] != 0 or re.findall(r'train_rc=(\d+)', driver) != ['0']:
        raise ValueError('missing successful exit')
    if '507018' in raw or 'ChildFailedError' in raw:
        raise ValueError('training error')
    if samples[0]['last_iter'] != 0 or samples[-1]['last_iter'] != 100:
        raise ValueError('sampling coverage incomplete')
    if any(s['last_iter'] not in range(101) for s in samples):
        raise ValueError('invalid sampled iteration')
    return {'median_ms_50_100': st.median([ms for i, ms in rows if i >= 50]),
            'steps_used': 51, 'phase_counters': phase_deltas(samples), 'valid': True}


def snapshot(log):
    text = log.read_text(errors='replace') if log.exists() else ''
    rows = PAT.findall(text)
    cpu = sweep.counters()
    p = Path(cpu['path'])
    root = p.parent
    quota = {name: (root/name).read_text() for name in
             ('cpu.max', 'cpu.cfs_quota_us', 'cpu.cfs_period_us') if (root/name).exists()}
    pid1_start = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'t': time.monotonic(), 'wall': time.time(), 'last_iter': int(rows[-1][0]) if rows else 0,
            'cpu': cpu, 'identity': {'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid1_start': pid1_start, 'inode': p.stat().st_ino, 'quota': quota}}


def summarize(results):
    if len(results) != 6 or [r['omp'] for r in results] != ORDER:
        raise ValueError('incomplete or wrong-order suite')
    if len({r['source_sha256'] for r in results}) != 1 or len({json.dumps(r['identity'], sort_keys=True) for r in results}) != 1:
        raise ValueError('cross-source/epoch suite')
    if len({r['mkl'] for r in results}) != 1:
        raise ValueError('MKL changed')
    groups = {omp: interval([r['median_ms_50_100'] for r in results if r['omp'] == omp]) for omp in ('default', '32')}
    diffs = []
    for i in range(0, 6, 2):
        pair = {r['omp']: r['median_ms_50_100'] for r in results[i:i+2]}
        diffs.append(pair['32'] - pair['default'])
    return {'complete': True, 'groups': groups, 'paired_omp32_minus_default_ms': interval(diffs),
            'limitations': ['log-observation phases, not instrumented exact load/train/save boundaries',
                'steady interior is observed after iteration 49 until before observing 100; not exact timing of window 50-100',
                'boundary intervals are unassigned; cgroup throttling time is not iteration time lost',
                'polling overhead uncalibrated; all six runs use identical sampling',
                'no in-rank torch.get_num_threads measurement', 'no accuracy verdict or performance acceptance claim']}


def no_training():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if any(x.endswith(b'/trainer.py') or x.endswith(b'/50_train.py') for x in p.read_bytes().split(b'\0')):
                raise RuntimeError('existing training: ' + str(p))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def execute(args):
    import fcntl
    lock = open('/tmp/qwen35_t1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    no_training()
    source = Path(args.config).read_bytes()
    base = yaml.safe_load(source)
    sweep.geometry(base)
    if base['training'].get('save_format') != 'hf':
        raise ValueError('expected HF save workaround')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'source.yaml').write_bytes(source)
    results = []
    for index, omp in enumerate(ORDER):
        no_training()
        case = out/('%02d_omp%s' % (index, omp))
        case.mkdir()
        cfg = copy.deepcopy(base)
        cfg['training']['train_iters'] = 100
        cfg['training']['save'] = str(case/'checkpoint')
        cfg['data']['dataloader_param']['num_workers'] = 24
        config = yaml.safe_dump(cfg, sort_keys=False).encode()
        assert yaml.safe_load(config) == cfg
        (case/'config.yaml').write_bytes(config)
        env = os.environ.copy()
        if omp == 'default':
            env.pop('OMP_NUM_THREADS', None)
        else:
            env['OMP_NUM_THREADS'] = omp
        meta = {'omp': omp, 'mkl': env.get('MKL_NUM_THREADS'), 'config_sha256': digest(config),
                'source_sha256': digest(source), 'affinity': sorted(os.sched_getaffinity(0))}
        (case/'meta.json').write_text(json.dumps(meta, indent=2))
        log = case/'train.log'
        samples = [snapshot(log)]
        print('START ' + case.name, flush=True)
        with (case/'samples.jsonl').open('w') as sf, (case/'driver.log').open('w') as driver:
            sf.write(json.dumps(samples[0])+'\n')
            sf.flush()
            proc = subprocess.Popen([sys.executable, str(Path(args.skill)/'scripts/50_train.py'),
                '--config', str(case/'config.yaml'), '--env', str(Path(args.skill)/'out/probe/env.json'),
                '--log', str(log), '--workdir', '/root/MindSpeed-MM', '--steps', '100',
                '--world-size', '2', '--timeout', '900', '--foreground'], cwd=args.skill, env=env,
                stdout=driver, stderr=subprocess.STDOUT)
            while True:
                time.sleep(0.5)
                samples.append(snapshot(log))
                sf.write(json.dumps(samples[-1])+'\n')
                sf.flush()
                if proc.poll() is not None:
                    break
            meta['driver_rc'] = proc.returncode
        (case/'meta.json').write_text(json.dumps(meta, indent=2))
        result = analyze(log.read_text(errors='replace'), (case/'driver.log').read_text(errors='replace'),
                         config, source, samples, meta)
        result.update(meta, identity=samples[0]['identity'])
        (case/'result.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        print('END ' + case.name + ' median_ms=' + str(result['median_ms_50_100']), flush=True)
    summary = summarize(results)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('REPEAT_PHASE_COMPLETE', flush=True)


def audit(path):
    results = []
    with tarfile.open(path, 'r:gz') as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
        roots = [n.rsplit('/', 1)[0] for n in names if n.endswith('/source.yaml')]
        if len(roots) != 1:
            raise ValueError('requires one suite root')
        root = roots[0]
        def read(name):
            return tf.extractfile(name).read()
        source = read(root+'/source.yaml')
        for index, omp in enumerate(ORDER):
            case = root + '/%02d_omp%s' % (index, omp)
            meta = json.loads(read(case+'/meta.json'))
            if meta['omp'] != omp:
                raise ValueError('OMP/order mismatch')
            samples = [json.loads(line) for line in read(case+'/samples.jsonl').splitlines()]
            result = analyze(read(case+'/train.log').decode(errors='replace'), read(case+'/driver.log').decode(errors='replace'),
                             read(case+'/config.yaml'), source, samples, meta)
            result.update(meta, identity=samples[0]['identity'])
            if result != json.loads(read(case+'/result.json')):
                raise ValueError('derived result mismatch')
            results.append(result)
        summary = summarize(results)
        if summary != json.loads(read(root+'/summary.json')):
            raise ValueError('summary mismatch')
    return {'archive_sha256': digest(Path(path).read_bytes()), 'results': results, 'summary': summary}


def markdown(report):
    lines = ['# T3 重复实验：归档回读生成', '', '证据 SHA256：`'+report['archive_sha256']+'`', '',
             '固定 GBS=8 / dp=2 / mbs=4 / gas=1 / workers=24；每次 100 步，性能窗口 50–100。', '',
             '| 运行序号 | OMP | 窗口中位 ms | 全程节流次数增量 |', '|---|---|---|---|']
    for i, r in enumerate(report['results']):
        lines.append('| %s | %s | %.3f | %s |' % (i, r['omp'], r['median_ms_50_100'],
                     r['phase_counters']['total'].get('nr_throttled', 'unavailable')))
    lines += ['', '## 运行级误差棒', '',
              '以下区间针对三次运行中位数的均值；不是总体中位数区间，也不以迭代数充当重复次数。', '',
              '| 分组/配对差 | 均值 ms | 样本标准差 ms | 均值 95% t 区间 ms |', '|---|---|---|---|']
    groups = dict(report['summary']['groups'])
    groups['OMP32-default 配对差'] = report['summary']['paired_omp32_minus_default_ms']
    for name, g in groups.items():
        lo, hi = g['mean_ci95_t_df2']
        lines.append('| %s | %.3f | %.3f | [%.3f, %.3f] |' % (name, g['mean'], g['sample_sd'], lo, hi))
    lines += ['', '## 日志观测阶段的节流增量', '',
              '单位直接取 cpu.stat：v1 为 ns，v2 为 usec；不是迭代损失时间。跨阶段区间独立列出。', '',
              '| 运行 | 阶段 | 节流计数 | 时间增量 | 单位 | 最大采样间隔 s |', '|---|---|---|---|---|---|']
    for i, r in enumerate(report['results']):
        for label, g in r['phase_counters']['groups'].items():
            d = g['delta']
            key = 'throttled_time' if 'throttled_time' in d else 'throttled_usec'
            unit = 'ns' if key == 'throttled_time' else 'usec'
            lines.append('| %s | %s | %s | %s | %s | %.3f |' %
                         (i, label, d.get('nr_throttled', 'unavailable'), d.get(key, 'unavailable'), unit, g['max_gap_seconds']))
    lines += ['', '## 限制', ''] + ['- '+s for s in report['summary']['limitations']]
    lines += ['', '- t 区间依赖运行值独立、近似正态假设，仅三次重复，证据较弱。',
              '- 精度仍为 NEEDS_EVIDENCE（黄）；data_identity/sample_order 未闭合；不宣称性能达标。', '']
    return '\n'.join(lines)


def selftest():
    s = lambda t, i, v: {'t': t, 'last_iter': i, 'identity': 'same', 'cpu': {'path': 'x', 'values': {'throttled_time': v}}}
    assert phase_deltas([s(0, 0, 0), s(1, 0, 2)])['groups']['before_first_iteration_observed']['delta']['throttled_time'] == 2
    assert phase_deltas([s(0, 48, 0), s(1, 50, 2)])['groups']['boundary_uncertain']['delta']['throttled_time'] == 2
    assert interval([10., 10., 10.])['mean_ci95_t_df2'] == [10., 10.]
    bad = [lambda: phase_deltas([s(0, 0, 2), s(1, 0, 0)]),
           lambda: phase_deltas([s(0, 50, 0), s(1, 48, 0)]), lambda: interval([10., 10.]),
           lambda: phase_deltas([s(1, 0, 0), s(0, 0, 0)])]
    for f in bad:
        try:
            f()
        except ValueError:
            continue
        raise AssertionError('bad sample accepted')
    cfg = yaml.safe_dump({'parallel': {'data_parallel_size': 2},
        'training': {'micro_batch_size': 4, 'gradient_accumulation_steps': 1,
                     'train_iters': 100, 'save_format': 'hf', 'save': '/synthetic'},
        'data': {'dataloader_param': {'num_workers': 24}}}).encode()
    raw = '\n'.join('iteration %d/100 elapsed time per iteration (ms): 10 global batch size: 8' % i for i in range(1, 101))
    meta = {'config_sha256': digest(cfg), 'source_sha256': digest(cfg), 'driver_rc': 0}
    samples = [s(0, 0, 0), s(1, 100, 2)]
    assert analyze(raw, 'train_rc=0', cfg, cfg, samples, meta)['median_ms_50_100'] == 10
    bad_meta = dict(meta, config_sha256='bad')
    for log, drv, m in [(raw, 'train_rc=1', meta), (raw.replace('size: 8', 'size: 4'), 'train_rc=0', meta),
                         ('\n'.join(raw.splitlines()[:-1]), 'train_rc=0', meta), (raw, 'train_rc=0', bad_meta)]:
        try:
            analyze(log, drv, cfg, cfg, samples, m)
        except ValueError:
            continue
        raise AssertionError('bad run evidence accepted')
    good = [dict(omp=o, source_sha256='same', identity={'epoch': 1}, mkl=None, median_ms_50_100=10.) for o in ORDER]
    assert summarize(good)['complete']
    variants = [good[:-1]]
    for key, value in [('source_sha256', 'changed'), ('identity', {'epoch': 2}), ('mkl', '32'), ('omp', 'unexpected')]:
        bad = copy.deepcopy(good)
        bad[0][key] = value
        variants.append(bad)
    for bad in variants:
        try:
            summarize(bad)
        except ValueError:
            continue
        raise AssertionError('bad suite accepted')
    print('REPEAT_PHASE_SELFTEST_OK cases=18')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--audit-archive')
    ap.add_argument('--report-md', help='optional generated Markdown report, audit mode only')
    ap.add_argument('--out', help='new directory for execution; JSON file for audit')
    ap.add_argument('--skill', default='/root/qwen35-ascend-migrator')
    ap.add_argument('--config', default='/root/qwen35-ascend-migrator/out/plan/train_config.yaml')
    a = ap.parse_args()
    if sum([a.selftest, a.execute, bool(a.audit_archive)]) != 1:
        ap.error('choose exactly one of --selftest, --execute, --audit-archive')
    if a.report_md and not a.audit_archive:
        ap.error('--report-md requires --audit-archive')
    if a.selftest:
        selftest()
    elif a.audit_archive:
        report = audit(a.audit_archive)
        data = json.dumps(report, indent=2)
        if a.out:
            Path(a.out).write_text(data)
        if a.report_md:
            rendered = markdown(report)
            Path(a.report_md).write_text(rendered, encoding='utf-8')
            assert Path(a.report_md).read_text(encoding='utf-8') == rendered
        print(data)
    else:
        if not a.out:
            ap.error('--execute requires --out NEW_DIRECTORY')
        if not Path(a.out).is_absolute():
            ap.error('--execute requires an absolute output path')
        execute(a)


if __name__ == '__main__':
    main()
