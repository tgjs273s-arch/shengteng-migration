#!/usr/bin/env python3
"""T1 diagnostic sweep, serialized, GBS=8. Outputs are not official benchmarks.

Run on NPU: python3 97_cpu_thread_sweep.py --skill /root/qwen35-ascend-migrator
    --config /root/qwen35-ascend-migrator/out/plan/train_config.yaml --out /root/ops/t1_results
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def geometry(doc):
    tr = doc['training']
    dp = doc['parallel']['data_parallel_size']
    if (dp, tr['micro_batch_size'], tr['gradient_accumulation_steps']) != (2, 4, 1):
        raise ValueError('T1 requires dp=2 / mbs=4 / gas=1')


def node_identity():
    """节点身份：让每组证据**自证来源**（坑 122/124 同族：引用证据必须能自证出处）。

    没有它，判断"这一组跑在哪台节点上"只能靠目录时间戳倒推 —— 本轮复核 T1 三组时
    正是这么倒推的（见 docs/CODEX_T1_T4_20260916.md 的 superseded 说明）。
    """
    return {'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'hostname': os.uname().nodename,
            'affinity': sorted(os.sched_getaffinity(0))}


def same_identity(ref, cur):
    """身份的**消费者**：中途换节点必须报错，而不是被静默平均掉。

    只写不读的字段属于坑表中的"声明了但无人消费"家族，因此这里显式消费它。
    """
    if json.dumps(ref, sort_keys=True) != json.dumps(cur, sort_keys=True):
        raise ValueError('node identity changed mid-sweep: %s -> %s' % (ref, cur))


def counters():
    for root in ('/sys/fs/cgroup', '/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct'):
        p = Path(root) / 'cpu.stat'
        if p.exists():
            return {'path': str(p), 'values': {a: int(b) for a, b in
                    (ln.split() for ln in p.read_text().splitlines())}}
    raise RuntimeError('No cgroup cpu.stat: refuse to infer zero throttling')


def delta(before, after):
    if before['path'] != after['path']:
        raise ValueError('cgroup identity changed')
    out = {k: after['values'][k] - v for k, v in before['values'].items()}
    if any(v < 0 for v in out.values()):
        raise ValueError('cgroup counters reset')
    return out


def valid_run(rc, train_rc, rows, text):
    return (rc == 0 and train_rc == ['0'] and
            [i for i, _ in rows] == list(range(1, 31)) and
            '507018' not in text and 'ChildFailedError' not in text)


def selftest():
    good = {'parallel': {'data_parallel_size': 2},
            'training': {'micro_batch_size': 4, 'gradient_accumulation_steps': 1}}
    geometry(good)
    bad = copy.deepcopy(good)
    bad['parallel']['data_parallel_size'] = 1
    bad['training']['gradient_accumulation_steps'] = 2
    try:
        geometry(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('compensating GBS geometry was accepted')
    before = {'path': 'x', 'values': {'throttled_time': 80}}
    assert delta(before, before)['throttled_time'] == 0
    try:
        delta(before, {'path': 'x', 'values': {'throttled_time': 2}})
    except ValueError:
        pass
    else:
        raise AssertionError('counter reset was accepted')
    rows = [(i, 1.0) for i in range(1, 31)]
    assert valid_run(0, ['0'], rows, '')
    assert not valid_run(0, ['1'], rows, '')
    assert not valid_run(0, ['0'], rows[:-1], '')
    assert not valid_run(0, ['0'], rows, 'ACL 507018')
    ident = {'boot': 'b1', 'hostname': 'h1', 'affinity': [0, 1]}
    same_identity(ident, copy.deepcopy(ident))
    moved = copy.deepcopy(ident)
    moved['boot'] = 'b2'
    try:
        same_identity(ident, moved)
    except ValueError:
        pass
    else:
        raise AssertionError('node change mid-sweep was accepted')
    print('CPU_SWEEP_SELFTEST_OK cases=10 good geometry; bad geometry; delta; reset; '
          'good run; bad exit; missing step; ACL; same identity; changed identity')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--skill', default='/root/qwen35-ascend-migrator')
    ap.add_argument('--config', default='/root/qwen35-ascend-migrator/out/plan/train_config.yaml')
    ap.add_argument('--out', default='/root/ops/t1_results')
    ap.add_argument('--start-index', type=int, choices=range(9), default=0,
                    help='Explicit restart in a NEW output directory; prior cases are not claimed here')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    import fcntl
    import yaml
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = open('/tmp/qwen35_t1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = p.read_bytes().split(b'\0')
            if any(x.endswith(b'/trainer.py') or x.endswith(b'/50_train.py') for x in cmd):
                raise RuntimeError('Existing training process: ' + str(p))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    source = Path(args.config).read_bytes()
    doc = yaml.safe_load(source)
    geometry(doc)
    if doc['training'].get('save_format') != 'hf':
        raise ValueError('Expected previously verified hf save workaround')
    identity = node_identity()
    results = []
    # Balanced deterministic order; every case retains official geometry.
    cases = [('default', 8), ('8', 16), ('32', 24), ('default', 16),
             ('8', 24), ('32', 8), ('default', 24), ('8', 8), ('32', 16)]
    for index, (omp, workers) in enumerate(cases):
        if index < args.start_index:
            continue
        tag = '%02d_omp%s_workers%d' % (index, omp, workers)
        case = out / tag
        case.mkdir(exist_ok=False)
        same_identity(identity, node_identity())   # 换节点即中止，不把两台节点的数据并成一组
        cfg = copy.deepcopy(doc)
        cfg['training']['train_iters'] = 30
        cfg['training']['save'] = str(case / 'checkpoint')
        cfg['data']['dataloader_param']['num_workers'] = workers
        geometry(cfg)
        config = case / 'config.yaml'
        content = yaml.safe_dump(cfg, sort_keys=False)
        assert yaml.safe_load(content) == cfg
        config.write_text(content)
        env = os.environ.copy()
        if omp == 'default':
            env.pop('OMP_NUM_THREADS', None)
        else:
            env['OMP_NUM_THREADS'] = omp
        # MKL stays fixed/inherited across cases; record it, never guess.
        record = {'tag': tag, 'omp_requested': omp, 'workers': workers,
                  'mkl': env.get('MKL_NUM_THREADS'), 'affinity': sorted(os.sched_getaffinity(0)),
                  'identity': identity,                       # ← 证据自证来源
                  'source_sha256': hashlib.sha256(source).hexdigest(),
                  'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                  'diagnostic_only': True, 'window': [11, 30], 'gbs': 8}
        probe_env = env.copy()
        if omp == 'default':
            probe_env['OMP_NUM_THREADS'] = '1'  # torchrun multi-rank default, probe only
        probe = subprocess.run([sys.executable, '-c',
            'import torch; print(torch.get_num_threads())'], env=probe_env,
            capture_output=True, text=True, timeout=60)
        record['torch_thread_probe'] = {'rc': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr,
                                         'scope': 'separate process with matching expected torchrun env'}
        record['before'] = counters()
        started = time.time()
        log = case / 'train.log'
        print('START ' + tag, flush=True)
        with (case / 'driver.log').open('w') as driver:
            r = subprocess.run([sys.executable, str(Path(args.skill) / 'scripts/50_train.py'),
                '--config', str(config), '--env', str(Path(args.skill) / 'out/probe/env.json'),
                '--log', str(log), '--workdir', '/root/MindSpeed-MM', '--steps', '30',
                '--world-size', '2', '--timeout', '900', '--foreground'],
                cwd=args.skill, env=env, stdout=driver, stderr=subprocess.STDOUT)
        record['wall_seconds'] = time.time() - started
        record['after'] = counters()
        record['delta'] = delta(record['before'], record['after'])
        driver_text = (case / 'driver.log').read_text(errors='replace')
        text = log.read_text(errors='replace') if log.exists() else ''
        rows = [(int(i), float(ms)) for i, ms in re.findall(
            r'iteration\s+(\d+)\s*/\s*30.*?elapsed time per iteration \(ms\):\s*([\d.]+)', text)]
        record['driver_rc'] = r.returncode
        record['train_rc'] = re.findall(r'train_rc=(\d+)', driver_text)
        record['rows'] = rows
        record['valid'] = valid_run(r.returncode, record['train_rc'], rows, text)
        window = [ms for i, ms in rows if 11 <= i <= 30]
        if record['valid']:
            record['median_ms'] = statistics.median(window)
            record['per_sample_ms'] = record['median_ms'] / 8
        results.append(record)
        (case / 'result.json').write_text(json.dumps(record, indent=2))
        (out / 'results.json').write_text(json.dumps(
            {'identity': identity, 'start_index': args.start_index, 'cases': results}, indent=2))
        print('END ' + tag + ' valid=' + str(record['valid']) +
              ' median_ms=' + str(record.get('median_ms')), flush=True)
        if not record['valid']:
            raise RuntimeError('Training failed: stop sweep; inspect ' + str(case))
    print('CPU_SWEEP_COMPLETE cases=' + str(len(results)), flush=True)


if __name__ == '__main__':
    main()
