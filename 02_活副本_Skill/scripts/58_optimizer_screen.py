#!/usr/bin/env python3
"""Isolated AdamW fused/foreach screen; NOT model performance or acceptance.

Pre-registered: fixed synthetic FP32 tensors, identical initial state/gradients,
three serial A/B/A blocks, A=fused and B=foreach. Reject a block if baseline
drift >=3%; candidate only if all blocks valid and mean improvement >=5%.
Numerical prerequisite: finite updates, max absolute difference <=1e-6.
This does not measure kernel launches, model loss/grad_norm, or elasticity.
No graph capture, package mutation, model/checkpoint mutation or GBS claims.
Run --selftest without torch; --execute requires a NEW absolute --out path.

Protocol warmed-v2 (separate batch, never relabel legacy evidence): exactly
100 warmup updates per run, then time all 100 updates (steps 101..200).
Same numerical/drift/gain thresholds. One batch only; if drift fails again,
stop this microbenchmark retry route rather than tune warmup to the result.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def protocol_plan(protocol):
    if protocol == 'legacy-v1':
        return {'warmup_steps': 0, 'measurement_steps': 100, 'window': [11, 100]}
    if protocol == 'warmed-v2':
        return {'warmup_steps': 100, 'measurement_steps': 100, 'window': [101, 200]}
    raise ValueError('unknown protocol')


def summarize(rows):
    if len(rows) != 9 or [r['variant'] for r in rows] != list('ABAABAABA'):
        raise ValueError('requires exactly three ABA blocks')
    if len({r['boot'] for r in rows}) != 1:
        raise ValueError('mixed boot identities')
    values = [r['median_ms'] for r in rows]
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('invalid timing')
    blocks = []
    for i in range(0, 9, 3):
        a, b, c = values[i:i+3]
        drift = abs(c-a)/min(a, c)
        blocks.append({'baseline_drift': drift, 'valid': drift < .03,
                       'improvement': 1-b/((a+c)/2)})
    improvements = [b['improvement'] for b in blocks]
    mean = statistics.mean(improvements)
    half = 4.302652729911275*statistics.stdev(improvements)/math.sqrt(3)
    return {'blocks': blocks, 'mean_improvement': mean,
            'ci95_t_df2': [mean-half, mean+half],
            'candidate': all(b['valid'] for b in blocks) and mean >= .05,
            'scope': 'synthetic optimizer screen only; no model speedup claim',
            'statistical_limit': 'n=3; independence/normality not established'}


def selftest():
    assert protocol_plan('legacy-v1')['window'] == [11, 100]
    assert protocol_plan('warmed-v2') == {
        'warmup_steps': 100, 'measurement_steps': 100, 'window': [101, 200]}
    try:
        protocol_plan('adaptive')
    except ValueError:
        pass
    else:
        raise AssertionError('unregistered protocol accepted')
    rows = [dict(variant=v, boot='x', median_ms=10 if v == 'A' else 9)
            for v in 'ABAABAABA']
    assert summarize(rows)['candidate']
    for change in ('missing', 'identity', 'nan'):
        bad = [dict(r) for r in rows]
        if change == 'missing':
            bad.pop()
        elif change == 'identity':
            bad[0]['boot'] = 'y'
        else:
            bad[0]['median_ms'] = float('nan')
        try:
            summarize(bad)
        except ValueError:
            continue
        raise AssertionError('bad sample accepted: '+change)
    rows[2]['median_ms'] = 11
    assert not summarize(rows)['candidate']
    print('OPTIMIZER_SCREEN_SELFTEST_OK good=1 bad_rejected=4')
    print('WARMUP_PROTOCOL_SELFTEST_OK good=2 bad_rejected=1')


def execute(out, protocol='legacy-v1'):
    plan = protocol_plan(protocol)
    import fcntl
    import torch
    import torch_npu
    lock = open('/tmp/qwen35_t1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        if proc.parent.name == str(os.getpid()):
            continue
        try:
            cmd = proc.read_bytes().replace(b'\0', b' ')
        except OSError:
            continue
        if any(x in cmd for x in (b'50_train.py', b'torchrun', b'mindspeed_mm.train')):
            raise RuntimeError('training process exists; refusing concurrent probe')
    if not out.is_absolute():
        raise ValueError('absolute output required')
    out.mkdir(parents=False, exist_ok=False)
    boot_path = Path('/proc/sys/kernel/random/boot_id')
    boot = boot_path.read_text().strip()
    torch.npu.set_device(0)
    seed = torch.linspace(.1, 1., 4096, dtype=torch.float32, device='npu')
    gradient = torch.full_like(seed, .01)
    rows, endpoints = [], {}
    metadata = {'boot': boot, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'tensor_count': 320, 'tensor_numel': 4096, 'dtype': 'float32',
                'steps': 100, 'window': plan['window'], 'synthetic': True,
                'protocol': protocol, 'warmup_steps': plan['warmup_steps'],
                'kernel_launch_count': None, 'elasticity': None,
                'official_comparable': False}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    for index, variant in enumerate('ABAABAABA'):
        if boot_path.read_text().strip() != boot:
            raise RuntimeError('boot changed')
        params = [torch.nn.Parameter(seed.clone()) for _ in range(320)]
        for p in params:
            p.grad = gradient.clone()
        optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .95),
                                     eps=1e-8, weight_decay=0,
                                     fused=variant == 'A', foreach=variant == 'B')
        times, warmup_times = [], []
        for step in range(plan['warmup_steps'] + plan['measurement_steps']):
            torch.npu.synchronize()
            started = time.perf_counter()
            optimizer.step()
            torch.npu.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            if step < plan['warmup_steps']:
                warmup_times.append(elapsed)
            else:
                times.append(elapsed)
        endpoint = torch.stack([p.detach() for p in params]).cpu()
        if not torch.isfinite(endpoint).all():
            raise RuntimeError('nonfinite updates')
        if variant not in endpoints:
            endpoints[variant] = endpoint
        difference = float((endpoint-endpoints['A']).abs().max())
        if difference > 1e-6:
            raise RuntimeError('numerical screening failed: '+str(difference))
        window_times = times[10:] if protocol == 'legacy-v1' else times
        row = {'variant': variant, 'boot': boot, 'median_ms': statistics.median(window_times),
               'protocol': protocol, 'warmup_times_ms': warmup_times,
               'times_ms': times, 'max_abs_delta_vs_first_A': difference}
        rows.append(row)
        (out/'runs.json').write_text(json.dumps(rows, indent=2))
        print('SCREEN_RUN', index, variant, row['median_ms'], difference, flush=True)
        del optimizer, params, endpoint
    summary = summarize(rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('OPTIMIZER_SCREEN_COMPLETE', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selftest', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--protocol', choices=['legacy-v1', 'warmed-v2'], default='legacy-v1')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.out is None:
        parser.error('--execute requires --out')
    else:
        execute(args.out, args.protocol)


if __name__ == '__main__':
    main()
