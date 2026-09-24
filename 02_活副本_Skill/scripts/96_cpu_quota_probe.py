#!/usr/bin/env python3
"""Read-only CPU quota snapshot; supports cgroup v1 and v2. No training."""
import argparse
import json
import os
from pathlib import Path


def snapshot():
    result = {"affinity": sorted(os.sched_getaffinity(0)),
              "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "cgroups": {}}
    for root in ("/sys/fs/cgroup", "/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        for name in ("cpu.max", "cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us"):
            p = Path(root) / name
            if p.is_file():
                result["cgroups"][str(p)] = p.read_text().strip()
    import torch
    result["torch_num_threads"] = torch.get_num_threads()
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(snapshot(), indent=2))
