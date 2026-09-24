# run_all.py  ——  串行编排 + 上游 gate; 只编排/校验, 不伪造任何环结果
import subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
SNAP = ROOT / "snapshots"; SNAP.mkdir(exist_ok=True)
STEPS = [
    ("poc1_parse.py",       ["poc1_nodes.json", "poc1_diff_report.md"], "环1 AutoMigrate"),
    ("poc2_npu_forward.py", ["poc2_ops.json"],                          "环2 PerfTune 侧(双轨)"),
    ("poc3_scene.py",       ["poc3_contract.json", "poc3_deps.json"],   "环3 DataForge 场景侧"),
]
def run(s):
    print(f"\n===== RUN {s} =====", flush=True)
    return subprocess.call([sys.executable, str(ROOT / s)])
def check(ps):
    miss = [p for p in ps if not (SNAP / p).exists()]
    print(("[GATE OK] " if not miss else f"[GATE FAIL] 缺产物 {miss} -> 下游不启动"), flush=True)
    return not miss
def selfcheck():
    print("===== 自检 =====", flush=True)
    try:
        r = subprocess.run([sys.executable,"-c",
            "import torch,torch_npu;print('NPU_OK' if torch.npu.is_available() else 'NPU_NO')"],
            capture_output=True, text=True)
        print("backend_probe:", (r.stdout or r.stderr).strip(), flush=True)
    except Exception as e:
        print("backend_probe: NPU_NO (probe err:", e, ") -> 环2 走轨B", flush=True)
    print("ref_probe:", "REF_OK" if (ROOT/"refs"/"modeling_qwen3_5__transformers_v5.2.0.py").exists() else "REF_MISSING(环1 将回退本地/自检)", flush=True)
def main():
    selfcheck()
    for s, ps, desc in STEPS:
        rc = run(s)
        if rc != 0: print(f"[FAIL] {desc} 退出码 {rc} -> 终止", flush=True); sys.exit(rc)
        if not check(ps): sys.exit(2)
    print("\n[ALL DONE] 三环串行通过(证据边界见 README §1/§3)", flush=True)
if __name__ == "__main__":
    main()