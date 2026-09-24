#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4-2 端到端 A/B —— 步骤 1/3：备份 + **env 门控注入** + 自证。

设计要点（对应项目纪律）
------------------------
1. **env 门控**：只有 `MM_CAND` 等于目标值时分支才进入 ⇒ 基线臂（不设该变量）走的是
   **与上游完全相同的代码路径**，不需要"两套框架文件"来回换。
2. **锚点唯一性断言**：插入前要求锚点在文件中**恰好出现 1 次**，否则拒绝写入（fail-closed）。
3. **幂等**：已注入则跳过，不重复插。
4. **先备份**：落到 `/root/ops/fwbackup_ab62/`，并记录原始 md5 供还原时逐字节核对。
5. **注入自证**（纪律 #3：复合开关要拆到底，否则"无效"可能是注入失败）：
   被注入的分支首次执行时写一个 marker 文件 ⇒ 事后可验证"代码路径真的被走到了"，
   而不是只看环境变量设没设。
"""
import hashlib
import os
import shutil
import sys

BK = "/root/ops/fwbackup_ab62"

C1_MARK = "    # === §4-2 INJECT C1 (env-gated: MM_CAND=C1) ==="
C1_BLOCK = """    # === §4-2 INJECT C1 (env-gated: MM_CAND=C1) ===
    import os as _mm_os
    if max_norm <= 0. and _mm_os.environ.get("MM_CAND") == "C1":
        _mm_mk = "/tmp/mm_cand_c1_active"
        if not _mm_os.path.exists(_mm_mk):
            try:
                with open(_mm_mk, "w") as _mm_fh:
                    _mm_fh.write("1")
            except OSError:
                pass
        return torch.tensor(0.0, device=torch.device(get_device_type()))
    # === /§4-2 INJECT C1 ===
"""

C2_MARK = "        # === §4-2 INJECT C2 (env-gated: MM_CAND=C2) ==="
C2_BLOCK = """        # === §4-2 INJECT C2 (env-gated: MM_CAND=C2) ===
        import os as _mm_os
        if _mm_os.environ.get("MM_CAND") == "C2":
            _mm_mk = "/tmp/mm_cand_c2_active"
            if not _mm_os.path.exists(_mm_mk):
                try:
                    with open(_mm_mk, "w") as _mm_fh:
                        _mm_fh.write("1")
                except OSError:
                    pass
            return torch.cat([loss.clone().detach().view(1) for loss in losses])
        # === /§4-2 INJECT C2 ===
"""

TARGETS = [
    {
        "path": "/root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py",
        "expect_md5": "207f9cbad3da6adba9cd19a00b016b88",
        "anchor": "    # EP-aware path (FSDP2 + EP)\n",
        "block": C1_BLOCK,
        "mark": C1_MARK,
        "name": "clip_grad_norm.py",
    },
    {
        "path": "/root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py",
        "expect_md5": "e0e8879ac9c94b9ece48bf18584d3166",
        "anchor": '        """Reduce a tensor of losses across all GPUs."""\n',
        "block": C2_BLOCK,
        "mark": C2_MARK,
        "name": "train_engine.py",
    },
]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(BK, exist_ok=True)
    ok = True
    for t in TARGETS:
        p = t["path"]
        if not os.path.isfile(p):
            print("SETUP_FAIL 找不到 %s" % p)
            ok = False
            continue
        cur = md5(p)
        print("== %s 当前 md5=%s" % (t["name"], cur))
        if t["mark"] in open(p, encoding="utf-8").read():
            print("   SKIP 已注入（幂等）")
            continue
        if cur != t["expect_md5"]:
            print("   SETUP_FAIL md5 与预期不符（期望 %s）⇒ 拒绝在未知版本上注入" % t["expect_md5"])
            ok = False
            continue
        bkp = os.path.join(BK, t["name"] + ".orig")
        if not os.path.exists(bkp):
            shutil.copy2(p, bkp)
            print("   备份 -> %s" % bkp)
        else:
            print("   备份已存在 -> %s" % bkp)
        text = open(p, encoding="utf-8").read()
        n = text.count(t["anchor"])
        if n != 1:
            print("   SETUP_FAIL 锚点出现 %d 次（必须恰好 1 次）⇒ 拒绝写入" % n)
            ok = False
            continue
        text = text.replace(t["anchor"], t["block"] + t["anchor"], 1)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        new = md5(p)
        print("   注入完成 md5 %s -> %s" % (cur, new))
        # 语法闸门：注入后必须仍能编译
        rc = os.system("%s -m py_compile %s" % (sys.executable, p))
        if rc != 0:
            print("   SETUP_FAIL 注入后 py_compile 失败 ⇒ 立刻还原")
            shutil.copy2(bkp, p)
            ok = False
            continue
        print("   注入后 py_compile OK")
    print("AB62_SETUP_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
