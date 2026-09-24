#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4-2 端到端 A/B —— 步骤 3/3：**还原框架并逐字节核对**。

纪律：改过别人的代码就必须给出"已经还原"的证据，而不是一句"我改回来了"。
本脚本用 md5 与备份对拍，并对每个文件打印 原md5 / 备份md5 / 还原后md5。
"""
import hashlib
import os
import shutil
import sys

BK = "/root/ops/fwbackup_ab62"
ORIG = {
    "clip_grad_norm.py": ("/root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py",
                          "207f9cbad3da6adba9cd19a00b016b88"),
    "train_engine.py": ("/root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py",
                        "e0e8879ac9c94b9ece48bf18584d3166"),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ok = True
    for name, (path, want) in ORIG.items():
        bkp = os.path.join(BK, name + ".orig")
        if not os.path.isfile(bkp):
            print("RESTORE_FAIL 缺备份 %s" % bkp)
            ok = False
            continue
        before = md5(path)
        shutil.copy2(bkp, path)
        after = md5(path)
        rm = os.system("%s -m py_compile %s" % (sys.executable, path))
        good = (after == want)
        ok = ok and good and (rm == 0)
        print("%-20s 备份md5=%s  还原后md5=%s  期望=%s  py_compile_rc=%d  %s"
              % (name, md5(bkp), after, want, rm, "OK" if good else "**不一致**"))
        print("%-20s （还原前 md5=%s）" % ("", before))
    # 注入残留扫描：marker 语句不得再出现在文件里
    for name, (path, _w) in ORIG.items():
        txt = open(path, encoding="utf-8").read()
        left = txt.count("§4-2 INJECT")
        if left:
            print("RESTORE_FAIL %s 仍残留 %d 处注入标记" % (name, left))
            ok = False
        else:
            print("%-20s 注入标记残留 = 0" % name)
    print("AB62_RESTORE_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
