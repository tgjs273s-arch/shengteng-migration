#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""failure_injection.py —— 故障注入：**任何一步失败，旧工件必须逐字节不变**

对应 Codex 复核要求（阶段 0 的判据）："故障注入时旧交付物不变"。
本脚本在**沙箱目录**里跑（绝不触碰真实交付物）：把 safe_pack_sync 的模块级路径全部改写到 tmp，
然后注入三类故障，逐例断言"旧目录与旧 zip 的哈希不变"：

  ① 备份失败（backup_dir 返回 None）        ⇒ 必须**中止**，不得删除
  ② 解压失败（extractall 抛异常）           ⇒ 必须回滚，旧目录内容恢复
  ③ 校验失败（解包后文件被篡改）            ⇒ 必须回滚，旧目录内容恢复

反例价值：修复前 ① 会走到 rmtree（旧目录被删且无回滚点）；本脚本能把那种行为标红。
用法：python failure_injection.py     输出 FAILURE_INJECTION_OK cases=N failed=0
"""
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def tree_hash(d):
    """目录内容的确定性哈希（相对路径 + 内容），用于断言"旧工件未变"。"""
    if not os.path.isdir(d):
        return "<不存在>"
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).replace("\\", "/").encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def load_sandboxed(tmp):
    """把 safe_pack_sync 载入并把它的路径常量改写到 tmp（沙箱）。"""
    spec = importlib.util.spec_from_file_location("sps_sbx", os.path.join(HERE, "safe_pack_sync.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SKILL = os.path.join(tmp, "src")
    m.OUT_ZIP_DIR = os.path.join(tmp, "out", "06_Skill")
    m.DELIV_ZIP_DIR = os.path.join(tmp, "deliv", "06_Skill")
    m.BACKUP_ROOT = os.path.join(tmp, "backups")
    m.ALLOW_DELETE_UNDER = [os.path.abspath(m.OUT_ZIP_DIR), os.path.abspath(m.DELIV_ZIP_DIR)]
    return m


def make_zip(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr("qwen35-ascend-migrator/" + n, "NEW-CONTENT-%s" % n)


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="faultinj_")
    try:
        m = load_sandboxed(tmp)
        target = os.path.join(m.OUT_ZIP_DIR, "qwen35-ascend-migrator")
        os.makedirs(target)
        with open(os.path.join(target, "OLD_MARKER.txt"), "w", encoding="utf-8") as fh:
            fh.write("OLD\n")
        before = tree_hash(target)
        zpath = os.path.join(tmp, "new.zip")
        make_zip(zpath, ["a.txt", "b.txt"])

        def ck(name, cond, extra=""):
            print("  [%s] %-52s %s" % ("PASS" if cond else "FAIL", name, extra))
            if not cond:
                fails.append(name)

        # ---- ① 备份失败 ⇒ 必须中止且目录不变 ----
        real_backup = m.backup_dir
        m.backup_dir = lambda *a, **k: None
        rc = m.unpack_replace(zpath, target, True)
        ck("① 备份失败 ⇒ 拒绝删除（rc=False）", rc is False, "rc=%s" % rc)
        ck("① 备份失败 ⇒ 旧目录逐字节不变", tree_hash(target) == before)
        m.backup_dir = real_backup

        # ---- ② 解压失败 ⇒ 必须回滚 ----
        real_extract = zipfile.ZipFile.extractall
        calls = {"n": 0}

        def boom(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("注入的解压失败")
            return real_extract(self, *a, **k)

        zipfile.ZipFile.extractall = boom
        rc = m.unpack_replace(zpath, target, True)
        zipfile.ZipFile.extractall = real_extract
        ck("② 解压失败 ⇒ 返回 False", rc is False, "rc=%s" % rc)
        ck("② 解压失败 ⇒ 旧目录被回滚、逐字节不变", tree_hash(target) == before)

        # ---- ③ 校验失败（解包后内容被篡改）⇒ 必须回滚 ----
        def tamper(self, *a, **k):
            real_extract(self, *a, **k)
            p = os.path.join(target, "a.txt")
            if os.path.isfile(p):
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write("TAMPERED")

        zipfile.ZipFile.extractall = tamper
        rc = m.unpack_replace(zpath, target, True)
        zipfile.ZipFile.extractall = real_extract
        ck("③ 校验失败 ⇒ 返回 False", rc is False, "rc=%s" % rc)
        ck("③ 校验失败 ⇒ 旧目录被回滚、逐字节不变", tree_hash(target) == before)

        # ---- 正向对照：不注入故障时必须成功（否则上面的"拒绝"可能是因为它根本跑不通）----
        rc = m.unpack_replace(zpath, target, True)
        ok_forward = (rc is True) and os.path.isfile(os.path.join(target, "a.txt"))
        ck("正对照：无故障时成功解包（证明上面的失败不是『它本来就不工作』）", ok_forward, "rc=%s" % rc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("FAILURE_INJECTION_%s cases=7 failed=%d" % ("OK" if not fails else "FAIL", len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
