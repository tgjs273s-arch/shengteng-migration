# -*- coding: utf-8 -*-
r"""_patch_safe_pack.py —— 给 safe_pack_sync.py 打三个补丁（每个都对应一次实测暴露的缺陷）

1) dry-run 不许读尚未生成的 zip：改为**按待打包文件清单**推算"将删除项"
2) 解包后内部校验：zip 条目已含顶层目录名，目标应是 `parent/n`（否则双重前缀 → 全量误判 → 假回滚）
3) 最终三方一致性：**比逐文件哈希**，不是只比文件名（后者正是当年"落后了却报 OK"的毛病）
"""
import io
import os
import sys

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\safe_pack_sync.py"
s = io.open(P, encoding="utf-8").read()
orig = s

# ---- 补丁 2：内部校验路径 ----
old2 = """                dp = os.path.join(target_dir, n)"""
new2 = """                # zip 条目本身已含顶层目录名（UNPACK_NAME/…），而解包目标是 parent ⇒ 拼 parent
                dp = os.path.join(parent, n)"""
assert old2 in s, "补丁2 锚点未找到"
s = s.replace(old2, new2, 1)

# ---- 补丁 1：dry-run 按清单推算，不读 zip ----
old1 = """            names, add, same, stale = unpack_manifest(zip_out, tgt)
            log("  目标 %s：zip 内 %d 文件；已存在 %d；**将删除 %d**"
                % (tgt, len(names), len(same), len(stale)))"""
new1 = """            # ★ 不许读尚未生成的 zip：按**待打包清单**推算（第一版在这里 FileNotFoundError）
            planned = {UNPACK_NAME + "/" + f.replace(os.sep, "/") for f in files}
            have = {os.path.relpath(os.path.join(r, f2), os.path.dirname(tgt)).replace("\\\\", "/")
                    for r, _dd, fs in os.walk(tgt) for f2 in fs}
            stale = sorted(have - planned)
            log("  目标 %s：计划内 %d 文件；现存在 %d；**将删除 %d**"
                % (tgt, len(planned), len(have), len(stale)))"""
assert old1 in s, "补丁1 锚点未找到"
s = s.replace(old1, new1, 1)

# ---- 补丁 3：最终一致性改为逐文件哈希 ----
old3 = """    with zipfile.ZipFile(zip_out) as z:
        names = {n for n in z.namelist() if not n.endswith("/")}
    for d in (OUT_ZIP_DIR, DELIV_ZIP_DIR):
        base = os.path.join(d, UNPACK_NAME)
        got = {os.path.relpath(os.path.join(r, f), d).replace("\\\\", "/")
               for r, _dd, fs in os.walk(base) for f in fs}
        log("一致性 %s 条目=%d 与 zip 相同=%s" % (d, len(got), got == names))
        ok = ok and got == names"""
new3 = """    # ★ 必须比**内容**：只比文件名的话，"解包副本落后于 zip"照样报 OK —— 当年 G22 就是这么漏的。
    with zipfile.ZipFile(zip_out) as z:
        want = {n: hashlib.sha256(z.read(n)).hexdigest()
                for n in z.namelist() if not n.endswith("/")}
    for d in (OUT_ZIP_DIR, DELIV_ZIP_DIR):
        mism = [n for n, h in want.items()
                if not os.path.isfile(os.path.join(d, n))
                or hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() != h]
        log("一致性 %s 条目=%d **逐文件哈希不一致=%d**%s"
            % (d, len(want), len(mism), ("  例：%s" % mism[:3]) if mism else ""))
        ok = ok and not mism"""
assert old3 in s, "补丁3 锚点未找到"
s = s.replace(old3, new3, 1)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("PATCH_OK 三处均已替换（%d → %d 字符）" % (len(orig), len(s)))
