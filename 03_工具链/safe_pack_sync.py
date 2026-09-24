#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""safe_pack_sync.py —— 安全的「打包 Skill + 同步两处解包副本」（坑 180 的重建版）

为什么要有它
-----------
原 `build_handover_skill_zip.py` + `sync_to_delivery_folder.py` 已在坑 180 里被删除，
而"再改 Skill 就没法安全打包"是交付前的硬缺口。重建时**不凭记忆照抄**，而是按坑 180 的
处置把四道闸门**写进设计**：

  G-a **默认 dry-run**：不带 `--apply` 时只打印"将写入/将删除"清单，不碰任何文件。
  G-b **删除白名单 + 前缀断言**：任何删除对象的规范化绝对路径**必须**位于允许的两个
      `06_Skill` 父目录之下；越界立即 abort（不是警告）。
  G-c **失败必回滚**：先备份、再删除、任何异常都用备份还原。
  G-d **运行前打备份点**：被替换的解包副本先整目录复制到 `_旧版本备份_<stamp>/`。
  另加 **G-e 内容闸门（fail-closed）**：敏感串（token/连接密码）、>5MB 文件、`.sh` 带 BOM
      任一命中即拒绝出包 —— 包是要交给评委的，宁可不生成也不带着问题生成。

它做什么（三步，幂等）
--------------------
  1. 打包：活副本 → `05_交付物成品/06_Skill/qwen35-ascend-migrator_<日期>_v1.0.zip`
     （排除运行态产物：`out/**`（保留 `out/README.md`）、`**/__pycache__/**`、`tests/tmp/**`）
  2. 同步：把该 zip 复制到交付目录 `C4AI复赛_交付材料/06_Skill/`
  3. 解包：在两处 `06_Skill/` 下各生成 `qwen35-ascend-migrator/` 解包副本，并**逐文件哈希**核对
     解包副本 == zip 内容（旧脚本当年就是这里落后于活副本，见坑 22 的 G22）

用法
----
    python safe_pack_sync.py                 # dry-run：打印清单
    python safe_pack_sync.py --apply         # 执行（自动备份 + 失败回滚）
    python safe_pack_sync.py --apply --keep-zip-name   # 不按日期重命名（沿用既有文件名）
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import zipfile

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

PKG = PATHS.root
SKILL = PATHS.skill
DELIV = PATHS.deliverables
OUT_ZIP_DIR = os.path.join(DELIV, "06_Skill")
DELIV_ZIP_DIR = os.path.join(DELIV, "06_Skill")
UNPACK_NAME = "qwen35-ascend-migrator"
BACKUP_ROOT = os.path.join(DELIV, "_旧版本备份_%s" % time.strftime("%Y%m%d"))
ZIP_BASENAME = "qwen35-ascend-migrator_%s_v1.0.zip" % time.strftime("%Y%m%d")

# 允许删除的父目录白名单（G-b：删任何东西都必须在这两个之下）
ALLOW_DELETE_UNDER = [os.path.abspath(OUT_ZIP_DIR), os.path.abspath(DELIV_ZIP_DIR)]
EXCLUDE_DIRS = {"__pycache__", ".git"}
MAX_FILE_MB = 5.0
# 敏感串：本项目真实出现过的凭据形态（宁可误报，不可漏报）
SECRET_PATTERNS = ["MD8NtRW1HDHNsO8K", "连接密码:", "jt_"]
# ★ 2026-09-22：把"真 token"的判定**提成模块级单一来源**（`TOKEN_RE`），供本闸门与
#   `_redact_tokens.py` **共用**。起因：我写脱敏脚本时自己另写了一个更严的正则
#   （要求两侧都足够长），于是"脱敏跑了 0 处、闸门却仍然报命中" ⇒ 判据一分为二就会分叉。
#   **要脱敏的东西，必须与判定它的东西用同一个定义。**
TOKEN_RE = re.compile(r"jt_[0-9A-Fa-f]{4,}:[0-9A-Fa-f]{8,}")
BOM = b"\xef\xbb\xbf"


def log(msg):
    print(msg, flush=True)


def sha12(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def walk_skill():
    """列出要打进包的文件（相对路径），执行排除规则。"""
    out = []
    for root, dirs, files in os.walk(SKILL):
        rel_root = os.path.relpath(root, SKILL)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        if rel_root.split(os.sep)[0] == "out":
            # out/ 只保留 README.md（与旧打包行为一致：排除运行态产物）
            pass
        for f in files:
            rel = os.path.normpath(os.path.join(rel_root, f))
            parts = rel.split(os.sep)
            if parts[0] == "out" and f != "README.md":
                continue
            if "tests" in parts and "tmp" in parts:
                continue
            out.append(rel)
    return sorted(out)


def content_gate(files):
    """G-e：不碰磁盘，只读取检查；任一问题即返回失败清单。"""
    problems, total = [], 0
    for rel in files:
        p = os.path.join(SKILL, rel)
        try:
            sz = os.path.getsize(p)
        except OSError as exc:
            problems.append("%s 不可读：%s" % (rel, exc))
            continue
        total += sz
        if sz > MAX_FILE_MB * 1024 * 1024:
            problems.append("%s 超过 %.1f MB（%.1f MB）" % (rel, MAX_FILE_MB, sz / 1048576.0))
        if sz == 0:
            problems.append("%s 是空文件（0 B）—— 不该入包" % rel)
        # ★ 2026-09-22 新增（坑 201）：**备份/临时文件禁止入包**。
        #   事故：我的补丁/追加/脱敏工具把备份写在**被改文件旁边**（`.bak_<时间戳>`），而这些文件在
        #   Skill 树内 ⇒ **被打进了交付 zip**；其中两份是"脱敏前"的坑表备份 ⇒ **真 token 进了交付物**。
        #   原闸门为什么没拦住：敏感串扫描**按扩展名过滤**，而 `.bak_20260922_105953` 没有扩展名 ⇒ 被跳过。
        #   两道修：① 名字像备份/临时的一律拒绝入包；② 敏感串扫描**对全部文件生效**（下一段，不再过滤）。
        if re.search(r"\.bak|\.orig|\.tmp$|~$|\.swp$", rel):
            problems.append("%s 像备份/临时文件 —— 禁止入包（备份必须写在 Skill 树之外）" % rel)
        if rel.endswith(".sh"):
            with open(p, "rb") as fh:
                if fh.read(3) == BOM:
                    problems.append("%s 是 .sh 但带 BOM（远端 bash 报 command not found，且首行静默失效）" % rel)
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except Exception:
            txt = ""                       # 二进制/不可解码：**不因此跳过敏感串扫描**（下面按空串处理）
        for pat in SECRET_PATTERNS:
            if pat in txt:
                # "jt_" 出现在坑表里是**记录**（说明我们检测过），不算泄露；但真 token 形如
                # jt_<hex>:<hex> 必须报出来 —— 用更严的形态判定，避免把自检文案误判成泄露。
                if pat == "jt_":
                    if TOKEN_RE.search(txt):
                        problems.append("%s 含疑似真 token（jt_<ID>:<HEX>）" % rel)
                else:
                    problems.append("%s 含敏感串 %r" % (rel, pat))
    return problems, total


def unpack_manifest(zip_path, target_dir):
    """算出"解包后相对目标目录"的清单；顺带给出将删除的陈旧文件（白名单断言在 apply 里做）。"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    add, same = [], []
    for n in sorted(names):
        dst = os.path.join(target_dir, n)
        if os.path.isfile(dst):
            with zipfile.ZipFile(zip_path) as z:
                data = z.read(n)
            if hashlib.sha256(data).hexdigest() == sha12(dst) + "" or True:
                pass
        (add if not os.path.isfile(dst) else same).append(n)
    stale = []
    if os.path.isdir(target_dir):
        for root, _d, fs in os.walk(target_dir):
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), target_dir).replace("\\", "/")
                if rel not in names:
                    stale.append(rel)
    return names, add, same, stale


def assert_deletable(paths):
    """G-b：删除白名单 + 前缀断言（规范化绝对路径必须在白名单父目录之下）。"""
    bad = []
    for p in paths:
        ap = os.path.abspath(p)
        if not any(ap == d or ap.startswith(d + os.sep) for d in ALLOW_DELETE_UNDER):
            bad.append(ap)
    return bad


def backup_dir(src_dir, tag):
    """G-d：把将被替换的目录整体备份到 _旧版本备份_<stamp>/<tag>_<唯一后缀>/。

    ★ 实测缺陷（本脚本首次 --apply 就在此处 traceback）：原实现用
      "%s_%s" % (tag, time.strftime("%H%M%S")) 作目录名 —— **秒级粒度且不处理"已存在"** ⇒
      同一秒内二次运行（或有残留）时 copytree 抛 FileExistsError 直接崩。
      一个会被反复调用的工具不能这样：① 名字必须有唯一后缀；② 备份失败要**打印原因并返回 None**
      （调用方已有 `if bk and os.path.isdir(bk)` 分支），**绝不 traceback**。
    """
    if not os.path.isdir(src_dir):
        return None
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    dst = None
    for i in range(100):
        cand = os.path.join(BACKUP_ROOT, "%s_%s_%02d" % (tag, time.strftime("%H%M%S"), i))
        if not os.path.exists(cand):
            dst = cand
            break
    if dst is None:
        print("    备份失败：%s 下同名目录过多 ⇒ 返回 None（**调用方将因此中止**）" % BACKUP_ROOT)
        return None
    try:
        shutil.copytree(src_dir, dst)
    except Exception as exc:
        print("    备份失败：%s: %s ⇒ 返回 None（**调用方将因此中止，不再删除**）"
              % (type(exc).__name__, exc))
        return None
    return dst


def unpack_replace(zip_path, target_dir, apply_it):
    """解包到 target_dir（替换式）。apply=False 时只打印清单。"""
    parent = os.path.dirname(os.path.abspath(target_dir))
    if not any(parent == d for d in ALLOW_DELETE_UNDER):
        log("ABORT %s 不在白名单父目录内 ⇒ 拒绝操作（G-b）" % target_dir)
        return False
    names, add, same, stale = unpack_manifest(zip_path, target_dir)
    log("  目标 %s" % target_dir)
    log("    zip 内文件 %d；已存在 %d；**将删除的陈旧文件 %d**" % (len(names), len(same), len(stale)))
    for s in stale[:10]:
        log("      - %s" % s)
    if len(stale) > 10:
        log("      …（共 %d）" % len(stale))
    if not apply_it:
        return True
    bad = assert_deletable([os.path.join(target_dir, s) for s in stale] + [target_dir])
    if bad:
        log("ABORT 删除清单越界 ⇒ 拒绝（G-b）：%s" % bad[:3])
        return False
    bk = backup_dir(target_dir, os.path.basename(parent))
    log("    备份点: %s" % bk)
    # ★★ P0 修复（Codex 复核指出，实测确认）：**没有回滚点就不许删**。
    #   原实现在 bk is None 时只打印一行、随后照样 rmtree ⇒ "不可逆删除 + 无回滚点"，
    #   一旦后续解压/校验再失败，旧目录就永久丢失（正是坑 180 的形态）。
    if bk is None and os.path.isdir(target_dir):
        log("  **中止**：备份失败而目标已存在 ⇒ 拒绝删除（无回滚点不做不可逆操作）")
        return False
    try:
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(parent)
        # 逐文件哈希核对：解包副本必须与 zip 内容一致
        with zipfile.ZipFile(zip_path) as z:
            mism = []
            for n in names:
                data = z.read(n)
                # zip 条目本身已含顶层目录名（UNPACK_NAME/…），而解包目标是 parent ⇒ 拼 parent
                dp = os.path.join(parent, n)
                if not os.path.isfile(dp) or hashlib.sha256(open(dp, "rb").read()).hexdigest() != \
                        hashlib.sha256(data).hexdigest():
                    mism.append(n)
        if mism:
            raise RuntimeError("解包后 %d 个文件与 zip 不一致：%s" % (len(mism), mism[:3]))
    except Exception as exc:
        log("  **失败** %s: %s ⇒ 用备份点回滚（G-c）" % (type(exc).__name__, exc))
        if bk and os.path.isdir(bk):
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(bk, target_dir)
            log("  已回滚自 %s" % bk)
        return False
    log("  OK 解包并逐文件哈希核对通过")
    return True


def verify_only():
    """**只读**核对：zip ↔ 两处解包副本 ↔ 活副本（逐文件哈希）。不写任何文件。

    为什么要单独一个模式：Codex 复核指出"检查命令必须只读" —— 原 G22 直接调 `--apply`，
    等于**闸门自己会写文件**（验证与发布没分离）。本模式只读、可随时跑。
    """
    zips = [f for f in os.listdir(OUT_ZIP_DIR) if f.endswith(".zip")] \
        if os.path.isdir(OUT_ZIP_DIR) else []
    print("zip 数=%d（应为 1）" % len(zips))
    problems = []
    if len(zips) != 1:
        problems.append("06_Skill 下 zip 数=%d ⇒ 多版本会让评委困惑" % len(zips))
        return False, problems
    zp = os.path.join(OUT_ZIP_DIR, zips[0])
    with zipfile.ZipFile(zp) as z:
        want = {n: hashlib.sha256(z.read(n)).hexdigest()
                for n in z.namelist() if not n.endswith("/")}
    # 活副本逐文件哈希（与 zip 内路径对齐）
    live = {}
    for rel in walk_skill():
        p = os.path.join(SKILL, rel)
        live[UNPACK_NAME + "/" + rel.replace(os.sep, "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    for d in dict.fromkeys((OUT_ZIP_DIR, DELIV_ZIP_DIR)):
        label = "交付目录"
        mism = [n for n, h in want.items()
                if not os.path.isfile(os.path.join(d, n))
                or hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() != h]
        print("  %s：条目=%d 与 zip 哈希不一致=%d%s"
              % (label, len(want), len(mism), ("  例：%s" % mism[:3]) if mism else ""))
        if mism:
            problems.append("%s 有 %d 个文件与 zip 不一致" % (label, len(mism)))
    miss_live = sorted(set(want) - set(live))
    diff_live = sorted(k for k in (set(want) & set(live)) if want[k] != live[k])
    print("  活副本：zip 内 %d 个路径中，活副本缺 %d、内容不同 %d"
          % (len(want), len(miss_live), len(diff_live)))
    if miss_live or diff_live:
        problems.append("活副本与 zip 不一致（缺 %d / 不同 %d）" % (len(miss_live), len(diff_live)))
    if problems:
        print("SAFE_PACK_VERIFY_FAIL " + "；".join(problems[:3]))
        return False, problems
    print("SAFE_PACK_VERIFY_OK items=%d zip=%s" % (len(want), zips[0]))
    return True, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-zip-name", action="store_true",
                    help="沿用现存唯一 zip 的名字（现存数 ≠1 则报错退出，不会新建第二个名字）")
    ap.add_argument("--zip-name", default="",
                    help="显式指定 zip 文件名（换名/首次打包用）；给了它就不看 --keep-zip-name")
    ap.add_argument("--verify-only", action="store_true",
                    help="只读核对（zip ↔ 两处解包副本 ↔ 活副本），不写任何文件")
    a = ap.parse_args()
    if a.apply:
        print("SAFE_PACK_BLOCKED 项目禁止批量删除；旧版 --apply 已禁用，仅支持只读核验或预览")
        return 2
    if a.verify_only:
        return 0 if verify_only()[0] else 1
    log("== safe_pack_sync %s ==" % ("**APPLY**" if a.apply else "DRY-RUN（不改任何文件）"))
    log("源（活副本）: %s" % SKILL)
    if not os.path.isdir(SKILL):
        log("FAIL 活副本不存在")
        return 2

    files = walk_skill()
    problems, total = content_gate(files)
    log("待打包文件 %d 个 / %.2f MB" % (len(files), total / 1048576.0))
    if problems:
        log("G-e 内容闸门未通过（%d 项）—— **拒绝出包**：" % len(problems))
        for p in problems[:10]:
            log("   ! %s" % p)
        return 3
    log("G-e 内容闸门通过（无敏感串 / 无 >%.1fMB / .sh 无 BOM）" % MAX_FILE_MB)

    # ★ 坑 189 同族：`--keep-zip-name` 原来是**硬编码 "20260921"**（不是"沿用现有名字"）。
    #   后果：它**制造了第二个 zip**——而"多版本 zip 会让评委困惑"正是 verify_only 自己的闸门要拦的。
    #   现在的语义才是名字说的那件事：**沿用现存的唯一一个 zip 的名字**；不唯一（0 或 ≥2）就报错，
    #   绝不自作主张再起一个新名字（要换名字必须显式 --zip-name）。
    if a.zip_name:
        zip_name = a.zip_name
    elif a.keep_zip_name:
        zips = sorted(f for f in os.listdir(OUT_ZIP_DIR)
                      if f.endswith(".zip")) if os.path.isdir(OUT_ZIP_DIR) else []
        if len(zips) != 1:
            log("FAIL --keep-zip-name 要求现存 zip 恰好 1 个，实际 %d 个：%s" % (len(zips), zips))
            log("     ⇒ 先人工收敛到 1 个（移到 _旧版本备份_*），或用 --zip-name 显式指定")
            return 4
        zip_name = zips[0]
        log("沿用现有 zip 名: %s" % zip_name)
    else:
        zip_name = ZIP_BASENAME
    zip_out = os.path.join(OUT_ZIP_DIR, zip_name)
    zip_deliv = os.path.join(DELIV_ZIP_DIR, zip_name)
    log("将写入: %s" % zip_out)
    log("将写入: %s" % zip_deliv)
    log("将重建解包副本: %s\\%s" % (OUT_ZIP_DIR, UNPACK_NAME))
    log("将重建解包副本: %s\\%s" % (DELIV_ZIP_DIR, UNPACK_NAME))
    if not a.apply:
        # ★ G-a 的全部意义就是"**先看到要删什么**" —— 第一版 dry-run 只打印了"将写入哪些路径"，
        #   把"将删除的陈旧文件清单"留到了 apply 里才算 ⇒ **闸门形同虚设**（还是先动手后知道）。
        #   这里在 dry-run 阶段就把两处解包副本的删除清单算出来打印。
        log("-- 将删除的陈旧文件清单（dry-run 预演；G-b 白名单断言在 apply 时执行）--")
        for d in (OUT_ZIP_DIR, DELIV_ZIP_DIR):
            tgt = os.path.join(d, UNPACK_NAME)
            if not os.path.isdir(tgt):
                log("  目标 %s 不存在（无删除）" % tgt)
                continue
            # ★ 不许读尚未生成的 zip：按**待打包清单**推算（第一版在这里 FileNotFoundError）
            planned = {UNPACK_NAME + "/" + f.replace(os.sep, "/") for f in files}
            have = {os.path.relpath(os.path.join(r, f2), os.path.dirname(tgt)).replace("\\", "/")
                    for r, _dd, fs in os.walk(tgt) for f2 in fs}
            stale = sorted(have - planned)
            log("  目标 %s：计划内 %d 文件；现存在 %d；**将删除 %d**"
                % (tgt, len(planned), len(have), len(stale)))
            for s in stale[:15]:
                log("      - %s" % s)
        log("DRY-RUN 结束（未改动任何文件）；加 --apply 执行")
        return 0

    os.makedirs(OUT_ZIP_DIR, exist_ok=True)
    os.makedirs(DELIV_ZIP_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(os.path.join(SKILL, rel), os.path.join(UNPACK_NAME, rel).replace("\\", "/"))
    log("已打包 %s（%d B，sha256=%s）" % (zip_out, os.path.getsize(zip_out), sha12(zip_out)))
    shutil.copy2(zip_out, zip_deliv)
    log("已复制到交付目录 %s" % zip_deliv)

    ok = True
    for d in (OUT_ZIP_DIR, DELIV_ZIP_DIR):
        ok = unpack_replace(zip_out, os.path.join(d, UNPACK_NAME), True) and ok
    # 三方一致性（zip / 两处解包副本）—— 旧脚本当年就落后在这里（坑 22 的 G22）
    # ★ 必须比**内容**：只比文件名的话，"解包副本落后于 zip"照样报 OK —— 当年 G22 就是这么漏的。
    with zipfile.ZipFile(zip_out) as z:
        want = {n: hashlib.sha256(z.read(n)).hexdigest()
                for n in z.namelist() if not n.endswith("/")}
    for d in (OUT_ZIP_DIR, DELIV_ZIP_DIR):
        mism = [n for n, h in want.items()
                if not os.path.isfile(os.path.join(d, n))
                or hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() != h]
        log("一致性 %s 条目=%d **逐文件哈希不一致=%d**%s"
            % (d, len(want), len(mism), ("  例：%s" % mism[:3]) if mism else ""))
        ok = ok and not mism
    log("SAFE_PACK_SYNC_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
