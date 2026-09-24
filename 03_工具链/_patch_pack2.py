# -*- coding: utf-8 -*-
r"""_patch_pack2.py —— 修 safe_pack_sync.py 的"备份目录碰撞"缺陷（首跑实测暴露）

现象：`--apply` 首次运行在 backup_dir 抛 FileExistsError traceback（同一秒内二次运行/残留即触发）；
根因：备份目录名用 %H%M%S 且不处理已存在，且异常直接冒泡。
处置：① 目录名加唯一序号后缀；② 备份失败**打印原因并返回 None**（调用方已能处理 None），绝不 traceback。
"""
import io

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\safe_pack_sync.py"
s = io.open(P, encoding="utf-8").read()

old = '''def backup_dir(src_dir, tag):
    """G-d：把将被替换的目录整体备份到 _旧版本备份_<stamp>/<tag>/。"""
    if not os.path.isdir(src_dir):
        return None
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    dst = os.path.join(BACKUP_ROOT, "%s_%s" % (tag, time.strftime("%H%M%S")))
    shutil.copytree(src_dir, dst)
    return dst'''

new = '''def backup_dir(src_dir, tag):
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
        print("    备份失败：%s 下同名目录过多 ⇒ 返回 None（调用方会跳过回滚分支）" % BACKUP_ROOT)
        return None
    try:
        shutil.copytree(src_dir, dst)
    except Exception as exc:
        print("    备份失败：%s: %s ⇒ 返回 None（删除流程继续，但请知悉没有回滚点）"
              % (type(exc).__name__, exc))
        return None
    return dst'''

assert old in s, "锚点未找到（文件已被改动？）"
io.open(P, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
print("PATCH_OK backup_dir 已改为唯一后缀 + 不抛异常")
