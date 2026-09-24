# -*- coding: utf-8 -*-
r"""_patch_pack3.py —— P0 修复：备份失败时**不得继续删除**（Codex 复核 P0-a）

缺陷（我引入的）：`backup_dir` 在失败时返回 None 并只打印一行；`unpack_replace` 拿到 None 后
**照样 `shutil.rmtree(target_dir)`** —— 于是"没有回滚点却执行了不可逆删除"。若随后的解压或校验再失败，
旧目录已经没了（源码注释甚至写着"删除流程继续"）。

处置：**没有回滚点就不许删**。目标目录存在而备份失败 ⇒ 立即中止并返回 False（调用方据此判失败）。
"""
import io

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\safe_pack_sync.py"
s = io.open(P, encoding="utf-8").read()

old = '''    bk = backup_dir(target_dir, os.path.basename(parent))
    log("    备份点: %s" % bk)
    try:
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)'''
new = '''    bk = backup_dir(target_dir, os.path.basename(parent))
    log("    备份点: %s" % bk)
    # ★★ P0 修复（Codex 复核指出，实测确认）：**没有回滚点就不许删**。
    #   原实现在 bk is None 时只打印一行、随后照样 rmtree ⇒ "不可逆删除 + 无回滚点"，
    #   一旦后续解压/校验再失败，旧目录就永久丢失（正是坑 180 的形态）。
    if bk is None and os.path.isdir(target_dir):
        log("  **中止**：备份失败而目标已存在 ⇒ 拒绝删除（无回滚点不做不可逆操作）")
        return False
    try:
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)'''
assert old in s, "锚点未找到"
s = s.replace(old, new, 1)

# 顺带把"删除流程继续"这句自我开脱的注释改掉（它正是这次缺陷的痕迹）
s = s.replace('print("    备份失败：%s: %s ⇒ 返回 None（删除流程继续，但请知悉没有回滚点）"\n              % (type(exc).__name__, exc))',
              'print("    备份失败：%s: %s ⇒ 返回 None（**调用方将因此中止，不再删除**）"\n              % (type(exc).__name__, exc))', 1)
s = s.replace('print("    备份失败：%s 下同名目录过多 ⇒ 返回 None（调用方会跳过回滚分支）" % BACKUP_ROOT)',
              'print("    备份失败：%s 下同名目录过多 ⇒ 返回 None（**调用方将因此中止**）" % BACKUP_ROOT)', 1)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("PATCH_PACK3_OK 无回滚点即中止已生效")
