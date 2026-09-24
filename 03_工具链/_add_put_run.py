# -*- coding: utf-8 -*-
r"""_add_put_run.py —— 给重建的 pull_evidence.py 补回 `--put` / `--run`

坑 180 后我重建的是**只读版**（只有 --status/--out/--files），于是要推脚本/发命令时发现能力缺失。
这里补齐（与丢失的原版行为一致）：
  · `--run <cmd>`：经会话执行命令（命令走 base64，引号/管道/`$()` 都安全 —— 坑 171）
  · `--put <local> <remote>`：**分块** base64 上传 + 两端 SHA256 对照（整份推送会撞 Windows
    命令行 32KB 上限并静默失败 —— 坑 177）
"""
import io

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\pull_evidence.py"
s = io.open(P, encoding="utf-8").read()

if "--put" in s and "def put(" in s:
    print("ADD_PUT_RUN_SKIP 已具备 --put/--run")
    raise SystemExit(0)

FUNCS = '''
def put_file(local, remote, chunk=14000):
    """分块上传 + 两端 SHA256 对照（不一致即报错，不落盘）。"""
    import hashlib
    data = open(local, "rb").read()
    want = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    tmp = remote + ".b64tmp"
    r = run("rm -f %s; touch %s" % (tmp, tmp))
    if r.get("rc") != 0:
        return False, "创建远端临时文件失败：%s" % (r.get("stderr") or "")[:120]
    for i in range(0, len(b64), chunk):
        part = b64[i:i + chunk]
        r = run("printf %%s %s >> %s" % (part, tmp))
        if r.get("rc") != 0:
            return False, "第 %d 块失败：%s" % (i // chunk + 1, (r.get("stderr") or "")[:120])
    r = run("base64 -d %s > %s; rm -f %s; sha256sum %s" % (tmp, remote, tmp, remote))
    got = ((r.get("stdout") or "").strip().split() or [""])[0]
    if got != want:
        return False, "SHA256 不一致：本地=%s 远端=%s" % (want[:12], got[:12])
    return True, "%d B / %d 块 sha256=%s" % (len(data), (len(b64) + chunk - 1) // chunk, got[:12])


'''
assert "def main():" in s
s = s.replace("def main():", FUNCS.lstrip("\n") + "def main():", 1)

s = s.replace('    ap.add_argument("--files", nargs="*", default=None)',
              '    ap.add_argument("--files", nargs="*", default=None)\n'
              '    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))\n'
              '    ap.add_argument("--run")', 1)

anchor = "    if a.status:"
assert anchor in s
s = s.replace(anchor,
              '    if a.put:\n'
              '        ok, why = put_file(a.put[0], a.put[1])\n'
              '        print("PUT_%s %s → %s  %s" % ("OK" if ok else "FAIL", a.put[0], a.put[1], why))\n'
              '        return 0 if ok else 1\n'
              '    if a.run:\n'
              '        r = run(a.run, timeout=600)\n'
              '        sys.stdout.write(r.get("stdout") or "")\n'
              '        if r.get("stderr"):\n'
              '            sys.stderr.write(r["stderr"])\n'
              '        print("REMOTE_RC=%s" % r.get("rc"))\n'
              '        return 0 if r.get("rc") == 0 else 1\n'
              + anchor, 1)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("ADD_PUT_RUN_OK 已补 --put / --run")
