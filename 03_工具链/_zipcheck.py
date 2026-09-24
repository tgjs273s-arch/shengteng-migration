import zipfile, sys
z = sys.argv[1]
with zipfile.ZipFile(z) as f:
    names = f.namelist()
    bad = [n for n in names if (".bak" in n) or n.endswith("~") or n.endswith(".tmp")]
    print("zip 条目 =", len(names))
    print("含备份/临时文件的条目 =", len(bad))
    for n in bad:
        print("  %-72s %d B" % (n, f.getinfo(n).file_size))
    sys.path.insert(0, r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建")
    from safe_pack_sync import TOKEN_RE
    hits = []
    for n in bad:
        txt = f.read(n).decode("utf-8", "ignore")
        if TOKEN_RE.search(txt):
            hits.append(n)
    print("备份文件里含疑似真 token 的个数 =", len(hits))
    for n in hits:
        print("  ★", n)
