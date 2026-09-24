import zipfile, hashlib
for z in [r"C:\Users\HUAWEI\Desktop\转交给codex的内容\05_交付物成品\06_Skill\qwen35-ascend-migrator_20260921_v1.0.zip",
          r"C:\Users\HUAWEI\Desktop\转交给codex的内容\05_交付物成品\06_Skill\qwen35-ascend-migrator_20260922_v1.0.zip"]:
    with zipfile.ZipFile(z) as f:
        n = f.namelist()
        pit = [x for x in n if "PITFALLS" in x][0]
        data = f.read(pit)
        txt = data.decode("utf-8-sig")
        import re
        ids = [int(m) for m in re.findall(r"^\|\s*\*\*(\d+)\*\*\s*\|", txt, re.M)]
        print(z.split("\\")[-1], "entries=%d" % len(n), "pit_max=%s count=%d" % (max(ids), len(ids)), "pit_sha=%s" % hashlib.sha256(data).hexdigest()[:12])
