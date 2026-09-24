import io, re, sys
sys.path.insert(0, ".")
from check_report_pdf import _squash, _extract
md = io.open(r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料\04_性能测试报告\性能优化报告.md", encoding="utf-8-sig").read()
body = "\n".join(l for l in md.splitlines() if l.strip())
head = _squash(body)[:40]
txt, err = _extract(r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料\04_性能测试报告\性能测试报告.pdf")
p = _squash(txt)
print("md head =", repr(head))
i = p.find("更正")
print("pdf 附近 =", repr(p[max(0,i-6):i+44]))
n = 0
while n < min(len(head), len(p)) and head[n] == p[p.find("更正")-2+n] if p.find("更正")>1 else False:
    n += 1
print("逐字相同前缀长度 =", n)
print("差异字符 md=%r pdf=%r" % (head[n:n+3], p[p.find("更正")-2+n:p.find("更正")-2+n+3]))
