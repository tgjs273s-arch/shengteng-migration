#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
registry_append.py — R2-SK04 · append-only 运行账本（来源 R2-C6-03）

功能
----
维护 evidence/registry.json 的 append-only 哈希链账本。每次追加一条运行记录：
    {seq, tag, run_dir, manifest_sha256, fingerprint_digest, observed_digest,
     prev_hash, ts, entry_hash}
链语义：
  * entry_hash = sha256(该 entry 去掉 entry_hash 后的规范 JSON) —— 追加后任何一条
    被篡改/重排都会在下次 verify 时被检出（防证据漂移，C6-03）；
  * prev_hash = 前一条 entry_hash（首条 = "GENESIS"）；
  * 只允许 append：tag 重复 → 拒绝；旧记录永不改写（写失败标 registry_pending，不阻塞
    主流程 —— C6-03"registry 写失败标 registry_pending 不阻塞"）。

确定性：哈希全部基于文件字节 + 时间戳字段（账本记录运行时刻是设计内行为；判定/指纹
         工具本身无墙钟）。纯 CPU；仅 Python 标准库。不依赖 torch/A2/pyyaml。

用法
----
  python3 scripts/registry_append.py --tag run_<tag> --run-dir <runs/run_<tag>>
      --manifest <manifest.json>          # sha256 记入 manifest_sha256
      --fingerprint <fingerprint_cfg.json>   # sha256 记入 fingerprint_digest
      [--observed <fingerprint_observed.json>]
      [--registry evidence/registry.json]  # 默认 skill 内 evidence/registry.json
      [--dry-run]                          # 只算不写
  python3 scripts/registry_append.py --verify --registry <file>   # 校验整链
  python3 scripts/registry_append.py --tail 3 --registry <file>   # 显示末 3 条
退出码：0=OK（已追加/校验通过）；2=用法/IO/写失败（打印 REGISTRY_PENDING）；3=链校验失败
       或 tag 重复（拒绝追加）；4=被追加文件缺失。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = SKILL_ROOT / "evidence" / "registry.json"
GENESIS = "GENESIS"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def entry_hash_of(entry):
    """entry_hash = sha256(去 entry_hash 字段后的规范 JSON)。"""
    e = {k: v for k, v in entry.items() if k != "entry_hash"}
    return sha256_bytes(canon(e).encode("utf-8"))


def load_registry(path):
    """读账本。文件不存在 → (空 entries, None)。损坏 → 抛 ValueError。"""
    p = Path(path)
    if not p.is_file():
        return [], None
    # utf-8-sig：容忍 Windows 工具写出的 BOM（json.loads 对 BOM 会炸）
    doc = json.loads(p.read_text(encoding="utf-8-sig"))
    entries = doc.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("registry 结构损坏（entries 非 list）")
    return entries, doc


def verify_chain(entries):
    """返回 (ok, problem)。逐条重算 entry_hash 并与存储值比对 + prev 链接。"""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("prev_hash") != prev:
            return False, "seq=%d prev_hash 断链（期望 %s，实际 %s）" % (e.get("seq"), prev, e.get("prev_hash"))
        calc = entry_hash_of(e)
        if calc != e.get("entry_hash"):
            return False, "seq=%d entry_hash 不匹配（重算 %s，存储 %s）→ 证据可能被改" % (e.get("seq"), calc[:16], (e.get("entry_hash") or "")[:16])
        prev = e.get("entry_hash")
    return True, ""


def dump_registry(path, entries, meta_extra=None):
    """原子写（tmp + os.replace）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "sk04_registry.v1", "entries": entries}
    if meta_extra:
        doc.update(meta_extra)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 append-only 运行账本（哈希链）")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--fingerprint", default=None)
    ap.add_argument("--observed", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="只校验整链")
    ap.add_argument("--tail", type=int, default=None, help="显示末 N 条")
    args = ap.parse_args()

    try:
        entries, doc = load_registry(args.registry)
        ok, problem = verify_chain(entries)
        if not ok:
            print("REGISTRY_CHAIN_BROKEN %s（%s）→ 拒绝追加/操作" % (args.registry, problem),
                  file=sys.stderr)
            return 3
    except ValueError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 3
    except OSError as e:
        print("REGISTRY_PENDING cannot_read %s (%s) —— 账本不可用不阻塞主流程" % (args.registry, e),
              file=sys.stderr)
        return 2

    if args.verify:
        print("REGISTRY_VERIFY_OK entries=%d prev_tail=%s" %
              (len(entries), entries[-1]["entry_hash"][:16] if entries else GENESIS))
        return 0
    if args.tail:
        for e in entries[-args.tail:]:
            print("REG[%d] tag=%s run_dir=%s manifest_sha=%s fp_digest=%s prev=%s hash=%s"
                  % (e.get("seq"), e.get("tag"), e.get("run_dir"),
                     (e.get("manifest_sha256") or "")[:12], (e.get("fingerprint_digest") or "")[:12],
                     (e.get("prev_hash") or "")[:12], (e.get("entry_hash") or "")[:16]))
        return 0

    if not (args.tag and args.run_dir and args.manifest and args.fingerprint):
        print("FATAL 追加模式需要 --tag/--run-dir/--manifest/--fingerprint", file=sys.stderr)
        return 2

    tag = args.tag
    if any(e.get("tag") == tag for e in entries):
        print("REGISTRY_DUP_TAG tag=%s 已存在 → 拒绝追加（换 tag；账本 append-only）" % tag,
              file=sys.stderr)
        return 3

    # 被引用文件必须存在（防悬空证据）
    for argname in ("manifest", "fingerprint", "observed"):
        v = getattr(args, argname)
        if v and not Path(v).is_file():
            print("FATAL --%s 文件不存在: %s" % (argname, v), file=sys.stderr)
            return 4

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {"seq": len(entries) + 1, "tag": tag, "run_dir": args.run_dir,
             "manifest_sha256": sha256_file(args.manifest),
             "fingerprint_digest": sha256_file(args.fingerprint),
             "observed_digest": sha256_file(args.observed) if args.observed else None,
             "prev_hash": entries[-1]["entry_hash"] if entries else GENESIS,
             "ts": ts}
    entry["entry_hash"] = entry_hash_of(entry)

    if args.dry_run:
        print("REGISTRY_DRYRUN seq=%d tag=%s entry_hash=%s prev=%s"
              % (entry["seq"], tag, entry["entry_hash"][:16], (entry["prev_hash"] or "")[:12]))
        return 0

    try:
        dump_registry(args.registry, entries + [entry])
    except OSError as e:
        print("REGISTRY_PENDING cannot_write %s (%s) —— 追加失败不阻塞主流程，运行记录仍以 "
              "run 目录证据为准（tag=%s）" % (args.registry, e, tag), file=sys.stderr)
        return 2
    print("REGISTRY_APPEND_OK seq=%d tag=%s entry_hash=%s prev=%s entries=%d"
          % (entry["seq"], tag, entry["entry_hash"][:16], (entry["prev_hash"] or "")[:12],
             len(entries) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
