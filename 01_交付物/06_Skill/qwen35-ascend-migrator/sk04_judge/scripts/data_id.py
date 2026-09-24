#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_id.py — R2-SK04 · 数据集身份指纹（可比性机器化三件套之二，来源 R2-C6-02）

功能
----
1) 单文件模式：对数据集 json 输出身份指纹
   {json_sha256, order_sha256, row_count, 字段 schema 摘要}：
     json_sha256   = 文件原始字节 sha256（字节级身份）
     order_sha256  = 逐样本 id 按行序串联的哈希（样本 id = --id-key 字段值；缺省时取
                     该样本的规范 JSON 序列化 —— 即"内容哈希当 id"，见下）
     row_count     = 样本行数（顶层 list 元素数 / 顶层 dict 的最大 list 值元素数）
     schema 摘要    = 前 N 行样本的顶层字段集 + 值类型 + 消息角色取值
2) --compare A B 双文件模式：输出 C6-02 三态（three_state）
     同字节 / 同内容（含异序）/ 未核对
   细粒度 status（同内容再按行序细分）：
     same_bytes                    （json_sha256 相同 → 同字节同序）
     same_content_same_order       （字节不同但逐样本内容与行序相同 = 仅格式差）
     same_content_reordered        （row_count 相同且样本 id 多重集合相同、行序哈希不同
                                    → 同内容异序）
     different / unverified        （其余；文件缺失/不可读 → 未核对）
判定语义（诚实边界）
  * order_sha256 的"样本 id"：llava/mock json 通常无显式样本 id 字段（schema 见下），
    缺省用"逐样本规范序列化哈希"做 id —— 它是内容强指纹：样本内容相同 ⇒ id 相同，
    与"逐样本 id"语义等价，且不依赖字段名。json 若有 id 字段（--id-key）则优先用它
    （更快、对'仅字段顺序不同'不敏感）。
  * "同内容异序"只证明 id 多重集合相等（内容层面），不证明两份文件字节/格式一致；
    要claim 逐步 batch 相同还需 shuffle=False + 同 sampler（由 fingerprint_cfg 的
    sample_order 门承接）。判断保守：id 多重集合不同 → different（不做内容近似推断）。
  * 官方 A/B 的数据 json 本机不可得（A2 COCO: annotations_slim.json 在 A2 上；
    B: output_llava_coco_data.json 未随附）→ 官方侧身份 unresolved（诚实标注，不回填假值）。

纯 CPU；仅 Python 标准库（hashlib/json/argparse）。不依赖 torch/A2/pyyaml。

用法
----
  python3 scripts/data_id.py --json <数据集.json> [--id-key id] [--out identity.json]
  python3 scripts/data_id.py --compare <A.json> <B.json> [--id-key id] [--out cmp.json]
  # 输出含一行机器可读: DATA_ID_OK json_sha256=.. order_sha256=.. row_count=..
  #                   DATA_CMP_OK status=same_bytes|same_content_reordered|different|unverified
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCHEMA_ROWS = 3          # schema 摘要扫描的样本行数（2000 样本全扫亦可行，取前 3 行做摘要足够）
_CANON_SEP = b"\x1f"      # 样本序列化分隔符（控制字符，避免样本内文本撞串）


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def load_rows(path: str):
    """返回 (rows, kind, note)：rows = 样本 list。
    kind: list | dict_list（顶层 dict 中取"元素数最多的 list 值"作为样本表）。"""
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8-sig"))   # utf-8-sig：容忍 BOM
    if isinstance(doc, list):
        return doc, "list", ""
    if isinstance(doc, dict):
        best_key, best = None, None
        for k, v in doc.items():
            if isinstance(v, list) and (best is None or len(v) > len(best)):
                best_key, best = k, v
        if best is not None:
            return best, "dict_list", "顶层 dict 的 list 字段 '%s'（%d 样本）" % (best_key, len(best))
        return [], "unrecognized", "顶层既非 list 也不是含 list 的 dict"
    return [], "unrecognized", "顶层类型 %s，无法识别样本表" % type(doc).__name__


def sample_ids(rows, id_key):
    """逐样本 id 列表：优先 id_key 字段值；否则样本规范 JSON 的 sha256 前缀（内容即身份）。"""
    out = []
    for r in rows:
        if id_key is not None and isinstance(r, dict) and id_key in r:
            v = r[id_key]
            out.append(v if isinstance(v, str) else _json_bytes(v).hex())
        else:
            out.append(sha256_bytes(_json_bytes(r))[:40])
    return out


def identity_of_file(path: str, id_key=None, sample_rows=_SCHEMA_ROWS):
    """单文件身份指纹 dict（含 schema 摘要）。文件不存在 → 返回 unresolved 结构。"""
    p = Path(path)
    if not p.is_file():
        return {"status": "unresolved", "note": "文件不存在: %s" % p}
    raw = p.read_bytes()
    json_sha = sha256_bytes(raw)
    try:
        rows, kind, note = load_rows(path)
    except Exception as e:
        return {"status": "unreadable", "note": "json 解析失败: %s" % e,
                "json_sha256": json_sha}
    ids = sample_ids(rows, id_key)
    order_sha = sha256_bytes(_CANON_SEP.join(s.encode("utf-8") for s in ids)) if ids \
        else sha256_bytes(b"")
    # ---- schema 摘要（前 sample_rows 行样本的字段/类型并集 + 角色取值）
    schema = {"field_types": {}, "role_values": [], "container": kind,
              "note": note, "sampled_rows": min(len(rows), sample_rows)}
    if isinstance(rows, list):
        for r in rows[:sample_rows]:
            if isinstance(r, dict):
                for k, v in r.items():
                    vt = type(v).__name__
                    prev = schema["field_types"].get(k)
                    schema["field_types"][k] = vt if prev is None or prev == vt else prev + "/" + vt
            else:
                schema["row_type"] = type(r).__name__
    # 消息角色取值（attr.role_tag=role 的口径；仅在 messages 存在时摘录）
    def _roles_of(row):
        roles = []
        msgs = row.get("messages") if isinstance(row, dict) else None
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and "role" in m:
                    roles.append(m["role"])
        return roles

    for r in rows[:sample_rows]:
        if isinstance(r, dict):
            schema["role_values"].extend(_roles_of(r))
    schema["role_values"] = sorted(set(str(x) for x in schema["role_values"]))
    return {"status": "ok", "path": str(p), "json_sha256": json_sha,
            "order_sha256": order_sha, "row_count": len(rows), "schema": schema}


def compare_two(path_a, path_b, id_key=None):
    """双文件判定。status（细粒度）∈ same_bytes | same_content_same_order |
    same_content_reordered | different | unverified；
    three_state（C6-02 三态）∈ 同字节 | 同内容（含异序） | 未核对/不同源。
    返回 dict + 一行机器可读。"""
    ia = identity_of_file(path_a, id_key)
    ib = identity_of_file(path_b, id_key)
    out = {"a": ia, "b": ib, "status": None, "three_state": None, "detail": ""}
    if ia.get("status") != "ok" or ib.get("status") != "ok":
        out["status"] = "unverified"
        out["three_state"] = "未核对"
        out["detail"] = "任一侧文件不可用（%s/%s）→ 未核对" % (
            ia.get("note", ia.get("status")), ib.get("note", ib.get("status")))
        return out
    if ia["json_sha256"] == ib["json_sha256"]:
        out["status"] = "same_bytes"
        out["three_state"] = "同字节"
        return out
    if ia["row_count"] != ib["row_count"]:
        out["status"] = "different"
        out["three_state"] = "未核对（row_count 不同，非同源）"
        out["detail"] = "row_count 不同: %d vs %d → 非同源数据" % (ia["row_count"], ib["row_count"])
        return out
    ids_a = sample_ids(load_rows(path_a)[0], id_key)
    ids_b = sample_ids(load_rows(path_b)[0], id_key)
    if ids_a == ids_b:
        out["status"] = "same_content_same_order"
        out["three_state"] = "同内容"
        out["detail"] = "字节不同但逐样本内容/行序相同（格式/字段序差异）——逐步 batch 组成仍一致"
        return out
    if sorted(ids_a) == sorted(ids_b):
        out["status"] = "same_content_reordered"
        out["three_state"] = "同内容（异序）"
        out["verdict"] = "同内容异序（样本 id 多重集合相同、行序不同）"
    else:
        out["status"] = "different"
        out["three_state"] = "未核对（样本 id 多重集合不同，非同源）"
        out["detail"] = "样本 id 多重集合不同 → 非同源数据"
    return out


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 数据集身份指纹（单文件/双文件三态）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", help="单文件模式：数据集 json 路径")
    g.add_argument("--compare", nargs=2, metavar=("JSON_A", "JSON_B"),
                   help="双文件模式：比较两份 json → 三态")
    ap.add_argument("--id-key", default=None,
                    help="样本 id 字段名（缺省=逐样本规范序列化哈希当 id）")
    ap.add_argument("--out", default=None, help="JSON 输出路径（默认 stdout）")
    args = ap.parse_args()

    if args.compare:
        out = compare_two(args.compare[0], args.compare[1], args.id_key)
    else:
        out = identity_of_file(args.json, args.id_key)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.compare:
        print("DATA_CMP_OK status=%s three_state=%s a_row=%s b_row=%s a_json_sha=%s b_json_sha=%s"
              % (out["status"], out["three_state"],
                 out["a"].get("row_count", "-"), out["b"].get("row_count", "-"),
                 out["a"].get("json_sha256", "-")[:12], out["b"].get("json_sha256", "-")[:12]))
        return 0 if out["status"] in ("same_bytes", "same_content_same_order",
                                      "same_content_reordered") else 1
    print("DATA_ID_OK status=%s json_sha256=%s order_sha256=%s row_count=%s"
          % (out.get("status"), out.get("json_sha256", "-"),
             out.get("order_sha256", "-"), out.get("row_count", "-")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
