"""Strict, shared parser for MindSpeed-MM iteration records.

The returned rows retain file order, rank and line number.  Consumers must
check ``issues`` and ``integrity`` before using rows for a formal result.
"""
import hashlib
import math
import re
from pathlib import Path


RANK = re.compile(r"\[Rank\s+(\d+)\s*\|\s*Local Rank\s+(\d+)\]")
ITER = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)\b")
FIELDS = {
    "consumed_samples": re.compile(r"\bconsumed samples:\s*(\d+)"),
    "ms": re.compile(r"\belapsed time per iteration \(ms\):\s*([^|\s]+)"),
    "gbs": re.compile(r"\bglobal batch size:\s*(\d+)"),
    "loss": re.compile(r"\bloss:\s*([^|\s]+)"),
    "grad_norm": re.compile(r"\bgrad norm:\s*([^|\s]+)"),
}


def read_log(path):
    raw = Path(path).read_bytes()
    rows, issues = [], []
    for line_no, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if not re.search(r"\biteration\s+\S*\s*/", line):
            continue
        match = ITER.search(line)
        rank_match = RANK.search(line)
        if not match or not rank_match:
            issues.append({"line": line_no, "kind": "malformed_iteration_or_rank", "text": line[:300]})
            continue
        values = {key: pat.search(line) for key, pat in FIELDS.items()}
        if any(value is None for value in values.values()):
            issues.append({"line": line_no, "kind": "missing_fields", "text": line[:300]})
            continue
        try:
            row = {"iter": int(match[1]), "total": int(match[2]),
                   "rank": int(rank_match[1]), "local_rank": int(rank_match[2]),
                   "line": line_no,
                   "consumed_samples": int(values["consumed_samples"][1]),
                   "gbs": int(values["gbs"][1]),
                   **{key: float(values[key][1]) for key in ("ms", "loss", "grad_norm")}}
        except (ValueError, OverflowError):
            issues.append({"line": line_no, "kind": "invalid_numeric", "text": line[:300]})
            continue
        if (row["iter"] < 1 or row["total"] < row["iter"] or row["gbs"] < 1
                or row["ms"] <= 0 or any(not math.isfinite(row[key])
                                          for key in ("ms", "loss", "grad_norm"))):
            issues.append({"line": line_no, "kind": "invalid_numeric", "text": line[:300]})
            continue
        rows.append(row)
    return {"rows": rows, "issues": issues, "log_sha256": hashlib.sha256(raw).hexdigest(),
            "log_bytes": len(raw)}


def integrity(parsed, expected_end=None, start_step=1, expected_gbs=None, rank=0):
    """Validate an inclusive step interval; None start means unknown resume point."""
    rows = parsed["rows"]
    selected = [row for row in rows if row["rank"] == rank]
    counts = {}
    for row in selected:
        counts[row["iter"]] = counts.get(row["iter"], 0) + 1
    duplicates = sorted(step for step, count in counts.items() if count > 1)
    out_of_order = []
    for previous, current in zip(selected, selected[1:]):
        if current["iter"] <= previous["iter"]:
            out_of_order.append({"previous_step": previous["iter"],
                                 "previous_line": previous["line"],
                                 "step": current["iter"], "line": current["line"]})
    totals = sorted({row["total"] for row in rows})
    gbs_values = sorted({row["gbs"] for row in rows})
    ranks = sorted({row["rank"] for row in rows})
    expected = set(range(start_step, expected_end + 1)) if (
        start_step is not None and expected_end is not None and
        1 <= start_step <= expected_end) else None
    missing = sorted(expected - counts.keys()) if expected is not None else None
    extra = sorted(counts.keys() - expected) if expected is not None else None
    problems = []
    if parsed["issues"]:
        problems.append("bad_iteration_records")
    if not selected:
        problems.append("no_selected_rank_records")
    if duplicates:
        problems.append("duplicate_steps")
    if out_of_order:
        problems.append("non_increasing_step_order")
    if ranks != [rank]:
        problems.append("unexpected_or_mixed_ranks")
    if len(totals) != 1 or (expected_end is not None and totals != [expected_end]):
        problems.append("total_conflict")
    if len(gbs_values) != 1 or (expected_gbs is not None and gbs_values != [expected_gbs]):
        problems.append("gbs_conflict")
    if missing:
        problems.append("missing_steps")
    if extra:
        problems.append("unexpected_steps")
    if expected is None:
        problems.append("unknown_expected_range")
    return {"schema": "train_integrity.v1", "state": "COMPLETE" if not problems else
            ("UNKNOWN" if problems == ["unknown_expected_range"] else "INCOMPLETE"),
            "expected_start": start_step, "expected_end": expected_end,
            "expected_gbs": expected_gbs, "selected_rank": rank,
            "ranks": ranks, "totals": totals, "gbs_values": gbs_values,
            "record_count": len(rows), "selected_count": len(selected),
            "unique_steps": sorted(counts), "duplicate_steps": duplicates,
            "out_of_order": out_of_order,
            "missing_steps": missing, "unexpected_steps": extra,
            "bad_records": parsed["issues"], "problems": problems,
            "log_sha256": parsed["log_sha256"], "log_bytes": parsed["log_bytes"]}
