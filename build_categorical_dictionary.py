#!/usr/bin/env python3
"""
Build a zero-based categorical dictionary from eda_cleaned_all_fields_report.txt.

Unified encoding (globally consistent, reproducible, safe for Train/Val/Test):
  0       = __NONE__   (missing / empty / true missing)
  1       = __UNKNOWN__ (system unknown / not available / not collected)
  2..N+1  = Top-N categories (by training frequency; N = actual retained count)
  N+2     = __OTHER__  (long tail + unseen in test)

Per-column: top_n (actual), reserved=2, others_index=2+top_n, vocab_size=others_index+1.
NONE vs UNKNOWN value sets are defined separately in meta.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "eda_cleaned_all_fields_report.txt"
OUT_PATH = SCRIPT_DIR / "categorical_dictionary.json"

# Canonical reserved labels (must match encoding)
LABEL_NONE = "__NONE__"
LABEL_UNKNOWN = "__UNKNOWN__"
LABEL_OTHER = "__OTHER__"

# 3.1 NONE (missing): value is truly empty / missing
NONE_VALUES = frozenset({
    "",                    # empty string
    " ",                   # single space (all-space treated in code)
    "(not provided)",
    "not available in demo dataset",
    "None",                # string form of Python None
})
# We also treat None (Python) and all-blank string as NONE

# 3.2 UNKNOWN (unknown but not missing): system placeholder for not available / unknown
UNKNOWN_VALUES = frozenset({
    "(not set)",
    "unknown.unknown",
    "Unknown",
    "unknown",
    "not set",
})

# (Other) in data maps to __OTHER__
OTHER_RAW_VALUES = frozenset({"(Other)"})

# Numerical columns (excluded from categorical dictionary)
NUMERIC_COLUMNS = frozenset({
    "average_price_product_viewed",
    "date",
    "total_category_viewed",
    "total_product_viewed",
    "total_revenue",
    "visitId",
    "visitNumber",
    "visitStartTime",
    "totals_bounces",
    "totals_hits",
    "totals_newVisits",
    "totals_pageviews",
    "totals_sessionQualityDim",
    "totals_timeOnSite",
    "totals_totalTransactionRevenue",
    "totals_transactionRevenue",
    "totals_transactions",
    "totals_visits",
})

# Joint-key columns: keep as raw identifiers, do not encode
JOINT_KEY_COLUMNS = frozenset({"fullVisitorId", "visitId"})


def _is_none(raw) -> bool:
    """True if value is NONE (missing)."""
    if raw is None:
        return True
    s = (raw.strip() if isinstance(raw, str) else str(raw)).strip()
    if s == "" or s in NONE_VALUES or s.lower() == "none":
        return True
    return False


def _is_unknown(raw) -> bool:
    """True if value is UNKNOWN (system placeholder)."""
    if raw is None:
        return False
    s = (raw.strip() if isinstance(raw, str) else str(raw)).strip()
    if s in UNKNOWN_VALUES or s.lower() in ("unknown", "not set"):
        return True
    return False


def _is_other_raw(raw) -> bool:
    """True if value is (Other) etc. that maps to __OTHER__."""
    if raw is None:
        return False
    s = (raw.strip() if isinstance(raw, str) else str(raw)).strip()
    return s in OTHER_RAW_VALUES


def _skip_for_top(raw) -> bool:
    """True if this value should not consume a top-N slot (reserved or other)."""
    return _is_none(raw) or _is_unknown(raw) or _is_other_raw(raw)


def _parse_value_count_line(line: str) -> Optional[Tuple[Optional[str], int]]:
    """Parse a line like "      'YouTube': 202,705" or "      '': 852,174". Returns (value, count) or None."""
    line = line.strip()
    m = re.match(r"^['\"](.*?)['\"]\s*:\s*([\d,]+)\s*$", line)
    if m:
        val, cnt = m.group(1), m.group(2).replace(",", "")
        return (val.replace("\\'", "'").replace('\\"', '"'), int(cnt))
    m = re.match(r"^None\s*:\s*([\d,]+)\s*$", line, re.IGNORECASE)
    if m:
        return (None, int(m.group(1).replace(",", "")))
    return None


def parse_eda_report(path: Path) -> dict:
    """Parse the EDA report; return dict of column_name -> { "unique": int, "value_counts": [(value, count), ...] }."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n----------------------------------------------------------------------\n", text)
    result = {}
    for block in blocks:
        block = block.strip()
        lines = block.split("\n")
        if not lines:
            continue
        col_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*$", lines[0])
        if not col_match:
            continue
        col_name = col_match.group(1)
        unique = None
        value_counts = []
        in_value_counts = False
        for line in lines[1:]:
            if "unique:" in line:
                m = re.search(r"unique:\s*([\d,]+)", line)
                if m:
                    unique = int(m.group(1).replace(",", ""))
            if "value_counts (top 15):" in line:
                in_value_counts = True
                continue
            if in_value_counts:
                parsed = _parse_value_count_line(line)
                if parsed is not None:
                    value_counts.append(parsed)
                elif line.strip() == "":
                    break
        if unique is not None and col_name:
            if value_counts or "dtype: String" in block:
                result[col_name] = {"unique": unique, "value_counts": value_counts}
    return result


def build_zero_based_dictionary(parsed: dict) -> dict:
    """
    Build the zero-based dictionary with unified encoding:
      0 = __NONE__, 1 = __UNKNOWN__, 2..N+1 = Top-N, N+2 = __OTHER__
    """
    out = {
        "meta": {
            "encoding": {
                "0": LABEL_NONE,
                "1": LABEL_UNKNOWN,
                "2..N+1": "Top-N categories (by frequency)",
                "N+2": LABEL_OTHER,
            },
            "none_values": sorted(NONE_VALUES),
            "unknown_values": sorted(UNKNOWN_VALUES),
            "other_raw_values": sorted(OTHER_RAW_VALUES),
            "reserved_indices": 2,
            "joint_key_columns": sorted(JOINT_KEY_COLUMNS),
            "description": "0=__NONE__, 1=__UNKNOWN__, 2..(N+1)=Top-N, (N+2)=__OTHER__. top_n=actual retained; others_index=2+top_n; vocab_size=others_index+1.",
        },
        "columns": {},
    }
    for col_name, data in parsed.items():
        requested_top_k = data["unique"]
        value_counts = data["value_counts"]
        max_from_report = len(value_counts)
        # Actual top-N: distinct categories from value_counts, skipping reserved/other
        seen = set()
        top_categories = []
        for val, _ in value_counts[:max_from_report]:
            if _skip_for_top(val):
                continue
            # Use literal value for category name (preserve casing etc.)
            key = val if isinstance(val, str) else str(val)
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                top_categories.append(key)
        top_n = len(top_categories)
        others_index = 2 + top_n
        vocab_size = others_index + 1

        # value_to_index
        value_to_idx = {}
        value_to_idx[LABEL_NONE] = 0
        value_to_idx[LABEL_UNKNOWN] = 1
        for i, cat in enumerate(top_categories):
            value_to_idx[cat] = 2 + i
        value_to_idx[LABEL_OTHER] = others_index
        # Map all NONE raw values -> 0
        value_to_idx[""] = 0
        value_to_idx["None"] = 0
        for v in NONE_VALUES:
            value_to_idx[v] = 0
        # Map all UNKNOWN raw values -> 1
        for v in UNKNOWN_VALUES:
            value_to_idx[v] = 1
        value_to_idx["unknown"] = 1
        value_to_idx["Unknown"] = 1
        value_to_idx["not set"] = 1
        # (Other) -> others_index
        value_to_idx["(Other)"] = others_index

        # index_to_value: canonical labels for 0, 1, others_index
        idx_to_val = {}
        idx_to_val["0"] = LABEL_NONE
        idx_to_val["1"] = LABEL_UNKNOWN
        for i, cat in enumerate(top_categories):
            idx_to_val[str(2 + i)] = cat
        idx_to_val[str(others_index)] = LABEL_OTHER

        out["columns"][col_name] = {
            "requested_top_k": requested_top_k,
            "top_n": top_n,
            "reserved": 2,
            "others_index": others_index,
            "vocab_size": vocab_size,
            "value_to_index": value_to_idx,
            "index_to_value": idx_to_val,
            "top_categories_order": top_categories,
        }
    return out


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"EDA report not found: {REPORT_PATH}")
    parsed = parse_eda_report(REPORT_PATH)
    string_cols = [
        c for c, d in parsed.items()
        if d["value_counts"] and c not in NUMERIC_COLUMNS and c not in JOINT_KEY_COLUMNS
    ]
    parsed = {c: parsed[c] for c in string_cols}
    d = build_zero_based_dictionary(parsed)
    OUT_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(d['columns'])} categorical columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
