#!/usr/bin/env python3
"""
Process all parquet files in train_v2_rawparquet:
- Flatten every field.
- customDimensions: flatten to value only ("North America", "apac", "EMEA", or "").
- hits: follow hits_metrics.py (total_product_viewed, average_price_product_viewed,
  total_category_viewed, categories_viewed_0..N, total_revenue).
- device, geoNetwork, totals, trafficSource: parse JSON and flatten to prefix_key columns.

Output: train_v2_parquet_logicapplied/<same_filename>.parquet

Run from project root: python3 flatten_all_parquet.py
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from flatten_customDimensions import flatten_custom_dimensions
from hits_metrics import compute_hits_metrics

PARQUET_DIR_DEFAULT = "train_v2_rawparquet"
OUT_DIR_DEFAULT = "train_v2_parquet_logicapplied"


def _flatten_dict(obj: Any, prefix: str = "") -> Dict[str, str]:
    """Recursively flatten a dict to prefix_key -> string value (for consistent parquet types)."""
    out: Dict[str, str] = {}
    if not isinstance(obj, dict):
        return {prefix: str(obj)} if prefix else {}
    for k, v in obj.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            out[key] = str(v) if v else ""
        else:
            out[key] = str(v) if v is not None else ""
    return out


def _parse_json_safe(raw: Optional[str]) -> Dict[str, Any]:
    """Parse JSON string; return {} on failure."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply flattening logic to a DataFrame; return new DataFrame with flattened columns."""
    flat_cols = [
        "channelGrouping",
        "date",
        "fullVisitorId",
        "visitId",
        "visitNumber",
        "visitStartTime",
        "socialEngagementType",
    ]

    rows_flat: List[Dict[str, Any]] = []

    n_rows = len(df)
    for i in range(n_rows):
        if (i + 1) % 50000 == 0 or i == 0:
            pct = 100 * (i + 1) / n_rows
            print(f"    progress: {i + 1:,}/{n_rows:,} rows ({pct:.1f}%)", flush=True)
        row_flat: Dict[str, Any] = {}
        for col in flat_cols:
            row_flat[col] = df.iloc[i][col]
        row_flat["customDimensions"] = flatten_custom_dimensions(df.iloc[i]["customDimensions"])

        for col, prefix in [
            ("device", "device"),
            ("geoNetwork", "geoNetwork"),
            ("totals", "totals"),
            ("trafficSource", "trafficSource"),
        ]:
            parsed = _parse_json_safe(df.iloc[i][col])
            for k, v in _flatten_dict(parsed, prefix).items():
                row_flat[k] = v

        hits_raw = df.iloc[i]["hits"]
        totals_raw = df.iloc[i]["totals"]
        metrics = compute_hits_metrics(
            hits_raw, totals_raw=totals_raw, use_totals_for_revenue_if_empty=True
        )
        row_flat["total_product_viewed"] = metrics["total_product_viewed"]
        row_flat["average_price_product_viewed"] = metrics["average_price_product_viewed"]
        row_flat["total_category_viewed"] = metrics["total_category_viewed"]
        for j, cat in enumerate(metrics["categories_viewed"]):
            row_flat[f"categories_viewed_{j}"] = cat
        row_flat["total_revenue"] = metrics["total_revenue"]

        rows_flat.append(row_flat)

    all_keys = sorted(set().union(*(r.keys() for r in rows_flat)))
    out_data: Dict[str, List[Any]] = {k: [] for k in all_keys}
    for row in rows_flat:
        for k in all_keys:
            out_data[k].append(row.get(k, ""))

    return pd.DataFrame(out_data)


def main(parquet_dir: str = PARQUET_DIR_DEFAULT, out_dir: str = OUT_DIR_DEFAULT) -> None:
    os.makedirs(out_dir, exist_ok=True)

    parts = sorted(
        f for f in os.listdir(parquet_dir) if f.endswith(".parquet")
    )
    if not parts:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    n_files = len(parts)
    print(f"Found {n_files} parquet file(s) in {parquet_dir}/", flush=True)

    for file_idx, filename in enumerate(parts, start=1):
        in_path = os.path.join(parquet_dir, filename)
        out_path = os.path.join(out_dir, filename)
        t0 = time.perf_counter()
        print(f"\n[{file_idx}/{n_files}] Processing {filename} ...", flush=True)
        df = pd.read_parquet(in_path)
        print(f"    loaded {len(df):,} rows", flush=True)
        result_df = process_df(df)
        result_df.to_parquet(out_path, index=False)
        elapsed = time.perf_counter() - t0
        print(f"    -> saved {len(result_df):,} rows, {len(result_df.columns)} cols to {out_path} ({elapsed:.1f}s)", flush=True)

    print(f"\nDone. Output in {out_dir}/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flatten parquet (device, geo, totals, hits, etc.)")
    ap.add_argument("--input-dir", default=PARQUET_DIR_DEFAULT, help="Input parquet directory")
    ap.add_argument("--output-dir", default=OUT_DIR_DEFAULT, help="Output parquet directory")
    args = ap.parse_args()
    main(parquet_dir=args.input_dir, out_dir=args.output_dir)
