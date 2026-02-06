#!/usr/bin/env python3
"""
Cast totals_* columns from object to numeric (in-place).
Default dir: train_v2_parquet_logicapplied_cleaned
"""

import argparse
import os
import time

import pandas as pd

PARQUET_DIR_DEFAULT = "train_v2_parquet_logicapplied_cleaned"

# Columns to cast to numeric (only those that exist)
TOTALS_INT_COLS = [
    "totals_bounces",
    "totals_hits",
    "totals_newVisits",
    "totals_pageviews",
    "totals_sessionQualityDim",
    "totals_timeOnSite",
    "totals_transactions",
    "totals_visits",
]
TOTALS_FLOAT_COLS = [
    "totals_totalTransactionRevenue",
    "totals_transactionRevenue",
]


def main(parquet_dir: str = PARQUET_DIR_DEFAULT) -> None:
    parts = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    for filename in parts:
        path = os.path.join(parquet_dir, filename)
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        for col in TOTALS_INT_COLS:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        for col in TOTALS_FLOAT_COLS:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df.to_parquet(path, index=False)
        elapsed = time.perf_counter() - t0
        print(f"  {filename}: cast totals_* to numeric ({len(df):,} rows) in {elapsed:.1f}s")
    print(f"Done. Updated {len(parts)} file(s) in {parquet_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cast totals_* to numeric (in-place)")
    ap.add_argument("--input-dir", default=PARQUET_DIR_DEFAULT, help="Parquet directory to update")
    args = ap.parse_args()
    main(parquet_dir=args.input_dir)
