#!/usr/bin/env python3
"""
Cast fullVisitorId and visitId to string (object), in-place.
Default dir: train_v2_parquet_logicapplied_cleaned
"""

import argparse
import os
import time

import pandas as pd

PARQUET_DIR_DEFAULT = "train_v2_parquet_logicapplied_cleaned"
ID_COLS = ["fullVisitorId", "visitId"]


def main(parquet_dir: str = PARQUET_DIR_DEFAULT) -> None:
    parts = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    for filename in parts:
        path = os.path.join(parquet_dir, filename)
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        for col in ID_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df.to_parquet(path, index=False)
        elapsed = time.perf_counter() - t0
        print(f"  {filename}: fullVisitorId, visitId -> string ({len(df):,} rows) in {elapsed:.1f}s")
    print(f"Done. Updated {len(parts)} file(s) in {parquet_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cast fullVisitorId, visitId to string (in-place)")
    ap.add_argument("--input-dir", default=PARQUET_DIR_DEFAULT, help="Parquet directory to update")
    args = ap.parse_args()
    main(parquet_dir=args.input_dir)
