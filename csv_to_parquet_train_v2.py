#!/usr/bin/env python3
"""
Convert a GA CSV to Parquet files (chunked).
Default: train_v2.csv -> train_v2_rawparquet/
Test: python csv_to_parquet_train_v2.py --csv test_v2.csv --output-dir test_v2_rawparquet
"""

import argparse
import os
import pandas as pd

CHUNK_ROWS = 500_000  # rows per parquet file


def main(csv_path: str = "train_v2.csv", out_dir: str = "train_v2_rawparquet") -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Reading {csv_path} and writing Parquet to {out_dir}/ ...")
    part = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS, low_memory=False):
        out_path = os.path.join(out_dir, f"part-{part:05d}.parquet")
        chunk.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(chunk):,} rows)")
        part += 1
    print(f"Done. Total {part} Parquet file(s) in {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CSV to Parquet (chunked)")
    p.add_argument("--csv", default="train_v2.csv", help="Input CSV path")
    p.add_argument("--output-dir", default="train_v2_rawparquet", help="Output parquet directory")
    args = p.parse_args()
    main(csv_path=args.csv, out_dir=args.output_dir)
