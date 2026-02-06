#!/usr/bin/env python3
"""
Drop zero/low-variance columns from parquet files.
Default: train_v2_parquet_logicapplied -> train_v2_parquet_logicapplied_cleaned
"""

import argparse
import os
import time

import pandas as pd

PARQUET_DIR_DEFAULT = "train_v2_parquet_logicapplied"
OUT_DIR_DEFAULT = "train_v2_parquet_logicapplied_cleaned"

# Zero/low variance columns to drop (unique=1 per EDA)
COLUMNS_TO_DROP = [
    "device_browserSize",
    "device_browserVersion",
    "device_flashVersion",
    "device_language",
    "device_mobileDeviceBranding",
    "device_mobileDeviceInfo",
    "device_mobileDeviceMarketingName",
    "device_mobileDeviceModel",
    "device_mobileInputSelector",
    "device_operatingSystemVersion",
    "device_screenColors",
    "device_screenResolution",
    "geoNetwork_cityId",
    "geoNetwork_latitude",
    "geoNetwork_longitude",
    "geoNetwork_networkLocation",
    "trafficSource_adwordsClickInfo_criteriaParameters",
]


def main(parquet_dir: str = PARQUET_DIR_DEFAULT, out_dir: str = OUT_DIR_DEFAULT) -> None:
    os.makedirs(out_dir, exist_ok=True)

    parts = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    n_files = len(parts)
    print(f"Found {n_files} parquet file(s) in {parquet_dir}/", flush=True)
    print(f"Dropping columns: {COLUMNS_TO_DROP}", flush=True)

    for file_idx, filename in enumerate(parts, start=1):
        in_path = os.path.join(parquet_dir, filename)
        out_path = os.path.join(out_dir, filename)
        t0 = time.perf_counter()
        print(f"\n[{file_idx}/{n_files}] Processing {filename} ...", flush=True)
        df = pd.read_parquet(in_path)
        before_cols = len(df.columns)
        # Drop only columns that exist
        to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        df.to_parquet(out_path, index=False)
        elapsed = time.perf_counter() - t0
        print(
            f"    -> dropped {len(to_drop)} cols ({before_cols} -> {len(df.columns)}), "
            f"saved {len(df):,} rows to {out_path} ({elapsed:.1f}s)",
            flush=True,
        )

    print(f"\nDone. Output in {out_dir}/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Drop low-variance columns")
    ap.add_argument("--input-dir", default=PARQUET_DIR_DEFAULT, help="Input parquet directory")
    ap.add_argument("--output-dir", default=OUT_DIR_DEFAULT, help="Output parquet directory")
    args = ap.parse_args()
    main(parquet_dir=args.input_dir, out_dir=args.output_dir)
