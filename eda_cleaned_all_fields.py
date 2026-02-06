#!/usr/bin/env python3
"""
EDA on ALL fields for ALL parquet files in train_v2_parquet_logicapplied_cleaned.
Loads every part, concatenates (relaxed schema), then reports per-column:
  - dtype, count, null_count, n_unique
  - Zero/low variance flag (unique <= 1)
  - For numeric: describe()
  - For string: top 15 value counts (and null sentinel treatment)

Run from project root:
  python "EDA code/eda_cleaned_all_fields.py"
  python "EDA code/eda_cleaned_all_fields.py" --sample 500000   # optional sampling
"""

import argparse
from pathlib import Path
from typing import Optional

import polars as pl

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PARQUET_DIR = PROJECT_ROOT / "train_v2_parquet_logicapplied_cleaned"
REPORT_PATH = SCRIPT_DIR / "eda_cleaned_all_fields_report.txt"

NULL_SENTINELS = (
    "(not set)",
    "not available in demo dataset",
    "(not provided)",
    "unknown.unknown",
)


def parse_args():
    p = argparse.ArgumentParser(description="EDA all fields — train_v2_parquet_logicapplied_cleaned")
    p.add_argument("--sample", type=int, default=None, help="If set, sample N rows for speed (default: use all)")
    p.add_argument("--no_file", action="store_true", help="Do not write report file, only stdout")
    return p.parse_args()


def load_all_cleaned(sample_n: Optional[int]) -> tuple:
    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Not found: {PARQUET_DIR}")
    parts = sorted(PARQUET_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet in {PARQUET_DIR}")

    dfs = [pl.read_parquet(p) for p in parts]
    df = pl.concat(dfs, how="diagonal_relaxed", rechunk=True)
    n = len(df)
    if sample_n is not None and n > sample_n:
        df = df.sample(n=sample_n, seed=42)
        n = len(df)
    return df, n


def run_eda(df: pl.DataFrame, total_rows: int, report_lines: list) -> None:
    report_lines.append("=" * 70)
    report_lines.append("EDA: ALL FIELDS — train_v2_parquet_logicapplied_cleaned")
    report_lines.append("=" * 70)
    report_lines.append(f"Total rows: {total_rows:,}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append("")

    # Sort columns for consistent output
    cols = sorted(df.columns)

    for col in cols:
        s = df[col]
        dtype = str(s.dtype)
        count = len(s)
        null_count = s.null_count()
        n_unique = s.n_unique()

        report_lines.append("-" * 70)
        report_lines.append(f"  {col}")
        report_lines.append(f"    dtype: {dtype}")
        report_lines.append(f"    non-null: {count - null_count:,}  |  null: {null_count:,}")
        report_lines.append(f"    unique: {n_unique:,}")

        if n_unique <= 1:
            report_lines.append(f"    >>> ZERO/LOW VARIANCE (unique={n_unique}) — exclude from modeling <<<")

        is_numeric = dtype in ("Int64", "Float64", "Int32", "Float32") or "Int" in dtype or "Float" in dtype
        if is_numeric:
            try:
                sub = df.select(pl.col(col))
                desc = sub.describe()
                report_lines.append("    describe:")
                report_lines.append(desc.to_pandas().to_string(index=True))
            except Exception as e:
                report_lines.append(f"    describe error: {e}")
        elif dtype == "Utf8" or "String" in dtype:
            # Top 15 value counts (treat null sentinels as valid categories for count)
            vc = s.value_counts().sort("count", descending=True).head(15)
            report_lines.append("    value_counts (top 15):")
            for row in vc.iter_rows():
                report_lines.append(f"      {row[0]!r}: {row[1]:,}")
        report_lines.append("")

    return


def main() -> None:
    args = parse_args()
    report_lines: list[str] = []

    print("Loading parquet from", PARQUET_DIR, "...", flush=True)
    df, total_rows = load_all_cleaned(args.sample)
    if args.sample:
        print(f"Sampled {total_rows:,} rows", flush=True)
    else:
        print(f"Loaded {total_rows:,} rows, {len(df.columns)} columns", flush=True)

    run_eda(df, total_rows, report_lines)

    report_text = "\n".join(report_lines)
    print(report_text)

    if not args.no_file:
        REPORT_PATH.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
