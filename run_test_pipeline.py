#!/usr/bin/env python3
"""
Run the full pipeline on test_v2.csv and write model input to test_v2_clean/.

Steps:
  1. test_v2.csv -> test_v2_rawparquet/
  2. test_v2_rawparquet/ -> test_v2_parquet_logicapplied/
  3. test_v2_parquet_logicapplied/ -> test_v2_parquet_logicapplied_cleaned/
  4. cast_totals_to_numeric on test_v2_parquet_logicapplied_cleaned/
  5. cast_ids_to_string on test_v2_parquet_logicapplied_cleaned/
  6. build_model_input: test_v2_parquet_logicapplied_cleaned/ -> test_v2_clean/

Uses same numeric_feature_policy.json and categorical_dictionary.json (training caps and vocab).
Run from project root:
  python3 run_test_pipeline.py                  # full test_v2.csv
  python3 run_test_pipeline.py --csv test_v2_small.csv   # quick run (small sample)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH_DEFAULT = SCRIPT_DIR / "test_v2.csv"
RAW_DIR = SCRIPT_DIR / "test_v2_rawparquet"
LOGICAPPLIED_DIR = SCRIPT_DIR / "test_v2_parquet_logicapplied"
CLEANED_DIR = SCRIPT_DIR / "test_v2_parquet_logicapplied_cleaned"
OUT_DIR = SCRIPT_DIR / "test_v2_clean"


def run(cmd: list[str], step_name: str) -> None:
    print(f"\n{'='*60}\n{step_name}\n{'='*60}", flush=True)
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        print(f"Failed: {cmd}", file=sys.stderr)
        sys.exit(r.returncode)


def main(csv_path: Path = CSV_PATH_DEFAULT) -> None:
    if not csv_path.exists():
        print(f"Not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 1. CSV -> raw parquet
    run(
        [
            sys.executable,
            "csv_to_parquet_train_v2.py",
            "--csv", str(csv_path),
            "--output-dir", str(RAW_DIR),
        ],
        "Step 1: test_v2.csv -> test_v2_rawparquet/",
    )

    # 2. Flatten
    run(
        [
            sys.executable,
            "flatten_all_parquet.py",
            "--input-dir", str(RAW_DIR),
            "--output-dir", str(LOGICAPPLIED_DIR),
        ],
        "Step 2: test_v2_rawparquet/ -> test_v2_parquet_logicapplied/",
    )

    # 3. Drop low-variance columns
    run(
        [
            sys.executable,
            "drop_low_variance_columns.py",
            "--input-dir", str(LOGICAPPLIED_DIR),
            "--output-dir", str(CLEANED_DIR),
        ],
        "Step 3: test_v2_parquet_logicapplied/ -> test_v2_parquet_logicapplied_cleaned/",
    )

    # 4. Cast totals to numeric (in-place on cleaned)
    run(
        [
            sys.executable,
            "cast_totals_to_numeric.py",
            "--input-dir", str(CLEANED_DIR),
        ],
        "Step 4: cast totals_* to numeric (test_v2_parquet_logicapplied_cleaned/)",
    )

    # 5. Cast IDs to string (in-place on cleaned)
    run(
        [
            sys.executable,
            "cast_ids_to_string.py",
            "--input-dir", str(CLEANED_DIR),
        ],
        "Step 5: cast fullVisitorId, visitId to string",
    )

    # 6. Build model input -> test_v2_clean/
    run(
        [
            sys.executable,
            "build_model_input.py",
            "--input-dir", str(CLEANED_DIR),
            "--output-dir", str(OUT_DIR),
        ],
        "Step 6: build model input -> test_v2_clean/",
    )

    print(f"\n{'='*60}\nDone. Test model input in {OUT_DIR}/\n{'='*60}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run test pipeline: test CSV -> test_v2_clean/")
    ap.add_argument("--csv", default=str(CSV_PATH_DEFAULT), help="Input test CSV (default: test_v2.csv)")
    args = ap.parse_args()
    main(csv_path=Path(args.csv))
