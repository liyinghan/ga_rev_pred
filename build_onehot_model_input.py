#!/usr/bin/env python3
"""
Build one-hot encoded model input from existing integer-encoded model input.
- Reads: train_v2_model_input/, test_v2_clean/
- Categorical columns (from categorical_dictionary.json) are converted from integer id to one-hot.
- Numeric and id/label columns unchanged.
- Writes: train_v2_model_input_onehot/, test_v2_clean_onehot/

Then you can run: python3 train_mlp.py --train-dir train_v2_model_input_onehot --test-dir test_v2_clean_onehot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CAT_DICT_PATH = SCRIPT_DIR / "EDA code" / "categorical_dictionary.json"
TRAIN_IN_DEFAULT = SCRIPT_DIR / "train_v2_model_input"
TEST_IN_DEFAULT = SCRIPT_DIR / "test_v2_clean"
TRAIN_OUT_DEFAULT = SCRIPT_DIR / "train_v2_model_input_onehot"
TEST_OUT_DEFAULT = SCRIPT_DIR / "test_v2_clean_onehot"

ID_COLS = ["fullVisitorId", "visitId"]
LABEL_COL = "total_revenue"


def load_cat_dict(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def one_hot_from_index(series: pd.Series, vocab_size: int, prefix: str) -> pd.DataFrame:
    """
    series: integer indices (0 .. vocab_size-1). Clipped to valid range.
    """
    idx = series.fillna(0).astype(np.int64).clip(0, max(0, vocab_size - 1)).values
    eye = np.eye(vocab_size, dtype=np.int8)
    arr = eye[idx]
    col_names = [f"{prefix}_{i}" for i in range(vocab_size)]
    return pd.DataFrame(arr, columns=col_names, index=series.index)


def process_df_to_onehot(
    df: pd.DataFrame,
    cat_cols_with_vocab: list,
    numeric_cols: list,
    id_cols: list,
    label_col: str,
) -> pd.DataFrame:
    """Convert one DataFrame: categorical columns -> one-hot; keep numeric and ids."""
    parts = []
    # IDs
    for c in id_cols:
        if c in df.columns:
            parts.append(df[[c]])
    # Label
    if label_col and label_col in df.columns:
        parts.append(df[[label_col]])
    # Numeric (excluding label if already added)
    for c in numeric_cols:
        if c in df.columns and c != label_col:
            parts.append(df[[c]])
    # One-hot for each categorical
    for col_name, vocab_size in cat_cols_with_vocab:
        if col_name in df.columns:
            oh = one_hot_from_index(df[col_name], vocab_size, col_name)
            parts.append(oh)
        else:
            # Missing column (e.g. in test): fill with zeros
            zero_cols = [f"{col_name}_{i}" for i in range(vocab_size)]
            parts.append(pd.DataFrame(0, index=df.index, columns=zero_cols, dtype=np.int8))
    return pd.concat(parts, axis=1)


def main(
    train_in: Path = TRAIN_IN_DEFAULT,
    test_in: Path = TEST_IN_DEFAULT,
    train_out: Path = TRAIN_OUT_DEFAULT,
    test_out: Path = TEST_OUT_DEFAULT,
) -> None:
    train_in = Path(train_in)
    test_in = Path(test_in)
    train_out = Path(train_out)
    test_out = Path(test_out)
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    cat_dict = load_cat_dict(CAT_DICT_PATH)
    cat_columns = list(cat_dict.get("columns", {}).keys())
    cat_cols_with_vocab = [
        (c, cat_dict["columns"][c]["vocab_size"])
        for c in cat_columns
    ]

    # Load one train part to get column layout
    train_parts = sorted(train_in.glob("*.parquet"))
    if not train_parts:
        raise FileNotFoundError(f"No parquet in {train_in}")
    sample = pd.read_parquet(train_parts[0])
    feature_cols = [c for c in sample.columns if c not in ID_COLS and c != LABEL_COL]
    numeric_cols = [c for c in feature_cols if c not in cat_columns]
    id_cols = [c for c in ID_COLS if c in sample.columns]
    label_col = LABEL_COL if LABEL_COL in sample.columns else None

    print("Train → one-hot")
    for path in train_parts:
        df = pd.read_parquet(path)
        out = process_df_to_onehot(df, cat_cols_with_vocab, numeric_cols, id_cols, label_col)
        out_path = train_out / path.name
        out.to_parquet(out_path, index=False)
        print(f"  {path.name} → {out_path.name} ({out.shape[1]} columns)")

    print("Test → one-hot")
    test_parts = sorted(test_in.glob("*.parquet"))
    if not test_parts:
        raise FileNotFoundError(f"No parquet in {test_in}")
    for path in test_parts:
        df = pd.read_parquet(path)
        out = process_df_to_onehot(df, cat_cols_with_vocab, numeric_cols, id_cols, label_col)
        out_path = test_out / path.name
        out.to_parquet(out_path, index=False)
        print(f"  {path.name} → {out_path.name} ({out.shape[1]} columns)")

    print(f"\nDone. One-hot train: {train_out}/")
    print(f"      One-hot test: {test_out}/")
    print("\nRun MLP on one-hot data (from project root):")
    print("  python3 train_mlp.py --train-dir train_v2_model_input_onehot --test-dir test_v2_clean_onehot --output-predictions mlp_onehot_test_predictions.csv --output-metrics mlp_onehot_evaluation_metrics.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build one-hot encoded train and test model input")
    ap.add_argument("--train-in", default=TRAIN_IN_DEFAULT, type=Path)
    ap.add_argument("--test-in", default=TEST_IN_DEFAULT, type=Path)
    ap.add_argument("--train-out", default=TRAIN_OUT_DEFAULT, type=Path)
    ap.add_argument("--test-out", default=TEST_OUT_DEFAULT, type=Path)
    args = ap.parse_args()
    main(
        train_in=args.train_in,
        test_in=args.test_in,
        train_out=args.train_out,
        test_out=args.test_out,
    )
