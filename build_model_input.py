#!/usr/bin/env python3
"""
Build model input from cleaned parquet (numeric policy + categorical encoding).
Default: train_v2_parquet_logicapplied_cleaned -> train_v2_model_input/
Test: python3 build_model_input.py --input-dir test_v2_parquet_logicapplied_cleaned --output-dir test_v2_clean
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_DIR_DEFAULT = SCRIPT_DIR / "train_v2_parquet_logicapplied_cleaned"
OUT_DIR_DEFAULT = SCRIPT_DIR / "train_v2_model_input"
NUMERIC_POLICY_PATH = SCRIPT_DIR / "numeric_feature_policy.json"
CATEGORICAL_DICT_PATH = SCRIPT_DIR / "EDA code" / "categorical_dictionary.json"


def load_numeric_policy(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_categorical_dictionary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _safe_float(x):
    """Coerce to float; NaN for non-numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def apply_numeric_policy(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Apply numeric feature policy: drop, log1p, log1p_cap, derived features, drop_raw."""
    out = df.copy()
    features = {f["field"]: f for f in policy["features"]}

    # Drop leakage columns
    for spec in policy["features"]:
        if spec["action"] == "drop" and spec["field"] in out.columns:
            out = out.drop(columns=[spec["field"]])

    # log1p
    for spec in policy["features"]:
        if spec["action"] != "log1p" or spec["field"] not in out.columns:
            continue
        col = pd.to_numeric(out[spec["field"]], errors="coerce").fillna(0)
        out[spec["field"]] = np.log1p(np.maximum(col, 0))

    # log1p_cap: cap at cap_value (if set from training) or cap_quantile then log1p
    for spec in policy["features"]:
        if spec["action"] != "log1p_cap" or spec["field"] not in out.columns:
            continue
        col = pd.to_numeric(out[spec["field"]], errors="coerce").fillna(0)
        cap_val = spec.get("cap_value")
        if cap_val is not None and isinstance(cap_val, (int, float)):
            cap = float(cap_val)
        else:
            q = spec.get("cap_quantile", 0.99)
            cap = col.quantile(q)
        capped = np.minimum(col, cap)
        out[spec["field"]] = np.log1p(np.maximum(capped, 0))

    # Derived from date (YYYYMMDD)
    derived = policy.get("derived_features") or {}
    if "date" in derived and "date" in out.columns:
        date_col = pd.to_numeric(out["date"], errors="coerce").dropna().astype("int64")
        dates = pd.to_datetime(date_col.astype(str), format="%Y%m%d", errors="coerce")
        out["date_dow"] = dates.dt.dayofweek  # 0=Mon, 6=Sun
        out["date_month"] = dates.dt.month
        out["date_is_weekend"] = (dates.dt.dayofweek >= 5).astype(np.int64)
        out = out.drop(columns=["date"])

    # Derived from visitStartTime (unix seconds UTC)
    if "visitStartTime" in derived and "visitStartTime" in out.columns:
        ts = pd.to_numeric(out["visitStartTime"], errors="coerce")
        dt = pd.to_datetime(ts, unit="s", utc=True)
        out["vst_hour"] = dt.dt.hour
        out["vst_dow"] = dt.dt.dayofweek
        out["vst_is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int64)
        out = out.drop(columns=["visitStartTime"])

    return out


def apply_categorical_encoding(
    df: pd.DataFrame, cat_dict: dict
) -> pd.DataFrame:
    """Encode categorical columns using value_to_index; unseen -> others_index."""
    columns_spec = cat_dict.get("columns") or {}
    out = df.copy()

    for col_name, spec in columns_spec.items():
        if col_name not in out.columns:
            continue
        value_to_index = spec.get("value_to_index")
        others_index = spec.get("others_index", 1)
        if not value_to_index:
            continue
        # Map; treat NaN/None as string "None" or use __NONE__ (0)
        series = out[col_name].astype(str).replace("nan", "").replace("None", "")
        mapped = series.map(lambda v: value_to_index.get(v, value_to_index.get(str(v).strip(), others_index)))
        # Fallback: if still missing (e.g. new string), use others_index
        out[col_name] = mapped.fillna(others_index).astype(np.int64)

    return out


def build_model_input_chunk(
    df: pd.DataFrame,
    numeric_policy: dict,
    cat_dict: dict,
    categorical_columns: list,
) -> pd.DataFrame:
    """Apply numeric policy and categorical encoding; keep only columns that exist and are used."""
    # 1) Numeric policy (drops leakage, adds derived, log1p, etc.)
    df = apply_numeric_policy(df, numeric_policy)

    # 2) Categorical encoding (only for columns present in both df and dictionary)
    df = apply_categorical_encoding(df, cat_dict)

    # 3) Order columns: ids, label, numeric features, categorical features
    id_cols = [c for c in ["fullVisitorId", "visitId"] if c in df.columns]
    label_col = "total_revenue" if "total_revenue" in df.columns else None
    numeric_candidates = [
        "total_revenue", "total_product_viewed", "total_category_viewed",
        "totals_hits", "totals_pageviews", "totals_timeOnSite",
        "average_price_product_viewed", "visitNumber",
        "date_dow", "date_month", "date_is_weekend",
        "vst_hour", "vst_dow", "vst_is_weekend",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    if label_col and label_col in numeric_cols:
        numeric_cols = [label_col] + [c for c in numeric_cols if c != label_col]
    cat_cols = [c for c in categorical_columns if c in df.columns]
    rest = [c for c in df.columns if c not in id_cols + numeric_cols + cat_cols]
    col_order = id_cols + numeric_cols + cat_cols + rest
    col_order = [c for c in col_order if c in df.columns]
    # Deduplicate while preserving order
    seen = set()
    col_order = [c for c in col_order if not (c in seen or seen.add(c))]
    return df[col_order]


def main(
    parquet_dir: Path = PARQUET_DIR_DEFAULT,
    out_dir: Path = OUT_DIR_DEFAULT,
) -> None:
    parquet_dir = Path(parquet_dir)
    out_dir = Path(out_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {parquet_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_policy = load_numeric_policy(NUMERIC_POLICY_PATH)
    cat_dict = load_categorical_dictionary(CATEGORICAL_DICT_PATH)
    categorical_columns = list((cat_dict.get("columns") or {}).keys())

    parts = sorted(parquet_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    for path in parts:
        df = pd.read_parquet(path)
        out_df = build_model_input_chunk(df, numeric_policy, cat_dict, categorical_columns)
        out_path = out_dir / path.name
        out_df.to_parquet(out_path, index=False)
        print(f"  {path.name}: {len(out_df):,} rows -> {out_path}")

    print(f"Done. Model input in {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build model input (numeric + categorical)")
    ap.add_argument("--input-dir", default=None, help="Input cleaned parquet directory")
    ap.add_argument("--output-dir", default=None, help="Output model input directory")
    args = ap.parse_args()
    in_dir = Path(args.input_dir) if args.input_dir else PARQUET_DIR_DEFAULT
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR_DEFAULT
    main(parquet_dir=in_dir, out_dir=out_dir)
