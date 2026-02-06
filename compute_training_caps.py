#!/usr/bin/env python3
"""
Compute training 0.99 quantiles for log1p_cap features and write them into
numeric_feature_policy.json so the same caps are used for train and test.

Reads: train_v2_parquet_logicapplied_cleaned/*.parquet, numeric_feature_policy.json
Writes: numeric_feature_policy.json (updates cap_value for each log1p_cap feature)

Run once after cleaning training data, before building train/test model input:
  python3 compute_training_caps.py
"""

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_DIR = SCRIPT_DIR / "train_v2_parquet_logicapplied_cleaned"
POLICY_PATH = SCRIPT_DIR / "numeric_feature_policy.json"


def main():
    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Not found: {PARQUET_DIR}")
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Not found: {POLICY_PATH}")

    policy = json.loads(POLICY_PATH.read_text())
    log1p_cap_specs = [f for f in policy["features"] if f.get("action") == "log1p_cap"]
    if not log1p_cap_specs:
        print("No log1p_cap features in policy. Nothing to do.")
        return

    fields = [s["field"] for s in log1p_cap_specs]
    quantiles = {s["field"]: s.get("cap_quantile", 0.99) for s in log1p_cap_specs}

    parts = sorted(PARQUET_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet in {PARQUET_DIR}")

    # Compute global quantiles over all training data (one column at a time to save memory)
    caps = {}
    for field in fields:
        values = []
        for path in parts:
            part_df = pd.read_parquet(path, columns=[field])
            if field not in part_df.columns:
                continue
            col = pd.to_numeric(part_df[field], errors="coerce").fillna(0)
            values.append(col)
        if not values:
            print(f"  {field}: column not found in any part, skipping")
            continue
        concat = pd.concat(values, ignore_index=True)
        q = quantiles[field]
        cap = float(concat.quantile(q))
        caps[field] = cap
        print(f"  {field}: cap_value = {cap:.6g} (quantile {q})")

    # Update policy in place
    for f in policy["features"]:
        if f.get("action") == "log1p_cap" and f["field"] in caps:
            f["cap_value"] = caps[f["field"]]

    POLICY_PATH.write_text(json.dumps(policy, indent=2))
    print(f"Updated {POLICY_PATH}")


if __name__ == "__main__":
    main()
