#!/usr/bin/env python3
"""
Train MLP for GA customer revenue prediction.
Uses all training data (train_v2_model_input), 90/10 time-split for validation.
- MLP: input -> 256 -> 128 -> 64 -> output (log-revenue)
- Applies to test_v2_clean, then evaluates validation + test (vs total_revenue).
- Writes: mlp_test_predictions.csv, mlp_evaluation_metrics.txt.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="divide by zero|overflow|invalid value", module="sklearn")
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR_DEFAULT = SCRIPT_DIR / "train_v2_model_input"
TEST_DIR_DEFAULT = SCRIPT_DIR / "test_v2_clean"
OUTPUT_PREDICTIONS_DEFAULT = SCRIPT_DIR / "mlp_test_predictions.csv"
OUTPUT_METRICS_DEFAULT = SCRIPT_DIR / "mlp_evaluation_metrics.txt"

ID_COLS = ["fullVisitorId", "visitId"]
LABEL_COL = "total_revenue"


def load_parquet_dir(path: Path) -> pd.DataFrame:
    """Load and concatenate all parquet files in directory."""
    parts = sorted(path.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files in {path}")
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Columns to use as features: all except IDs and label."""
    exclude = set(ID_COLS) | {LABEL_COL}
    return [c for c in df.columns if c not in exclude]


def time_split(df: pd.DataFrame, train_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time: sort by visitId, first train_frac for train, rest for validation.
    Validation is strictly later in time than training.
    """
    df = df.sort_values("visitId").reset_index(drop=True)
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def main(
    train_dir: Path = TRAIN_DIR_DEFAULT,
    test_dir: Path = TEST_DIR_DEFAULT,
    train_frac: float = 0.9,
    output_predictions: Path = OUTPUT_PREDICTIONS_DEFAULT,
    output_metrics: Path = OUTPUT_METRICS_DEFAULT,
    random_state: int = 42,
) -> None:
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    print("Loading training data...")
    train_full = load_parquet_dir(train_dir)
    feature_cols = get_feature_columns(train_full)
    print(f"  Total rows: {len(train_full):,}, features: {len(feature_cols)}")

    print("Time-based train/validation split (90% / 10%)...")
    train_df, val_df = time_split(train_full, train_frac=train_frac)
    print(f"  Train: {len(train_df):,}, Validation: {len(val_df):,}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[LABEL_COL].values
    X_val = val_df[feature_cols].copy()
    y_val = val_df[LABEL_COL].values

    # Fill NaN: use train medians for float, 0 for int/categorical
    fill_values = {}
    for c in feature_cols:
        if X_train[c].dtype in ("float64", "float32"):
            fill_values[c] = X_train[c].median()
            X_train[c] = X_train[c].fillna(fill_values[c])
            X_val[c] = X_val[c].fillna(fill_values[c])
        else:
            fill_values[c] = 0
            X_train[c] = X_train[c].fillna(0)
            X_val[c] = X_val[c].fillna(0)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    # Clip to avoid overflow in MLP (log-scale targets are small; keep features bounded)
    clip = 10.0
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    X_val_s = np.clip(np.nan_to_num(X_val_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

    print("Training MLP (hidden 256, 128, 64)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True,
    )
    mlp.fit(X_train_s, y_train)

    # Validation evaluation
    y_val_pred = mlp.predict(X_val_s)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print("\nValidation metrics:")
    print(f"  RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R2: {val_r2:.6f}")

    # Test
    print("\nLoading test data...")
    test_df = load_parquet_dir(test_dir)
    X_test = pd.DataFrame(index=test_df.index)
    for c in feature_cols:
        if c in test_df.columns:
            X_test[c] = test_df[c].fillna(fill_values.get(c, 0))
        else:
            X_test[c] = fill_values.get(c, 0)
    X_test = X_test[feature_cols]

    X_test_s = scaler.transform(X_test)
    X_test_s = np.clip(np.nan_to_num(X_test_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    test_pred = mlp.predict(X_test_s)

    # Submission format: fullVisitorId, PredictedLogRevenue
    out = test_df[["fullVisitorId"]].copy()
    out["PredictedLogRevenue"] = np.maximum(test_pred, 0.0)
    # Competition often aggregates by fullVisitorId (sum of log revenue per visitor)
    out = out.groupby("fullVisitorId", as_index=False)["PredictedLogRevenue"].sum()
    out.to_csv(output_predictions, index=False)
    print(f"Test predictions saved to {output_predictions} ({len(out):,} visitors)")

    # Test evaluation (vs total_revenue in test data, aggregated by fullVisitorId)
    test_actual = test_df.groupby("fullVisitorId", as_index=False)[LABEL_COL].sum()
    test_actual = test_actual.rename(columns={LABEL_COL: "actual"})
    compare = out.merge(test_actual, on="fullVisitorId", how="inner")
    y_test_true = compare["actual"].values
    y_test_pred = compare["PredictedLogRevenue"].values
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)
    metrics_lines = [
        "=== MLP evaluation (validation set, time-split 90/10) ===",
        f"Validation RMSE (log scale): {val_rmse:.6f}",
        f"Validation MAE:  {val_mae:.6f}",
        f"Validation R2:  {val_r2:.6f}",
        "",
        "=== Test (predictions vs total_revenue in test data, per fullVisitorId) ===",
        f"Test RMSE: {test_rmse:.6f}",
        f"Test MAE:  {test_mae:.6f}",
        f"Test R2:   {test_r2:.6f}",
        "",
        f"Predictions written to: {output_predictions}",
    ]
    print("\nTest metrics (vs test total_revenue):")
    print(f"  RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R2: {test_r2:.6f}")

    Path(output_metrics).write_text("\n".join(metrics_lines))
    print(f"Metrics saved to {output_metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train MLP and predict on test")
    ap.add_argument("--train-dir", default=TRAIN_DIR_DEFAULT, type=Path)
    ap.add_argument("--test-dir", default=TEST_DIR_DEFAULT, type=Path)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--output-predictions", default=OUTPUT_PREDICTIONS_DEFAULT, type=Path)
    ap.add_argument("--output-metrics", default=OUTPUT_METRICS_DEFAULT, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_frac=args.train_frac,
        output_predictions=args.output_predictions,
        output_metrics=args.output_metrics,
        random_state=args.seed,
    )
