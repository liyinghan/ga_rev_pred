#!/usr/bin/env python3
"""
Train MLP with Batch Normalization (PyTorch) for GA revenue prediction.
Uses all training data by default (train_v2_model_input), 90/10 time-split for val.
- Architecture: input -> Linear->BN->ReLU -> 256 -> BN -> ReLU -> 128 -> BN -> ReLU -> 64 -> Linear -> output.
- Applies to test_v2_clean, then evaluates validation + test (vs total_revenue).
- Writes: mlp_bn_test_predictions.csv, mlp_bn_evaluation_metrics.txt.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR_DEFAULT = SCRIPT_DIR / "train_v2_model_input"
TEST_DIR_DEFAULT = SCRIPT_DIR / "test_v2_clean"
OUTPUT_PREDICTIONS_DEFAULT = SCRIPT_DIR / "mlp_bn_test_predictions.csv"
OUTPUT_METRICS_DEFAULT = SCRIPT_DIR / "mlp_bn_evaluation_metrics.txt"

ID_COLS = ["fullVisitorId", "visitId"]
LABEL_COL = "total_revenue"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_parquet_dir(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No .parquet files in {path}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(ID_COLS) | {LABEL_COL}
    return [c for c in df.columns if c not in exclude]


def time_split(df: pd.DataFrame, train_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("visitId").reset_index(drop=True)
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


class MLPBN(nn.Module):
    """MLP with Batch Normalization after each hidden layer."""

    def __init__(self, in_dim: int, hidden: tuple = (256, 128, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.out(x).squeeze(-1)


def prepare_X_y(df: pd.DataFrame, feature_cols: list, fill_values: dict, scaler: StandardScaler, clip: float):
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c].fillna(fill_values.get(c, 0))
        else:
            X[c] = fill_values.get(c, 0)
    X = X[feature_cols]
    X_s = scaler.transform(X)
    X_s = np.clip(np.nan_to_num(X_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    return X_s.astype(np.float32)


def main(
    train_dir: Path = TRAIN_DIR_DEFAULT,
    test_dir: Path = TEST_DIR_DEFAULT,
    train_frac: float = 0.9,
    output_predictions: Path = OUTPUT_PREDICTIONS_DEFAULT,
    output_metrics: Path = OUTPUT_METRICS_DEFAULT,
    batch_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
    max_train: Optional[int] = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    clip = 10.0

    print("Loading training data...")
    train_full = load_parquet_dir(train_dir)
    feature_cols = get_feature_columns(train_full)
    print(f"  Total rows: {len(train_full):,}, features: {len(feature_cols)}")

    if max_train is not None:
        train_full = train_full.sort_values("visitId").reset_index(drop=True).head(max_train)
        print(f"  Subset: using first {len(train_full):,} rows (--max-train {max_train})")
    print("Time-based train/validation split (90% / 10%)...")
    train_df, val_df = time_split(train_full, train_frac=train_frac)
    print(f"  Train: {len(train_df):,}, Validation: {len(val_df):,}")

    fill_values = {}
    X_train = train_df[feature_cols].copy()
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    X_val = val_df[feature_cols].copy()
    y_val = val_df[LABEL_COL].values.astype(np.float32)
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
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip).astype(np.float32)
    X_val_s = np.clip(np.nan_to_num(X_val_s, nan=0.0, posinf=clip, neginf=-clip), -clip, clip).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_t = (torch.from_numpy(X_val_s), torch.from_numpy(y_val))

    in_dim = X_train_s.shape[1]
    model = MLPBN(in_dim, hidden=(256, 128, 64)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"Training MLP+BN on {DEVICE} (hidden 256, 128, 64)...")
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_t[0].to(DEVICE)).cpu().numpy()
        val_loss = mean_squared_error(y_val, val_pred)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch % 10 == 0) or epoch == 1:
            print(f"  Epoch {epoch}, val RMSE: {np.sqrt(val_loss):.6f}")
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_val_pred = model(val_t[0].to(DEVICE)).cpu().numpy()
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print("\nValidation metrics (MLP+BN):")
    print(f"  RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R2: {val_r2:.6f}")

    # Test
    print("\nLoading test data...")
    test_df = load_parquet_dir(test_dir)
    X_test_s = prepare_X_y(test_df, feature_cols, fill_values, scaler, clip)
    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test_s).to(DEVICE)).cpu().numpy()
    test_pred = np.maximum(test_pred, 0.0)
    out = test_df[["fullVisitorId"]].copy()
    out["PredictedLogRevenue"] = test_pred
    out = out.groupby("fullVisitorId", as_index=False)["PredictedLogRevenue"].sum()
    out.to_csv(output_predictions, index=False)
    print(f"Test predictions saved to {output_predictions} ({len(out):,} visitors)")

    # Test metrics vs total_revenue in test data
    test_actual = test_df.groupby("fullVisitorId", as_index=False)[LABEL_COL].sum()
    test_actual = test_actual.rename(columns={LABEL_COL: "actual"})
    compare = out.merge(test_actual, on="fullVisitorId", how="inner")
    y_test_true = compare["actual"].values
    y_test_pred = compare["PredictedLogRevenue"].values
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)

    metrics_lines = [
        "=== MLP+BN (Batch Normalization) evaluation ===",
        "",
        "Validation (time-split 90/10):",
        f"  RMSE (log scale): {val_rmse:.6f}",
        f"  MAE:              {val_mae:.6f}",
        f"  R2:               {val_r2:.6f}",
        "",
        "Test (vs total_revenue in test data, aggregated by fullVisitorId):",
        f"  RMSE: {test_rmse:.6f}",
        f"  MAE:  {test_mae:.6f}",
        f"  R2:   {test_r2:.6f}",
        "",
        f"Predictions: {output_predictions}",
    ]
    Path(output_metrics).write_text("\n".join(metrics_lines))
    print(f"Metrics saved to {output_metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=TRAIN_DIR_DEFAULT, type=Path)
    ap.add_argument("--test-dir", default=TEST_DIR_DEFAULT, type=Path)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--output-predictions", default=OUTPUT_PREDICTIONS_DEFAULT, type=Path)
    ap.add_argument("--output-metrics", default=OUTPUT_METRICS_DEFAULT, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None, help="Use first N rows for quick run")
    args = ap.parse_args()
    main(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        train_frac=args.train_frac,
        output_predictions=args.output_predictions,
        output_metrics=args.output_metrics,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        max_train=args.max_train,
    )
