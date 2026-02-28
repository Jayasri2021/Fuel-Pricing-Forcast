from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/modeling_weekly.parquet")
REPORTS_DIR = Path("reports")
PRED_PATH = REPORTS_DIR / "backtest_predictions.csv"
METRICS_PATH = REPORTS_DIR / "backtest_metrics_summary.csv"

# Backtest settings
TRAIN_WEEKS = 104          # 2 years
HORIZON_WEEKS = 4          # predict next 4 weeks
STEP_WEEKS = 4             # move forward by 4 weeks each iteration


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    # avoid division by zero
    denom = np.where(denom == 0, 1e-8, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def evaluate_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    s = smape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"mae": mae, "rmse": rmse, "smape": s, "bias": bias}


def fit_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    # Start with a sensible baseline; we can tune later
    model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)

    # Features: everything except date + target
    feature_cols = [c for c in df.columns if c not in ["date", "y_gas_price"]]

    predictions_rows = []
    metrics_rows = []

    # Rolling cutoffs: ensure we have train window + horizon available
    max_start = len(df) - (TRAIN_WEEKS + HORIZON_WEEKS)
    if max_start <= 0:
        raise ValueError("Not enough rows for the chosen train window + horizon.")

    # We slide forward by STEP_WEEKS
    for start_idx in range(0, max_start + 1, STEP_WEEKS):
        train_start = start_idx
        train_end = start_idx + TRAIN_WEEKS  # exclusive
        test_start = train_end
        test_end = train_end + HORIZON_WEEKS  # exclusive

        train = df.iloc[train_start:train_end]
        test = df.iloc[test_start:test_end]

        # sanity
        if len(test) < HORIZON_WEEKS:
            break

        X_train = train[feature_cols]
        y_train = train["y_gas_price"].astype(float)

        X_test = test[feature_cols]
        y_test = test["y_gas_price"].astype(float).to_numpy()

        # -----------------------------
        # Baseline predictions
        # -----------------------------
        naive_pred = test["y_gas_price_lag_1"].astype(float).to_numpy()
        seasonal_pred = test["y_gas_price_lag_52"].astype(float).to_numpy()

        # -----------------------------
        # XGBoost
        # -----------------------------
        model = fit_xgb(X_train, y_train)
        xgb_pred = model.predict(X_test).astype(float)

        # -----------------------------
        # Save per-row predictions
        # -----------------------------
        for i in range(len(test)):
            predictions_rows.append({
                "cutoff_end_date": train["date"].iloc[-1],
                "date": test["date"].iloc[i],
                "y_true": y_test[i],
                "naive_lag1": naive_pred[i],
                "seasonal_lag52": seasonal_pred[i],
                "xgb": xgb_pred[i],
            })

        # -----------------------------
        # Metrics per window (block)
        # -----------------------------
        window_id = f"{train['date'].iloc[0].date()}__{train['date'].iloc[-1].date()}"
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "Naive (lag_1)",
            **evaluate_block(y_test, naive_pred),
        })
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "Seasonal Naive (lag_52)",
            **evaluate_block(y_test, seasonal_pred),
        })
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "XGBoost",
            **evaluate_block(y_test, xgb_pred),
        })

    # Write outputs
    pred_df = pd.DataFrame(predictions_rows)
    pred_df.to_csv(PRED_PATH, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)

    # Print overall summary across all windows
    print(f"Saved predictions: {PRED_PATH} ({len(pred_df)} rows)")
    print(f"Saved metrics:     {METRICS_PATH} ({len(metrics_df)} rows)")

    print("\nOverall metrics (mean across windows):")
    summary = (
        metrics_df.groupby("model")[["mae", "rmse", "smape", "bias"]]
        .mean()
        .sort_values("mae")
    )
    print(summary.to_string())

    print("\nTip: The backtest truth is the *distribution* across windows, not a single split.")


if __name__ == "__main__":
    main()