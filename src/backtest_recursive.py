from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import uuid
from src.db import get_engine, schema_name
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/modeling_weekly.parquet")
REPORTS_DIR = Path("reports")

PRED_PATH = REPORTS_DIR / "backtest_recursive_predictions.csv"
METRICS_PATH = REPORTS_DIR / "backtest_recursive_metrics_summary.csv"

# Backtest settings
TRAIN_WEEKS = 104
HORIZON_WEEKS = 4
STEP_WEEKS = 4

# These must match what you used in features.py
Y_LAGS = [1, 4, 8, 12, 26, 52]
Y_WINDOWS = [4, 8, 12, 26]
Y_SLOPE_WINDOW = 14


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1e-8, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def evaluate_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    s = smape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"mae": mae, "rmse": rmse, "smape": s, "bias": bias}


def fit_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
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


def _rolling_std(arr: np.ndarray) -> float:
    # match pandas default ddof=1 behavior
    if len(arr) < 2:
        return np.nan
    return float(np.std(arr, ddof=1))


def compute_y_dependent_features(history_y: list[float]) -> dict[str, float]:
    """
    Compute leakage-safe y-dependent features for the NEXT step,
    using history up to the previous time step.

    history_y[-1] is the last known/forecast value (t-1 relative to next step).
    """
    feats: dict[str, float] = {}

    # lags
    for l in Y_LAGS:
        feats[f"y_gas_price_lag_{l}"] = float(history_y[-l])

    # rolling stats on past values (equivalent to shift(1) then rolling)
    for w in Y_WINDOWS:
        window_vals = np.array(history_y[-w:], dtype=float)
        feats[f"y_gas_price_roll_mean_{w}"] = float(np.mean(window_vals))
        feats[f"y_gas_price_roll_median_{w}"] = float(np.median(window_vals))
        feats[f"y_gas_price_roll_std_{w}"] = _rolling_std(window_vals)

    # rolling slope (past-only)
    slope_vals = np.array(history_y[-Y_SLOPE_WINDOW:], dtype=float)
    x = np.arange(len(slope_vals), dtype=float)
    slope = float(np.polyfit(x, slope_vals, 1)[0])
    feats[f"y_gas_price_roll_slope_{Y_SLOPE_WINDOW}"] = slope

    return feats


def recursive_forecast_block(
    model: xgb.XGBRegressor,
    train_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - naive_recursive preds
      - seasonal_recursive preds
      - xgb_recursive preds
    """
    # Build history from TRAIN ACTUALS only (what you truly know at cutoff)
    history_y = train_df["y_gas_price"].astype(float).tolist()

    # Naive recursive: repeat last value forward
    naive_preds = []
    seasonal_preds = []
    xgb_preds = []

    # seasonal recursive needs history at least 52
    # (train window 104 ensures this)
    for step in range(len(horizon_df)):
        # baselines (recursive)
        naive_next = float(history_y[-1])
        seasonal_next = float(history_y[-52])

        naive_preds.append(naive_next)
        seasonal_preds.append(seasonal_next)

        # Build X row: start from horizon row's existing exogenous/time features,
        # then overwrite y-dependent features computed from history (actual + preds so far).
        row = horizon_df.iloc[step].copy()

        y_feats = compute_y_dependent_features(history_y)
        for k, v in y_feats.items():
            row[k] = v

        X_row = row[feature_cols].to_frame().T.apply(pd.to_numeric, errors="coerce")
        xgb_next = float(model.predict(X_row)[0])
        xgb_preds.append(xgb_next)

        # Update history for next step (use XGB forecast as the "known" value)
        history_y.append(xgb_next)

    return (
        np.array(naive_preds, dtype=float),
        np.array(seasonal_preds, dtype=float),
        np.array(xgb_preds, dtype=float),
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["date", "y_gas_price"]]

    predictions_rows = []
    metrics_rows = []

    max_start = len(df) - (TRAIN_WEEKS + HORIZON_WEEKS)
    if max_start <= 0:
        raise ValueError("Not enough rows for the chosen train window + horizon.")

    for start_idx in range(0, max_start + 1, STEP_WEEKS):
        train_start = start_idx
        train_end = start_idx + TRAIN_WEEKS
        test_start = train_end
        test_end = train_end + HORIZON_WEEKS

        train = df.iloc[train_start:train_end]
        test = df.iloc[test_start:test_end]

        if len(test) < HORIZON_WEEKS:
            break

        X_train = train[feature_cols]
        y_train = train["y_gas_price"].astype(float)

        model = fit_xgb(X_train, y_train)

        y_true = test["y_gas_price"].astype(float).to_numpy()

        naive_pred, seasonal_pred, xgb_pred = recursive_forecast_block(
            model=model,
            train_df=train,
            horizon_df=test,
            feature_cols=feature_cols,
        )

        cutoff_end = train["date"].iloc[-1]
        window_id = f"{train['date'].iloc[0].date()}__{train['date'].iloc[-1].date()}"

        # per-row predictions
        for i in range(len(test)):
            predictions_rows.append({
                "window_id": window_id,
                "cutoff_end_date": cutoff_end,
                "horizon_step": i + 1,  # 1..4
                "date": test["date"].iloc[i],
                "y_true": y_true[i],
                "naive_recursive": naive_pred[i],
                "seasonal_recursive": seasonal_pred[i],
                "xgb_recursive": xgb_pred[i],
            })

        # block metrics (overall across 4 weeks)
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "Naive (recursive)",
            **evaluate_block(y_true, naive_pred),
        })
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "Seasonal Naive (recursive)",
            **evaluate_block(y_true, seasonal_pred),
        })
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": test["date"].iloc[0],
            "test_end": test["date"].iloc[-1],
            "model": "XGBoost (recursive)",
            **evaluate_block(y_true, xgb_pred),
        })


    pred_df = pd.DataFrame(predictions_rows)
    pred_df.to_csv(PRED_PATH, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)
    run_id = str(uuid.uuid4())
    pred_df.insert(0, "run_id", run_id)
    metrics_df.insert(0, "run_id", run_id)

    engine = get_engine()
    schema = schema_name()

    pred_df.to_sql("backtest_predictions", engine, schema=schema, if_exists="append", index=False)
    metrics_df.to_sql("backtest_metrics", engine, schema=schema, if_exists="append", index=False)

    print(f"Wrote backtest outputs to Postgres with run_id={run_id}")
    
    print(f"Saved predictions: {PRED_PATH} ({len(pred_df)} rows)")
    print(f"Saved metrics:     {METRICS_PATH} ({len(metrics_df)} rows)")

    print("\nOverall metrics (mean across windows):")
    summary = (
        metrics_df.groupby("model")[["mae", "rmse", "smape", "bias"]]
        .mean()
        .sort_values("mae")
    )
    print(summary.to_string())

    print("\nHorizon-wise (mean MAE by step):")
    horizon_summary = (
        pred_df.assign(err_xgb=np.abs(pred_df["xgb_recursive"] - pred_df["y_true"]),
                       err_naive=np.abs(pred_df["naive_recursive"] - pred_df["y_true"]))
        .groupby("horizon_step")[["err_naive", "err_xgb"]]
        .mean()
    )
    print(horizon_summary.to_string())


if __name__ == "__main__":
    main()