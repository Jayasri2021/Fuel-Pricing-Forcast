from __future__ import annotations

from pathlib import Path
from src.db import get_engine, schema_name
import json
import uuid
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/modeling_weekly.parquet")
MODEL_DIR = Path("models/multi_horizon")
REPORTS_DIR = Path("reports")

PRED_PATH = REPORTS_DIR / "backtest_multi_horizon_predictions.csv"
METRICS_PATH = REPORTS_DIR / "backtest_multi_horizon_metrics_summary.csv"

TRAIN_WEEKS = 104
HORIZON_WEEKS = 4
STEP_WEEKS = 4
HORIZONS = [1, 2, 3, 4]
VALID_WEEKS = 8
N_ESTIMATORS = 600
EARLY_STOPPING_ROUNDS = 50
TREE_METHOD = "hist"

run_id = str(uuid.uuid4())


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


def make_supervised(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    for h in horizons:
        df[f"y_t_plus_{h}"] = df["y_gas_price"].shift(-h)
    df = df.dropna(subset=[f"y_t_plus_{h}" for h in horizons]).reset_index(drop=True)
    return df


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method=TREE_METHOD,
        random_state=42,
        n_jobs=-1,
    )
    if X_valid is not None and y_valid is not None:
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        except TypeError:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
    else:
        model.fit(X_train, y_train)
    return model


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df = make_supervised(df, HORIZONS)
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["date", "y_gas_price"] and not c.startswith("y_t_plus_")]

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

        train = df.iloc[train_start:train_end].copy()
        test = df.iloc[test_start:test_end].copy()
        if len(test) < HORIZON_WEEKS:
            break

        cutoff_end_date = train["date"].iloc[-1]
        window_id = f"{train['date'].iloc[0].date()}__{train['date'].iloc[-1].date()}"

        # Naive recursive baseline (repeat last observed)
        last_y = float(train["y_gas_price"].iloc[-1])
        naive_recursive = np.array([last_y] * HORIZON_WEEKS, dtype=float)

        # Train one model per horizon on this window (direct)
        if len(train) <= VALID_WEEKS:
            raise ValueError("Train window too small for the chosen VALID_WEEKS.")

        train_fit = train.iloc[:-VALID_WEEKS].copy()
        train_valid = train.iloc[-VALID_WEEKS:].copy()

        X_train = train_fit[feature_cols]
        X_valid = train_valid[feature_cols]
        X_test = test[feature_cols].head(1)

        xgb_preds_by_h = {}
        for h in HORIZONS:
            y_train = train_fit[f"y_t_plus_{h}"].astype(float)
            y_valid = train_valid[f"y_t_plus_{h}"].astype(float)
            model_h = fit_xgb(X_train, y_train, X_valid, y_valid)
            xgb_preds_by_h[h] = model_h.predict(X_test).astype(float)

        # For each horizon step i=1..4, use the corresponding model's prediction at that row.
        # Because test has 4 rows; for step i, we take pred from model_hi at test row 0
        # BUT to stay consistent, we align: prediction for t+i at the cutoff corresponds to test row i-1.
        # So pick xgb_preds_by_h[i][0] OR xgb_preds_by_h[i][i-1]? Let's do the correct mapping:
        # - label y_t_plus_i is defined per row. In test row 0 (t+1 from cutoff), y_t_plus_1 equals actual at t+1.
        # - In test row 0, model_h1 prediction targets that.
        # - For t+2 relative to cutoff, use test row 0's y_t_plus_2; but we only need horizon 2 step.
        # We want a 4-step forecast from cutoff, so we should use test row 0 and its y_t_plus_h labels/preds.
        # Therefore: use predictions on test row 0 for each horizon h.
        base_row_preds = {h: float(xgb_preds_by_h[h][0]) for h in HORIZONS}
        xgb_direct = np.array([base_row_preds[1], base_row_preds[2], base_row_preds[3], base_row_preds[4]], dtype=float)

        # True future values relative to cutoff are test row 0's y_t_plus_h
        y_true = np.array(
            [float(test["y_t_plus_1"].iloc[0]),
             float(test["y_t_plus_2"].iloc[0]),
             float(test["y_t_plus_3"].iloc[0]),
             float(test["y_t_plus_4"].iloc[0])],
            dtype=float
        )

        # Save per-step predictions
        horizon_dates = test["date"].iloc[:HORIZON_WEEKS].to_list()
        for i in range(HORIZON_WEEKS):
            predictions_rows.append({
                "window_id": window_id,
                "cutoff_end_date": cutoff_end_date,
                "horizon_step": i + 1,
                "date": horizon_dates[i],
                "y_true": y_true[i],
                "naive_recursive": naive_recursive[i],
                "xgb_direct": xgb_direct[i],
            })

        # Metrics (block)
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": horizon_dates[0],
            "test_end": horizon_dates[-1],
            "model": "Naive (recursive)",
            **evaluate_block(y_true, naive_recursive),
        })
        metrics_rows.append({
            "window_id": window_id,
            "train_start": train["date"].iloc[0],
            "train_end": train["date"].iloc[-1],
            "test_start": horizon_dates[0],
            "test_end": horizon_dates[-1],
            "model": "XGBoost (direct multi-horizon)",
            **evaluate_block(y_true, xgb_direct),
        })

    pred_df = pd.DataFrame(predictions_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    pred_df.to_csv(PRED_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    pred_df.insert(0, "run_id", run_id)
    metrics_df.insert(0, "run_id", run_id)

    engine = get_engine()
    schema = schema_name()

    pred_df.to_sql("backtest_multi_horizon_predictions", engine, schema=schema, if_exists="append", index=False)
    metrics_df.to_sql("backtest_multi_horizon_metrics_summary", engine, schema=schema, if_exists="append", index=False)

    print(f"Saved predictions: {PRED_PATH} ({len(pred_df)} rows)")
    print(f"Saved metrics:     {METRICS_PATH} ({len(metrics_df)} rows)")

    print(f"✅ Wrote backtest outputs to Postgres (run_id={run_id})")

    print("\nOverall metrics (mean across windows):")
    summary = (
        metrics_df.groupby("model")[["mae", "rmse", "smape", "bias"]]
        .mean()
        .sort_values("mae")
    )
    print(summary.to_string())

    print("\nHorizon-wise mean MAE:")
    horizon_summary = (
        pred_df.assign(err_naive=np.abs(pred_df["naive_recursive"] - pred_df["y_true"]),
                       err_xgb=np.abs(pred_df["xgb_direct"] - pred_df["y_true"]))
        .groupby("horizon_step")[["err_naive", "err_xgb"]]
        .mean()
    )
    print(horizon_summary.to_string())


if __name__ == "__main__":
    main()
