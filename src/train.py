from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/modeling_weekly.parquet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_model.json"


def train_test_split_time(df: pd.DataFrame, test_weeks: int = 52):
    df = df.sort_values("date")
    train = df.iloc[:-test_weeks]
    test = df.iloc[-test_weeks:]
    return train, test


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    s = smape(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"SMAPE: {s:.2f}%")

    return {"model": name, "mae": mae, "rmse": rmse, "smape": s}


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    train, test = train_test_split_time(df, test_weeks=52)

    y_train = train["y_gas_price"]
    y_test = test["y_gas_price"]

    feature_cols = [c for c in df.columns if c not in ["date", "y_gas_price"]]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    results = []

    # ----------------------------
    # Baseline 1: Naive (lag_1)
    # ----------------------------
    naive_pred = test["y_gas_price_lag_1"]
    results.append(evaluate("Naive (lag_1)", y_test, naive_pred))

    # ----------------------------
    # Baseline 2: Seasonal naive (lag_52)
    # ----------------------------
    seasonal_pred = test["y_gas_price_lag_52"]
    results.append(evaluate("Seasonal Naive (lag_52)", y_test, seasonal_pred))

    # ----------------------------
    # XGBoost
    # ----------------------------
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    xgb_pred = model.predict(X_test)
    results.append(evaluate("XGBoost", y_test, xgb_pred))

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("reports/train_test_metrics.csv", index=False)

    print("\nSaved model and metrics.")


if __name__ == "__main__":
    main()