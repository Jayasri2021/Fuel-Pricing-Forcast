from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/processed/modeling_weekly.parquet")
MODEL_DIR = Path("models/multi_horizon")
REPORTS_DIR = Path("reports")

HORIZONS = [1, 2, 3, 4]
TEST_WEEKS = 52  # simple final holdout for quick sanity


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1e-8, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    s = smape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    print(f"{name:>10} | MAE {mae:.4f} | RMSE {rmse:.4f} | SMAPE {s:.2f}% | Bias {bias:+.4f}")
    return {"horizon": name, "mae": mae, "rmse": rmse, "smape": s, "bias": bias}


def make_supervised(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    for h in horizons:
        df[f"y_t_plus_{h}"] = df["y_gas_price"].shift(-h)
    # drop rows where any horizon label is missing (tail)
    df = df.dropna(subset=[f"y_t_plus_{h}" for h in horizons]).reset_index(drop=True)
    return df


def fit_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    # Sensible default; tune later if needed
    model = xgb.XGBRegressor(
        n_estimators=2000,
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
    df = pd.read_parquet(DATA_PATH)
    df = make_supervised(df, HORIZONS)

    feature_cols = [c for c in df.columns if c not in ["date", "y_gas_price"] and not c.startswith("y_t_plus_")]

    # time split
    train = df.iloc[:-TEST_WEEKS].copy()
    test = df.iloc[-TEST_WEEKS:].copy()

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    meta = {"feature_cols": feature_cols, "horizons": HORIZONS, "test_weeks": TEST_WEEKS}

    print("\nHoldout metrics by horizon (direct models):")
    for h in HORIZONS:
        y_train = train[f"y_t_plus_{h}"].astype(float)
        y_test = test[f"y_t_plus_{h}"].astype(float).to_numpy()

        model = fit_xgb(X_train, y_train)
        pred = model.predict(X_test).astype(float)

        model_path = MODEL_DIR / f"xgb_h{h}.json"
        model.save_model(model_path)

        results.append(evaluate(f"h={h}", y_test, pred))

    # Save metrics table
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(REPORTS_DIR / "train_test_metrics_multi_horizon.csv", index=False)

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved models to:", MODEL_DIR)
    print("Saved metrics to:", REPORTS_DIR / "train_test_metrics_multi_horizon.csv")


if __name__ == "__main__":
    main()