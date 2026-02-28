from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/base_weekly.parquet")
OUT_PATH = Path("data/processed/modeling_weekly.parquet")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month.astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    df["time_idx"] = np.arange(len(df), dtype=int)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for l in lags:
        df[f"{col}_lag_{l}"] = df[col].shift(l)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """
    Leakage-safe: rolling stats only use past values.
    We do shift(1) first, then rolling(window).
    """
    df = df.copy()
    s = df[col].shift(1)

    for w in windows:
        df[f"{col}_roll_mean_{w}"] = s.rolling(window=w, min_periods=w).mean()
        df[f"{col}_roll_median_{w}"] = s.rolling(window=w, min_periods=w).median()
        df[f"{col}_roll_std_{w}"] = s.rolling(window=w, min_periods=w).std()
    return df


def add_rolling_slope(df: pd.DataFrame, col: str, window: int = 14) -> pd.DataFrame:
    """
    Simple trend feature: slope of last `window` points (past-only).
    """
    df = df.copy()
    s = df[col].shift(1)

    def slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        y = arr.astype(float)
        # If any nan, slope = nan
        if np.any(np.isnan(y)):
            return np.nan
        # slope of linear fit
        return np.polyfit(x, y, 1)[0]

    df[f"{col}_roll_slope_{window}"] = (
        s.rolling(window=window, min_periods=window)
        .apply(lambda x: slope(x.to_numpy()), raw=False)
    )
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Rename target explicitly (already y_gas_price in your base)
    # Add time features
    df = add_time_features(df)

    # Lags (weekly)
    target_lags = [1, 4, 8, 12, 26, 52]  # 1w, 1m, 2m, 3m, ~6m, 1y
    df = add_lag_features(df, "y_gas_price", target_lags)

    # Rolling windows (weekly)
    windows = [4, 8, 12, 26]  # 1m, 2m, 3m, ~6m
    df = add_rolling_features(df, "y_gas_price", windows)

    # Trend slope
    df = add_rolling_slope(df, "y_gas_price", window=14)

    # Exogenous WTI features (optional but strong)
    df = add_lag_features(df, "wti_usd_per_barrel", [1, 4, 12])
    df = add_rolling_features(df, "wti_usd_per_barrel", [4, 12])

    return df


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows without full history needed for the max lag/window.
    """
    feature_cols = [c for c in df.columns if c not in ["date"]]
    # Keep only rows where all features exist (no NaNs)
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    feats = make_features(df)
    modeling = drop_warmup(feats)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_parquet(OUT_PATH, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Modeling rows (after warmup drop): {len(modeling)}")
    print("Columns:", len(modeling.columns))
    print(modeling.head(5).to_string(index=False))


if __name__ == "__main__":
    main()