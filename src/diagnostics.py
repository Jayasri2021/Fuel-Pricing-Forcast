from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = Path("reports/backtest_multi_horizon_predictions.csv")
METRICS_PATH = Path("reports/backtest_multi_horizon_metrics_summary.csv")
OUT_DIR = Path("reports/diagnostics")

def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_PATH, parse_dates=["cutoff_end_date", "date"])
    metrics = pd.read_csv(METRICS_PATH, parse_dates=["train_start", "train_end", "test_start", "test_end"])

    pred["abs_err_naive"] = (pred["naive_recursive"] - pred["y_true"]).abs()
    pred["abs_err_xgb"] = (pred["xgb_direct"] - pred["y_true"]).abs()

    horizon_mae = pred.groupby("horizon_step")[["abs_err_naive", "abs_err_xgb"]].mean()

    plt.figure()
    plt.plot(horizon_mae.index, horizon_mae["abs_err_naive"], marker="o", label="Naive (recursive)")
    plt.plot(horizon_mae.index, horizon_mae["abs_err_xgb"], marker="o", label="XGBoost (direct multi-horizon)")
    plt.xlabel("Horizon step (weeks)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Error Growth by Horizon")
    plt.legend()
    save_fig(OUT_DIR / "horizon_mae.png")

    # --- Residual distribution (XGB) ---
    pred["resid_xgb"] = pred["xgb_direct"] - pred["y_true"]
    plt.figure()
    plt.hist(pred["resid_xgb"], bins=50)
    plt.xlabel("Residual (prediction - actual)")
    plt.ylabel("Count")
    plt.title("Residual Distribution (XGBoost direct multi-horizon)")
    save_fig(OUT_DIR / "residual_hist_xgb.png")

    # --- Bias over time (rolling) ---
    pred = pred.sort_values("date")
    pred["bias_xgb"] = pred["resid_xgb"]
    pred["bias_xgb_roll"] = pred["bias_xgb"].rolling(52, min_periods=20).mean()  # ~1 year rolling mean

    plt.figure()
    plt.plot(pred["date"], pred["bias_xgb_roll"])
    plt.xlabel("Date")
    plt.ylabel("Rolling mean residual")
    plt.title("Rolling Bias (52-week mean) - XGBoost")
    save_fig(OUT_DIR / "rolling_bias_xgb.png")

    # --- Worst windows (where model struggles) ---
    # Recompute per-window MAE using pred file (most reliable)
    window_mae = (
        pred.groupby(["window_id"])
        .apply(lambda g: pd.Series({
            "mae_naive": float((g["naive_recursive"] - g["y_true"]).abs().mean()),
            "mae_xgb": float((g["xgb_direct"] - g["y_true"]).abs().mean()),
            "cutoff_end_date": g["cutoff_end_date"].iloc[0],
        }))
        .reset_index()
        .sort_values("mae_xgb", ascending=False)
    )

    window_mae.to_csv(OUT_DIR / "worst_windows.csv", index=False)

    # --- Print a tight text summary for README ---
    overall = {
        "naive_mae": float(pred["abs_err_naive"].mean()),
        "xgb_mae": float(pred["abs_err_xgb"].mean()),
    }
    improve = (overall["naive_mae"] - overall["xgb_mae"]) / overall["naive_mae"] * 100.0

    summary_lines = [
        f"Overall MAE - Naive (recursive): {overall['naive_mae']:.4f}",
        f"Overall MAE - XGBoost (direct multi-horizon): {overall['xgb_mae']:.4f}",
        f"Relative improvement: {improve:.1f}%",
        "",
        "Horizon-wise MAE:",
        horizon_mae.to_string(),
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved diagnostics to:", OUT_DIR)
    print((OUT_DIR / "summary.txt").read_text(encoding="utf-8")[:800])


if __name__ == "__main__":
    main() 
