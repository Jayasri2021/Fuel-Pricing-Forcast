from __future__ import annotations

from pathlib import Path
from src.db import get_engine, schema_name

import pandas as pd

RAW_GAS_PATH = Path("../data/raw/gas_weekly.csv")
RAW_WTI_PATH = Path("../data/raw/wti_daily.csv")
OUT_PATH = Path("../data/processed/base_weekly.parquet")

GAS_DATE_COL = "Week of"
GAS_VALUE_COL = "Weekly U.S. All Grades All Formulations Retail Gasoline Prices Dollars per Gallon"

WTI_DATE_COL = "observation_date"
WTI_VALUE_COL = "DCOILWTICO"


def load_gas(gas_path: Path = RAW_GAS_PATH) -> pd.DataFrame:
    """
    EIA download includes metadata lines before the header row.
    We'll detect the header row by finding the line that starts with 'Week of,'.
    """
    # Find header line index
    with gas_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Week of,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row starting with 'Week of,' in gas_weekly.csv")

    gas = pd.read_csv(gas_path, skiprows=header_idx)
    return gas


def load_wti(wti_path: Path = RAW_WTI_PATH) -> pd.DataFrame:
    wti = pd.read_csv(wti_path)
    return wti


def clean_and_standardize(gas: pd.DataFrame, wti: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Parse dates
    gas[GAS_DATE_COL] = pd.to_datetime(gas[GAS_DATE_COL])
    wti[WTI_DATE_COL] = pd.to_datetime(wti[WTI_DATE_COL])

    # Numeric conversion (FRED sometimes has '.' for missing)
    gas[GAS_VALUE_COL] = pd.to_numeric(gas[GAS_VALUE_COL], errors="coerce")
    wti[WTI_VALUE_COL] = pd.to_numeric(wti[WTI_VALUE_COL], errors="coerce")

    # Keep contract columns
    gas = gas[[GAS_DATE_COL, GAS_VALUE_COL]].rename(
        columns={GAS_DATE_COL: "date", GAS_VALUE_COL: "y_gas_price"}
    )
    wti = wti[[WTI_DATE_COL, WTI_VALUE_COL]].rename(
        columns={WTI_DATE_COL: "date", WTI_VALUE_COL: "wti_usd_per_barrel"}
    )

    # Sort + drop missing
    gas = gas.sort_values("date").dropna(subset=["date", "y_gas_price"])
    wti = wti.sort_values("date").dropna(subset=["date", "wti_usd_per_barrel"])

    return gas, wti


def align_wti_to_gas_dates(gas: pd.DataFrame, wti: pd.DataFrame) -> pd.DataFrame:
    """
    For each weekly gas date, attach the most recent WTI value at/before that date.
    This avoids week-ending alignment issues.
    """
    merged = pd.merge_asof(gas, wti, on="date", direction="backward")
    return merged


def save_parquet(df: pd.DataFrame, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    gas_raw = load_gas()
    wti_raw = load_wti()
    engine = get_engine()
    schema = schema_name()

    gas.to_sql("raw_gas_weekly", engine, schema=schema, if_exists="replace", index=False)
    wti.to_sql("raw_wti_daily", engine, schema=schema, if_exists="replace", index=False)
    
    gas, wti = clean_and_standardize(gas_raw, wti_raw)
    base = align_wti_to_gas_dates(gas, wti)

    save_parquet(base, OUT_PATH)
    print(f"Saved {len(base)} rows to {OUT_PATH}")
    print(base.head(10).to_string(index=False))
    print("\nNull rate:")
    print(base.isna().mean().to_string())

    base.to_sql("base_weekly", engine, schema=schema, if_exists="replace", index=False)

    print("Wrote raw + base tables to Postgres.")

if __name__ == "__main__":
    main()