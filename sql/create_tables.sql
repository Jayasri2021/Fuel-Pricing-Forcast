CREATE SCHEMA IF NOT EXISTS pricing;

-- 1) Raw tables
CREATE TABLE IF NOT EXISTS pricing.raw_gas_weekly (
  week_of DATE PRIMARY KEY,
  gas_price_usd_per_gallon DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing.raw_wti_daily (
  date DATE PRIMARY KEY,
  wti_usd_per_barrel DOUBLE PRECISION
);

-- 2) Base merged weekly (gas weekly + aligned wti)
CREATE TABLE IF NOT EXISTS pricing.base_weekly (
  date DATE PRIMARY KEY,
  y_gas_price DOUBLE PRECISION NOT NULL,
  wti_usd_per_barrel DOUBLE PRECISION
);

-- 3) Modeling features table (you can also make this a VIEW later)
-- We'll create it later via pandas.to_sql(replace)
-- but here's a placeholder schema if you want explicit columns later.
-- For now, we’ll create dynamically from pandas.

-- 4) Monitoring-ready outputs
CREATE TABLE IF NOT EXISTS pricing.backtest_metrics (
  run_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  model TEXT NOT NULL,
  train_start DATE NOT NULL,
  train_end DATE NOT NULL,
  test_start DATE NOT NULL,
  test_end DATE NOT NULL,
  mae DOUBLE PRECISION,
  rmse DOUBLE PRECISION,
  smape DOUBLE PRECISION,
  bias DOUBLE PRECISION,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pricing.backtest_predictions (
  run_id TEXT NOT NULL,
  window_id TEXT NOT NULL,
  cutoff_end_date DATE NOT NULL,
  date DATE NOT NULL,
  horizon_step INT,
  y_true DOUBLE PRECISION,
  naive DOUBLE PRECISION,
  seasonal DOUBLE PRECISION,
  xgb DOUBLE PRECISION,
  created_at TIMESTAMP DEFAULT NOW()
);