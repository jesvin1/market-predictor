# NSE Prediction Storage Layout

This project now separates raw market files, DuckDB storage, feature datasets, and prediction exports so the data pipeline stays predictable as it grows.

## Recommended Folder Structure

```text
NSE_Prediction/
├── data/
│   ├── duckdb/          # Canonical DuckDB database file lives here
│   ├── raw/
│   │   ├── equity/      # Raw equity OHLCV / bhavcopy files
│   │   ├── index/       # Raw index-level inputs
│   │   ├── derivatives/ # Futures/options source files
│   │   └── reference/   # Symbol master, holiday calendar, corporate actions
│   ├── staging/         # Cleaned intermediate files before merge/load
│   ├── features/        # Model-ready feature tables or parquet extracts
│   └── exports/         # Predictions, reports, backtest outputs
├── nse_prediction/
│   ├── config.py        # Shared project paths
│   └── db/
│       └── duckdb.py    # DuckDB bootstrap and connection helpers
├── scripts/
│   └── init_storage.py  # Initializes folders and DuckDB schemas
└── sql/
    └── duckdb/
        └── 001_create_schemas.sql
```

## DuckDB Schema Layout

Inside the DuckDB database, use separate schemas for each stage:

- `raw`: source tables loaded with minimal transformation
- `staging`: cleaned and standardized tables
- `features`: model features and training datasets
- `serving`: prediction-ready tables and views
- `meta`: load audit and pipeline metadata

## Bootstrap

Run either of these commands to create the folders and initialize the DuckDB file:

```bash
python main.py
```

```bash
python scripts/init_storage.py
```

The database file is created at `data/duckdb/market_data.duckdb`.
