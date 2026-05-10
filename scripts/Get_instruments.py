# =========================================================
# NSE Index Constituents Scraper -> DuckDB Loader
# =========================================================
#
# Installs:
# pip install pandas requests duckdb lxml
#
# What this script does:
# 1. Downloads NSE index constituent CSV
# 2. Cleans symbols
# 3. Loads data into DuckDB
# 4. Keeps historical snapshots
#
# =========================================================

import requests
import pandas as pd
import duckdb
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

DUCKDB_PATH = "market_data.duckdb"

INDEX_URLS = {
    "NIFTY_50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY_NEXT_50": "https://www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
    "NIFTY_100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY_200": "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTY_500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
}

BENCHMARK_INDEX_DETAILS = [
    {
        "index_name": "BENCHMARK_INDICES",
        "company_name": "NIFTY 50",
        "symbol": "^NSEI",
        "industry": "INDEX",
        "isin_code": None,
    },
    {
        "index_name": "BENCHMARK_INDICES",
        "company_name": "INDIA VIX",
        "symbol": "^INDIAVIX",
        "industry": "INDEX",
        "isin_code": None,
    },
    {
        "index_name": "BENCHMARK_INDICES",
        "company_name": "NASDAQ Composite",
        "symbol": "^IXIC",
        "industry": "INDEX",
        "isin_code": None,
    },
    {
        "index_name": "BENCHMARK_INDICES",
        "company_name": "S&P 500",
        "symbol": "^GSPC",
        "industry": "INDEX",
        "isin_code": None,
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =========================================================
# CREATE TABLES
# =========================================================

con = duckdb.connect(DUCKDB_PATH)

con.execute("""
CREATE TABLE IF NOT EXISTS index_constituents (
    index_name VARCHAR,
    company_name VARCHAR,
    symbol VARCHAR,
    industry VARCHAR,
    isin_code VARCHAR,
    instrument_type VARCHAR,
    snapshot_date DATE,
    loaded_at TIMESTAMP
)
""")

con.execute("ALTER TABLE index_constituents ADD COLUMN IF NOT EXISTS instrument_type VARCHAR")

# =========================================================
# DOWNLOAD + LOAD
# =========================================================

snapshot_date = datetime.today().date()

all_data = []

for index_name, url in INDEX_URLS.items():

    print(f"\nDownloading {index_name} ...")

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed: {index_name}")
        continue

    # Read CSV directly from response
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    print(f"Rows fetched: {len(df)}")

    # Standardize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename columns safely
    rename_map = {
        "company name": "company_name",
        "industry": "industry",
        "symbol": "symbol",
        "isin code": "isin_code"
    }

    df.rename(columns=rename_map, inplace=True)

    # Add Yahoo Finance suffix
    df["symbol"] = df["symbol"].astype(str).str.strip() + ".NS"

    # Add metadata
    df["index_name"] = index_name
    df["instrument_type"] = "STOCK"
    df["snapshot_date"] = snapshot_date
    df["loaded_at"] = datetime.now()

    # Select columns
    final_df = df[
        [
            "index_name",
            "company_name",
            "symbol",
            "industry",
            "isin_code",
            "instrument_type",
            "snapshot_date",
            "loaded_at"
        ]
    ]

    all_data.append(final_df)

# Add benchmark index symbols into the same catalog table
benchmark_df = pd.DataFrame(BENCHMARK_INDEX_DETAILS)
benchmark_df["instrument_type"] = "INDEX"
benchmark_df["snapshot_date"] = snapshot_date
benchmark_df["loaded_at"] = datetime.now()
all_data.append(
    benchmark_df[
        [
            "index_name",
            "company_name",
            "symbol",
            "industry",
            "isin_code",
            "instrument_type",
            "snapshot_date",
            "loaded_at",
        ]
    ]
)

# =========================================================
# COMBINE + INSERT
# =========================================================

if all_data:

    final_dataset = pd.concat(all_data, ignore_index=True)

    print("\nTotal rows:", len(final_dataset))

    # Insert into DuckDB
    con.register("temp_df", final_dataset)

    con.execute("""
    INSERT INTO index_constituents (
        index_name,
        company_name,
        symbol,
        industry,
        isin_code,
        instrument_type,
        snapshot_date,
        loaded_at
    )
    SELECT
        index_name,
        company_name,
        symbol,
        industry,
        isin_code,
        instrument_type,
        snapshot_date,
        loaded_at
    FROM temp_df
    """)

    print("\nData loaded successfully!")

else:
    print("No data downloaded.")

# =========================================================
# SAMPLE QUERY
# =========================================================

result = con.execute("""
SELECT
    index_name,
    COUNT(*) AS total_companies
FROM index_constituents
WHERE snapshot_date = CURRENT_DATE
GROUP BY 1
ORDER BY 1
""").fetchdf()

print("\nLatest Snapshot Counts:")
print(result)

# =========================================================
# CLOSE
# =========================================================

con.close()