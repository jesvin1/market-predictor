import duckdb

# Connect DuckDB
conn = duckdb.connect("data/market.duckdb")

conn.execute("""
CREATE TABLE IF NOT EXISTS market_data (
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    ticker VARCHAR,
    rsi DOUBLE,
    macd DOUBLE,
    ema20 DOUBLE,
    ema50 DOUBLE,
    volume_spike BOOLEAN,
    predicted VARCHAR,
    actual VARCHAR
)
""")

# Ensure existing tables also get new indicator columns.
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS rsi DOUBLE")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS macd DOUBLE")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema20 DOUBLE")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema50 DOUBLE")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS volume_spike BOOLEAN")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS predicted VARCHAR")
conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS actual VARCHAR")

print(conn.execute("SELECT COUNT(*) FROM market_data").fetchall())
