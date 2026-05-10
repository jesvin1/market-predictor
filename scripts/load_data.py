from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DB = ROOT / "data" / "market.duckdb"
CONSTITUENTS_DB = ROOT / "market_data.duckdb"
PERIOD = "6mo"


def normalize_column_name(col: object, ticker: str) -> str:
    if isinstance(col, tuple):
        parts = [str(part) for part in col if part and str(part) != ticker]
        col = parts[0] if parts else "_".join(str(part) for part in col if part)
    return str(col).lower().replace(" ", "_")


def fetch_symbols_for(index_name: str, instrument_type: str | None = None) -> list[str]:
    if not CONSTITUENTS_DB.is_file():
        raise FileNotFoundError(
            f"Missing {CONSTITUENTS_DB}. Run from project root: python scripts/Get_instruments.py"
        )
    conn = duckdb.connect(str(CONSTITUENTS_DB.resolve()), read_only=True)
    try:
        filter_type = "AND instrument_type = ?" if instrument_type else ""
        params: list[str] = [index_name]
        if instrument_type:
            params.append(instrument_type)
        params.append(index_name)
        df = conn.execute(
            f"""
            SELECT DISTINCT symbol
            FROM index_constituents
            WHERE index_name = ?
              {filter_type}
              AND snapshot_date = (
                  SELECT MAX(snapshot_date)
                  FROM index_constituents
                  WHERE index_name = ?
              )
            ORDER BY symbol
            """,
            params,
        ).fetchdf()
    finally:
        conn.close()
    return df["symbol"].tolist() if not df.empty else []


def build_ticker_list(nifty_symbols: list[str], benchmark_symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in benchmark_symbols + nifty_symbols:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def ensure_market_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
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
        """
    )
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS rsi DOUBLE")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS macd DOUBLE")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema20 DOUBLE")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema50 DOUBLE")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS volume_spike BOOLEAN")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS predicted VARCHAR")
    conn.execute("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS actual VARCHAR")


def indicators_and_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw.empty:
        return None

    df = raw.copy()
    df.reset_index(inplace=True)
    df.columns = [normalize_column_name(col, ticker) for col in df.columns]

    close_series = pd.to_numeric(df["close"], errors="coerce")
    volume_series = pd.to_numeric(df["volume"], errors="coerce")

    df["rsi"] = RSIIndicator(close=close_series, window=14).rsi()
    df["macd"] = MACD(
        close=close_series, window_slow=26, window_fast=12, window_sign=9
    ).macd()
    df["ema20"] = EMAIndicator(close=close_series, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=close_series, window=50).ema_indicator()
    df["volume_spike"] = volume_series > (
        volume_series.rolling(window=20, min_periods=1).mean() * 1.5
    )

    df["ticker"] = ticker
    df["predicted"] = None
    df["actual"] = None

    return df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticker",
            "rsi",
            "macd",
            "ema20",
            "ema50",
            "volume_spike",
            "predicted",
            "actual",
        ]
    ]


def merge_into_market(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    conn.register("source_df", df)
    try:
        conn.execute(
            """
            MERGE INTO market_data AS target
            USING source_df AS source
            ON target.date = source.date AND target.ticker = source.ticker
            WHEN MATCHED THEN
                UPDATE SET
                    open = source.open,
                    high = source.high,
                    low = source.low,
                    close = source.close,
                    volume = source.volume,
                    rsi = source.rsi,
                    macd = source.macd,
                    ema20 = source.ema20,
                    ema50 = source.ema50,
                    volume_spike = source.volume_spike,
                    predicted = COALESCE(source.predicted, target.predicted),
                    actual = COALESCE(source.actual, target.actual)
            WHEN NOT MATCHED THEN
                INSERT (
                    date, open, high, low, close, volume, ticker, rsi, macd, ema20, ema50, volume_spike, predicted, actual
                )
                VALUES (
                    source.date, source.open, source.high, source.low, source.close, source.volume, source.ticker,
                    source.rsi, source.macd, source.ema20, source.ema50, source.volume_spike, source.predicted, source.actual
                )
            """
        )
    finally:
        conn.unregister("source_df")


def next_working_day(last_date: pd.Timestamp) -> pd.Timestamp:
    nxt = last_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def label_to_text(value: int) -> str:
    return "positive" if value == 1 else "negative"


def update_actual_labels(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        UPDATE market_data AS t
        SET actual = s.actual_label
        FROM (
            SELECT
                ticker,
                date,
                CASE
                    WHEN prev_close IS NULL THEN NULL
                    WHEN close > prev_close THEN 'positive'
                    ELSE 'negative'
                END AS actual_label
            FROM (
                SELECT
                    ticker,
                    date,
                    close,
                    LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
                FROM market_data
                WHERE close IS NOT NULL
            ) z
        ) AS s
        WHERE t.ticker = s.ticker AND t.date = s.date
        """
    )


def upsert_prediction_row(
    conn: duckdb.DuckDBPyConnection, ticker: str, pred_date: pd.Timestamp, label: str
) -> None:
    conn.execute(
        """
        MERGE INTO market_data AS t
        USING (
            SELECT ?::DATE AS date, ?::VARCHAR AS ticker, ?::VARCHAR AS predicted
        ) AS s
        ON t.date = s.date AND t.ticker = s.ticker
        WHEN MATCHED THEN
            UPDATE SET predicted = s.predicted
        WHEN NOT MATCHED THEN
            INSERT (date, ticker, predicted)
            VALUES (s.date, s.ticker, s.predicted)
        """,
        [pred_date.date(), ticker, label],
    )


def predict_single_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> str | None:
    df = conn.execute(
        """
        SELECT date, close, rsi, macd, ema20, ema50, volume_spike
        FROM market_data
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date
        """,
        [ticker],
    ).fetchdf()
    if len(df) < 40:
        return None

    df["ret_1d"] = df["close"].pct_change()
    df["volume_spike"] = df["volume_spike"].fillna(False).astype(int)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    feature_cols = ["close", "rsi", "macd", "ema20", "ema50", "volume_spike", "ret_1d"]
    model_df = df.dropna(subset=feature_cols + ["target"]).copy()
    if len(model_df) < 30 or model_df["target"].nunique() < 2:
        return None

    X = model_df[feature_cols]
    y = model_df["target"]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    pred_value = int(model.predict(X.tail(1))[0])
    return label_to_text(pred_value)


def predict_nsei_with_cross_features(
    conn: duckdb.DuckDBPyConnection, tickers: list[str]
) -> str | None:
    if "^NSEI" not in tickers:
        return None
    placeholders = ",".join(["?"] * len(tickers))
    prices = conn.execute(
        f"""
        SELECT date, ticker, close
        FROM market_data
        WHERE ticker IN ({placeholders}) AND close IS NOT NULL
        """,
        tickers,
    ).fetchdf()
    if prices.empty:
        return None

    pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = pivot.pct_change().dropna(how="any")
    if "^NSEI" not in returns.columns or len(returns) < 40:
        return None

    target = (pivot["^NSEI"].shift(-1) > pivot["^NSEI"]).reindex(returns.index).astype(float)
    model_df = returns.copy()
    model_df["target"] = target
    model_df = model_df.dropna()
    if len(model_df) < 30 or model_df["target"].nunique() < 2:
        return None

    X = model_df.drop(columns=["target"])
    y = model_df["target"].astype(int)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    pred_value = int(model.predict(X.tail(1))[0])
    return label_to_text(pred_value)


def run_predictions(conn: duckdb.DuckDBPyConnection, tickers: list[str]) -> None:
    if not tickers:
        return
    placeholders = ",".join(["?"] * len(tickers))
    latest_dates = conn.execute(
        f"""
        SELECT ticker, MAX(date) AS max_date
        FROM market_data
        WHERE ticker IN ({placeholders}) AND close IS NOT NULL
        GROUP BY ticker
        """,
        tickers,
    ).fetchdf()
    if latest_dates.empty:
        return

    for _, row in latest_dates.iterrows():
        ticker = row["ticker"]
        last_date = pd.Timestamp(row["max_date"])
        pred_date = next_working_day(last_date)

        if ticker == "^NSEI":
            pred = predict_nsei_with_cross_features(conn, tickers)
        else:
            pred = predict_single_ticker(conn, ticker)

        if pred is None:
            print(f"  [pred-skip] insufficient training data: {ticker}")
            continue

        upsert_prediction_row(conn, ticker, pred_date, pred)
        print(f"  [pred] {ticker} -> {pred} for {pred_date.date()}")


def main() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for XGBoost predictions. Install with: pip install scikit-learn"
        ) from exc

    DATA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DATA_DB))
    ensure_market_schema(conn)

    nifty_symbols = fetch_symbols_for("NIFTY_50")
    benchmark_symbols = fetch_symbols_for("BENCHMARK_INDICES", "INDEX")
    tickers = build_ticker_list(nifty_symbols, benchmark_symbols)
    print(f"Loading {len(tickers)} tickers (NIFTY_50 + BENCHMARK_INDICES)...")

    for ticker in tickers:
        raw = yf.download(
            ticker,
            period=PERIOD,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        prepared = indicators_and_frame(raw, ticker)
        if prepared is None:
            print(f"  [skip] no data: {ticker}")
            continue
        merge_into_market(conn, prepared)
        print(f"  [ok] {ticker} ({len(prepared)} rows)")

    update_actual_labels(conn)
    run_predictions(conn, tickers)

    summary = conn.execute(
        """
        SELECT ticker, MAX(date) AS last_date, COUNT(*) AS n
        FROM market_data
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    print("\nRows per ticker (sample):")
    print(summary.head(20))
    print(f"\n... total tickers in summary: {len(summary)}")

    conn.close()


if __name__ == "__main__":
    main()
