from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "market.duckdb"
CONSTITUENTS_DB_PATH = ROOT / "market_data.duckdb"
INDEX_TICKERS = ["^NSEI", "^INDIAVIX"]


def compute_signal(row: pd.Series) -> tuple[str, int, list[str]]:
    score = 0
    reasons: list[str] = []

    close_price = float(row["close"])
    open_price = float(row["open"])
    ema20 = float(row["ema20"]) if pd.notna(row["ema20"]) else None
    ema50 = float(row["ema50"]) if pd.notna(row["ema50"]) else None
    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else None
    macd = float(row["macd"]) if pd.notna(row["macd"]) else None
    volume_spike = bool(row["volume_spike"]) if pd.notna(row["volume_spike"]) else False

    if ema20 is not None and ema50 is not None:
        if close_price > ema20 > ema50:
            score += 2
            reasons.append("Price is above EMA20 and EMA50 (uptrend).")
        elif close_price < ema20 < ema50:
            score -= 2
            reasons.append("Price is below EMA20 and EMA50 (downtrend).")

    if rsi is not None:
        if rsi < 35:
            score += 1
            reasons.append("RSI is in/near oversold zone (<35).")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI is overbought (>70).")

    if macd is not None:
        if macd > 0:
            score += 1
            reasons.append("MACD is positive (bullish momentum).")
        elif macd < 0:
            score -= 1
            reasons.append("MACD is negative (bearish momentum).")

    if volume_spike:
        if close_price >= open_price:
            score += 1
            reasons.append("Volume spike with green candle confirms buying interest.")
        else:
            score -= 1
            reasons.append("Volume spike with red candle confirms selling pressure.")

    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, reasons


def load_tickers(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM market_data
        WHERE ticker IS NOT NULL
        ORDER BY ticker
        """
    ).fetchall()
    return [r[0] for r in rows]


def load_index_names() -> list[str]:
    if not CONSTITUENTS_DB_PATH.is_file():
        return ["NIFTY_50"]
    c = duckdb.connect(str(CONSTITUENTS_DB_PATH), read_only=True)
    try:
        names = c.execute(
            """
            SELECT DISTINCT index_name
            FROM index_constituents
            ORDER BY index_name
            """
        ).fetchall()
    finally:
        c.close()
    return [n[0] for n in names] or ["NIFTY_50"]


def load_index_symbols(index_name: str, instrument_type: str | None = None) -> list[str]:
    if not CONSTITUENTS_DB_PATH.is_file():
        return []
    c = duckdb.connect(str(CONSTITUENTS_DB_PATH), read_only=True)
    try:
        filter_type = "AND instrument_type = ?" if instrument_type else ""
        params: list[str] = [index_name]
        if instrument_type:
            params.append(instrument_type)
        params.append(index_name)
        rows = c.execute(
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
        ).fetchall()
    finally:
        c.close()
    return [r[0] for r in rows]


def load_latest_row(conn: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            date,
            ticker,
            open,
            high,
            low,
            close,
            volume,
            rsi,
            macd,
            ema20,
            ema50,
            volume_spike,
            predicted,
            actual
        FROM market_data
        WHERE ticker = ?
          AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """,
        [ticker],
    ).fetchdf()


def load_latest_rows_for_tickers(
    conn: duckdb.DuckDBPyConnection, tickers: list[str]
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(tickers))
    return conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                date,
                ticker,
                open,
                high,
                low,
                close,
                volume,
                rsi,
                macd,
                ema20,
                ema50,
                volume_spike,
                predicted,
                actual,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM market_data
            WHERE ticker IN ({placeholders})
              AND close IS NOT NULL
        )
        SELECT date, ticker, open, high, low, close, volume, rsi, macd, ema20, ema50, volume_spike, predicted, actual
        FROM ranked
        WHERE rn = 1
        ORDER BY ticker
        """,
        tickers,
    ).fetchdf()


def load_last_prediction(conn: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT date, ticker, predicted
        FROM market_data
        WHERE ticker = ?
          AND predicted IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """,
        [ticker],
    ).fetchdf()


def load_last_prediction_dates(
    conn: duckdb.DuckDBPyConnection, tickers: list[str]
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "last_predicted_date", "last_predicted"])
    placeholders = ",".join(["?"] * len(tickers))
    return conn.execute(
        f"""
        SELECT
            ticker,
            MAX(date) FILTER (WHERE predicted IS NOT NULL) AS last_predicted_date,
            ANY_VALUE(predicted) FILTER (
                WHERE date = (
                    SELECT MAX(m2.date)
                    FROM market_data m2
                    WHERE m2.ticker = m1.ticker
                      AND m2.predicted IS NOT NULL
                )
            ) AS last_predicted
        FROM market_data m1
        WHERE ticker IN ({placeholders})
        GROUP BY ticker
        """,
        tickers,
    ).fetchdf()


def consolidated_signal(latest_rows: pd.DataFrame) -> tuple[str, int]:
    if latest_rows.empty:
        return "NO_DATA", 0
    scores = []
    for _, row in latest_rows.iterrows():
        _, score, _ = compute_signal(row)
        scores.append(score)
    total = int(sum(scores))
    avg = total / len(scores)
    if avg >= 1.2:
        return "BULLISH", total
    if avg <= -1.2:
        return "BEARISH", total
    return "SIDEWAYS", total


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df = display_df.dropna(axis=1, how="all")
    return display_df


def main() -> None:
    st.set_page_config(page_title="Market Signal Dashboard", layout="wide")
    st.title("NSE Signal Dashboard")
    st.caption("Single ticker + consolidated index dashboard with indicator-based signals")

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        tickers = load_tickers(conn)
        if not tickers:
            st.warning("No tickers found in `market_data`. Load data first.")
            return

        page = st.sidebar.radio(
            "Page",
            ["NIFTY50 Total Dashboard", "Ticker Analysis"],
            index=0,
        )

        if page == "NIFTY50 Total Dashboard":
            nifty50_symbols = load_index_symbols("NIFTY_50", instrument_type="STOCK")
            ticker_choices = [t for t in nifty50_symbols if t in tickers]
            if "^NSEI" in tickers and "^NSEI" not in ticker_choices:
                ticker_choices.append("^NSEI")

            latest_rows = load_latest_rows_for_tickers(conn, ticker_choices)
            if latest_rows.empty:
                st.warning("No latest rows found for NIFTY_50 tickers.")
                return

            market_view, total_score = consolidated_signal(latest_rows)
            pred_nsei = load_last_prediction(conn, "^NSEI")
            nsei_pred_label = "-"
            nsei_pred_date = "-"
            if not pred_nsei.empty:
                nsei_pred_label = str(pred_nsei.iloc[0]["predicted"])
                nsei_pred_date = str(pred_nsei.iloc[0]["date"])
            buy_count = 0
            sell_count = 0
            hold_count = 0
            scored_rows = []
            prediction_dates = load_last_prediction_dates(conn, ticker_choices)
            prediction_map = (
                prediction_dates.set_index("ticker").to_dict("index")
                if not prediction_dates.empty
                else {}
            )
            for _, row in latest_rows.iterrows():
                sig, score, _ = compute_signal(row)
                if sig == "BUY":
                    buy_count += 1
                elif sig == "SELL":
                    sell_count += 1
                else:
                    hold_count += 1
                row_dict = row.to_dict()
                row_dict["signal"] = sig
                row_dict["score"] = score
                pred_data = prediction_map.get(row_dict["ticker"], {})
                row_dict["last_predicted"] = pred_data.get("last_predicted")
                row_dict["last_predicted_date"] = pred_data.get("last_predicted_date")
                scored_rows.append(row_dict)

            st.subheader("NIFTY 50 Consolidated Dashboard")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Universe Size", len(latest_rows))
            c2.metric("BUY / HOLD / SELL", f"{buy_count} / {hold_count} / {sell_count}")
            c3.metric("Total Signal Score", f"{total_score:+d}")
            c4.metric("NIFTY 50 View", market_view)
            c5.metric("^NSEI Prediction", f"{nsei_pred_label} ({nsei_pred_date})")

            df_view = pd.DataFrame(scored_rows).sort_values(
                by=["score", "ticker"], ascending=[False, True]
            )
            st.dataframe(prepare_display_df(df_view), use_container_width=True)
            return

        index_names = load_index_names()
        selected_index = st.selectbox("Select index", index_names, index=0)
        index_symbols = load_index_symbols(selected_index, instrument_type="STOCK")
        ticker_choices = [t for t in index_symbols if t in tickers]
        if selected_index == "BENCHMARK_INDICES":
            benchmark_tickers = load_index_symbols("BENCHMARK_INDICES", instrument_type="INDEX")
            for t in benchmark_tickers:
                if t in tickers and t not in ticker_choices:
                    ticker_choices.append(t)
        if not ticker_choices:
            ticker_choices = tickers

        selected_ticker = st.selectbox("Select ticker", ticker_choices, index=0)

        latest_df = load_latest_row(conn, selected_ticker)

        if latest_df.empty:
            st.warning(f"No records found for ticker `{selected_ticker}`.")
            return

        row = latest_df.iloc[0]
        signal, score, reasons = compute_signal(row)
        last_pred_df = load_last_prediction(conn, selected_ticker)
        last_pred_label = "-"
        last_pred_date = "-"
        if not last_pred_df.empty:
            last_pred_label = str(last_pred_df.iloc[0]["predicted"])
            last_pred_date = str(last_pred_df.iloc[0]["date"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", str(row["ticker"]))
        c2.metric("Latest Date", str(row["date"]))
        c3.metric("Signal Score", f"{score:+d}")
        c4.metric("Last Predicted", f"{last_pred_label} ({last_pred_date})")

        if signal == "BUY":
            st.success(f"Signal: {signal}")
        elif signal == "SELL":
            st.error(f"Signal: {signal}")
        else:
            st.info(f"Signal: {signal}")

        st.subheader("Why this signal")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("- Not enough indicator data yet (likely early rows).")

        st.subheader("Latest Record")
        display_df = prepare_display_df(latest_df)
        display_df["last_predicted"] = last_pred_label
        display_df["last_predicted_date"] = last_pred_date
        st.dataframe(display_df, use_container_width=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
