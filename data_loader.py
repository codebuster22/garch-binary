# data_loader.py

import pandas as pd
import sqlalchemy as sa
from config import settings

# Create a SQLAlchemy engine from DATABASE_URL
_engine = sa.create_engine(settings.DATABASE_URL, echo=False)

def fetch_returns(delta: int, start_ts: str = None, end_ts: str = None) -> pd.DataFrame:
    """
    Fetch precomputed log-returns for a given delta (in minutes).
    Optionally restrict to start/end timestamps (ISO strings).
    Returns DataFrame indexed by ts with column 'log_return'.
    """
    sql = """
    SELECT ts, log_return
      FROM returns
     WHERE delta_min = :delta
       {start_clause}
       {end_clause}
     ORDER BY ts
    """
    start_clause = "AND ts >= :start_ts" if start_ts else ""
    end_clause   = "AND ts <= :end_ts"   if end_ts   else ""
    sql = sql.format(start_clause=start_clause, end_clause=end_clause)

    params = {"delta": delta}
    if start_ts: params["start_ts"] = start_ts
    if end_ts:   params["end_ts"]   = end_ts

    df = pd.read_sql(sql, _engine, params=params, parse_dates=["ts"])
    df.set_index("ts", inplace=True)
    return df

def fetch_binary_labels(start_ts: str = None, end_ts: str = None) -> pd.DataFrame:
    """
    Fetch historical binary outcomes (0 or 1) and market mid_price.
    Assumes a binary_quotes table with ts, expiry = ts + 24h, mid_price.
    Returns DataFrame with columns ['mid_price', 'outcome'] indexed by ts.
    """
    sql = """
    SELECT q.ts,
           q.mid_price,
           CASE WHEN f.close_price > f_prev.close_price THEN 1 ELSE 0 END AS outcome
      FROM binary_quotes q
      JOIN binance_klines f
        ON f.symbol = 'BTCUSDT'
       AND f.interval = '1m'
       AND f.open_time = EXTRACT(EPOCH FROM q.expiry)*1000
      JOIN binance_klines f_prev
        ON f_prev.symbol = 'BTCUSDT'
       AND f_prev.interval = '1m'
       AND f_prev.open_time = EXTRACT(EPOCH FROM q.ts)*1000
     WHERE 1=1
       {start_clause}
       {end_clause}
     ORDER BY q.ts
    """
    start_clause = "AND q.ts >= :start_ts" if start_ts else ""
    end_clause   = "AND q.ts <= :end_ts"   if end_ts   else ""
    sql = sql.format(start_clause=start_clause, end_clause=end_clause)

    params = {}
    if start_ts: params["start_ts"] = start_ts
    if end_ts:   params["end_ts"]   = end_ts

    df = pd.read_sql(sql, _engine, params=params, parse_dates=["ts"])
    df.set_index("ts", inplace=True)
    return df

def prepare_garch_backtest(delta: int, window_days: int) -> pd.DataFrame:
    """
    Join returns and binary labels for backtesting GARCH(1,1).
    - delta: return interval in minutes
    - window_days: number of past days for calibration (unused here)
    Returns DataFrame with columns ['log_return', 'mid_price', 'outcome'] indexed by ts.
    """
    # load full series
    rets = fetch_returns(delta)
    bins = fetch_binary_labels()
    # align on same timestamps (only when both available)
    df = rets.join(bins, how="inner")
    # we only need log_return for t-1 when forecasting at t
    df["log_return_prev"] = df["log_return"].shift(1)
    df.dropna(inplace=True)
    return df[["log_return_prev", "mid_price", "outcome"]]
