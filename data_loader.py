# data_loader.py

import pandas as pd
import sqlalchemy as sa
import numpy as np
import logging
from decimal import Decimal, getcontext
from config import settings
import pytz

# Set decimal precision for financial calculations
getcontext().prec = 18  # 18 decimal places for price precision

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_float_division(numerator, denominator, default=np.nan):
    """
    Safely perform floating-point division with fallback.
    Useful for log-return calculations where division by zero can occur.
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        result = numerator / denominator
        if np.isinf(result) or np.isnan(result):
            return default
        return result
    except (ZeroDivisionError, OverflowError):
        return default

# Create a SQLAlchemy engine from DATABASE_URL
try:
    _engine = sa.create_engine(settings.DATABASE_URL, echo=False)
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

def fetch_returns(delta: int, start_ts: str = None, end_ts: str = None) -> pd.DataFrame:
    """
    Compute log-returns for a given delta (in minutes) from binance_klines data.
    Since database only contains 1-minute data, we aggregate to the requested interval.
    Optionally restrict to start/end timestamps (ISO strings or milliseconds).
    Returns DataFrame indexed by ts_ms with column 'log_return'.
    All timestamps are in milliseconds.
    """
    if delta not in [1, 5, 10]:
        raise ValueError(f"Invalid delta: {delta}. Must be one of [1, 5, 10]")
    
    # Always fetch 1-minute data (only interval available in database)
    sql = """
    SELECT 
        open_time as ts_ms,
        close_time,
        open_price,
        close_price,
        high_price,
        low_price,
        volume,
        num_trades,
        quote_volume,
        taker_buy_base_vol,
        taker_buy_quote_vol
    FROM binance_klines
    WHERE symbol = :symbol 
      AND interval = '1m'
      {start_clause}
      {end_clause}
    ORDER BY open_time
    """
    
    start_clause = "AND open_time >= :start_ts_ms" if start_ts else ""
    end_clause = "AND open_time <= :end_ts_ms" if end_ts else ""
    sql = sql.format(start_clause=start_clause, end_clause=end_clause)

    params = {"symbol": "btcusdt"}  # Database uses lowercase symbols
    if start_ts: 
        # Convert ISO string to milliseconds if needed
        if isinstance(start_ts, str) and 'T' in start_ts:
            start_ts_ms = int(pd.Timestamp(start_ts).timestamp() * 1000)
        else:
            start_ts_ms = int(start_ts)
        params["start_ts_ms"] = start_ts_ms
    if end_ts: 
        if isinstance(end_ts, str) and 'T' in end_ts:
            end_ts_ms = int(pd.Timestamp(end_ts).timestamp() * 1000)
        else:
            end_ts_ms = int(end_ts)
        params["end_ts_ms"] = end_ts_ms

    try:
        df = pd.read_sql(sa.text(sql), _engine, params=params)
        
        if df.empty:
            logger.warning(f"No 1-minute klines data found for start_ts={start_ts}, end_ts={end_ts}")
            return pd.DataFrame(columns=['log_return']).set_index('ts_ms')
        
        logger.info(f"Fetched {len(df)} 1-minute klines, aggregating to {delta}-minute intervals...")
        
        # If delta is 1, no aggregation needed
        if delta == 1:
            df = df.sort_values('ts_ms')
            
            # Compute log returns directly
            df['prev_close_price'] = df['close_price'].shift(1)
            df = df.dropna()
            df['log_return'] = np.log(df['close_price'] / df['prev_close_price'])
            
            df.set_index("ts_ms", inplace=True)
            result = df[['log_return']].copy()
            
            logger.info(f"Computed {len(result)} 1-minute log return records")
            return result
        
        # For 5-minute or 10-minute intervals, aggregate the 1-minute data
        aggregated_klines = _aggregate_klines_to_interval(df, delta)
        
        if aggregated_klines.empty:
            logger.warning(f"No data after aggregating to {delta}-minute intervals")
            return pd.DataFrame(columns=['log_return']).set_index('ts_ms')
        
        # Compute log returns from aggregated data
        aggregated_klines = aggregated_klines.sort_values('ts_ms')
        aggregated_klines['prev_close_price'] = aggregated_klines['close_price'].shift(1)
        aggregated_klines = aggregated_klines.dropna()
        aggregated_klines['log_return'] = np.log(aggregated_klines['close_price'] / aggregated_klines['prev_close_price'])
        
        # Set millisecond timestamp as index
        aggregated_klines.set_index("ts_ms", inplace=True)
        result = aggregated_klines[['log_return']].copy()
        
        logger.info(f"Computed {len(result)} {delta}-minute log return records from aggregated data")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch returns data: {e}")
        raise

def _aggregate_klines_to_interval(df_1m: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """
    Aggregate 1-minute klines to the specified interval using OHLCV logic.
    Based on the JavaScript aggregation logic provided.
    All timestamps are in milliseconds.
    
    Args:
        df_1m: DataFrame with 1-minute kline data (ts_ms column)
        interval_minutes: Target interval (5 or 10 minutes)
    
    Returns:
        DataFrame with aggregated klines
    """
    if df_1m.empty:
        return pd.DataFrame()
    
    # Sort by ts_ms to ensure proper ordering
    df_1m = df_1m.sort_values('ts_ms').copy()
    
    # Convert interval to milliseconds
    window_ms = interval_minutes * 60 * 1000
    
    aggregated_rows = []
    current_window = None
    window_start = 0
    
    for _, kline in df_1m.iterrows():
        # Calculate which window this kline belongs to
        kline_window_start = (kline['ts_ms'] // window_ms) * window_ms
        
        if current_window is None or kline_window_start != window_start:
            # Start new window
            if current_window is not None:
                aggregated_rows.append(current_window)
            
            window_start = kline_window_start
            current_window = {
                'ts_ms': window_start,
                'close_time': window_start + window_ms - 1,
                'open_price': kline['open_price'],
                'close_price': kline['close_price'],
                'high_price': kline['high_price'],
                'low_price': kline['low_price'],
                'volume': kline['volume'],
                'num_trades': kline['num_trades'],
                'quote_volume': kline['quote_volume'],
                'taker_buy_base_vol': kline['taker_buy_base_vol'],
                'taker_buy_quote_vol': kline['taker_buy_quote_vol'],
            }
        else:
            # Aggregate into current window (OHLCV logic)
            if current_window is not None:
                # Open: keep the first kline's open (already set)
                # High: maximum of all highs in window
                current_window['high_price'] = max(current_window['high_price'], kline['high_price'])
                # Low: minimum of all lows in window  
                current_window['low_price'] = min(current_window['low_price'], kline['low_price'])
                # Close: use latest close
                current_window['close_price'] = kline['close_price']
                # Volume: sum all volumes
                current_window['volume'] += kline['volume']
                current_window['quote_volume'] += kline['quote_volume']
                current_window['taker_buy_base_vol'] += kline['taker_buy_base_vol']
                current_window['taker_buy_quote_vol'] += kline['taker_buy_quote_vol']
                # Trades: sum number of trades
                current_window['num_trades'] += kline['num_trades']
                # Close time: use latest close time
                current_window['close_time'] = max(current_window['close_time'], kline['close_time'])
    
    # Add the last window
    if current_window is not None:
        aggregated_rows.append(current_window)
    
    if not aggregated_rows:
        return pd.DataFrame()
    
    # Convert to DataFrame
    aggregated_df = pd.DataFrame(aggregated_rows)
    
    logger.debug(f"Aggregated {len(df_1m)} 1-minute klines into {len(aggregated_df)} {interval_minutes}-minute klines")
    
    return aggregated_df

def fetch_binary_labels(start_ts: str = None, end_ts: str = None, limit: int = None) -> pd.DataFrame:
    """
    Fetch complete binary options order book data from order_books table.
    Store full order books in memory and calculate mid_price in Python.
    Returns DataFrame with columns ['mid_price', 'outcome', 'bids', 'asks', 'asset_id'] indexed by ts_ms.
    
    Binary options settle daily at 12 PM ET (Eastern Time).
    The outcome compares Bitcoin price at 12 PM ET on the option day vs 12 PM ET the next day.
    All timestamps are in milliseconds.
    """
    
    # Fetch complete order book data with full bids/asks JSONB
    sql_books = """
    SELECT 
        EXTRACT(EPOCH FROM published_at) * 1000 as ts_ms,
        asset_id,
        bids,
        asks,
        server_timestamp,
        client_timestamp,
        market,
        hash,
        source
    FROM order_books
    WHERE asset_id = ANY(:bitcoin_asset_ids)  -- Only Bitcoin binary options
      AND bids != '[]'::jsonb 
      AND asks != '[]'::jsonb
      {start_clause}
      {end_clause}
    ORDER BY published_at
    {limit_clause}
    """
    
    start_clause = "AND published_at >= to_timestamp(:start_ts_ms / 1000)" if start_ts else ""
    end_clause = "AND published_at <= to_timestamp(:end_ts_ms / 1000)" if end_ts else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    sql_books = sql_books.format(start_clause=start_clause, end_clause=end_clause, limit_clause=limit_clause)

    params = {
        "bitcoin_asset_ids": settings.BITCOIN_BINARY_OPTION_ASSET_IDS
    }
    
    # Check if we have Bitcoin asset IDs configured
    if not settings.BITCOIN_BINARY_OPTION_ASSET_IDS:
        logger.error("No Bitcoin binary option asset IDs configured in settings.BITCOIN_BINARY_OPTION_ASSET_IDS")
        return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
    
    if start_ts: 
        # Convert ISO string to milliseconds if needed
        if isinstance(start_ts, str) and 'T' in start_ts:
            start_ts_ms = int(pd.Timestamp(start_ts).timestamp() * 1000)
        else:
            start_ts_ms = int(start_ts)
        params["start_ts_ms"] = start_ts_ms
    if end_ts: 
        if isinstance(end_ts, str) and 'T' in end_ts:
            end_ts_ms = int(pd.Timestamp(end_ts).timestamp() * 1000)
        else:
            end_ts_ms = int(end_ts)
        params["end_ts_ms"] = end_ts_ms

    try:
        df_books = pd.read_sql(sa.text(sql_books), _engine, params=params)
        
        if df_books.empty:
            logger.warning(f"No order books found for start_ts={start_ts}, end_ts={end_ts}")
            return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
        
        logger.info(f"Fetched {len(df_books)} complete order books, calculating mid_prices in memory...")
        
        # Calculate mid_price in memory from bids/asks JSONB data with Decimal precision
        mid_prices = []
        for _, row in df_books.iterrows():
            try:
                bids = row['bids']  # Already parsed as Python list from JSONB
                asks = row['asks']  # Already parsed as Python list from JSONB
                
                if bids and asks and len(bids) > 0 and len(asks) > 0:
                    # Use Decimal for precise financial calculations
                    # Bids/asks are dictionaries with 'price' key, not nested arrays
                    best_bid = Decimal(str(bids[0]['price']))  # First bid price  
                    best_ask = Decimal(str(asks[0]['price']))  # First ask price
                    mid_price_decimal = (best_bid + best_ask) / Decimal('2')
                    # Convert back to float for compatibility with rest of pipeline
                    mid_price = float(mid_price_decimal)
                else:
                    mid_price = None
                    
            except (ValueError, IndexError, TypeError, ArithmeticError) as e:
                logger.warning(f"Could not parse bid/ask for timestamp {row['ts_ms']}: {e}")
                mid_price = None
                
            mid_prices.append(mid_price)
        
        df_books['mid_price'] = mid_prices
        
        # Remove rows where mid_price couldn't be calculated
        df_books = df_books.dropna(subset=['mid_price'])
        
        if df_books.empty:
            logger.warning("No order books with valid mid_prices")
            return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
        
        logger.info(f"Processing {len(df_books)} binary options for outcome computation...")
        
        # Convert millisecond timestamps to ET timezone for date calculation
        et_tz = pytz.timezone('US/Eastern')
        
        # Convert ts_ms to datetime in ET timezone
        df_books['ts_dt'] = pd.to_datetime(df_books['ts_ms'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(et_tz)
        df_books['option_date'] = df_books['ts_dt'].dt.date
        
        # Get unique settlement dates and calculate 12 PM ET in milliseconds
        unique_dates = df_books['option_date'].unique()
        settlement_times_ms = {}  # date -> (today_12pm_ms, tomorrow_12pm_ms)
        
        for date in unique_dates:
            # Today's 12 PM ET
            today_12pm_et = et_tz.localize(pd.Timestamp.combine(date, pd.Timestamp('12:00:00').time()))
            today_12pm_ms = int(today_12pm_et.timestamp() * 1000)
            
            # Tomorrow's 12 PM ET  
            tomorrow_12pm_et = et_tz.localize(pd.Timestamp.combine(date + pd.Timedelta(days=1), pd.Timestamp('12:00:00').time()))
            tomorrow_12pm_ms = int(tomorrow_12pm_et.timestamp() * 1000)
            
            settlement_times_ms[date] = (today_12pm_ms, tomorrow_12pm_ms)
        
        if not settlement_times_ms:
            logger.error("No settlement dates found")
            return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
        
        # Get Bitcoin price data covering all settlement times
        all_settlement_times = [time_ms for times in settlement_times_ms.values() for time_ms in times]
        min_settlement_ms = min(all_settlement_times)
        max_settlement_ms = max(all_settlement_times)
        
        price_sql = """
        SELECT 
            open_time as ts_ms,
            close_price
        FROM binance_klines 
        WHERE symbol = 'btcusdt'  -- Database uses lowercase symbols 
          AND interval = '1m'
          AND open_time BETWEEN :min_ts_ms AND :max_ts_ms
        ORDER BY open_time
        """
        
        btc_prices = pd.read_sql(sa.text(price_sql), _engine, 
                                params={"min_ts_ms": min_settlement_ms, "max_ts_ms": max_settlement_ms})
        
        if btc_prices.empty:
            logger.error("No Bitcoin price data found for settlement periods")
            return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
        
        btc_prices.set_index('ts_ms', inplace=True)
        
        # For each binary option, compute outcome using 12 PM ET settlements
        outcomes = []
        for _, row in df_books.iterrows():
            option_date = row['option_date']
            
            try:
                today_12pm_ms, tomorrow_12pm_ms = settlement_times_ms[option_date]
                
                # Find closest Bitcoin prices to settlement times
                today_price_idx = btc_prices.index.get_indexer([today_12pm_ms], method='nearest')[0]
                tomorrow_price_idx = btc_prices.index.get_indexer([tomorrow_12pm_ms], method='nearest')[0]
                
                if today_price_idx >= 0 and tomorrow_price_idx >= 0:
                    today_price = btc_prices.iloc[today_price_idx]['close_price']
                    tomorrow_price = btc_prices.iloc[tomorrow_price_idx]['close_price']
                    
                    # Outcome: 1 if Bitcoin went UP from today 12 PM ET to tomorrow 12 PM ET
                    outcome = 1 if tomorrow_price > today_price else 0
                    
                    logger.debug(f"Option {option_date}: {today_price:.2f} -> {tomorrow_price:.2f} = outcome {outcome}")
                else:
                    outcome = None
                    
            except Exception as e:
                logger.warning(f"Could not determine outcome for option on {option_date}: {e}")
                outcome = None
            
            outcomes.append(outcome)
        
        df_books['outcome'] = outcomes
        
        # Remove rows where we couldn't determine outcome
        df_books = df_books.dropna(subset=['outcome'])
        
        if df_books.empty:
            logger.warning("No binary options with determinable outcomes")
            return pd.DataFrame(columns=['mid_price', 'outcome', 'bids', 'asks', 'asset_id']).set_index('ts_ms')
        
        # Validate data
        if df_books['mid_price'].isna().any():
            logger.warning(f"Found {df_books['mid_price'].isna().sum()} records with missing mid_price")
        
        # Check for reasonable price ranges (basic sanity check)
        if len(df_books) > 0:
            min_price, max_price = df_books['mid_price'].min(), df_books['mid_price'].max()
            if min_price <= 0 or max_price > 1.0:
                logger.warning(f"Unusual price range detected: {min_price:.6f} to {max_price:.6f}")
        
        if not df_books['outcome'].isin([0, 1]).all():
            logger.error("Invalid outcome values found (must be 0 or 1)")
            raise ValueError("Invalid outcome values in binary labels")
        
        # Set index to millisecond timestamp and return rich dataset
        df_books.set_index("ts_ms", inplace=True)
        result = df_books[['mid_price', 'outcome', 'bids', 'asks', 'asset_id', 'market', 'hash', 'source']].copy()
        
        logger.info(f"Fetched {len(result)} binary label records with complete order books using 12 PM ET settlements")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch binary labels: {e}")
        raise

def prepare_garch_backtest(delta: int, window: int, limit: int = None) -> pd.DataFrame:
    """
    Prepare data for GARCH(1,1) backtest by joining returns and binary options labels.
    
    Args:
        delta: Time interval in minutes for returns calculation (1, 5, or 10)
        window: Rolling window in days for GARCH estimation (30, 60, or 90)
        limit: Optional limit on number of records to process
        
    Returns:
        DataFrame with columns ['log_return_prev', 'mid_price', 'outcome', 'bids', 'asks'] 
        indexed by ts_ms (milliseconds timestamp)
    """
    
    logger.info(f"Preparing GARCH backtest data with delta={delta}min, window={window}days, limit={limit}")
    
    try:
        # Fetch returns data (indexed by ts_ms)
        logger.info(f"Fetching returns data with delta={delta} minutes...")
        returns_df = fetch_returns(delta=delta)
        
        if returns_df.empty:
            logger.error("No returns data available")
            return pd.DataFrame()
        
        logger.info(f"Fetched {len(returns_df)} return records")
        
        # Fetch binary options data with complete order books (indexed by ts_ms)  
        logger.info("Fetching binary options data with complete order books...")
        binary_df = fetch_binary_labels(limit=limit)
        
        if binary_df.empty:
            logger.error("No binary options data available")
            return pd.DataFrame()
        
        logger.info(f"Fetched {len(binary_df)} binary option records with complete order books")
        
        # Join on millisecond timestamps using inner join
        logger.info("Joining returns and binary options data on millisecond timestamps...")
        combined_df = returns_df.join(binary_df, how='inner', lsuffix='_returns', rsuffix='_binary')
        
        if combined_df.empty:
            logger.error("No matching timestamps between returns and binary options data")
            return pd.DataFrame()
        
        logger.info(f"Successfully joined datasets: {len(combined_df)} matching records")
        
        # Create lagged returns for GARCH modeling
        # GARCH(1,1) uses previous period's return to predict current period's volatility
        combined_df = combined_df.sort_index()  # Ensure chronological order by ts_ms
        combined_df['log_return_prev'] = combined_df['log_return'].shift(1)
        
        # Drop rows with missing lagged returns (first observation)
        combined_df = combined_df.dropna(subset=['log_return_prev'])
        
        if combined_df.empty:
            logger.error("No data remaining after creating lagged returns")
            return pd.DataFrame()
        
        # Select final columns for GARCH backtest
        # Include complete order book data for rich analysis
        result_columns = ['log_return_prev', 'mid_price', 'outcome', 'bids', 'asks', 'asset_id']
        
        # Add optional metadata columns if available
        optional_columns = ['market', 'hash', 'source']
        for col in optional_columns:
            if col in combined_df.columns:
                result_columns.append(col)
        
        result_df = combined_df[result_columns].copy()
        
        # Data validation
        missing_data = result_df[['log_return_prev', 'mid_price', 'outcome']].isna().any(axis=1).sum()
        if missing_data > 0:
            logger.warning(f"Found {missing_data} records with missing critical data, dropping them")
            result_df = result_df.dropna(subset=['log_return_prev', 'mid_price', 'outcome'])
        
        if not result_df['outcome'].isin([0, 1]).all():
            logger.error("Invalid outcome values found in joined data")
            raise ValueError("Binary outcomes must be 0 or 1")
        
        # Check data quality
        if len(result_df) < window:
            logger.warning(f"Only {len(result_df)} records available, less than window size {window}")
        
        logger.info(f"Prepared {len(result_df)} records for GARCH backtest")
        logger.info(f"Data range: {pd.to_datetime(result_df.index.min(), unit='ms')} to {pd.to_datetime(result_df.index.max(), unit='ms')}")
        logger.info(f"Average mid_price: {result_df['mid_price'].mean():.4f}")
        logger.info(f"Binary outcome distribution: {result_df['outcome'].value_counts().to_dict()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to prepare GARCH backtest data: {e}")
        raise
