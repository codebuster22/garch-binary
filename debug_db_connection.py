#!/usr/bin/env python3
"""
Database Connection Debugging Script

This script helps diagnose database connectivity and query issues.
"""

import pandas as pd
import sqlalchemy as sa
import logging
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test basic database connection."""
    logger.info("🔍 Testing database connection...")
    
    try:
        engine = sa.create_engine(settings.DATABASE_URL, echo=True)  # Enable SQL logging
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1 as test_value"))
            test_row = result.fetchone()
            logger.info(f"✅ Basic connection successful: {test_row}")
            
        return engine
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return None

def test_table_existence(engine):
    """Test if our required tables exist."""
    logger.info("🔍 Testing table existence...")
    
    tables_to_check = ['binance_klines', 'order_books']
    
    for table_name in tables_to_check:
        try:
            query = f"""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            result = pd.read_sql(sa.text(query), engine)
            
            if result.empty:
                logger.warning(f"⚠️  Table '{table_name}' does not exist or is not accessible")
            else:
                logger.info(f"✅ Table '{table_name}' exists with {len(result)} columns:")
                for _, row in result.iterrows():
                    logger.info(f"   - {row['column_name']}: {row['data_type']}")
                
        except Exception as e:
            logger.error(f"❌ Error checking table '{table_name}': {e}")

def test_data_availability(engine):
    """Test if tables have any data."""
    logger.info("🔍 Testing data availability...")
    
    # Test binance_klines
    try:
        klines_count_query = "SELECT COUNT(*) as count FROM binance_klines"
        klines_result = pd.read_sql(sa.text(klines_count_query), engine)
        klines_count = klines_result.iloc[0]['count']
        logger.info(f"📊 binance_klines table has {klines_count} records")
        
        if klines_count > 0:
            # Show sample data
            sample_query = """
            SELECT symbol, interval, open_time, close_time, open_price, close_price, is_closed
            FROM binance_klines 
            ORDER BY open_time DESC 
            LIMIT 3
            """
            sample_data = pd.read_sql(sa.text(sample_query), engine)
            logger.info("📋 Sample binance_klines data:")
            for _, row in sample_data.iterrows():
                logger.info(f"   Symbol: {row['symbol']}, Interval: {row['interval']}, "
                          f"Time: {row['open_time']}, Price: {row['close_price']}, "
                          f"Closed: {row['is_closed']}")
        
    except Exception as e:
        logger.error(f"❌ Error testing binance_klines: {e}")
    
    # Test order_books
    try:
        books_count_query = "SELECT COUNT(*) as count FROM order_books"
        books_result = pd.read_sql(sa.text(books_count_query), engine)
        books_count = books_result.iloc[0]['count']
        logger.info(f"📊 order_books table has {books_count} records")
        
        if books_count > 0:
            # Show sample data
            sample_query = """
            SELECT subject, asset_id, market, published_at
            FROM order_books 
            ORDER BY published_at DESC 
            LIMIT 3
            """
            sample_data = pd.read_sql(sa.text(sample_query), engine)
            logger.info("📋 Sample order_books data:")
            for _, row in sample_data.iterrows():
                logger.info(f"   Subject: {row['subject']}, Asset: {row['asset_id']}, "
                          f"Market: {row['market']}, Time: {row['published_at']}")
        
    except Exception as e:
        logger.error(f"❌ Error testing order_books: {e}")

def test_our_specific_queries(engine):
    """Test the exact queries our system uses."""
    logger.info("🔍 Testing our specific queries...")
    
    # Test the exact binance_klines query
    logger.info("Testing binance_klines query...")
    try:
        klines_query = """
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
          AND is_closed = true
        ORDER BY open_time
        LIMIT 5
        """
        
        params = {"symbol": "BTCUSDT"}
        result = pd.read_sql(sa.text(klines_query), engine, params=params)
        
        logger.info(f"📊 Klines query returned {len(result)} rows with columns: {list(result.columns)}")
        
        if result.empty:
            logger.warning("⚠️  Query returned empty result - this explains our error!")
            logger.info("Possible reasons:")
            logger.info("   1. No data with symbol='BTCUSDT'")
            logger.info("   2. No data with interval='1m'") 
            logger.info("   3. No data with is_closed=true")
            
            # Check what symbols exist
            symbol_query = "SELECT DISTINCT symbol FROM binance_klines LIMIT 10"
            symbols = pd.read_sql(sa.text(symbol_query), engine)
            logger.info(f"Available symbols: {symbols['symbol'].tolist() if not symbols.empty else 'None'}")
            
            # Check what intervals exist
            interval_query = "SELECT DISTINCT interval FROM binance_klines LIMIT 10"
            intervals = pd.read_sql(sa.text(interval_query), engine)
            logger.info(f"Available intervals: {intervals['interval'].tolist() if not intervals.empty else 'None'}")
            
        else:
            logger.info("✅ Klines query successful!")
            logger.info(f"Sample data: {result.head(1).to_dict('records')}")
        
    except Exception as e:
        logger.error(f"❌ Klines query failed: {e}")
    
    # Test the exact order_books query
    logger.info("\nTesting order_books query...")
    try:
        books_query = """
        SELECT 
            EXTRACT(EPOCH FROM published_at) * 1000 as ts_ms,
            asset_id,
            bids,
            asks,
            subject,
            market
        FROM order_books
        WHERE subject = 'order_book'
          AND bids != '[]'::jsonb 
          AND asks != '[]'::jsonb
          AND asset_id ~ '^[0-9]+$'
        ORDER BY published_at
        LIMIT 5
        """
        
        result = pd.read_sql(sa.text(books_query), engine)
        
        logger.info(f"📊 Order books query returned {len(result)} rows with columns: {list(result.columns)}")
        
        if result.empty:
            logger.warning("⚠️  Order books query returned empty result!")
            
            # Check what subjects exist
            subject_query = "SELECT DISTINCT subject FROM order_books LIMIT 10"
            subjects = pd.read_sql(sa.text(subject_query), engine)
            logger.info(f"Available subjects: {subjects['subject'].tolist() if not subjects.empty else 'None'}")
            
        else:
            logger.info("✅ Order books query successful!")
            logger.info(f"Sample data: {result.head(1).to_dict('records')}")
        
    except Exception as e:
        logger.error(f"❌ Order books query failed: {e}")

def main():
    """Run all database diagnostics."""
    logger.info("🚀 Starting Database Diagnostics")
    logger.info("=" * 50)
    
    # Test 1: Basic connection
    engine = test_database_connection()
    if not engine:
        logger.error("❌ Cannot proceed - database connection failed")
        return
    
    print()
    
    # Test 2: Table existence
    test_table_existence(engine)
    
    print()
    
    # Test 3: Data availability
    test_data_availability(engine)
    
    print()
    
    # Test 4: Specific queries
    test_our_specific_queries(engine)
    
    logger.info("=" * 50)
    logger.info("🏁 Database diagnostics completed")

if __name__ == "__main__":
    main() 