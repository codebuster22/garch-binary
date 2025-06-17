#!/usr/bin/env python3
"""
Helper script to find Bitcoin binary option asset IDs in the database.
Run this script to discover the asset IDs you need to add to config.py
"""

import pandas as pd
import sqlalchemy as sa
from config import settings

def find_bitcoin_asset_ids():
    """Find all unique asset IDs and their associated markets to identify Bitcoin-related ones."""
    engine = sa.create_engine(settings.DATABASE_URL)
    
    print("🔍 Finding Bitcoin Binary Option Asset IDs...")
    print("="*60)
    
    # Get all unique asset IDs with sample data
    query = '''
    SELECT 
        asset_id,
        market,
        COUNT(*) as record_count,
        MIN(published_at) as first_seen,
        MAX(published_at) as last_seen
    FROM order_books 
    WHERE bids != '[]'::jsonb 
      AND asks != '[]'::jsonb
    GROUP BY asset_id, market
    ORDER BY record_count DESC
    LIMIT 20
    '''
    
    result = pd.read_sql(sa.text(query), engine)
    
    print(f"Found {len(result)} unique asset_id/market combinations:")
    print("\nTop 20 by record count:")
    print("-" * 60)
    
    for i, row in result.iterrows():
        print(f"\n{i+1}. Asset ID: {row['asset_id']}")
        print(f"   Market: {row['market']}")
        print(f"   Records: {row['record_count']}")
        print(f"   Period: {row['first_seen']} to {row['last_seen']}")
    
    print("\n" + "="*60)
    print("📋 TO CONFIGURE:")
    print("1. Review the asset IDs and markets above")
    print("2. Identify which ones are Bitcoin-related binary options")
    print("3. Add the Bitcoin asset IDs to config.py in BITCOIN_BINARY_OPTION_ASSET_IDS")
    print("\nExample config.py update:")
    print("BITCOIN_BINARY_OPTION_ASSET_IDS = [")
    
    # Show first few as examples
    for i, row in result.head(3).iterrows():
        print(f'    "{row["asset_id"]}",  # {row["record_count"]} records')
    
    print("    # Add more Bitcoin-specific asset IDs...")
    print("]")

if __name__ == "__main__":
    find_bitcoin_asset_ids() 