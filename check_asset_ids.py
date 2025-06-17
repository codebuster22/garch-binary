#!/usr/bin/env python3
"""
Check asset IDs in order_books table to identify Bitcoin-specific binary options
"""

import pandas as pd
import sqlalchemy as sa
from config import settings

def check_asset_ids():
    engine = sa.create_engine(settings.DATABASE_URL)
    
    # Get top asset IDs by record count
    query = '''
    SELECT DISTINCT asset_id, COUNT(*) as count
    FROM order_books 
    WHERE subject LIKE 'market.v1.dev.polymarket.l2book.%'
      AND bids != '[]'::jsonb 
      AND asks != '[]'::jsonb
    GROUP BY asset_id
    ORDER BY count DESC
    LIMIT 10
    '''
    
    result = pd.read_sql(sa.text(query), engine)
    print('Top asset IDs by record count:')
    print(result.to_string(index=False))
    
    print("\n" + "="*50)
    
    # Get sample market data to understand what these represent
    sample_query = '''
    SELECT asset_id, market, subject, published_at
    FROM order_books 
    WHERE subject LIKE 'market.v1.dev.polymarket.l2book.%'
      AND bids != '[]'::jsonb 
      AND asks != '[]'::jsonb
    ORDER BY published_at DESC
    LIMIT 5
    '''
    
    sample_result = pd.read_sql(sa.text(sample_query), engine)
    print('\nSample market data:')
    for i, row in sample_result.iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  Asset ID: {row['asset_id']}")
        print(f"  Market: {row['market']}")
        print(f"  Published: {row['published_at']}")

if __name__ == "__main__":
    check_asset_ids() 