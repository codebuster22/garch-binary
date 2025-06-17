#!/usr/bin/env python3
"""
Debug script to examine bids/asks structure in order_books table
"""

import pandas as pd
import sqlalchemy as sa
from config import settings

def examine_bids_asks():
    engine = sa.create_engine(settings.DATABASE_URL)
    
    query = '''
    SELECT bids, asks, asset_id, published_at
    FROM order_books 
    WHERE subject LIKE 'market.v1.dev.polymarket.l2book.%'
      AND bids != '[]'::jsonb 
      AND asks != '[]'::jsonb
    LIMIT 5
    '''
    
    result = pd.read_sql(sa.text(query), engine)
    
    print(f"Found {len(result)} records with non-empty bids/asks")
    print("\nSample bids/asks structure:")
    
    for i, row in result.iterrows():
        print(f"\n--- Record {i+1} ---")
        print(f"Asset ID: {row['asset_id']}")
        print(f"Published: {row['published_at']}")
        print(f"Bids type: {type(row['bids'])}")
        print(f"Asks type: {type(row['asks'])}")
        print(f"Bids: {row['bids']}")
        print(f"Asks: {row['asks']}")
        
        # Try to access first bid and ask
        try:
            bids = row['bids']
            asks = row['asks']
            
            if bids and len(bids) > 0:
                print(f"First bid: {bids[0]}")
                print(f"First bid type: {type(bids[0])}")
                print(f"First bid length: {len(bids[0]) if hasattr(bids[0], '__len__') else 'N/A'}")
                
            if asks and len(asks) > 0:
                print(f"First ask: {asks[0]}")
                print(f"First ask type: {type(asks[0])}")
                print(f"First ask length: {len(asks[0]) if hasattr(asks[0], '__len__') else 'N/A'}")
                
        except Exception as e:
            print(f"Error processing bids/asks: {e}")

if __name__ == "__main__":
    examine_bids_asks() 