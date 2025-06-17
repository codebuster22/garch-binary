# config.py

import os
from dotenv import load_dotenv

load_dotenv()  # read from .env at project root

class Settings:
    """Application settings loaded from environment."""
    DATABASE_URL: str = os.getenv("DATABASE_URL")  # e.g. postgres://user:pass@host:port/dbname
    
    # GARCH backtest defaults
    DELTAS = [1, 5, 10]         # minute intervals to test
    WINDOW_DAYS = [30, 60, 90]  # calibration windows in days
    
    # Bitcoin Binary Options Asset IDs
    # Replace these with the actual Bitcoin binary option asset IDs from your database
    BITCOIN_BINARY_OPTION_ASSET_IDS = [
        # TODO: Replace with actual Bitcoin binary option asset IDs
        # Example format (these are placeholders):
        # "100890937137399127151127456856506253118780205854182929749795579740352258083088",
        # "61254456251883347135096329978302889434424298045167660035050871490225505675402",
        # Add more Bitcoin-specific asset IDs as needed
    ]

settings = Settings()
