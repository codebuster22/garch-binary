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

settings = Settings()
