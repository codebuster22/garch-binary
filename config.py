# config.py

import os
from dotenv import load_dotenv

load_dotenv()  # read from .env at project root

class Settings:
    """Application settings loaded from environment."""
    DATABASE_URL: str = os.getenv("DATABASE_URL")  # e.g. postgres://user:pass@host:port/dbname
    # Data grid
    DELTAS_MIN = [1, 5, 10]       # aggregation intervals in minutes
    WINDOW_DAYS = [30, 60, 90]    # calibration windows

    # GARCH calibration
    MLE_BOUNDS = [(1e-9, None), (0.0, 1.0), (0.0, 1.0)]  # ω>0, α≥0, β≥0
    MLE_CONSTRAINT = lambda x: x[1] + x[2] < 1.0         # α+β <1
    MLE_MAXITER = 10000

    # Daily roll frequency
    ROLL_FREQ = '1D'              # use pandas date_range with freq='1D'

    # Backtest metrics
    METRICS = ['mse', 'mae']

settings = Settings()
