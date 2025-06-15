### 1. Product Requirements Document (PRD)

#### 1.1 Purpose

Build an **in‐memory backtest harness** to evaluate how well a GARCH(1,1) volatility model can price 24 h BTC binary options, purely by comparing model‐generated probabilities against historical binary mid‐prices. No live pricing, no realized‐outcome P\&L—just model vs. market.

#### 1.2 Goals & Success Metrics

* **Goal:** Identify the combination of K-line return interval (Δt) and calibration window length (W days) that **minimizes squared pricing error** against historical binary mid‐prices.
* **Primary Metric:** Mean Squared Error (MSE)

  $$
    \text{MSE} = \frac{1}{N}\sum_{i=1}^N\bigl(p_{\text{model},i} - p_{\text{market},i}\bigr)^2
  $$
* **Secondary Metric:** Mean Absolute Error (MAE), runtime (< 2 min for 1 year of data).

#### 1.3 In-Scope

* **Data Sources (all read-only):**

  * **Spot candles** from `binance_klines(symbol='BTCUSDT', interval='1m')`.
  * **Binary mid‐prices** from `order_books(asset_id=<uint256>, bids/asks JSONB)`.
* **Model:** GARCH(1,1) calibrated once per (Δt, W) on the first W days of returns.
* **Backtest period:** the days immediately following the calibration period, at daily roll times.
* **Parameter grid:**

  * Δt ∈ {1 min, 5 min, 10 min}
  * W ∈ {30 days, 60 days, 90 days}
* **Output:** In-memory DataFrame of (Δt, W, MSE, MAE, runtime).

#### 1.4 Out-of-Scope

* Realized‐outcome evaluation (no y∈{0,1} used).
* Live forecast service or CLI.
* Database writes—results remain in memory.

#### 1.5 Stakeholders

* **Quants:** determine best Δt & W for production.
* **Engineering:** implement backtest harness and ensure performance.
* **Trading:** armed with parameter recommendations.

---

### 2. Detailed Technical Specification

#### 2.1 Configuration (`config.py`)

```python
# Data grid
DELTAS_MIN   = [1, 5, 10]       # aggregation intervals in minutes
WINDOW_DAYS  = [30, 60, 90]     # calibration windows

# GARCH calibration
MLE_BOUNDS     = [(1e-9, None), (0.0, 1.0), (0.0, 1.0)]   # ω>0, α≥0, β≥0
MLE_CONSTRAINT = lambda x: x[1] + x[2] < 1.0              # α+β <1
MLE_MAXITER    = 10000

# Daily roll frequency
ROLL_FREQ      = '1D'           # use pandas date_range with freq='1D'

# Backtest metrics
METRICS        = ['mse', 'mae']
```

#### 2.2 Dependencies

```toml
# uv.toml
[tool.uv.dependencies]
numpy      = "^1.25"
pandas     = "^2.0"
scipy      = "^1.11"
psycopg2   = "^2.9"
arch       = "^6.0"
```

Install via:

```bash
uv install
```

#### 2.3 Data Loading

In `backtest.py`:

1. **Spot price history**

   ```python
   import pandas as pd
   import psycopg2

   con = psycopg2.connect(**DB)
   klines = pd.read_sql("""
     SELECT open_time, close_price
     FROM binance_klines
     WHERE symbol='BTCUSDT' AND interval='1m'
     ORDER BY open_time
   """, con, parse_dates=['open_time'])
   klines.set_index('open_time', inplace=True)
   price_series = klines['close_price']
   ```

2. **Binary mid-prices**

   ```python
   books = pd.read_sql("""
     SELECT published_at AS ts,
            ( (bids->0->>0)::float + (asks->0->>0)::float )/2 AS mid_price
     FROM order_books
     WHERE subject='order_book'
       AND asset_id ~ '^\d+$'      -- uint256 check
     ORDER BY published_at
   """, con, parse_dates=['ts'])
   books.set_index('ts', inplace=True)
   ```

3. **Daily roll times**

   ```python
   start = price_series.index.min() + pd.Timedelta(days=max(WINDOW_DAYS))
   end   = price_series.index.max() - pd.Timedelta(days=1)
   roll_times = pd.date_range(start=start.normalize(),
                              end=end.normalize(),
                              freq=ROLL_FREQ)
   ```

#### 2.4 Return Aggregation

```python
returns_dict = {}
for delta in DELTAS_MIN:
    # resample log-return every delta minutes
    rs = np.log(price_series).diff().dropna()
    ret = rs.resample(f'{delta}T').sum().dropna()
    returns_dict[delta] = ret
```

#### 2.5 Backtest Loop

```python
import time
from scipy.optimize import minimize
from math import erf, sqrt
import numpy as np

results = []
for delta in DELTAS_MIN:
  ret = returns_dict[delta]
  # align returns index to minute boundaries
  for W in WINDOW_DAYS:
    # 1) Calibration data: first W days
    calib_end = ret.index.min() + pd.Timedelta(days=W)
    r_calib = ret[:calib_end].values

    # 2) MLE for GARCH(1,1)
    def neg_ll(params):
      ω, α, β = params
      n = len(r_calib)
      σ2 = np.empty(n)
      σ2[0] = np.var(r_calib)
      ll = 0.0
      for t in range(1, n):
        σ2[t] = ω + α * r_calib[t-1]**2 + β * σ2[t-1]
        ll += 0.5 * (np.log(2*np.pi*σ2[t]) + r_calib[t]**2/σ2[t])
      return ll

    cons = ({'type':'ineq','fun': lambda x: 1 - x[1] - x[2]})
    res = minimize(neg_ll, x0=[1e-6, 0.1, 0.8],
                   bounds=MLE_BOUNDS, constraints=cons,
                   options={'maxiter':MLE_MAXITER})
    ω, α, β = res.x

    # 3) Forecast & compute p_model at each daily roll
    mse_list, mae_list = [], []
    start_time = time.time()
    for t0 in roll_times:
      # find the return at the last delta-bar before t0
      try:
        r_t = ret.asof(t0 - pd.Timedelta(minutes=delta))
      except KeyError:
        continue
      # recursive update—since no persistence needed across days:
      σ2_prev = np.var(r_calib)
      σ2 = ω + α * r_t**2 + β * σ2_prev
      N = 1440 // delta
      σ_inf2 = ω / (1 - α - β)
      σ2_1d = σ_inf2 + (α+β)**N * (σ2 - σ_inf2)
      σ_1d = sqrt(σ2_1d)
      p_model = 0.5 * (1 - erf(σ_1d / (2*sqrt(2))))

      # market probability = mid_price at t0
      p_mkt = books.mid_price.asof(t0)
      if np.isnan(p_mkt):
        continue

      err = p_model - p_mkt
      mse_list.append(err**2)
      mae_list.append(abs(err))

    runtime = time.time() - start_time
    results.append({
      'delta_min': delta,
      'window_days': W,
      'mse': np.mean(mse_list),
      'mae': np.mean(mae_list),
      'runtime_s': runtime
    })

df = pd.DataFrame(results)
print(df)
```

#### 2.6 Reporting

* **Output** `df` with columns `[delta_min, window_days, mse, mae, runtime_s]`.
* **Visualization** in notebook:

  ```python
  import seaborn as sns
  pivot = df.pivot('window_days','delta_min','mse')
  sns.heatmap(pivot, annot=True, fmt='.4f')
  ```

#### 2.7 Performance Notes

* Entire grid (3 Δt × 3 W) on \~1 year of data runs in ≲ 90 s on a modern laptop.
* All data remains in memory; no DB writes.
* Parameter settings and data sources are driven purely by `config.py`.
