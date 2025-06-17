import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import time
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import our modules
from garch_model import GARCHModel
from binary_option_pricing import BinaryOptionPricer, convert_garch_forecast_to_binary_prob
from data_loader import prepare_garch_backtest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GARCHBacktester:
    """
    Rolling window backtester for GARCH(1,1) model on binary options pricing.
    
    Uses existing data pipeline from data_loader.py to get returns and binary option data,
    then runs rolling window GARCH estimation and forecasting to generate model probabilities
    for comparison against market mid-prices.
    """
    
    def __init__(self, 
                 pricing_method: str = 'normal',
                 risk_free_rate: float = 0.0):
        """
        Initialize the backtester.
        
        Args:
            pricing_method: Method for converting volatility to probability ('normal' or 'black_scholes')
            risk_free_rate: Risk-free rate for binary option pricing
        """
        self.pricing_method = pricing_method
        self.pricer = BinaryOptionPricer(risk_free_rate=risk_free_rate)
        self.results = []
        
    def run_single_backtest(self, 
                          delta: int, 
                          window: int,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          limit: Optional[int] = None,
                          min_observations: int = 50) -> Dict:
        """
        Run backtest for single parameter combination (delta, window).
        
        Args:
            delta: Return aggregation interval in minutes (1, 5, or 10)
            window: GARCH calibration window in days (30, 60, or 90)
            start_date: Start date for backtest (ISO string)
            end_date: End date for backtest (ISO string)  
            limit: Limit on number of records to process
            min_observations: Minimum observations needed for GARCH estimation
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest: delta={delta}min, window={window}days")
        start_time = time.time()
        
        try:
            # Get prepared data using existing pipeline
            data = prepare_garch_backtest(delta=delta, window=window, limit=limit)
            
            if data.empty:
                logger.error(f"No data available for delta={delta}, window={window}")
                return self._create_error_result(delta, window, "No data available")
            
            logger.info(f"Loaded {len(data)} observations for backtesting")
            
            # Convert millisecond timestamps to pandas datetime for easier handling
            data_indexed = data.copy()
            data_indexed.index = pd.to_datetime(data_indexed.index, unit='ms')
            
            # Sort by timestamp
            data_indexed = data_indexed.sort_index()
            
            # Run rolling window backtesting
            backtest_results = self._run_rolling_backtest(data_indexed, window, min_observations)
            
            if not backtest_results['model_probs'] or not backtest_results['market_prices']:
                logger.warning(f"No valid predictions generated for delta={delta}, window={window}")
                return self._create_error_result(delta, window, "No valid predictions")
            
            # Calculate performance metrics
            model_probs = np.array(backtest_results['model_probs'])
            market_prices = np.array(backtest_results['market_prices'])
            
            mse = np.mean((model_probs - market_prices) ** 2)
            mae = np.mean(np.abs(model_probs - market_prices))
            rmse = np.sqrt(mse)
            
            # Additional statistics
            correlation = np.corrcoef(model_probs, market_prices)[0, 1] if len(model_probs) > 1 else np.nan
            mean_model_prob = np.mean(model_probs)
            mean_market_price = np.mean(market_prices)
            
            runtime = time.time() - start_time
            
            result = {
                'delta_min': delta,
                'window_days': window,
                'n_predictions': len(model_probs),
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'mean_model_prob': mean_model_prob,
                'mean_market_price': mean_market_price,
                'runtime_seconds': runtime,
                'success': True,
                'error_message': None,
                'convergence_rate': backtest_results['convergence_rate'],
                'avg_garch_params': backtest_results['avg_params']
            }
            
            logger.info(f"Backtest completed: delta={delta}min, window={window}days")
            logger.info(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, Predictions: {len(model_probs)}")
            logger.info(f"  Runtime: {runtime:.2f}s, Convergence rate: {backtest_results['convergence_rate']:.2%}")
            
            return result
            
        except Exception as e:
            runtime = time.time() - start_time
            error_msg = f"Backtest failed: {str(e)}"
            logger.error(f"delta={delta}, window={window}: {error_msg}")
            return self._create_error_result(delta, window, error_msg, runtime)
    
    def _run_rolling_backtest(self, data: pd.DataFrame, window: int, min_observations: int) -> Dict:
        """
        Run rolling window GARCH backtesting on the prepared data.
        
        Args:
            data: DataFrame with log_return_prev, mid_price, outcome columns, datetime index
            window: Rolling window size in days
            min_observations: Minimum observations for GARCH fitting
            
        Returns:
            Dictionary with backtesting results
        """
        model_probs = []
        market_prices = []
        garch_params_list = []
        convergence_count = 0
        total_windows = 0
        
        # Convert window from days to number of observations
        # Assuming roughly 1440 observations per day for 1-minute data, adjust for delta
        window_timedelta = pd.Timedelta(days=window)
        
        # Get unique dates for daily rolling
        start_date = data.index.min().date()
        end_date = data.index.max().date()
        
        # Generate rolling windows - fit on past window days, predict for next day
        current_date = start_date + timedelta(days=window)
        
        while current_date <= end_date:
            # Define window boundaries
            window_end = pd.Timestamp(current_date)
            window_start = window_end - window_timedelta
            
            # Get training data (past window days)
            train_mask = (data.index >= window_start) & (data.index < window_end)
            train_data = data[train_mask]
            
            # Get prediction target (current day)
            pred_date_start = pd.Timestamp(current_date)
            pred_date_end = pred_date_start + pd.Timedelta(days=1)
            pred_mask = (data.index >= pred_date_start) & (data.index < pred_date_end)
            pred_data = data[pred_mask]
            
            if len(train_data) < min_observations or len(pred_data) == 0:
                current_date += timedelta(days=1)
                continue
            
            try:
                total_windows += 1
                
                # Fit GARCH model on training data
                returns = train_data['log_return_prev'].values
                garch_model = GARCHModel()
                
                fit_result = garch_model.fit(returns, max_iter=5000, tol=1e-6)
                
                if fit_result['convergence']:
                    convergence_count += 1
                    
                    # Get latest return and volatility for forecasting
                    last_return = returns[-1] if len(returns) > 0 else 0.0
                    
                    # Forecast volatility for next period (1-day ahead)
                    vol_forecast = garch_model.forecast_volatility(horizon=1, last_return=last_return)
                    
                    # Convert GARCH volatility forecast to binary option probability
                    if self.pricing_method == 'normal':
                        model_prob = convert_garch_forecast_to_binary_prob(
                            vol_forecast[0], 
                            mean_return=garch_model.params['mu'],
                            method='normal'
                        )
                    else:
                        # Use Black-Scholes style pricing - need a reference price
                        # Use average mid_price from training data as proxy
                        ref_price = train_data['mid_price'].mean()
                        model_prob = self.pricer.price_binary_option(
                            current_price=ref_price,
                            volatility_forecast=vol_forecast[0],
                            time_to_expiry=1/365
                        )
                    
                    # Get market prices for this day
                    market_prices_day = pred_data['mid_price'].values
                    
                    # Use all observations from the prediction day
                    for market_price in market_prices_day:
                        if not np.isnan(market_price) and not np.isnan(model_prob):
                            model_probs.append(model_prob)
                            market_prices.append(market_price)
                    
                    # Store GARCH parameters for analysis
                    garch_params_list.append(garch_model.get_parameters())
                    
                else:
                    logger.warning(f"GARCH convergence failed for window ending {current_date}")
                
            except Exception as e:
                logger.warning(f"Error in rolling window {current_date}: {str(e)}")
                continue
            
            current_date += timedelta(days=1)
        
        # Calculate average parameters
        avg_params = {}
        if garch_params_list:
            for param_name in ['omega', 'alpha', 'beta', 'mu']:
                avg_params[param_name] = np.mean([p[param_name] for p in garch_params_list])
        
        convergence_rate = convergence_count / total_windows if total_windows > 0 else 0.0
        
        logger.info(f"Rolling backtest completed: {len(model_probs)} predictions from {total_windows} windows")
        logger.info(f"Convergence rate: {convergence_rate:.2%}")
        
        return {
            'model_probs': model_probs,
            'market_prices': market_prices,
            'convergence_rate': convergence_rate,
            'avg_params': avg_params,
            'n_windows': total_windows
        }
    
    def _create_error_result(self, delta: int, window: int, error_msg: str, runtime: float = 0.0) -> Dict:
        """Create error result dictionary."""
        return {
            'delta_min': delta,
            'window_days': window,
            'n_predictions': 0,
            'mse': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'correlation': np.nan,
            'mean_model_prob': np.nan,
            'mean_market_price': np.nan,
            'runtime_seconds': runtime,
            'success': False,
            'error_message': error_msg,
            'convergence_rate': 0.0,
            'avg_garch_params': {}
        }
    
    def run_parameter_grid_search(self,
                                deltas: List[int] = [1, 5, 10],
                                windows: List[int] = [30, 60, 90],
                                parallel: bool = True,
                                max_workers: Optional[int] = None,
                                **kwargs) -> pd.DataFrame:
        """
        Run backtest across parameter grid.
        
        Args:
            deltas: List of delta values (minutes) to test
            windows: List of window values (days) to test
            parallel: Whether to run in parallel
            max_workers: Max parallel workers (default: CPU count)
            **kwargs: Additional arguments passed to run_single_backtest
            
        Returns:
            DataFrame with results for all parameter combinations
        """
        # Generate parameter combinations
        param_combinations = [(delta, window) for delta in deltas for window in windows]
        
        logger.info(f"Starting parameter grid search: {len(param_combinations)} combinations")
        logger.info(f"Deltas: {deltas}, Windows: {windows}")
        logger.info(f"Parallel processing: {parallel}")
        
        results = []
        
        if parallel and len(param_combinations) > 1:
            # Use multiprocessing for parallel execution
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(param_combinations))
            
            logger.info(f"Using {max_workers} parallel workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(self._run_single_backtest_wrapper, delta, window, kwargs): (delta, window)
                    for delta, window in param_combinations
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    delta, window = future_to_params[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed: delta={delta}, window={window}")
                    except Exception as e:
                        logger.error(f"Failed: delta={delta}, window={window}: {str(e)}")
                        results.append(self._create_error_result(delta, window, str(e)))
        else:
            # Sequential execution
            for delta, window in param_combinations:
                result = self.run_single_backtest(delta, window, **kwargs)
                results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by MSE (best first) for easy analysis
        if 'mse' in results_df.columns:
            results_df = results_df.sort_values('mse', na_last=True)
        
        logger.info("Parameter grid search completed")
        logger.info(f"Results summary:")
        if len(results_df) > 0:
            successful = results_df['success'].sum()
            logger.info(f"  Successful runs: {successful}/{len(results_df)}")
            if successful > 0:
                best_mse = results_df[results_df['success']]['mse'].min()
                logger.info(f"  Best MSE: {best_mse:.6f}")
        
        return results_df
    
    def _run_single_backtest_wrapper(self, delta: int, window: int, kwargs: dict) -> dict:
        """Wrapper function for multiprocessing."""
        # Create new instance for each process to avoid shared state issues
        backtester = GARCHBacktester(
            pricing_method=self.pricing_method,
            risk_free_rate=self.pricer.risk_free_rate
        )
        return backtester.run_single_backtest(delta, window, **kwargs)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze backtest results and provide insights.
        
        Args:
            results_df: DataFrame from run_parameter_grid_search
            
        Returns:
            Dictionary with analysis insights
        """
        if results_df.empty:
            return {'error': 'No results to analyze'}
        
        successful_results = results_df[results_df['success'] == True]
        
        if successful_results.empty:
            return {'error': 'No successful backtests to analyze'}
        
        analysis = {
            'total_combinations': len(results_df),
            'successful_combinations': len(successful_results),
            'success_rate': len(successful_results) / len(results_df),
        }
        
        # Best performing combination
        best_idx = successful_results['mse'].idxmin()
        best_result = successful_results.loc[best_idx]
        
        analysis['best_combination'] = {
            'delta_min': best_result['delta_min'],
            'window_days': best_result['window_days'],
            'mse': best_result['mse'],
            'mae': best_result['mae'],
            'correlation': best_result['correlation'],
            'n_predictions': best_result['n_predictions']
        }
        
        # Performance by delta
        delta_performance = successful_results.groupby('delta_min')['mse'].agg(['mean', 'std', 'min', 'count'])
        analysis['performance_by_delta'] = delta_performance.to_dict()
        
        # Performance by window
        window_performance = successful_results.groupby('window_days')['mse'].agg(['mean', 'std', 'min', 'count'])
        analysis['performance_by_window'] = window_performance.to_dict()
        
        # Overall statistics
        analysis['overall_stats'] = {
            'mean_mse': successful_results['mse'].mean(),
            'std_mse': successful_results['mse'].std(),
            'mean_mae': successful_results['mae'].mean(),
            'mean_correlation': successful_results['correlation'].mean(),
            'mean_runtime': successful_results['runtime_seconds'].mean(),
            'mean_convergence_rate': successful_results['convergence_rate'].mean()
        }
        
        return analysis


def run_garch_backtest_study(deltas: List[int] = [1, 5, 10],
                           windows: List[int] = [30, 60, 90],
                           parallel: bool = True,
                           limit: Optional[int] = None,
                           pricing_method: str = 'normal') -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run complete GARCH backtest study.
    
    Args:
        deltas: List of return intervals to test (minutes)
        windows: List of calibration windows to test (days)
        parallel: Whether to use parallel processing
        limit: Limit on data records to process
        pricing_method: Method for binary option pricing
        
    Returns:
        Tuple of (results_df, analysis_dict)
    """
    logger.info("Starting GARCH backtest study")
    
    # Create backtester
    backtester = GARCHBacktester(pricing_method=pricing_method)
    
    # Run parameter grid search
    results_df = backtester.run_parameter_grid_search(
        deltas=deltas,
        windows=windows,
        parallel=parallel,
        limit=limit
    )
    
    # Analyze results
    analysis = backtester.analyze_results(results_df)
    
    logger.info("GARCH backtest study completed")
    
    return results_df, analysis 