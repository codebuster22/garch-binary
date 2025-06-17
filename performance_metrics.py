import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for GARCH models and binary option pricing.
    
    Provides statistical metrics, model diagnostics, and trading performance evaluation.
    """
    
    def __init__(self):
        """Initialize the performance evaluator."""
        pass
    
    def calculate_pricing_metrics(self,
                                model_predictions: np.ndarray,
                                market_prices: np.ndarray,
                                return_details: bool = False) -> Dict[str, float]:
        """
        Calculate comprehensive pricing performance metrics.
        
        Args:
            model_predictions: Model-generated probabilities/prices
            market_prices: Observed market mid-prices
            return_details: Whether to return detailed error statistics
            
        Returns:
            Dictionary with performance metrics
        """
        model_predictions = np.asarray(model_predictions)
        market_prices = np.asarray(market_prices)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(model_predictions) | np.isnan(market_prices))
        if not np.any(valid_mask):
            raise ValueError("No valid data points for metric calculation")
        
        model_clean = model_predictions[valid_mask]
        market_clean = market_prices[valid_mask]
        
        if len(model_clean) == 0:
            raise ValueError("No valid observations after cleaning")
        
        # Calculate errors
        errors = model_clean - market_clean
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        # Basic metrics
        metrics = {
            'n_observations': len(model_clean),
            'mse': np.mean(squared_errors),
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(squared_errors)),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors)
        }
        
        # Relative metrics
        if np.mean(market_clean) != 0:
            metrics['mape'] = np.mean(abs_errors / np.abs(market_clean)) * 100  # Mean Absolute Percentage Error
        else:
            metrics['mape'] = np.nan
        
        # Correlation metrics
        if len(model_clean) > 1:
            metrics['correlation'] = np.corrcoef(model_clean, market_clean)[0, 1]
            metrics['r_squared'] = metrics['correlation'] ** 2
            
            # Spearman rank correlation (non-parametric)
            metrics['spearman_correlation'] = stats.spearmanr(model_clean, market_clean)[0]
        else:
            metrics['correlation'] = np.nan
            metrics['r_squared'] = np.nan
            metrics['spearman_correlation'] = np.nan
        
        # Directional accuracy (for binary options)
        if len(model_clean) > 1:
            # Check if model and market move in same direction
            model_changes = np.diff(model_clean)
            market_changes = np.diff(market_clean)
            same_direction = np.sign(model_changes) == np.sign(market_changes)
            metrics['directional_accuracy'] = np.mean(same_direction[~np.isnan(same_direction)])
        else:
            metrics['directional_accuracy'] = np.nan
        
        # Quantile-based metrics
        metrics['error_5_percentile'] = np.percentile(errors, 5)
        metrics['error_95_percentile'] = np.percentile(errors, 95)
        metrics['error_iqr'] = np.percentile(errors, 75) - np.percentile(errors, 25)
        
        # Additional details if requested
        if return_details:
            metrics['min_error'] = np.min(errors)
            metrics['max_error'] = np.max(errors)
            metrics['skewness'] = stats.skew(errors)
            metrics['kurtosis'] = stats.kurtosis(errors)
            metrics['mean_model_prediction'] = np.mean(model_clean)
            metrics['mean_market_price'] = np.mean(market_clean)
            metrics['std_model_prediction'] = np.std(model_clean)
            metrics['std_market_price'] = np.std(market_clean)
        
        return metrics
    
    def evaluate_garch_diagnostics(self,
                                 garch_model,
                                 returns: np.ndarray,
                                 confidence_level: float = 0.05) -> Dict[str, Union[float, bool]]:
        """
        Perform GARCH model diagnostic tests.
        
        Args:
            garch_model: Fitted GARCH model object
            returns: Original return series
            confidence_level: Significance level for tests
            
        Returns:
            Dictionary with diagnostic test results
        """
        if not hasattr(garch_model, 'fitted') or not garch_model.fitted:
            raise ValueError("GARCH model must be fitted first")
        
        diagnostics = {}
        
        # Basic model information
        params = garch_model.get_parameters()
        diagnostics['persistence'] = params['alpha'] + params['beta']
        diagnostics['long_run_volatility'] = np.sqrt(params['omega'] / (1 - diagnostics['persistence']))
        diagnostics['unconditional_volatility'] = np.std(returns)
        
        # Parameter stability
        diagnostics['stationarity_condition'] = diagnostics['persistence'] < 1.0
        diagnostics['parameter_constraints_satisfied'] = (
            params['omega'] > 0 and 
            params['alpha'] >= 0 and 
            params['beta'] >= 0 and
            diagnostics['persistence'] < 1.0
        )
        
        # Standardized residuals tests
        if hasattr(garch_model, 'standardized_residuals') and garch_model.standardized_residuals is not None:
            residuals = garch_model.standardized_residuals
            
            # Jarque-Bera normality test
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            diagnostics['jarque_bera_statistic'] = jb_stat
            diagnostics['jarque_bera_pvalue'] = jb_pvalue
            diagnostics['normality_rejected'] = jb_pvalue < confidence_level
            
            # Ljung-Box test for serial correlation in residuals
            diagnostics.update(self._ljung_box_test(residuals, lags=10))
            
            # Ljung-Box test for serial correlation in squared residuals (ARCH effects)
            diagnostics.update(self._ljung_box_test(residuals**2, lags=10, prefix='squared_'))
            
            # Basic residual statistics
            diagnostics['residual_mean'] = np.mean(residuals)
            diagnostics['residual_std'] = np.std(residuals)
            diagnostics['residual_skewness'] = stats.skew(residuals)
            diagnostics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        return diagnostics
    
    def _ljung_box_test(self, data: np.ndarray, lags: int = 10, prefix: str = '') -> Dict[str, float]:
        """Perform Ljung-Box test for serial correlation."""
        try:
            # Simple implementation of Ljung-Box test
            n = len(data)
            autocorrs = []
            
            for lag in range(1, lags + 1):
                if lag < n:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)
                    else:
                        autocorrs.append(0.0)
                else:
                    autocorrs.append(0.0)
            
            # Ljung-Box statistic
            lb_stat = n * (n + 2) * sum([(autocorrs[i]**2) / (n - i - 1) for i in range(len(autocorrs))])
            lb_pvalue = 1 - stats.chi2.cdf(lb_stat, lags)
            
            return {
                f'{prefix}ljung_box_statistic': lb_stat,
                f'{prefix}ljung_box_pvalue': lb_pvalue,
                f'{prefix}serial_correlation_rejected': lb_pvalue < 0.05
            }
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {str(e)}")
            return {
                f'{prefix}ljung_box_statistic': np.nan,
                f'{prefix}ljung_box_pvalue': np.nan,
                f'{prefix}serial_correlation_rejected': False
            }
    
    def evaluate_forecast_accuracy(self,
                                 forecasts: np.ndarray,
                                 realized_values: np.ndarray,
                                 forecast_horizon: int = 1) -> Dict[str, float]:
        """
        Evaluate volatility forecasting accuracy.
        
        Args:
            forecasts: Model volatility forecasts
            realized_values: Realized volatility measures
            forecast_horizon: Forecast horizon (days)
            
        Returns:
            Dictionary with forecast evaluation metrics
        """
        forecasts = np.asarray(forecasts)
        realized_values = np.asarray(realized_values)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(forecasts) | np.isnan(realized_values))
        if not np.any(valid_mask):
            raise ValueError("No valid data points for forecast evaluation")
        
        forecasts_clean = forecasts[valid_mask]
        realized_clean = realized_values[valid_mask]
        
        if len(forecasts_clean) == 0:
            raise ValueError("No valid forecasts after cleaning")
        
        # Forecast errors
        errors = forecasts_clean - realized_clean
        
        metrics = {
            'n_forecasts': len(forecasts_clean),
            'forecast_horizon': forecast_horizon,
            'mse': np.mean(errors**2),
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'forecast_bias': np.mean(errors) / np.mean(realized_clean) if np.mean(realized_clean) != 0 else np.nan
        }
        
        # Directional accuracy for volatility forecasts
        if len(forecasts_clean) > 1:
            forecast_changes = np.diff(forecasts_clean)
            realized_changes = np.diff(realized_clean)
            same_direction = np.sign(forecast_changes) == np.sign(realized_changes)
            metrics['directional_accuracy'] = np.mean(same_direction[~np.isnan(same_direction)])
        else:
            metrics['directional_accuracy'] = np.nan
        
        # Correlation metrics
        if len(forecasts_clean) > 1:
            metrics['correlation'] = np.corrcoef(forecasts_clean, realized_clean)[0, 1]
            metrics['r_squared'] = metrics['correlation'] ** 2
        else:
            metrics['correlation'] = np.nan
            metrics['r_squared'] = np.nan
        
        return metrics
    
    def compare_models(self,
                     results_list: List[Dict],
                     model_names: List[str] = None,
                     primary_metric: str = 'mse') -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_list: List of result dictionaries from different models
            model_names: Names for the models
            primary_metric: Primary metric for ranking
            
        Returns:
            DataFrame with model comparison
        """
        if not results_list:
            raise ValueError("No results provided for comparison")
        
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(results_list))]
        
        if len(model_names) != len(results_list):
            raise ValueError("Number of model names must match number of results")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for i, (result, name) in enumerate(zip(results_list, model_names)):
            row = {'model_name': name}
            
            # Extract key metrics
            key_metrics = ['mse', 'mae', 'rmse', 'correlation', 'r_squared', 
                          'directional_accuracy', 'n_observations']
            
            for metric in key_metrics:
                if metric in result:
                    row[metric] = result[metric]
                else:
                    row[metric] = np.nan
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by primary metric (lower is better for error metrics)
        if primary_metric in comparison_df.columns:
            ascending = primary_metric in ['mse', 'mae', 'rmse']  # Lower is better
            comparison_df = comparison_df.sort_values(primary_metric, ascending=ascending)
            comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def generate_performance_report(self,
                                  pricing_metrics: Dict,
                                  garch_diagnostics: Dict = None,
                                  forecast_metrics: Dict = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            pricing_metrics: Pricing performance metrics
            garch_diagnostics: GARCH model diagnostics (optional)
            forecast_metrics: Forecast accuracy metrics (optional)
            
        Returns:
            Formatted performance report string
        """
        report = []
        report.append("=" * 60)
        report.append("GARCH MODEL PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Pricing performance section
        report.append("\n📊 PRICING PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"Observations: {pricing_metrics.get('n_observations', 'N/A')}")
        report.append(f"MSE: {pricing_metrics.get('mse', np.nan):.6f}")
        report.append(f"MAE: {pricing_metrics.get('mae', np.nan):.6f}")
        report.append(f"RMSE: {pricing_metrics.get('rmse', np.nan):.6f}")
        report.append(f"Correlation: {pricing_metrics.get('correlation', np.nan):.4f}")
        report.append(f"R²: {pricing_metrics.get('r_squared', np.nan):.4f}")
        
        if 'directional_accuracy' in pricing_metrics:
            report.append(f"Directional Accuracy: {pricing_metrics['directional_accuracy']:.2%}")
        
        # GARCH diagnostics section
        if garch_diagnostics:
            report.append("\n🔧 GARCH MODEL DIAGNOSTICS:")
            report.append("-" * 30)
            report.append(f"Persistence (α+β): {garch_diagnostics.get('persistence', np.nan):.6f}")
            report.append(f"Stationarity: {'✅' if garch_diagnostics.get('stationarity_condition', False) else '❌'}")
            report.append(f"Parameter Constraints: {'✅' if garch_diagnostics.get('parameter_constraints_satisfied', False) else '❌'}")
            
            if 'jarque_bera_pvalue' in garch_diagnostics:
                jb_p = garch_diagnostics['jarque_bera_pvalue']
                normality = "✅ Normal" if jb_p > 0.05 else "❌ Non-normal"
                report.append(f"Residual Normality: {normality} (p={jb_p:.4f})")
            
            if 'ljung_box_pvalue' in garch_diagnostics:
                lb_p = garch_diagnostics['ljung_box_pvalue']
                serial_corr = "❌ Present" if lb_p < 0.05 else "✅ Absent"
                report.append(f"Serial Correlation: {serial_corr} (p={lb_p:.4f})")
        
        # Forecast accuracy section
        if forecast_metrics:
            report.append("\n📈 FORECAST ACCURACY:")
            report.append("-" * 30)
            report.append(f"Forecast Horizon: {forecast_metrics.get('forecast_horizon', 'N/A')} days")
            report.append(f"Number of Forecasts: {forecast_metrics.get('n_forecasts', 'N/A')}")
            report.append(f"Forecast MSE: {forecast_metrics.get('mse', np.nan):.6f}")
            report.append(f"Forecast MAE: {forecast_metrics.get('mae', np.nan):.6f}")
            report.append(f"Forecast Correlation: {forecast_metrics.get('correlation', np.nan):.4f}")
            
            if 'directional_accuracy' in forecast_metrics:
                report.append(f"Forecast Directional Accuracy: {forecast_metrics['directional_accuracy']:.2%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def calculate_model_comparison_statistics(results_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate comprehensive statistics for model comparison.
    
    Args:
        results_df: DataFrame with backtest results for different parameter combinations
        
    Returns:
        Dictionary with comparison statistics
    """
    if results_df.empty:
        return {'error': 'No results provided'}
    
    successful_results = results_df[results_df['success'] == True]
    
    if successful_results.empty:
        return {'error': 'No successful results to analyze'}
    
    stats = {}
    
    # Overall statistics
    stats['overall'] = {
        'total_combinations': len(results_df),
        'successful_combinations': len(successful_results),
        'success_rate': len(successful_results) / len(results_df),
        'mean_mse': successful_results['mse'].mean(),
        'std_mse': successful_results['mse'].std(),
        'min_mse': successful_results['mse'].min(),
        'max_mse': successful_results['mse'].max(),
        'median_mse': successful_results['mse'].median()
    }
    
    # Statistics by delta
    if 'delta_min' in successful_results.columns:
        stats['by_delta'] = {}
        for delta in successful_results['delta_min'].unique():
            delta_data = successful_results[successful_results['delta_min'] == delta]
            stats['by_delta'][delta] = {
                'count': len(delta_data),
                'mean_mse': delta_data['mse'].mean(),
                'std_mse': delta_data['mse'].std(),
                'min_mse': delta_data['mse'].min(),
                'mean_correlation': delta_data['correlation'].mean() if 'correlation' in delta_data.columns else np.nan
            }
    
    # Statistics by window
    if 'window_days' in successful_results.columns:
        stats['by_window'] = {}
        for window in successful_results['window_days'].unique():
            window_data = successful_results[successful_results['window_days'] == window]
            stats['by_window'][window] = {
                'count': len(window_data),
                'mean_mse': window_data['mse'].mean(),
                'std_mse': window_data['mse'].std(),
                'min_mse': window_data['mse'].min(),
                'mean_correlation': window_data['correlation'].mean() if 'correlation' in window_data.columns else np.nan
            }
    
    return stats 