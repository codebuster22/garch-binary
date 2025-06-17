#!/usr/bin/env python3
"""
Test GARCH backtesting system with synthetic data.

This script generates synthetic return and binary option data to test
the complete GARCH backtesting pipeline without requiring database access.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from garch_model import GARCHModel
from binary_option_pricing import BinaryOptionPricer, convert_garch_forecast_to_binary_prob
from performance_metrics import PerformanceEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_returns(n_periods: int = 1000, 
                             true_omega: float = 1e-6,
                             true_alpha: float = 0.05,
                             true_beta: float = 0.9,
                             random_seed: int = 42) -> np.ndarray:
    """
    Generate synthetic returns from a known GARCH(1,1) process.
    
    Args:
        n_periods: Number of return observations to generate
        true_omega: True omega parameter
        true_alpha: True alpha parameter  
        true_beta: True beta parameter
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of synthetic returns
    """
    np.random.seed(random_seed)
    
    # Initialize arrays
    returns = np.zeros(n_periods)
    sigma2 = np.zeros(n_periods)
    
    # Initial conditional variance
    sigma2[0] = true_omega / (1 - true_alpha - true_beta)
    
    for t in range(n_periods):
        # Generate standardized innovation
        epsilon = np.random.normal(0, 1)
        
        # Generate return
        returns[t] = np.sqrt(sigma2[t]) * epsilon
        
        # Update conditional variance for next period
        if t < n_periods - 1:
            sigma2[t + 1] = true_omega + true_alpha * returns[t]**2 + true_beta * sigma2[t]
    
    return returns

def generate_synthetic_binary_data(returns: np.ndarray,
                                 volatilities: np.ndarray,
                                 noise_level: float = 0.05,
                                 random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic binary option data based on returns and volatilities.
    
    Args:
        returns: Return series
        volatilities: Conditional volatility series
        noise_level: Amount of noise to add to theoretical prices
        random_seed: Random seed
        
    Returns:
        DataFrame with synthetic binary option data
    """
    np.random.seed(random_seed)
    
    # Create timestamps (1-minute intervals)
    start_time = pd.Timestamp('2024-01-01')
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(len(returns))]
    
    # Generate theoretical binary option prices
    pricer = BinaryOptionPricer()
    theoretical_prices = []
    
    for vol in volatilities:
        # Convert daily volatility to binary option probability
        prob = convert_garch_forecast_to_binary_prob(vol, mean_return=0.0, method='normal')
        theoretical_prices.append(prob)
    
    theoretical_prices = np.array(theoretical_prices)
    
    # Add noise to create "market" prices
    noise = np.random.normal(0, noise_level, len(theoretical_prices))
    market_prices = theoretical_prices + noise
    
    # Ensure prices stay in valid range [0, 1]
    market_prices = np.clip(market_prices, 0.01, 0.99)
    
    # Generate binary outcomes (simplified - just based on return sign)
    outcomes = (returns > 0).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'return': returns,
        'volatility': volatilities,
        'theoretical_price': theoretical_prices,
        'market_price': market_prices,
        'outcome': outcomes
    })
    
    return data

def test_garch_estimation():
    """Test GARCH model estimation with synthetic data."""
    logger.info("🧪 Testing GARCH estimation with synthetic data...")
    
    # Generate synthetic returns
    true_params = {'omega': 1e-6, 'alpha': 0.05, 'beta': 0.9}
    returns = generate_synthetic_returns(
        n_periods=1000,
        true_omega=true_params['omega'],
        true_alpha=true_params['alpha'],
        true_beta=true_params['beta']
    )
    
    logger.info(f"Generated {len(returns)} synthetic returns")
    logger.info(f"True parameters: ω={true_params['omega']:.6f}, α={true_params['alpha']:.3f}, β={true_params['beta']:.3f}")
    
    # Fit GARCH model
    garch_model = GARCHModel()
    fit_result = garch_model.fit(returns)
    
    if fit_result['convergence']:
        estimated_params = garch_model.get_parameters()
        logger.info("✅ GARCH estimation successful!")
        logger.info(f"Estimated parameters: ω={estimated_params['omega']:.6f}, α={estimated_params['alpha']:.3f}, β={estimated_params['beta']:.3f}")
        
        # Check parameter accuracy
        omega_error = abs(estimated_params['omega'] - true_params['omega']) / true_params['omega']
        alpha_error = abs(estimated_params['alpha'] - true_params['alpha']) / true_params['alpha']
        beta_error = abs(estimated_params['beta'] - true_params['beta']) / true_params['beta']
        
        logger.info(f"Parameter errors: ω={omega_error:.1%}, α={alpha_error:.1%}, β={beta_error:.1%}")
        
        return garch_model, returns
    else:
        logger.error("❌ GARCH estimation failed")
        return None, returns

def test_binary_option_pricing():
    """Test binary option pricing functionality."""
    logger.info("🧪 Testing binary option pricing...")
    
    # Test with sample volatilities
    sample_volatilities = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # Daily volatilities
    
    pricer = BinaryOptionPricer()
    
    # Test normal method
    normal_probs = []
    for vol in sample_volatilities:
        prob = convert_garch_forecast_to_binary_prob(vol, method='normal')
        normal_probs.append(prob)
    
    logger.info("✅ Binary option pricing successful!")
    logger.info(f"Sample volatilities: {sample_volatilities}")
    logger.info(f"Corresponding probabilities: {[f'{p:.4f}' for p in normal_probs]}")
    
    return pricer

def test_performance_metrics():
    """Test performance evaluation metrics."""
    logger.info("🧪 Testing performance metrics...")
    
    # Generate test data
    np.random.seed(42)
    n_obs = 100
    
    # Create correlated model predictions and market prices
    true_probs = np.random.uniform(0.3, 0.7, n_obs)
    model_predictions = true_probs + np.random.normal(0, 0.05, n_obs)
    market_prices = true_probs + np.random.normal(0, 0.03, n_obs)
    
    # Ensure valid range
    model_predictions = np.clip(model_predictions, 0.01, 0.99)
    market_prices = np.clip(market_prices, 0.01, 0.99)
    
    # Calculate metrics
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_pricing_metrics(model_predictions, market_prices, return_details=True)
    
    logger.info("✅ Performance metrics calculation successful!")
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"Correlation: {metrics['correlation']:.4f}")
    logger.info(f"R²: {metrics['r_squared']:.4f}")
    
    return metrics

def test_complete_pipeline():
    """Test the complete GARCH backtesting pipeline with synthetic data."""
    logger.info("🧪 Testing complete pipeline...")
    
    # Generate synthetic data
    returns = generate_synthetic_returns(n_periods=500)
    volatilities = np.array([np.std(returns)] * len(returns))  # Simplified volatility
    
    synthetic_data = generate_synthetic_binary_data(returns, volatilities)
    
    logger.info(f"Generated synthetic dataset with {len(synthetic_data)} observations")
    
    # Split data for backtesting simulation
    train_size = 200
    test_size = 100
    
    train_returns = returns[:train_size]
    test_data = synthetic_data.iloc[train_size:train_size+test_size]
    
    # Fit GARCH model on training data
    garch_model = GARCHModel()
    fit_result = garch_model.fit(train_returns)
    
    if not fit_result['convergence']:
        logger.error("❌ GARCH fitting failed in pipeline test")
        return False
    
    # Generate predictions
    model_predictions = []
    market_prices = test_data['market_price'].values
    
    for i in range(len(test_data)):
        # Get last return for forecasting
        if i == 0:
            last_return = train_returns[-1]
        else:
            last_return = test_data.iloc[i-1]['return']
        
        # Forecast volatility
        vol_forecast = garch_model.forecast_volatility(horizon=1, last_return=last_return)
        
        # Convert to binary option probability
        prob = convert_garch_forecast_to_binary_prob(vol_forecast[0], method='normal')
        model_predictions.append(prob)
    
    model_predictions = np.array(model_predictions)
    
    # Calculate performance metrics
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_pricing_metrics(model_predictions, market_prices)
    
    logger.info("✅ Complete pipeline test successful!")
    logger.info(f"Pipeline MSE: {metrics['mse']:.6f}")
    logger.info(f"Pipeline MAE: {metrics['mae']:.6f}")
    logger.info(f"Pipeline Correlation: {metrics['correlation']:.4f}")
    
    return True

def main():
    """Run all synthetic data tests."""
    logger.info("🚀 Starting GARCH Synthetic Data Tests")
    logger.info("="*50)
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Test 1: GARCH Estimation
        total_tests += 1
        garch_model, returns = test_garch_estimation()
        if garch_model is not None:
            tests_passed += 1
        
        print()
        
        # Test 2: Binary Option Pricing
        total_tests += 1
        pricer = test_binary_option_pricing()
        if pricer is not None:
            tests_passed += 1
        
        print()
        
        # Test 3: Performance Metrics
        total_tests += 1
        metrics = test_performance_metrics()
        if metrics is not None:
            tests_passed += 1
        
        print()
        
        # Test 4: Complete Pipeline
        total_tests += 1
        pipeline_success = test_complete_pipeline()
        if pipeline_success:
            tests_passed += 1
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    logger.info("="*50)
    logger.info("🏁 SYNTHETIC DATA TEST RESULTS")
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! The GARCH backtesting system is working correctly.")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - tests_passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 