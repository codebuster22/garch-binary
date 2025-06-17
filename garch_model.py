import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from typing import Tuple, Optional, Dict, Any
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GARCHModel:
    """
    GARCH(1,1) volatility model implementation with MLE estimation.
    
    Model specification:
    Returns: r_t = μ + ε_t, where ε_t ~ N(0, σ²_t)
    Volatility: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    Parameter constraints:
    - ω > 0 (long-term volatility component)
    - α ≥ 0 (ARCH coefficient)
    - β ≥ 0 (GARCH coefficient)  
    - α + β < 1 (stationarity condition)
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
        self.log_likelihood = None
        self.convergence_info = None
        self.conditional_volatility = None
        self.standardized_residuals = None
        
    def fit(self, returns: np.ndarray, method: str = 'MLE', 
            max_iter: int = 10000, tol: float = 1e-8) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) model to return series using Maximum Likelihood Estimation.
        
        Args:
            returns: Array of log returns
            method: Estimation method ('MLE' only for now)
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            
        Returns:
            Dictionary with estimation results
        """
        if len(returns) < 10:
            raise ValueError("Need at least 10 observations to fit GARCH model")
            
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) < 10:
            raise ValueError("Insufficient valid observations after removing NaN values")
        
        logger.info(f"Fitting GARCH(1,1) model to {len(returns)} returns")
        
        try:
            # Get initial parameter estimates
            initial_params = self._get_initial_params(returns)
            
            # Define bounds for parameters
            bounds = [
                (1e-8, None),    # ω > 0
                (0.0, 1.0),      # 0 ≤ α ≤ 1
                (0.0, 1.0)       # 0 ≤ β ≤ 1
            ]
            
            # Stationarity constraint: α + β < 1
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}
            ]
            
            # Optimize log-likelihood
            result = minimize(
                fun=self._negative_log_likelihood,
                x0=initial_params,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': tol}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
                # Try with different initial values
                result = self._try_alternative_optimization(returns, bounds, constraints, max_iter, tol)
            
            # Store results
            self.params = {
                'omega': result.x[0],
                'alpha': result.x[1], 
                'beta': result.x[2],
                'mu': np.mean(returns)  # Mean return
            }
            
            self.log_likelihood = -result.fun
            self.convergence_info = result
            self.fitted = True
            
            # Calculate conditional volatilities and residuals
            self._calculate_conditional_volatility(returns)
            
            # Log results
            logger.info(f"GARCH(1,1) estimation completed:")
            logger.info(f"  ω = {self.params['omega']:.6f}")
            logger.info(f"  α = {self.params['alpha']:.6f}")
            logger.info(f"  β = {self.params['beta']:.6f}")
            logger.info(f"  μ = {self.params['mu']:.6f}")
            logger.info(f"  Log-likelihood = {self.log_likelihood:.2f}")
            logger.info(f"  α + β = {self.params['alpha'] + self.params['beta']:.6f}")
            
            return {
                'params': self.params,
                'log_likelihood': self.log_likelihood,
                'convergence': result.success,
                'message': result.message,
                'n_iterations': result.nit if hasattr(result, 'nit') else None
            }
            
        except Exception as e:
            logger.error(f"GARCH model fitting failed: {e}")
            raise
    
    def _get_initial_params(self, returns: np.ndarray) -> np.ndarray:
        """Get reasonable initial parameter estimates."""
        var_returns = np.var(returns)
        
        # Initial values based on typical GARCH parameters
        alpha_init = 0.05
        beta_init = 0.90
        omega_init = var_returns * (1 - alpha_init - beta_init)
        
        # Ensure omega is positive
        omega_init = max(omega_init, 1e-6)
        
        return np.array([omega_init, alpha_init, beta_init])
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for GARCH(1,1) model.
        
        Args:
            params: [ω, α, β]
            returns: Return series
            
        Returns:
            Negative log-likelihood value
        """
        omega, alpha, beta = params
        
        # Parameter validation
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10  # Return large value for invalid parameters
        
        n = len(returns)
        
        # Initialize conditional variance
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Unconditional variance as starting value
        
        # Calculate log-likelihood
        log_likelihood = 0.0
        
        try:
            for t in range(1, n):
                # GARCH(1,1) volatility equation
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
                
                # Ensure positive variance
                if sigma2[t] <= 0:
                    return 1e10
                
                # Log-likelihood contribution
                log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + 
                                         returns[t]**2 / sigma2[t])
        
        except (FloatingPointError, OverflowError):
            return 1e10
        
        # Return negative log-likelihood (for minimization)
        return -log_likelihood
    
    def _try_alternative_optimization(self, returns: np.ndarray, bounds: list, 
                                    constraints: list, max_iter: int, tol: float):
        """Try alternative optimization with different starting points."""
        logger.info("Trying alternative optimization strategies...")
        
        # Try multiple starting points
        starting_points = [
            [1e-6, 0.01, 0.95],
            [1e-5, 0.1, 0.8],
            [1e-4, 0.2, 0.7],
            [np.var(returns) * 0.1, 0.05, 0.9]
        ]
        
        best_result = None
        best_ll = -np.inf
        
        for start_params in starting_points:
            try:
                result = minimize(
                    fun=self._negative_log_likelihood,
                    x0=start_params,
                    args=(returns,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': max_iter, 'ftol': tol}
                )
                
                if result.success and -result.fun > best_ll:
                    best_result = result
                    best_ll = -result.fun
                    
            except Exception as e:
                logger.warning(f"Alternative optimization failed with start {start_params}: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All optimization attempts failed")
        
        return best_result
    
    def _calculate_conditional_volatility(self, returns: np.ndarray):
        """Calculate conditional volatilities and standardized residuals."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        n = len(returns)
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        
        # Calculate conditional variances
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        self.conditional_volatility = np.sqrt(sigma2)
        self.standardized_residuals = returns / self.conditional_volatility
    
    def forecast_volatility(self, horizon: int = 1, last_return: float = None, 
                          last_volatility: float = None) -> np.ndarray:
        """
        Forecast volatility for given horizon.
        
        Args:
            horizon: Forecast horizon (number of periods)
            last_return: Last observed return (if None, uses last from fitted data)
            last_volatility: Last conditional volatility (if None, uses last from fitted data)
            
        Returns:
            Array of volatility forecasts
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        
        # Use last values from fitted data if not provided
        if last_return is None:
            if self.standardized_residuals is None:
                raise ValueError("No fitted data available for forecasting")
            last_return = self.standardized_residuals[-1] * self.conditional_volatility[-1]
        
        if last_volatility is None:
            if self.conditional_volatility is None:
                raise ValueError("No fitted data available for forecasting")
            last_volatility = self.conditional_volatility[-1]
        
        # Forecast volatility
        forecasts = np.zeros(horizon)
        
        # One-step ahead forecast
        forecasts[0] = np.sqrt(omega + alpha * last_return**2 + beta * last_volatility**2)
        
        # Multi-step ahead forecasts
        if horizon > 1:
            # Long-run variance
            long_run_var = omega / (1 - alpha - beta)
            
            for h in range(1, horizon):
                # Multi-step ahead forecast using persistence
                forecasts[h] = np.sqrt(long_run_var + (alpha + beta)**h * 
                                     (forecasts[0]**2 - long_run_var))
        
        return forecasts
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostic statistics.
        
        Returns:
            Dictionary with diagnostic statistics
        """
        if not self.fitted or self.standardized_residuals is None:
            raise ValueError("Model must be fitted first")
        
        residuals = self.standardized_residuals
        
        # Basic statistics
        diagnostics = {
            'n_observations': len(residuals),
            'log_likelihood': self.log_likelihood,
            'aic': -2 * self.log_likelihood + 2 * 3,  # 3 parameters
            'bic': -2 * self.log_likelihood + np.log(len(residuals)) * 3,
            'persistence': self.params['alpha'] + self.params['beta']
        }
        
        # Residual statistics
        diagnostics.update({
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': self._calculate_skewness(residuals),
            'residual_kurtosis': self._calculate_kurtosis(residuals)
        })
        
        return diagnostics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        skew = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate sample excess kurtosis."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        # Excess kurtosis (normal distribution has kurtosis = 3)
        kurt = np.sum(((data - mean) / std) ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
        kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurt
    
    def simulate(self, n_periods: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate GARCH(1,1) process.
        
        Args:
            n_periods: Number of periods to simulate
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (returns, volatilities)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        omega, alpha, beta = self.params['omega'], self.params['alpha'], self.params['beta']
        mu = self.params['mu']
        
        # Initialize arrays
        returns = np.zeros(n_periods)
        volatilities = np.zeros(n_periods)
        
        # Initial values
        volatilities[0] = np.sqrt(omega / (1 - alpha - beta))  # Long-run volatility
        eps = np.random.normal(0, 1)
        returns[0] = mu + volatilities[0] * eps
        
        # Simulate the process
        for t in range(1, n_periods):
            # Update volatility
            volatilities[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatilities[t-1]**2)
            
            # Generate return
            eps = np.random.normal(0, 1)
            returns[t] = mu + volatilities[t] * eps
        
        return returns, volatilities
    
    def get_parameters(self) -> Dict[str, float]:
        """Get fitted parameters."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        return self.params.copy()
    
    def summary(self) -> str:
        """Return a summary string of the fitted model."""
        if not self.fitted:
            return "GARCH(1,1) model - Not fitted"
        
        diagnostics = self.get_model_diagnostics()
        
        summary = f"""
GARCH(1,1) Model Summary
========================
Parameters:
  ω (omega): {self.params['omega']:.6f}
  α (alpha): {self.params['alpha']:.6f}
  β (beta):  {self.params['beta']:.6f}
  μ (mu):    {self.params['mu']:.6f}

Model Statistics:
  Observations: {diagnostics['n_observations']}
  Log-likelihood: {diagnostics['log_likelihood']:.2f}
  AIC: {diagnostics['aic']:.2f}
  BIC: {diagnostics['bic']:.2f}
  Persistence (α+β): {diagnostics['persistence']:.6f}

Residual Diagnostics:
  Mean: {diagnostics['residual_mean']:.6f}
  Std: {diagnostics['residual_std']:.6f}
  Skewness: {diagnostics['residual_skewness']:.4f}
  Excess Kurtosis: {diagnostics['residual_kurtosis']:.4f}
"""
        return summary 