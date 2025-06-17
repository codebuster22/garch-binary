import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

class BinaryOptionPricer:
    """
    Binary option pricing using GARCH volatility forecasts.
    
    Prices 24-hour binary options that settle based on whether Bitcoin price
    goes up or down from current time to 24 hours later.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the binary option pricer.
        
        Args:
            risk_free_rate: Risk-free rate for discounting (annualized)
        """
        self.risk_free_rate = risk_free_rate
    
    def price_binary_option(self, 
                          current_price: float,
                          volatility_forecast: Union[float, np.ndarray],
                          time_to_expiry: float = 1.0,  # 1 day in years
                          option_type: str = 'call') -> Union[float, np.ndarray]:
        """
        Price binary option using Black-Scholes style approach with GARCH volatility.
        
        For binary options that pay $1 if Bitcoin goes up, $0 if it goes down.
        
        Args:
            current_price: Current Bitcoin price
            volatility_forecast: GARCH volatility forecast (daily)
            time_to_expiry: Time to expiry in years (default 1 day = 1/365)
            option_type: 'call' (up) or 'put' (down)
            
        Returns:
            Binary option probability/price
        """
        if current_price <= 0:
            raise ValueError("Current price must be positive")
        
        if isinstance(volatility_forecast, (list, np.ndarray)):
            volatility_forecast = np.asarray(volatility_forecast)
        else:
            volatility_forecast = np.array([volatility_forecast])
        
        if np.any(volatility_forecast <= 0):
            raise ValueError("Volatility forecast must be positive")
        
        # For binary options, we assume log-normal price distribution
        # Strike price = current price (at-the-money binary)
        strike = current_price
        
        # Convert daily volatility to annualized volatility
        # volatility_forecast is daily, so annualize by sqrt(365)
        annual_vol = volatility_forecast * np.sqrt(365)
        
        # Black-Scholes d2 parameter for binary options
        # d2 = (ln(S/K) + (r - 0.5*σ²)*T) / (σ*√T)
        d2 = ((np.log(current_price / strike) + 
               (self.risk_free_rate - 0.5 * annual_vol**2) * time_to_expiry) / 
              (annual_vol * np.sqrt(time_to_expiry)))
        
        if option_type.lower() == 'call':
            # Probability that S_T > K (Bitcoin goes up)
            probability = norm.cdf(d2)
        elif option_type.lower() == 'put':
            # Probability that S_T < K (Bitcoin goes down)  
            probability = norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Apply discounting (though typically small for short-term options)
        discounted_price = probability * np.exp(-self.risk_free_rate * time_to_expiry)
        
        # Return scalar if input was scalar
        if len(discounted_price) == 1:
            return float(discounted_price[0])
        else:
            return discounted_price
    
    def garch_to_binary_probability(self,
                                  garch_volatility: Union[float, np.ndarray],
                                  mean_return: float = 0.0,
                                  horizon_days: float = 1.0) -> Union[float, np.ndarray]:
        """
        Convert GARCH volatility forecast to binary option probability using simplified approach.
        
        This assumes that log returns follow normal distribution with GARCH volatility.
        For binary option that pays if price goes up over horizon_days.
        
        Args:
            garch_volatility: GARCH volatility forecast (daily standard deviation)
            mean_return: Expected daily return (default 0 for martingale)
            horizon_days: Forecast horizon in days
            
        Returns:
            Probability that price goes up over the horizon
        """
        if isinstance(garch_volatility, (list, np.ndarray)):
            garch_volatility = np.asarray(garch_volatility)
        else:
            garch_volatility = np.array([garch_volatility])
        
        if np.any(garch_volatility <= 0):
            raise ValueError("GARCH volatility must be positive")
        
        # For multi-day horizon, scale volatility and return
        scaled_volatility = garch_volatility * np.sqrt(horizon_days)
        scaled_return = mean_return * horizon_days
        
        # Probability that log return > 0 (price goes up)
        # Under normal distribution: P(R > 0) = Φ(μ/σ) where Φ is CDF
        z_score = scaled_return / scaled_volatility
        probability = norm.cdf(z_score)
        
        # For martingale (μ=0), this simplifies to 0.5
        # But with non-zero volatility, we get slight deviations
        
        # Return scalar if input was scalar
        if len(probability) == 1:
            return float(probability[0])
        else:
            return probability
    
    def implied_volatility_from_price(self,
                                    binary_price: float,
                                    current_price: float,
                                    time_to_expiry: float = 1.0,
                                    option_type: str = 'call',
                                    max_iter: int = 100,
                                    tol: float = 1e-6) -> float:
        """
        Calculate implied volatility from binary option price using Newton-Raphson.
        
        Args:
            binary_price: Observed binary option price/probability
            current_price: Current underlying price
            time_to_expiry: Time to expiry in years
            option_type: 'call' or 'put'
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Implied daily volatility
        """
        if not (0 < binary_price < 1):
            raise ValueError("Binary price must be between 0 and 1")
        
        # Initial guess for volatility (20% annualized = ~1.26% daily)
        vol_guess = 0.20 / np.sqrt(365)
        
        for i in range(max_iter):
            # Calculate price and vega at current volatility guess
            price = self.price_binary_option(current_price, vol_guess, 
                                           time_to_expiry, option_type)
            
            # Calculate vega (derivative w.r.t. volatility) numerically
            vol_up = vol_guess * 1.001
            price_up = self.price_binary_option(current_price, vol_up,
                                              time_to_expiry, option_type)
            vega = (price_up - price) / (vol_up - vol_guess)
            
            if abs(vega) < 1e-10:
                logger.warning("Vega too small, implied volatility may be unstable")
                break
            
            # Newton-Raphson update
            price_diff = price - binary_price
            vol_new = vol_guess - price_diff / vega
            
            # Ensure positive volatility
            vol_new = max(vol_new, 1e-6)
            
            if abs(vol_new - vol_guess) < tol:
                return vol_new
            
            vol_guess = vol_new
        
        logger.warning(f"Implied volatility did not converge after {max_iter} iterations")
        return vol_guess
    
    def calculate_pricing_error(self,
                              model_probabilities: np.ndarray,
                              market_prices: np.ndarray,
                              error_type: str = 'mse') -> float:
        """
        Calculate pricing error between model and market.
        
        Args:
            model_probabilities: Model-generated probabilities
            market_prices: Market mid-prices
            error_type: 'mse', 'mae', or 'rmse'
            
        Returns:
            Pricing error
        """
        model_probabilities = np.asarray(model_probabilities)
        market_prices = np.asarray(market_prices)
        
        if len(model_probabilities) != len(market_prices):
            raise ValueError("Model probabilities and market prices must have same length")
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(model_probabilities) | np.isnan(market_prices))
        if not np.any(valid_mask):
            raise ValueError("No valid data points for error calculation")
        
        model_clean = model_probabilities[valid_mask]
        market_clean = market_prices[valid_mask]
        
        # Calculate errors
        errors = model_clean - market_clean
        
        if error_type.lower() == 'mse':
            return np.mean(errors**2)
        elif error_type.lower() == 'mae':
            return np.mean(np.abs(errors))
        elif error_type.lower() == 'rmse':
            return np.sqrt(np.mean(errors**2))
        else:
            raise ValueError("error_type must be 'mse', 'mae', or 'rmse'")
    
    def get_pricing_statistics(self,
                             model_probabilities: np.ndarray,
                             market_prices: np.ndarray) -> dict:
        """
        Get comprehensive pricing statistics.
        
        Args:
            model_probabilities: Model-generated probabilities
            market_prices: Market mid-prices
            
        Returns:
            Dictionary with pricing statistics
        """
        model_probabilities = np.asarray(model_probabilities)
        market_prices = np.asarray(market_prices)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(model_probabilities) | np.isnan(market_prices))
        model_clean = model_probabilities[valid_mask]
        market_clean = market_prices[valid_mask]
        
        if len(model_clean) == 0:
            raise ValueError("No valid data points")
        
        errors = model_clean - market_clean
        
        stats = {
            'n_observations': len(model_clean),
            'mse': np.mean(errors**2),
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'mean_model_prob': np.mean(model_clean),
            'mean_market_price': np.mean(market_clean),
            'correlation': np.corrcoef(model_clean, market_clean)[0, 1] if len(model_clean) > 1 else np.nan
        }
        
        return stats


def convert_garch_forecast_to_binary_prob(garch_model_forecast: float,
                                        mean_return: float = 0.0,
                                        method: str = 'normal') -> float:
    """
    Utility function to convert single GARCH volatility forecast to binary probability.
    
    Args:
        garch_model_forecast: Daily volatility forecast from GARCH model
        mean_return: Expected daily return
        method: Method for conversion ('normal' or 'black_scholes')
        
    Returns:
        Probability that Bitcoin price goes up in next 24 hours
    """
    pricer = BinaryOptionPricer()
    
    if method == 'normal':
        return pricer.garch_to_binary_probability(garch_model_forecast, mean_return)
    elif method == 'black_scholes':
        # Assume current price = 100 for relative calculation
        return pricer.price_binary_option(current_price=100.0, 
                                        volatility_forecast=garch_model_forecast,
                                        time_to_expiry=1/365)
    else:
        raise ValueError("method must be 'normal' or 'black_scholes'") 