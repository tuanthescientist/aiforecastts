import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Optional, Tuple
import joblib
import logging
import warnings

logger = logging.getLogger(__name__)


class AutoARIMAModel:
    """
    AutoARIMA wrapper using pmdarima for automatic order selection.
    
    Automatically selects (p,d,q)(P,D,Q,m) parameters using:
    - Stepwise search with AIC/BIC optimization
    - Automatic stationarity and seasonality detection
    - Support for exogenous variables
    """
    def __init__(
        self,
        seasonal: bool = True,
        m: int = 1,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        information_criterion: str = "aic",
        stepwise: bool = True,
        suppress_warnings: bool = True,
    ):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        
        self.model = None
        self.is_fitted = False
        self.fit_diagnostics: Dict = {}
        self.target_col: str = ""
    
    def fit(self, df: pd.DataFrame, target_col: str, exog: pd.DataFrame = None):
        """
        Fit AutoARIMA model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target_col : str
            Target column name
        exog : pd.DataFrame, optional
            Exogenous variables
        """
        self.target_col = target_col
        y = df[target_col].values
        
        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            
            self.model = pm.auto_arima(
                y,
                exogenous=exog.values if exog is not None else None,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                information_criterion=self.information_criterion,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                error_action="ignore",
                trace=False,
            )
        
        self.is_fitted = True
        self.fit_diagnostics = {
            'order': self.model.order,
            'seasonal_order': self.model.seasonal_order if self.seasonal else None,
            'aic': round(self.model.aic(), 4),
            'bic': round(self.model.bic(), 4),
            'n_train': len(y),
        }
        
        logger.info(f"AutoARIMA fitted: order={self.model.order}, AIC={self.model.aic():.4f}")
        return self
    
    def predict(self, steps: int, last_df: pd.DataFrame = None, return_ci: bool = False, exog: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate forecasts with optional confidence intervals.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        preds, conf_int = self.model.predict(
            n_periods=steps,
            exogenous=exog.values if exog is not None else None,
            return_conf_int=True,
        )
        
        result = pd.DataFrame({
            'arima_forecast': preds,
        })
        
        if return_ci:
            result['ci_lower'] = conf_int[:, 0]
            result['ci_upper'] = conf_int[:, 1]
        
        return result
    
    def summary(self) -> Dict:
        if not self.is_fitted:
            return {"status": "Not fitted"}
        return {
            "model": "AutoARIMA",
            "status": "Fitted",
            **self.fit_diagnostics,
        }
    
    def save(self, path: str):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> "AutoARIMAModel":
        return joblib.load(path)


class ExponentialSmoothingModel:
    """
    Holt-Winters Exponential Smoothing wrapper.
    
    Supports:
    - Simple, Double (Holt), and Triple (Holt-Winters) exponential smoothing
    - Additive and Multiplicative seasonality
    - Automatic parameter optimization
    """
    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        use_boxcox: bool = False,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.fit_diagnostics: Dict = {}
        self.target_col: str = ""
    
    def fit(self, df: pd.DataFrame, target_col: str):
        """Fit Exponential Smoothing model."""
        self.target_col = target_col
        y = df[target_col].values
        
        # Auto-detect seasonal period if not provided
        if self.seasonal_periods is None and self.seasonal is not None:
            if isinstance(df.index, pd.DatetimeIndex):
                freq = pd.infer_freq(df.index)
                period_map = {'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'H': 24, 'T': 60}
                self.seasonal_periods = period_map.get(freq, 7)
            else:
                self.seasonal_periods = 7
        
        # Ensure enough data for seasonal period
        if self.seasonal is not None and self.seasonal_periods and len(y) < 2 * self.seasonal_periods:
            logger.warning("Not enough data for seasonal ETS, falling back to non-seasonal.")
            self.seasonal = None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
            )
            self.fitted_model = self.model.fit(optimized=True, use_brute=True)
        
        self.is_fitted = True
        
        # In-sample metrics
        fitted_values = self.fitted_model.fittedvalues
        residuals = y - fitted_values
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        self.fit_diagnostics = {
            'n_train': len(y),
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'aic': round(self.fitted_model.aic, 4) if hasattr(self.fitted_model, 'aic') else None,
            'rmse': round(rmse, 4),
        }
        
        return self
    
    def predict(self, steps: int, last_df: pd.DataFrame = None, return_ci: bool = False) -> pd.DataFrame:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        forecast = self.fitted_model.forecast(steps)
        
        result = pd.DataFrame({
            'ets_forecast': forecast.values if hasattr(forecast, 'values') else forecast,
        })
        
        if return_ci:
            # Approximate CI using in-sample residual std
            residuals = self.fitted_model.resid
            std = np.std(residuals)
            z = 1.96
            expanding_std = std * np.sqrt(np.arange(1, steps + 1))
            result['ci_lower'] = result['ets_forecast'] - z * expanding_std
            result['ci_upper'] = result['ets_forecast'] + z * expanding_std
        
        return result
    
    def summary(self) -> Dict:
        if not self.is_fitted:
            return {"status": "Not fitted"}
        return {
            "model": "Exponential Smoothing (ETS)",
            "status": "Fitted",
            **self.fit_diagnostics,
        }
    
    def save(self, path: str):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> "ExponentialSmoothingModel":
        return joblib.load(path)
