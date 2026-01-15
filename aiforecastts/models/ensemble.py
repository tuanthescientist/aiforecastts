import pandas as pd
import numpy as np
from prophet import Prophet
from .hgr import HarmonicGradientResonance
from ..analytics.evaluator import ModelEvaluator
from typing import Dict, Any, List

class ScientificEnsemble:
    """
    Scientific Ensemble Model.
    Combines:
    1. Facebook Prophet (Robust trend/seasonality detection)
    2. HGR (Harmonic-Gradient Resonance) - Physics-aware model
    
    This ensemble averages the predictions of the industry-standard Prophet
    with our specialized HGR algorithm for maximum robustness.
    """
    def __init__(self, weights: Dict[str, float] = None):
        self.prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.hgr = HarmonicGradientResonance()
        self.is_fitted = False
        self.weights = weights or {'prophet': 0.5, 'hgr': 0.5}
        self.fit_summary = {}

    @staticmethod
    def _prepare_prophet_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            prophet_df = df.reset_index().rename(columns={df.index.name or 'index': 'ds'})
        else:
            date_col = None
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
            if date_col:
                prophet_df = df.rename(columns={date_col: 'ds'}).copy()
            else:
                prophet_df = df.copy()
                prophet_df['ds'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df[['ds', target_col]].rename(columns={target_col: 'y'})
        return prophet_df

    @staticmethod
    def _optimize_weights(y_true: np.ndarray, y_prophet: np.ndarray, y_hgr: np.ndarray) -> Dict[str, float]:
        best_w = 0.5
        best_rmse = float('inf')
        for w in np.linspace(0.0, 1.0, 21):
            y_pred = (w * y_prophet) + ((1 - w) * y_hgr)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w
        return {'prophet': round(float(best_w), 3), 'hgr': round(float(1 - best_w), 3), 'rmse': round(float(best_rmse), 4)}

    def fit(self, df: pd.DataFrame, target_col: str, validation_size: int = 14, optimize_weights: bool = True):
        """
        Fits both models.
        """
        self.target_col = target_col
        df = df.copy()
        validation_size = max(0, int(validation_size))

        # Time-based split for validation
        if validation_size > 0 and len(df) > validation_size:
            train_df = df.iloc[:-validation_size]
            val_df = df.iloc[-validation_size:]
        else:
            train_df = df
            val_df = None
        
        # 1. Fit Prophet
        prophet_train = self._prepare_prophet_df(train_df, target_col)
        self.prophet.fit(prophet_train)
        
        # 2. Fit HGR
        self.hgr.fit(train_df, target_col)

        # 3. Optimize weights on validation if available
        if val_df is not None and optimize_weights:
            # Prophet validation forecast
            future = self.prophet.make_future_dataframe(periods=len(val_df))
            prophet_pred = self.prophet.predict(future).tail(len(val_df))['yhat'].values

            # HGR validation forecast
            hgr_pred = self.hgr.predict(steps=len(val_df), last_df=train_df)['hgr_forecast'].values

            y_true = val_df[target_col].values
            opt = self._optimize_weights(y_true, prophet_pred, hgr_pred)
            self.weights = {'prophet': opt['prophet'], 'hgr': opt['hgr']}
            self.fit_summary = {
                'validation_size': len(val_df),
                'optimized_weights': self.weights,
                'validation_rmse': opt['rmse'],
            }

        # 4. Refit on full data for final model
        self.prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
        prophet_full = self._prepare_prophet_df(df, target_col)
        self.prophet.fit(prophet_full)
        self.hgr.fit(df, target_col)
        
        self.is_fitted = True
        return self

    def predict(self, steps: int, last_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasting with weighted ensemble.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")

        # 1. Prophet Forecast
        future = self.prophet.make_future_dataframe(periods=steps)
        prophet_raw = self.prophet.predict(future).tail(steps)
        prophet_vals = prophet_raw['yhat'].values
        prophet_dates = prophet_raw['ds'].values
        
        # 2. HGR Forecast
        hgr_out = self.hgr.predict(steps, last_df)
        hgr_vals = hgr_out['hgr_forecast'].values
        
        # 3. Combine
        final_preds = (prophet_vals * self.weights['prophet']) + (hgr_vals * self.weights['hgr'])
        
        return pd.DataFrame({
            'date': prophet_dates,
            'prophet': prophet_vals,
            'hgr': hgr_vals,
            'ensemble_forecast': final_preds
        })

    def evaluate(self, y_true, y_pred):
        return ModelEvaluator.calculate_metrics(y_true, y_pred)

class SuperAlgorithm(ScientificEnsemble):
    """Legacy alias"""
    pass
