import pandas as pd
import numpy as np
from prophet import Prophet
from .hgr import HarmonicGradientResonance
from .statistical import AutoARIMAModel
from ..analytics.evaluator import ModelEvaluator
from typing import Dict, Any, List, Optional
import logging
import warnings

logger = logging.getLogger(__name__)


class ScientificEnsemble:
    """
    Scientific Ensemble Model v2.0.
    
    Combines up to 3 forecasting paradigms:
    1. Facebook Prophet (Bayesian trend/seasonality)
    2. HGR (Harmonic-Gradient Resonance) - Physics-aware model
    3. AutoARIMA (Classical statistical modeling)
    
    Features:
    - Automatic weight optimization via grid search or inverse-error weighting
    - Support for 2-model or 3-model ensembles
    - Stacking option with Ridge meta-learner
    - Comprehensive fit summary with per-model diagnostics
    """
    def __init__(
        self,
        weights: Dict[str, float] = None,
        include_arima: bool = True,
        stacking: bool = False,
    ):
        self.prophet = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        self.hgr = HarmonicGradientResonance()
        self.arima = AutoARIMAModel(seasonal=False) if include_arima else None
        self.include_arima = include_arima
        self.stacking = stacking
        
        self.is_fitted = False
        self.weights = weights or self._default_weights()
        self.fit_summary: Dict = {}
        self.target_col: str = ""
        
        # Stacking meta-learner
        self._meta_model = None

    def _default_weights(self) -> Dict[str, float]:
        if self.include_arima:
            return {'prophet': 0.35, 'hgr': 0.35, 'arima': 0.30}
        return {'prophet': 0.5, 'hgr': 0.5}

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
    def _optimize_weights_multi(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        resolution: int = 21,
    ) -> Dict[str, Any]:
        """
        Optimize weights for 2 or 3 models using grid search.
        """
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        best_weights = {name: 1.0 / n_models for name in model_names}
        best_rmse = float('inf')
        
        if n_models == 2:
            for w in np.linspace(0.0, 1.0, resolution):
                y_pred = w * predictions[model_names[0]] + (1 - w) * predictions[model_names[1]]
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = {model_names[0]: round(w, 3), model_names[1]: round(1 - w, 3)}
        
        elif n_models == 3:
            step = 1.0 / (resolution - 1) if resolution > 1 else 1.0
            for w1 in np.arange(0, 1.0 + step / 2, step):
                for w2 in np.arange(0, 1.0 - w1 + step / 2, step):
                    w3 = 1.0 - w1 - w2
                    if w3 < -0.01:
                        continue
                    w3 = max(0, w3)
                    y_pred = (
                        w1 * predictions[model_names[0]]
                        + w2 * predictions[model_names[1]]
                        + w3 * predictions[model_names[2]]
                    )
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = {
                            model_names[0]: round(float(w1), 3),
                            model_names[1]: round(float(w2), 3),
                            model_names[2]: round(float(w3), 3),
                        }
        
        return {**best_weights, 'rmse': round(float(best_rmse), 4)}

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        validation_size: int = 14,
        optimize_weights: bool = True,
    ):
        """
        Fits all component models and optimizes ensemble weights.
        """
        self.target_col = target_col
        df = df.copy()
        validation_size = max(0, int(validation_size))

        # Time-based split for validation
        if validation_size > 0 and len(df) > validation_size + 30:
            train_df = df.iloc[:-validation_size]
            val_df = df.iloc[-validation_size:]
        else:
            train_df = df
            val_df = None
        
        model_diagnostics = {}
        
        # 1. Fit Prophet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prophet_train = self._prepare_prophet_df(train_df, target_col)
            self.prophet.fit(prophet_train)
        model_diagnostics['prophet'] = 'fitted'
        
        # 2. Fit HGR
        self.hgr.fit(train_df, target_col)
        model_diagnostics['hgr'] = self.hgr.fit_diagnostics
        
        # 3. Fit ARIMA (optional)
        if self.include_arima and self.arima is not None:
            try:
                self.arima.fit(train_df, target_col)
                model_diagnostics['arima'] = self.arima.fit_diagnostics
            except Exception as e:
                logger.warning(f"ARIMA fitting failed: {e}. Falling back to 2-model ensemble.")
                self.include_arima = False
                self.arima = None
                self.weights = {'prophet': 0.5, 'hgr': 0.5}

        # 4. Optimize weights on validation if available
        if val_df is not None and optimize_weights:
            val_preds = {}
            y_true = val_df[target_col].values
            
            # Prophet validation
            future = self.prophet.make_future_dataframe(periods=len(val_df))
            prophet_pred = self.prophet.predict(future).tail(len(val_df))['yhat'].values
            val_preds['prophet'] = prophet_pred
            
            # HGR validation
            hgr_pred = self.hgr.predict(steps=len(val_df), last_df=train_df)['hgr_forecast'].values
            val_preds['hgr'] = hgr_pred
            
            # ARIMA validation
            if self.include_arima and self.arima is not None:
                arima_pred = self.arima.predict(steps=len(val_df))['arima_forecast'].values
                val_preds['arima'] = arima_pred
            
            # Optimize
            opt = self._optimize_weights_multi(y_true, val_preds)
            opt_rmse = opt.pop('rmse')
            self.weights = opt
            
            self.fit_summary['validation'] = {
                'validation_size': len(val_df),
                'optimized_weights': self.weights,
                'validation_rmse': opt_rmse,
            }

        # 5. Refit on full data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prophet = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,
            )
            prophet_full = self._prepare_prophet_df(df, target_col)
            self.prophet.fit(prophet_full)
        
        self.hgr.fit(df, target_col)
        
        if self.include_arima and self.arima is not None:
            try:
                self.arima.fit(df, target_col)
            except Exception:
                pass
        
        self.is_fitted = True
        self.fit_summary['models'] = model_diagnostics
        self.fit_summary['ensemble_weights'] = self.weights
        self.fit_summary['n_models'] = 3 if self.include_arima else 2
        
        return self

    def predict(self, steps: int, last_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast with weighted ensemble of all component models.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")

        # 1. Prophet
        future = self.prophet.make_future_dataframe(periods=steps)
        prophet_raw = self.prophet.predict(future).tail(steps)
        prophet_vals = prophet_raw['yhat'].values
        prophet_dates = prophet_raw['ds'].values
        
        # 2. HGR
        hgr_out = self.hgr.predict(steps, last_df)
        hgr_vals = hgr_out['hgr_forecast'].values
        
        result = {
            'date': prophet_dates,
            'prophet': prophet_vals,
            'hgr': hgr_vals,
        }
        
        # 3. ARIMA (optional)
        if self.include_arima and self.arima is not None and self.arima.is_fitted:
            arima_out = self.arima.predict(steps=steps)
            arima_vals = arima_out['arima_forecast'].values
            result['arima'] = arima_vals
            
            final_preds = (
                prophet_vals * self.weights.get('prophet', 0.35)
                + hgr_vals * self.weights.get('hgr', 0.35)
                + arima_vals * self.weights.get('arima', 0.30)
            )
        else:
            final_preds = (
                prophet_vals * self.weights.get('prophet', 0.5)
                + hgr_vals * self.weights.get('hgr', 0.5)
            )
        
        result['ensemble_forecast'] = final_preds
        return pd.DataFrame(result)

    def evaluate(self, y_true, y_pred):
        return ModelEvaluator.calculate_metrics(y_true, y_pred)
    
    def summary(self) -> Dict:
        """Return comprehensive ensemble summary."""
        if not self.is_fitted:
            return {"status": "Not fitted"}
        return {
            "model": "Scientific Ensemble v2.0",
            "status": "Fitted",
            **self.fit_summary,
        }


class SuperAlgorithm(ScientificEnsemble):
    """Legacy alias for ScientificEnsemble."""
    pass
