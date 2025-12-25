import pandas as pd
import numpy as np
from prophet import Prophet
from pmdarima import auto_arima
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..analytics.features import FeatureEngineer
from typing import Dict, Any

class SuperAlgorithm:
    """
    The 'Super Algorithm': A hybrid ensemble of Prophet, AutoARIMA, and XGBoost.
    Designed for high-accuracy scientific forecasting.
    """
    def __init__(self):
        self.prophet_model = None
        self.arima_model = None
        self.xgb_model = None
        self.fe = FeatureEngineer()
        self.best_weights = None

    def fit(self, df: pd.DataFrame, target_col: str):
        """
        Train the hybrid ensemble.
        """
        # 1. Prophet
        prophet_df = df.reset_index().rename(columns={df.index.name or 'index': 'ds', target_col: 'y'})
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        self.prophet_model.fit(prophet_df)
        
        # 2. AutoARIMA
        self.arima_model = auto_arima(df[target_col], seasonal=True, m=7)
        
        # 3. XGBoost on residuals or features
        df_features = self.fe.transform(df, target_col)
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]
        
        self.xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5)
        self.xgb_model.fit(X, y)
        
        return self

    def predict(self, steps: int, last_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble predictions.
        """
        # Prophet forecast
        future = self.prophet_model.make_future_dataframe(periods=steps)
        prophet_pred = self.prophet_model.predict(future).tail(steps)['yhat'].values
        
        # ARIMA forecast
        arima_pred = self.arima_model.predict(n_periods=steps)
        
        # XGBoost forecast (simplified for this example - usually requires recursive prediction)
        # For now, let's use a simple average or weighted ensemble
        
        final_pred = (prophet_pred + arima_pred) / 2
        return pd.DataFrame({
            'prophet': prophet_pred,
            'arima': arima_pred,
            'ensemble': final_pred
        })

    def evaluate(self, y_true, y_pred):
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
