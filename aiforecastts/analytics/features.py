import pandas as pd
import numpy as np
import ta
from typing import List

class FeatureEngineer:
    """
    Advanced feature engineering for time series forecasting.
    """
    def __init__(self, lags: List[int] = [1, 7, 14, 30]):
        self.lags = lags

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Lag features
        for lag in self.lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            
        # 2. Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            
        # 3. Technical Indicators (for financial/scientific TS)
        df['rsi'] = ta.momentum.RSIIndicator(df[target_col]).rsi()
        df['macd'] = ta.trend.MACD(df[target_col]).macd()
        
        # 4. Calendar features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
            
        return df.dropna()
