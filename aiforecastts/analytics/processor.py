import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, Tuple

class DataProcessor:
    """
    Advanced data processing for scientific time series research.
    """
    @staticmethod
    def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        # Handle missing values using interpolation for time series
        df[target_col] = df[target_col].interpolate(method='time')
        df = df.dropna()
        return df

    @staticmethod
    def check_stationarity(series: pd.Series) -> dict:
        result = adfuller(series.dropna())
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }

    @staticmethod
    def decompose_series(series: pd.Series, model='additive', period=None):
        return seasonal_decompose(series, model=model, period=period)
