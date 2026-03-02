import pandas as pd
import numpy as np
import ta
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for time series forecasting.
    
    Feature categories:
    - Lag features (configurable)
    - Rolling statistics (multi-scale)
    - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
    - Calendar/temporal features
    - Fourier features for seasonality
    - Interaction features
    - Target statistics features
    """
    def __init__(
        self,
        lags: List[int] = None,
        rolling_windows: List[int] = None,
        fourier_order: int = 3,
        include_technical: bool = True,
        include_calendar: bool = True,
        include_fourier: bool = True,
    ):
        self.lags = lags or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.fourier_order = fourier_order
        self.include_technical = include_technical
        self.include_calendar = include_calendar
        self.include_fourier = include_fourier

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Generate comprehensive feature set.
        
        Returns DataFrame with all features appended.
        """
        df = df.copy()
        
        # 1. Lag features
        df = self._add_lag_features(df, target_col)
        
        # 2. Rolling statistics
        df = self._add_rolling_features(df, target_col)
        
        # 3. Technical Indicators
        if self.include_technical:
            df = self._add_technical_indicators(df, target_col)
        
        # 4. Calendar features
        if self.include_calendar:
            df = self._add_calendar_features(df)
        
        # 5. Fourier features
        if self.include_fourier:
            df = self._add_fourier_features(df)
        
        # 6. Target statistics
        df = self._add_target_stats(df, target_col)
            
        return df.dropna()
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add configurable lag features."""
        for lag in self.lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            
            # Lag differences
            if lag > 1:
                df[f'lag_diff_{lag}'] = df[target_col].diff(lag)
        
        # Percentage change lags
        for lag in [1, 7]:
            if lag in self.lags:
                df[f'pct_change_{lag}'] = df[target_col].pct_change(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add multi-scale rolling statistics."""
        for window in self.rolling_windows:
            if window >= len(df):
                continue
                
            roll = df[target_col].rolling(window=window)
            
            df[f'rolling_mean_{window}'] = roll.mean()
            df[f'rolling_std_{window}'] = roll.std()
            df[f'rolling_min_{window}'] = roll.min()
            df[f'rolling_max_{window}'] = roll.max()
            df[f'rolling_median_{window}'] = roll.median()
            
            # Rolling range (volatility proxy)
            df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
            
            # Rolling skewness
            if window >= 8:
                df[f'rolling_skew_{window}'] = roll.skew()
        
        # Exponential weighted moving averages
        for span in [7, 14, 30]:
            if span < len(df):
                df[f'ewma_{span}'] = df[target_col].ewm(span=span, min_periods=1).mean()
        
        # Distance from rolling stats (useful for mean-reversion detection)
        if 'rolling_mean_7' in df.columns:
            df['distance_from_ma7'] = df[target_col] - df['rolling_mean_7']
        if 'rolling_mean_30' in df.columns:
            df['distance_from_ma30'] = df[target_col] - df['rolling_mean_30']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add technical analysis indicators."""
        try:
            series = df[target_col]
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(series, window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(series)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(series, window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_pct'] = bb.bollinger_pband()
            
            # Stochastic Oscillator (using close=target as proxy)
            if len(series) >= 14:
                df['stoch_k'] = ta.momentum.StochasticOscillator(
                    high=series, low=series, close=series, window=14
                ).stoch()
            
            # Rate of Change
            df['roc_5'] = ta.momentum.ROCIndicator(series, window=5).roc()
            df['roc_10'] = ta.momentum.ROCIndicator(series, window=10).roc()
            
        except Exception as e:
            logger.warning(f"Could not compute some technical indicators: {e}")
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/calendar features from datetime index."""
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            
            df['hour'] = idx.hour
            df['dayofweek'] = idx.dayofweek
            df['dayofmonth'] = idx.day
            df['dayofyear'] = idx.dayofyear
            df['month'] = idx.month
            df['quarter'] = idx.quarter
            df['weekofyear'] = idx.isocalendar().week.values.astype(int)
            df['is_weekend'] = idx.dayofweek.isin([5, 6]).astype(int)
            df['is_month_start'] = idx.is_month_start.astype(int)
            df['is_month_end'] = idx.is_month_end.astype(int)
            df['is_quarter_start'] = idx.is_quarter_start.astype(int)
            df['is_quarter_end'] = idx.is_quarter_end.astype(int)
            
            # Cyclic encoding for temporal features
            df['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
            df['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
            
            if idx.hour.max() > 0:  # Has time component
                df['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier features for capturing multiple seasonal patterns.
        """
        n = len(df)
        t = np.arange(n, dtype=np.float64)
        
        # Standard seasonal periods to model
        periods = [7, 30.44, 91.31, 365.25]  # weekly, monthly, quarterly, yearly
        
        for period in periods:
            if n < 2 * period:
                continue
            for order in range(1, self.fourier_order + 1):
                col_sin = f'fourier_sin_p{int(period)}_o{order}'
                col_cos = f'fourier_cos_p{int(period)}_o{order}'
                df[col_sin] = np.sin(2 * np.pi * order * t / period)
                df[col_cos] = np.cos(2 * np.pi * order * t / period)
        
        return df
    
    def _add_target_stats(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add expanding (cumulative) target statistics."""
        series = df[target_col]
        
        # Expanding stats
        df['expanding_mean'] = series.expanding(min_periods=2).mean()
        df['expanding_std'] = series.expanding(min_periods=2).std()
        
        # Relative to expanding stats
        df['zscore_expanding'] = (series - df['expanding_mean']) / (df['expanding_std'] + 1e-10)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Return list of all generated feature names (excludes target)."""
        transformed = self.transform(df, target_col)
        return [col for col in transformed.columns if col != target_col]
    
    def feature_importance_summary(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Quick correlation-based feature importance.
        """
        transformed = self.transform(df, target_col)
        features = [col for col in transformed.columns if col != target_col]
        
        correlations = {}
        for feat in features:
            try:
                corr = transformed[feat].corr(transformed[target_col])
                correlations[feat] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlations[feat] = 0.0
        
        importance_df = pd.DataFrame(
            list(correlations.items()),
            columns=['feature', 'abs_correlation']
        ).sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        
        return importance_df
