import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy import stats
from typing import Optional, Tuple, Dict, List
import logging
import warnings

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Advanced data processing for scientific time series research.
    
    Capabilities:
    - Intelligent missing value handling (multiple strategies)
    - Outlier detection and treatment (IQR, Z-score, Modified Z-score)
    - Stationarity testing (ADF + KPSS for confirmatory analysis)
    - Automatic differencing
    - Normalization and scaling
    - Seasonal decomposition (classical + STL)
    - Data quality report
    """
    
    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        target_col: str,
        method: str = "interpolate",
        interpolation_method: str = "time",
    ) -> pd.DataFrame:
        """
        Handle missing values with multiple strategies.
        
        Parameters
        ----------
        method : str
            'interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop'
        interpolation_method : str
            Method for pd.interpolate() - 'time', 'linear', 'cubic', 'spline'
        """
        df = df.copy()
        
        if method == "interpolate":
            try:
                df[target_col] = df[target_col].interpolate(method=interpolation_method)
            except Exception:
                df[target_col] = df[target_col].interpolate(method='linear')
        elif method == "ffill":
            df[target_col] = df[target_col].ffill()
        elif method == "bfill":
            df[target_col] = df[target_col].bfill()
        elif method == "mean":
            df[target_col] = df[target_col].fillna(df[target_col].mean())
        elif method == "median":
            df[target_col] = df[target_col].fillna(df[target_col].median())
        elif method == "drop":
            df = df.dropna(subset=[target_col])
        
        # Final cleanup
        df = df.dropna(subset=[target_col])
        return df

    @staticmethod
    def check_stationarity(series: pd.Series, significance: float = 0.05) -> Dict:
        """
        Confirmatory stationarity analysis using both ADF and KPSS tests.
        
        ADF: H0 = unit root (non-stationary)
        KPSS: H0 = stationary
        
        Interpretation:
        - ADF rejects + KPSS fails to reject → Stationary
        - ADF fails to reject + KPSS rejects → Non-stationary
        - Both reject → Trend-stationary (difference needed)
        - Neither rejects → Inconclusive
        """
        series = series.dropna()
        
        # ADF test
        adf_result = adfuller(series)
        adf_stationary = adf_result[1] < significance
        
        # KPSS test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kpss_result = kpss(series, regression='c', nlags='auto')
                kpss_stationary = kpss_result[1] > significance
            except Exception:
                kpss_result = None
                kpss_stationary = None
        
        # Confirmatory interpretation
        if adf_stationary and kpss_stationary:
            conclusion = "Stationary"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "Non-stationary"
        elif adf_stationary and not kpss_stationary:
            conclusion = "Trend-stationary (consider differencing)"
        else:
            conclusion = "Inconclusive"
        
        result = {
            'adf_stat': round(adf_result[0], 4),
            'adf_p_value': round(adf_result[1], 4),
            'adf_stationary': adf_stationary,
            'critical_values': adf_result[4],
            'conclusion': conclusion,
        }
        
        if kpss_result:
            result['kpss_stat'] = round(kpss_result[0], 4)
            result['kpss_p_value'] = round(kpss_result[1], 4)
            result['kpss_stationary'] = kpss_stationary
        
        return result

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.Series:
        """
        Detect outliers using various methods.
        
        Parameters
        ----------
        method : str
            'iqr' (IQR), 'zscore', 'modified_zscore'
        threshold : float
            IQR multiplier (default 1.5) or Z-score threshold (default 3.0)
        
        Returns
        -------
        Boolean Series where True indicates an outlier
        """
        series = series.dropna()
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)
        
        elif method == "zscore":
            z = np.abs(stats.zscore(series))
            return pd.Series(z > (threshold if threshold > 1 else 3.0), index=series.index)
        
        elif method == "modified_zscore":
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / (mad + 1e-10)
            return pd.Series(np.abs(modified_z) > (threshold if threshold > 1 else 3.5), index=series.index)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def treat_outliers(
        df: pd.DataFrame,
        target_col: str,
        method: str = "clip",
        detection_method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Detect and treat outliers.
        
        Parameters
        ----------
        method : str
            'clip' (winsorize), 'interpolate', 'nan' (replace with NaN)
        """
        df = df.copy()
        outlier_mask = DataProcessor.detect_outliers(
            df[target_col], method=detection_method, threshold=threshold
        )
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.info(f"Found {n_outliers} outliers using {detection_method} method.")
        
        if method == "clip":
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            df[target_col] = df[target_col].clip(Q1 - threshold * IQR, Q3 + threshold * IQR)
        
        elif method == "interpolate":
            df.loc[outlier_mask, target_col] = np.nan
            df[target_col] = df[target_col].interpolate(method='linear')
        
        elif method == "nan":
            df.loc[outlier_mask, target_col] = np.nan
        
        return df
    
    @staticmethod
    def difference(series: pd.Series, order: int = 1, seasonal_period: int = None) -> pd.Series:
        """
        Apply differencing for stationarity.
        
        Parameters
        ----------
        order : int
            Number of regular differences
        seasonal_period : int, optional
            If provided, applies seasonal differencing
        """
        result = series.copy()
        
        # Regular differencing
        for _ in range(order):
            result = result.diff()
        
        # Seasonal differencing
        if seasonal_period:
            result = result.diff(seasonal_period)
        
        return result.dropna()
    
    @staticmethod
    def auto_difference(series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Automatically find the minimum differencing order to achieve stationarity.
        
        Returns
        -------
        Tuple of (differenced_series, differencing_order)
        """
        for d in range(max_diff + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
            
            result = adfuller(test_series.dropna())
            if result[1] < 0.05:
                return test_series, d
        
        return series.diff(max_diff).dropna(), max_diff
    
    @staticmethod
    def normalize(series: pd.Series, method: str = "minmax") -> Tuple[pd.Series, Dict]:
        """
        Normalize series values.
        
        Returns series and parameters needed for inverse transform.
        """
        if method == "minmax":
            min_val = series.min()
            max_val = series.max()
            normalized = (series - min_val) / (max_val - min_val + 1e-10)
            params = {"method": "minmax", "min": min_val, "max": max_val}
        
        elif method == "zscore":
            mean_val = series.mean()
            std_val = series.std()
            normalized = (series - mean_val) / (std_val + 1e-10)
            params = {"method": "zscore", "mean": mean_val, "std": std_val}
        
        elif method == "robust":
            median_val = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            normalized = (series - median_val) / (iqr + 1e-10)
            params = {"method": "robust", "median": median_val, "iqr": iqr}
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return normalized, params
    
    @staticmethod
    def inverse_normalize(series: pd.Series, params: Dict) -> pd.Series:
        """Inverse normalization using stored parameters."""
        method = params["method"]
        
        if method == "minmax":
            return series * (params["max"] - params["min"]) + params["min"]
        elif method == "zscore":
            return series * params["std"] + params["mean"]
        elif method == "robust":
            return series * params["iqr"] + params["median"]
        
        return series

    @staticmethod
    def decompose_series(
        series: pd.Series,
        model: str = 'additive',
        period: int = None,
        method: str = "classical",
    ):
        """
        Decompose time series using classical or STL method.
        
        Parameters
        ----------
        method : str
            'classical' or 'stl'
        """
        if method == "stl":
            if period is None:
                period = 7
            result = STL(series, period=period).fit()
        else:
            result = seasonal_decompose(series, model=model, period=period)
        
        return result
    
    @staticmethod
    def data_quality_report(df: pd.DataFrame, target_col: str) -> Dict:
        """
        Generate a comprehensive data quality report.
        """
        series = df[target_col]
        
        report = {
            'n_rows': len(df),
            'n_missing': int(series.isna().sum()),
            'pct_missing': round(series.isna().mean() * 100, 2),
            'n_duplicates': int(df.duplicated().sum()),
            'dtype': str(series.dtype),
            'stats': {
                'mean': round(float(series.mean()), 4),
                'std': round(float(series.std()), 4),
                'min': round(float(series.min()), 4),
                'max': round(float(series.max()), 4),
                'median': round(float(series.median()), 4),
                'skewness': round(float(series.skew()), 4),
                'kurtosis': round(float(series.kurtosis()), 4),
            },
            'n_zeros': int((series == 0).sum()),
            'n_negative': int((series < 0).sum()),
        }
        
        # Outlier count
        outliers = DataProcessor.detect_outliers(series.dropna())
        report['n_outliers_iqr'] = int(outliers.sum())
        
        # Date range if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            report['date_range'] = {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'frequency': pd.infer_freq(df.index),
            }
        
        return report
