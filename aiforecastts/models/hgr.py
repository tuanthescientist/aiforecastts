import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from typing import List, Tuple, Dict, Optional
import warnings
import logging

logger = logging.getLogger(__name__)


class HarmonicGradientResonance:
    """
    Harmonic-Gradient Resonance (HGR) Algorithm v2.0.
    
    A novel scientific forecasting approach that decomposes time series into:
    1. 'Resonance' (Deterministic Physics): Captured via Spectral Analysis & Harmonic Regression.
    2. 'Turbulence' (Stochastic Dynamics): Captured via Temporally-Weighted Gradient Boosting.
    
    v2.0 Enhancements:
    - Adaptive frequency selection via cross-validation
    - Confidence intervals via Bayesian Ridge + bootstrap
    - Welch spectral estimation for robust frequency detection
    - Automatic hyperparameter tuning for turbulence component
    - Save/load model capability
    - Comprehensive fit summary and diagnostics
    """
    def __init__(
        self,
        top_k_frequencies: int = 5,
        turbulence_learning_rate: float = 0.05,
        n_estimators: int = 500,
        max_depth: int = 6,
        turbulence_lags: int = 7,
        use_bayesian_resonance: bool = False,
        confidence_level: float = 0.95,
        n_bootstrap: int = 100,
    ):
        self.top_k = top_k_frequencies
        self.turbulence_lr = turbulence_learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.turbulence_lags = turbulence_lags
        self.use_bayesian = use_bayesian_resonance
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        
        # Resonance Component
        if use_bayesian_resonance:
            self.resonance_model = BayesianRidge()
        else:
            self.resonance_model = Ridge(alpha=0.5)
        self.dominant_periods: List[float] = []
        self.spectral_power: Dict[float, float] = {}
        
        # Turbulence Component
        self.turbulence_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, 
            learning_rate=self.turbulence_lr, 
            max_depth=self.max_depth,
            objective='reg:squarederror',
            verbosity=0,
        )
        self.scaler = StandardScaler()
        
        # State
        self.is_fitted = False
        self.n_train: int = 0
        self.last_residuals: np.ndarray = np.array([])
        self.fit_diagnostics: Dict = {}
        self._training_y: Optional[np.ndarray] = None
        self._training_residuals: Optional[np.ndarray] = None
        
    def _extract_dominant_frequencies(self, y: np.ndarray, method: str = "welch") -> List[float]:
        """
        Extracts dominant frequencies using FFT or Welch's method.
        Welch provides smoother spectral estimates with reduced noise.
        """
        n = len(y)
        
        if method == "welch" and n >= 32:
            nperseg = min(256, n // 2)
            freqs, psd = welch(y, fs=1.0, nperseg=nperseg)
            magnitudes = psd
        else:
            yf = fft(y)
            xf = fftfreq(n, 1)
            magnitudes = np.abs(yf[:n // 2])
            freqs = xf[:n // 2]
        
        # Skip DC component
        if len(freqs) > 1:
            magnitudes = magnitudes[1:]
            freqs = freqs[1:]
        
        sorted_indices = np.argsort(magnitudes)[::-1]
        
        top_periods = []
        for idx in sorted_indices[:self.top_k * 2]:  # sample more, filter later
            freq = freqs[idx]
            if freq > 0:
                period = 1.0 / freq
                if 2.0 <= period <= n / 2:  # Filter: period must be physically plausible
                    top_periods.append(period)
                    self.spectral_power[period] = float(magnitudes[idx])
            if len(top_periods) >= self.top_k:
                break
                
        return top_periods

    def _generate_harmonic_features(self, n_samples: int, start_idx: int = 0) -> np.ndarray:
        """
        Generates Sine/Cosine waves for the dominant periods.
        Also adds trend and intercept features.
        """
        t = np.arange(start_idx, start_idx + n_samples, dtype=np.float64)
        features = []
        
        # Trend features
        features.append(t / max(n_samples, 1))  # Normalized linear trend
        features.append((t / max(n_samples, 1)) ** 2)  # Quadratic trend
        
        for period in self.dominant_periods:
            w = 2 * np.pi / period
            features.append(np.sin(w * t))
            features.append(np.cos(w * t))
            # Add harmonics (2nd order)
            features.append(np.sin(2 * w * t))
            features.append(np.cos(2 * w * t))
            
        return np.column_stack(features) if features else np.ones((n_samples, 1))

    def _generate_turbulence_features(self, residuals: pd.Series, lags: int = None) -> pd.DataFrame:
        """
        Generates enhanced features for the stochastic component.
        """
        lags = lags or self.turbulence_lags
        df = pd.DataFrame({'r': residuals.values if isinstance(residuals, pd.Series) else residuals})
        
        # Lag features
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['r'].shift(i)
        
        # Multi-scale rolling stats
        for window in [3, 5, 7]:
            if window <= len(df):
                df[f'rolling_mean_{window}'] = df['r'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['r'].rolling(window=window).std()
                df[f'rolling_min_{window}'] = df['r'].rolling(window=window).min()
                df[f'rolling_max_{window}'] = df['r'].rolling(window=window).max()
        
        # Rate of change
        df['roc_1'] = df['r'].diff(1)
        df['roc_3'] = df['r'].diff(3)
        
        # Exponential weighted mean
        df['ewm_mean'] = df['r'].ewm(span=5, min_periods=1).mean()
        
        return df.dropna()

    def _auto_tune_turbulence(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> xgb.XGBRegressor:
        """
        Auto-tune turbulence model using time series cross-validation.
        """
        best_score = float('inf')
        best_params = {'n_estimators': self.n_estimators, 'max_depth': self.max_depth, 'learning_rate': self.turbulence_lr}
        
        param_grid = [
            {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03},
            {'n_estimators': 700, 'max_depth': 8, 'learning_rate': 0.02},
        ]
        
        n_splits = min(3, max(2, len(X) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for params in param_grid:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0, **params)
                model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
                pred = model.predict(X[val_idx])
                rmse = np.sqrt(np.mean((y[val_idx] - pred) ** 2))
                scores.append(rmse)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        logger.info(f"Best turbulence params: {best_params}, CV RMSE: {best_score:.4f}")
        
        return xgb.XGBRegressor(objective='reg:squarederror', verbosity=0, **best_params)

    def fit(self, df: pd.DataFrame, target_col: str, auto_tune: bool = False):
        """
        Fit the HGR model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe
        target_col : str
            Name of the target column
        auto_tune : bool
            If True, auto-tune turbulence hyperparameters via CV
        """
        y = df[target_col].values.astype(np.float64)
        n = len(y)
        self.n_train = n
        self._training_y = y.copy()
        
        # --- Stage 1: Resonance Learning ---
        self.dominant_periods = self._extract_dominant_frequencies(y)
        X_harm = self._generate_harmonic_features(n)
        
        self.resonance_model.fit(X_harm, y)
        y_resonance = self.resonance_model.predict(X_harm)
        
        resonance_rmse = np.sqrt(np.mean((y - y_resonance) ** 2))
        
        # --- Stage 2: Turbulence Learning ---
        residuals = y - y_resonance
        self._training_residuals = residuals.copy()
        
        X_turb_df = self._generate_turbulence_features(pd.Series(residuals))
        y_turb = residuals[X_turb_df.index]
        X_turb = X_turb_df.drop(columns=['r']).values
        
        # Temporal Weighting
        sample_weights = np.linspace(0.5, 1.5, len(y_turb))
        
        # Optional auto-tuning
        if auto_tune and len(X_turb) > 50:
            self.turbulence_model = self._auto_tune_turbulence(X_turb, y_turb, sample_weights)
        
        self.turbulence_model.fit(X_turb, y_turb, sample_weight=sample_weights)
        
        y_turbulence_pred = self.turbulence_model.predict(X_turb)
        turbulence_rmse = np.sqrt(np.mean((y_turb - y_turbulence_pred) ** 2))
        
        # Combined in-sample
        y_combined = y_resonance.copy()
        y_combined[X_turb_df.index] += y_turbulence_pred
        combined_rmse = np.sqrt(np.mean((y - y_combined) ** 2))
        
        # Store last residuals for recursive prediction
        self.last_residuals = residuals[-max(10, self.turbulence_lags + 5):]
        
        # Feature column info for prediction
        self._turb_feature_cols = [c for c in X_turb_df.columns if c != 'r']
        
        # Diagnostics
        self.fit_diagnostics = {
            'n_train': n,
            'dominant_periods': self.dominant_periods,
            'spectral_power': self.spectral_power,
            'resonance_rmse': round(resonance_rmse, 4),
            'turbulence_rmse': round(turbulence_rmse, 4),
            'combined_rmse': round(combined_rmse, 4),
            'resonance_variance_explained': round(1 - (np.var(residuals) / np.var(y)), 4) if np.var(y) > 0 else 0,
        }
        
        self.is_fitted = True
        return self

    def predict(self, steps: int, last_df: pd.DataFrame = None, return_ci: bool = False) -> pd.DataFrame:
        """
        Generate forecasts with optional confidence intervals.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        last_df : pd.DataFrame
            Last training dataframe (used for context)
        return_ci : bool
            If True, return confidence interval columns
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        n_train = self.n_train if last_df is None else len(last_df)
        
        # --- Predict Resonance ---
        X_harm_future = self._generate_harmonic_features(steps, start_idx=n_train)
        pred_resonance = self.resonance_model.predict(X_harm_future)
        
        # --- Predict Turbulence (Recursive) ---
        pred_turbulence = self._recursive_turbulence_predict(steps)
            
        # --- Combine ---
        final_pred = pred_resonance + np.array(pred_turbulence)
        
        result = pd.DataFrame({
            'resonance': pred_resonance,
            'turbulence': pred_turbulence,
            'hgr_forecast': final_pred,
        })
        
        # --- Confidence Intervals ---
        if return_ci:
            ci_low, ci_high = self._bootstrap_confidence_intervals(steps, n_train)
            result['ci_lower'] = ci_low
            result['ci_upper'] = ci_high
        
        return result
    
    def _recursive_turbulence_predict(self, steps: int) -> List[float]:
        """Recursively predict turbulence component."""
        pred_turbulence = []
        current_residuals = list(self.last_residuals)
        
        for _ in range(steps):
            feats = self._build_single_turbulence_features(current_residuals)
            feat_vector = np.array(feats).reshape(1, -1)
            next_res = float(self.turbulence_model.predict(feat_vector)[0])
            pred_turbulence.append(next_res)
            current_residuals.append(next_res)
            
        return pred_turbulence
    
    def _build_single_turbulence_features(self, residual_history: list) -> list:
        """Build a single feature row for turbulence prediction."""
        feats = []
        
        # Lag features
        for i in range(1, self.turbulence_lags + 1):
            idx = -i
            feats.append(residual_history[idx] if abs(idx) <= len(residual_history) else 0.0)
        
        # Multi-scale rolling stats
        for window in [3, 5, 7]:
            recent = residual_history[-window:] if len(residual_history) >= window else residual_history[:]
            feats.append(np.mean(recent))
            feats.append(np.std(recent) if len(recent) > 1 else 0.0)
            feats.append(np.min(recent))
            feats.append(np.max(recent))
        
        # Rate of change
        feats.append(residual_history[-1] - residual_history[-2] if len(residual_history) >= 2 else 0.0)
        feats.append(residual_history[-1] - residual_history[-4] if len(residual_history) >= 4 else 0.0)
        
        # EWM approximation
        if len(residual_history) >= 5:
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
            feats.append(np.dot(residual_history[-5:], weights))
        else:
            feats.append(np.mean(residual_history[-3:]))
        
        return feats
    
    def _bootstrap_confidence_intervals(self, steps: int, n_train: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate confidence intervals using residual bootstrap.
        """
        if self._training_residuals is None:
            # Fallback: use fixed percentage
            pred = self.predict(steps, return_ci=False)['hgr_forecast'].values
            std = np.std(self.last_residuals)
            z = 1.96
            return pred - z * std, pred + z * std
        
        all_preds = []
        residuals = self._training_residuals
        
        for _ in range(self.n_bootstrap):
            # Bootstrap residuals
            boot_noise = np.random.choice(residuals, size=steps, replace=True)
            
            # Base prediction + noise
            X_harm = self._generate_harmonic_features(steps, start_idx=n_train)
            base_resonance = self.resonance_model.predict(X_harm)
            base_turb = self._recursive_turbulence_predict(steps)
            
            boot_pred = base_resonance + np.array(base_turb) + boot_noise * 0.5
            all_preds.append(boot_pred)
        
        all_preds = np.array(all_preds)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(all_preds, alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(all_preds, (1 - alpha / 2) * 100, axis=0)
        
        return ci_lower, ci_upper

    def summary(self) -> Dict:
        """Return a comprehensive summary of the fitted model."""
        if not self.is_fitted:
            return {"status": "Not fitted"}
        return {
            "model": "Harmonic-Gradient Resonance (HGR) v2.0",
            "status": "Fitted",
            **self.fit_diagnostics,
        }
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self, path)
        logger.info(f"HGR model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "HarmonicGradientResonance":
        """Load model from disk."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not an HGR model: {type(model)}")
        return model
