import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import List, Tuple, Dict

class HarmonicGradientResonance:
    """
    Harmonic-Gradient Resonance (HGR) Algorithm.
    
    A novel scientific forecasting approach that decomposes time series into:
    1. 'Resonance' (Deterministic Physics): Captured via Spectral Analysis & Harmonic Regression.
    2. 'Turbulence' (Stochastic Dynamics): Captured via Temporally-Weighted Gradient Boosting.
    
    This is NOT an ensemble. It is a dual-stage hybrid architecture.
    """
    def __init__(self, top_k_frequencies: int = 5, turbulence_learning_rate: float = 0.05):
        self.top_k = top_k_frequencies
        self.turbulence_lr = turbulence_learning_rate
        
        # Resonance Component
        self.resonance_model = Ridge(alpha=0.5)
        self.dominant_periods = []
        
        # Turbulence Component
        self.turbulence_model = xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=self.turbulence_lr, 
            max_depth=6,
            objective='reg:squarederror'
        )
        self.scaler = StandardScaler()
        
    def _extract_dominant_frequencies(self, y: np.array) -> List[float]:
        """
        Performs Fast Fourier Transform (FFT) to find the 'Resonance' of the series.
        """
        n = len(y)
        yf = fft(y)
        xf = fftfreq(n, 1)  # Assuming daily frequency (1) for simplicity, relative unit
        
        # Get magnitudes, ignore DC component (0 freq)
        magnitudes = np.abs(yf[:n//2])
        frequencies = xf[:n//2]
        
        # Sort by magnitude (skip index 0)
        sorted_indices = np.argsort(magnitudes[1:])[::-1]
        
        # Get top K periods (1/freq)
        top_periods = []
        for idx in sorted_indices[:self.top_k]:
            freq = frequencies[idx+1]
            if freq > 0:
                top_periods.append(1/freq)
                
        return top_periods

    def _generate_harmonic_features(self, n_samples: int, start_idx: int = 0) -> np.ndarray:
        """
        Generates Sine/Cosine waves for the dominant periods.
        """
        t = np.arange(start_idx, start_idx + n_samples)
        features = []
        for period in self.dominant_periods:
            w = 2 * np.pi / period
            features.append(np.sin(w * t))
            features.append(np.cos(w * t))
        return np.column_stack(features)

    def _generate_turbulence_features(self, residuals: pd.Series, lags: int = 7) -> pd.DataFrame:
        """
        Generates features for the stochastic component.
        """
        df = pd.DataFrame({'r': residuals})
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['r'].shift(i)
        
        df['rolling_mean'] = df['r'].rolling(window=3).mean()
        df['rolling_std'] = df['r'].rolling(window=3).std()
        return df.dropna()

    def fit(self, df: pd.DataFrame, target_col: str):
        y = df[target_col].values
        n = len(y)
        
        # --- Stage 1: Resonance Learning ---
        self.dominant_periods = self._extract_dominant_frequencies(y)
        X_harm = self._generate_harmonic_features(n)
        
        self.resonance_model.fit(X_harm, y)
        y_resonance = self.resonance_model.predict(X_harm)
        
        # --- Stage 2: Turbulence Learning ---
        residuals = y - y_resonance
        
        # Create lag features for residuals
        X_turb_df = self._generate_turbulence_features(pd.Series(residuals))
        y_turb = residuals[X_turb_df.index]
        X_turb = X_turb_df.values
        
        # Temporal Weighting: Give more weight to recent data
        # Weight grows linearly from 0.5 to 1.5
        sample_weights = np.linspace(0.5, 1.5, len(y_turb))
        
        self.turbulence_model.fit(X_turb, y_turb, sample_weight=sample_weights)
        
        # Store last residuals for recursive prediction
        self.last_residuals = residuals[-10:] # Keep enough history
        
        return self

    def predict(self, steps: int, last_df: pd.DataFrame = None) -> pd.DataFrame:
        # --- Predict Resonance ---
        # We need to know the absolute time index relative to training start
        # Assuming steps continue immediately after training
        # In a real scenario, we'd track the global time index. 
        # Here we approximate by continuing t from fit.
        
        # Re-generate training harmonics to get the length
        # (In production, store n_train)
        n_train = len(last_df)
        X_harm_future = self._generate_harmonic_features(steps, start_idx=n_train)
        pred_resonance = self.resonance_model.predict(X_harm_future)
        
        # --- Predict Turbulence (Recursive) ---
        pred_turbulence = []
        current_residuals = list(self.last_residuals)
        
        for _ in range(steps):
            # Build single row feature
            # We need to reconstruct the exact features used in _generate_turbulence_features
            # Lags:
            feats = []
            for i in range(1, 8): # lags 1..7
                feats.append(current_residuals[-i])
            
            # Rolling mean/std (window 3)
            recent_3 = current_residuals[-3:]
            feats.append(np.mean(recent_3))
            feats.append(np.std(recent_3))
            
            # Predict
            feat_vector = np.array(feats).reshape(1, -1)
            next_res = self.turbulence_model.predict(feat_vector)[0]
            
            pred_turbulence.append(next_res)
            current_residuals.append(next_res)
            
        # --- Combine ---
        final_pred = pred_resonance + np.array(pred_turbulence)
        
        return pd.DataFrame({
            'resonance': pred_resonance,
            'turbulence': pred_turbulence,
            'hgr_forecast': final_pred
        })
