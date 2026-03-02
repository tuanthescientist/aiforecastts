# AIForecastTS v1.0.0: Scientific Framework for Time Series Resonance & Turbulence Analysis

[![PyPI version](https://badge.fury.io/py/aiforecastts.svg)](https://badge.fury.io/py/aiforecastts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 1. Abstract
AIForecastTS is a high-performance scientific library designed to bridge the gap between classical signal processing and modern machine learning for time series forecasting. 

**v1.0.0** is a major release introducing:
- **Harmonic-Gradient Resonance (HGR) v2.0** with confidence intervals, Bayesian resonance, and Welch spectral analysis
- **AutoARIMA & Exponential Smoothing (ETS)** models
- **3-Model Scientific Ensemble** (Prophet + HGR + AutoARIMA) with auto weight optimization
- **Walk-Forward Backtesting** framework with multi-model comparison
- **Professional Visualization** module (forecast plots, residual diagnostics, component decomposition)
- **Enhanced Analytics** (outlier detection, confirmatory stationarity testing, data quality reports)
- **Upgraded AI Agent** with structured scientific reports via Google Gemini

## 2. Theoretical Framework

### 2.1. Harmonic-Gradient Resonance (HGR) v2.0
Traditional forecasting often fails by treating signal and noise as a monolithic entity. HGR operates on the principle of **Dual-Component Decomposition**:

#### Resonance Module (Deterministic Physics)
The Resonance module assumes that every time series contains underlying periodicities driven by systemic cycles.
- **Spectral Identification**: Uses Welch's method (robust) or FFT to map the series into the frequency domain.
- **Harmonic Regression**: Selects the top $K$ dominant frequencies with 2nd-order harmonics and reconstructs the signal:
  $$y_{res}(t) = \beta_0 t + \beta_1 t^2 + \sum_{i=1}^{K} \sum_{j=1}^{2} [a_{ij} \sin(j\omega_i t) + b_{ij} \cos(j\omega_i t)]$$
- **v2.0**: Bayesian Ridge option for probabilistic resonance estimation

#### Turbulence Module (Stochastic Dynamics)
The residuals from the Resonance module ($y - y_{res}$) represent "Turbulence" — the chaotic, non-linear interactions.
- Multi-scale rolling statistics (windows: 3, 5, 7)
- Rate of change and exponential weighted features
- Temporal weighting with auto-tuned XGBoost
- **v2.0**: Confidence intervals via residual bootstrap

### 2.2. Scientific Ensemble v2.0
The ensemble now combines **3 forecasting paradigms**:

| Component | Paradigm | Strength |
|-----------|----------|----------|
| **Prophet** | Bayesian | Trend & seasonality |
| **HGR** | Physics-based | Frequency decomposition |
| **AutoARIMA** | Statistical | Classical Box-Jenkins |

Weight optimization uses 3D grid search on validation data. Falls back gracefully to 2-model ensemble if ARIMA fails.

### 2.3. AutoARIMA & ETS
- **AutoARIMA**: Automatic $(p,d,q)(P,D,Q,m)$ selection via stepwise AIC optimization (pmdarima)
- **ETS**: Holt-Winters with automatic seasonal period detection and parameter optimization

## 3. Architecture & Modules

### 3.1. Advanced Analytics (`aiforecastts.analytics`)
- **DataProcessor**: 
  - Multiple missing value strategies (interpolate, ffill, bfill, mean, median, drop)
  - Outlier detection (IQR, Z-score, Modified Z-score) and treatment (clip, interpolate)
  - Confirmatory stationarity testing (ADF + KPSS) with diagnostic conclusion
  - Auto-differencing, normalization (MinMax, Z-score, Robust)
  - Data quality reports
- **FeatureEngineer**: 
  - Multi-scale lag features with differences and pct_change
  - Rolling statistics with range, skewness, EWMA
  - Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ROC)
  - Cyclic calendar encoding (sin/cos for month, day-of-week, hour)
  - Fourier features for multiple seasonal periods
  - Correlation-based feature importance
- **ModelEvaluator**: 
  - Metrics: MAE, RMSE, MAPE, SMAPE, R², MedAE, MaxError
  - Directional Accuracy, Theil's U statistic
  - Residual diagnostics (normality, autocorrelation, skewness)
  - Multi-model comparison tables

### 3.2. Backtesting (`aiforecastts.backtesting`)
- Walk-forward validation (expanding/sliding window)
- Configurable forecast horizon and step size
- Multi-model comparison backtesting
- Per-fold and aggregate performance metrics

### 3.3. Visualization (`aiforecastts.visualization`)
- Forecast plots with confidence intervals
- HGR component decomposition (Resonance + Turbulence)
- Multi-model comparison plots
- 4-panel residual diagnostics (residuals, histogram, Q-Q, ACF)
- Backtest performance charts

### 3.4. AI Research Agent (`aiforecastts.agents`)
Powered by **Google Gemini**, the agent generates structured scientific reports with:
- Executive Summary
- Trend & Stationarity Analysis
- Forecast Interpretation with component analysis
- Confidence assessment and risk factors
- Actionable recommendations

## 4. Installation

```bash
pip install aiforecastts
```

For development:
```bash
pip install aiforecastts[dev]
```

## 5. Quick Start

### 5.1. Basic HGR Forecasting
```python
import pandas as pd
from aiforecastts import TimeSeriesResearch

df = pd.read_csv("data.csv", index_col='date', parse_dates=True)

research = TimeSeriesResearch(df, target_col='value', model_type='hgr')
results = research.run_full_analysis(steps=14, return_ci=True)

print(results['forecast'][['hgr_forecast', 'ci_lower', 'ci_upper']])
print(results['stationarity'])
print(results['data_quality'])
```

### 5.2. Scientific Ensemble (3-Model)
```python
research = TimeSeriesResearch(df, target_col='value', model_type='ensemble')
results = research.run_full_analysis(steps=7, treat_outliers=True)

print(results['forecast'][['prophet', 'hgr', 'arima', 'ensemble_forecast']])
print(results['model_summary'])
```

### 5.3. Model Comparison
```python
research = TimeSeriesResearch(df, target_col='value')
comparison = research.compare_models(model_types=['hgr', 'arima', 'ets'], steps=14)

print(comparison['comparison_table'])
```

### 5.4. Walk-Forward Backtesting
```python
research = TimeSeriesResearch(df, target_col='value', model_type='hgr')
bt_results = research.backtest(forecast_horizon=7, step_size=7, expanding=True)

print(f"Overall RMSE: {bt_results['overall_metrics']['RMSE']}")
print(f"Folds: {bt_results['n_folds']}")
```

### 5.5. Multi-Model Backtest Comparison
```python
bt_compare = research.backtest_compare(
    model_types=['hgr', 'arima'],
    forecast_horizon=7,
    step_size=7,
)
print(bt_compare['comparison'])
```

### 5.6. Visualization
```python
from aiforecastts import ForecastPlotter

# Forecast plot with CI
ForecastPlotter.plot_forecast(
    actual=df['value'],
    forecast=results['forecast'],
    title="HGR Forecast with Confidence Intervals",
    show_last_n=60,
)

# Component decomposition
ForecastPlotter.plot_components(results['forecast'])

# Residual diagnostics
ForecastPlotter.plot_residual_diagnostics(y_true, y_pred)

# Backtest results
ForecastPlotter.plot_backtest_results(bt_results)
```

### 5.7. AI Agent Analysis
```python
results = research.run_full_analysis(
    agent_query="Analyze the resonance-turbulence interaction and assess forecast stability.",
    api_key="YOUR_GEMINI_API_KEY",
    steps=7,
)
print(results['agent_report'])
```

### 5.8. Individual Model Usage
```python
from aiforecastts import HarmonicGradientResonance, AutoARIMAModel, ExponentialSmoothingModel

# HGR with confidence intervals
hgr = HarmonicGradientResonance(top_k_frequencies=5, use_bayesian_resonance=True)
hgr.fit(df, 'value', auto_tune=True)
forecast = hgr.predict(steps=14, return_ci=True)
print(hgr.summary())

# AutoARIMA
arima = AutoARIMAModel(seasonal=True, m=7)
arima.fit(df, 'value')
forecast = arima.predict(steps=14, return_ci=True)

# Exponential Smoothing
ets = ExponentialSmoothingModel(trend='add', seasonal='add')
ets.fit(df, 'value')
forecast = ets.predict(steps=14)
```

## 6. API Reference

### Models
| Class | Description |
|-------|-------------|
| `HarmonicGradientResonance` | HGR v2.0 with CI, Bayesian, auto-tune |
| `ScientificEnsemble` | 3-model ensemble (Prophet + HGR + ARIMA) |
| `AutoARIMAModel` | Auto-tuned ARIMA via pmdarima |
| `ExponentialSmoothingModel` | Holt-Winters ETS |

### Analytics
| Class | Description |
|-------|-------------|
| `DataProcessor` | Cleaning, outliers, stationarity, normalization |
| `FeatureEngineer` | Lag, rolling, technical, Fourier features |
| `ModelEvaluator` | Metrics, diagnostics, model comparison |

### Tools
| Class | Description |
|-------|-------------|
| `TimeSeriesResearch` | Main pipeline orchestrator |
| `Backtester` | Walk-forward validation framework |
| `ForecastPlotter` | Professional visualization |
| `AIForecastAgent` | Gemini-powered analysis agent |

## 7. What's New in v1.0.0

### Breaking Changes from v0.3.x
- `check_stationarity()` now returns `conclusion` instead of `is_stationary`
- `ScientificEnsemble` now includes ARIMA by default (use `include_arima=False` for v0.3 behavior)

### New Features
- HGR v2.0: Confidence intervals, Bayesian Ridge, Welch spectral, auto-tuning
- AutoARIMA and ETS model wrappers
- 3-model ensemble with 3D weight optimization
- Walk-forward backtesting with multi-model comparison
- Professional visualization (forecast, components, diagnostics, backtest)
- Outlier detection/treatment (IQR, Z-score, Modified Z-score)
- Confirmatory stationarity (ADF + KPSS)
- Data quality reports
- Enhanced feature engineering (Fourier, cyclic encoding, EWMA)
- Model save/load functionality
- Structured AI agent reports

## 8. Future Research Directions
- Integration of Quantum-inspired optimization for frequency selection
- Support for Multi-variate HGR (MV-HGR) to capture cross-series correlations
- Real-time streaming analytics for high-frequency turbulence monitoring
- Neural network hybrid (HGR + Transformer)
- Conformal prediction intervals

## 9. Citation
If you use this library in your research, please cite it as:
> Tuan, T. A. (2025). AIForecastTS: A Scientific Framework for Time Series Resonance & Turbulence Analysis. GitHub Repository. https://github.com/tuanthescientist/aiforecastts

---
*Developed by Tran Anh Tuan - (AI Forecast) aiconsultant.org*
```
