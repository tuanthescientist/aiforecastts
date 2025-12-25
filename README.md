# AIForecastTS: A Scientific Framework for Time Series Resonance & Turbulence Analysis

[![PyPI version](https://badge.fury.io/py/aiforecastts.svg)](https://badge.fury.io/py/aiforecastts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Abstract
AIForecastTS is a high-performance scientific library designed to bridge the gap between classical signal processing and modern machine learning. At its core, the library introduces the **Harmonic-Gradient Resonance (HGR)** algorithm, a novel approach that mathematically decomposes time series into deterministic physical components (Resonance) and stochastic dynamical components (Turbulence). Integrated with **Gemini 3 Flash Preview**, it provides an automated AI Research Agent capable of interpreting complex temporal patterns.

## 2. Theoretical Framework: The HGR Algorithm
Traditional forecasting often fails by treating signal and noise as a monolithic entity. HGR operates on the principle of **Dual-Component Decomposition**:

### 2.1. Resonance Module (Deterministic Physics)
The Resonance module assumes that every time series contains underlying periodicities driven by systemic cycles.
- **Spectral Identification**: Uses Fast Fourier Transform (FFT) to map the series into the frequency domain.
- **Harmonic Regression**: Selects the top $K$ dominant frequencies and reconstructs the signal using a basis of sine and cosine functions:
  $$y_{res}(t) = \sum_{i=1}^{K} [a_i \sin(\omega_i t) + b_i \cos(\omega_i t)]$$
- **Purpose**: Captures long-term trends and seasonal cycles with mathematical precision.

### 2.2. Turbulence Module (Stochastic Dynamics)
The residuals from the Resonance module ($y - y_{res}$) represent "Turbulence" — the chaotic, non-linear interactions of the system.
- **Temporally-Weighted Gradient Boosting**: Employs a modified XGBoost architecture where training samples are weighted by their temporal proximity:
  $$W(t) = \alpha + \beta \cdot \frac{t}{T}$$
- **Recursive Stochastic Mapping**: Models the short-term dependencies and volatility clusters within the noise.

## 3. Architecture & Modules

### 3.1. Advanced Analytics (`aiforecastts.analytics`)
- **DataProcessor**: Implements time-aware interpolation and rigorous stationarity testing (Augmented Dickey-Fuller).
- **FeatureEngineer**: Automated generation of high-dimensional feature spaces, including multi-scale lags and technical indicators (RSI, MACD, Bollinger Bands).

### 3.2. AI Research Agent (`aiforecastts.agents`)
Powered by **Gemini 3 Flash Preview**, the agent acts as an automated peer-reviewer. It analyzes the mathematical outputs of the HGR algorithm and generates a scientific synthesis, explaining the interaction between systemic resonance and market turbulence.

## 4. Installation & Setup

```bash
pip install aiforecastts
```

## 5. Scientific Workflow Example

```python
import pandas as pd
from aiforecastts import TimeSeriesResearch

# Load dataset (e.g., Financial or Sensor data)
df = pd.read_csv("data.csv", index_col='timestamp', parse_dates=True)

# Initialize the Research Pipeline
research = TimeSeriesResearch(df, target_col='target')

# Execute Full Analysis with Gemini 3 Flash Agent
results = research.run_full_analysis(
    agent_query="Analyze the resonance-turbulence interaction and assess forecast stability.",
    api_key="YOUR_GEMINI_API_KEY"
)

# Access Scientific Report
print(results['agent_report'])

# Access Mathematical Forecast
forecast_df = results['forecast']
print(forecast_df[['resonance', 'turbulence', 'hgr_forecast']])
```

## 6. Future Research Directions
- Integration of Quantum-inspired optimization for frequency selection.
- Support for Multi-variate HGR (MV-HGR) to capture cross-series correlations.
- Real-time streaming analytics for high-frequency turbulence monitoring.

## 7. Citation
If you use this library in your research, please cite it as:
> Tuan, T. A. (2025). AIForecastTS: A Scientific Framework for Time Series Resonance & Turbulence Analysis. GitHub Repository.

---
*Developed by Tran Anh Tuan - (AI Forecast) aiconsultant.org*
```
