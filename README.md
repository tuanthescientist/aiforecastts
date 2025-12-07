# AIForecastTS

[![PyPI version](https://badge.fury.io/py/aiforecastts.svg)](https://badge.fury.io/py/aiforecastts)
[![Tests](https://github.com/tuanthescientist/aiforecastts/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanthescientist/aiforecastts/actions)

This project is part of **AI Forecast** at **aiconsultant.org**. AIForecastTS is a practical Python library for time series analysis and forecasting — it includes classic utilities (moving average, seasonal decomposition, stationarity testing, ARIMA) and a higher-level ensemble forecaster (SuperForecaster) that combines Prophet, AutoARIMA, and XGBoost.

## Features
- Time series exploration utilities (moving averages, descriptive statistics)
- Seasonal decomposition (trend / seasonal / residual)
- Stationarity tests (ADF)
- ARIMA forecasting utilities
- SuperForecaster: an ensemble that trains Prophet, AutoARIMA and XGBoost with feature engineering (lags, rolling stats, RSI, MACD, Bollinger bands)

## Installation

```bash
pip install aiforecastts
```

## Quick Start

```python
import pandas as pd
from aiforecastts import TimeSeriesAnalyzer, SuperForecaster

# Sample data
data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range('2020-01-01', periods=10))

analyzer = TimeSeriesAnalyzer(data)
print(analyzer.moving_average(3))          # MA
print(analyzer.forecast_arima(steps=5))    # ARIMA
print(analyzer.is_stationary())            # ADF

# SuperForecaster (ensemble Prophet + AutoARIMA + XGBoost)
series = pd.Series(range(1, 121), index=pd.date_range('2020-01-01', periods=120))
forecaster = SuperForecaster(series)
metrics = forecaster.fit_ensemble(train_size=0.8)
print(metrics['mae'])
print(forecaster.predict(steps=7))
```

**Important**: This library does not fetch market data automatically. Please provide your own time series (CSV files, data APIs or a DataFrame) when using TimeSeriesAnalyzer or SuperForecaster.

## Development

```bash
git clone https://github.com/tuanthescientist/aiforecastts
cd aiforecastts
pip install -e .[dev]
python -m unittest discover -v tests
ruff check . --fix
black .
```

## Build & Publish

```bash
python -m build
twine upload dist/*
```

## Contact / project

- Repository: https://github.com/tuanthescientist/aiforecastts
- Project: https://aiconsultant.org (AI Forecast)
```
