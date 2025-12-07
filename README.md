# AIForecastTS

[![PyPI version](https://badge.fury.io/py/aiforecastts.svg)](https://badge.fury.io/py/aiforecastts)
[![Tests](https://github.com/tuanthescientist/aiforecastts/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanthescientist/aiforecastts/actions)

AI-powered Time Series Forecasting Library với ARIMA, decomposition, stationarity tests.

## Features
- Moving Average
- Seasonal Decomposition
- Stationarity Test (ADF)
- ARIMA Forecasting

## Installation

```bash
pip install aiforecastts
```

## Quick Start

```python
import pandas as pd
from aiforecastts import TimeSeriesAnalyzer, SuperForecaster

# Sample data
data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                 index=pd.date_range('2020-01-01', periods=10))

analyzer = TimeSeriesAnalyzer(data)

# Moving average
print(analyzer.moving_average(3))

# ARIMA forecast
forecast = analyzer.forecast_arima(steps=5)
print(forecast)

# Check stationarity
print(analyzer.is_stationary())

# SuperForecaster (ensemble Prophet + AutoARIMA + XGBoost)
series = pd.Series(
    range(1, 121),
    index=pd.date_range('2020-01-01', periods=120)
)
forecaster = SuperForecaster(series)
metrics = forecaster.fit_ensemble(train_size=0.8)
print(metrics['mae'])
print(forecaster.predict(steps=7))
```

## Development

```bash
git clone https://github.com/tuanthescientist/aiforecastts
cd aiforecastts
pip install -e .[dev]
pytest tests/
black .
ruff check .
```

## Build & Publish

```bash
python -m build
twine upload dist/*
```

## GitHub

[tuanthescientist/aiforecastts](https://github.com/tuanthescientist/aiforecastts)
```
