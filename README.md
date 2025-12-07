# AIForecastTS

[![PyPI version](https://badge.fury.io/py/aiforecastts.svg)](https://badge.fury.io/py/aiforecastts)
[![Tests](https://github.com/tuanthescientist/aiforecastts/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanthescientist/aiforecastts/actions)

Thuộc dự án **AI Forecast** của **aiconsultant.org**. Thư viện dự báo chuỗi thời gian với ARIMA, decomposition, kiểm định dừng, và ensemble Prophet + AutoARIMA + XGBoost.

## Features
- Moving Average, thống kê mô tả nhanh
- Seasonal Decomposition (trend/seasonal/residual)
- Stationarity Test (ADF)
- ARIMA Forecasting
- **SuperForecaster**: Ensemble (Prophet + AutoARIMA + XGBoost) với feature engineering (lags, rolling stats, RSI, MACD, Bollinger)

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

**Lưu ý dữ liệu**: Thư viện không tự kéo dữ liệu (không dùng yfinance); bạn cần cung cấp dữ liệu (CSV, API, v.v.).

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

## Liên hệ / GitHub

- Repo: https://github.com/tuanthescientist/aiforecastts
- Dự án: https://aiconsultant.org (AI Forecast)
```
