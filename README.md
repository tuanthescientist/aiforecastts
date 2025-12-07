# TS Library

Đây là một thư viện Python mẫu để phân tích chuỗi thời gian.

## Cài đặt

```bash
pip install ts-library-example
```

## Sử dụng

```python
import pandas as pd
from ts_library import TimeSeriesAnalyzer

data = pd.Series([1, 2, 3, 4, 5])
analyzer = TimeSeriesAnalyzer(data)

print(analyzer.moving_average(window=3))
```

## Phát triển

Để cài đặt môi trường phát triển:

```bash
pip install -e .
```
