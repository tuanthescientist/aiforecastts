import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """
        Khởi tạo TimeSeriesAnalyzer.

        Args:
            data (pd.Series hoặc pd.DataFrame): Dữ liệu chuỗi thời gian.
        """
        self.data = data

    def moving_average(self, window=3):
        """
        Tính trung bình động (Moving Average).

        Args:
            window (int): Kích thước cửa sổ.

        Returns:
            pd.Series: Kết quả trung bình động.
        """
        return self.data.rolling(window=window).mean()

    def summary(self):
        """
        Trả về thống kê mô tả cơ bản.
        """
        return self.data.describe()

    def decompose(self, model='additive', period=12):
        """
        Phân tích thành phần chuỗi thời gian (trend, seasonal, residual).

        Args:
            model (str): 'additive' hoặc 'multiplicative'.
            period (int): Chu kỳ mùa vụ.

        Returns:
            DecomposeResult: Kết quả phân tích.
        """
        return seasonal_decompose(self.data, model=model, period=period)

    def is_stationary(self, alpha=0.05):
        """
        Kiểm tra tính dừng của chuỗi thời gian bằng ADF test.

        Returns:
            bool: True nếu dừng.
        """
        result = adfuller(self.data.dropna())
        return result[1] < alpha

    def forecast_arima(self, steps=5, order=(1, 1, 1)):
        """
        Dự báo bằng ARIMA.

        Args:
            steps (int): Số bước dự báo.
            order (tuple): (p,d,q) cho ARIMA.

        Returns:
            pd.Series: Dự báo.
        """
        model = ARIMA(self.data, order=order).fit()
        return model.forecast(steps=steps)
