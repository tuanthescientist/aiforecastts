import pandas as pd
import numpy as np

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
