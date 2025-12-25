import unittest
import pandas as pd
import numpy as np
from aiforecastts.core import TimeSeriesAnalyzer, SuperForecaster


class TestTimeSeriesAnalyzer(unittest.TestCase):
    def setUp(self):
        self.data = pd.Series([1, 2, 3, 4, 5])
        self.analyzer = TimeSeriesAnalyzer(self.data)

    def test_moving_average(self):
        ma = self.analyzer.moving_average(window=3)
        # Giá trị đầu tiên và thứ hai sẽ là NaN
        self.assertTrue(pd.isna(ma.iloc[0]))
        self.assertTrue(pd.isna(ma.iloc[1]))
        # (1+2+3)/3 = 2.0
        self.assertEqual(ma.iloc[2], 2.0)

    def test_decompose(self):
        data = pd.Series(
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            index=pd.date_range("2020", periods=9, freq="ME"),
        )
        analyzer = TimeSeriesAnalyzer(data)
        result = analyzer.decompose(period=3)
        self.assertIsNotNone(result.trend)

    def test_is_stationary(self):
        data = pd.Series(np.random.randn(100).cumsum())  # Non-stationary
        analyzer = TimeSeriesAnalyzer(data)
        self.assertFalse(analyzer.is_stationary())

    def test_forecast_arima(self):
        data = pd.Series([1, 2, 3, 4, 5])
        analyzer = TimeSeriesAnalyzer(data)
        forecast = analyzer.forecast_arima(steps=3, order=(1, 0, 0))
        self.assertEqual(len(forecast), 3)
        self.assertTrue(all(isinstance(x, (int, float)) for x in forecast))


class TestSuperForecaster(unittest.TestCase):
    def setUp(self):
        self.forecaster = SuperForecaster()

    # Skip fetch_data test - no yfinance dep
    def test_super_forecaster_fit_predict(self):
        np.random.seed(42)
        self.forecaster.data = pd.Series(
            np.cumsum(np.random.randn(120)),
            index=pd.date_range("2020-01-01", periods=120, freq="D"),
        )
        metrics = self.forecaster.fit_ensemble(train_size=0.8)
        self.assertIn("mae", metrics)
        pred = self.forecaster.predict(steps=5)
        self.assertEqual(len(pred), 5)
        self.assertTrue(pred.index.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
