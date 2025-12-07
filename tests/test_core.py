import unittest
import pandas as pd
from ts_library.core import TimeSeriesAnalyzer

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

if __name__ == '__main__':
    unittest.main()
