import unittest
import pandas as pd
import numpy as np
from aiforecastts.core import TimeSeriesResearch
from aiforecastts.models.ensemble import ScientificEnsemble


class TestTimeSeriesResearch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"target": np.cumsum(np.random.randn(80))},
            index=pd.date_range("2020-01-01", periods=80, freq="D"),
        )

    def test_run_full_analysis_hgr(self):
        research = TimeSeriesResearch(self.df, target_col="target", model_type="hgr")
        results = research.run_full_analysis(steps=3)
        self.assertIn("forecast", results)
        self.assertIn("hgr_forecast", results["forecast"].columns)


class TestScientificEnsemble(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.df = pd.DataFrame(
            {"target": np.cumsum(np.random.randn(100))},
            index=pd.date_range("2021-01-01", periods=100, freq="D"),
        )

    def test_ensemble_fit_predict(self):
        model = ScientificEnsemble()
        model.fit(self.df, target_col="target", validation_size=10, optimize_weights=True)
        pred = model.predict(steps=5, last_df=self.df)
        self.assertEqual(len(pred), 5)
        self.assertIn("ensemble_forecast", pred.columns)
        self.assertAlmostEqual(
            model.weights["prophet"] + model.weights["hgr"], 1.0, places=2
        )


if __name__ == "__main__":
    unittest.main()
