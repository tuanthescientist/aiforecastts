import unittest
import pandas as pd
import numpy as np
from aiforecastts.core import TimeSeriesResearch
from aiforecastts.models.hgr import HarmonicGradientResonance
from aiforecastts.models.ensemble import ScientificEnsemble
from aiforecastts.models.statistical import AutoARIMAModel, ExponentialSmoothingModel
from aiforecastts.analytics.evaluator import ModelEvaluator
from aiforecastts.analytics.processor import DataProcessor
from aiforecastts.analytics.features import FeatureEngineer
from aiforecastts.backtesting.backtester import Backtester


class TestTimeSeriesResearch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 120
        t = np.arange(n)
        seasonal = 5 * np.sin(2 * np.pi * t / 7)  # weekly seasonality
        trend = 0.05 * t
        noise = np.random.randn(n) * 0.5
        self.df = pd.DataFrame(
            {"target": trend + seasonal + noise},
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_run_full_analysis_hgr(self):
        research = TimeSeriesResearch(self.df, target_col="target", model_type="hgr")
        results = research.run_full_analysis(steps=3)
        self.assertIn("forecast", results)
        self.assertIn("hgr_forecast", results["forecast"].columns)
        self.assertIn("stationarity", results)
        self.assertIn("data_quality", results)
        self.assertIn("model_summary", results)

    def test_run_full_analysis_arima(self):
        research = TimeSeriesResearch(self.df, target_col="target", model_type="arima")
        results = research.run_full_analysis(steps=5)
        self.assertIn("forecast", results)
        self.assertIn("arima_forecast", results["forecast"].columns)

    def test_run_full_analysis_ets(self):
        research = TimeSeriesResearch(self.df, target_col="target", model_type="ets")
        results = research.run_full_analysis(steps=5)
        self.assertIn("forecast", results)
        self.assertIn("ets_forecast", results["forecast"].columns)

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            TimeSeriesResearch(self.df, target_col="target", model_type="invalid")

    def test_compare_models(self):
        research = TimeSeriesResearch(self.df, target_col="target")
        comparison = research.compare_models(model_types=["hgr", "arima"], steps=7)
        self.assertIn("comparison_table", comparison)
        self.assertGreater(len(comparison["comparison_table"]), 0)

    def test_run_with_ci(self):
        research = TimeSeriesResearch(self.df, target_col="target", model_type="hgr")
        results = research.run_full_analysis(steps=5, return_ci=True)
        self.assertIn("ci_lower", results["forecast"].columns)
        self.assertIn("ci_upper", results["forecast"].columns)


class TestHGR(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        self.df = pd.DataFrame(
            {"target": 3 * np.sin(2 * np.pi * t / 10) + np.random.randn(n) * 0.3},
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_hgr_fit_predict(self):
        hgr = HarmonicGradientResonance(top_k_frequencies=3)
        hgr.fit(self.df, "target")
        pred = hgr.predict(steps=7, last_df=self.df)
        self.assertEqual(len(pred), 7)
        self.assertIn("hgr_forecast", pred.columns)
        self.assertIn("resonance", pred.columns)
        self.assertIn("turbulence", pred.columns)

    def test_hgr_confidence_intervals(self):
        hgr = HarmonicGradientResonance(n_bootstrap=20)
        hgr.fit(self.df, "target")
        pred = hgr.predict(steps=5, return_ci=True)
        self.assertIn("ci_lower", pred.columns)
        self.assertIn("ci_upper", pred.columns)
        # CI lower should be less than upper
        self.assertTrue((pred["ci_lower"] <= pred["ci_upper"]).all())

    def test_hgr_summary(self):
        hgr = HarmonicGradientResonance()
        self.assertEqual(hgr.summary()["status"], "Not fitted")
        hgr.fit(self.df, "target")
        summary = hgr.summary()
        self.assertEqual(summary["status"], "Fitted")
        self.assertIn("dominant_periods", summary)
        self.assertIn("resonance_rmse", summary)

    def test_hgr_bayesian(self):
        hgr = HarmonicGradientResonance(use_bayesian_resonance=True)
        hgr.fit(self.df, "target")
        pred = hgr.predict(steps=5)
        self.assertEqual(len(pred), 5)


class TestAutoARIMA(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.df = pd.DataFrame(
            {"target": np.cumsum(np.random.randn(80))},
            index=pd.date_range("2021-01-01", periods=80, freq="D"),
        )

    def test_arima_fit_predict(self):
        model = AutoARIMAModel(seasonal=False)
        model.fit(self.df, "target")
        pred = model.predict(steps=7)
        self.assertEqual(len(pred), 7)
        self.assertIn("arima_forecast", pred.columns)

    def test_arima_with_ci(self):
        model = AutoARIMAModel(seasonal=False)
        model.fit(self.df, "target")
        pred = model.predict(steps=5, return_ci=True)
        self.assertIn("ci_lower", pred.columns)
        self.assertIn("ci_upper", pred.columns)

    def test_arima_summary(self):
        model = AutoARIMAModel()
        model.fit(self.df, "target")
        summary = model.summary()
        self.assertIn("order", summary)
        self.assertIn("aic", summary)


class TestETS(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        self.df = pd.DataFrame(
            {"target": 10 + 0.1 * t + np.random.randn(n) * 0.5},
            index=pd.date_range("2021-01-01", periods=n, freq="D"),
        )

    def test_ets_fit_predict(self):
        model = ExponentialSmoothingModel(trend="add", seasonal=None)
        model.fit(self.df, "target")
        pred = model.predict(steps=7)
        self.assertEqual(len(pred), 7)
        self.assertIn("ets_forecast", pred.columns)


class TestScientificEnsemble(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.df = pd.DataFrame(
            {"target": np.cumsum(np.random.randn(100))},
            index=pd.date_range("2021-01-01", periods=100, freq="D"),
        )

    def test_ensemble_fit_predict(self):
        model = ScientificEnsemble(include_arima=False)
        model.fit(self.df, target_col="target", validation_size=10, optimize_weights=True)
        pred = model.predict(steps=5, last_df=self.df)
        self.assertEqual(len(pred), 5)
        self.assertIn("ensemble_forecast", pred.columns)

    def test_ensemble_3_models(self):
        model = ScientificEnsemble(include_arima=True)
        model.fit(self.df, target_col="target", validation_size=10)
        pred = model.predict(steps=5, last_df=self.df)
        self.assertIn("ensemble_forecast", pred.columns)


class TestModelEvaluator(unittest.TestCase):
    def test_calculate_metrics(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)
        self.assertIn("MedAE", metrics)
        self.assertGreater(metrics["R2"], 0.9)

    def test_directional_accuracy(self):
        y_true = np.array([1, 2, 3, 2, 4])
        y_pred = np.array([1, 2.5, 3.5, 1.5, 3.8])
        da = ModelEvaluator.directional_accuracy(y_true, y_pred)
        self.assertGreaterEqual(da, 0)
        self.assertLessEqual(da, 100)

    def test_theils_u(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        u = ModelEvaluator.theils_u(y_true, y_pred)
        self.assertLess(u, 1)  # Should beat naive forecast

    def test_residual_diagnostics(self):
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        diag = ModelEvaluator.residual_diagnostics(y_true, y_pred)
        self.assertIn("mean", diag)
        self.assertIn("std", diag)
        self.assertIn("skewness", diag)

    def test_compare_models(self):
        y_true = np.array([1, 2, 3, 4, 5])
        predictions = {
            "model_a": np.array([1.1, 2.0, 3.2, 3.9, 5.1]),
            "model_b": np.array([1.5, 2.5, 2.5, 4.5, 4.5]),
        }
        table = ModelEvaluator.compare_models(y_true, predictions)
        self.assertEqual(len(table), 2)


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"target": np.random.randn(100)},
            index=pd.date_range("2020-01-01", periods=100, freq="D"),
        )

    def test_stationarity_confirmatory(self):
        result = DataProcessor.check_stationarity(self.df["target"])
        self.assertIn("conclusion", result)
        self.assertIn("adf_p_value", result)

    def test_outlier_detection(self):
        series = self.df["target"].copy()
        series.iloc[10] = 100  # inject outlier
        outliers = DataProcessor.detect_outliers(series, method="iqr")
        self.assertTrue(outliers.iloc[10])

    def test_treat_outliers(self):
        df = self.df.copy()
        df.loc[df.index[5], "target"] = 50  # outlier
        treated = DataProcessor.treat_outliers(df, "target", method="clip")
        self.assertLess(treated["target"].max(), 50)

    def test_data_quality_report(self):
        report = DataProcessor.data_quality_report(self.df, "target")
        self.assertIn("n_rows", report)
        self.assertIn("stats", report)
        self.assertEqual(report["n_rows"], 100)

    def test_normalize(self):
        normalized, params = DataProcessor.normalize(self.df["target"], method="minmax")
        self.assertAlmostEqual(normalized.min(), 0, places=5)
        self.assertAlmostEqual(normalized.max(), 1, places=5)

    def test_auto_difference(self):
        series = pd.Series(np.cumsum(np.random.randn(100)))
        diff_series, order = DataProcessor.auto_difference(series)
        self.assertGreaterEqual(order, 0)
        self.assertLessEqual(order, 2)


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            {"target": np.random.randn(200)},
            index=pd.date_range("2020-01-01", periods=200, freq="D"),
        )

    def test_transform_generates_features(self):
        fe = FeatureEngineer(include_fourier=True, include_calendar=True)
        result = fe.transform(self.df, "target")
        self.assertGreater(len(result.columns), 10)
        self.assertIn("lag_1", result.columns)
        self.assertIn("rsi", result.columns)
        self.assertIn("month_sin", result.columns)

    def test_feature_importance(self):
        fe = FeatureEngineer()
        importance = fe.feature_importance_summary(self.df, "target")
        self.assertIn("feature", importance.columns)
        self.assertIn("abs_correlation", importance.columns)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 120
        t = np.arange(n)
        self.df = pd.DataFrame(
            {"target": 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(n) * 0.3},
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )

    def test_backtest_hgr(self):
        bt = Backtester(
            initial_train_size=80,
            forecast_horizon=7,
            step_size=14,
            expanding=True,
        )
        model = HarmonicGradientResonance()
        results = bt.run(model, self.df, "target", verbose=False)
        self.assertGreater(results["n_folds"], 0)
        self.assertIn("overall_metrics", results)

    def test_backtest_compare(self):
        bt = Backtester(
            initial_train_size=80,
            forecast_horizon=7,
            step_size=20,
        )
        models = {
            "hgr": HarmonicGradientResonance(),
            "arima": AutoARIMAModel(seasonal=False),
        }
        results = bt.compare_models(models, self.df, "target", verbose=False)
        self.assertIn("comparison", results)


if __name__ == "__main__":
    unittest.main()
