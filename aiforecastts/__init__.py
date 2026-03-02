from .core import TimeSeriesResearch
from .analytics.processor import DataProcessor
from .analytics.features import FeatureEngineer
from .analytics.evaluator import ModelEvaluator
from .models.hgr import HarmonicGradientResonance
from .models.ensemble import ScientificEnsemble, SuperAlgorithm
from .models.statistical import AutoARIMAModel, ExponentialSmoothingModel
from .backtesting.backtester import Backtester
from .visualization.plots import ForecastPlotter

# Lazy import for AIForecastAgent (requires google-generativeai)
def __getattr__(name):
    if name == "AIForecastAgent":
        from .agents.forecasting_agent import AIForecastAgent
        return AIForecastAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"

__all__ = [
	"TimeSeriesResearch",
	"DataProcessor",
	"FeatureEngineer",
	"ModelEvaluator",
	"HarmonicGradientResonance",
	"ScientificEnsemble",
	"SuperAlgorithm",
	"AutoARIMAModel",
	"ExponentialSmoothingModel",
	"AIForecastAgent",
	"Backtester",
	"ForecastPlotter",
]
