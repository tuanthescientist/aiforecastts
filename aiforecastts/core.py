from .analytics.processor import DataProcessor
from .analytics.evaluator import ModelEvaluator
from .models.hgr import HarmonicGradientResonance
from .models.ensemble import ScientificEnsemble
from .models.statistical import AutoARIMAModel, ExponentialSmoothingModel
from .backtesting.backtester import Backtester
from typing import Dict, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TimeSeriesResearch:
    """
    Main entry point for Scientific Time Series Research v2.0.
    
    Integrates:
    - Data Analytics (cleaning, stationarity, outlier detection)
    - HGR Algorithm (Harmonic-Gradient Resonance)
    - AutoARIMA (Classical statistical modeling)
    - ETS (Exponential Smoothing)
    - Scientific Ensemble (multi-model)
    - AI Agent (Gemini-powered analysis)
    - Backtesting Framework
    
    Supported model types: 'hgr', 'ensemble', 'arima', 'ets'
    """
    
    SUPPORTED_MODELS = ('hgr', 'ensemble', 'arima', 'ets')
    
    def __init__(self, data: pd.DataFrame, target_col: str, model_type: str = "hgr"):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {self.SUPPORTED_MODELS}")
        
        self.data = data
        self.target_col = target_col
        self.model_type = model_type
        self.processor = DataProcessor()
        self.model = self._create_model(model_type)
        self._fitted = False
        
    def _create_model(self, model_type: str):
        """Factory method for model creation."""
        if model_type == "ensemble":
            return ScientificEnsemble()
        elif model_type == "arima":
            return AutoARIMAModel()
        elif model_type == "ets":
            return ExponentialSmoothingModel()
        else:
            return HarmonicGradientResonance()
        
    def run_full_analysis(
        self,
        agent_query: Optional[str] = None,
        api_key: Optional[str] = None,
        steps: int = 7,
        clean_method: str = "interpolate",
        treat_outliers: bool = False,
        return_ci: bool = False,
    ) -> Dict:
        """
        Executes a complete research pipeline:
        1. Data Cleaning & Outlier Treatment
        2. Statistical Analysis (Stationarity - ADF + KPSS)
        3. Model Fitting & Forecasting
        4. Confidence Intervals (optional)
        5. AI Agent Interpretation (optional)
        
        Parameters
        ----------
        agent_query : str, optional
            Query for the AI agent
        api_key : str, optional
            Gemini API key
        steps : int
            Number of forecast steps
        clean_method : str
            Missing value strategy ('interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop')
        treat_outliers : bool
            Whether to detect and treat outliers
        return_ci : bool
            Whether to return confidence intervals
        """
        # 1. Clean data
        df = self.processor.clean_data(self.data, self.target_col, method=clean_method)
        
        # 1b. Outlier treatment
        if treat_outliers:
            df = self.processor.treat_outliers(df, self.target_col)
        
        # 2. Stationarity analysis
        stationarity = self.processor.check_stationarity(df[self.target_col])
        
        # 3. Data quality report
        quality = self.processor.data_quality_report(df, self.target_col)
        
        # 4. Fit and Predict
        self.model.fit(df, self.target_col)
        self._fitted = True
        
        # Handle return_ci for models that support it
        if return_ci and hasattr(self.model, 'predict') and self.model_type in ('hgr', 'arima'):
            forecast = self.model.predict(steps=steps, last_df=df, return_ci=True)
        else:
            forecast = self.model.predict(steps=steps, last_df=df)
        
        # 5. Model summary
        model_summary = self.model.summary() if hasattr(self.model, 'summary') else {}
        
        # 6. Agent interpretation (optional)
        report = None
        if agent_query:
            try:
                from .agents.forecasting_agent import AIForecastAgent
                agent = AIForecastAgent(api_key=api_key)
                model_label = self.model_type.upper()
                if self.model_type == "ensemble":
                    model_label = "ScientificEnsemble"
                
                report = agent.analyze_and_forecast(
                    df,
                    self.target_col,
                    agent_query,
                    forecast=forecast,
                    model_label=model_label,
                )
            except Exception as e:
                logger.warning(f"AI Agent failed: {e}")
                report = {"agent_report": f"Agent error: {str(e)}"}
        
        return {
            "stationarity": stationarity,
            "data_quality": quality,
            "model_used": self.model_type,
            "model_summary": model_summary,
            "forecast": forecast,
            "agent_report": report["agent_report"] if report else "No agent query provided.",
        }
    
    def compare_models(
        self,
        model_types: List[str] = None,
        steps: int = 7,
        validation_size: int = None,
    ) -> Dict:
        """
        Compare multiple forecasting models on the same dataset.
        
        Parameters
        ----------
        model_types : list, optional
            Models to compare. Default: all supported models.
        steps : int
            Forecast horizon for comparison
        validation_size : int, optional
            Size of validation set. Default: same as steps.
        
        Returns
        -------
        Dict with comparison metrics and per-model forecasts
        """
        model_types = model_types or ['hgr', 'arima', 'ets']
        validation_size = validation_size or steps
        
        df = self.processor.clean_data(self.data, self.target_col)
        
        if len(df) <= validation_size + 30:
            raise ValueError("Not enough data for model comparison.")
        
        train_df = df.iloc[:-validation_size]
        val_df = df.iloc[-validation_size:]
        y_true = val_df[self.target_col].values
        
        predictions = {}
        model_summaries = {}
        
        for mt in model_types:
            try:
                model = self._create_model(mt)
                model.fit(train_df, self.target_col)
                pred_df = model.predict(steps=validation_size, last_df=train_df)
                
                # Find forecast column
                for col in ['ensemble_forecast', 'hgr_forecast', 'arima_forecast', 'ets_forecast']:
                    if col in pred_df.columns:
                        predictions[mt] = pred_df[col].values[:len(y_true)]
                        break
                
                if mt not in predictions and len(pred_df.columns) > 0:
                    predictions[mt] = pred_df.iloc[:, -1].values[:len(y_true)]
                
                if hasattr(model, 'summary'):
                    model_summaries[mt] = model.summary()
                    
            except Exception as e:
                logger.warning(f"Model '{mt}' failed: {e}")
                continue
        
        # Compare
        comparison = ModelEvaluator.compare_models(y_true, predictions)
        
        return {
            'comparison_table': comparison,
            'model_summaries': model_summaries,
            'validation_size': validation_size,
            'y_true': y_true,
            'predictions': predictions,
        }
    
    def backtest(
        self,
        forecast_horizon: int = 7,
        step_size: int = 7,
        expanding: bool = True,
        initial_train_pct: float = 0.6,
    ) -> Dict:
        """
        Run walk-forward backtesting.
        
        Parameters
        ----------
        forecast_horizon : int
            Steps to forecast each fold
        step_size : int
            Steps to advance each fold
        expanding : bool
            Expanding (True) or sliding (False) window
        initial_train_pct : float
            Percentage of data for initial training window
        """
        df = self.processor.clean_data(self.data, self.target_col)
        
        bt = Backtester(
            initial_train_size=int(len(df) * initial_train_pct),
            forecast_horizon=forecast_horizon,
            step_size=step_size,
            expanding=expanding,
        )
        
        return bt.run(self.model, df, self.target_col)
    
    def backtest_compare(
        self,
        model_types: List[str] = None,
        forecast_horizon: int = 7,
        step_size: int = 7,
        expanding: bool = True,
        initial_train_pct: float = 0.6,
    ) -> Dict:
        """
        Backtest and compare multiple models.
        """
        model_types = model_types or ['hgr', 'arima']
        df = self.processor.clean_data(self.data, self.target_col)
        
        models = {}
        for mt in model_types:
            try:
                models[mt] = self._create_model(mt)
            except Exception as e:
                logger.warning(f"Cannot create model '{mt}': {e}")
        
        bt = Backtester(
            initial_train_size=int(len(df) * initial_train_pct),
            forecast_horizon=forecast_horizon,
            step_size=step_size,
            expanding=expanding,
        )
        
        return bt.compare_models(models, df, self.target_col)
