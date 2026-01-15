import os
import pandas as pd
import google.generativeai as genai
from ..analytics.processor import DataProcessor
from ..models.hgr import HarmonicGradientResonance

class AIForecastAgent:
    """
    AI Agent for Time Series Research.
    Integrates Gemini 3 Flash with the library's analytical and forecasting capabilities.
    """
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please provide it or set it as an environment variable.")
        
        genai.configure(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"
        self.client = genai.GenerativeModel(self.model_name)
        
        self.processor = DataProcessor()
        self.model = HarmonicGradientResonance()

    def analyze_and_forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        query: str,
        forecast: pd.DataFrame = None,
        model_label: str = "HGR",
    ):
        """
        The agent interprets the user's query, performs analysis, and returns a scientific report.
        """
        # 1. Data Analysis
        stats = self.processor.check_stationarity(df[target_col])
        
        # 2. Forecasting (optional if provided externally)
        if forecast is None:
            self.model.fit(df, target_col)
            forecast = self.model.predict(steps=7, last_df=df)
        
        # 3. LLM Interpretation
        # Build contextual forecast text
        forecast_lines = []
        if "ensemble_forecast" in forecast.columns:
            forecast_lines.append(
                f"Ensemble Forecast (next steps): {forecast['ensemble_forecast'].tolist()}"
            )
        if "hgr_forecast" in forecast.columns:
            forecast_lines.append(
                f"HGR Forecast (next steps): {forecast['hgr_forecast'].tolist()}"
            )
        if "resonance" in forecast.columns:
            forecast_lines.append(
                f"Resonance Component: {forecast['resonance'].tolist()}"
            )
        if "turbulence" in forecast.columns:
            forecast_lines.append(
                f"Turbulence Component: {forecast['turbulence'].tolist()}"
            )

        forecast_block = "\n".join(forecast_lines) if forecast_lines else "No forecast details available."

        prompt = f"""
        You are a Senior Data Scientist. Analyze the following time series results and answer the user's query.
        
        Data Statistics: {stats}
        Model Used: {model_label}
        {forecast_block}
        
        User Query: {query}
        
        Provide a scientific explanation, including trend analysis, stationarity, and forecast confidence.
        Explain how the Resonance and Turbulence components interacted.
        """
        
        response = self.client.generate_content(prompt)
        
        return {
            "analysis": stats,
            "forecast": forecast,
            "agent_report": response.text
        }
