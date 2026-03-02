import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from ..analytics.processor import DataProcessor
from ..models.hgr import HarmonicGradientResonance
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AIForecastAgent:
    """
    AI Agent for Time Series Research v2.0.
    Integrates Google Gemini with the library's analytical and forecasting capabilities.
    
    Enhancements v2.0:
    - Structured scientific reports
    - Multi-model comparison interpretation
    - Error handling and graceful degradation
    - Customizable system prompts
    - Streaming support
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a Senior Data Scientist and Time Series Expert. 
You analyze forecasting results with scientific rigor and provide actionable insights.
Always structure your response with: Summary, Trend Analysis, Stationarity Assessment, 
Forecast Interpretation, Model Performance, and Recommendations."""

    def __init__(self, api_key: str = None, model_name: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Provide it directly or set it as an environment variable."
            )
        
        genai.configure(api_key=api_key)
        self.model_name = model_name or "gemini-2.5-flash"
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
        comparison_results: Dict = None,
        system_prompt: str = None,
    ) -> Dict:
        """
        The agent interprets the user's query, performs analysis, and returns a scientific report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        target_col : str
            Target column
        query : str
            User's query
        forecast : pd.DataFrame, optional
            Pre-computed forecast
        model_label : str
            Name of the model used
        comparison_results : dict, optional
            Results from model comparison
        system_prompt : str, optional
            Custom system prompt
        """
        # 1. Data Analysis
        stats = self.processor.check_stationarity(df[target_col])
        quality = self.processor.data_quality_report(df, target_col)
        
        # 2. Forecasting (if not provided)
        if forecast is None:
            self.model.fit(df, target_col)
            forecast = self.model.predict(steps=7, last_df=df)
        
        # 3. Build context
        context = self._build_context(
            stats, quality, forecast, model_label, comparison_results, df, target_col
        )
        
        # 4. LLM Interpretation
        prompt = self._build_prompt(query, context, system_prompt)
        
        try:
            response = self.client.generate_content(prompt)
            agent_text = response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            agent_text = f"[Agent Error: {str(e)}] Unable to generate AI interpretation."
        
        return {
            "analysis": stats,
            "data_quality": quality,
            "forecast": forecast,
            "agent_report": agent_text,
        }
    
    def _build_context(
        self,
        stats: Dict,
        quality: Dict,
        forecast: pd.DataFrame,
        model_label: str,
        comparison_results: Optional[Dict],
        df: pd.DataFrame,
        target_col: str,
    ) -> str:
        """Build structured context block for the LLM."""
        sections = []
        
        # Data overview
        sections.append(f"## Data Overview\n{self._format_quality(quality)}")
        
        # Stationarity
        sections.append(f"## Stationarity Analysis\n{self._format_dict(stats)}")
        
        # Descriptive stats
        recent = df[target_col].tail(30)
        sections.append(
            f"## Recent Data Statistics (Last 30 points)\n"
            f"- Mean: {recent.mean():.4f}\n"
            f"- Std: {recent.std():.4f}\n"
            f"- Trend: {'Upward' if recent.iloc[-1] > recent.iloc[0] else 'Downward'}\n"
            f"- Range: [{recent.min():.4f}, {recent.max():.4f}]"
        )
        
        # Forecast details
        sections.append(f"## Model: {model_label}")
        forecast_lines = []
        for col in forecast.columns:
            vals = forecast[col].tolist()
            val_str = [f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v) for v in vals]
            forecast_lines.append(f"- {col}: [{', '.join(val_str)}]")
        sections.append("## Forecast Results\n" + "\n".join(forecast_lines))
        
        # Model comparison (if available)
        if comparison_results and 'comparison_table' in comparison_results:
            sections.append(
                f"## Model Comparison\n{comparison_results['comparison_table'].to_string()}"
            )
        
        return "\n\n".join(sections)
    
    def _build_prompt(self, query: str, context: str, system_prompt: str = None) -> str:
        """Build the full prompt."""
        sys = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        return f"""{sys}

--- ANALYSIS CONTEXT ---

{context}

--- USER QUERY ---

{query}

--- INSTRUCTIONS ---

Provide a scientific explanation covering:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Trend & Stationarity Analysis**: What the statistical tests reveal
3. **Forecast Interpretation**: What the model predicts and why
4. **Component Analysis**: How Resonance (deterministic) and Turbulence (stochastic) interact (if HGR)
5. **Confidence & Risks**: Uncertainty factors and potential risks
6. **Actionable Recommendations**: What actions to take based on the forecast
"""
    
    @staticmethod
    def _format_dict(d: Dict) -> str:
        """Format dict for display in prompt."""
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"- {k}:")
                for kk, vv in v.items():
                    lines.append(f"  - {kk}: {vv}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_quality(quality: Dict) -> str:
        """Format quality report for prompt."""
        return (
            f"- Total rows: {quality.get('n_rows', 'N/A')}\n"
            f"- Missing values: {quality.get('n_missing', 0)} ({quality.get('pct_missing', 0)}%)\n"
            f"- Outliers (IQR): {quality.get('n_outliers_iqr', 0)}\n"
            f"- Mean: {quality.get('stats', {}).get('mean', 'N/A')}\n"
            f"- Std: {quality.get('stats', {}).get('std', 'N/A')}"
        )
