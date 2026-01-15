from .analytics.processor import DataProcessor
from .models.hgr import HarmonicGradientResonance
from .models.ensemble import ScientificEnsemble
from .agents.forecasting_agent import AIForecastAgent

class TimeSeriesResearch:
    """
    Main entry point for Scientific Time Series Research.
    Integrates Data Analytics, HGR Algorithm, Ensemble Models, and AI Agents.
    """
    def __init__(self, data, target_col, model_type="hgr"):
        self.data = data
        self.target_col = target_col
        self.processor = DataProcessor()
        
        if model_type == "ensemble":
            self.model = ScientificEnsemble()
        else:
            self.model = HarmonicGradientResonance()
        
    def run_full_analysis(self, agent_query=None, api_key=None, steps: int = 7):
        """
        Executes a complete research pipeline:
        1. Data Cleaning & Processing
        2. Statistical Analysis (Stationarity)
        3. HGR (Harmonic-Gradient Resonance) Modeling
        4. AI Agent Interpretation (Optional)
        """
        # Clean data
        df = self.processor.clean_data(self.data, self.target_col)
        
        # Stationarity
        stationarity = self.processor.check_stationarity(df[self.target_col])
        
        # Fit and Predict
        self.model.fit(df, self.target_col)
        forecast = self.model.predict(steps=steps, last_df=df)
        
        report = None
        if agent_query:
            agent = AIForecastAgent(api_key=api_key)
            model_label = "ScientificEnsemble" if isinstance(self.model, ScientificEnsemble) else "HGR"
            # Update agent to use HGR results context if needed, 
            # but for now passing the dataframe and query is standard.
            # We might want to pass the forecast explicitly to the agent in a future update.
            # For now, the agent re-runs a model internally or we can inject the forecast.
            # Let's inject the forecast into the prompt inside the agent class later.
            # For this step, we keep the interface simple.
            
            # Note: The current agent implementation re-instantiates a model. 
            # Ideally, we should pass the fitted model or results.
            # But to keep changes minimal and safe:
            report = agent.analyze_and_forecast(
                df,
                self.target_col,
                agent_query,
                forecast=forecast,
                model_label=model_label,
            )
            
        return {
            "stationarity": stationarity,
            "model_used": "ensemble" if isinstance(self.model, ScientificEnsemble) else "hgr",
            "forecast": forecast,
            "agent_report": report["agent_report"] if report else "No agent query provided."
        }
