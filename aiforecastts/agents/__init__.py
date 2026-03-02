# Lazy import to avoid hard dependency on google-generativeai
def __getattr__(name):
    if name == "AIForecastAgent":
        from .forecasting_agent import AIForecastAgent
        return AIForecastAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AIForecastAgent"]
