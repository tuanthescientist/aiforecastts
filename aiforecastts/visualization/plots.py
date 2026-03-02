import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization features disabled. Install with: pip install matplotlib")


class ForecastPlotter:
    """
    Professional visualization module for time series forecasting.
    
    Plot types:
    - Forecast with confidence intervals
    - Component decomposition (Resonance + Turbulence)
    - Model comparison
    - Residual diagnostics
    - ACF/PACF
    - Backtest results
    """
    
    # Professional color palette
    COLORS = {
        'actual': '#2C3E50',
        'forecast': '#E74C3C',
        'ci': '#F5B7B1',
        'prophet': '#3498DB',
        'hgr': '#E67E22',
        'arima': '#2ECC71',
        'ensemble': '#9B59B6',
        'resonance': '#1ABC9C',
        'turbulence': '#E74C3C',
        'grid': '#ECF0F1',
    }
    
    @staticmethod
    def _check_matplotlib():
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )
    
    @staticmethod
    def plot_forecast(
        actual: pd.Series,
        forecast: pd.DataFrame,
        title: str = "Time Series Forecast",
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 6),
        show_last_n: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """
        Plot actual values with forecast and optional confidence intervals.
        """
        ForecastPlotter._check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Actual data
        plot_actual = actual.tail(show_last_n) if show_last_n else actual
        ax.plot(
            range(len(plot_actual)),
            plot_actual.values,
            color=ForecastPlotter.COLORS['actual'],
            linewidth=2,
            label='Actual',
        )
        
        # Determine forecast column
        forecast_col = None
        for col in ['ensemble_forecast', 'hgr_forecast', 'arima_forecast', 'ets_forecast']:
            if col in forecast.columns:
                forecast_col = col
                break
        if forecast_col is None:
            forecast_col = forecast.columns[-1]
        
        # Forecast
        forecast_x = range(len(plot_actual), len(plot_actual) + len(forecast))
        ax.plot(
            forecast_x,
            forecast[forecast_col].values,
            color=ForecastPlotter.COLORS['forecast'],
            linewidth=2,
            linestyle='--',
            marker='o',
            markersize=4,
            label=f'Forecast ({forecast_col})',
        )
        
        # Confidence Intervals
        if ci_lower is not None and ci_upper is not None:
            ax.fill_between(
                forecast_x,
                ci_lower,
                ci_upper,
                color=ForecastPlotter.COLORS['ci'],
                alpha=0.3,
                label='95% Confidence Interval',
            )
        elif 'ci_lower' in forecast.columns and 'ci_upper' in forecast.columns:
            ax.fill_between(
                forecast_x,
                forecast['ci_lower'].values,
                forecast['ci_upper'].values,
                color=ForecastPlotter.COLORS['ci'],
                alpha=0.3,
                label='95% Confidence Interval',
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_components(
        forecast: pd.DataFrame,
        title: str = "HGR Component Decomposition",
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """
        Plot Resonance and Turbulence components from HGR forecast.
        """
        ForecastPlotter._check_matplotlib()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Combined forecast
        forecast_col = 'hgr_forecast' if 'hgr_forecast' in forecast.columns else 'ensemble_forecast'
        if forecast_col in forecast.columns:
            axes[0].plot(
                forecast[forecast_col].values,
                color=ForecastPlotter.COLORS['forecast'],
                linewidth=2,
            )
            axes[0].set_title('Combined Forecast', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        # Resonance
        if 'resonance' in forecast.columns:
            axes[1].plot(
                forecast['resonance'].values,
                color=ForecastPlotter.COLORS['resonance'],
                linewidth=2,
            )
            axes[1].set_title('Resonance Component (Deterministic)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        # Turbulence
        if 'turbulence' in forecast.columns:
            axes[2].bar(
                range(len(forecast)),
                forecast['turbulence'].values,
                color=ForecastPlotter.COLORS['turbulence'],
                alpha=0.7,
            )
            axes[2].set_title('Turbulence Component (Stochastic)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Forecast Steps', fontsize=11)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_model_comparison(
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Model Comparison",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """
        Plot multiple model predictions against actual values.
        """
        ForecastPlotter._check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(actual))
        ax.plot(x, actual, color=ForecastPlotter.COLORS['actual'], linewidth=2.5, label='Actual')
        
        model_colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6', '#E74C3C', '#1ABC9C']
        
        for i, (name, preds) in enumerate(predictions.items()):
            color = model_colors[i % len(model_colors)]
            ax.plot(
                x[:len(preds)],
                preds,
                color=color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.8,
                label=name,
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_residual_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Diagnostics",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """
        4-panel residual diagnostic plot:
        1. Residuals over time
        2. Histogram of residuals
        3. Q-Q plot
        4. ACF of residuals
        """
        ForecastPlotter._check_matplotlib()
        from scipy import stats as sp_stats
        
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals, color='#2C3E50', linewidth=1)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram
        axes[0, 1].hist(residuals, bins=30, color='#3498DB', edgecolor='white', alpha=0.7, density=True)
        # Overlay normal distribution
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(
            x_range,
            sp_stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
            'r-', linewidth=2, label='Normal fit'
        )
        axes[0, 1].set_title('Residual Distribution', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        sp_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF of residuals
        n_lags = min(30, len(residuals) // 2 - 1)
        if n_lags > 0:
            acf_vals = []
            for lag in range(n_lags + 1):
                if lag == 0:
                    acf_vals.append(1.0)
                else:
                    acf_vals.append(np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1])
            
            axes[1, 1].bar(range(len(acf_vals)), acf_vals, color='#2C3E50', alpha=0.7)
            # Significance bounds
            sig = 1.96 / np.sqrt(len(residuals))
            axes[1, 1].axhline(y=sig, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].axhline(y=-sig, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('ACF of Residuals', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_backtest_results(
        backtest_results: Dict,
        metric: str = "RMSE",
        title: str = "Walk-Forward Backtest Results",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """
        Plot backtest fold-by-fold performance metrics.
        """
        ForecastPlotter._check_matplotlib()
        
        fold_results = backtest_results.get('fold_results', [])
        if not fold_results:
            logger.warning("No fold results to plot.")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Metric per fold
        folds = [r['fold'] for r in fold_results]
        values = [r.get(metric, 0) for r in fold_results]
        
        axes[0].bar(folds, values, color='#3498DB', alpha=0.7, edgecolor='white')
        axes[0].axhline(
            y=np.mean(values),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean {metric}: {np.mean(values):.4f}',
        )
        axes[0].set_title(f'{metric} per Fold', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Fold', fontsize=11)
        axes[0].set_ylabel(metric, fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Actual vs Predicted scatter
        actuals = backtest_results.get('all_actuals', [])
        preds = backtest_results.get('all_predictions', [])
        
        if actuals and preds:
            axes[1].scatter(actuals, preds, color='#3498DB', alpha=0.5, s=20)
            min_val = min(min(actuals), min(preds))
            max_val = max(max(actuals), max(preds))
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Fit')
            axes[1].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Actual', fontsize=11)
            axes[1].set_ylabel('Predicted', fontsize=11)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
