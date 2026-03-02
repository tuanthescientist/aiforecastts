import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation metrics and diagnostics for time series forecasting.
    
    Metrics:
    - MAE, RMSE, MAPE, SMAPE, R², MedAE
    - Directional Accuracy (DA)
    - Theil's U Statistic
    - Residual diagnostics (Ljung-Box, normality, autocorrelation)
    
    Cross-validation:
    - Walk-forward (expanding window)
    - Sliding window
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
        """
        Returns a comprehensive dictionary of forecasting metrics.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        epsilon = 1e-10
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
        
        # SMAPE
        smape = 100 / len(y_true) * np.sum(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
        )
        
        # R²
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
        
        # Median Absolute Error
        medae = float(np.median(np.abs(y_true - y_pred)))
        
        # Max Error
        max_error = float(np.max(np.abs(y_true - y_pred)))
        
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
            "SMAPE": round(smape, 2),
            "R2": round(r2, 4),
            "MedAE": round(medae, 4),
            "MaxError": round(max_error, 4),
        }
    
    @staticmethod
    def directional_accuracy(y_true, y_pred) -> float:
        """
        Measures how often the forecast correctly predicts the direction of change.
        Returns percentage (0-100).
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) < 2:
            return 0.0
        
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        
        da = np.mean(true_dir == pred_dir) * 100
        return round(da, 2)
    
    @staticmethod
    def theils_u(y_true, y_pred) -> float:
        """
        Theil's U statistic. 
        U < 1: model is better than naive forecast.
        U = 1: equivalent to naive.
        U > 1: worse than naive.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        
        if len(y_true) < 2:
            return float('inf')
        
        # Naive forecast: last value
        naive_pred = np.roll(y_true, 1)[1:]
        y_true_shifted = y_true[1:]
        y_pred_shifted = y_pred[1:]
        
        model_mse = np.mean((y_true_shifted - y_pred_shifted) ** 2)
        naive_mse = np.mean((y_true_shifted - naive_pred) ** 2)
        
        if naive_mse == 0:
            return 0.0 if model_mse == 0 else float('inf')
        
        return round(np.sqrt(model_mse / naive_mse), 4)
    
    @staticmethod
    def residual_diagnostics(y_true, y_pred) -> Dict[str, any]:
        """
        Comprehensive residual diagnostics:
        - Normality (Shapiro-Wilk or Jarque-Bera)
        - Autocorrelation (Ljung-Box proxy)
        - Mean, std, skewness, kurtosis
        """
        residuals = np.array(y_true, dtype=np.float64) - np.array(y_pred, dtype=np.float64)
        n = len(residuals)
        
        diagnostics = {
            'mean': round(float(np.mean(residuals)), 4),
            'std': round(float(np.std(residuals)), 4),
            'skewness': round(float(stats.skew(residuals)), 4),
            'kurtosis': round(float(stats.kurtosis(residuals)), 4),
        }
        
        # Normality test
        if 3 <= n <= 5000:
            stat, p_value = stats.shapiro(residuals)
            diagnostics['normality_test'] = 'Shapiro-Wilk'
            diagnostics['normality_stat'] = round(float(stat), 4)
            diagnostics['normality_p_value'] = round(float(p_value), 4)
            diagnostics['residuals_normal'] = p_value > 0.05
        elif n > 5000:
            stat, p_value = stats.jarque_bera(residuals)
            diagnostics['normality_test'] = 'Jarque-Bera'
            diagnostics['normality_stat'] = round(float(stat), 4)
            diagnostics['normality_p_value'] = round(float(p_value), 4)
            diagnostics['residuals_normal'] = p_value > 0.05
        
        # Autocorrelation check (first 5 lags)
        if n > 10:
            acf_values = []
            for lag in range(1, min(6, n // 2)):
                r = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
                acf_values.append(round(float(r), 4))
            diagnostics['autocorrelation_lags_1_5'] = acf_values
            diagnostics['significant_autocorrelation'] = any(abs(v) > 2 / np.sqrt(n) for v in acf_values)
        
        return diagnostics
    
    @staticmethod
    def compare_models(
        y_true,
        predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Parameters
        ----------
        y_true : array-like
            Actual values
        predictions : dict
            {model_name: predicted_values}
        
        Returns
        -------
        pd.DataFrame with metrics for each model
        """
        results = []
        y_true = np.array(y_true)
        
        for model_name, y_pred in predictions.items():
            y_pred = np.array(y_pred)
            metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            metrics['DA'] = ModelEvaluator.directional_accuracy(y_true, y_pred)
            metrics['Theils_U'] = ModelEvaluator.theils_u(y_true, y_pred)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        return df.sort_values('RMSE')
    
    @staticmethod
    def walk_forward_validation(
        model,
        df: pd.DataFrame,
        target_col: str,
        initial_train_size: int,
        forecast_horizon: int = 1,
        step_size: int = 1,
        expanding: bool = True,
    ) -> Dict[str, any]:
        """
        Walk-forward (expanding or sliding window) cross-validation.
        
        Parameters
        ----------
        model : object
            Model with fit() and predict() methods
        df : pd.DataFrame
            Full dataset
        target_col : str
            Target column name
        initial_train_size : int
            Size of first training window
        forecast_horizon : int
            Number of steps to forecast each iteration
        step_size : int
            Number of steps to advance the window
        expanding : bool
            True for expanding window, False for sliding window
        
        Returns
        -------
        Dict with fold results and aggregated metrics
        """
        n = len(df)
        all_actuals = []
        all_preds = []
        fold_results = []
        fold_idx = 0
        
        start = initial_train_size
        while start + forecast_horizon <= n:
            if expanding:
                train = df.iloc[:start]
            else:
                train = df.iloc[max(0, start - initial_train_size):start]
            
            test = df.iloc[start:start + forecast_horizon]
            
            try:
                model_copy = model.__class__(**{
                    k: v for k, v in model.__dict__.items()
                    if not k.startswith('_') and k not in ('is_fitted', 'model', 'fitted_model')
                })
            except Exception:
                model_copy = model
            
            try:
                model_copy.fit(train, target_col)
                pred_df = model_copy.predict(steps=forecast_horizon, last_df=train)
                
                # Find the forecast column
                forecast_col = None
                for col in ['hgr_forecast', 'ensemble_forecast', 'arima_forecast', 'ets_forecast']:
                    if col in pred_df.columns:
                        forecast_col = col
                        break
                
                if forecast_col is None:
                    forecast_col = pred_df.columns[-1]
                
                y_pred = pred_df[forecast_col].values[:len(test)]
                y_true = test[target_col].values[:len(y_pred)]
                
                fold_metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
                fold_metrics['fold'] = fold_idx
                fold_results.append(fold_metrics)
                
                all_actuals.extend(y_true)
                all_preds.extend(y_pred)
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
            
            start += step_size
            fold_idx += 1
        
        # Aggregate metrics
        overall = ModelEvaluator.calculate_metrics(all_actuals, all_preds) if all_actuals else {}
        
        return {
            'n_folds': fold_idx,
            'fold_results': fold_results,
            'overall_metrics': overall,
            'all_actuals': all_actuals,
            'all_predictions': all_preds,
        }
