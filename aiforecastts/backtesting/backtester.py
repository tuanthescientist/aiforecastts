import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from ..analytics.evaluator import ModelEvaluator
import logging
import copy

logger = logging.getLogger(__name__)


class Backtester:
    """
    Comprehensive backtesting framework for time series models.
    
    Supports:
    - Walk-forward validation (expanding window)
    - Sliding window validation
    - Multi-model comparison backtesting
    - Configurable forecast horizons
    - Detailed per-fold and aggregate metrics
    """
    
    def __init__(
        self,
        initial_train_size: int = None,
        forecast_horizon: int = 7,
        step_size: int = 1,
        expanding: bool = True,
        min_train_pct: float = 0.5,
    ):
        """
        Parameters
        ----------
        initial_train_size : int
            Size of initial training window. If None, uses min_train_pct of data.
        forecast_horizon : int
            Steps to forecast in each fold
        step_size : int
            How many steps to advance each fold
        expanding : bool
            True = expanding window, False = sliding window
        min_train_pct : float
            Minimum percentage of data for initial training (used if initial_train_size is None)
        """
        self.initial_train_size = initial_train_size
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.expanding = expanding
        self.min_train_pct = min_train_pct
    
    def run(
        self,
        model,
        df: pd.DataFrame,
        target_col: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtesting for a single model.
        
        Parameters
        ----------
        model : object
            Model instance with fit() and predict() methods
        df : pd.DataFrame
            Complete dataset
        target_col : str
            Target column name
        verbose : bool
            Print progress
        
        Returns
        -------
        Dict with fold_results, overall_metrics, all_actuals, all_predictions
        """
        n = len(df)
        train_size = self.initial_train_size or int(n * self.min_train_pct)
        
        if train_size + self.forecast_horizon > n:
            raise ValueError(
                f"Not enough data: need at least {train_size + self.forecast_horizon} rows, "
                f"got {n}."
            )
        
        all_actuals = []
        all_preds = []
        fold_results = []
        
        start = train_size
        fold_idx = 0
        
        while start + self.forecast_horizon <= n:
            if self.expanding:
                train = df.iloc[:start]
            else:
                train = df.iloc[max(0, start - train_size):start]
            
            test = df.iloc[start:start + self.forecast_horizon]
            
            try:
                # Create a fresh model instance for each fold
                model_copy = self._clone_model(model)
                
                # Fit
                model_copy.fit(train, target_col)
                
                # Predict
                pred_df = model_copy.predict(steps=self.forecast_horizon, last_df=train)
                
                # Find forecast column
                forecast_col = self._find_forecast_col(pred_df)
                
                y_pred = pred_df[forecast_col].values[:len(test)]
                y_true = test[target_col].values[:len(y_pred)]
                
                # Metrics
                fold_metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
                fold_metrics['fold'] = fold_idx
                fold_metrics['train_size'] = len(train)
                fold_metrics['test_size'] = len(test)
                fold_results.append(fold_metrics)
                
                all_actuals.extend(y_true.tolist())
                all_preds.extend(y_pred.tolist())
                
                if verbose:
                    logger.info(
                        f"Fold {fold_idx}: train={len(train)}, "
                        f"RMSE={fold_metrics['RMSE']:.4f}, "
                        f"MAE={fold_metrics['MAE']:.4f}"
                    )
                    
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                fold_results.append({'fold': fold_idx, 'error': str(e)})
            
            start += self.step_size
            fold_idx += 1
        
        # Overall metrics
        overall = ModelEvaluator.calculate_metrics(all_actuals, all_preds) if all_actuals else {}
        
        # Summary stats across folds
        successful_folds = [f for f in fold_results if 'RMSE' in f]
        fold_summary = {}
        if successful_folds:
            for metric in ['MAE', 'RMSE', 'MAPE', 'SMAPE']:
                values = [f[metric] for f in successful_folds]
                fold_summary[f'{metric}_mean'] = round(np.mean(values), 4)
                fold_summary[f'{metric}_std'] = round(np.std(values), 4)
        
        return {
            'n_folds': fold_idx,
            'n_successful_folds': len(successful_folds),
            'fold_results': fold_results,
            'fold_summary': fold_summary,
            'overall_metrics': overall,
            'all_actuals': all_actuals,
            'all_predictions': all_preds,
            'config': {
                'initial_train_size': train_size,
                'forecast_horizon': self.forecast_horizon,
                'step_size': self.step_size,
                'expanding': self.expanding,
            },
        }
    
    def compare_models(
        self,
        models: Dict[str, Any],
        df: pd.DataFrame,
        target_col: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtesting for multiple models and compare.
        
        Parameters
        ----------
        models : dict
            {model_name: model_instance}
        df : pd.DataFrame
            Complete dataset
        target_col : str
            Target column
        
        Returns
        -------
        Dict with per_model results and comparison DataFrame
        """
        results = {}
        
        for name, model in models.items():
            if verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"Backtesting: {name}")
                logger.info(f"{'='*50}")
            
            try:
                results[name] = self.run(model, df, target_col, verbose=verbose)
            except Exception as e:
                logger.warning(f"Model {name} backtesting failed: {e}")
                results[name] = {'error': str(e)}
        
        # Build comparison table
        comparison = []
        for name, res in results.items():
            if 'overall_metrics' in res and res['overall_metrics']:
                row = {'Model': name, **res['overall_metrics']}
                row['n_folds'] = res.get('n_successful_folds', 0)
                if 'fold_summary' in res:
                    row['RMSE_std'] = res['fold_summary'].get('RMSE_std', 0)
                comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        if not comparison_df.empty:
            comparison_df = comparison_df.set_index('Model').sort_values('RMSE')
        
        return {
            'per_model': results,
            'comparison': comparison_df,
        }
    
    @staticmethod
    def _clone_model(model):
        """Create a fresh model instance."""
        try:
            # Try to create new instance with same init params
            cls = model.__class__
            
            # For models that store their init params
            init_params = {}
            import inspect
            sig = inspect.signature(cls.__init__)
            for param_name in sig.parameters:
                if param_name == 'self':
                    continue
                if hasattr(model, param_name):
                    init_params[param_name] = getattr(model, param_name)
            
            return cls(**init_params) if init_params else cls()
        except Exception:
            try:
                return copy.deepcopy(model)
            except Exception:
                return model
    
    @staticmethod
    def _find_forecast_col(pred_df: pd.DataFrame) -> str:
        """Find the main forecast column in predictions."""
        priority = ['ensemble_forecast', 'hgr_forecast', 'arima_forecast', 'ets_forecast']
        for col in priority:
            if col in pred_df.columns:
                return col
        return pred_df.columns[-1]
