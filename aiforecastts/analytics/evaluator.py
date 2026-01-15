import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelEvaluator:
    """
    Standardized evaluation metrics for time series forecasting.
    """
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Returns a dictionary of metrics: MAE, RMSE, MAPE, SMAPE.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        epsilon = 1e-10
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
        
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
            "SMAPE": round(smape, 2)
        }
