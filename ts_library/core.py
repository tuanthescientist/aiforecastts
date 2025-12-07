import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import ta
from prophet import Prophet
from pmdarima import auto_arima
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """
        Khởi tạo TimeSeriesAnalyzer.

        Args:
            data (pd.Series hoặc pd.DataFrame): Dữ liệu chuỗi thời gian.
        """
        self.data = data

    def moving_average(self, window=3):
        """
        Tính trung bình động (Moving Average).

        Args:
            window (int): Kích thước cửa sổ.

        Returns:
            pd.Series: Kết quả trung bình động.
        """
        return self.data.rolling(window=window).mean()

    def summary(self):
        """
        Trả về thống kê mô tả cơ bản.
        """
        return self.data.describe()

    def decompose(self, model='additive', period=12):
        """
        Phân tích thành phần chuỗi thời gian (trend, seasonal, residual).

        Args:
            model (str): 'additive' hoặc 'multiplicative'.
            period (int): Chu kỳ mùa vụ.

        Returns:
            DecomposeResult: Kết quả phân tích.
        """
        return seasonal_decompose(self.data, model=model, period=period)

    def is_stationary(self, alpha=0.05):
        """
        Kiểm tra tính dừng của chuỗi thời gian bằng ADF test.

        Returns:
            bool: True nếu dừng.
        """
        result = adfuller(self.data.dropna())
        return result[1] < alpha

    def forecast_arima(self, steps=5, order=(1, 1, 1)):
        """
        Dự báo bằng ARIMA.

        Args:
            steps (int): Số bước dự báo.
            order (tuple): (p,d,q) cho ARIMA.

        Returns:
            pd.Series: Dự báo.
        """
        model = ARIMA(self.data, order=order).fit()
        return model.forecast(steps=steps)

class SuperForecaster(TimeSeriesAnalyzer):
    def __init__(self, data: Optional[pd.Series] = None):
        super().__init__(data) if data is not None else None
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def add_features(self, lags: int = 14) -> pd.DataFrame:
        """Engineer features: lags, rolling stats, TA indicators."""
        df = pd.DataFrame({'y': self.data})
        for i in range(1, lags+1):
            df[f'lag_{i}'] = df['y'].shift(i)
        df['rolling_mean_7'] = df['y'].rolling(7).mean()
        df['rolling_std_7'] = df['y'].rolling(7).std()
        df['rsi'] = ta.momentum.RSIIndicator(df['y']).rsi()
        df['macd'] = ta.trend.MACD(df['y']).macd()
        df['bb_upper'] = ta.volatility.BollingerBands(df['y']).bollinger_hband()
        return df.dropna()

    def fit_ensemble(self, train_size: float = 0.8, cv_folds: int = 5) -> Dict:
        """Fit Prophet, AutoARIMA, XGBoost ensemble with CV weights."""
        features = self.add_features()
        split = int(len(features) * train_size)
        X_train, y_train = features.iloc[:split].drop('y', axis=1), features.iloc[:split]['y']
        X_test, y_test = features.iloc[split:].drop('y', axis=1), features.iloc[split:]['y']

        # Prophet requires columns ds, y
        prophet_df = self.data.reset_index()
        prophet_df.columns = ['ds', 'y']
        m_prophet = Prophet().fit(prophet_df.iloc[:split])
        future_prophet = m_prophet.make_future_dataframe(periods=len(y_test))
        forecast_prophet = m_prophet.predict(future_prophet)['yhat'].iloc[-len(y_test):]

        # AutoARIMA
        arima_model = auto_arima(
            y_train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        forecast_arima = arima_model.predict(n_periods=len(y_test))

        # XGBoost
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        forecast_xgb = xgb_model.predict(X_test_scaled)

        # Weights by CV MAE (simplified)
        mae_prophet = mean_absolute_error(y_test, forecast_prophet)
        mae_arima = mean_absolute_error(y_test, forecast_arima)
        mae_xgb = mean_absolute_error(y_test, forecast_xgb)
        total_mae = mae_prophet + mae_arima + mae_xgb
        weights = {
            'prophet': (1 - mae_prophet / total_mae),
            'arima': (1 - mae_arima / total_mae),
            'xgb': (1 - mae_xgb / total_mae)
        }

        self.models = {'prophet': m_prophet, 'arima': arima_model, 'xgb': xgb_model}
        self.weights = weights
        self.is_fitted = True

        metrics = {
            'mae': {'prophet': mae_prophet, 'arima': mae_arima, 'xgb': mae_xgb},
            'rmse': {
                'prophet': np.sqrt(mean_squared_error(y_test, forecast_prophet)),
                'arima': np.sqrt(mean_squared_error(y_test, forecast_arima)),
                'xgb': np.sqrt(mean_squared_error(y_test, forecast_xgb))
            }
        }
        return metrics

    def predict(self, steps: int = 30, return_ci: bool = True) -> pd.Series:
        """Super prediction: Weighted ensemble forecast."""
        if not self.is_fitted:
            raise ValueError("Fit ensemble first!")
        
        # Prophet forecast
        future = self.models['prophet'].make_future_dataframe(periods=steps)
        prophet_fc = self.models['prophet'].predict(future)['yhat'].iloc[-steps:]
        if return_ci:
            prophet_ci = self.models['prophet'].predict(future)[['yhat_lower', 'yhat_upper']].iloc[-steps:]

        # ARIMA forecast
        arima_fc = self.models['arima'].predict(n_periods=steps)

        # XGBoost forecast (recursive)
        last_features = self.add_features().iloc[-1:].drop('y', axis=1)
        xgb_fc = []
        last_scaled = self.scaler.transform(last_features)
        for _ in range(steps):
            pred = self.models['xgb'].predict(last_scaled.reshape(1, -1))[0]
            xgb_fc.append(pred)
            # Update features (simplified)
            last_scaled = np.roll(last_scaled, -1)
            last_scaled[0, 0] = pred  # Update lag_1

        ensemble_fc = (self.weights['prophet'] * prophet_fc +
                       self.weights['arima'] * arima_fc +
                       self.weights['xgb'] * np.array(xgb_fc))
        
        return pd.Series(ensemble_fc, index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D'))
