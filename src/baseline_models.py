"""
Baseline models for stock prediction.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class BaselineModels:
    def __init__(self):
        pass

    def arima_model(self, data, order=(5,1,0)):
        """Fit ARIMA model."""
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        return model_fit

    def prophet_model(self, df):
        """Fit Prophet model."""
        # Prepare data for Prophet
        prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        return model

    def random_forest_model(self, X_train, y_train):
        """Train Random Forest model."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def xgboost_model(self, X_train, y_train):
        """Train XGBoost model."""
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Example usage
if __name__ == "__main__":
    models = BaselineModels()
    # This would be used with actual data
    print("Baseline models module loaded")