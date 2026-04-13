"""
Unit tests for the Hybrid AI Stock Prediction System.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import DataPreprocessor
from src.baseline_models import BaselineModels

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=100)
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)

    def test_clean_stock_data(self):
        cleaned = self.preprocessor.clean_stock_data(self.sample_data)
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertGreater(len(cleaned), 0)

    def test_add_technical_indicators(self):
        data_with_indicators = self.preprocessor.add_technical_indicators(self.sample_data)
        self.assertIn('SMA_20', data_with_indicators.columns)
        self.assertIn('RSI', data_with_indicators.columns)

    def test_get_sentiment_score(self):
        text = "The stock market is performing well!"
        score = self.preprocessor.get_sentiment_score(text)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

class TestBaselineModels(unittest.TestCase):
    def setUp(self):
        self.models = BaselineModels()
        self.X = np.random.rand(100, 5)
        self.y = np.random.rand(100)

    def test_random_forest_model(self):
        model = self.models.random_forest_model(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_xgboost_model(self):
        model = self.models.xgboost_model(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = BaselineModels() 
        self.y_true = np.array([100, 105, 102, 108, 110])
        self.y_pred = np.array([101, 104, 103, 107, 109])

    def test_calculate_metrics(self):
        metrics = self.evaluator.evaluate_model(self.y_true, self.y_pred) 
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertIsInstance(metrics['MAE'], (int, float))

if __name__ == '__main__':
    unittest.main()