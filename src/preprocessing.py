"""
Data preprocessing module for cleaning and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.sia = SentimentIntensityAnalyzer()

    def clean_stock_data(self, df):
        """Clean stock price data."""
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.sort_index()
        return df

    def add_technical_indicators(self, df):
        """Add technical indicators to stock data."""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def preprocess_text(self, text):
        """Basic text preprocessing."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return tokens

    def get_sentiment_score(self, text):
        """Get sentiment score using VADER."""
        return self.sia.polarity_scores(text)['compound']

    def scale_features(self, df, features):
        """Scale numerical features."""
        scaled = self.scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled, columns=features, index=df.index)
        return scaled_df

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Sample data
    sample_text = "The stock market is performing well today!"
    sentiment = preprocessor.get_sentiment_score(sample_text)
    print(f"Sentiment score: {sentiment}")