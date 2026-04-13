"""
Deep learning models for stock prediction using LSTM and Transformers.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DeepLearningModels:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_gru_model(self, input_shape):
        """Build GRU model."""
        model = Sequential([
            Input(shape=input_shape),
            tf.keras.layers.GRU(50, return_sequences=True),
            Dropout(0.2),
            tf.keras.layers.GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def load_transformer_model(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Load pre-trained transformer model for sentiment analysis."""
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def predict_sentiment(self, model, tokenizer, text):
        """Predict sentiment using transformer model."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model.to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.cpu().numpy()

# Example usage
if __name__ == "__main__":
    dl_models = DeepLearningModels()
    print("Deep learning models module loaded")