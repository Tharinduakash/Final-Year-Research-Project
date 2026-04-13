"""
Advanced time-series transformer models for stock prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    def __init__(self, data, seq_length=60, pred_length=1):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        # Return only the first feature (price) as target for forecasting
        return x, y[:, 0]  # Shape: (seq_length, features), (pred_length,)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class InformerEncoder(nn.Module):
    """Simplified Informer Encoder for time series forecasting."""
    def __init__(self, input_size, d_model=512, n_heads=8, n_layers=6, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, 1)  # Project to single value

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = src.shape

        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)  # (batch_size, seq_len, d_model)

        # Global average pooling across sequence dimension
        output = torch.mean(output, dim=1)  # (batch_size, d_model)
        output = self.output_projection(output)  # (batch_size, 1)
        return output

class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer for multivariate time series."""
    def __init__(self, input_size, output_size=1, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Static enrichment (simplified)
        self.static_enrichment = nn.Linear(d_model, d_model)

        # Temporal processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Gating mechanism (simplified)
        self.gate = nn.Linear(d_model, d_model)
        self.final_output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Static enrichment
        static_context = torch.mean(x, dim=1, keepdim=True)
        static_context = self.static_enrichment(static_context)
        x = x + static_context

        # Temporal processing
        x = self.temporal_encoder(x)

        # Gating
        gate_output = torch.sigmoid(self.gate(x))
        x = x * gate_output

        # Global pooling and final output
        x = torch.mean(x, dim=1)
        output = self.final_output_projection(x)
        return output

class AdvancedTransformers:
    """Wrapper class for advanced transformer models."""
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.informer = None
        self.tft = None

    def build_informer(self, input_size, seq_length=60):
        """Build Informer model."""
        self.informer = InformerEncoder(input_size=input_size).to(self.device)
        return self.informer

    def build_tft(self, input_size, output_size=1):
        """Build Temporal Fusion Transformer."""
        self.tft = TemporalFusionTransformer(input_size=input_size, output_size=output_size).to(self.device)
        return self.tft

    def train_model(self, model, train_loader, num_epochs=10, learning_rate=0.001):
        """Train a transformer model."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

        return model

    def predict(self, model, data_loader):
        """Make predictions with trained model."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    transformers = AdvancedTransformers()
    print("Advanced transformer models module loaded")